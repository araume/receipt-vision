import os
import re
import csv
import threading
from datetime import datetime, timezone

import pytesseract
from PIL import Image
import cv2
import numpy as np
from pytesseract import Output

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import hashlib
import json
import urllib.request
import urllib.error
import email.utils
import webbrowser

from pathlib import Path
import sys

def _try_bundle_tesseract():
	# When frozen by PyInstaller, data lives in _MEIPASS
	app_dir = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
	tesseract_path = app_dir / 'Tesseract-OCR' / 'tesseract.exe'
	if tesseract_path.exists():
		os.environ['TESSDATA_PREFIX'] = str((tesseract_path.parent / 'tessdata').resolve())
		pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)

_try_bundle_tesseract()

# Fallback to default system install if bundle not found
_default_tess_path = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
if 'TESSDATA_PREFIX' not in os.environ and os.path.exists(_default_tess_path):
    pytesseract.pytesseract.tesseract_cmd = _default_tess_path


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS


def preprocess_image(image_path: str) -> Image.Image:
    """Load image, apply light denoise and binarization, return as PIL.Image for OCR."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)
    # Adaptive threshold can help with uneven lighting
    image_bin = cv2.adaptiveThreshold(
        image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    # Slight morphology opening to clean small noise
    kernel = np.ones((2, 2), np.uint8)
    image_clean = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, kernel)
    pil_img = Image.fromarray(image_clean)
    return pil_img


# -----------
# OCR helpers
# -----------

def _rotate_to_orientation(image_bgr: np.ndarray) -> np.ndarray:
    try:
        osd = pytesseract.image_to_osd(image_bgr)
        m = re.search(r"Rotate:\s*(\d+)", osd)
        if not m:
            return image_bgr
        deg = int(m.group(1)) % 360
        if deg == 90:
            return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        if deg == 180:
            return cv2.rotate(image_bgr, cv2.ROTATE_180)
        if deg == 270:
            return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass
    return image_bgr


def _scale_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    target_max_dim = 1600
    max_dim = max(h, w)
    if max_dim >= target_max_dim:
        return image_bgr
    scale = target_max_dim / float(max_dim)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _preprocess_variants(image_bgr: np.ndarray) -> list[Image.Image]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gray_clahe = _clahe(gray_blur)

    variants = []

    # Adaptive Gaussian (existing style)
    v1 = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    variants.append(Image.fromarray(v1))
    variants.append(Image.fromarray(cv2.bitwise_not(v1)))

    # OTSU
    _, v2 = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(Image.fromarray(v2))
    variants.append(Image.fromarray(cv2.bitwise_not(v2)))

    # Adaptive mean
    v3 = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    variants.append(Image.fromarray(v3))
    variants.append(Image.fromarray(cv2.bitwise_not(v3)))

    # Plain CLAHE gray (no threshold)
    variants.append(Image.fromarray(gray_clahe))

    return variants


def _ocr_union_text(variants: list[Image.Image]) -> str:
    configs = [
        "--oem 3 --psm 6 -l eng",
        "--oem 3 --psm 4 -l eng",
        "--oem 3 --psm 11 -l eng",
    ]
    texts = []
    for img in variants:
        for cfg in configs:
            try:
                texts.append(pytesseract.image_to_string(img, config=cfg))
            except Exception:
                continue
    return "\n".join(t for t in texts if t)


def _tesseract_data(image_bgr: np.ndarray, config: str = "--oem 3 --psm 6 -l eng") -> dict:
    try:
        return pytesseract.image_to_data(image_bgr, config=config, output_type=Output.DICT)
    except Exception:
        return {"text": []}


def _group_words_by_line(data: dict) -> list[dict]:
    if not data or not data.get("text"):
        return []
    n = len(data["text"])
    lines = {}
    for i in range(n):
        try:
            if int(data.get("conf", ["-1"][i])[i]) < -1:  # guard
                pass
        except Exception:
            pass
        key = (data.get("block_num", [0])[i], data.get("par_num", [0])[i], data.get("line_num", [0])[i])
        if key not in lines:
            lines[key] = {
                "left": [], "top": [], "right": [], "bottom": [], "words": []
            }
        text = data["text"][i] or ""
        if text.strip() == "":
            continue
        l = data["left"][i]; t = data["top"][i]
        w = data["width"][i]; h = data["height"][i]
        lines[key]["left"].append(l); lines[key]["top"].append(t)
        lines[key]["right"].append(l + w); lines[key]["bottom"].append(t + h)
        lines[key]["words"].append(text)
    merged = []
    for key, rec in lines.items():
        if not rec["words"]:
            continue
        left = min(rec["left"]); top = min(rec["top"]) 
        right = max(rec["right"]); bottom = max(rec["bottom"])
        merged.append({
            "bbox": (left, top, right, bottom),
            "text": " ".join(rec["words"]).strip()
        })
    return merged


def _crop_expand(image_bgr: np.ndarray, bbox: tuple[int, int, int, int], expand: float = 0.25) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    dx = int((x2 - x1) * expand)
    dy = int((y2 - y1) * expand)
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(w, x2 + dx)
    ny2 = min(h, y2 + dy)
    return image_bgr[ny1:ny2, nx1:nx2]


def _ocr_roi_digits(image_bgr: np.ndarray) -> str:
    roi_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    roi_gray = _clahe(roi_gray)
    _, th = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.resize(th, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    pil = Image.fromarray(th)
    cfg = "--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=+0123456789()"
    try:
        return pytesseract.image_to_string(pil, config=cfg)
    except Exception:
        return ""


def _ocr_roi_code(image_bgr: np.ndarray) -> str:
    roi_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    roi_gray = _clahe(roi_gray)
    _, th = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.resize(th, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    pil = Image.fromarray(th)
    cfg = "--oem 3 --psm 7 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    try:
        return pytesseract.image_to_string(pil, config=cfg)
    except Exception:
        return ""


def _extract_phone_from_layout(image_bgr: np.ndarray) -> str:
    data = _tesseract_data(image_bgr)
    lines = _group_words_by_line(data)
    # First pass: keyword lines
    for line in lines:
        txt = line["text"]
        if _PHONE_KEYWORD_PATTERN.search(txt) and not _REF_LINE_PATTERN.search(txt):
            roi = _crop_expand(image_bgr, line["bbox"], 0.4)
            raw = _ocr_roi_digits(roi)
            m = _PHONE_PH_PATTERN.search(raw) or _PHONE_GENERIC_PATTERN.search(raw)
            if m:
                return _format_ph_phone(m.group(0))
    # Second pass: any line with 09 or +63 9
    for line in lines:
        txt = line["text"]
        if "+63" in txt or re.search(r"\b0?9\d{2}\b", txt):
            roi = _crop_expand(image_bgr, line["bbox"], 0.3)
            raw = _ocr_roi_digits(roi)
            m = _PHONE_PH_PATTERN.search(raw) or _PHONE_GENERIC_PATTERN.search(raw)
            if m:
                return _format_ph_phone(m.group(0))
    return ""


def _extract_total_from_layout(image_bgr: np.ndarray) -> str:
    data = _tesseract_data(image_bgr)
    lines = _group_words_by_line(data)
    for line in lines:
        txt = line["text"]
        if _AMOUNT_KEYWORDS.search(txt):
            roi = _crop_expand(image_bgr, line["bbox"], 0.5)
            # OCR with money-friendly whitelist
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray = _clahe(roi_gray)
            _, th = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th = cv2.resize(th, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            pil = Image.fromarray(th)
            cfg = "--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=$€£₱0123456789., "
            try:
                raw = pytesseract.image_to_string(pil, config=cfg)
            except Exception:
                raw = ""
            m = _MONEY_PATTERN.search(raw)
            if m:
                return m.group(1).strip()
    return ""


def _extract_total_global(image_bgr: np.ndarray) -> str:
    # Scan the whole image with money whitelist across variants
    variants = _preprocess_variants(image_bgr)
    cfg = "--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=$€£₱PpHh0123456789., "
    texts = []
    for img in variants:
        try:
            texts.append(pytesseract.image_to_string(img, config=cfg))
        except Exception:
            continue
    text = "\n".join(texts)
    candidates = [m.group(1).strip() for m in _MONEY_PATTERN.finditer(text)]
    if not candidates:
        return ""
    def parse_amount(val: str) -> float:
        s = val.replace("$", "").replace("€", "").replace("£", "").replace("₱", "").replace("P", "").replace("p", "")
        s = s.strip().replace(",", "")
        try:
            return float(s)
        except Exception:
            return -1.0
    return max(candidates, key=parse_amount)


def _extract_reference_from_layout(image_bgr: np.ndarray) -> str:
    data = _tesseract_data(image_bgr)
    lines = _group_words_by_line(data)
    for idx, line in enumerate(lines):
        txt = line["text"]
        if _REF_LINE_PATTERN.search(txt):
            # Try same line
            roi1 = _crop_expand(image_bgr, line["bbox"], 0.5)
            raw1 = _ocr_roi_code(roi1).upper()
            for m in _REF_CODE_PATTERN.finditer(raw1):
                cand = m.group(0)
                if not re.fullmatch(r"63?9\d{9}", _digits_only(cand)):
                    return cand
            # Try next line if available
            if idx + 1 < len(lines):
                roi2 = _crop_expand(image_bgr, lines[idx + 1]["bbox"], 0.4)
                raw2 = _ocr_roi_code(roi2).upper()
                for m in _REF_CODE_PATTERN.finditer(raw2):
                    cand = m.group(0)
                    if not re.fullmatch(r"63?9\d{9}", _digits_only(cand)):
                        return cand
    return ""


def extract_all_fields(image_path: str, mode: str = "Balanced") -> dict:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Unable to read image: {image_path}")
    # Mode controls: orientation, scaling, variants, layout fallbacks
    do_rotate = True
    do_scale = True
    use_layout_fallbacks = True
    psm_configs = ["--oem 3 --psm 6 -l eng", "--oem 3 --psm 4 -l eng", "--oem 3 --psm 11 -l eng"]
    if mode == "Fast":
        do_rotate = False
        do_scale = False
        use_layout_fallbacks = False
        psm_configs = ["--oem 3 --psm 6 -l eng"]
    elif mode == "Balanced":
        do_rotate = True
        do_scale = True
        use_layout_fallbacks = True
        psm_configs = ["--oem 3 --psm 6 -l eng", "--oem 3 --psm 4 -l eng"]
    elif mode == "Max accuracy":
        do_rotate = True
        do_scale = True
        use_layout_fallbacks = True
        psm_configs = ["--oem 3 --psm 6 -l eng", "--oem 3 --psm 4 -l eng", "--oem 3 --psm 11 -l eng"]

    if do_rotate:
        image_bgr = _rotate_to_orientation(image_bgr)
    if do_scale:
        image_bgr = _scale_for_ocr(image_bgr)

    variants = _preprocess_variants(image_bgr)
    # Build OCR text using mode-specific PSM configs
    texts = []
    for img in variants:
        for cfg in psm_configs:
            try:
                texts.append(pytesseract.image_to_string(img, config=cfg))
            except Exception:
                continue
    union_text = "\n".join(t for t in texts if t)

    # First pass from union text
    reference = extract_reference_number(union_text)
    phone = extract_phone(union_text, forbidden_codes={reference} if reference else None)
    total = extract_total_amount(union_text)
    date = extract_date(union_text)

    # Fallbacks via layout-driven ROIs
    if use_layout_fallbacks and not phone:
        try:
            phone = _extract_phone_from_layout(image_bgr)
        except Exception:
            pass
    if use_layout_fallbacks and not reference:
        try:
            reference = _extract_reference_from_layout(image_bgr)
        except Exception:
            pass
    if use_layout_fallbacks and not total:
        try:
            total = _extract_total_from_layout(image_bgr)
        except Exception:
            pass
    if use_layout_fallbacks and not total:
        try:
            total = _extract_total_global(image_bgr)
        except Exception:
            pass

    # Final phone normalization and exclusion
    if reference:
        phone = extract_phone(phone or union_text, forbidden_codes={reference}) or phone

    return {
        "date": date,
        "phone": phone,
        "total": total,
        "reference": reference,
        "ocr_text": union_text,
    }


def ocr_image(pil_image: Image.Image) -> str:
    config = "--oem 3 --psm 6"
    try:
        text = pytesseract.image_to_string(pil_image, config=config)
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "Tesseract-OCR not found. Please install it and ensure it's on PATH or update the tesseract_cmd path."
        ) from e
    return text


# ------------------------
# Field extraction helpers
# ------------------------

_DATE_PATTERNS = [
    # 2025-08-19, 2025/08/19, 2025.08.19
    re.compile(r"\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b"),
    # 19-08-2025, 19/08/2025, 19.08.2025
    re.compile(r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b"),
    # Aug 19, 2025
    re.compile(r"\b([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})\b"),
    # 19 Aug 2025
    re.compile(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b"),
]


def extract_date(text: str) -> str:
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return ""


_PHONE_KEYWORD_PATTERN = re.compile(r"\b(Phone|Tel|Mobile|Contact)\b\s*[:#-]?[\s]*", re.IGNORECASE)
_PHONE_GENERIC_PATTERN = re.compile(r"\+?\d[\d\-\s()]{7,}\d")
# Philippine mobile number focused pattern: +63 9XX XXX XXXX | 09XX XXX XXXX | 9XX XXX XXXX
_PHONE_PH_PATTERN = re.compile(
    r"(?:\+?63[\s-]?9\d{2}[\s-]?\d{3}[\s-]?\d{4}|09\d{2}[\s-]?\d{3}[\s-]?\d{4}|9\d{2}[\s-]?\d{3}[\s-]?\d{4})"
)
_PHONE_MIN_DIGITS = 9
_PHONE_MAX_DIGITS = 15


def normalize_phone(raw: str) -> str:
    digits = re.sub(r"[^\d+]", "", raw)
    # Keep leading + if present
    if digits.startswith("+"):
        return "+" + re.sub(r"\D", "", digits[1:])
    return re.sub(r"\D", "", digits)


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def _extract_long_digit_chunks(raw: str, min_len: int = 7) -> set[str]:
    return {m.group(0) for m in re.finditer(r"\d{" + str(min_len) + r",}", raw)}


def _format_ph_phone(raw: str) -> str:
    # Normalize to local PH format: 09XXXXXXXXX (11 digits, no spaces)
    digits = _digits_only(raw)
    # +63 9XXXXXXXXX → 09XXXXXXXXX
    if digits.startswith("63") and len(digits) >= 12 and digits[2] == "9":
        core10 = digits[2:12]
        return "0" + core10
    # 09XXXXXXXXX → keep first 11
    if digits.startswith("09") and len(digits) >= 11:
        return digits[:11]
    # 9XXXXXXXXX → 09XXXXXXXXX
    if digits.startswith("9") and len(digits) >= 10:
        return "0" + digits[:10]
    # Fallback to generic normalization (digits only)
    generic = normalize_phone(raw)
    # If generic begins with +63, convert to 0 + next 10 digits
    if generic.startswith("+63"):
        rest = _digits_only(generic[3:])
        if rest.startswith("9") and len(rest) >= 10:
            return "0" + rest[:10]
    # If generic is 11 digits not starting with 0 but starting with 9, make it local
    gdig = _digits_only(generic)
    if gdig.startswith("9") and len(gdig) >= 10:
        return "0" + gdig[:10]
    return gdig or generic


def extract_phone(text: str, forbidden_codes: set[str] | None = None) -> str:
    if forbidden_codes is None:
        forbidden_codes = set()

    # Build forbidden digits set
    forbidden_digits_exact = {_digits_only(code) for code in forbidden_codes if code}
    forbidden_substrings = set()
    for code in forbidden_codes:
        if code:
            forbidden_substrings |= _extract_long_digit_chunks(code, min_len=7)

    def is_valid_phone(candidate: str) -> bool:
        digits = _digits_only(candidate)
        if not (_PHONE_MIN_DIGITS <= len(digits) <= _PHONE_MAX_DIGITS):
            return False
        if digits in forbidden_digits_exact:
            return False
        # Reject if candidate contains any long forbidden digit substring (reference spillover)
        for bad in forbidden_substrings:
            if bad and bad in digits:
                return False
        return True

    # Tier 1: Prefer PH-specific formats first
    for line in text.splitlines():
        if _REF_LINE_PATTERN.search(line):
            continue
        m = _PHONE_PH_PATTERN.search(line)
        if m and is_valid_phone(m.group(0)):
            return _format_ph_phone(m.group(0))

    # Tier 2: Lines with phone-like keywords (generic)
    for line in text.splitlines():
        if _PHONE_KEYWORD_PATTERN.search(line) and not _REF_LINE_PATTERN.search(line):
            m = _PHONE_GENERIC_PATTERN.search(line)
            if m and is_valid_phone(m.group(0)):
                return _format_ph_phone(m.group(0))

    # Tier 3: Any plausible phone sequence not overlapping reference
    for m in _PHONE_GENERIC_PATTERN.finditer(text):
        # Skip if on a reference-labeled line
        line = text[max(0, text.rfind("\n", 0, m.start())): text.find("\n", m.start())]
        if _REF_LINE_PATTERN.search(line):
            continue
        if is_valid_phone(m.group(0)):
            return _format_ph_phone(m.group(0))
    return ""


_AMOUNT_KEYWORDS = re.compile(
    r"\b(Total|Amount Due|Grand Total|Balance Due|Total Due|Total Amount)\b",
    re.IGNORECASE,
)
_MONEY_PATTERN = re.compile(
    r"([\$€£₱]?\s?\d{1,3}(?:[ ,]\d{3})*(?:[\.,]\d{2})|[\$€£₱]?\s?\d+\.[0-9]{2})"
)


def extract_total_amount(text: str) -> str:
    # Prefer amounts near total keywords
    for line in text.splitlines():
        if _AMOUNT_KEYWORDS.search(line):
            m = _MONEY_PATTERN.search(line)
            if m:
                return m.group(1).strip()
    # Fallback: choose the largest monetary value found
    candidates = [m.group(1).strip() for m in _MONEY_PATTERN.finditer(text)]
    def parse_amount(val: str) -> float:
        val = val.replace("$", "").replace("€", "").replace("£", "").replace("₱", "")
        val = val.strip()
        # Normalize thousand separators (assume , as thousand and . as decimal if both present)
        if "," in val and "." in val:
            val = val.replace(",", "")
        else:
            # If only commas, treat last comma as decimal (unlikely) – fallback to removing commas
            val = val.replace(",", "")
        try:
            return float(val)
        except ValueError:
            return -1.0
    if not candidates:
        return ""
    best = max(candidates, key=parse_amount)
    return best


_REF_LINE_PATTERN = re.compile(
    r"\b(Ref(?:erence)?|RRN|Txn|Transaction(?: ID)?|Invoice(?: No\.| #| Number)?|Receipt(?: No\.| #| Number)?|Order(?: No\.| #| Number)?)\b",
    re.IGNORECASE,
)
_REF_CODE_PATTERN = re.compile(r"(?=[A-Z0-9\-]{6,})(?=.*\d)[A-Z0-9\-]+")


def extract_reference_number(text: str) -> str:
    def _looks_like_phone_digits(d: str) -> bool:
        if not (_PHONE_MIN_DIGITS <= len(d) <= _PHONE_MAX_DIGITS):
            return False
        if d.startswith("63") and len(d) >= 12 and d[2] == "9":
            return True
        if d.startswith("09") and len(d) >= 11:
            return True
        if d.startswith("9") and len(d) >= 10:
            return True
        return False

    lines = text.splitlines()
    for i, line in enumerate(lines):
        label_match = _REF_LINE_PATTERN.search(line)
        if not label_match:
            continue
        segment = line[label_match.end():].upper()
        for m in _REF_CODE_PATTERN.finditer(segment):
            cand = m.group(0)
            if not _looks_like_phone_digits(_digits_only(cand)):
                return cand
        # Try anywhere on the same line
        for m in _REF_CODE_PATTERN.finditer(line.upper()):
            cand = m.group(0)
            if not _looks_like_phone_digits(_digits_only(cand)):
                return cand
        # Try next line in case the code is on the following line
        if i + 1 < len(lines):
            for m in _REF_CODE_PATTERN.finditer(lines[i + 1].upper()):
                cand = m.group(0)
                if not _looks_like_phone_digits(_digits_only(cand)):
                    return cand
    # Fallback: first generic long alphanumeric-with-digits code
    for m in _REF_CODE_PATTERN.finditer(text.upper()):
        cand = m.group(0)
        if not _looks_like_phone_digits(_digits_only(cand)):
            return cand
    return ""


def ensure_receipts_dir() -> str:
    out_dir = os.path.join(os.getcwd(), "receipts")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


class OCRApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Receipt OCR Scanner")
        self.root.geometry("900x600")

        self.folder_path_var = tk.StringVar()
        self.progress_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Select a folder to begin.")
        self.mode_var = tk.StringVar(value="Balanced")

        self.results = []  # list of dicts: {filename, date, phone, total, reference}
        self._scan_thread = None
        self._stop_requested = False
        self.current_folder = ""

        self._build_ui()

    def _build_ui(self) -> None:
        # Top: folder selection
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="Folder:").pack(side=tk.LEFT)
        entry = ttk.Entry(top_frame, textvariable=self.folder_path_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.browse_btn = ttk.Button(top_frame, text="Browse", command=self._on_browse)
        self.browse_btn.pack(side=tk.LEFT)
        self.start_btn = ttk.Button(top_frame, text="Start Scan", command=self._on_start_scan)
        self.start_btn.pack(side=tk.LEFT, padx=(5, 0))
        self.cancel_btn = ttk.Button(top_frame, text="Cancel", command=self._on_cancel_scan)
        try:
            self.cancel_btn.state(["disabled"])  # disabled initially
        except Exception:
            pass
        self.cancel_btn.pack(side=tk.LEFT, padx=(5, 0))

        ttk.Label(top_frame, text="Mode:").pack(side=tk.LEFT, padx=(10, 0))
        self.mode_combo = ttk.Combobox(
            top_frame,
            textvariable=self.mode_var,
            width=14,
            state="readonly",
            values=["Fast", "Balanced", "Max accuracy"],
        )
        self.mode_combo.pack(side=tk.LEFT, padx=(5, 0))

        ttk.Button(top_frame, text="Export CSV", command=self._on_export).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(top_frame, text="Rename Files", command=self._on_rename_files).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(top_frame, text="View OCR Text", command=self._on_view_ocr_text).pack(side=tk.LEFT, padx=(5, 0))

        # Progress bar and status
        prog_frame = ttk.Frame(self.root)
        prog_frame.pack(fill=tk.X, padx=10)
        self.progress = ttk.Progressbar(prog_frame, orient=tk.HORIZONTAL, mode='determinate', maximum=100, variable=self.progress_var)
        self.progress.pack(fill=tk.X, pady=(0, 5))
        self.status_label = ttk.Label(prog_frame, textvariable=self.status_var)
        self.status_label.pack(fill=tk.X)

        # Results table
        columns = ("filename", "date", "phone", "total", "reference")
        self.tree = ttk.Treeview(self.root, columns=columns, show='headings', height=20)
        for col, text in zip(columns, ["File", "Date", "Phone", "Total", "Reference"]):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=150 if col != "filename" else 250, anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Footer: Developer credit with clickable link
        footer = ttk.Frame(self.root)
        footer.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Label(footer, text="Connect with the developer:").pack(side=tk.LEFT)
        link = tk.Label(footer, text=" Mark-V", fg="#1a73e8", cursor="hand2")
        link.pack(side=tk.LEFT)
        link.bind("<Button-1>", lambda e: self._open_url("https://github.com/araume"))

    def _on_browse(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path_var.set(folder)

    def _on_start_scan(self) -> None:
        if self._scan_thread and self._scan_thread.is_alive():
            messagebox.showinfo("Scan in Progress", "Please wait for the current scan to finish.")
            return
        folder = self.folder_path_var.get().strip()
        if not folder:
            messagebox.showwarning("No Folder", "Please select a folder containing images.")
            return
        if not os.path.isdir(folder):
            messagebox.showerror("Invalid Folder", "The selected path is not a folder.")
            return
        self.results.clear()
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.progress_var.set(0)
        self.status_var.set("Scanning...")
        self._stop_requested = False
        self.current_folder = folder

        # Set UI states for scanning
        self._set_scan_ui_state(True)

        self._scan_thread = threading.Thread(target=self._scan_folder, args=(folder,), daemon=True)
        self._scan_thread.start()

    def _scan_folder(self, folder: str) -> None:
        files = [f for f in os.listdir(folder) if is_image_file(f)]
        files.sort()
        total = len(files)
        if total == 0:
            self._set_status("No image files found in the selected folder.")
            self._set_scan_ui_state(False)
            return
        for idx, filename in enumerate(files, start=1):
            if self._stop_requested:
                break
            filepath = os.path.join(folder, filename)
            try:
                fields = extract_all_fields(filepath, mode=self.mode_var.get())
                info = {
                    "filename": filename,
                    "date": fields.get("date", ""),
                    "phone": fields.get("phone", ""),
                    "total": fields.get("total", ""),
                    "reference": fields.get("reference", ""),
                    "ocr_text": fields.get("ocr_text", ""),
                }
            except Exception as e:
                info = {
                    "filename": filename,
                    "date": "",
                    "phone": "",
                    "total": "",
                    "reference": "",
                    "ocr_text": "",
                }
            self.results.append(info)
            self._append_row(info)
            progress_pct = int((idx / total) * 100)
            self._set_progress(progress_pct)
            self._set_status(f"Processed {idx}/{total} files")

        if self._stop_requested:
            self._set_status("Scan cancelled.")
        else:
            self._set_status("Scan completed.")
        self._set_scan_ui_state(False)

    def _set_scan_ui_state(self, scanning: bool) -> None:
        def _apply():
            try:
                if scanning:
                    self.cancel_btn.state(["!disabled"])
                    self.start_btn.state(["disabled"])
                    self.browse_btn.state(["disabled"])
                    self.mode_combo.state(["disabled"])
                else:
                    self.cancel_btn.state(["disabled"])
                    self.start_btn.state(["!disabled"])
                    self.browse_btn.state(["!disabled"])
                    self.mode_combo.state(["readonly"])  # selectable
            except Exception:
                pass
        self.root.after(0, _apply)

    def _on_cancel_scan(self) -> None:
        self._stop_requested = True
        self._set_status("Cancelling...")
        try:
            self.cancel_btn.state(["disabled"])
        except Exception:
            pass

    def _on_view_ocr_text(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No Selection", "Please select a row in the table first.")
            return
        item_id = sel[0]
        values = self.tree.item(item_id, "values")
        if not values:
            messagebox.showinfo("No Data", "Selected row has no data.")
            return
        filename = values[0]
        match = None
        for info in self.results:
            if info.get("filename") == filename:
                match = info
                break
        ocr_text = (match or {}).get("ocr_text", "")
        top = tk.Toplevel(self.root)
        top.title(f"OCR Text - {filename}")
        top.geometry("800x600")
        frm = ttk.Frame(top)
        frm.pack(fill=tk.BOTH, expand=True)
        txt = tk.Text(frm, wrap=tk.WORD)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", ocr_text)
        txt.configure(state=tk.DISABLED)

    # -----------------
    # Renaming helpers
    # -----------------

    def _parse_date_to_mdy_yy(self, date_str: str) -> str:
        if not date_str:
            return ""
        s = date_str.strip()
        fmts = [
            "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
            "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
            "%m-%d-%Y", "%m/%d/%Y", "%m.%d.%Y",
            "%d-%m-%y", "%d/%m/%y", "%d.%m.%y",
            "%m-%d-%y", "%m/%d/%y", "%m.%d.%y",
            "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y",
        ]
        for fmt in fmts:
            try:
                dt = datetime.strptime(s, fmt)
                return f"{dt.month}.{dt.day}.{str(dt.year)[-2:]}"
            except ValueError:
                continue
        # Try to normalize separators if present
        s2 = re.sub(r"[\\/.]", "-", s)
        for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d-%m-%y", "%m-%d-%y"]:
            try:
                dt = datetime.strptime(s2, fmt)
                return f"{dt.month}.{dt.day}.{str(dt.year)[-2:]}"
            except ValueError:
                continue
        return ""

    def _sanitize_reference(self, ref: str) -> str:
        if not ref:
            return ""
        return re.sub(r"[^A-Z0-9\-]", "", ref.upper())

    def _sanitize_amount(self, amount: str) -> str:
        if not amount:
            return ""
        s = amount.strip()
        s = s.replace("$", "").replace("€", "").replace("£", "").replace("₱", "")
        s = s.replace(",", "")
        s = re.sub(r"\s+", "", s)
        # Keep digits and a single dot
        m = re.match(r"(\d+(?:\.\d{1,2})?)", s)
        return m.group(1) if m else re.sub(r"[^0-9.]", "", s)

    def _unique_path(self, base_dir: str, base_name: str, ext: str) -> str:
        candidate = os.path.join(base_dir, f"{base_name}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx = 1
        while True:
            candidate = os.path.join(base_dir, f"{base_name}_{idx}{ext}")
            if not os.path.exists(candidate):
                return candidate
            idx += 1

    def _on_rename_files(self) -> None:
        if not self.results:
            messagebox.showinfo("No Results", "Please run a scan first.")
            return
        folder = self.current_folder or self.folder_path_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("No Folder", "The original scan folder is not available.")
            return

        # Decide which files can be renamed
        candidates = []
        for info in self.results:
            orig_name = info.get("filename", "")
            date_raw = info.get("date", "")
            ref_raw = info.get("reference", "")
            total_raw = info.get("total", "")
            mdy = self._parse_date_to_mdy_yy(date_raw)
            if not mdy:
                # Fallback: use file's modified time
                try:
                    mtime = os.path.getmtime(os.path.join(folder, orig_name))
                    dt = datetime.fromtimestamp(mtime)
                    mdy = f"{dt.month}.{dt.day}.{str(dt.year)[-2:]}"
                except Exception:
                    mdy = ""
            ref = self._sanitize_reference(ref_raw)
            total = self._sanitize_amount(total_raw)
            if not (orig_name and mdy and ref and total):
                continue
            candidates.append((info, orig_name, mdy, ref, total))

        if not candidates:
            messagebox.showinfo("Nothing to Rename", "No files have all required fields (date, reference, total).")
            return

        if not messagebox.askyesno(
            "Confirm Rename",
            f"Rename {len(candidates)} files in '{folder}' to 'M.D.YY_REFERENCE_TOTAL.ext'?",
        ):
            return

        renamed = 0
        skipped = 0
        failed = 0
        for info, orig_name, mdy, ref, total in candidates:
            old_path = os.path.join(folder, orig_name)
            if not os.path.exists(old_path):
                skipped += 1
                continue
            name_no_ext, ext = os.path.splitext(orig_name)
            # Build base name and sanitize for Windows
            base = f"{mdy}_{ref}_{total}"
            base = re.sub(r"[\\/:*?\"<>|]", "-", base)
            target_path = self._unique_path(folder, base, ext)
            try:
                os.rename(old_path, target_path)
                new_name = os.path.basename(target_path)
                info["filename"] = new_name
                renamed += 1
            except Exception:
                failed += 1
                continue

        # Rebuild the table to reflect new filenames
        for row in self.tree.get_children():
            self.tree.delete(row)
        for info in self.results:
            self._append_row(info)

        messagebox.showinfo(
            "Rename Complete",
            f"Renamed: {renamed}\nSkipped: {skipped}\nFailed: {failed}",
        )

    def _append_row(self, info: dict) -> None:
        def _insert():
            self.tree.insert("", tk.END, values=(
                info.get("filename", ""),
                info.get("date", ""),
                info.get("phone", ""),
                info.get("total", ""),
                info.get("reference", ""),
            ))
        self.root.after(0, _insert)

    def _set_progress(self, value: int) -> None:
        self.root.after(0, lambda: self.progress_var.set(value))

    def _set_status(self, msg: str) -> None:
        self.root.after(0, lambda: self.status_var.set(msg))

    def _on_export(self) -> None:
        if not self.results:
            messagebox.showinfo("No Data", "No results to export. Please run a scan first.")
            return
        out_dir = ensure_receipts_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = os.path.join(out_dir, f"results_{timestamp}.csv")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=os.path.basename(default_path),
            filetypes=[("CSV Files", "*.csv")],
            initialdir=out_dir,
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                fieldnames = ["filename", "date", "phone", "total", "reference"]
                writer = csv.DictWriter(
                    f, fieldnames=fieldnames, extrasaction='ignore'
                )
                writer.writeheader()
                for row in self.results:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
            messagebox.showinfo("Export Complete", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))

    def _open_url(self, url: str) -> None:
        try:
            webbrowser.open_new(url)
        except Exception:
            self.status_var.set(f"Open in browser: {url}")


def main() -> None:
    root = tk.Tk()
    root.withdraw()

    # Activation policy
    ACTIVATION_ENFORCE_DATE = datetime(2025, 9, 20).date()  # Start enforcing on this date (UTC)
    ACTIVATION_HASH = "73427e0ec2d74565ee01c6d90da35d85ed091d4b40df8e6cb3984816904b9709"  # 64-char lowercase hex; set this value directly

    def _sha256_hex(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    # Persistent state helpers (prevent clock rollback bypass)
    def _storage_path() -> str:
        base = os.getenv("APPDATA") or os.getcwd()
        return os.path.join(base, "receipt_vision_state.json")

    def _load_last_seen_utc() -> datetime | None:
        try:
            with open(_storage_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            val = data.get("last_seen_utc", "")
            if not val:
                return None
            # Stored as ISO without tz; interpret as UTC
            return datetime.fromisoformat(val)
        except Exception:
            return None

    def _save_last_seen_utc(dt_utc: datetime) -> None:
        try:
            payload = {"last_seen_utc": dt_utc.replace(microsecond=0).isoformat()}
            with open(_storage_path(), "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            pass

    # Persistent activation/block state
    def _load_state() -> dict:
        try:
            with open(_storage_path(), "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _save_state(state: dict) -> None:
        try:
            with open(_storage_path(), "w", encoding="utf-8") as f:
                json.dump(state, f)
        except Exception:
            pass

    def _is_activated() -> bool:
        try:
            state = _load_state()
            return state.get("activated_hash") == ACTIVATION_HASH
        except Exception:
            return False

    def _mark_activated(at_utc: datetime | None = None) -> None:
        try:
            state = _load_state()
            state["activated_hash"] = ACTIVATION_HASH
            if at_utc is None:
                try:
                    at_utc = datetime.utcnow()
                except Exception:
                    at_utc = None
            if at_utc is not None:
                state["activated_at_utc"] = at_utc.replace(microsecond=0).isoformat()
            _save_state(state)
        except Exception:
            pass

    def _is_blocked() -> bool:
        try:
            state = _load_state()
            return bool(state.get("blocked", False))
        except Exception:
            return False

    def _mark_blocked(at_utc: datetime | None = None) -> None:
        try:
            state = _load_state()
            state["blocked"] = True
            if at_utc is None:
                try:
                    at_utc = datetime.utcnow()
                except Exception:
                    at_utc = None
            if at_utc is not None:
                state["blocked_at_utc"] = at_utc.replace(microsecond=0).isoformat()
            _save_state(state)
        except Exception:
            pass

    def _online_utc_now() -> datetime | None:
        # Try to get reliable UTC time from HTTP Date header
        for url in ("https://www.google.com", "https://www.microsoft.com", "https://cloudflare.com"):
            try:
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    date_hdr = resp.headers.get("Date")
                if not date_hdr:
                    continue
                dt = email.utils.parsedate_to_datetime(date_hdr)
                if dt is None:
                    continue
                # Normalize to naive UTC
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                return dt
            except Exception:
                continue
        return None

    def _observed_utc_today() -> datetime.date:
        last_seen = _load_last_seen_utc()
        try:
            local_now = datetime.utcnow()
        except Exception:
            local_now = None
        online_now = _online_utc_now()

        candidates = [d for d in (last_seen, local_now, online_now) if d is not None]
        if not candidates:
            # Fallback to enforcement date to avoid accidental bypass
            return ACTIVATION_ENFORCE_DATE
        best = max(candidates)
        # Persist the max to prevent rollback bypass next run
        try:
            _save_last_seen_utc(best)
        except Exception:
            pass
        return best.date()

    def _is_enforcement_active() -> bool:
        if _is_activated():
            return False
        today_utc = _observed_utc_today()
        return today_utc >= ACTIVATION_ENFORCE_DATE

    def _prompt_activation() -> bool:
        if not ACTIVATION_HASH:
            messagebox.showerror(
                "Activation Required",
                "Activation is required but the activation hash is not configured. Please contact the developer.",
            )
            return False

        attempts_total = 5
        attempts_left = attempts_total
        result = {"ok": False}

        top = tk.Toplevel(root)
        top.title("Activation Required")
        top.resizable(False, False)
        top.grab_set()

        container = ttk.Frame(top, padding=(12, 12, 12, 12))
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="Activation is required to continue.").pack(anchor=tk.W)
        warn = ttk.Label(
            container,
            text=(
                "Warning: You have 5 attempts. Using all attempts will permanently disable "
                "this application on this device."
            ),
            foreground="#d93025",
            wraplength=420,
            justify=tk.LEFT,
        )
        warn.pack(anchor=tk.W, pady=(6, 6))

        attempts_var = tk.StringVar(value=f"Attempts remaining: {attempts_left}")
        ttk.Label(container, textvariable=attempts_var).pack(anchor=tk.W)

        ttk.Label(container, text="Activation code:").pack(anchor=tk.W, pady=(8, 2))
        code_entry = ttk.Entry(container, show="*")
        code_entry.pack(fill=tk.X)
        code_entry.focus_set()

        buttons = ttk.Frame(container)
        buttons.pack(fill=tk.X, pady=(10, 0))

        def on_submit() -> None:
            nonlocal attempts_left
            code = code_entry.get().strip()
            if not code:
                return
            if _sha256_hex(code) == ACTIVATION_HASH.strip().lower():
                try:
                    _mark_activated()
                except Exception:
                    pass
                result["ok"] = True
                top.destroy()
                return
            attempts_left -= 1
            if attempts_left <= 0:
                try:
                    _mark_blocked()
                except Exception:
                    pass
                messagebox.showerror(
                    "Activation Failed",
                    "Activation attempts exceeded. The application will now close permanently on this device.",
                    parent=top,
                )
                result["ok"] = False
                top.destroy()
                return
            attempts_var.set(f"Attempts remaining: {attempts_left}")
            messagebox.showwarning(
                "Invalid Code",
                f"Activation failed. {attempts_left} attempt(s) left.",
                parent=top,
            )
            code_entry.delete(0, tk.END)
            code_entry.focus_set()

        def on_cancel() -> None:
            result["ok"] = False
            top.destroy()

        ttk.Button(buttons, text="Activate", command=on_submit).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=(0, 8))

        top.bind("<Return>", lambda e: on_submit())
        top.protocol("WM_DELETE_WINDOW", on_cancel)
        root.wait_window(top)
        return bool(result.get("ok"))

    # Hard block if previously exhausted attempts
    if _is_blocked():
        messagebox.showerror("Application Blocked", "This application has been permanently disabled due to failed activation attempts.")
        root.destroy()
        return

    if _is_enforcement_active():
        ok = _prompt_activation()
        if not ok:
            root.destroy()
            return

    root.deiconify()
    app = OCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

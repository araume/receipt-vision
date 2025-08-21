import os
import re
import csv
import threading
from datetime import datetime

import pytesseract
from PIL import Image
import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
        ttk.Button(top_frame, text="Browse", command=self._on_browse).pack(side=tk.LEFT)
        ttk.Button(top_frame, text="Start Scan", command=self._on_start_scan).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(top_frame, text="Export CSV", command=self._on_export).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(top_frame, text="Rename Files", command=self._on_rename_files).pack(side=tk.LEFT, padx=(5, 0))

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

        self._scan_thread = threading.Thread(target=self._scan_folder, args=(folder,), daemon=True)
        self._scan_thread.start()

    def _scan_folder(self, folder: str) -> None:
        files = [f for f in os.listdir(folder) if is_image_file(f)]
        files.sort()
        total = len(files)
        if total == 0:
            self._set_status("No image files found in the selected folder.")
            return
        for idx, filename in enumerate(files, start=1):
            if self._stop_requested:
                break
            filepath = os.path.join(folder, filename)
            try:
                pil_img = preprocess_image(filepath)
                text = ocr_image(pil_img)
                reference = extract_reference_number(text)
                phone = extract_phone(text, forbidden_codes={reference} if reference else None)
                info = {
                    "filename": filename,
                    "date": extract_date(text),
                    "phone": phone,
                    "total": extract_total_amount(text),
                    "reference": reference,
                }
            except Exception as e:
                info = {
                    "filename": filename,
                    "date": "",
                    "phone": "",
                    "total": "",
                    "reference": "",
                }
            self.results.append(info)
            self._append_row(info)
            progress_pct = int((idx / total) * 100)
            self._set_progress(progress_pct)
            self._set_status(f"Processed {idx}/{total} files")

        self._set_status("Scan completed.")

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
                writer = csv.DictWriter(
                    f, fieldnames=["filename", "date", "phone", "total", "reference"]
                )
                writer.writeheader()
                for row in self.results:
                    writer.writerow(row)
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
    app = OCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

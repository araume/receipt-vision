## Receipt Vision (ComVision)

### Overview
Receipt Vision is a Windows Tkinter app for bulk OCR of receipt images. It scans a folder, extracts key fields, shows results in a table with progress, exports CSV, and can rename files with a consistent pattern.

### Features
- **Folder scan**: Recursively processes supported image files (.jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp)
- **OCR**: Tesseract-based extraction with light preprocessing (grayscale, blur, adaptive threshold)
- **Field extraction**:
  - **Date**: Common formats (e.g., 2025-08-19, 19/08/2025, Aug 19, 2025)
  - **Phone**: Philippine mobile formats prioritized; normalized to local 09XXXXXXXXX
  - **Total amount**: Prefers totals near keywords; falls back to largest value
  - **Reference number**: Looks after labels (Reference, RRN, Transaction ID, Invoice/Receipt No.) and next line; ensures it’s not a phone number
- **UI**: Progress bar, table of results, CSV export, file rename utility
- **File renaming**: One-click rename to `M.D.YY_REFERENCE_TOTAL.ext`
- **Developer link**: “Connect with the developer: Mark-V” opens the GitHub profile

### Requirements
- Windows 10/11
- Python 3.10+ (recommended)
- Tesseract OCR 5.x (system installed or bundled next to the EXE)
- Python packages: `opencv-python`, `pillow`, `pytesseract`, `numpy` (managed via `requirements.txt`)

### Setup (development)
1) Create and activate a virtual environment (PowerShell):
```powershell
python -m venv comvis
& .\comvis\Scripts\Activate.ps1
```

2) Install dependencies:
```powershell
python -m pip install -r requirements.txt
```

3) Install Tesseract OCR (system-wide) or ensure a bundled copy is available (see Build section). If system-installed, the default path `C:\Program Files\Tesseract-OCR\tesseract.exe` is auto-detected; otherwise put Tesseract on PATH or bundle it.

4) Run the app:
```powershell
python main.py
```

### Usage
1) Click “Browse” and select a folder containing receipt images
2) Click “Start Scan” to OCR all images; results populate the table
3) Optional: Click “Export CSV” to save results in the `receipts` folder
4) Optional: Click “Rename Files” to rename files to `M.D.YY_REFERENCE_TOTAL.ext`

### Extraction details
- **Date**: Multiple regex patterns for numeric and month-name formats
- **Phone**:
  - Prefers Philippine formats: `+63 9XX XXX XXXX`, `09XX XXX XXXX`, `9XX XXX XXXX`
  - Always normalized to `09XXXXXXXXX` (e.g., `+639638655707` → `09638655707`)
  - Avoids reference-labeled lines and excludes digits from the extracted reference
- **Total**: Prefers lines near keywords (Total, Amount Due, Grand Total); falls back to the largest monetary value found
- **Reference**: After labels and also checks next line; requires alphanumeric with at least one digit; rejects phone-like digit strings

### File renaming format
- Pattern: `M.D.YY_REFERENCE_TOTAL.{original extension}`
- Example: `8.19.25_ABC123_1345.67.jpg`
- The app sanitizes invalid filename characters and resolves collisions with suffixes like `_1`, `_2`, etc.

### Build a Windows executable

#### Option A: Use system-installed Tesseract (simplest)
1) Ensure Tesseract is installed (e.g., `C:\Program Files\Tesseract-OCR`) and accessible
2) Install the packager:
```powershell
python -m pip install pyinstaller
```
3) Build (no console window):
```powershell
pyinstaller --noconfirm --windowed --onedir --name Receipt-Vision main.py
```
4) Run: `dist\Receipt-Vision\Receipt-Vision.exe`

Notes:
- Use `--onefile` for a single EXE (slower startup)
- Add an icon with `--icon path\to\icon.ico`

#### Option B: Bundle UB‑Mannheim Tesseract inside the build (portable)
This produces a self-contained folder that includes Tesseract.

1) Install/download the UB‑Mannheim Windows build of Tesseract and obtain the `Tesseract-OCR` folder, which contains:
   - `Tesseract-OCR\tesseract.exe`
   - `Tesseract-OCR\tessdata\` (traineddata files)

   You can copy it from your system install or download from the UB‑Mannheim wiki: `https://github.com/UB-Mannheim/tesseract/wiki`.

2) Place the entire `Tesseract-OCR` folder in your project root (same directory as `main.py`). The app contains logic to prefer a bundled copy at runtime.

3) Build with PyInstaller, bundling the Tesseract binaries and data:
```powershell
pyinstaller --noconfirm --windowed --onedir --name Receipt-Vision ^
  --add-binary "Tesseract-OCR\tesseract.exe;Tesseract-OCR" ^
  --add-data  "Tesseract-OCR\tessdata;Tesseract-OCR\tessdata" ^
  main.py
```

4) Run: `dist\Receipt-Vision\Receipt-Vision.exe`

Notes:
- The app sets `TESSDATA_PREFIX` to the bundled `Tesseract-OCR\tessdata` folder when found
- The same flags work with `--onefile`; PyInstaller unpacks to a temp dir and the app locates it

### Troubleshooting
- **Tesseract not found**: Install it, put it on PATH, or bundle `Tesseract-OCR` as described
- **Can’t find tessdata**: Ensure `Tesseract-OCR\tessdata` is included and readable
- **OpenCV DLL warnings**: Upgrade PyInstaller and rebuild
- **SmartScreen/Antivirus flags**: Prefer `--onedir`, sign the binaries, or distribute via trusted channels
- **OCR quality issues**: Adjust preprocessing in `preprocess_image` or rescan higher-resolution images
- **Fields misdetected**: Share the OCR text snippet for the specific receipt; patterns are adjustable

### Developer
- **Connect with the developer:** [Mark-V](https://github.com/araume)

### References
- UB‑Mannheim Tesseract for Windows — `https://github.com/UB-Mannheim/tesseract/wiki`
- Developer profile — `https://github.com/araume`



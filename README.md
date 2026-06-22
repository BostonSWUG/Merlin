# 🧙 Merlin — Balloon Quantity Analyzer v1.1

Merlin analyzes mechanical assembly drawings (SOLIDWORKS PDFs and images) to automatically detect Find number balloons, interpret quantity multipliers, and produce a per-Find-number tally.

## Features

- **Vector PDF detection** — reads balloon circles and text directly from CAD-generated PDFs (no OCR needed for SOLIDWORKS exports)
- **Raster fallback** — Hough Circle Transform + Tesseract OCR for scanned images
- **Multiplier recognition** — nX, Xn, (n) PLACES, BOTH SIDES, TYP, custom phrases
- **View-level multipliers** — uses PDF bookmarks to associate BOTH SIDES with the correct drawing view
- **Multi-page support** — handles drawings with any number of sheets
- **Interactive UI** — Streamlit web app with PDF preview, balloon highlighting, and sortable tables
- **BOM comparison** — import a PLM BOM export (Excel) to compare drawing quantities against the BOM
- **CLI** — command-line interface for scripting and automation
- **JSON + tabular export**

---

## Setup Instructions (Step-by-Step)

These instructions are written for anyone on the team. The Merlin folder lives on our shared SharePoint at:

```
C:\Users\<your-alias>\OneDrive - amazon.com\Product Development Engineering - Documents\AI\Tools\Merlin
```

> Your local path will differ based on your Windows username / alias. The folder syncs automatically via OneDrive once you have access to the SharePoint library.

### Step 1 — Install Python 3.11 or newer

1. Open https://www.python.org/downloads/ and download the latest Python 3.11+ installer for Windows.
2. Run the installer. **Check the box "Add python.exe to PATH"** before clicking Install.
3. Verify by opening a terminal (Command Prompt or PowerShell) and running:
   ```
   python --version
   ```
   You should see `Python 3.11.x` or higher.

### Step 2 — Install Tesseract OCR (required for scanned images)

> If you only analyze vector SOLIDWORKS PDFs, you can skip this step. Tesseract is only needed for raster/scanned images.

1. Download the Windows installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer and accept the defaults. Note the install path (typically `C:\Program Files\Tesseract-OCR`).
3. Add Tesseract to your system PATH:
   - Open **Start → Settings → System → About → Advanced system settings → Environment Variables**.
   - Under **System variables**, find `Path`, click **Edit**, and add `C:\Program Files\Tesseract-OCR`.
   - Click OK to save.
4. Verify by opening a new terminal and running:
   ```
   tesseract --version
   ```

### Step 3 — Install Poppler (required for PDF rendering)

1. Download the latest Poppler for Windows release from: https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract the zip to a permanent location, e.g. `C:\poppler`.
3. Add the `bin` folder to your system PATH (same process as Step 2):
   - Add `C:\poppler\Library\bin` (check the exact subfolder — it should contain `pdftoppm.exe`).
4. Verify by opening a new terminal and running:
   ```
   pdftoppm -h
   ```
   You should see usage information (not "command not found").

### Step 4 — Open a terminal in the Merlin folder

1. Open File Explorer and navigate to your synced Merlin folder.
2. Click the address bar, type `cmd`, and press Enter. This opens a Command Prompt in that folder.
   - Alternatively, right-click in the folder and choose **Open in Terminal**.

### Step 5 — Create a virtual environment (one-time setup)

This keeps Merlin's dependencies isolated from other Python projects.

```
python -m venv .venv
```

### Step 6 — Activate the virtual environment

Every time you open a new terminal to use Merlin, activate the environment first:

**Command Prompt:**
```
.venv\Scripts\activate
```

**PowerShell:**
```
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the beginning of your prompt.

### Step 7 — Install Merlin and its dependencies

With the virtual environment active, run:

```
pip install -e ".[dev]"
```

This installs Merlin in editable mode so it stays up to date as the SharePoint folder syncs new versions.

### Step 8 — Launch Merlin

**Web UI (recommended):**
```
merlin
```
This starts a local Streamlit server and opens your browser to http://localhost:8501. Drag and drop a PDF to analyze.

**Command Line:**
```
balloon-analyzer drawing.pdf --format tabular
balloon-analyzer drawing.pdf --format json --output results.json
```

---

## Updating Merlin

When a new version is pushed to the SharePoint folder, OneDrive will sync the files automatically. To pick up new dependencies, re-run the install command with your virtual environment active:

```
pip install -e ".[dev]"
```

---

## Quick Reference

| Action                            | Command                                    |
|-----------------------------------|--------------------------------------------|
| Activate environment (cmd)        | `.venv\Scripts\activate`                   |
| Activate environment (PowerShell) | `.venv\Scripts\Activate.ps1`               |
| Launch web UI                     | `merlin`                                   |
| Analyze a file (CLI)              | `balloon-analyzer <file> --format tabular` |
| Re-install after update           | `pip install -e ".[dev]"`                  |
| Run tests                         | `python -m pytest tests/ -v`               |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognized | Re-run the Python installer and check "Add to PATH", then open a new terminal. |
| `merlin` not recognized | Make sure the virtual environment is activated (you see `(.venv)` in your prompt). |
| PDF rendering fails / blank images | Poppler is not installed or not on PATH. Verify with `pdftoppm -h`. |
| OCR returns garbage on scanned images | Tesseract is not installed or not on PATH. Verify with `tesseract --version`. |
| `pip install` fails with permission errors | Make sure you're inside the activated `.venv` — never install with `--user` into the global Python. |
| OneDrive sync conflicts | Close Merlin, let OneDrive finish syncing, delete any `*.conflict` files, then re-launch. |

---

## Configuration

Create a `config.json` in the Merlin folder to customize analysis parameters:

```json
{
  "proximity_radius": 50.0,
  "confidence_threshold": 0.5,
  "custom_multiplier_phrases": {
    "EACH SIDE": 2,
    "PER ASSY": 1
  }
}
```

Use it with the CLI:
```
balloon-analyzer drawing.pdf --config config.json
```

---

## Python API

```python
from balloon_quantity_analyzer.analyzer import BalloonAnalyzer
from balloon_quantity_analyzer.models import AnalyzerConfig

analyzer = BalloonAnalyzer(config=AnalyzerConfig(proximity_radius=80.0))
report = analyzer.analyze("drawing.pdf")
print(report.tally)
```

## Testing

```bash
python -m pytest tests/ -v
```

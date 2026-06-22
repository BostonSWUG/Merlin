"""
Build script for Merlin standalone distribution.

Run from the project root with the .venv activated:

    python build_merlin.py

This will:
  1. Install PyInstaller (if missing)
  2. Run PyInstaller with merlin.spec
  3. Copy bundled Poppler/Tesseract into dist/ if found on PATH
  4. Print next steps
"""

import shutil
import subprocess
import sys
from pathlib import Path

DIST_DIR = Path("dist", "Merlin")


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def run_pyinstaller() -> None:
    print("\n=== Running PyInstaller ===\n")
    subprocess.check_call([sys.executable, "-m", "PyInstaller", "merlin.spec", "--noconfirm"])


def bundle_poppler() -> None:
    """Copy Poppler binaries into dist if pdftoppm is on PATH."""
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        print("\nWARNING: Poppler (pdftoppm) not found on PATH.")
        print("  PDF rendering will not work unless the end user has Poppler installed.")
        print("  To bundle it, install Poppler and make sure pdftoppm is on PATH.\n")
        return

    poppler_bin = Path(pdftoppm).parent
    dest = DIST_DIR / "poppler" / "Library" / "bin"
    print(f"Bundling Poppler from {poppler_bin} -> {dest}")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(poppler_bin, dest)


def bundle_tesseract() -> None:
    """Copy Tesseract binaries + tessdata into dist if tesseract is on PATH."""
    tesseract = shutil.which("tesseract")
    if not tesseract:
        print("\nWARNING: Tesseract not found on PATH.")
        print("  OCR features will not work unless the end user has Tesseract installed.")
        print("  To bundle it, install Tesseract and make sure it is on PATH.\n")
        return

    tess_dir = Path(tesseract).parent
    dest = DIST_DIR / "tesseract"
    print(f"Bundling Tesseract from {tess_dir} -> {dest}")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(tess_dir, dest)


def main() -> None:
    ensure_pyinstaller()
    run_pyinstaller()
    bundle_poppler()
    bundle_tesseract()

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {DIST_DIR.resolve()}")
    print("\nTo distribute:")
    print("  1. Zip the dist/Merlin folder")
    print("  2. Share the .zip via SharePoint")
    print("  3. Users extract and double-click Merlin.exe")
    print()


if __name__ == "__main__":
    main()

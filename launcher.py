"""
Merlin Launcher — Standalone entry point for PyInstaller builds.

Starts the Streamlit server and opens the user's default browser.
"""

import multiprocessing
import os
import sys
import webbrowser
from pathlib import Path


def _get_bundle_dir() -> Path:
    """Return the directory where bundled resources live."""
    # PyInstaller sets sys._MEIPASS when running from a bundle
    if getattr(sys, "_MEIPASS", None):
        return Path(sys._MEIPASS)
    return Path(__file__).parent


def _setup_env() -> None:
    """Point PATH at bundled Poppler / Tesseract binaries (if present)."""
    bundle = _get_bundle_dir()

    poppler_bin = bundle / "poppler" / "Library" / "bin"
    if poppler_bin.exists():
        os.environ["PATH"] = str(poppler_bin) + os.pathsep + os.environ.get("PATH", "")

    tesseract_dir = bundle / "tesseract"
    if tesseract_dir.exists():
        os.environ["PATH"] = str(tesseract_dir) + os.pathsep + os.environ.get("PATH", "")
        tessdata = tesseract_dir / "tessdata"
        if tessdata.exists():
            os.environ["TESSDATA_PREFIX"] = str(tessdata)


def main() -> None:
    _setup_env()

    # Streamlit needs to find app.py — resolve from bundle
    bundle = _get_bundle_dir()
    app_path = str(bundle / "balloon_quantity_analyzer" / "app.py")

    if not os.path.isfile(app_path):
        print(f"ERROR: Cannot find app.py at {app_path}", file=sys.stderr)
        input("Press Enter to exit...")
        sys.exit(1)

    port = "8501"
    url = f"http://localhost:{port}"

    # Open browser after a short delay
    import threading
    threading.Timer(2.0, webbrowser.open, args=[url]).start()

    # Launch Streamlit
    sys.argv = [
        "streamlit", "run", app_path,
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
    ]

    from streamlit.web.cli import main as st_main
    st_main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

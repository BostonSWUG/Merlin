"""Streamlit Community Cloud entry point for Merlin.

Streamlit re-runs this file on every session and interaction. We must
*re-execute* the app module each run rather than `import` it: a plain
import only runs the module body once (Python caches it in sys.modules),
which leaves every subsequent session with a blank page.

`runpy.run_path` executes app.py fresh on every rerun, so all the
Streamlit UI commands fire each time. We add src/ to sys.path first so
the balloon_quantity_analyzer package imports resolve.
"""

import runpy
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

APP = SRC / "balloon_quantity_analyzer" / "app.py"
runpy.run_path(str(APP), run_name="__main__")

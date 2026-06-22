"""Streamlit Community Cloud entry point for Merlin.

Adds the src/ directory to the Python path so the
balloon_quantity_analyzer package is importable, then loads the app UI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Importing the module executes the Streamlit UI defined at module level.
import balloon_quantity_analyzer.app  # noqa: E402,F401

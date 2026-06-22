"""Launch the Merlin Streamlit UI."""

import sys
from pathlib import Path


def main() -> None:
    """Entry point for the `merlin` console script."""
    import streamlit.web.cli as stcli

    app_path = str(Path(__file__).parent / "app.py")
    sys.argv = ["streamlit", "run", app_path, "--server.headless=true"]
    stcli.main()


if __name__ == "__main__":
    main()

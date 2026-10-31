"""Streamlit Cloud & Hugging Face Spaces entry point for infersize."""

import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from infersize.ui.app import main  # noqa: E402

if __name__ == "__main__":
    main()

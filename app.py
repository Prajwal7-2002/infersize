"""Hugging Face Spaces & Streamlit entry point for infersize."""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from infersize.ui.app import *  # noqa: F401, F403

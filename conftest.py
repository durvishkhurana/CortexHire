"""Pytest bootstrap: put the repo root on sys.path so tests can import
``src.*``, ``rank``, and ``validate_submission`` without installation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

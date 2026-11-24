"""
Convenience Root Runner: Invokes scripts/run_all.py with all forwarded arguments.
"""

from __future__ import annotations
import runpy
from pathlib import Path
import sys

if __name__ == "__main__":
    script_path = Path(__file__).resolve().parent / "scripts" / "run_all.py"
    sys.argv[0] = str(script_path)
    runpy.run_path(str(script_path), run_name="__main__")

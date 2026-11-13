"""
Launcher for dataset audit.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_audit import run_dataset_audit

if __name__ == "__main__":
    run_dataset_audit()


"""
Convenience launcher for Roboflow dataset acquisition.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_dataset import get_roboflow_dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Roboflow Dataset for FoodWise AI")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API Key")
    parser.add_argument("--version", type=int, default=6, help="Dataset version (default: 6)")
    parser.add_argument("--format", type=str, default="yolov8", help="Export format")
    args = parser.parse_args()

    get_roboflow_dataset(api_key=args.api_key, version=args.version, export_format=args.format)


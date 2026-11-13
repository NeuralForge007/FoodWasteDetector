"""
FOODWISE AI: Evaluation Launcher Script
Runs evaluation on test or validation splits for trained YOLOv11 models.
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 Segmentation on PROJ3 Dataset")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Dataset split to evaluate")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-preds", action="store_true", help="Do not save visual prediction overlays")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        split=args.split,
        conf_threshold=args.conf,
        save_predictions=not args.no_preds
    )


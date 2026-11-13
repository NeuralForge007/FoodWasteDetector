"""
Convenience launcher for YOLOv11 model training.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_yolo import train_yolo_model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLOv11 Segmentation on PROJ3 Dataset")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt", help="YOLO model variant")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch smoke test")
    args = parser.parse_args()

    train_yolo_model(
        model_variant=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        smoke_test=args.smoke_test
    )


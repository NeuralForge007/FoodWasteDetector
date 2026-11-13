"""
FOODWISE AI: YOLOv11 Instance Segmentation Training Pipeline
Implements the training methodology from Ferreira et al. (Applied Sciences 2025):
- Synchronous geometric augmentations
- Multi-task box, classification, and instance mask loss
- Automatic checkpointing to models/best.pt and models/last.pt
- Training log and curve preservation in runs/foodwise_yolo11/
"""

import os
import sys
from pathlib import Path
import shutil
import yaml
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import get_yolo_augmentation_config

def train_yolo_model(
    model_variant: str = "yolo11n-seg.pt",
    dataset_yaml: Path = None,
    epochs: int = 30,
    batch_size: int = 4,
    imgsz: int = 640,
    lr0: float = 0.01,
    project_dir: str = "runs/foodwise_yolo11",
    experiment_name: str = "train",
    smoke_test: bool = False
):
    if dataset_yaml is None:
        dataset_yaml = PROJECT_ROOT / "data" / "processed" / "dataset.yaml"
    dataset_yaml = Path(dataset_yaml).resolve()

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Missing dataset YAML at {dataset_yaml}")

    output_models_dir = PROJECT_ROOT / "models"
    output_models_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("   FOODWISE AI: TRAINING YOLOV11 INSTANCE SEGMENTATION")
    print(f"   Model Variant:    {model_variant}")
    print(f"   Dataset Config:   {dataset_yaml}")
    print(f"   Device:           {device}")
    print(f"   Target Epochs:    {1 if smoke_test else epochs}")
    print(f"   Batch Size:       {batch_size}")
    print(f"   Image Size:       {imgsz}")
    print(f"   Learning Rate:    {lr0}")
    print("=" * 70)

    # 1. Initialize YOLO model
    model = YOLO(model_variant)

    # 2. Get augmentation hyperparameter configuration
    aug_cfg = get_yolo_augmentation_config()

    # 3. Execute training
    actual_epochs = 1 if smoke_test else epochs
    train_results = model.train(
        data=str(dataset_yaml),
        epochs=actual_epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr0,
        device=device,
        project=project_dir,
        name=experiment_name,
        exist_ok=True,
        workers=2,
        val=True,
        plots=True,
        verbose=True,
        **aug_cfg
    )

    exp_dir = Path(train_results.save_dir) if hasattr(train_results, 'save_dir') else (Path(project_dir) / experiment_name)
    best_weights = exp_dir / "weights" / "best.pt"
    last_weights = exp_dir / "weights" / "last.pt"

    print("\nTraining completed.")
    print(f"Artifacts saved to: {exp_dir}")

    # Copy checkpoints to models/
    if best_weights.exists():
        dst_best = output_models_dir / "best.pt"
        shutil.copy2(best_weights, dst_best)
        print(f"Copied best model checkpoint to: {dst_best}")

    if last_weights.exists():
        dst_last = output_models_dir / "last.pt"
        shutil.copy2(last_weights, dst_last)
        print(f"Copied last model checkpoint to: {dst_last}")

    return train_results, exp_dir

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

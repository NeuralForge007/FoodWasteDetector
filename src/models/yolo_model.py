"""
FOODWISE AI: YOLOv11 Instance Segmentation Model Wrapper
Manages model loading, device selection, and raw inference.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class FoodWiseYOLOModel:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto"
    ):
        if weights_path is None:
            # Prefer best.pt if exists, else reference weights
            best_pt = PROJECT_ROOT / "models" / "best.pt"
            ref_pt = PROJECT_ROOT / "models" / "yolov11-x-weights-v6.pt"
            if best_pt.exists():
                weights_path = str(best_pt)
            elif ref_pt.exists():
                weights_path = str(ref_pt)
            else:
                weights_path = "yolo11s-seg.pt" # Default fallback
        
        self.weights_path = str(Path(weights_path).resolve()) if Path(weights_path).exists() else weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Device selection
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.load_model()

    def load_model(self):
        """Instantiates Ultralytics YOLO model."""
        print(f"Loading YOLOv11 model from {self.weights_path} on {self.device}...")
        self.model = YOLO(self.weights_path)
        self.model.to(self.device)
        print(f"Model successfully loaded. Task: {self.model.task}, Classes: {len(self.model.names)}")

    def predict(self, source, conf: Optional[float] = None, imgsz: int = 640):
        """Runs segmentation inference."""
        conf = conf if conf is not None else self.conf_threshold
        with torch.no_grad():
            results = self.model(
                source=source,
                conf=conf,
                iou=self.iou_threshold,
                imgsz=imgsz,
                device=self.device,
                verbose=False
            )
        return results


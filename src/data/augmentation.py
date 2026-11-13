"""
FOODWISE AI: Data Augmentation Strategy for YOLOv11 Instance Segmentation
Configures synchronous image and polygon mask augmentations adhering to the research methodology:
- Horizontal flipping
- Controlled rotations
- Controlled zooming / scaling
- Perspective & Translation
Preserves food region integrity on the plate.
"""

from typing import Dict, Any

def get_yolo_augmentation_config() -> Dict[str, Any]:
    """
    Returns Ultralytics YOLOv11 augmentation hyperparameter dictionary.
    Ensures geometric transformations are synchronously applied to both image and segmentation polygons.
    """
    return {
        "fliplr": 0.5,          # 50% probability horizontal flip
        "flipud": 0.0,          # Disable vertical flip to preserve plate gravity orientation
        "degrees": 15.0,        # Controlled rotation (+/- 15 degrees)
        "translate": 0.1,       # Controlled translation (+/- 10%)
        "scale": 0.2,           # Controlled scaling/zoom (80% to 120%)
        "shear": 2.0,           # Slight shear (+/- 2 degrees)
        "perspective": 0.0001,  # Minimal perspective distortion
        "hsv_h": 0.015,         # Mild hue adjustment
        "hsv_s": 0.2,           # Mild saturation adjustment
        "hsv_v": 0.2,           # Mild brightness adjustment
        "mosaic": 0.5,          # Mosaic augmentation for multi-item plates
        "mixup": 0.0,           # Disabled mixup to avoid confusing overlapping food types
        "copy_paste": 0.1       # Segment copy-paste for rare ingredient instances
    }


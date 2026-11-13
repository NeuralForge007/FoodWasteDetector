"""
Image processing and visual utility functions for Streamlit.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_sample_images() -> List[Tuple[str, Path]]:
    """Loads a collection of available sample images from the dataset test directory."""
    test_dir = PROJECT_ROOT / "data" / "raw" / "roboflow" / "PROJ3---Food-waste-detection-6" / "test" / "images"
    sample_list = []
    if test_dir.exists():
        for p in sorted(test_dir.glob("*.jpg")):
            # Create readable friendly label
            name = p.stem.split("_")[0]
            sample_list.append((f"Test Sample: {name} ({p.name[:18]}...)", p))
    return sample_list

def create_mask_cluster_image(
    image_shape: Tuple[int, int, int],
    detections: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Creates a colorful categorical mask cluster map showing different segmented food items.
    """
    h, w = image_shape[:2]
    cluster_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Color palette for different item types
    colors = [
        (46, 204, 113),  # Green
        (231, 76, 60),   # Red
        (52, 152, 219),  # Blue
        (241, 196, 15),  # Yellow
        (155, 89, 182),  # Purple
        (230, 126, 34),  # Orange
        (26, 188, 156),  # Turquoise
        (243, 156, 18),  # Amber
    ]

    for idx, det in enumerate(detections):
        poly = det.get("polygon")
        if poly is not None and len(poly) > 2:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            color = colors[idx % len(colors)]
            cv2.fillPoly(cluster_canvas, [pts], color)
            cv2.polylines(cluster_canvas, [pts], True, (255, 255, 255), 2)

    return cluster_canvas


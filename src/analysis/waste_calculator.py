"""
FOODWISE AI: Plate-Area-Based Food Waste Calculator
Reproduces and extends the research methodology from:
Ferreira et al., "Food Waste Detection in Canteen Plates Using YOLOv11" (Applied Sciences 2025).

Research Formula:
    Waste (%) = FoodArea / (PlateArea - GarbageArea) * 100

Scientific Clarification:
    This is an image-area ratio metric representing the percentage of available plate
    surface covered by leftover food. It does NOT directly measure leftover food weight in grams
    because plate depth, food volume, density, and original served portions are not observable
    from a single post-consumption 2D image.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from src.config.category_mapping import (
    PLATE_CLASS_ID,
    GARBAGE_CLASS_IDS,
    UTENSIL_CLASS_IDS,
    is_edible_food,
    get_item_semantic_type
)

class WasteAreaCalculator:
    def __init__(self, min_plate_area: int = 1000):
        """
        Args:
            min_plate_area: Minimum pixel area to consider a plate detection valid.
        """
        self.min_plate_area = min_plate_area

    def calculate_from_masks(
        self,
        detected_objects: List[Dict[str, Any]],
        image_shape: tuple = None
    ) -> Dict[str, Any]:
        """
        Calculates food waste percentage using binary masks to prevent double-counting
        overlapping ingredient regions.

        Args:
            detected_objects: List of dicts, each containing:
                - 'class_id' (int)
                - 'class_name' (str)
                - 'confidence' (float)
                - 'area' (int or float)
                - 'mask' (optional 2D numpy boolean array)
            image_shape: (height, width) of the image.

        Returns:
            Dict containing:
                - 'waste_percentage': float (0.0 to 100.0)
                - 'food_area': int
                - 'plate_area': int
                - 'garbage_area': int
                - 'usable_plate_area': int (PlateArea - GarbageArea)
                - 'plate_detected': bool
                - 'status': str ('success', 'no_plate_detected', 'invalid_plate_area', 'zero_usable_area')
                - 'message': str
                - 'metric_name': str
        """
        metric_name = "Estimated Leftover Food Area (%)"

        if not detected_objects:
            return {
                "waste_percentage": 0.0,
                "food_area": 0,
                "plate_area": 0,
                "garbage_area": 0,
                "usable_plate_area": 0,
                "plate_detected": False,
                "status": "no_objects_detected",
                "message": "No objects were detected on the plate.",
                "metric_name": metric_name
            }

        # Check if masks are provided for overlap-free pixel calculation
        has_numpy_masks = all(
            "mask" in obj and isinstance(obj["mask"], np.ndarray)
            for obj in detected_objects
            if "mask" in obj
        ) and any("mask" in obj for obj in detected_objects)

        if has_numpy_masks and image_shape is not None:
            # Overlap-safe pixel-exact computation
            h, w = image_shape[:2]
            plate_mask = np.zeros((h, w), dtype=bool)
            garbage_mask = np.zeros((h, w), dtype=bool)
            food_mask = np.zeros((h, w), dtype=bool)

            for obj in detected_objects:
                cid = obj["class_id"]
                mask = obj.get("mask")
                if mask is None:
                    continue
                # Ensure mask matches image dimensions
                if mask.shape != (h, w):
                    import cv2
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

                if cid == PLATE_CLASS_ID:
                    plate_mask = np.logical_or(plate_mask, mask)
                elif cid in GARBAGE_CLASS_IDS:
                    garbage_mask = np.logical_or(garbage_mask, mask)
                elif is_edible_food(cid):
                    food_mask = np.logical_or(food_mask, mask)

            plate_area = int(np.sum(plate_mask))
            garbage_area = int(np.sum(garbage_mask))
            # Food area inside plate boundaries
            food_area = int(np.sum(food_mask))
        else:
            # Sum of reported areas from detections
            plate_area = 0
            garbage_area = 0
            food_area = 0

            for obj in detected_objects:
                cid = obj["class_id"]
                area = obj.get("area", 0)

                if cid == PLATE_CLASS_ID:
                    plate_area += area
                elif cid in GARBAGE_CLASS_IDS:
                    garbage_area += area
                elif is_edible_food(cid):
                    food_area += area

        # Apply Safeguards
        plate_detected = plate_area >= self.min_plate_area

        if not plate_detected:
            return {
                "waste_percentage": 0.0,
                "food_area": int(food_area),
                "plate_area": int(plate_area),
                "garbage_area": int(garbage_area),
                "usable_plate_area": 0,
                "plate_detected": False,
                "status": "no_plate_detected",
                "message": "No valid plate region detected in the photograph. Ensure the entire plate is visible.",
                "metric_name": metric_name
            }

        usable_plate_area = max(0, plate_area - garbage_area)

        if usable_plate_area <= 0:
            return {
                "waste_percentage": 0.0,
                "food_area": int(food_area),
                "plate_area": int(plate_area),
                "garbage_area": int(garbage_area),
                "usable_plate_area": 0,
                "plate_detected": True,
                "status": "zero_usable_area",
                "message": "Discarded non-edible materials cover the entire plate surface.",
                "metric_name": metric_name
            }

        # Compute research formula: FoodArea / (PlateArea - GarbageArea) * 100
        raw_percentage = (food_area / usable_plate_area) * 100.0
        # Safeguard: clamp between 0.0% and 100.0%
        waste_percentage = round(float(np.clip(raw_percentage, 0.0, 100.0)), 2)

        return {
            "waste_percentage": waste_percentage,
            "food_area": int(food_area),
            "plate_area": int(plate_area),
            "garbage_area": int(garbage_area),
            "usable_plate_area": int(usable_plate_area),
            "plate_detected": True,
            "status": "success",
            "message": f"Leftover food covers {waste_percentage:.1f}% of usable plate surface.",
            "metric_name": metric_name
        }


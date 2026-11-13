"""
Unit and Integration tests for the FoodWisePredictor pipeline.
"""

import sys
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import FoodWisePredictor

class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize predictor once
        cls.predictor = FoodWisePredictor(
            weights_path="models/yolov11-x-weights-v6.pt",
            conf_threshold=0.20
        )

    def test_predictor_on_blank_image(self):
        """Blank image should yield no reliable detections and clean fallback message."""
        blank = np.full((480, 640, 3), 255, dtype=np.uint8)
        result = self.predictor.predict(blank)
        self.assertFalse(result["has_reliable_detections"])
        self.assertIn("No reliable detection", result["message"])
        self.assertEqual(len(result["detected_objects"]), 0)
        self.assertEqual(result["waste_analysis"]["waste_percentage"], 0.0)

    def test_predictor_output_structure(self):
        """Result dictionary must contain all required UI keys."""
        dummy = Image.new("RGB", (320, 320), color=(100, 150, 200))
        result = self.predictor.predict(dummy)
        required_keys = [
            "annotated_image", "clustering_image", "detected_objects",
            "edible_items", "waste_analysis", "table_rows",
            "has_reliable_detections", "message"
        ]
        for k in required_keys:
            self.assertIn(k, result)

if __name__ == "__main__":
    unittest.main()


"""
Unit & Integration tests for the FoodWise inference pipeline.
"""
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import FoodWisePredictor

class TestInferencePipeline(unittest.TestCase):
    def setUp(self):
        # Use available model weights
        best_pt = PROJECT_ROOT / "models" / "best.pt"
        benchmark_pt = PROJECT_ROOT / "models" / "yolov11-x-weights-v6.pt"
        default_n = PROJECT_ROOT / "yolo11n-seg.pt"

        if benchmark_pt.exists():
            self.model_path = str(benchmark_pt)
        elif best_pt.exists():
            self.model_path = str(best_pt)
        else:
            self.model_path = str(default_n)
            
        self.predictor = FoodWisePredictor(weights_path=self.model_path, conf_threshold=0.25)

    def test_pipeline_on_synthetic_blank_image(self):
        # Blank black image should return no detections or safe empty analysis
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        res = self.predictor.predict(blank)

        self.assertIn("annotated_image", res)
        self.assertIn("clustering_image", res)
        self.assertIn("waste_analysis", res)
        self.assertIn("table_rows", res)
        self.assertFalse(res["waste_analysis"]["plate_detected"])
        self.assertEqual(res["waste_analysis"]["waste_percentage"], 0.0)

    def test_pipeline_on_test_dataset_image(self):
        test_dir = PROJECT_ROOT / "data" / "raw" / "roboflow" / "PROJ3---Food-waste-detection-6" / "test" / "images"
        test_images = list(test_dir.glob("*.jpg"))
        if not test_images:
            self.skipTest("No test images found in dataset folder.")

        img_path = test_images[0]
        res = self.predictor.predict(img_path)

        self.assertIn("annotated_image", res)
        self.assertIn("waste_analysis", res)
        self.assertIsInstance(res["annotated_image"], Image.Image)
        self.assertGreaterEqual(res["waste_analysis"]["waste_percentage"], 0.0)
        self.assertLessEqual(res["waste_analysis"]["waste_percentage"], 100.0)

if __name__ == "__main__":
    unittest.main()


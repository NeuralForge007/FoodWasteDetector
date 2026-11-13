"""
Unit tests for the Plate-Area-Based Food Waste Calculator
Verifies the research formula and all 9 safeguards.
"""

import sys
from pathlib import Path
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.waste_calculator import WasteAreaCalculator
from src.config.category_mapping import PLATE_CLASS_ID, GARBAGE_CLASS_IDS

class TestWasteCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = WasteAreaCalculator(min_plate_area=100)

    def test_normal_calculation(self):
        """Standard valid plate with food and no garbage."""
        objs = [
            {"class_id": PLATE_CLASS_ID, "class_name": "plate", "area": 1000},
            {"class_id": 63, "class_name": "rice", "area": 300},
            {"class_id": 20, "class_name": "chicken", "area": 100}
        ]
        res = self.calc.calculate_from_masks(objs)
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["plate_detected"])
        self.assertEqual(res["plate_area"], 1000)
        self.assertEqual(res["food_area"], 400)
        self.assertEqual(res["garbage_area"], 0)
        self.assertEqual(res["usable_plate_area"], 1000)
        # 400 / 1000 * 100 = 40.0%
        self.assertAlmostEqual(res["waste_percentage"], 40.0)

    def test_calculation_with_garbage(self):
        """Plate with garbage deducted from denominator."""
        objs = [
            {"class_id": PLATE_CLASS_ID, "class_name": "plate", "area": 1000},
            {"class_id": 35, "class_name": "garbage", "area": 200}, # Garbage
            {"class_id": 63, "class_name": "rice", "area": 400}
        ]
        res = self.calc.calculate_from_masks(objs)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["usable_plate_area"], 800) # 1000 - 200
        # 400 / 800 * 100 = 50.0%
        self.assertAlmostEqual(res["waste_percentage"], 50.0)

    def test_no_plate_safeguard(self):
        """Missing plate should trigger graceful fallback."""
        objs = [
            {"class_id": 63, "class_name": "rice", "area": 400}
        ]
        res = self.calc.calculate_from_masks(objs)
        self.assertFalse(res["plate_detected"])
        self.assertEqual(res["status"], "no_plate_detected")
        self.assertEqual(res["waste_percentage"], 0.0)

    def test_plate_too_small_safeguard(self):
        """Plate below min_plate_area should be considered invalid."""
        objs = [
            {"class_id": PLATE_CLASS_ID, "class_name": "plate", "area": 50}, # Below 100
            {"class_id": 63, "class_name": "rice", "area": 20}
        ]
        res = self.calc.calculate_from_masks(objs)
        self.assertFalse(res["plate_detected"])

    def test_division_by_zero_prevention(self):
        """Garbage >= PlateArea should not crash with ZeroDivisionError."""
        objs = [
            {"class_id": PLATE_CLASS_ID, "class_name": "plate", "area": 500},
            {"class_id": 35, "class_name": "garbage", "area": 500},
            {"class_id": 63, "class_name": "rice", "area": 100}
        ]
        res = self.calc.calculate_from_masks(objs)
        self.assertEqual(res["usable_plate_area"], 0)
        self.assertEqual(res["waste_percentage"], 0.0)
        self.assertEqual(res["status"], "zero_usable_area")

    def test_clamping_safeguard(self):
        """Food area exceeding usable plate area must be clamped to 100%."""
        objs = [
            {"class_id": PLATE_CLASS_ID, "class_name": "plate", "area": 500},
            {"class_id": 63, "class_name": "rice", "area": 600} # Piled up food
        ]
        res = self.calc.calculate_from_masks(objs)
        self.assertEqual(res["waste_percentage"], 100.0)

    def test_mask_overlap_safety(self):
        """Two overlapping food detections should not double count common pixels."""
        shape = (100, 100)
        plate_mask = np.ones(shape, dtype=bool)
        
        # Overlapping food masks
        mask1 = np.zeros(shape, dtype=bool)
        mask1[20:60, 20:60] = True # 40x40 = 1600 px
        
        mask2 = np.zeros(shape, dtype=bool)
        mask2[40:80, 40:80] = True # 40x40 = 1600 px (with 20x20=400 overlap)

        objs = [
            {"class_id": PLATE_CLASS_ID, "class_name": "plate", "mask": plate_mask, "area": 10000},
            {"class_id": 63, "class_name": "rice", "mask": mask1, "area": 1600},
            {"class_id": 20, "class_name": "chicken", "mask": mask2, "area": 1600}
        ]
        res = self.calc.calculate_from_masks(objs, image_shape=shape)
        # Expected unique food union = 1600 + 1600 - 400 = 2800
        self.assertEqual(res["food_area"], 2800)
        # 2800 / 10000 * 100 = 28.0%
        self.assertAlmostEqual(res["waste_percentage"], 28.0)

if __name__ == "__main__":
    unittest.main()


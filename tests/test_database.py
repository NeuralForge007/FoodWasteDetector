"""
Unit tests for the SQLite Database Manager.
"""

import sys
import unittest
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.database import FoodWiseDatabase

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_file.name
        self.temp_file.close()
        self.db = FoodWiseDatabase(db_path=self.db_path)

    def tearDown(self):
        p = Path(self.db_path)
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    def test_insert_and_get_records(self):
        record_id = self.db.insert_record(
            image_filename="plate_test.jpg",
            detected_classes=["rice", "chicken"],
            confidence_scores={"rice": 0.89, "chicken": 0.94},
            categories=["Cereals and Starches", "Protein and Meat"],
            waste_percentage=35.5,
            food_area=3550,
            plate_area=10000,
            garbage_area=0,
            model_version="YOLO11x-seg-v6"
        )
        self.assertIsNotNone(record_id)
        self.assertGreater(record_id, 0)

        records = self.db.get_recent_records(limit=10)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["image_filename"], "plate_test.jpg")
        self.assertIn("rice", records[0]["detected_classes"])
        self.assertAlmostEqual(records[0]["waste_percentage"], 35.5)

    def test_analytics_aggregation(self):
        self.db.insert_record(
            image_filename="plate_1.jpg",
            detected_classes=["rice"],
            confidence_scores={"rice": 0.9},
            categories=["Cereals and Starches"],
            waste_percentage=40.0,
            food_area=4000,
            plate_area=10000,
            garbage_area=0
        )
        self.db.insert_record(
            image_filename="plate_2.jpg",
            detected_classes=["apple"],
            confidence_scores={"apple": 0.95},
            categories=["Fruits"],
            waste_percentage=20.0,
            food_area=2000,
            plate_area=10000,
            garbage_area=0
        )

        analytics = self.db.get_analytics_summary()
        self.assertEqual(analytics["total_scans"], 2)
        self.assertAlmostEqual(analytics["avg_waste_percentage"], 30.0)
        self.assertEqual(analytics["class_frequency"]["rice"], 1)
        self.assertEqual(analytics["class_frequency"]["apple"], 1)
        self.assertEqual(analytics["category_frequency"]["Fruits"], 1)

if __name__ == "__main__":
    unittest.main()


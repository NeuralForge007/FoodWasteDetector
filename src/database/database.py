"""
FOODWISE AI: SQLite Database Manager
Provides ACID-compliant relational persistence for inference audit logs,
food items, waste percentages, and longitudinal analytics.
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class FoodWiseDatabase:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(PROJECT_ROOT / "data" / "foodwise.db")
        self.db_path = str(Path(db_path).resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self):
        """Initializes tables for inference records and audit metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_filename TEXT NOT NULL,
                    detected_classes TEXT NOT NULL,       -- JSON list of strings
                    confidence_scores TEXT NOT NULL,      -- JSON dict {class: conf}
                    categories TEXT NOT NULL,             -- JSON list of broad categories
                    waste_percentage REAL NOT NULL,       -- Estimated leftover area %
                    food_area INTEGER NOT NULL,           -- Total edible food pixels
                    plate_area INTEGER NOT NULL,          -- Plate area in pixels
                    garbage_area INTEGER NOT NULL,        -- Non-edible / garbage pixels
                    model_version TEXT NOT NULL,          -- Model architecture / checkpoint
                    notes TEXT
                )
            """)
            conn.commit()

    def insert_record(
        self,
        image_filename: str,
        detected_classes: List[str],
        confidence_scores: Dict[str, float],
        categories: List[str],
        waste_percentage: float,
        food_area: int,
        plate_area: int,
        garbage_area: int,
        model_version: str = "YOLOv11-seg",
        notes: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> int:
        """Inserts a new inference scan record and returns its ID."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cls_json = json.dumps(detected_classes, ensure_ascii=False)
        conf_json = json.dumps(confidence_scores, ensure_ascii=False)
        cat_json = json.dumps(categories, ensure_ascii=False)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO inference_logs (
                    timestamp, image_filename, detected_classes,
                    confidence_scores, categories, waste_percentage,
                    food_area, plate_area, garbage_area, model_version, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, image_filename, cls_json,
                conf_json, cat_json, float(waste_percentage),
                int(food_area), int(plate_area), int(garbage_area),
                model_version, notes
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent inference records ordered by latest first."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM inference_logs ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["detected_classes"] = json.loads(item["detected_classes"])
                item["confidence_scores"] = json.loads(item["confidence_scores"])
                item["categories"] = json.loads(item["categories"])
                results.append(item)
            return results

    def get_record_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single record by its primary key ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM inference_logs WHERE id = ?", (record_id,))
            row = cursor.fetchone()
            if not row:
                return None
            item = dict(row)
            item["detected_classes"] = json.loads(item["detected_classes"])
            item["confidence_scores"] = json.loads(item["confidence_scores"])
            item["categories"] = json.loads(item["categories"])
            return item

    def delete_record(self, record_id: int) -> bool:
        """Deletes a specific record by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM inference_logs WHERE id = ?", (record_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_all(self):
        """Clears all records from the table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM inference_logs")
            conn.commit()

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Computes aggregate analytics across all recorded plate scans."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total_count FROM inference_logs")
            total_count = cursor.fetchone()["total_count"]

            if total_count == 0:
                return {
                    "total_scans": 0,
                    "avg_waste_percentage": 0.0,
                    "total_food_pixels": 0,
                    "class_frequency": {},
                    "category_frequency": {},
                    "timeline": []
                }

            cursor.execute("""
                SELECT 
                    AVG(waste_percentage) as avg_waste,
                    SUM(food_area) as sum_food_area
                FROM inference_logs
            """)
            row = cursor.fetchone()
            avg_waste = float(row["avg_waste"] or 0.0)
            sum_food_area = int(row["sum_food_area"] or 0)

            cursor.execute("SELECT timestamp, detected_classes, categories, waste_percentage FROM inference_logs ORDER BY id ASC")
            all_rows = cursor.fetchall()

            class_freq = {}
            cat_freq = {}
            timeline = []

            for r in all_rows:
                classes = json.loads(r["detected_classes"])
                cats = json.loads(r["categories"])
                for c in classes:
                    class_freq[c] = class_freq.get(c, 0) + 1
                for cat in cats:
                    cat_freq[cat] = cat_freq.get(cat, 0) + 1
                timeline.append({
                    "timestamp": r["timestamp"],
                    "waste_percentage": r["waste_percentage"]
                })

            return {
                "total_scans": total_count,
                "avg_waste_percentage": round(avg_waste, 2),
                "total_food_pixels": sum_food_area,
                "class_frequency": class_freq,
                "category_frequency": cat_freq,
                "timeline": timeline
            }


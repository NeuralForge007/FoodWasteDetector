"""
Unit tests for Streamlit application components and module imports.
"""
import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class TestStreamlitAppModules(unittest.TestCase):
    def test_imports(self):
        try:
            import app.components.ui_elements as ui
            import app.components.image_utils as iu
            import app.pages.home as p1
            import app.pages.detection as p2
            import app.pages.analytics as p3
            import app.pages.model_info as p4
            import app.pages.sdg_impact as p5
        except ImportError as e:
            self.fail(f"Failed to import app modules: {e}")

    def test_waste_badge_rendering(self):
        from app.components.ui_elements import render_waste_badge
        badge_low = render_waste_badge(12.5)
        self.assertIn("Low Waste", badge_low)

        badge_med = render_waste_badge(30.0)
        self.assertIn("Moderate Waste", badge_med)

        badge_high = render_waste_badge(65.0)
        self.assertIn("High Waste", badge_high)

if __name__ == "__main__":
    unittest.main()

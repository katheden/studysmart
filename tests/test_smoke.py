import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


class StudySmartSmokeTests(unittest.TestCase):
    def test_template_coaching_returns_text(self):
        from nlp_coach import get_coaching

        text = get_coaching(
            risk_label="Medium Risk",
            confidence=0.72,
            top_features=[("G2", 0.40), ("absences", 0.12), ("study_time", 0.08)],
            student_goal="improve my final grade",
            use_api=False,
            habit_context={"sleep_hours": 7, "social_media_hours": 2, "motivation_level": "medium"},
        )

        self.assertIn("Medium Risk", text)
        self.assertIn("Recommendations", text)
        self.assertGreater(len(text), 100)

    def test_saved_model_metadata_exists(self):
        meta_path = os.path.join(ROOT, "models", "model_meta.json")
        self.assertTrue(os.path.exists(meta_path))


if __name__ == "__main__":
    unittest.main()

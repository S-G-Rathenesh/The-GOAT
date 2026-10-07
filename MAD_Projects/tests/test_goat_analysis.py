import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from goat_analysis import GoatAnalyzer


class GoatAnalysisWorkflowTest(unittest.TestCase):
    def test_project_flow_includes_registration_and_anomaly_fields(self):
        project_root = Path(__file__).resolve().parents[1]
        dataset_dir = project_root / "Goat_Dataset"
        image_dir = dataset_dir / "healthy_goat"
        image_path = next(p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"})

        analyzer = GoatAnalyzer(data_dir=dataset_dir)
        result = analyzer.analyze_goat(
            str(image_path),
            breed="Boer",
            age_months=12,
            sex="Male",
            observed_weight_kg=30.0,
            goat_id="G-101",
            owner_name="Demo Farmer",
        )

        self.assertTrue(hasattr(result, "fair_trade_status"))
        self.assertTrue(hasattr(result, "expected_range_kg"))
        self.assertTrue(hasattr(result, "goat_id"))
        self.assertTrue(hasattr(result, "owner_name"))
        self.assertIn(result.fair_trade_status, ["Fair trade", "Review needed", "High anomaly"])

    def test_estimate_expected_weight_builds_regressor_when_missing(self):
        project_root = Path(__file__).resolve().parents[1]
        dataset_dir = project_root / "Goat_Dataset"
        image_dir = dataset_dir / "healthy_goat"
        image_path = next(p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"})

        analyzer = GoatAnalyzer(data_dir=dataset_dir)
        analyzer.regressor = None
        analyzer.classifier = None
        features = analyzer._extract_features_from_image(str(image_path))

        weight = analyzer.estimate_expected_weight(features, breed="Boer", age_months=12, sex="Male")
        self.assertGreater(weight, 0)


if __name__ == "__main__":
    unittest.main()

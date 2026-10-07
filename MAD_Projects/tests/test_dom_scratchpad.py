"""
DOM Scratchpad & Automated Functionality Verification for Goat Fair-Weight Analyzer.
Uses Streamlit's official AppTest framework to interact with DOM elements,
trigger widget actions, step through all 8 pages, and verify outputs.
"""

import sys
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from goat_analysis import DATA_DIR, GoatAnalyzer


class DomScratchpadWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.app_path = str(ROOT / "app.py")

    def test_complete_dom_workflow(self):
        print("\n" + "=" * 70)
        print("DOM SCRATCHPAD: INITIATING FULL WORKFLOW VERIFICATION")
        print("=" * 70)

        at = AppTest.from_file(self.app_path)
        at.run(timeout=15)
        self.assertFalse(at.exception, f"App crashed on load: {at.exception}")

        # -------------------------------------------------------------
        # STEP 0: Goat Profile
        # -------------------------------------------------------------
        print("\n[STEP 0: Goat Profile]")
        self.assertEqual(at.session_state.current_step, 0)
        subheaders = [s.value for s in at.subheader]
        self.assertIn("Goat profile", subheaders)

        # Inspect and set DOM inputs
        goat_id_input = at.text_input[0]
        owner_input = at.text_input[1]
        breed_select = at.selectbox[0]
        sex_select = at.selectbox[1]
        age_slider = at.slider[0]
        weight_input = at.number_input[0]

        print(f"  DOM Element [Goat ID]: {goat_id_input.value}")
        print(f"  DOM Element [Owner]: {owner_input.value}")
        print(f"  DOM Element [Breed]: {breed_select.value}")
        print(f"  DOM Element [Sex]: {sex_select.value}")
        print(f"  DOM Element [Age]: {age_slider.value} months")
        print(f"  DOM Element [Observed Weight]: {weight_input.value} kg")

        # Mutate an input to test interactivity
        goat_id_input.input("G-999")
        owner_input.input("Farmer Rajesh")
        at.run(timeout=10)
        self.assertEqual(at.session_state.goat_id, "G-999")
        self.assertEqual(at.session_state.owner_name, "Farmer Rajesh")
        print("  [OK] Profile inputs successfully updated and verified in state.")

        # Click Next button
        next_button = next(b for b in at.button if b.label == "Next")
        next_button.click().run(timeout=10)
        self.assertFalse(at.exception, f"Exception on navigating to Step 1: {at.exception}")

        # -------------------------------------------------------------
        # STEP 1: Image Capture
        # -------------------------------------------------------------
        print("\n[STEP 1: Image Capture]")
        self.assertEqual(at.session_state.current_step, 1)
        subheaders = [s.value for s in at.subheader]
        self.assertIn("Image Capture", subheaders)

        # Verify that sample buttons have been removed
        sample_btns = [b for b in at.button if "Use Healthy Goat" in b.label or "Use Unhealthy Goat" in b.label]
        self.assertEqual(len(sample_btns), 0, "Dataset sample buttons must not be present in Image Capture DOM")

        # Verify that file_uploader and camera_input are rendered in DOM
        self.assertGreaterEqual(len(at.get("file_uploader")), 1, "File uploader should be present in DOM")
        self.assertGreaterEqual(len(at.get("camera_input")), 1, "Camera input should be present in DOM")
        print("  [OK] Confirmed only Upload and Camera sessions are provided in Image Capture.")

        # Simulate image capture / upload
        dataset_dir = ROOT / "Goat_Dataset" / "healthy_goat"
        sample_img_path = next(p for p in sorted(dataset_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        at.session_state.image_path = str(sample_img_path)
        at.run(timeout=10)

        self.assertIsNotNone(at.session_state.image_path)
        self.assertTrue(Path(at.session_state.image_path).exists())
        print(f"  [OK] Image set for analysis: {Path(at.session_state.image_path).name}")

        # Advance to step 2 using "Continue to goat detection" or Next
        cont_btn = next((b for b in at.button if "Continue to goat detection" in b.label), None)
        if cont_btn:
            cont_btn.click().run(timeout=10)
        else:
            next(b for b in at.button if b.label == "Next").click().run(timeout=10)
        self.assertFalse(at.exception, f"Exception advancing to Step 2: {at.exception}")

        # -------------------------------------------------------------
        # STEP 2: Goat Detection & Segmentation
        # -------------------------------------------------------------
        print("\n[STEP 2: Goat Detection & Segmentation]")
        self.assertEqual(at.session_state.current_step, 2)
        subheaders = [s.value for s in at.subheader]
        self.assertIn("Goat Detection & Segmentation", subheaders)
        self.assertTrue(any("segmentation complete" in s.value.lower() for s in at.success))
        print("  [OK] Segmentation rendered and verified via DOM success banner.")

        next(b for b in at.button if b.label == "Next").click().run(timeout=10)
        self.assertFalse(at.exception, f"Exception advancing to Step 3: {at.exception}")

        # -------------------------------------------------------------
        # STEP 3: Body Feature Extraction
        # -------------------------------------------------------------
        print("\n[STEP 3: Body Feature Extraction]")
        self.assertEqual(at.session_state.current_step, 3)
        subheaders = [s.value for s in at.subheader]
        self.assertIn("Body Feature Extraction", subheaders)
        self.assertIsNotNone(at.session_state.current_features)
        features = at.session_state.current_features

        print(f"  Extracted Morphometrics:")
        print(f"    - Body Length: {features['body_length_cm']:.2f} cm")
        print(f"    - Body Height: {features['body_height_cm']:.2f} cm")
        print(f"    - Chest Girth: {features['chest_girth_cm']:.2f} cm")
        print(f"    - Body Fullness: {features['body_fullness']:.3f}")
        print(f"    - Aspect Ratio: {features['aspect_ratio']:.3f}")

        self.assertGreater(features['body_length_cm'], 0)
        self.assertGreater(features['body_height_cm'], 0)
        self.assertGreater(features['chest_girth_cm'], 0)
        self.assertGreater(features['body_fullness'], 0)

        next(b for b in at.button if b.label == "Next").click().run(timeout=10)
        self.assertFalse(at.exception, f"Exception advancing to Step 4: {at.exception}")

        # -------------------------------------------------------------
        # STEP 4: Breed-Aware Weight Estimation
        # -------------------------------------------------------------
        print("\n[STEP 4: Breed-Aware Weight Estimation]")
        self.assertEqual(at.session_state.current_step, 4)
        subheaders = [s.value for s in at.subheader]
        self.assertIn("Breed-Aware Weight Estimation", subheaders)

        metrics = at.metric
        self.assertGreaterEqual(len(metrics), 2)
        print(f"  DOM Metric [{metrics[0].label}]: {metrics[0].value}")
        print(f"  DOM Metric [{metrics[1].label}]: {metrics[1].value}")

        next(b for b in at.button if b.label == "Next").click().run(timeout=10)
        self.assertFalse(at.exception, f"Exception advancing to Step 5: {at.exception}")

        # -------------------------------------------------------------
        # STEP 5: Temporary Weight Anomaly Detection
        # -------------------------------------------------------------
        print("\n[STEP 5: Temporary Weight Anomaly Detection]")
        self.assertEqual(at.session_state.current_step, 5)
        subheaders = [s.value for s in at.subheader]
        self.assertIn("Temporary Weight Anomaly Detection", subheaders)

        anomaly_metrics = {m.label: m.value for m in at.metric}
        print(f"  DOM Metrics in Anomaly Step:")
        for label, val in anomaly_metrics.items():
            print(f"    - {label}: {val}")

        self.assertIn("Temporary Weight Anomaly Score", anomaly_metrics)

        next(b for b in at.button if b.label == "Next").click().run(timeout=10)
        self.assertFalse(at.exception, f"Exception advancing to Step 6: {at.exception}")

        # -------------------------------------------------------------
        # STEP 6: Result & Fair-Trade Analysis
        # -------------------------------------------------------------
        print("\n[STEP 6: Result & Fair-Trade Analysis]")
        self.assertEqual(at.session_state.current_step, 6)
        subheaders = [s.value for s in at.subheader]
        self.assertIn("Result & Fair-Trade Analysis", subheaders)

        self.assertIsNotNone(at.session_state.analysis_result)
        res = at.session_state.analysis_result
        print(f"  Analysis Result Summary:")
        print(f"    - Goat ID: {res.goat_id}")
        print(f"    - Owner: {res.owner_name}")
        print(f"    - Health Label: {res.health_label} ({res.confidence*100:.1f}%)")
        print(f"    - Fair-Trade Status: {res.fair_trade_status}")
        print(f"    - Estimated Weight: {res.estimated_weight_kg:.2f} kg")
        print(f"    - Expected Range: {res.expected_range_kg[0]:.2f} - {res.expected_range_kg[1]:.2f} kg")
        print(f"    - Anomaly Score: {res.anomaly_score:.2f}/100")

        self.assertEqual(res.goat_id, "G-999")
        self.assertEqual(res.owner_name, "Farmer Rajesh")
        self.assertIn(res.health_label, ["Healthy Goat", "Unhealthy Goat"])
        self.assertIn(res.fair_trade_status, ["Fair trade", "Review needed", "High anomaly"])

        next(b for b in at.button if b.label == "Next").click().run(timeout=10)
        self.assertFalse(at.exception, f"Exception advancing to Step 7: {at.exception}")

        # -------------------------------------------------------------
        # STEP 7: History & Reports
        # -------------------------------------------------------------
        print("\n[STEP 7: History & Reports]")
        self.assertEqual(at.session_state.current_step, 7)
        subheaders = [s.value for s in at.subheader]
        self.assertIn("History & Reports", subheaders)

        # Check DataFrame in DOM
        dataframes = at.dataframe
        self.assertGreaterEqual(len(dataframes), 1, "Report DataFrame not found in DOM")
        df_value = dataframes[0].value
        print(f"  [OK] Report DataFrame rendered ({len(df_value)} rows, {len(df_value.columns)} columns)")
        print(f"    Columns: {list(df_value.columns)}")

        # Check Download Button in DOM
        download_btns = at.get("download_button")
        self.assertGreaterEqual(len(download_btns), 1, "Download button not found in DOM")
        dl_btn = download_btns[0]
        print(f"  [OK] Download button verified: '{dl_btn.label}'")

        print("\n" + "=" * 70)
        print("DOM SCRATCHPAD: ALL 8 WORKFLOW STEPS VERIFIED SUCCESSFULLY!")
        print("=" * 70 + "\n")

    def test_image_capture_tabs_only_upload_and_camera(self):
        at = AppTest.from_file(self.app_path)
        at.session_state.current_step = 1
        at.run(timeout=10)
        self.assertFalse(at.exception, f"App exception: {at.exception}")

        # Ensure no sample dataset buttons exist
        sample_btns = [b for b in at.button if "Use Healthy" in b.label or "Use Unhealthy" in b.label]
        self.assertEqual(len(sample_btns), 0, "No sample dataset buttons should exist")

        # Ensure upload and camera input exist
        self.assertGreaterEqual(len(at.get("file_uploader")), 1)
        self.assertGreaterEqual(len(at.get("camera_input")), 1)

    def test_history_and_result_tabs_without_image_render_cleanly(self):
        # Test Step 6 without image
        at = AppTest.from_file(self.app_path)
        at.session_state.current_step = 6
        at.session_state.image_path = None
        at.run(timeout=10)
        self.assertFalse(at.exception, f"Step 6 crashed without image: {at.exception}")
        self.assertTrue(any("Goat image is required" in w.value for w in at.warning))

        # Test Step 7 without image or records
        at2 = AppTest.from_file(self.app_path)
        at2.session_state.current_step = 7
        at2.session_state.image_path = None
        at2.session_state.analysis_result = None
        at2.session_state.history_list = []
        at2.run(timeout=10)
        self.assertFalse(at2.exception, f"Step 7 crashed without records: {at2.exception}")
        self.assertTrue(any("No report generated yet" in inf.value for inf in at2.info))


if __name__ == "__main__":
    unittest.main()


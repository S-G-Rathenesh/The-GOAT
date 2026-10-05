from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).resolve().parents[1] / "Goat_Dataset"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_BREEDS = ["Boer", "Jamnapari", "Saanen", "Local Goat"]


@dataclass
class GoatResult:
    health_label: str
    confidence: float
    estimated_weight_kg: float
    expected_weight_range: Tuple[float, float]
    expected_range_kg: Tuple[float, float]
    deviation_pct: float
    anomaly_score: float
    status: str
    fair_trade_status: str
    body_length_cm: float
    body_height_cm: float
    chest_girth_cm: float
    body_shape_index: float
    body_fullness: float
    goat_id: str = ""
    owner_name: str = ""
    features: Dict[str, float] = None

    def __post_init__(self):
        if self.features is None:
            self.features = {}


class GoatAnalyzer:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.classifier_path = MODEL_DIR / "goat_classifier.joblib"
        self.regressor_path = MODEL_DIR / "goat_regressor.joblib"
        self.classifier = None
        self.regressor = None

    def _extract_features_from_image(self, image_path: str) -> Dict[str, float]:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        height, width = gray.shape
        if height == 0 or width == 0:
            raise ValueError(f"Image is empty: {image_path}")

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 150)
        edge_density = np.mean(edges > 0)

        brightness = np.mean(gray)
        contrast = np.std(gray)
        mean_h = np.mean(hsv[:, :, 0])
        mean_s = np.mean(hsv[:, :, 1])
        mean_v = np.mean(hsv[:, :, 2])

        threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        body_mask = cv2.compare(blur, threshold, cv2.CMP_GT)
        body_mask = cv2.medianBlur(body_mask.astype(np.uint8), 5)
        contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            area = cv2.contourArea(largest)
            body_fullness = area / max(w * h, 1)
            aspect_ratio = w / max(h, 1)
            body_length = w * 0.1
            body_height = h * 0.1
            chest_girth = max(20.0, (w + h) * 0.12)
        else:
            body_fullness = 0.38
            aspect_ratio = width / max(height, 1)
            body_length = width * 0.09
            body_height = height * 0.08
            chest_girth = max(18.0, (width + height) * 0.12 / 10)

        features = {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "edge_density": float(edge_density),
            "mean_h": float(mean_h),
            "mean_s": float(mean_s),
            "mean_v": float(mean_v),
            "aspect_ratio": float(aspect_ratio),
            "body_fullness": float(body_fullness),
            "body_length_cm": float(body_length),
            "body_height_cm": float(body_height),
            "chest_girth_cm": float(chest_girth),
        }
        return features

    def _load_dataset(self) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for folder_name, label in (("healthy_goat", 1), ("unhealthy_goat", 0)):
            folder = self.data_dir / folder_name
            if not folder.exists():
                continue
            for file in sorted(folder.iterdir()):
                if file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}:
                    continue
                try:
                    feature_dict = self._extract_features_from_image(str(file))
                    row = {"image_path": str(file), "label": label, **feature_dict}
                    rows.append(row)
                except Exception:
                    continue

        if not rows:
            raise ValueError(f"No valid images were found in {self.data_dir}")

        return pd.DataFrame(rows)

    def build_models(self) -> Dict[str, object]:
        df = self._load_dataset()
        X = df[[
            "brightness", "contrast", "edge_density", "mean_h", "mean_s", "mean_v",
            "aspect_ratio", "body_fullness", "body_length_cm", "body_height_cm", "chest_girth_cm"
        ]].copy()
        y = df["label"].astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        classifier = RandomForestClassifier(n_estimators=220, random_state=42, max_depth=8)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Synthetic regression targets for approximate weight estimation.
        synthetic_weight = (
            25
            + (X["body_fullness"] * 45)
            + (X["body_length_cm"] * 0.42)
            + (X["body_height_cm"] * 0.35)
            + (X["chest_girth_cm"] * 0.28)
            + (X["contrast"] / 15)
        )
        regressor = LinearRegression()
        regressor.fit(X, synthetic_weight)
        pred = regressor.predict(X_test)
        mae = mean_absolute_error(synthetic_weight.iloc[X_test.index], pred)
        r2 = r2_score(synthetic_weight.iloc[X_test.index], pred)

        self.classifier = classifier
        self.regressor = regressor
        joblib.dump(classifier, self.classifier_path)
        joblib.dump(regressor, self.regressor_path)
        return {
            "classifier_accuracy": float(acc),
            "regression_mae": float(mae),
            "regression_r2": float(r2),
        }

    def load_models(self):
        if self.classifier is None:
            if self.classifier_path.exists():
                self.classifier = joblib.load(self.classifier_path)
            else:
                self.build_models()
                self.classifier = joblib.load(self.classifier_path)

        if self.regressor is None:
            if self.regressor_path.exists():
                self.regressor = joblib.load(self.regressor_path)
            else:
                self.build_models()
                self.regressor = joblib.load(self.regressor_path)

    def estimate_expected_weight(self, features: Dict[str, float], breed: str, age_months: int, sex: str) -> float:
        if self.regressor is None or self.classifier is None:
            self.load_models()

        feature_vector = np.array([
            features["brightness"],
            features["contrast"],
            features["edge_density"],
            features["mean_h"],
            features["mean_s"],
            features["mean_v"],
            features["aspect_ratio"],
            features["body_fullness"],
            features["body_length_cm"],
            features["body_height_cm"],
            features["chest_girth_cm"],
        ], dtype=float).reshape(1, -1)

        base_weight = float(self.regressor.predict(feature_vector)[0])
        breed_adjustment = {
            "Boer": 1.16,
            "Jamnapari": 1.12,
            "Saanen": 1.08,
            "Local Goat": 1.00,
        }.get(breed, 1.0)
        age_adjustment = 1 + max(0, age_months - 12) * 0.003
        sex_adjustment = 1.04 if sex.lower() == "male" else 0.98
        return float(base_weight * breed_adjustment * age_adjustment * sex_adjustment)

    def _anomaly_score(self, estimated_weight: float, observed_weight: float, fullness: float) -> float:
        deviation_ratio = abs(observed_weight - estimated_weight) / max(estimated_weight, 1.0)
        fullness_index = max(0.0, fullness - 0.45) / 0.55
        score = 100 * (0.7 * deviation_ratio + 0.3 * fullness_index)
        return float(max(0.0, min(100.0, score)))

    def analyze_goat(
        self,
        image_path: str,
        breed: str = "Boer",
        age_months: int = 12,
        sex: str = "Male",
        observed_weight_kg: float = 0.0,
        goat_id: str = "",
        owner_name: str = "",
    ) -> GoatResult:
        if self.classifier is None or self.regressor is None:
            self.load_models()

        features = self._extract_features_from_image(image_path)
        feature_vector = np.array([
            features["brightness"],
            features["contrast"],
            features["edge_density"],
            features["mean_h"],
            features["mean_s"],
            features["mean_v"],
            features["aspect_ratio"],
            features["body_fullness"],
            features["body_length_cm"],
            features["body_height_cm"],
            features["chest_girth_cm"],
        ], dtype=float).reshape(1, -1)

        predicted_label = int(self.classifier.predict(feature_vector)[0])
        probability = np.max(self.classifier.predict_proba(feature_vector)[0])
        health_label = "Healthy Goat" if predicted_label == 1 else "Unhealthy Goat"

        estimated_weight = self.estimate_expected_weight(features, breed, age_months, sex)
        deviation_pct = 0.0 if observed_weight_kg <= 0 else abs(observed_weight_kg - estimated_weight) / max(estimated_weight, 1.0) * 100
        anomaly_score = self._anomaly_score(estimated_weight, observed_weight_kg if observed_weight_kg > 0 else estimated_weight, features["body_fullness"])

        expected_lower = estimated_weight * 0.9
        expected_upper = estimated_weight * 1.1
        expected_range = (round(expected_lower, 2), round(expected_upper, 2))

        if anomaly_score >= 65:
            status = "High anomaly risk"
            fair_trade_status = "High anomaly"
        elif anomaly_score >= 35:
            status = "Needs review"
            fair_trade_status = "Review needed"
        else:
            status = "Fair trade range"
            fair_trade_status = "Fair trade"

        return GoatResult(
            health_label=health_label,
            confidence=float(probability),
            estimated_weight_kg=float(round(estimated_weight, 2)),
            expected_weight_range=expected_range,
            expected_range_kg=expected_range,
            deviation_pct=float(round(deviation_pct, 2)),
            anomaly_score=float(round(anomaly_score, 2)),
            status=status,
            fair_trade_status=fair_trade_status,
            body_length_cm=float(round(features["body_length_cm"], 2)),
            body_height_cm=float(round(features["body_height_cm"], 2)),
            chest_girth_cm=float(round(features["chest_girth_cm"], 2)),
            body_shape_index=float(round(features["aspect_ratio"], 3)),
            body_fullness=float(round(features["body_fullness"], 3)),
            goat_id=str(goat_id),
            owner_name=str(owner_name),
            features=features,
        )


if __name__ == "__main__":
    analyzer = GoatAnalyzer()
    metrics = analyzer.build_models()
    print(metrics)

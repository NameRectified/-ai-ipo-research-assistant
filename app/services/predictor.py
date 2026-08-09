"""Model loading, prediction, and SHAP explanation service."""

import joblib
import numpy as np
import pandas as pd
import shap
from loguru import logger
from xgboost import XGBClassifier

from app.api.schemas import FeatureContribution, IPOApplication, IPOResearchAssessment
from app.services.feature_meta import FEATURE_META, describe_magnitude, humanize_value


class Predictor:
    """Wraps a trained XGBoost model for IPO profitability prediction."""

    def __init__(self, model_path: str) -> None:
        saved = joblib.load(model_path)
        self._model: XGBClassifier = saved["model"]
        self._features: list[str] = saved["features"]
        self._threshold: float = saved["threshold"]
        self._feature_ranges: dict = saved.get("feature_ranges", {})
        self._baseline_profitability_rate: float = float(
            saved.get("baseline_profitability_rate", 0.5)
        )
        self._explainer = shap.TreeExplainer(self._model)
        logger.info(
            f"Loaded model with {len(self._features)} features, "
            f"threshold={self._threshold:.2f}, "
            f"baseline_profitability_rate={self._baseline_profitability_rate:.4f}"
        )
        if self._feature_ranges:
            logger.info("Feature ranges loaded for OOD detection")

    @property
    def features(self) -> list[str]:
        return list(self._features)

    @property
    def threshold(self) -> float:
        return self._threshold

    def _check_out_of_distribution(self, row: dict) -> list[str]:
        """Check if any input features are outside training distribution ranges."""
        warnings = []
        for feat, rng in self._feature_ranges.items():
            if feat in row:
                val = row[feat]
                if val < rng["min"] or val > rng["max"]:
                    label = FEATURE_META.get(feat, {}).get("label", feat)
                    warnings.append(
                        f"{label} ({humanize_value(feat, val)}) is outside "
                        f"training range [{humanize_value(feat, rng['min'])}, "
                        f"{humanize_value(feat, rng['max'])}]"
                    )
        return warnings

    def predict(self, application: IPOApplication) -> IPOResearchAssessment:
        total_sub = (
            application.subscription_qib
            + application.subscription_hni
            + application.subscription_rii
        )
        eps = 1e-6

        row = {
            "Total_Sub": total_sub,
            "QIB": application.subscription_qib,
            "HNI": application.subscription_hni,
            "HNI_pct": application.subscription_hni / (total_sub + eps),
            "RII_pct": application.subscription_rii / (total_sub + eps),
            "Issue_Size_crores": application.issue_size,
        }

        # Check for out-of-distribution inputs
        ood_warnings = self._check_out_of_distribution(row)

        features_df = pd.DataFrame([row])[self._features]

        proba = float(self._model.predict_proba(features_df)[0, 1])
        is_profitable = proba > self._threshold

        shap_values = self._explainer.shap_values(features_df)
        shap_row = shap_values[0]

        explanations = [
            FeatureContribution(
                feature_name=feat,
                feature_label=FEATURE_META[feat]["label"],
                feature_value=row[feat],
                value_label=humanize_value(feat, row[feat]),
                shap_value=round(float(shap_row[i]), 4),
                impact=(
                    "increases_profitability"
                    if shap_row[i] >= 0
                    else "decreases_profitability"
                ),
                magnitude=describe_magnitude(float(shap_row[i])),
            )
            for i, feat in enumerate(self._features)
        ]
        explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)

        return IPOResearchAssessment(
            ipo_name=application.ipo_name,
            prediction="Profitable" if is_profitable else "Not Profitable",
            profitability_probability=round(proba, 4),
            baseline_profitability_rate=round(self._baseline_profitability_rate, 4),
            features_used=self._features,
            shap_explanations=explanations,
            input_warnings=ood_warnings,
        )

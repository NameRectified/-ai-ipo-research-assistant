"""Model loading, prediction, and SHAP explanation service."""

import joblib
import numpy as np
import pandas as pd
import shap
from loguru import logger
from xgboost import XGBClassifier

from app.api.schemas import FeatureContribution, IPOApplication, IPOResearchAssessment


class Predictor:
    """Wraps a trained XGBoost model for IPO profitability prediction."""

    def __init__(self, model_path: str) -> None:
        saved = joblib.load(model_path)
        self._model: XGBClassifier = saved["model"]
        self._features: list[str] = saved["features"]
        self._threshold: float = saved["threshold"]
        self._explainer = shap.TreeExplainer(self._model)
        logger.info(
            f"Loaded model with {len(self._features)} features, "
            f"threshold={self._threshold:.2f}"
        )

    @property
    def features(self) -> list[str]:
        return list(self._features)

    @property
    def threshold(self) -> float:
        return self._threshold

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

        features_df = pd.DataFrame([row])[self._features]

        proba = float(self._model.predict_proba(features_df)[0, 1])
        is_profitable = proba > self._threshold

        shap_values = self._explainer.shap_values(features_df)
        shap_row = shap_values[0]

        explanations = [
            FeatureContribution(
                feature_name=feat,
                feature_value=row[feat],
                shap_value=round(float(shap_row[i]), 4),
                impact=(
                    "increases_profitability"
                    if shap_row[i] >= 0
                    else "decreases_profitability"
                ),
            )
            for i, feat in enumerate(self._features)
        ]
        explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)

        return IPOResearchAssessment(
            ipo_name=application.ipo_name,
            prediction="Profitable" if is_profitable else "Not Profitable",
            profitability_probability=round(proba, 4),
            features_used=self._features,
            shap_explanations=explanations,
        )

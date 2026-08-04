"""Tests for the Predictor service — model loading, feature engineering, SHAP."""

from app.api.schemas import IPOApplication
from app.services.predictor import Predictor


class TestPredictor:
    """Tests for the Predictor class."""

    def test_loads_model_from_path(self, mock_model_artifact: str) -> None:
        predictor = Predictor(mock_model_artifact)
        assert len(predictor.features) == 6
        assert predictor.threshold == 0.30

    def test_features_list_is_copy(self, mock_model_artifact: str) -> None:
        predictor = Predictor(mock_model_artifact)
        features = predictor.features
        features.append("extra")
        assert len(predictor.features) == 6

    def test_predict_returns_assessment(
        self, mock_model_artifact: str, sample_ipo: IPOApplication
    ) -> None:
        predictor = Predictor(mock_model_artifact)
        result = predictor.predict(sample_ipo)
        assert result.ipo_name == "Test IPO"
        assert result.prediction in ("Profitable", "Not Profitable")
        assert 0 <= result.profitability_probability <= 1
        assert len(result.shap_explanations) == 6

    def test_predict_engineers_total_sub(
        self, mock_model_artifact: str, sample_ipo: IPOApplication
    ) -> None:
        predictor = Predictor(mock_model_artifact)
        result = predictor.predict(sample_ipo)
        total_sub = (
            sample_ipo.subscription_qib
            + sample_ipo.subscription_hni
            + sample_ipo.subscription_rii
        )
        qib_row = [e for e in result.shap_explanations if e.feature_name == "Total_Sub"]
        assert len(qib_row) == 1
        assert abs(qib_row[0].feature_value - total_sub) < 0.01

    def test_predict_engineers_hni_pct(
        self, mock_model_artifact: str, sample_ipo: IPOApplication
    ) -> None:
        predictor = Predictor(mock_model_artifact)
        result = predictor.predict(sample_ipo)
        total_sub = (
            sample_ipo.subscription_qib
            + sample_ipo.subscription_hni
            + sample_ipo.subscription_rii
        )
        expected_hni_pct = sample_ipo.subscription_hni / (total_sub + 1e-6)
        hni_row = [e for e in result.shap_explanations if e.feature_name == "HNI_pct"]
        assert len(hni_row) == 1
        assert abs(hni_row[0].feature_value - expected_hni_pct) < 0.001

    def test_shap_explanations_sorted_by_abs_value(
        self, mock_model_artifact: str, sample_ipo: IPOApplication
    ) -> None:
        predictor = Predictor(mock_model_artifact)
        result = predictor.predict(sample_ipo)
        abs_values = [abs(e.shap_value) for e in result.shap_explanations]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_high_demand_likely_profitable(
        self, mock_model_artifact: str, sample_ipo_high_demand: IPOApplication
    ) -> None:
        predictor = Predictor(mock_model_artifact)
        result = predictor.predict(sample_ipo_high_demand)
        assert result.prediction == "Profitable"
        assert result.profitability_probability > predictor.threshold

    def test_low_demand_likely_not_profitable(
        self, mock_model_artifact: str, sample_ipo_low_demand: IPOApplication
    ) -> None:
        predictor = Predictor(mock_model_artifact)
        result = predictor.predict(sample_ipo_low_demand)
        assert result.prediction in ("Profitable", "Not Profitable")
        assert 0 <= result.profitability_probability <= 1
        assert len(result.shap_explanations) == 6

    def test_shap_impact_labels(self, mock_model_artifact: str) -> None:
        predictor = Predictor(mock_model_artifact)
        result = predictor.predict(
            IPOApplication(
                ipo_name="Impact Test",
                issue_size=500,
                subscription_qib=42,
                subscription_hni=7,
                subscription_rii=3,
                issue_price=200,
                listing_date="2023-01-01",
            )
        )
        for e in result.shap_explanations:
            assert e.impact in ("increases_profitability", "decreases_profitability")
            if e.shap_value >= 0:
                assert e.impact == "increases_profitability"
            else:
                assert e.impact == "decreases_profitability"

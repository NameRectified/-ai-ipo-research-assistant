"""Tests for Pydantic schemas — input validation and serialization."""

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    ErrorResponse,
    FeatureContribution,
    IPOApplication,
    IPOResearchAssessment,
)


class TestIPOApplication:
    """Tests for the IPOApplication input schema."""

    def test_valid_application(self, sample_ipo: IPOApplication) -> None:
        assert sample_ipo.ipo_name == "Test IPO"
        assert sample_ipo.issue_size == 500.0
        assert sample_ipo.subscription_qib == 42.42
        assert sample_ipo.subscription_hni == 7.13
        assert sample_ipo.subscription_rii == 2.84
        assert sample_ipo.issue_price == 220.0
        assert sample_ipo.listing_date == "2022-08-26"

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            IPOApplication(
                ipo_name="Test",
                issue_size=100,
                subscription_qib=5,
                subscription_hni=3,
                subscription_rii=2,
                issue_price=100,
            )

    def test_negative_issue_size(self) -> None:
        with pytest.raises(ValidationError):
            IPOApplication(
                ipo_name="Test",
                issue_size=-100,
                subscription_qib=5,
                subscription_hni=3,
                subscription_rii=2,
                issue_price=100,
                listing_date="2023-01-01",
            )

    def test_negative_subscription(self) -> None:
        with pytest.raises(ValidationError):
            IPOApplication(
                ipo_name="Test",
                issue_size=100,
                subscription_qib=-5,
                subscription_hni=3,
                subscription_rii=2,
                issue_price=100,
                listing_date="2023-01-01",
            )

    def test_zero_values_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IPOApplication(
                ipo_name="Zero IPO",
                issue_size=0,
                subscription_qib=0,
                subscription_hni=0,
                subscription_rii=0,
                issue_price=0,
                listing_date="2023-01-01",
            )

    def test_out_of_range_values_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IPOApplication(
                ipo_name="Mega IPO",
                issue_size=100000,
                subscription_qib=500,
                subscription_hni=300,
                subscription_rii=100,
                issue_price=5000,
                listing_date="2023-12-31",
            )


class TestFeatureContribution:
    """Tests for the FeatureContribution schema."""

    def test_valid_contribution(self) -> None:
        fc = FeatureContribution(
            feature_name="QIB",
            feature_value=42.42,
            shap_value=0.15,
            impact="increases_profitability",
        )
        assert fc.feature_name == "QIB"
        assert fc.impact == "increases_profitability"

    def test_negative_shap(self) -> None:
        fc = FeatureContribution(
            feature_name="RII_pct",
            feature_value=0.2,
            shap_value=-0.08,
            impact="decreases_profitability",
        )
        assert fc.shap_value < 0
        assert fc.impact == "decreases_profitability"


class TestIPOResearchAssessment:
    """Tests for the assessment response schema."""

    def test_defaults(self) -> None:
        assessment = IPOResearchAssessment(
            ipo_name="Test",
            prediction="Profitable",
            profitability_probability=0.85,
            features_used=["QIB"],
            shap_explanations=[],
        )
        assert assessment.research_report == ""
        assert assessment.report_generated is False

    def test_probability_bounds(self) -> None:
        with pytest.raises(ValidationError):
            IPOResearchAssessment(
                ipo_name="Test",
                prediction="Profitable",
                profitability_probability=1.5,
                features_used=["QIB"],
                shap_explanations=[],
            )

    def test_negative_probability(self) -> None:
        with pytest.raises(ValidationError):
            IPOResearchAssessment(
                ipo_name="Test",
                prediction="Profitable",
                profitability_probability=-0.1,
                features_used=["QIB"],
                shap_explanations=[],
            )


class TestErrorResponse:
    """Tests for the error response schema."""

    def test_error_response(self) -> None:
        err = ErrorResponse(detail="Something went wrong")
        assert err.detail == "Something went wrong"

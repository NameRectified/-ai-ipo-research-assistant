"""Tests for the ReportGenerator service — prompt formatting and LLM calls."""

from unittest.mock import MagicMock

from app.api.schemas import FeatureContribution, IPOApplication, IPOResearchAssessment
from app.services.report_generator import (
    ReportGenerator,
    _format_ipo_data,
    _format_shap,
)


class TestFormatIPOData:
    """Tests for IPO data formatting."""

    def test_format_basic(self, sample_ipo: IPOApplication) -> None:
        result = _format_ipo_data(sample_ipo)
        assert "500" in result
        assert "42.42x" in result
        assert "7.13x" in result
        assert "2.84x" in result
        assert "220" in result
        assert "2022-08-26" in result

    def test_format_contains_all_fields(self, sample_ipo: IPOApplication) -> None:
        result = _format_ipo_data(sample_ipo)
        assert "Issue Size" in result
        assert "QIB Subscription" in result
        assert "HNI Subscription" in result
        assert "RII Subscription" in result
        assert "Issue Price" in result
        assert "Listing Date" in result


class TestFormatSHAP:
    """Tests for SHAP explanation formatting."""

    def test_format_empty_list(self) -> None:
        result = _format_shap([])
        assert result == ""

    def test_format_single_feature(self) -> None:
        explanations = [
            FeatureContribution(
                feature_name="QIB",
                feature_value=42.42,
                shap_value=0.15,
                impact="increases_profitability",
            )
        ]
        result = _format_shap(explanations)
        assert "QIB" in result
        assert "42.42" in result
        assert "+0.15" in result
        assert "increases_profitability" in result

    def test_format_negative_shap(self) -> None:
        explanations = [
            FeatureContribution(
                feature_name="RII_pct",
                feature_value=0.2,
                shap_value=-0.08,
                impact="decreases_profitability",
            )
        ]
        result = _format_shap(explanations)
        assert "-0.08" in result
        assert "+" not in result.split("SHAP=")[1].split(" ")[0]


class TestReportGenerator:
    """Tests for the ReportGenerator class."""

    def test_generate_calls_llm(self, sample_ipo: IPOApplication) -> None:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Generated report content"
        generator = ReportGenerator(mock_llm, prompt_path="prompts/report.yaml")
        assessment = IPOResearchAssessment(
            ipo_name="Test",
            prediction="Profitable",
            profitability_probability=0.85,
            features_used=["QIB"],
            shap_explanations=[
                FeatureContribution(
                    feature_name="QIB",
                    feature_value=42.42,
                    shap_value=0.15,
                    impact="increases_profitability",
                )
            ],
        )
        result = generator.generate(sample_ipo, assessment, threshold=0.30)
        assert result == "Generated report content"
        mock_llm.generate.assert_called_once()

    def test_generate_passes_correct_prompt_parts(
        self, sample_ipo: IPOApplication
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "report"
        generator = ReportGenerator(mock_llm, prompt_path="prompts/report.yaml")
        assessment = IPOResearchAssessment(
            ipo_name="Test IPO",
            prediction="Profitable",
            profitability_probability=0.85,
            features_used=["QIB"],
            shap_explanations=[],
        )
        generator.generate(sample_ipo, assessment, threshold=0.30)
        call_args = mock_llm.generate.call_args
        system_prompt = call_args[0][0]
        user_prompt = call_args[0][1]
        assert "IPO research analyst" in system_prompt
        assert "Test IPO" in user_prompt
        assert "Profitable" in user_prompt
        assert "85.0%" in user_prompt

    def test_generate_raises_on_llm_failure(
        self, sample_ipo: IPOApplication
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("All providers failed")
        generator = ReportGenerator(mock_llm, prompt_path="prompts/report.yaml")
        assessment = IPOResearchAssessment(
            ipo_name="Test",
            prediction="Profitable",
            profitability_probability=0.85,
            features_used=["QIB"],
            shap_explanations=[],
        )
        try:
            generator.generate(sample_ipo, assessment, threshold=0.30)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "All providers failed" in str(e)

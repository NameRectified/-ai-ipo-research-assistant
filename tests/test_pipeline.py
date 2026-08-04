"""Tests for the IPOPipeline orchestration service."""

from unittest.mock import MagicMock

from app.api.schemas import IPOApplication
from app.services.pipeline import IPOPipeline


class TestIPOPipeline:
    """Tests for pipeline orchestration."""

    def test_process_without_report_generator(
        self, mock_model_artifact: str, sample_ipo: IPOApplication
    ) -> None:
        from app.services.predictor import Predictor

        predictor = Predictor(mock_model_artifact)
        pipeline = IPOPipeline(predictor=predictor, report_generator=None)
        result = pipeline.process(sample_ipo)
        assert result.ipo_name == "Test IPO"
        assert result.research_report == ""
        assert result.report_generated is False

    def test_process_with_report_generator(
        self, mock_model_artifact: str, sample_ipo: IPOApplication
    ) -> None:
        from app.services.predictor import Predictor

        predictor = Predictor(mock_model_artifact)
        mock_report_gen = MagicMock()
        mock_report_gen.generate.return_value = "# Test Report\nBullish outlook."
        pipeline = IPOPipeline(predictor=predictor, report_generator=mock_report_gen)
        result = pipeline.process(sample_ipo)
        assert result.research_report == "# Test Report\nBullish outlook."
        assert result.report_generated is True
        mock_report_gen.generate.assert_called_once()

    def test_report_generation_failure_graceful(
        self, mock_model_artifact: str, sample_ipo: IPOApplication
    ) -> None:
        from app.services.predictor import Predictor

        predictor = Predictor(mock_model_artifact)
        mock_report_gen = MagicMock()
        mock_report_gen.generate.side_effect = RuntimeError("LLM API down")
        pipeline = IPOPipeline(predictor=predictor, report_generator=mock_report_gen)
        result = pipeline.process(sample_ipo)
        assert result.research_report == ""
        assert result.report_generated is False
        assert result.prediction in ("Profitable", "Not Profitable")

    def test_report_generator_called_with_correct_args(
        self, mock_model_artifact: str, sample_ipo: IPOApplication
    ) -> None:
        from app.services.predictor import Predictor

        predictor = Predictor(mock_model_artifact)
        mock_report_gen = MagicMock()
        mock_report_gen.generate.return_value = "report"
        pipeline = IPOPipeline(predictor=predictor, report_generator=mock_report_gen)
        pipeline.process(sample_ipo)
        call_args = mock_report_gen.generate.call_args
        assert call_args[0][0] == sample_ipo
        assert call_args[1]["threshold"] == predictor.threshold

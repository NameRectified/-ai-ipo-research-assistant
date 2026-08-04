"""Orchestrates the full IPO research pipeline."""

from typing import Optional

from loguru import logger

from app.api.schemas import IPOApplication, IPOResearchAssessment
from app.services.predictor import Predictor
from app.services.report_generator import ReportGenerator


class IPOPipeline:
    """Coordinates the end-to-end IPO research flow.

    For a single IPO, this pipeline:
    1. Runs the ML model to get a profitability prediction + SHAP explanations.
    2. Generates a narrative research report via LLM (if configured).
    """

    def __init__(
        self,
        predictor: Predictor,
        report_generator: Optional[ReportGenerator] = None,
    ) -> None:
        self._predictor = predictor
        self._report_generator = report_generator

    def process(self, application: IPOApplication) -> IPOResearchAssessment:
        """Run the full research pipeline on a single IPO.

        Args:
            application: Validated IPO subscription data.

        Returns:
            Complete research assessment with prediction, SHAP explanations,
            and optionally an LLM-generated research report.
        """
        assessment = self._predictor.predict(application)

        if self._report_generator is not None:
            try:
                report = self._report_generator.generate(
                    application,
                    assessment,
                    threshold=self._predictor.threshold,
                )
                assessment.research_report = report
                assessment.report_generated = True
            except Exception as exc:
                logger.warning(f"Report generation failed: {exc}")
                assessment.research_report = ""

        return assessment

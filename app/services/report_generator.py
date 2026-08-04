"""Generates narrative research reports using an LLM."""

from loguru import logger
import yaml

from app.api.schemas import FeatureContribution, IPOApplication, IPOResearchAssessment
from app.services.llm_client import LLMClient

PROMPT_PATH = "prompts/report.yaml"


def _load_prompt(path: str) -> tuple[str, str]:
    """Load the system prompt and user prompt template from a YAML file.

    Args:
        path: Path to the YAML prompt file.

    Returns:
        (system_prompt, user_prompt_template)
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    prompts = data["report_generation"]
    return prompts["system_prompt"], prompts["user_prompt_template"]


def _format_ipo_data(application: IPOApplication) -> str:
    """Format IPO subscription data as a readable table for the LLM."""
    return (
        f"Issue Size: {application.issue_size:,.0f} crores\n"
        f"QIB Subscription: {application.subscription_qib:.2f}x\n"
        f"HNI Subscription: {application.subscription_hni:.2f}x\n"
        f"RII Subscription: {application.subscription_rii:.2f}x\n"
        f"Issue Price: INR {application.issue_price:,.0f}\n"
        f"Listing Date: {application.listing_date}"
    )


def _format_shap(explanations: list[FeatureContribution]) -> str:
    """Format SHAP explanations as a readable table for the LLM."""
    lines = []
    for e in explanations:
        direction = "+" if e.shap_value >= 0 else ""
        lines.append(
            f"  {e.feature_name}: value={e.feature_value}, "
            f"SHAP={direction}{e.shap_value:.4f} ({e.impact})"
        )
    return "\n".join(lines)


class ReportGenerator:
    """Generates narrative research reports from model predictions."""

    def __init__(
        self, llm_client: LLMClient, prompt_path: str = PROMPT_PATH
    ) -> None:
        self._llm = llm_client
        self._system_prompt, self._user_template = _load_prompt(prompt_path)
        logger.info("Report generator initialized")

    def generate(
        self,
        application: IPOApplication,
        assessment: IPOResearchAssessment,
        threshold: float,
    ) -> str:
        """Generate a narrative research report for investors.

        Args:
            application: The original IPO subscription data.
            assessment: The model's prediction and SHAP explanations.
            threshold: Decision threshold used by the model.

        Returns:
            A narrative research report in plain text.

        Raises:
            RuntimeError: If no LLM provider is available.
        """
        ipo_data = _format_ipo_data(application)
        shap_text = _format_shap(assessment.shap_explanations)

        user_prompt = self._user_template.format(
            ipo_name=application.ipo_name,
            prediction=assessment.prediction,
            probability=assessment.profitability_probability,
            threshold=threshold,
            applicant_data=ipo_data,
            shap_explanations=shap_text,
        )

        return self._llm.generate(self._system_prompt, user_prompt)

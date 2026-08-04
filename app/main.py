"""FastAPI application for the AI IPO Research Assistant."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.schemas import ErrorResponse, IPOApplication, IPOResearchAssessment
from app.config.settings import settings
from app.services.llm_client import LLMClient
from app.services.pipeline import IPOPipeline
from app.services.predictor import Predictor
from app.services.report_generator import ReportGenerator

pipeline: Optional[IPOPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialize clients on startup. Clean up on shutdown."""
    global pipeline

    logger.info(f"Loading model from {settings.model_path}")
    predictor = Predictor(settings.model_path)

    llm_client = LLMClient()
    report_generator: Optional[ReportGenerator] = None
    if llm_client.available:
        report_generator = ReportGenerator(llm_client)
        logger.info("LLM report generator initialized")
    else:
        logger.warning("No LLM providers configured — reports will be skipped")

    pipeline = IPOPipeline(predictor=predictor, report_generator=report_generator)
    logger.info("IPO pipeline initialized")

    yield

    pipeline = None
    logger.info("Shutdown complete.")


app = FastAPI(
    title="AI IPO Research Assistant",
    description=(
        "Predicts IPO listing-day profitability using XGBoost, "
        "explains predictions via SHAP, and generates research reports."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the minimal frontend."""
    path = Path("app/static/index.html")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.post(
    "/api/v1/predict",
    response_model=IPOResearchAssessment,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def predict(application: IPOApplication):
    """Submit IPO subscription data for a research assessment.

    Runs the full pipeline: prediction → SHAP explanation →
    LLM report → persistence. Returns the complete research assessment.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline.process(application)

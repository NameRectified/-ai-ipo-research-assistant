"""Shared test fixtures for the IPO Research Assistant test suite."""

from typing import Any
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from app.api.schemas import IPOApplication

FEATURES = [
    "Total_Sub",
    "QIB",
    "HNI",
    "HNI_pct",
    "RII_pct",
    "Issue_Size_crores",
]

THRESHOLD = 0.30


def _build_mock_model() -> XGBClassifier:
    """Build a tiny XGBoost model that returns predictable probabilities.

    The model is trained on a minimal dataset so that predict_proba
    returns deterministic results for test assertions.
    """
    X = pd.DataFrame(
        {
            "Total_Sub": [10, 50, 100, 5, 200, 80],
            "QIB": [5, 25, 50, 2, 100, 40],
            "HNI": [3, 15, 30, 1, 60, 24],
            "HNI_pct": [0.3, 0.3, 0.3, 0.2, 0.3, 0.3],
            "RII_pct": [0.2, 0.2, 0.2, 0.6, 0.1, 0.2],
            "Issue_Size_crores": [100, 500, 1000, 50, 2000, 800],
        }
    )
    y = np.array([0, 1, 1, 0, 1, 1])
    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return model


@pytest.fixture()
def mock_model_artifact(tmp_path: Any) -> str:
    """Save a mock model artifact to a temp file and return the path."""
    model = _build_mock_model()
    artifact = {
        "model": model,
        "features": FEATURES,
        "threshold": THRESHOLD,
        "baseline_profitability_rate": 0.67,
        "feature_ranges": {
            "Total_Sub": {"min": 0.0, "max": 300.0},
            "QIB": {"min": 0.0, "max": 150.0},
            "HNI": {"min": 0.0, "max": 100.0},
            "HNI_pct": {"min": 0.0, "max": 0.9},
            "RII_pct": {"min": 0.0, "max": 0.99},
            "Issue_Size_crores": {"min": 23.0, "max": 27858.8},
        },
    }
    path = tmp_path / "model.pkl"
    joblib.dump(artifact, path)
    return str(path)


@pytest.fixture()
def sample_ipo() -> IPOApplication:
    """Return a valid IPO application for testing."""
    return IPOApplication(
        ipo_name="Test IPO",
        issue_size=500.0,
        subscription_qib=42.42,
        subscription_hni=7.13,
        subscription_rii=2.84,
        issue_price=220.0,
        listing_date="2022-08-26",
    )


@pytest.fixture()
def sample_ipo_high_demand() -> IPOApplication:
    """Return an IPO with high subscription (likely profitable)."""
    return IPOApplication(
        ipo_name="High Demand IPO",
        issue_size=200.0,
        subscription_qib=150.0,
        subscription_hni=80.0,
        subscription_rii=20.0,
        issue_price=350.0,
        listing_date="2023-01-15",
    )


@pytest.fixture()
def sample_ipo_low_demand() -> IPOApplication:
    """Return an IPO with low subscription (likely not profitable)."""
    return IPOApplication(
        ipo_name="Low Demand IPO",
        issue_size=1000.0,
        subscription_qib=1.5,
        subscription_hni=0.8,
        subscription_rii=0.5,
        issue_price=100.0,
        listing_date="2023-06-01",
    )

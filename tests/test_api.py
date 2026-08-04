"""Tests for FastAPI endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestIndexEndpoint:
    """Tests for GET / (frontend)."""

    def test_index_returns_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "AI IPO Research Assistant" in resp.text


class TestPredictEndpoint:
    """Tests for POST /api/v1/predict."""

    def test_predict_requires_pipeline(self) -> None:
        from fastapi import HTTPException

        from app.api.schemas import IPOApplication
        from app.main import predict

        import app.main as main_module

        original = main_module.pipeline
        main_module.pipeline = None
        try:
            body = IPOApplication(
                ipo_name="Test",
                issue_size=500,
                subscription_qib=42,
                subscription_hni=7,
                subscription_rii=3,
                issue_price=220,
                listing_date="2022-08-26",
            )
            try:
                predict(body)
                assert False, "Should have raised HTTPException"
            except HTTPException as e:
                assert e.status_code == 503
        finally:
            main_module.pipeline = original

    def test_predict_validation_error(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/predict",
            json={"ipo_name": "Test"},
        )
        assert resp.status_code == 422

    def test_predict_missing_fields(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/predict",
            json={
                "ipo_name": "Test",
                "issue_size": 500,
            },
        )
        assert resp.status_code == 422

    def test_predict_negative_values(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/predict",
            json={
                "ipo_name": "Test",
                "issue_size": -100,
                "subscription_qib": 42,
                "subscription_hni": 7,
                "subscription_rii": 3,
                "issue_price": 220,
                "listing_date": "2022-08-26",
            },
        )
        assert resp.status_code == 422

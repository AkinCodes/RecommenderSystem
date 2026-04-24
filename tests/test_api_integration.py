"""Comprehensive API integration tests for CinemaScopeAI Recommender.

Uses FastAPI TestClient with mocking so tests pass without a trained model
or live TMDB credentials.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_MOVIES = [
    {
        "title": f"Movie {i}",
        "genre": "Action",
        "rating": "PG-13",
        "score": round(0.1 * i, 2),
        "poster_url": f"https://example.com/poster{i}.jpg",
        "director": f"Director {i}",
        "release_year": 2020 + i,
        "summary": f"Summary for movie {i}.",
    }
    for i in range(1, 11)
]

VALID_PAYLOAD = {
    "continuous_features": [0.5, 0.8, 0.1, 0.3, 0.6, 0.4, 0.7, 0.5],
    "categorical_features": [1, 2],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """Lazily import TestClient to avoid model-loading side effects."""
    from fastapi.testclient import TestClient

    from api.app import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# Root & Health
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_status_healthy(self, client):
        data = client.get("/").json()
        assert data["status"] == "healthy"

    def test_root_has_message(self, client):
        data = client.get("/").json()
        assert "message" in data


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_model_loaded(self, client):
        data = client.get("/health").json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)


# ---------------------------------------------------------------------------
# Model Info
# ---------------------------------------------------------------------------


class TestModelInfoEndpoint:
    def test_models_returns_200_or_503(self, client):
        resp = client.get("/api/v1/models")
        assert resp.status_code in (200, 503)

    def test_models_architecture_info(self, client):
        resp = client.get("/api/v1/models")
        if resp.status_code == 200:
            data = resp.json()
            assert "architecture" in data
            assert "num_parameters" in data
            assert data["num_parameters"] > 0
            assert "device" in data
            assert "num_continuous_features" in data
            assert "num_categorical_features" in data
            assert "mlp_layers" in data
            assert isinstance(data["mlp_layers"], list)

    def test_models_503_structured_error(self, client):
        resp = client.get("/api/v1/models")
        if resp.status_code == 503:
            data = resp.json()
            assert data["success"] is False
            assert "error" in data
            assert "detail" in data


# ---------------------------------------------------------------------------
# Predict — valid requests
# ---------------------------------------------------------------------------


class TestPredictValid:
    """Tests that mock the model so predictions work even without a .pth file."""

    def _mock_predict(self, client, payload):
        """Helper: patch model_loaded, model, and fetch_real_movies."""
        import api.app as app_module

        fake_model = MagicMock()
        fake_model.return_value = torch.tensor([[0.75]])

        with (
            patch.object(app_module, "model_loaded", True),
            patch.object(app_module, "model", fake_model),
            patch("api.app.fetch_real_movies", new_callable=AsyncMock) as mock_fetch,
        ):
            mock_fetch.return_value = MOCK_MOVIES
            return client.post("/api/v1/predict", json=payload)

    def test_predict_returns_list(self, client):
        """POST /api/v1/predict with valid input returns a list of movies."""
        resp = self._mock_predict(client, VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) <= 5

    def test_predict_movie_schema(self, client):
        """Each returned movie has all required fields."""
        resp = self._mock_predict(client, VALID_PAYLOAD)
        assert resp.status_code == 200
        for movie in resp.json():
            assert "title" in movie
            assert "genre" in movie
            assert "rating" in movie
            assert "score" in movie
            assert "poster_url" in movie
            assert "director" in movie
            assert "release_year" in movie
            assert "summary" in movie

    def test_predict_at_most_five(self, client):
        """Even with 10 movies from TMDB, only 5 are returned."""
        resp = self._mock_predict(client, VALID_PAYLOAD)
        assert resp.status_code == 200
        assert len(resp.json()) == 5


# ---------------------------------------------------------------------------
# Predict — invalid requests
# ---------------------------------------------------------------------------


class TestPredictInvalid:
    def test_wrong_number_of_categorical_features(self, client):
        payload = {
            "continuous_features": [0.5] * 8,
            "categorical_features": [1],  # need 2
        }
        resp = client.post("/api/v1/predict", json=payload)
        # 400 if model loaded, 503 if not
        assert resp.status_code in (400, 503)
        if resp.status_code == 400:
            data = resp.json()
            assert data["success"] is False
            assert "categorical" in data["detail"].lower()

    def test_too_many_categorical_features(self, client):
        payload = {
            "continuous_features": [0.5] * 8,
            "categorical_features": [1, 2, 3],  # need 2
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code in (400, 503)

    def test_out_of_range_categorical_user(self, client):
        payload = {
            "continuous_features": [0.5] * 8,
            "categorical_features": [99999, 2],  # user_id out of range
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code in (400, 503)
        if resp.status_code == 400:
            data = resp.json()
            assert "out of range" in data["detail"].lower()

    def test_out_of_range_categorical_item(self, client):
        payload = {
            "continuous_features": [0.5] * 8,
            "categorical_features": [1, 99999],  # item_id out of range
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code in (400, 503)
        if resp.status_code == 400:
            data = resp.json()
            assert "out of range" in data["detail"].lower()

    def test_negative_categorical(self, client):
        payload = {
            "continuous_features": [0.5] * 8,
            "categorical_features": [-1, 2],
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code in (400, 503)

    def test_wrong_number_of_continuous_features(self, client):
        payload = {
            "continuous_features": [0.5],  # need 8
            "categorical_features": [1, 2],
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code in (400, 503)
        if resp.status_code == 400:
            data = resp.json()
            assert "continuous" in data["detail"].lower()

    def test_missing_fields_returns_422(self, client):
        resp = client.post("/api/v1/predict", json={})
        assert resp.status_code == 422

    def test_missing_categorical_returns_422(self, client):
        resp = client.post("/api/v1/predict", json={"continuous_features": [0.5] * 8})
        assert resp.status_code == 422

    def test_missing_continuous_returns_422(self, client):
        resp = client.post("/api/v1/predict", json={"categorical_features": [1, 2]})
        assert resp.status_code == 422

    def test_string_instead_of_numbers_returns_422(self, client):
        payload = {
            "continuous_features": ["not", "a", "number"],
            "categorical_features": [1, 2],
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_string_categorical_returns_422(self, client):
        payload = {
            "continuous_features": [0.5] * 8,
            "categorical_features": ["a", "b"],
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_null_body_returns_422(self, client):
        resp = client.post(
            "/api/v1/predict",
            content="null",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Recommend
# ---------------------------------------------------------------------------


class TestRecommendEndpoint:
    def test_recommend_known_user_or_503(self, client):
        """If model is loaded and serving_context exists, known user gets recs."""
        resp = client.get("/api/v1/recommend/1")
        # 503 if model/context not loaded, 200 if known, cold-start dict if unknown
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "user_id" in data
            assert "recommendations" in data
            assert isinstance(data["recommendations"], list)

    def test_recommend_unknown_user_cold_start(self, client):
        """An unknown user_id should trigger cold-start fallback or 404/503."""
        resp = client.get("/api/v1/recommend/999999")
        assert resp.status_code in (200, 404, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "note" in data
            assert "cold-start" in data["note"]
            assert "recommendations" in data

    def test_recommend_top_k_param(self, client):
        """top_k query parameter should limit results."""
        resp = client.get("/api/v1/recommend/1?top_k=3")
        if resp.status_code == 200:
            data = resp.json()
            assert len(data["recommendations"]) <= 3


# ---------------------------------------------------------------------------
# Structured Errors
# ---------------------------------------------------------------------------


class TestStructuredErrors:
    def test_404_returns_json(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data

    def test_404_detail_content(self, client):
        data = client.get("/totally/made/up/path").json()
        assert data["detail"] == "Not Found"

    def test_method_not_allowed(self, client):
        """PUT on a GET-only endpoint should return 405."""
        resp = client.put("/health")
        assert resp.status_code == 405

    def test_custom_http_exception_structured(self, client):
        """Endpoints that raise HTTPException return structured errors."""
        # Trigger a 503 by hitting predict without a loaded model
        payload = {
            "continuous_features": [0.5] * 8,
            "categorical_features": [1, 2],
        }
        resp = client.post("/api/v1/predict", json=payload)
        if resp.status_code in (400, 503):
            data = resp.json()
            assert data["success"] is False
            assert "error" in data
            assert "detail" in data

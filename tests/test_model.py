import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
import torch

# Ensure the project root is on sys.path so imports work when running pytest
# from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.dlrm import DLRMModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dlrm_model():
    """Return a small DLRM model suitable for unit tests."""
    return DLRMModel(
        num_features=10,
        embedding_sizes=[10, 10, 10, 10, 10],
        mlp_layers=[64, 32, 16],
    )


@pytest.fixture
def sample_inputs():
    """Return a (continuous, categorical) tensor pair for testing."""
    continuous = torch.randn(1, 10)
    categorical = torch.randint(0, 5, (1, 5))
    return continuous, categorical


@pytest.fixture
def test_client():
    """Return a TestClient for the FastAPI app.

    Imported lazily so model-loading side effects don't block other tests
    if the trained_model.pth is missing.
    """
    from fastapi.testclient import TestClient

    from api.app import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestDLRMModel:
    def test_forward_output_shape(self, dlrm_model, sample_inputs):
        continuous, categorical = sample_inputs
        output = dlrm_model(continuous, categorical)
        assert output.shape == torch.Size([1, 1])

    def test_forward_output_range(self, dlrm_model, sample_inputs):
        continuous, categorical = sample_inputs
        output = dlrm_model(continuous, categorical)
        assert 0.0 <= output.item() <= 1.0, "Output should be a probability in [0, 1]"

    def test_forward_rejects_none_inputs(self, dlrm_model):
        with pytest.raises(ValueError, match="cannot be None"):
            dlrm_model(None, torch.randint(0, 5, (1, 5)))

    def test_forward_rejects_none_categorical(self, dlrm_model):
        with pytest.raises(ValueError, match="cannot be None"):
            dlrm_model(torch.randn(1, 10), None)

    def test_forward_rejects_too_many_categorical(self, dlrm_model):
        continuous = torch.randn(1, 10)
        categorical = torch.randint(0, 5, (1, 100))  # way more than 5
        with pytest.raises(ValueError, match="Too many categorical features"):
            dlrm_model(continuous, categorical)

    def test_batch_forward(self, dlrm_model):
        continuous = torch.randn(4, 10)
        categorical = torch.randint(0, 5, (4, 5))
        output = dlrm_model(continuous, categorical)
        assert output.shape == torch.Size([4, 1])

    def test_single_categorical_feature(self, dlrm_model):
        """Model should handle fewer categorical features than embedding tables."""
        continuous = torch.randn(1, 10)
        categorical = torch.randint(0, 5, (1, 1))
        output = dlrm_model(continuous, categorical)
        assert output.shape == torch.Size([1, 1])

    def test_empty_batch_raises(self, dlrm_model):
        """Zero-batch tensors should propagate gracefully (no crash)."""
        continuous = torch.randn(0, 10)
        categorical = torch.randint(0, 5, (0, 5))
        output = dlrm_model(continuous, categorical)
        assert output.shape == torch.Size([0, 1])

    def test_parameter_count_positive(self, dlrm_model):
        """Model should have a non-trivial number of parameters."""
        total = sum(p.numel() for p in dlrm_model.parameters())
        assert total > 0


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    def test_root_endpoint(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_endpoint(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)
        assert data["version"] == "1.0.0"


class TestModelInfoEndpoint:
    def test_model_info(self, test_client):
        response = test_client.get("/api/v1/models")
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            data = response.json()
            assert "architecture" in data
            assert "num_parameters" in data
            assert data["num_parameters"] > 0
            assert "device" in data
            assert "mlp_layers" in data


class TestPredictEndpoint:
    def test_predict_endpoint(self, test_client):
        payload = {
            "continuous_features": [0.5, 0.8],
            "categorical_features": [1, 2],
        }
        with patch("api.app.fetch_real_movies", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [
                {
                    "title": "Test Movie",
                    "genre": "Action",
                    "rating": "PG-13",
                    "score": 0.85,
                    "poster_url": "https://example.com/poster.jpg",
                    "director": "Test Director",
                    "release_year": 2024,
                    "summary": "A test movie.",
                }
            ]
            response = test_client.post("/api/v1/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert "title" in data[0]

    def test_predict_legacy_endpoint(self, test_client):
        """Legacy /predict/ endpoint should still work."""
        payload = {
            "continuous_features": [0.5, 0.8],
            "categorical_features": [1, 2],
        }
        with patch("api.app.fetch_real_movies", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [
                {
                    "title": "Legacy Movie",
                    "genre": "Drama",
                    "rating": "R",
                    "score": 0.72,
                    "poster_url": "https://example.com/poster2.jpg",
                    "director": "Director Two",
                    "release_year": 2023,
                    "summary": "A legacy test.",
                }
            ]
            response = test_client.post("/predict/", json=payload)

        assert response.status_code == 200

    def test_predict_bad_categorical_count(self, test_client):
        payload = {
            "continuous_features": [0.5, 0.8],
            "categorical_features": [1],  # too few
        }
        response = test_client.post("/api/v1/predict", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data

    def test_predict_empty_continuous(self, test_client):
        """Empty continuous features should still be accepted by the endpoint."""
        payload = {
            "continuous_features": [],
            "categorical_features": [1, 2],
        }
        with patch("api.app.fetch_real_movies", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [
                {
                    "title": "Movie",
                    "genre": "Action",
                    "rating": "PG",
                    "score": 0.5,
                    "poster_url": "https://example.com/p.jpg",
                    "director": "Dir",
                    "release_year": 2020,
                    "summary": "Summary.",
                }
            ]
            response = test_client.post("/api/v1/predict", json=payload)
        # Might succeed or fail depending on model input -- just ensure it
        # doesn't return a 500
        assert response.status_code in (200, 400, 422, 503)

    def test_predict_wrong_types_rejected(self, test_client):
        """Sending strings instead of numbers should fail validation."""
        payload = {
            "continuous_features": ["not", "a", "number"],
            "categorical_features": [1, 2],
        }
        response = test_client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_fields(self, test_client):
        """Omitting required fields should fail."""
        response = test_client.post("/api/v1/predict", json={})
        assert response.status_code == 422


class TestStructuredErrors:
    def test_404_returns_structured_error(self, test_client):
        response = test_client.get("/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert "detail" in data

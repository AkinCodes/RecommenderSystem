# CinemaScopeAI Recommender System -- Technical Documentation

> **Version:** 1.0.0 | **Last Updated:** 2026-03-20 | **Python:** >=3.10 | **License:** Proprietary

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Getting Started](#4-getting-started)
5. [API Reference](#5-api-reference)
6. [DLRM Model Deep Dive](#6-dlrm-model-deep-dive)
7. [Training](#7-training)
8. [Inference](#8-inference)
9. [Configuration](#9-configuration)
10. [Testing](#10-testing)
11. [Deployment](#11-deployment)
12. [Makefile Reference](#12-makefile-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [Contributing](#14-contributing)
15. [Changelog](#15-changelog)

---

## 1. Project Overview

CinemaScopeAI Recommender System is a production-grade movie recommendation service that combines a **Deep Learning Recommendation Model (DLRM)** with real-time movie data from the **TMDB (The Movie Database) API**. The system accepts user feature vectors, computes a recommendation score via the DLRM, and returns ranked movie recommendations enriched with metadata from TMDB.

### Key Features

- DLRM-based recommendation scoring with continuous and categorical feature support
- Real-time movie data enrichment via the TMDB API
- RESTful API with versioned endpoints, structured error handling, and request logging
- PyTorch Lightning training pipeline with checkpoint management
- Docker-ready deployment with health checks
- AWS ECS container deployment configuration
- Comprehensive test suite covering model logic and all API endpoints

### Tech Stack

| Layer             | Technology                                      |
|-------------------|-------------------------------------------------|
| **Language**      | Python 3.10+                                    |
| **Web Framework** | FastAPI + Uvicorn                                |
| **ML Framework**  | PyTorch 2.0+                                    |
| **Training**      | PyTorch Lightning 2.0+                           |
| **HTTP Client**   | httpx (async)                                    |
| **Config**        | PyYAML + python-dotenv                           |
| **Data Science**  | NumPy, Pandas, scikit-learn                      |
| **Package Mgmt**  | uv (Astral)                                      |
| **Linting**       | Ruff                                             |
| **Testing**       | pytest + pytest-asyncio                          |
| **Containerization** | Docker (Python 3.11-slim + uv)               |
| **CI/CD**         | GitHub Actions                                   |
| **Cloud**         | AWS ECS (Fargate), ECR                           |

---

## 2. Architecture

### System Diagram

```
+-----------+       +-------------------+       +------------------+
|  Client   | ----> |  FastAPI Server   | ----> |   TMDB API       |
| (curl/UI) |       |  (Uvicorn)        |       |  (popular movies)|
+-----------+       +--------+----------+       +------------------+
                             |
                    +--------v----------+
                    |   DLRM Model      |
                    |  (PyTorch, CPU)   |
                    +-------------------+
```

### Request Flow

1. Client sends a `POST /api/v1/predict` request with continuous and categorical features.
2. FastAPI validates the input via Pydantic schemas.
3. The DLRM model runs a forward pass and produces a recommendation score in `[0, 1]`.
4. The API fetches popular movies from TMDB asynchronously via `httpx`.
5. Movies are ranked by proximity to the prediction score and the top 5 are returned.

### ML Pipeline

```
+-------------------+     +------------------+     +-------------------+
| Data Preparation  | --> | Model Training   | --> | Model Export      |
| (synthetic/real)  |     | (PyTorch Lightning) |  | (trained_model.pth)|
+-------------------+     +------------------+     +-------------------+
                                                           |
                                                   +-------v-----------+
                                                   | API Serving       |
                                                   | (FastAPI + torch) |
                                                   +-------------------+
```

### DLRM Architecture

```
Continuous Features (float)          Categorical Features (int)
        |                                     |
        v                                     v
  [Linear Layer]                   [Embedding Tables x N]
        |                                     |
        v                                     v
  Dense Repr (dim=mlp[0])          Embedded Repr (dim=mlp[0] each)
        |                                     |
        +----------------+--------------------+
                         |
                    [Concatenate]
                         |
                    [MLP Layers]
                    Linear -> ReLU
                    Linear -> ReLU
                         |
                  [Output Linear]
                         |
                    [Sigmoid]
                         |
                  Score in [0, 1]
```

---

## 3. Project Structure

```
RecommenderSystem/
|-- api/
|   |-- __init__.py              # Package marker
|   |-- app.py                   # FastAPI application, endpoints, middleware, model loading
|
|-- models/
|   |-- __init__.py              # Package marker
|   |-- dlrm.py                  # PyTorch DLRM model implementation
|
|-- scripts/
|   |-- train.py                 # PyTorch Lightning training script with checkpointing
|   |-- inference.py             # Standalone inference script for testing the model
|
|-- tests/
|   |-- test_model.py            # pytest suite: model unit tests + API integration tests
|
|-- configs/
|   |-- config.yaml              # Model hyperparameters (single source of truth)
|
|-- ci_cd/
|   |-- github_actions.yml       # GitHub Actions CI/CD pipeline definition
|
|-- data/                        # Training data directory (gitignored)
|-- lightning_logs/              # PyTorch Lightning logs and checkpoints (gitignored)
|
|-- .env.example                 # Template for environment variables
|-- .gitignore                   # Git ignore rules
|-- .dockerignore                # Docker build context exclusions
|-- container-definition.json    # AWS ECS task container definition
|-- iam-policy.json              # AWS IAM policy for ECS task execution role
|-- trust-policy.json            # AWS IAM trust policy for ECS service
|-- Dockerfile                   # Container image definition (Python 3.11-slim + uv)
|-- Makefile                     # Developer workflow targets
|-- pyproject.toml               # Project metadata, dependencies, tool config
|-- trained_model.pth            # Serialized model weights (gitignored in .gitignore)
|-- README.md                    # Quick-start guide
|-- DOCUMENTATION.md             # This file -- comprehensive technical docs
```

---

## 4. Getting Started

### Prerequisites

- **Python 3.10+**
- **uv** -- fast Python package manager from Astral ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **TMDB API credentials** -- free at https://www.themoviedb.org/settings/api

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd RecommenderSystem

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install all dependencies (including dev extras)
make install
# Equivalent to: uv sync --all-extras

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your TMDB credentials
```

### Running the Server

```bash
make run
# Equivalent to: uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

The server starts on `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI, or `http://localhost:8000/redoc` for ReDoc.

### Verifying the Setup

```bash
# Check the root endpoint
curl http://localhost:8000/

# Check health (includes model load status)
curl http://localhost:8000/health
```

---

## 5. API Reference

All endpoints return JSON. Error responses follow a consistent structure:

```json
{
  "success": false,
  "error": "HTTP <status_code>",
  "detail": "Human-readable error message."
}
```

### `GET /` -- Root

Returns a basic status message confirming the API is running.

**Response (200):**

```json
{
  "status": "healthy",
  "message": "Recommendation API is running."
}
```

**Example:**

```bash
curl http://localhost:8000/
```

---

### `GET /health` -- Health Check

Reports server health, model load status, and application version.

**Response (200):**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

**Example:**

```bash
curl http://localhost:8000/health
```

> **Note:** If `model_loaded` is `false`, the model weights file (`trained_model.pth`) was not found at startup. Prediction endpoints will return `503`.

---

### `GET /api/v1/models` -- Model Info

Returns metadata about the loaded DLRM model.

**Response (200):**

```json
{
  "architecture": "DLRM (Deep Learning Recommendation Model)",
  "num_parameters": 12345,
  "device": "cpu",
  "num_continuous_features": 2,
  "num_categorical_features": 2,
  "mlp_layers": [64, 32, 16]
}
```

**Response (503):** Model is not loaded.

**Example:**

```bash
curl http://localhost:8000/api/v1/models
```

---

### `POST /api/v1/predict` -- Predict (Versioned)

Accepts user features, runs the DLRM, fetches popular movies from TMDB, and returns the top 5 recommendations ranked by score proximity.

**Request Body:**

```json
{
  "continuous_features": [0.5, 0.8],
  "categorical_features": [1, 2]
}
```

| Field                    | Type         | Description                                      |
|--------------------------|--------------|--------------------------------------------------|
| `continuous_features`    | `list[float]`| Dense (numeric) features for the model           |
| `categorical_features`   | `list[int]`  | Exactly 2 categorical feature indices            |

**Response (200):**

```json
[
  {
    "title": "Movie Title",
    "genre": "Unknown",
    "rating": "N/A",
    "score": 0.85,
    "poster_url": "https://image.tmdb.org/t/p/w500/...",
    "director": "N/A",
    "release_year": 2024,
    "summary": "Movie overview text."
  }
]
```

**Error Responses:**

| Status | Condition                                   |
|--------|---------------------------------------------|
| 400    | Wrong number of categorical features        |
| 422    | Invalid request body (Pydantic validation)  |
| 502    | TMDB API error or no movies returned        |
| 503    | Model not loaded                            |

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"continuous_features": [0.5, 0.8], "categorical_features": [1, 2]}'
```

---

### `POST /predict/` -- Predict (Legacy)

Backwards-compatible endpoint that delegates to `/api/v1/predict`. Accepts the same request body and returns the same response.

**Example:**

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"continuous_features": [0.5, 0.8], "categorical_features": [1, 2]}'
```

---

## 6. DLRM Model Deep Dive

### Overview

The Deep Learning Recommendation Model (DLRM) is implemented in `models/dlrm.py` as a PyTorch `nn.Module`. It is designed to process both continuous (dense) and categorical (sparse) input features to produce a recommendation score.

### Architecture Details

**File:** `models/dlrm.py` -- class `DLRMModel`

**Constructor Parameters:**

| Parameter         | Type         | Description                                      |
|-------------------|--------------|--------------------------------------------------|
| `num_features`    | `int`        | Number of continuous input features               |
| `embedding_sizes` | `list[int]`  | Vocabulary size for each categorical embedding    |
| `mlp_layers`      | `list[int]`  | Hidden layer dimensions for the interaction MLP   |

**Layer Breakdown:**

1. **Continuous Layer:** `nn.Linear(num_features, mlp_layers[0])` -- projects dense features into the embedding dimension.
2. **Embedding Tables:** One `nn.Embedding(vocab_size, mlp_layers[0])` per categorical feature. For the API serving configuration, this includes 2 base categorical features plus 73 genre embeddings (75 total).
3. **MLP (Multi-Layer Perceptron):** A `nn.Sequential` stack of `Linear -> ReLU` layers. The input dimension equals `mlp_layers[0] + sum(embedding_dims)`. The hidden layers follow `mlp_layers[1:]`.
4. **Output Layer:** `nn.Linear(mlp_layers[-1], 1)` followed by `nn.Sigmoid()`, producing a score in `[0, 1]`.

**Forward Pass:**

```python
def forward(self, continuous_features: Tensor, categorical_features: Tensor) -> Tensor:
    # 1. Project continuous features
    # 2. Look up embeddings for each categorical feature
    # 3. Concatenate continuous projection + all embeddings
    # 4. Pass through MLP layers
    # 5. Output layer + sigmoid -> score in [0, 1]
```

**Validation:**
- Raises `ValueError` if inputs are `None`.
- Raises `ValueError` if categorical feature count exceeds embedding table count.
- Raises `ValueError` if concatenated tensor width mismatches MLP input dimension.

### API Serving Configuration

When loaded by `api/app.py`, the model uses:
- `num_continuous_features = 2`
- `num_categorical_features = 2` (user-provided) + 73 genre padding = 75 embedding tables
- `embedding_sizes = [2, 18] + [2] * 73`
- `mlp_layers` from `configs/config.yaml` (default: `[64, 32, 16]`)

---

## 7. Training

### Training Pipeline

The training script (`scripts/train.py`) wraps the DLRM in a **PyTorch Lightning** module for managed training with automatic logging and checkpointing.

**File:** `scripts/train.py` -- class `DLRMTrainer(pl.LightningModule)`

### Running Training

```bash
uv run python scripts/train.py
```

### How It Works

1. **Data:** Currently uses synthetic data via `get_dataloader()` -- 1000 samples with 10 continuous features, 5 categorical features, and binary labels. Replace with real data for production use.
2. **Optimization:** Manual optimization with two separate optimizers:
   - `Adam` for dense (non-embedding) parameters
   - `SparseAdam` for embedding parameters
3. **Loss Function:** Binary Cross-Entropy (`nn.BCELoss`)
4. **Checkpointing:** Top-3 checkpoints saved by `train_loss` to `lightning_logs/checkpoints/` with format `dlrm-{epoch:02d}-{train_loss:.2f}`. The last checkpoint is always saved.
5. **Accelerator:** Automatically uses GPU if CUDA is available, otherwise CPU.

### Training Configuration (from `config.yaml`)

| Parameter         | Default Value           | Description                         |
|-------------------|-------------------------|-------------------------------------|
| `num_features`    | `10`                    | Number of continuous input features |
| `embedding_sizes` | `[10, 10, 10, 10, 10]` | Vocabulary sizes per categorical    |
| `mlp_layers`      | `[64, 32, 16]`         | MLP hidden layer dimensions         |
| `learning_rate`   | `0.001`                 | Adam/SparseAdam learning rate       |
| `epochs`          | `5`                     | Number of training epochs           |
| `batch_size`      | `32`                    | Training batch size                 |

### Checkpoint Output

```
lightning_logs/
|-- checkpoints/
|   |-- dlrm-epoch=00-train_loss=0.70.ckpt
|   |-- dlrm-epoch=04-train_loss=0.65.ckpt
|   |-- last.ckpt
```

### Exporting for API Serving

After training, export model weights as a `.pth` file for the API:

```python
torch.save(model.model.state_dict(), "trained_model.pth")
```

The API loads `trained_model.pth` from the project root at startup.

---

## 8. Inference

### Standalone Inference Script

**File:** `scripts/inference.py`

A lightweight script for testing model predictions outside the API.

```bash
uv run python scripts/inference.py
```

**How it works:**

1. Instantiates a `DLRMModel` with the training configuration (10 features, 5 embeddings of size 10, MLP `[64, 32, 16]`).
2. Loads weights from `trained_model.pth`.
3. Sets the model to evaluation mode (`model.eval()`).
4. Creates sample input tensors (1 continuous sample of 10 features, 1 categorical sample of 5 features).
5. Runs a forward pass and logs the recommendation score.

### API Inference Flow

When the API receives a prediction request:

1. **Input validation:** Pydantic checks types; the endpoint verifies exactly 2 categorical features.
2. **Feature preparation:** Categorical features are padded with 73 zero-valued genre slots to match the 75 embedding tables.
3. **Tensor creation:** Continuous features become a `float32` tensor; categorical features become an `int64` tensor.
4. **Forward pass:** `model(continuous_tensor, categorical_tensor)` returns a scalar score.
5. **TMDB integration:** Popular movies are fetched asynchronously from TMDB via `httpx.AsyncClient`.
6. **Ranking:** Movies are sorted by `abs(movie_score - prediction_score)` in descending order; top 5 are returned.

---

## 9. Configuration

### config.yaml

**File:** `configs/config.yaml`

The single source of truth for model hyperparameters. Loaded and validated at API startup.

```yaml
model:
  num_features: 10
  embedding_sizes: [10, 10, 10, 10, 10]
  mlp_layers: [64, 32, 16]
  learning_rate: 0.001
  epochs: 5
  batch_size: 32
```

**Required Keys (validated at startup):**

| Section | Key               | Type         | Description                              |
|---------|-------------------|--------------|------------------------------------------|
| `model` | `num_features`    | `int`        | Number of continuous features             |
| `model` | `embedding_sizes` | `list[int]`  | Embedding vocabulary sizes                |
| `model` | `mlp_layers`      | `list[int]`  | MLP hidden layer widths                   |
| `model` | `learning_rate`   | `float`      | Optimizer learning rate                   |
| `model` | `epochs`          | `int`        | Training epochs                           |
| `model` | `batch_size`      | `int`        | Batch size for training                   |

If any required key is missing, the API exits with a `SystemExit(1)` and logs the specific missing key.

### Environment Variables

**File:** `.env.example` (copy to `.env`)

| Variable             | Required | Description                                  |
|----------------------|----------|----------------------------------------------|
| `TMDB_API_KEY`       | Yes      | TMDB v3 API key                              |
| `TMDB_BEARER_TOKEN`  | Yes      | TMDB v4 read-access bearer token             |

Obtain both at https://www.themoviedb.org/settings/api.

Environment variables are loaded via `python-dotenv` at startup. If either is missing, the API logs a warning but still starts -- however, prediction endpoints will return `502` when they attempt to fetch movies from TMDB.

---

## 10. Testing

### Running Tests

```bash
# Via Makefile
make test

# Directly
uv run pytest tests/ -v
```

### Test Suite Structure

**File:** `tests/test_model.py`

The test suite is organized into four test classes:

#### `TestDLRMModel` -- Model Unit Tests

| Test                                    | Description                                       |
|-----------------------------------------|---------------------------------------------------|
| `test_forward_output_shape`             | Output shape is `(1, 1)` for single sample        |
| `test_forward_output_range`             | Output is in `[0, 1]` (sigmoid)                   |
| `test_forward_rejects_none_inputs`      | `ValueError` on `None` continuous input            |
| `test_forward_rejects_none_categorical` | `ValueError` on `None` categorical input           |
| `test_forward_rejects_too_many_categorical` | `ValueError` when exceeding embedding tables   |
| `test_batch_forward`                    | Correct shape `(4, 1)` for batch of 4             |
| `test_single_categorical_feature`       | Handles fewer categorical features than tables     |
| `test_empty_batch_raises`               | Zero-batch tensors produce `(0, 1)` output         |
| `test_parameter_count_positive`         | Model has a non-trivial parameter count            |

#### `TestHealthEndpoints` -- Health Check Tests

| Test                    | Description                                    |
|-------------------------|------------------------------------------------|
| `test_root_endpoint`    | `GET /` returns 200 with `"healthy"` status    |
| `test_health_endpoint`  | `GET /health` returns version and model status |

#### `TestModelInfoEndpoint` -- Model Info Tests

| Test              | Description                                           |
|-------------------|-------------------------------------------------------|
| `test_model_info` | `GET /api/v1/models` returns architecture metadata    |

#### `TestPredictEndpoint` -- Prediction Tests

| Test                              | Description                                      |
|-----------------------------------|--------------------------------------------------|
| `test_predict_endpoint`           | Valid prediction returns 200 with movie data     |
| `test_predict_legacy_endpoint`    | Legacy `/predict/` still works                   |
| `test_predict_bad_categorical_count` | Wrong categorical count returns 400           |
| `test_predict_empty_continuous`   | Empty continuous features handled gracefully     |
| `test_predict_wrong_types_rejected` | String inputs return 422                       |
| `test_predict_missing_fields`     | Missing fields return 422                        |

#### `TestStructuredErrors` -- Error Handling Tests

| Test                              | Description                                      |
|-----------------------------------|--------------------------------------------------|
| `test_404_returns_structured_error` | 404s return structured JSON with `success: false`|

### Fixtures

- **`dlrm_model`** -- A small `DLRMModel(num_features=10, embedding_sizes=[10]*5, mlp_layers=[64,32,16])` for unit tests.
- **`sample_inputs`** -- A tuple of `(continuous_tensor, categorical_tensor)` with shapes `(1, 10)` and `(1, 5)`.
- **`test_client`** -- A `fastapi.testclient.TestClient` wrapping the FastAPI app. Imported lazily to avoid model-loading side effects.

### Mocking

Prediction tests use `unittest.mock.patch` with `AsyncMock` to mock `api.app.fetch_real_movies`, avoiding real TMDB API calls during tests.

---

## 11. Deployment

### Docker

**File:** `Dockerfile`

The image is built on `python:3.11-slim` with `uv` for fast dependency installation.

```bash
# Build
make docker-build
# Equivalent to: docker build -t recommender-system .

# Run
make docker-run
# Equivalent to: docker run --rm -p 8000:8000 --env-file .env recommender-system
```

**Dockerfile Highlights:**

| Stage                  | Detail                                                     |
|------------------------|------------------------------------------------------------|
| Base image             | `python:3.11-slim`                                         |
| Package manager        | `uv` (copied from `ghcr.io/astral-sh/uv:latest`)          |
| System deps            | `gcc`, `build-essential`, `libffi-dev`, `curl`             |
| Dependency caching     | `pyproject.toml` copied first for layer caching            |
| Port                   | `8000`                                                     |
| Health check           | `curl -f http://localhost:8000/health` every 30s           |
| Entrypoint             | `uvicorn api.app:app --host 0.0.0.0 --port 8000`          |

**Docker Ignore (`.dockerignore`):** Excludes `.venv/`, `__pycache__/`, `.git/`, `.env`, `lightning_logs/`, `data/`, `tests/`, `ci_cd/`, `*.md`, and caches from the build context.

### AWS ECS Deployment

The project includes AWS ECS configuration files for container deployment.

**Container Definition (`container-definition.json`):**

```json
[
  {
    "name": "cinemascoperecsys",
    "image": "804859200836.dkr.ecr.us-east-1.amazonaws.com/cinemascoperecsys:latest",
    "portMappings": [{ "containerPort": 8000, "protocol": "tcp" }],
    "essential": true
  }
]
```

**IAM Policies:**

- `trust-policy.json` -- Allows the `ecs-tasks.amazonaws.com` service to assume the execution role.
- `iam-policy.json` -- Grants `iam:UpdateAssumeRolePolicy` on the ECS task execution role.

**Deployment Steps (high-level):**

1. Build and push the Docker image to AWS ECR.
2. Register the task definition using `container-definition.json`.
3. Create or update the ECS service to use the new task definition.
4. ECS health checks use the `/health` endpoint (Docker `HEALTHCHECK` directive).

### CI/CD Pipeline

**File:** `ci_cd/github_actions.yml`

Triggered on push to `main`. The pipeline:

1. Checks out the repository.
2. Sets up Python 3.9.
3. Installs dependencies.
4. Runs the test suite with `pytest tests/`.

> **Note:** The CI/CD pipeline currently uses `pip install -r requirements.txt`. Consider updating to use `uv` for consistency with local development.

---

## 12. Makefile Reference

All developer workflow commands are available through the Makefile.

| Target          | Command                                                          | Description                                         |
|-----------------|------------------------------------------------------------------|-----------------------------------------------------|
| `make install`  | `uv sync --all-extras`                                           | Install all dependencies including dev extras        |
| `make run`      | `uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload`| Start the FastAPI dev server with hot reload         |
| `make test`     | `uv run pytest tests/ -v`                                        | Run the full test suite with verbose output          |
| `make docker-build` | `docker build -t recommender-system .`                       | Build the Docker image                               |
| `make docker-run`   | `docker run --rm -p 8000:8000 --env-file .env recommender-system` | Run the container with env vars from `.env`    |
| `make lint`     | `uv run ruff check . && uv run ruff format --check .`           | Lint and check formatting with Ruff                  |
| `make clean`    | Removes `__pycache__/`, `*.pyc`, `.pytest_cache`, `.ruff_cache`, `dist/`, `build/`, `*.egg-info`, `.venv` | Remove all build artifacts and caches |

---

## 13. Troubleshooting

### Model Not Found

**Symptom:** `/health` reports `"model_loaded": false`; `/api/v1/predict` returns `503`.

**Cause:** `trained_model.pth` is missing from the project root.

**Fix:**
1. Run the training script: `uv run python scripts/train.py`
2. Export the model weights: `torch.save(model.model.state_dict(), "trained_model.pth")`
3. Restart the server.

---

### TMDB API Key Errors

**Symptom:** `/api/v1/predict` returns `502` with `"TMDB_BEARER_TOKEN is not configured"`.

**Cause:** `.env` file is missing or TMDB credentials are not set.

**Fix:**
1. Copy the example: `cp .env.example .env`
2. Register at https://www.themoviedb.org/settings/api to obtain your API key and bearer token.
3. Fill in `TMDB_API_KEY` and `TMDB_BEARER_TOKEN` in `.env`.
4. Restart the server (dotenv is loaded at startup).

---

### CUDA / GPU Issues

**Symptom:** Training crashes with CUDA-related errors, or the model loads on the wrong device.

**Cause:** PyTorch CUDA mismatch or no GPU available.

**Fix:**
- The API forces CPU mode (`map_location=torch.device("cpu")` in `app.py`; `torch.set_num_threads(1)` in `dlrm.py`).
- Training auto-detects GPU via `accelerator="gpu" if torch.cuda.is_available() else "cpu"`.
- If you encounter CUDA errors during training, install the correct PyTorch CUDA version: `uv pip install torch --index-url https://download.pytorch.org/whl/cu121`

---

### Port Already in Use

**Symptom:** `[Errno 48] Address already in use` when starting the server.

**Fix:**
```bash
# Find the process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or start on a different port
uv run uvicorn api.app:app --host 0.0.0.0 --port 8001 --reload
```

---

### Config Validation Failure

**Symptom:** `SystemExit: 1` at startup with `"Missing required config section"` or `"Missing required config key"`.

**Cause:** `configs/config.yaml` is missing or incomplete.

**Fix:** Ensure `configs/config.yaml` contains all required keys under the `model` section: `num_features`, `embedding_sizes`, `mlp_layers`, `learning_rate`, `epochs`, `batch_size`.

---

### Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'models'`

**Fix:** Run scripts from the project root, or ensure the project is installed:
```bash
cd RecommenderSystem
uv run python scripts/train.py
```

---

## 14. Contributing

### Development Setup

```bash
# Install with dev dependencies
make install

# Run linting
make lint

# Run tests
make test
```

### Code Style

- **Linter/Formatter:** Ruff (configured in `pyproject.toml`)
- **Line length:** 120 characters
- **Target version:** Python 3.10
- **Lint rules:** `E` (pycodestyle errors), `F` (pyflakes), `I` (isort), `W` (pycodestyle warnings)

### Branching Strategy

- `main` -- production branch; CI/CD triggers on push.
- Feature branches should be merged via pull request.

### Pull Request Checklist

- [ ] All tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] New endpoints have corresponding tests in `tests/test_model.py`
- [ ] Config changes are reflected in `configs/config.yaml` and this documentation
- [ ] Breaking API changes update the version in `api/app.py`

---

## 15. Changelog

### v1.0.0 (Current)

- Initial release of the CinemaScopeAI Recommender System.
- DLRM model implementation with continuous and categorical feature support.
- FastAPI serving with versioned endpoints (`/api/v1/predict`, `/api/v1/models`).
- Legacy `/predict/` endpoint for backwards compatibility.
- Structured error handling with consistent JSON error responses.
- Request/response logging middleware.
- PyTorch Lightning training pipeline with checkpoint management.
- TMDB API integration for real-time movie data.
- Docker containerization with uv and health checks.
- AWS ECS deployment configuration.
- GitHub Actions CI/CD pipeline.
- Comprehensive test suite with model and API coverage.

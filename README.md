# 🎬 RecommenderSystem

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Live on Render](https://img.shields.io/badge/Live%20on-Render-46E3B7?logo=render&logoColor=white)](https://recommendersystem-l993.onrender.com)

A movie recommendation API powered by a Deep Learning Recommendation Model (DLRM). You send it your preferences, it runs them through a neural network, pulls real movie data from TMDB (posters, ratings, summaries — the works), and hands you back personalized recommendations. This is the backend that powers [CinemaScopeAI](https://github.com/AkinCodes/CinemaScopeAI), an iOS app for discovering movies you'll actually want to watch.

---

## Live Demo

The API is live right now. Try it:

```bash
curl -X POST https://recommendersystem-l993.onrender.com/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"continuous_features": [0.5, 0.8], "categorical_features": [1, 2]}'
```

> **Note:** It's on Render's free tier, so the first request might take ~30 seconds to wake up. After that, responses come back fast.

You'll get something like this:

```json
[
  {
    "title": "The Shawshank Redemption",
    "genre": "Unknown",
    "rating": "N/A",
    "score": 0.87,
    "poster_url": "https://image.tmdb.org/t/p/w500/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg",
    "director": "N/A",
    "release_year": 1994,
    "summary": "Imprisoned in the 1940s for the double murder of his wife and her lover..."
  },
  {
    "title": "Interstellar",
    "genre": "Unknown",
    "rating": "N/A",
    "score": 0.84,
    "poster_url": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
    "director": "N/A",
    "release_year": 2014,
    "summary": "The adventures of a group of explorers who make use of a newly discovered wormhole..."
  }
]
```

Five movies, ranked by how well they match your taste profile, each with a poster URL you can render directly in a client app.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT REQUEST                              │
│  POST /api/v1/predict                                               │
│  { "continuous_features": [0.5, 0.8],                               │
│    "categorical_features": [1, 2] }                                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INPUT VALIDATION                               │
│  Pydantic checks types + FastAPI validates feature counts           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DLRM MODEL (PyTorch)                          │
│                                                                     │
│  continuous features ──► Dense Layer ──────────┐                    │
│                                                 ├──► MLP ──► Score  │
│  categorical features ──► Embedding Tables ────┘    [64→32→16→1]   │
│                                                      + Sigmoid      │
│                                                      → [0.0, 1.0]  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     TMDB API (async via httpx)                      │
│  Fetches popular movies: titles, ratings, poster paths, summaries   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       RANKING + RESPONSE                            │
│  Sorts movies by how closely they match the prediction score        │
│  Returns top 5 with full metadata + poster URLs                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Step by step:**

1. **You send preferences** — two continuous features (numbers describing your taste, like how much you value action vs. drama) and two categorical features (discrete choices, like preferred era or language).
2. **DLRM processes them** — the model takes your number preferences through a dense layer and your category choices through separate embedding tables. Each embedding turns a category ID into a learned vector. Then everything gets concatenated and pushed through a multi-layer perceptron (64 → 32 → 16 neurons) that outputs a single score between 0 and 1.
3. **Real movies get fetched** — the API calls TMDB asynchronously to grab currently popular movies with their metadata.
4. **Results get ranked** — movies are sorted by how well their TMDB rating aligns with the model's prediction score, and the top 5 are returned with titles, posters, release years, and summaries.

---

## Features

### The Model

The DLRM (Deep Learning Recommendation Model) is built from scratch in PyTorch. What makes it interesting:

- **Handles mixed feature types** — continuous features (like numeric taste scores) go through a dense layer, while categorical features (like genre preferences) each get their own embedding table. This is how real recommendation systems at companies like Meta handle the mix of "how much" and "which one" data.
- **Embedding tables** — 75 total embeddings (2 for user-level categories + 73 for genre-level features), each mapping category IDs to learned vectors.
- **MLP interaction layer** — a 3-layer network (64 → 32 → 16) with ReLU activations that learns how continuous and categorical signals interact.
- **Sigmoid output** — produces a score between 0 and 1, representing predicted preference strength.
- **Input validation** — the forward pass checks for None inputs, mismatched feature counts, and MLP shape mismatches, raising clear errors instead of crashing silently.

### Real Movie Data

- **TMDB integration** — fetches currently popular movies from The Movie Database API, so recommendations always reflect what's actually out there.
- **Async fetching** — uses `httpx.AsyncClient` so the API doesn't block while waiting on TMDB.
- **Full metadata** — each recommendation comes with title, rating, poster URL (ready to render in a UI), release year, and a plot summary.

### API Design

- **Versioned endpoints** — `/api/v1/predict` with a legacy `/predict/` that delegates to it for backward compatibility.
- **Health checks** — `GET /health` reports model status, app version, and whether everything is loaded.
- **Model introspection** — `GET /api/v1/models` returns architecture details, parameter count, device info, and layer configuration.
- **Structured errors** — every error (400, 404, 422, 500, 502, 503) returns consistent JSON with `success`, `error`, and `detail` fields. No raw stack traces in production.
- **Request logging middleware** — every request gets logged with method, path, status code, and duration in milliseconds.

### Developer Experience

- **Makefile** — one command for anything: `make run`, `make test`, `make lint`, `make docker-build`.
- **uv for dependency management** — fast, reproducible installs via `pyproject.toml`.
- **Docker with health checks** — production Dockerfile includes a `HEALTHCHECK` directive that pings `/health` every 30 seconds.
- **Comprehensive tests** — 16+ tests covering model forward passes, API endpoints, input validation, edge cases (empty batches, wrong types, missing fields), and structured error responses.
- **Ruff for linting** — fast Python linting and formatting with a 120-char line length.

---

## Tech Stack

| Technology | Why |
|---|---|
| **PyTorch** | Full control over the DLRM architecture — custom forward pass, manual embedding tables, easy to extend |
| **PyTorch Lightning** | Handles the training loop boilerplate — checkpointing, logging, GPU/CPU switching, validation splits |
| **FastAPI** | Async by default, automatic OpenAPI docs, Pydantic validation on every request, and it's fast |
| **httpx** | Async HTTP client for non-blocking TMDB API calls inside async FastAPI endpoints |
| **uv** | 10-100x faster than pip for dependency resolution and installs |
| **Pydantic** | Type-safe request/response schemas that auto-generate API documentation |
| **Docker** | Consistent environment from dev to production, with multi-stage builds and health checks |
| **Ruff** | Linting + formatting in one tool, written in Rust, runs in milliseconds |
| **pytest** | Test framework with fixtures, parametrize, and async support for testing FastAPI endpoints |
| **Render** | Simple container deployment with auto-deploy from GitHub pushes |

---

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A TMDB API key ([get one free here](https://www.themoviedb.org/settings/api))

### 1. Clone the repo

```bash
git clone https://github.com/AkinCodes/RecommenderSystem.git
cd RecommenderSystem
```

### 2. Install dependencies

```bash
make install
```

This runs `uv sync --all-extras`, which installs everything from `pyproject.toml` including dev dependencies (pytest, ruff, etc.).

### 3. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and add your TMDB credentials:

```
TMDB_API_KEY=your_tmdb_api_key_here
TMDB_BEARER_TOKEN=your_tmdb_bearer_token_here
```

### 4. Start the dev server

```bash
make run
```

This starts uvicorn with hot reload at `http://localhost:8000`. You should see:

```
INFO:     Configuration loaded and validated successfully.
INFO:     DLRM model loaded successfully from 'trained_model.pth'.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 5. Try it out

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"continuous_features": [0.5, 0.8], "categorical_features": [1, 2]}'
```

Or visit `http://localhost:8000/docs` for the interactive Swagger UI.

### 6. Run the tests

```bash
make test
```

You should see all tests pass:

```
tests/test_model.py::TestDLRMModel::test_forward_output_shape PASSED
tests/test_model.py::TestDLRMModel::test_forward_output_range PASSED
tests/test_model.py::TestPredictEndpoint::test_predict_endpoint PASSED
...
```

---

## API Reference

### `GET /`

Status check. Returns a simple alive message.

```json
{ "status": "healthy", "message": "Recommendation API is running." }
```

### `GET /health`

Health check with model status and version info.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### `GET /api/v1/models`

Returns details about the loaded DLRM model.

```json
{
  "architecture": "DLRM (Deep Learning Recommendation Model)",
  "num_parameters": 42817,
  "device": "cpu",
  "num_continuous_features": 2,
  "num_categorical_features": 2,
  "mlp_layers": [64, 32, 16]
}
```

### `POST /api/v1/predict`

The main endpoint. Send user preferences, get back 5 movie recommendations.

**Request:**

```json
{
  "continuous_features": [0.5, 0.8],
  "categorical_features": [1, 2]
}
```

**Response** (200):

```json
[
  {
    "title": "Movie Title",
    "genre": "Unknown",
    "rating": "N/A",
    "score": 0.85,
    "poster_url": "https://image.tmdb.org/t/p/w500/poster.jpg",
    "director": "N/A",
    "release_year": 2024,
    "summary": "A brief plot overview from TMDB."
  }
]
```

**Error responses:**

| Status | When |
|---|---|
| 400 | Wrong number of categorical features |
| 422 | Invalid types (strings instead of numbers, missing fields) |
| 502 | TMDB API is down or returned no movies |
| 503 | Model isn't loaded |

All errors return structured JSON:

```json
{
  "success": false,
  "error": "HTTP 400",
  "detail": "Expected 2 categorical features, got 1."
}
```

### `POST /predict/`

Legacy endpoint. Delegates to `/api/v1/predict` — same request format, same response.

---

## Training the Model

The training pipeline uses PyTorch Lightning, which handles checkpointing, logging, and hardware switching automatically.

```bash
uv run python scripts/train.py
```

This will:
1. Create a synthetic dataset (1000 samples with 10 continuous + 5 categorical features)
2. Train the DLRM for 5 epochs with BCE loss
3. Save checkpoints to `lightning_logs/checkpoints/`
4. Run validation after training
5. Load the best checkpoint and run a test inference

### Config Parameters

All hyperparameters live in `configs/config.yaml`:

| Parameter | Default | What it controls |
|---|---|---|
| `num_features` | `10` | Number of continuous input features |
| `embedding_sizes` | `[10, 10, 10, 10, 10]` | Vocabulary size for each categorical embedding table |
| `mlp_layers` | `[64, 32, 16]` | Hidden layer dimensions for the interaction MLP |
| `learning_rate` | `0.001` | Adam optimizer learning rate |
| `epochs` | `5` | Number of training epochs |
| `batch_size` | `32` | Samples per training batch |

### How Checkpoints Work

The `ModelCheckpoint` callback saves the top 3 models by training loss, plus a `last.ckpt` that always has the most recent weights. Checkpoint files are named like `dlrm-epoch=04-train_loss=0.65.ckpt`. To use a trained model in the API, export its state dict to `trained_model.pth` in the project root.

### Dual Optimizer Setup

The trainer separates dense parameters (linear layers) and sparse parameters (embeddings) into two different optimizers — Adam for dense, SparseAdam for sparse. This is a standard pattern in recommendation systems because embedding gradients are naturally sparse (only the rows that were looked up get updated).

---

## Makefile Reference

| Command | What it does |
|---|---|
| `make install` | Install all dependencies including dev extras via `uv sync --all-extras` |
| `make run` | Start the FastAPI dev server with hot reload on port 8000 |
| `make test` | Run the full test suite with verbose output |
| `make lint` | Check code with Ruff (linting + format check) |
| `make docker-build` | Build the Docker image as `recommender-system` |
| `make docker-run` | Run the container on port 8000, injecting `.env` variables |
| `make clean` | Remove `__pycache__`, `.pyc`, `.pytest_cache`, `.ruff_cache`, and build artifacts |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `TMDB_API_KEY` | Yes | TMDB v3 API key (used for authentication) |
| `TMDB_BEARER_TOKEN` | Yes | TMDB v4 read-access token (used in Bearer auth header) |

Get both at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api) — you'll need a free account.

The app will still start without them, but `model_loaded` will be true and predictions will fail at the TMDB fetch step with a 502 error.

---

## Project Structure

```
RecommenderSystem/
├── api/
│   └── app.py                 # FastAPI app — endpoints, middleware, TMDB client, error handling
├── models/
│   ├── __init__.py
│   └── dlrm.py                # DLRM model — embeddings, dense layers, MLP, forward pass
├── scripts/
│   ├── train.py               # PyTorch Lightning training loop with checkpointing
│   └── inference.py           # Standalone inference script for testing the model locally
├── configs/
│   └── config.yaml            # Hyperparameters — features, layers, learning rate, epochs
├── tests/
│   ├── __init__.py
│   └── test_model.py          # 16+ tests — model unit tests, API integration, error handling
├── data/
│   └── netflix_titles.csv     # Dataset for experimentation
├── docker/
│   └── Dockerfile             # Lightweight Dockerfile (Python 3.9, pip)
├── ci_cd/
│   └── github_actions.yml     # GitHub Actions CI pipeline — test on push to main
├── Dockerfile                 # Production Dockerfile (Python 3.11, uv, health checks)
├── Makefile                   # Dev shortcuts — install, run, test, lint, docker, clean
├── pyproject.toml             # Project metadata, dependencies, tool config (ruff, pytest)
├── task-definition.json       # AWS ECS Fargate task definition
├── trust-policy.json          # AWS IAM trust policy for ECS execution role
├── .env.example               # Template for required environment variables
├── trained_model.pth          # Serialized model weights (loaded at startup)
└── README.md                  # You are here
```

---

## Deployment

### Docker (local)

```bash
make docker-build
make docker-run
```

The production Dockerfile uses Python 3.11 with uv for fast installs and includes a health check that pings `/health` every 30 seconds.

### Render

The API is deployed on Render at **https://recommendersystem-l993.onrender.com**. It auto-deploys on every push to `main`. Render builds the Docker image, sets the environment variables, and runs the container.

### AWS ECS (Fargate)

The repo includes an ECS task definition (`task-definition.json`) configured for Fargate with 256 CPU units and 512MB memory. The container image is pushed to ECR at `804859200836.dkr.ecr.us-east-1.amazonaws.com/cinemascope-recsys:latest`.

To deploy to ECS:

```bash
# Build and push to ECR
docker build -t cinemascope-recsys .
docker tag cinemascope-recsys:latest 804859200836.dkr.ecr.us-east-1.amazonaws.com/cinemascope-recsys:latest
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 804859200836.dkr.ecr.us-east-1.amazonaws.com
docker push 804859200836.dkr.ecr.us-east-1.amazonaws.com/cinemascope-recsys:latest

# Register task and run
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs update-service --cluster your-cluster --service your-service --force-new-deployment
```

---

## What I'd Improve Next

- **More model architectures** — add a DCN (Deep & Cross Network) and a two-tower model, let users pick which one scores their preferences
- **A/B testing framework** — serve different models to different users and track which one drives better engagement
- **User profiles** — store preference history so the model can learn from past interactions, not just a single request
- **Genre enrichment** — use TMDB genre IDs to fill in the "Unknown" genre field and enable genre-aware filtering
- **Director/cast data** — hit the TMDB credits endpoint to return actual director names and top cast
- **Batch predictions** — accept multiple user profiles in one request for bulk recommendation scenarios
- **Model versioning** — track which model version served each prediction for reproducibility
- **Rate limiting** — protect the TMDB integration from getting throttled under heavy load
- **Caching** — Redis or in-memory cache for TMDB responses since popular movies don't change every second

---

## Related Projects

- **[CinemaScopeAI](https://github.com/AkinCodes/CinemaScopeAI)** — the iOS app that talks to this API. Built with SwiftUI, displays recommendations with posters, ratings, and summaries.
- **[MoviePosterAI](https://github.com/AkinCodes/MoviePosterAI)** — poster analysis using computer vision.

---

## Author

**Akin Olusanya**

[LinkedIn](https://www.linkedin.com/in/akindeveloper/) · [GitHub](https://github.com/AkinCodes) · workwithakin@gmail.com

# 🎬 RecommenderSystem

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Live on Render](https://img.shields.io/badge/Live%20on-Render-46E3B7?logo=render&logoColor=white)](https://recommendersystem-l993.onrender.com)

A movie recommendation API powered by a Deep Learning Recommendation Model (DLRM). You send it your preferences, it runs them through a neural network, pulls real movie data from TMDB (posters, ratings, summaries — the works), and hands you back personalized recommendations. On top of the model, a LinUCB contextual-bandit layer adds **tunable exploration** — balancing proven favorites against items the model is unsure about — with online learning from feedback and off-policy evaluation. This is the backend that powers [CinemaScopeAI](https://github.com/AkinCodes/CinemaScopeAI), an iOS app for discovering movies you'll actually want to watch.

---

## Live Demo

The API is live right now. Try it:

```bash
curl -X POST https://recommendersystem-l993.onrender.com/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"continuous_features": [0.76, 0.5, 0.3, 0.4, 0.72, 0.6, 0.8, 0.52], "categorical_features": [1, 2]}'
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
│  { "continuous_features": [0.76, 0.5, 0.3, 0.4, 0.72, 0.6, 0.8, 0.52],│
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
│  categorical features ──► Embedding Tables ────┘    [128→64→32→1]  │
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

1. **You send preferences** — 8 continuous features (engineered stats like average rating, activity, item popularity) and 2 categorical features (user ID and item ID).
2. **DLRM processes them** — the model takes your 8 continuous features through a dense layer and your category choices through separate embedding tables (`[943, 1682]`, each 64-dim). It also computes an element-wise user × item interaction (the two embedding vectors multiplied together). Everything gets concatenated and pushed through a multi-layer perceptron (128 → 64 → 32 neurons) with Dropout(0.2) that outputs a single score between 0 and 1.
3. **Real movies get fetched** — the API calls TMDB asynchronously to grab currently popular movies with their metadata.
4. **Results get ranked** — movies are sorted by how well their TMDB rating aligns with the model's prediction score, and the top 5 are returned with titles, posters, release years, and summaries.

---

## Features

### The Model

The DLRM (Deep Learning Recommendation Model) is built from scratch in PyTorch. What makes it interesting:

- **Handles mixed feature types** — 8 continuous features (engineered user/item stats) go through a dense layer, while categorical features (user ID, item ID) each get their own embedding table. This is how real recommendation systems at companies like Meta handle the mix of "how much" and "which one" data.
- **Embedding tables** — `[943, 1682]` — two embedding tables (users, items), each 64-dimensional (`embedding_dim`, decoupled from the MLP widths).
- **Explicit feature interaction** — computes element-wise user × item embedding product before the MLP, capturing direct collaborative filtering signal.
- **MLP interaction layer** — a 3-layer network (128 → 64 → 32) with ReLU activations and Dropout(0.2) that learns how continuous features, embeddings, and their interaction combine.
- **Binary classification** — ratings >= 4 are positive, trained with BCELoss. Sigmoid output produces a score between 0 and 1.
- **Regularization** — Dropout(0.2), weight decay (1e-5), early stopping (patience=5).
- **Input validation** — the forward pass checks for None inputs, mismatched feature counts, and MLP shape mismatches, raising clear errors instead of crashing silently.

### Exploration & Online Learning

The recommender doesn't only exploit what the DLRM already knows — it can explore.

- **Contextual bandit (LinUCB)** — `/api/v1/recommend_v3` reranks candidates by a common-scale blend of the DLRM score and a LinUCB exploration bonus, tuned by a single explore-weight `w` (default `0.2`). This surfaces items the model is *unsure* about instead of only its safest bets. See [Exploration layer](#exploration-layer-contextual-bandit).
- **Online learning** — `POST /api/v1/feedback` updates the bandit from observed rewards and atomically persists its state, so the policy improves after deployment without retraining the DLRM.
- **Off-policy evaluation** — the layer is validated on logged data with replay + IPS/SNIPS estimators and an explore-weight coverage/reward frontier, not just asserted to work.

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
- **Comprehensive tests** — 132 tests covering model forward passes, API endpoints, input validation, edge cases (empty batches, wrong types, missing fields), structured error responses, preprocessing config, DLRM interaction verification, plus the bandit math (Sherman-Morrison updates, scale-invariant blending), off-policy estimators (replay/IPS/SNIPS against synthetic ground truth), and exploration ordering effects.
- **Ruff for linting** — fast Python linting and formatting with a 120-char line length.

---

## Tech Stack

| Technology | Why |
|---|---|
| **PyTorch** | Full control over the DLRM architecture — custom forward pass, manual embedding tables, easy to extend |
| **FastAPI** | Async by default, automatic OpenAPI docs, Pydantic validation on every request, and it's fast |
| **httpx** | Async HTTP client for non-blocking TMDB API calls inside async FastAPI endpoints |
| **uv** | 10-100x faster than pip for dependency resolution and installs |
| **Pydantic** | Type-safe request/response schemas that auto-generate API documentation |
| **Docker** | Consistent environment from dev to production, with multi-stage builds and health checks |
| **Ruff** | Linting + formatting in one tool, written in Rust, runs in milliseconds |
| **pytest** | Test framework with fixtures, parametrize, and async support for testing FastAPI endpoints |
| **Faiss** | Approximate nearest-neighbor search for fast candidate retrieval in the two-stage pipeline |
| **W&B** | Experiment tracking — logs training curves, hyperparameters, and evaluation metrics |
| **Render** | Simple container deployment with auto-deploy from GitHub pushes |

---

## Benchmarks & Results

See [EVALUATION.md](EVALUATION.md) for full ablation studies, failure analysis, and reproduction instructions.

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Dataset | MovieLens 100K (100,000 ratings, 943 users, 1,682 items) |
| Embeddings | `[943, 1682]` x 64-dim (user, item) |
| Continuous features | 8 dense features → Linear(8, 64) |
| MLP | `[128 → 64 → 32]` with ReLU + Dropout(0.2) between layers |
| Output | Linear(32, 1) → Sigmoid |
| Loss | BCELoss (binary: rating >= 4 is positive) |
| Optimizer | Adam (lr=0.001, weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Regularization | Dropout 0.2, early stopping (patience=5) |
| Training | 20 epochs max, batch_size=256 |
| Parameters | ~372K total (~90% embeddings, ~10% MLP + dense layers) |

### Input Features (8 total)

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 0 | `user_mean_rating` | User | Average rating the user gave, normalised by max rating (5.0) |
| 1 | `user_rating_count` | User | Number of ratings by user, normalised by max count in training set |
| 2 | `user_rating_var` | User | Variance of user's ratings, normalised by max possible variance (4.0) |
| 3 | `user_days_active` | User | Days between first and last rating, normalised |
| 4 | `item_mean_rating` | Item | Average rating the item received, normalised |
| 5 | `item_rating_count` | Item | Number of ratings for item, normalised |
| 6 | `item_popularity_rank` | Item | Rank-ordered popularity, normalised to [0, 1] (1.0 = most popular) |
| 7 | `user_item_deviation` | Interaction | `user_mean - item_mean`, shifted from [-1, 1] to [0, 1] |

### Offline Evaluation Metrics (DLRM)

| Metric | Value |
|--------|-------|
| NDCG@10 | 0.6723 |
| Precision@10 | 0.5774 |
| Recall@10 | 0.3758 |
| HitRate@10 | 0.9965 |
| AUC | 0.5541 |

> Evaluated on 283 held-out users, 20K test ratings. Train time: ~13s. Inference latency: ~0.05ms/user.

### Classical Baselines Comparison

| Model | Approach | NDCG@10 | Prec@10 | Hit@10 |
|-------|----------|---------|---------|--------|
| **LogReg** | StandardScaler → Logistic Regression (C=1.0, lbfgs) on 8 features | **0.8023** | **0.6880** | **1.0000** |
| XGBoost | 200 trees, max_depth=6, lr=0.1, subsample=0.8 on 8 features | 0.7836 | 0.6707 | 1.0000 |
| LightGBM | 200 trees, max_depth=6, lr=0.1, subsample=0.8 on 8 features | 0.7820 | 0.6661 | 0.9965 |
| DLRM | Learned user/item embeddings (64-dim) + interaction + 3-layer MLP | 0.6723 | 0.5774 | 0.9965 |

### Serving Architecture

| Endpoint | Strategy | Description |
|----------|----------|-------------|
| `/recommend/{user_id}` | Brute-force | Scores all 1,682 items in a single forward pass, returns top-K |
| `/recommend_v2/{user_id}` | Two-stage (ANN + rerank) | Faiss ANN retrieves top-100 candidates, DLRM reranks to top-K |
| `/recommend_v3/{user_id}` | Two-stage + bandit | Same retrieval as v2, reranked by a **common-scale blend** `(1-w)·norm(DLRM) + w·norm(LinUCB bonus)` (default `w=0.2`) |
| Cold-start | Popularity + explore blend | Unknown users get the popularity candidates reranked by the same `combine_explore` blend on v3 |

### Two-Stage Retrieval Pipeline

```
Stage 1: Candidate Generation (~1ms)          Stage 2: Reranking
┌──────────────────────────────┐       ┌─────────────────────────────┐
│  User embedding (64-dim)     │       │  Full DLRM forward pass     │
│         ↓                    │       │  on 100 candidates          │
│  Faiss IndexFlatIP (cosine)  │──────►│  8 features + embeddings    │
│  1,682 items → top 100       │       │  → top-K ranked results     │
└──────────────────────────────┘       └─────────────────────────────┘
```

- **Stage 1** uses Faiss inner-product search over normalised item embeddings to retrieve ~100 candidates in ~1ms.
- **Stage 2** runs the full DLRM (8 continuous features + user/item embeddings → MLP) on the candidate set and returns the final top-K.
- Falls back to sklearn `NearestNeighbors` (brute cosine) if `faiss-cpu` is not installed.

### Exploration layer (contextual bandit)

Pure exploitation always shows a user the items the DLRM already scores highest, so it never learns whether items it's *unsure* about would have landed. `/recommend_v3` adds a **LinUCB contextual bandit** on top of the two-stage pipeline to close that loop.

- **What it does** — for each candidate it builds an *interaction context* (the elementwise `user × item` embedding product — the same signal the DLRM's own interaction term uses) and computes the bandit's **exploration bonus** `alpha · √(xᵀ A⁻¹ x)`. Because the DLRM sigmoid score (~0.9–1.0) and the raw bonus (mean ~3.3) live on **completely different scales**, they are **not** summed directly (a raw sum just ranks by whichever term is larger). Instead both are min-max normalized across the candidate set to `[0, 1]` and convex-combined by a single explore-weight knob `w`: **`final = (1 - w)·norm(dlrm) + w·norm(bonus)`** (`combine_explore` in `models/bandit.py`). `w = 0` is pure exploit (identical to DLRM order), `w = 1` is pure exploration; the served default is **`w = 0.2`**. The bonus is large for contexts the bandit has seen little of and shrinks as it accumulates observations, so exploration decays automatically. Each result carries its `dlrm_score`, `exploration_bonus`, and an `is_exploratory` flag (true when the blend changed that item's rank vs pure-DLRM order). Cold-start users get the same blend applied over the popularity candidates using a **mean-user-embedding context** (popularity as the base, normalized the same way), so a new user's order is genuinely perturbed rather than being the identical static popularity list.
- **Online learning** — `POST /api/v1/feedback` takes `{user_id, item_id, reward}` (reward in `[0, 1]`, e.g. `1.0` if the user liked it), rebuilds the context from the stored embeddings, applies a rank-1 `LinUCB.update`, and atomically persists the state to `bandit_state.pkl` (loaded at startup, mirroring `serving_context.pkl`). The update + persist run under a lock, and the write is atomic (`os.replace`). This makes the layer a *real* online learner rather than a static bonus — post-deployment feedback moves the policy. **Single-worker assumption:** the bandit state is in-memory; running multiple uvicorn workers means each holds its own copy and they race the pickle (see EVALUATION.md).
- The bandit is pure NumPy (Sherman-Morrison rank-1 updates, no matrix re-inversion) and lives in `models/bandit.py`. `v1`/`v2` are left unchanged so v3 can be A/B-compared against them. `scripts/offline_bandit_eval.py` sweeps the explore weight `w` and saves an **explore-weight coverage/reward frontier** (`experiments/results/bandit_explore_frontier.png`) — the exploit signal it measures is the **real served DLRM sigmoid**, and it ranks by the same `combine_explore` blend the API serves.

**Result — the explore/exploit frontier** (off-policy evaluation over 20K held-out MovieLens events):

![Explore-weight coverage vs. reward frontier](experiments/results/bandit_explore_frontier.png)

| explore weight `w` | reward (SNIPS) | catalog coverage (distinct items) |
|---|---|---|
| 0.0 (pure DLRM) | 0.731 | 107 |
| 0.1 | 0.700 | 164 |
| **0.2 (served default)** | **0.695** | **180** |
| 0.3 | 0.662 | 190 |
| 0.5 | 0.654 | 195 |
| 1.0 (pure explore) | 0.585 | 196 |
| Most-Popular (reference) | 0.754 | 58 |

At the default `w = 0.2` the bandit keeps **~95% of the DLRM's reward** while lifting **catalog coverage +68% (107 → 180 distinct items)**. Most-Popular's higher SNIPS is a self-selection/replay artifact — it retains a tiny, popularity-biased subset of events while surfacing only 58 items (SNIPS is the estimator to trust here; absolute IPS is uninterpretable). MovieLens is static, so exploration is replay-simulated — the true online lift can only come from a live A/B test.

```bash
# Explore-aware recommendations (returns dlrm_score, exploration_bonus, is_exploratory per item)
curl "http://localhost:8000/api/v1/recommend_v3/1?top_k=5"

# Feed an observed reward back into the bandit (online update + persist)
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "item_id": 50, "reward": 1.0}'
```

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
  -d '{"continuous_features": [0.76, 0.5, 0.3, 0.4, 0.72, 0.6, 0.8, 0.52], "categorical_features": [1, 2]}'
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
  "num_parameters": 372097,
  "device": "cpu",
  "num_continuous_features": 8,
  "num_categorical_features": 2,
  "mlp_layers": [128, 64, 32]
}
```

### `POST /api/v1/predict`

The main endpoint. Send user preferences, get back 5 movie recommendations.

**Request:**

```json
{
  "continuous_features": [0.76, 0.5, 0.3, 0.4, 0.72, 0.6, 0.8, 0.52],
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

### `GET /api/v1/recommend/{user_id}`

Personalized top-K recommendations for a known MovieLens user. Scores every item in one forward pass and returns the highest-scoring movies with metadata.

**Query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `top_k` | `10` | Number of recommendations to return |

**Example:**

```bash
curl http://localhost:8000/api/v1/recommend/1?top_k=5
```

**Response** (200):

```json
{
  "user_id": 1,
  "recommendations": [
    {
      "item_id": 50,
      "score": 0.9312,
      "title": "Star Wars (1977)",
      "genres": ["Action", "Adventure", "Romance", "Sci-Fi", "War"]
    }
  ]
}
```

**Error responses:**

| Status | When |
|---|---|
| 200 | Unknown user — returns popularity-based fallback with `"note": "cold-start"` |
| 404 | Unknown user AND no popular items available (rare) |
| 503 | Model or serving context not loaded |

### `GET /api/v1/recommend_v3/{user_id}`

Same two-stage retrieval as `/recommend_v2`, but reranked by the common-scale blend `(1-w)·norm(dlrm_score) + w·norm(exploration_bonus)` with default `w=0.2` (see [Exploration layer](#exploration-layer-contextual-bandit)). Known users get the ANN + DLRM + bandit path; unknown users get the same blend applied over popularity candidates. `score` is the blended `[0, 1]` value; `dlrm_score` and `exploration_bonus` are the raw pre-blend terms.

**Query parameters:** `top_k` (default `10`), `num_candidates` (default `100`).

**Example:**

```bash
curl "http://localhost:8000/api/v1/recommend_v3/1?top_k=5"
```

**Response** (200):

```json
{
  "user_id": 1,
  "recommendations": [
    {
      "item_id": 50,
      "score": 0.8123,
      "dlrm_score": 0.9312,
      "exploration_bonus": 0.352901,
      "is_exploratory": true,
      "title": "Star Wars (1977)",
      "genres": ["Action", "Adventure", "Sci-Fi"]
    }
  ],
  "pipeline": "three-stage (ANN + DLRM rerank + bandit exploration)",
  "policy": "combine_explore(w=0.2): (1-w)*norm(DLRM) + w*norm(LinUCB alpha=1.0 bonus)",
  "explore_weight": 0.2,
  "retrieval": "item-centroid",
  "num_candidates": 100,
  "latency_ms": { "stage1_ann": 1.0, "stage2_rerank": 0.6, "stage3_bandit": 0.7, "total": 2.3 },
  "ann_backend": "faiss-cpu"
}
```

Cold-start users receive `"pipeline": "bandit-reranked (cold-start)"` with `"note": "cold-start: bandit-explored popularity path"`.

### `POST /api/v1/feedback`

Online-learning update. Applies an observed `(user_id, item_id, reward)` to the bandit and persists the new state to `bandit_state.pkl`.

**Request:**

```json
{ "user_id": 1, "item_id": 50, "reward": 1.0 }
```

**Response** (200):

```json
{
  "success": true,
  "user_id": 1,
  "item_id": 50,
  "reward": 1.0,
  "updates_applied": 1,
  "exploit_mean": 0.4821,
  "exploration_bonus": 0.3410,
  "persisted": true
}
```

**Error responses:**

| Status | When |
|---|---|
| 400 | `reward` outside `[0, 1]`, or unknown `item_id` |
| 503 | Model, serving context, or bandit not loaded |

### `POST /predict/`

Legacy endpoint. Delegates to `/api/v1/predict` — same request format, same response.

---

## Training the Model

The training pipeline uses PyTorch with early stopping, learning rate scheduling, and W&B logging.

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
| `cardinalities` | `[10, 10, 10, 10, 10]` | Vocabulary size (cardinality) for each categorical embedding table |
| `embedding_dim` | `64` | Dimension of every embedding vector (decoupled from the MLP) |
| `mlp_layers` | `[128, 64, 32]` | Hidden layer dimensions for the interaction MLP |
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
│   └── app.py                      # FastAPI app — endpoints, middleware, TMDB client, error handling
├── models/
│   ├── __init__.py
│   ├── dlrm.py                     # DLRM model — embeddings, interaction, MLP, forward pass
│   └── classical.py                # Classical baselines (XGBoost, LightGBM, LogReg)
├── data/
│   ├── preprocessing.py            # Shared preprocessing — PrepConfig, validation, feature engineering
│   └── ml-100k/                    # MovieLens 100K dataset
├── scripts/
│   ├── train_movielens.py          # Train DLRM on MovieLens with W&B logging
│   ├── retrain_and_compare.py      # Train DLRM + 3 classical baselines, generate comparison report
│   ├── run_experiment.py           # A/B experiment runner (Welch's t-test, Cohen's d, power analysis)
│   ├── experiment_framework.py     # A/B testing framework — statistical comparison engine
│   ├── baseline_comparison.py      # Heuristic baselines (random, popular, user-mean) vs DLRM
│   ├── benchmark_inference.py      # PyTorch vs ONNX inference latency benchmarks
│   ├── drift_detection.py          # Data drift detection between train/test distributions
│   ├── fairness_analysis.py        # Fairness audit — popularity bias, user activity gap, diversity
│   ├── export_onnx.py              # Export DLRM to ONNX format
│   └── train.py                    # Demo training script with synthetic data
├── tests/
│   ├── test_api_integration.py     # 31 API endpoint tests (predict, recommend, errors, health)
│   ├── test_model.py               # 20 DLRM unit tests + API mock tests
│   ├── test_upgrades.py            # 10 tests for PrepConfig + interaction feature
│   └── test_validation.py          # 11 tests for data validation and split checks
├── reports/
│   └── model_comparison.json       # Benchmark results (DLRM vs XGBoost vs LightGBM vs LogReg)
├── configs/
│   └── config.yaml                 # Hyperparameters — features, layers, learning rate, epochs
├── deploy/
│   ├── task-definition.json        # AWS ECS Fargate task definition
│   └── trust-policy.json           # AWS IAM trust policy for ECS execution role
├── Dockerfile                      # Production Dockerfile (Python 3.11, uv, health checks)
├── Makefile                        # Dev shortcuts — install, run, test, lint, docker, clean
├── pyproject.toml                  # Project metadata, dependencies, tool config (ruff, pytest)
├── EVALUATION.md                   # Full ablation studies and reproduction instructions
├── .env.example                    # Template for required environment variables
└── README.md                       # You are here
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

The repo includes an ECS task definition (`task-definition.json`) configured for Fargate with 256 CPU units and 512MB memory. The container image is pushed to ECR at `YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cinemascope-recsys:latest`.

To deploy to ECS:

```bash
# Build and push to ECR
docker build -t cinemascope-recsys .
docker tag cinemascope-recsys:latest YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cinemascope-recsys:latest
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker push YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cinemascope-recsys:latest

# Register task and run
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs update-service --cluster your-cluster --service your-service --force-new-deployment
```

---

## What I'd Improve Next

- **Sequential model** — add SASRec or BERT4Rec to capture temporal user behavior (what the user watched recently, not just aggregate stats)
- **Genre features in DLRM** — genre data is already loaded but not used in training. Adding content features would improve cold-start for items.
- **Negative sampling** — explicit hard-negative mining during training to improve ranking quality
- **Feature store** — replace the pickle-based serving context with a feature store (Feast/Redis) for fresh user features
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

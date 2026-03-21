# RecommenderSystem

A movie recommendation API. You give it some preferences, it runs them through a neural network (DLRM), grabs real movie data from TMDB, and gives you back 5 movies with posters and everything. It's the backend that powers [CinemaScopeAI](https://github.com/AkinCodes/CinemaScopeAI).

## How it works

```
You send preferences (numbers describing your taste)
        |
        v
   DLRM model scores movies (PyTorch)
        |
        v
   Fetches real movie data from TMDB
        |
        v
   Returns 5 recommendations
   (titles, posters, genres, scores)
```

## What's inside

FastAPI for the API. PyTorch for the model. TMDB for real movie data. Packaged with Docker, deployed on Render.

## Live demo

The API is running at **https://cinemascope-api.onrender.com** (free tier, so first request might take a few seconds to wake up).

```bash
curl -X POST https://cinemascope-api.onrender.com/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"continuous_features": [0.5, 0.8], "categorical_features": [1, 2]}'
```

## Getting started

```bash
uv sync --all-extras
cp .env.example .env   # add your TMDB keys
make run               # starts at localhost:8000
```

## API endpoints

| Method | Path | What it does |
|--------|------|-------------|
| GET | `/` | Status check |
| GET | `/health` | Health check + model status |
| GET | `/api/v1/models` | Model architecture info |
| POST | `/api/v1/predict` | Get 5 movie recommendations |
| POST | `/predict/` | Legacy predict (redirects to v1) |

## Makefile shortcuts

| Command | What it does |
|---------|-------------|
| `make install` | Install dependencies |
| `make run` | Start dev server (hot reload) |
| `make test` | Run tests |
| `make lint` | Lint with Ruff |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run container |
| `make clean` | Remove caches |

## Training

The DLRM model trains on movie rating data. Hyperparameters live in `configs/config.yaml`, and checkpoints get saved to `lightning_logs/checkpoints/`.

```bash
uv run python scripts/train.py
```

## Environment variables

| Variable | What it is |
|----------|-----------|
| `TMDB_API_KEY` | TMDB v3 API key |
| `TMDB_BEARER_TOKEN` | TMDB v4 read-access token |

Get both at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api).

## Project structure

```
api/app.py            FastAPI app and endpoints
models/dlrm.py        DLRM model (PyTorch)
scripts/train.py      Training loop (PyTorch Lightning)
scripts/inference.py  Standalone inference
configs/config.yaml   Hyperparameters
tests/test_model.py   Tests
```

## Other projects

- **[CinemaScopeAI](https://github.com/AkinCodes/CinemaScopeAI)** — the iOS app that talks to this API
- **[MoviePosterAI](https://github.com/AkinCodes/MoviePosterAI)** — poster analysis

## Author

Akin Olusanya

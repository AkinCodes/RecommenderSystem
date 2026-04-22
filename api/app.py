"""CinemaScopeAI Recommender API.

Provides movie recommendation endpoints powered by a DLRM model and the TMDB API.
"""

import logging
import os
import pickle
import time

import httpx
import torch
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from data.preprocessing import load_item_metadata
from models.dlrm import DLRMModel

# Configuration

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")

if not TMDB_API_KEY or not TMDB_BEARER_TOKEN:
    logger.warning(
        "TMDB_API_KEY or TMDB_BEARER_TOKEN not set. "
        "Movie fetching will fail until they are configured."
    )


class TMDBError(Exception):
    """Raised when the TMDB API returns an error."""


class ModelInputError(Exception):
    """Raised when prediction input is invalid."""


# Load and validate config from YAML

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")

REQUIRED_CONFIG_KEYS = {
    "model": ["num_features", "embedding_sizes", "mlp_layers", "learning_rate", "epochs", "batch_size"],
}


def load_and_validate_config(path: str) -> dict:
    """Load YAML config and validate that all required keys are present."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError("Configuration file is empty.")

    for section, keys in REQUIRED_CONFIG_KEYS.items():
        if section not in cfg:
            raise ValueError(f"Missing required config section: '{section}'")
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"Missing required config key: '{section}.{key}'")

    return cfg


try:
    config = load_and_validate_config(CONFIG_PATH)
    model_cfg = config["model"]
    logger.info("Configuration loaded and validated successfully.")
    logger.info(
        "Effective config: model.num_features=%s, model.mlp_layers=%s, "
        "model.epochs=%s, model.batch_size=%s",
        model_cfg["num_features"],
        model_cfg["mlp_layers"],
        model_cfg["epochs"],
        model_cfg["batch_size"],
    )
except (FileNotFoundError, ValueError) as exc:
    logger.error("Configuration error: %s", exc)
    raise SystemExit(1) from exc

# Request / response schemas


class PredictionRequest(BaseModel):
    continuous_features: list[float]
    categorical_features: list[int]


class RecommendationResponse(BaseModel):
    title: str
    genre: str
    rating: str
    score: float
    poster_url: str
    director: str
    release_year: int
    summary: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    architecture: str
    num_parameters: int
    device: str
    num_continuous_features: int
    num_categorical_features: int
    mlp_layers: list[int]


# Model loading

num_continuous_features = model_cfg["num_features"]  # 2: mean_rating, normalised_count
num_categorical_features = len(model_cfg["embedding_sizes"])  # 2: user_id, item_id
embedding_sizes = model_cfg["embedding_sizes"]  # [943, 1682]

model_loaded = False
model: DLRMModel | None = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_model_movielens.pth")

try:
    model = DLRMModel(
        num_features=num_continuous_features,
        embedding_sizes=embedding_sizes,
        mlp_layers=model_cfg["mlp_layers"],
    )

    if not os.path.exists(MODEL_PATH):
        logger.error(
            "Model file not found at '%s'. The /health endpoint will report "
            "model_loaded=false. Predictions will fail.",
            MODEL_PATH,
        )
    else:
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        model_loaded = True
        logger.info("DLRM model loaded successfully from '%s'.", MODEL_PATH)
except Exception as exc:
    logger.error("Failed to load DLRM model: %s", exc, exc_info=True)
    model = None

# Serving context (user/item mappings + per-user features)

SERVING_CONTEXT_PATH = os.path.join(os.path.dirname(__file__), "..", "serving_context.pkl")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k")

serving_context: dict | None = None
item_metadata: dict = {}

if os.path.exists(SERVING_CONTEXT_PATH):
    try:
        with open(SERVING_CONTEXT_PATH, "rb") as _f:
            serving_context = pickle.load(_f)
        logger.info(
            "Serving context loaded: %d users, %d items.",
            len(serving_context["user2idx"]),
            len(serving_context["item2idx"]),
        )
    except (pickle.UnpicklingError, EOFError, Exception) as e:
        logger.error("Failed to load serving context: %s", e)
        serving_context = None
else:
    logger.warning("Serving context not found at '%s'. /recommend endpoint will be unavailable.", SERVING_CONTEXT_PATH)

if os.path.exists(os.path.join(DATA_DIR, "u.item")):
    item_metadata = load_item_metadata(DATA_DIR)
    logger.info("Item metadata loaded: %d movies.", len(item_metadata))
else:
    logger.warning("Item metadata file not found at '%s/u.item'.", DATA_DIR)

# Cold-start fallback: pre-compute popular items from item metadata
popular_items: list[dict] = []
if item_metadata:
    popular_ids = sorted(item_metadata.keys())[:50]
    popular_items = [
        {
            "item_id": int(iid), "score": 0.0,
            "title": item_metadata[iid]["title"],
            "genres": item_metadata[iid]["genres"],
        }
        for iid in popular_ids
    ]
    logger.info("Cold-start fallback ready: %d popular items.", len(popular_items))

# App

app = FastAPI(title="CinemaScopeAI Recommender", version=APP_VERSION)


# Middleware: request/response logging


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status code and duration for every request."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "method=%s path=%s status_code=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# Structured error handling


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    """Return structured JSON error responses for all HTTPExceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": f"HTTP {exc.status_code}",
            "detail": str(exc.detail),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    """Catch-all for unhandled exceptions -- return a structured 500."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred.",
        },
    )


# Root endpoint


@app.get("/")
async def root():
    """Basic root endpoint."""
    return {"status": "healthy", "message": "Recommendation API is running."}


# Health check


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health-check endpoint with model status."""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "version": APP_VERSION,
    }


# Model info


@app.get("/api/v1/models", response_model=ModelInfoResponse)
async def model_info():
    """Return information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    total_params = sum(p.numel() for p in model.parameters())
    device = str(next(model.parameters()).device) if model_loaded else "N/A"

    return {
        "architecture": "DLRM (Deep Learning Recommendation Model)",
        "num_parameters": total_params,
        "device": device,
        "num_continuous_features": num_continuous_features,
        "num_categorical_features": num_categorical_features,
        "mlp_layers": model_cfg["mlp_layers"],
    }


# TMDB helpers


async def fetch_real_movies() -> list[dict]:
    """Fetch popular movies from the TMDB API (async / non-blocking)."""
    if not TMDB_BEARER_TOKEN:
        raise TMDBError("TMDB_BEARER_TOKEN is not configured.")

    url = "https://api.themoviedb.org/3/movie/popular?language=en-US&page=1"
    headers = {
        "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
        "accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, headers=headers)

    if response.status_code != 200:
        logger.error(
            "TMDB API request failed: status=%s body=%s",
            response.status_code,
            response.text[:300],
        )
        raise TMDBError(f"TMDB API returned status {response.status_code}")

    movies = response.json()["results"]
    return [
        {
            "title": movie["title"],
            "genre": "Unknown",
            "rating": "N/A",
            "score": movie["vote_average"] / 10,
            "poster_url": (
                f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                if movie.get("poster_path")
                else "https://via.placeholder.com/500"
            ),
            "director": "N/A",
            "release_year": (
                int(movie["release_date"].split("-")[0])
                if movie.get("release_date")
                else 0
            ),
            "summary": movie.get("overview", ""),
        }
        for movie in movies
    ]


# Predict (versioned)


@app.post("/api/v1/predict", response_model=list[RecommendationResponse])
async def predict(request: PredictionRequest):
    """Return the top-5 movie recommendations based on the DLRM prediction score."""
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if len(request.categorical_features) != num_categorical_features:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {num_categorical_features} categorical features "
                f"(user_id 0-{embedding_sizes[0] - 1}, item_id 0-{embedding_sizes[1] - 1}), "
                f"got {len(request.categorical_features)}."
            ),
        )

    # Validate categorical feature ranges
    for i, (val, max_val) in enumerate(
        zip(request.categorical_features, embedding_sizes)
    ):
        if val < 0 or val >= max_val:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Categorical feature {i} out of range: got {val}, "
                    f"expected 0-{max_val - 1}."
                ),
            )

    if len(request.continuous_features) != num_continuous_features:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {num_continuous_features} continuous features "
                f"(mean_rating, normalised_count), "
                f"got {len(request.continuous_features)}."
            ),
        )

    continuous_tensor = torch.tensor(
        [request.continuous_features], dtype=torch.float32
    )
    categorical_tensor = torch.tensor(
        [request.categorical_features], dtype=torch.int64
    )

    prediction_score = model(continuous_tensor, categorical_tensor).item()
    logger.info("Prediction score: %.4f", prediction_score)

    try:
        movie_database = await fetch_real_movies()
    except TMDBError as exc:
        logger.error("Failed to fetch movies: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))

    if not movie_database:
        raise HTTPException(status_code=502, detail="TMDB returned no movies.")

    recommended_movies = sorted(
        movie_database,
        key=lambda x: abs(x["score"] - prediction_score),
        reverse=True,
    )

    return recommended_movies[:5]


# Personalized recommendations


@app.get("/api/v1/recommend/{user_id}")
async def recommend(user_id: int, top_k: int = 10):
    """Return top-K personalized recommendations for a known user."""
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if serving_context is None:
        raise HTTPException(status_code=503, detail="Serving context is not loaded.")

    if user_id not in serving_context["user2idx"]:
        if popular_items:
            return {
                "user_id": user_id,
                "recommendations": popular_items[:top_k],
                "note": "cold-start: popularity-based fallback",
            }
        raise HTTPException(status_code=404, detail=f"Unknown user: {user_id}")

    user_idx = serving_context["user2idx"][user_id]
    user_cont = serving_context["user_features"][user_idx]
    num_items = len(serving_context["item2idx"])

    # Build batch: same user features for every item
    cont = torch.tensor([user_cont] * num_items, dtype=torch.float32)
    cat = torch.tensor([[user_idx, i] for i in range(num_items)], dtype=torch.int64)

    with torch.no_grad():
        scores = model(cont, cat).squeeze()

    top_indices = scores.topk(top_k).indices.tolist()

    results = []
    for idx in top_indices:
        raw_item_id = int(serving_context["idx2item"][idx])
        meta = item_metadata.get(raw_item_id, {"title": f"Item {raw_item_id}", "genres": []})
        results.append({
            "item_id": raw_item_id,
            "score": round(scores[idx].item(), 4),
            "title": meta["title"],
            "genres": meta["genres"],
        })

    return {"user_id": user_id, "recommendations": results}


# Keep old endpoint for backwards compatibility (redirects to versioned)
@app.post("/predict/", response_model=list[RecommendationResponse])
async def predict_legacy(request: PredictionRequest):
    """Legacy predict endpoint -- delegates to the versioned endpoint."""
    return await predict(request)

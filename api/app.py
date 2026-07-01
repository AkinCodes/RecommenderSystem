"""CinemaScopeAI Recommender API."""

import logging
import os
import pickle
import threading
import time

import httpx
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from data.preprocessing import load_item_metadata, load_movielens_data
from models.bandit import (
    LinUCBBandit,
    build_interaction_context,
    combine_explore,
    load_state,
    save_state,
)
from models.dlrm import DLRMModel

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    from sklearn.neighbors import NearestNeighbors

    FAISS_AVAILABLE = False

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


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")

REQUIRED_CONFIG_KEYS = {
    "model": ["num_features", "cardinalities", "mlp_layers", "learning_rate", "epochs", "batch_size"],
}


def load_and_validate_config(path: str) -> dict:
    """Load YAML config and validate required keys."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError("Configuration file is empty.")

    model_cfg = cfg.get("model", {})
    if "embedding_sizes" in model_cfg:
        raise ValueError(
            "'embedding_sizes' was renamed to 'cardinalities'. Update your config "
            "(and add 'embedding_dim', default 64)."
        )

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


class FeedbackRequest(BaseModel):
    user_id: int
    item_id: int
    reward: float


class FeedbackResponse(BaseModel):
    success: bool = True
    user_id: int
    item_id: int
    reward: float
    updates_applied: int
    exploit_mean: float
    exploration_bonus: float
    persisted: bool


num_continuous_features = model_cfg["num_features"]
num_categorical_features = len(model_cfg["cardinalities"])
cardinalities = model_cfg["cardinalities"]

model_loaded = False
model: DLRMModel | None = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_model_movielens.pth")

try:
    model = DLRMModel(
        num_features=num_continuous_features,
        cardinalities=cardinalities,
        embedding_dim=model_cfg.get("embedding_dim", 64),
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

# Training-set popularity + per-user liked-item history (train split only).
# Loaded once at startup; powers a popularity-ranked cold-start fallback and
# in-space (item-embedding) retrieval in /recommend_v2.
item_train_counts: dict[int, int] = {}
user_history: dict[int, list[int]] = {}
_data_file = os.path.join(DATA_DIR, "u.data")
if serving_context is not None and os.path.exists(_data_file):
    try:
        _raw = load_movielens_data(_data_file)
        _sorted = _raw[np.argsort(_raw[:, 3])]            # sort by timestamp
        _train = _sorted[: int(len(_sorted) * 0.8)]       # same 80% split used in training
        _u2i = serving_context["user2idx"]
        _i2i = serving_context["item2idx"]
        for _row in _train:
            _uid, _iid, _rating = int(_row[0]), int(_row[1]), float(_row[2])
            item_train_counts[_iid] = item_train_counts.get(_iid, 0) + 1
            if _rating >= 4 and _uid in _u2i and _iid in _i2i:
                user_history.setdefault(_u2i[_uid], []).append(_i2i[_iid])
        logger.info(
            "Train popularity (%d items) and user history (%d users) loaded.",
            len(item_train_counts), len(user_history),
        )
    except Exception as exc:
        logger.error("Failed to build popularity/history: %s", exc, exc_info=True)

popular_items: list[dict] = []
if item_metadata:
    # Rank cold-start items by ACTUAL training interaction count (popularity),
    # not by item id. Items unseen in training sort last (count 0).
    popular_ids = sorted(
        item_metadata.keys(),
        key=lambda _iid: item_train_counts.get(int(_iid), 0),
        reverse=True,
    )[:50]
    popular_items = [
        {
            "item_id": int(iid),
            "score": float(item_train_counts.get(int(iid), 0)),
            "title": item_metadata[iid]["title"],
            "genres": item_metadata[iid]["genres"],
        }
        for iid in popular_ids
    ]
    logger.info(
        "Cold-start fallback ready: %d popular items (ranked by train count).",
        len(popular_items),
    )

# Faiss ANN for sub-millisecond candidate generation
faiss_index = None
item_embeddings_np: np.ndarray | None = None

if model_loaded and model is not None:
    try:
        item_emb_weights = model.embeddings[1].weight.detach().numpy().astype(np.float32)
        # L2-normalise so inner-product search == cosine similarity
        norms = np.linalg.norm(item_emb_weights, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        item_embeddings_np = item_emb_weights / norms

        if FAISS_AVAILABLE:
            faiss_index = faiss.IndexFlatIP(item_embeddings_np.shape[1])
            faiss_index.add(item_embeddings_np)
            logger.info(
                "Faiss ANN index built: %d items, dim=%d (using faiss-cpu).",
                faiss_index.ntotal,
                item_embeddings_np.shape[1],
            )
        else:
            faiss_index = NearestNeighbors(
                n_neighbors=100, metric="cosine", algorithm="brute"
            )
            faiss_index.fit(item_embeddings_np)
            logger.info(
                "Sklearn ANN index built: %d items, dim=%d (faiss-cpu unavailable, using sklearn fallback).",
                item_embeddings_np.shape[0],
                item_embeddings_np.shape[1],
            )
    except Exception as exc:
        logger.error("Failed to build ANN index: %s", exc, exc_info=True)
        faiss_index = None
        item_embeddings_np = None

# --- Contextual bandit exploration layer -----------------------------------
# A shared-weight LinUCB bandit adds an exploration bonus on top of the DLRM
# exploit score (see /recommend_v3) and learns online from /feedback. The
# context is the elementwise user x item embedding product -- the SAME signal
# the DLRM's interaction term uses -- so the bandit "sees" what the ranker sees.
BANDIT_DIM = model_cfg.get("embedding_dim", 64)
BANDIT_ALPHA = 1.0
# Explore weight: the single user-facing exploit<->explore knob for combine_explore.
# 0.0 = pure DLRM exploit, 1.0 = pure exploration. Default 0.2 was chosen from the offline
# explore-weight frontier sweep (scripts/offline_bandit_eval.py): it keeps ~all of the DLRM
# SNIPS reward while adding a modest catalog-coverage gain. alpha shapes the bonus; w trades it.
BANDIT_EXPLORE_WEIGHT = 0.2
BANDIT_STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "bandit_state.pkl")

# Serializes the read-modify-write (update + atomic persist) of the bandit in /feedback so
# concurrent feedback requests within a single worker cannot race the in-memory A / A_inv or the
# pickle. NOTE: this only protects a SINGLE worker; multiple uvicorn workers each hold their own
# in-memory bandit and would still race the on-disk state (see EVALUATION.md).
bandit_lock = threading.Lock()

bandit: LinUCBBandit | None = None
try:
    if os.path.exists(BANDIT_STATE_PATH):
        bandit = load_state(BANDIT_STATE_PATH)
        logger.info(
            "Bandit state loaded from '%s' (dim=%d, alpha=%.2f).",
            BANDIT_STATE_PATH, bandit.dim, bandit.alpha,
        )
    else:
        bandit = LinUCBBandit(dim=BANDIT_DIM, alpha=BANDIT_ALPHA)
        logger.info(
            "Bandit state not found at '%s'. Initialised fresh LinUCB (dim=%d, alpha=%.2f).",
            BANDIT_STATE_PATH, BANDIT_DIM, BANDIT_ALPHA,
        )
except Exception as exc:
    logger.error("Failed to initialise bandit: %s", exc, exc_info=True)
    bandit = None

# Mean user embedding -- the default context anchor for the bandit cold-start
# path (unknown users have no personalised embedding). Computed once at startup.
mean_user_emb_np: np.ndarray | None = None
if model_loaded and model is not None:
    try:
        mean_user_emb_np = (
            model.embeddings[0].weight.detach().numpy().astype(np.float32).mean(axis=0)
        )
    except Exception as exc:
        logger.error("Failed to compute mean user embedding: %s", exc, exc_info=True)
        mean_user_emb_np = None


app = FastAPI(title="CinemaScopeAI Recommender", version=APP_VERSION)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status and duration for every request."""
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


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    """Return structured JSON for HTTPExceptions."""
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
    """Catch-all handler returning a structured 500."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred.",
        },
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {"status": "healthy", "message": "Recommendation API is running."}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with model status."""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "version": APP_VERSION,
    }


@app.get("/api/v1/models", response_model=ModelInfoResponse)
async def model_info():
    """Return loaded model metadata."""
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


async def fetch_real_movies() -> list[dict]:
    """Fetch popular movies from the TMDB API."""
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


@app.post("/api/v1/predict", response_model=list[RecommendationResponse])
async def predict(request: PredictionRequest):
    """Return top-5 movie recommendations from DLRM score."""
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if len(request.categorical_features) != num_categorical_features:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {num_categorical_features} categorical features "
                f"(user_id 0-{cardinalities[0] - 1}, item_id 0-{cardinalities[1] - 1}), "
                f"got {len(request.categorical_features)}."
            ),
        )

    for i, (val, max_val) in enumerate(
        zip(request.categorical_features, cardinalities)
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


def _build_user_item_features(user_feats, item_indices, item_features, n_features):
    """Build (N, n_features) tensor combining user, item, and interaction features."""
    n = len(item_indices)
    cont_np = np.zeros((n, n_features), dtype=np.float32)

    cont_np[:, 0:4] = user_feats[0:4]

    item_mat = np.zeros((n, 3), dtype=np.float32)
    for i, item_idx in enumerate(item_indices):
        item_mat[i] = item_features.get(item_idx, [0.6, 0.0, 0.0])
    cont_np[:, 4:7] = item_mat

    cont_np[:, 7] = (user_feats[0] - item_mat[:, 0] + 1.0) / 2.0

    return torch.tensor(cont_np, dtype=torch.float32)


@app.get("/api/v1/recommend/{user_id}")
async def recommend(user_id: int, top_k: int = 10):
    """Return top-K personalized recommendations for a user."""
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
    user_feats = serving_context["user_features"][user_idx]
    item_features = serving_context.get("item_features", {})
    num_items = len(serving_context["item2idx"])

    cont = _build_user_item_features(user_feats, list(range(num_items)), item_features, num_continuous_features)
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


@app.get("/api/v1/recommend_v2/{user_id}")
async def recommend_v2(user_id: int, top_k: int = 10, num_candidates: int = 100):
    """Two-stage recommendation: ANN candidate generation + DLRM reranking."""
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if serving_context is None:
        raise HTTPException(status_code=503, detail="Serving context is not loaded.")

    if faiss_index is None or item_embeddings_np is None:
        raise HTTPException(status_code=503, detail="ANN index is not available.")

    if user_id not in serving_context["user2idx"]:
        if popular_items:
            return {
                "user_id": user_id,
                "recommendations": popular_items[:top_k],
                "note": "cold-start: popularity-based fallback",
                "pipeline": "two-stage (skipped: cold-start)",
            }
        raise HTTPException(status_code=404, detail=f"Unknown user: {user_id}")

    user_idx = serving_context["user2idx"][user_id]
    user_feats = serving_context["user_features"][user_idx]
    item_features = serving_context.get("item_features", {})
    num_items = len(serving_context["item2idx"])

    num_candidates = min(num_candidates, num_items)

    stage1_start = time.perf_counter()

    # Retrieve from the item-embedding index using the centroid of the user's
    # liked items, so the query and the index share the SAME trained space.
    # The DLRM trains user/item embeddings through an MLP interaction, not a
    # shared dot-product space, so querying with the raw user embedding is only
    # a heuristic; we keep it as a fallback for users with no liked history.
    liked = user_history.get(user_idx, [])
    if liked and item_embeddings_np is not None:
        query_vec = item_embeddings_np[liked].mean(axis=0, keepdims=True).astype(np.float32)
        retrieval_method = "item-centroid"
    else:
        query_vec = model.embeddings[0](torch.tensor([user_idx])).detach().numpy().astype(np.float32)
        retrieval_method = "user-embedding-fallback"
    query_vec = (query_vec / (np.linalg.norm(query_vec) or 1.0)).astype(np.float32)

    if FAISS_AVAILABLE:
        _, candidate_indices = faiss_index.search(query_vec, num_candidates)
        candidate_indices = candidate_indices[0].tolist()
    else:
        distances, indices = faiss_index.kneighbors(query_vec, n_neighbors=num_candidates)
        candidate_indices = indices[0].tolist()

    stage1_ms = (time.perf_counter() - stage1_start) * 1000

    stage2_start = time.perf_counter()

    cont = _build_user_item_features(user_feats, candidate_indices, item_features, num_continuous_features)
    cat = torch.tensor([[user_idx, idx] for idx in candidate_indices], dtype=torch.int64)

    with torch.no_grad():
        scores = model(cont, cat).squeeze()

    # Handle single-candidate edge case (squeeze collapses to scalar)
    if scores.dim() == 0:
        scores = scores.unsqueeze(0)

    top_within_candidates = scores.topk(min(top_k, len(candidate_indices))).indices.tolist()

    stage2_ms = (time.perf_counter() - stage2_start) * 1000

    brute_start = time.perf_counter()
    cont_all = _build_user_item_features(user_feats, list(range(num_items)), item_features, num_continuous_features)
    cat_all = torch.tensor([[user_idx, i] for i in range(num_items)], dtype=torch.int64)
    with torch.no_grad():
        _ = model(cont_all, cat_all).squeeze()
    brute_ms = (time.perf_counter() - brute_start) * 1000

    results = []
    for rank_idx in top_within_candidates:
        item_idx = candidate_indices[rank_idx]
        raw_item_id = int(serving_context["idx2item"][item_idx])
        meta = item_metadata.get(raw_item_id, {"title": f"Item {raw_item_id}", "genres": []})
        results.append({
            "item_id": raw_item_id,
            "score": round(scores[rank_idx].item(), 4),
            "title": meta["title"],
            "genres": meta["genres"],
        })

    return {
        "user_id": user_id,
        "recommendations": results,
        "pipeline": "two-stage",
        "retrieval": retrieval_method,
        "num_candidates": num_candidates,
        "latency_ms": {
            "stage1_ann": round(stage1_ms, 2),
            "stage2_rerank": round(stage2_ms, 2),
            "total_two_stage": round(stage1_ms + stage2_ms, 2),
            "brute_force_all_items": round(brute_ms, 2),
            "speedup": round(brute_ms / max(stage1_ms + stage2_ms, 0.01), 2),
        },
        "ann_backend": "faiss-cpu" if FAISS_AVAILABLE else "sklearn-NearestNeighbors",
    }


def _raw_user_emb(user_idx: int) -> np.ndarray:
    """Return the DLRM's raw (un-normalised) user embedding, shape (dim,)."""
    return model.embeddings[0](torch.tensor([user_idx])).detach().numpy()[0].astype(np.float64)


def _raw_item_embs(item_indices: list[int]) -> np.ndarray:
    """Return raw (un-normalised) item embeddings for the given indices, shape (n, dim)."""
    idx = torch.tensor(item_indices, dtype=torch.int64)
    return model.embeddings[1](idx).detach().numpy().astype(np.float64)


@app.get("/api/v1/recommend_v3/{user_id}")
async def recommend_v3(user_id: int, top_k: int = 10, num_candidates: int = 100):
    """Two-stage retrieval (like v2) reranked by a common-scale blend of DLRM exploit + bonus.

    Final rank key = ``combine_explore(dlrm_score, exploration_bonus, w)``: both terms are min-max
    normalized across the candidate set to [0, 1] and convex-combined as
    ``(1 - w) * norm(dlrm) + w * norm(bonus)``. This is the fix for the scale mismatch -- the DLRM
    sigmoid (~0.9-1.0) and the LinUCB bonus (mean ~3.3) are not comparable raw, so a plain sum
    ranked by the bonus alone. The context is the elementwise user x item embedding product (the
    same interaction signal the DLRM uses). ``is_exploratory`` marks items whose rank changed
    versus pure-DLRM (w=0) order because of the blend.
    """
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if serving_context is None:
        raise HTTPException(status_code=503, detail="Serving context is not loaded.")

    if faiss_index is None or item_embeddings_np is None:
        raise HTTPException(status_code=503, detail="ANN index is not available.")

    if bandit is None:
        raise HTTPException(status_code=503, detail="Bandit exploration layer is not available.")

    # Cold-start: rank the popularity candidates with the bandit using a default
    # context (mean user embedding x each item embedding), so even unknown users
    # get an exploration mechanism instead of the static popularity list.
    if user_id not in serving_context["user2idx"]:
        if popular_items and mean_user_emb_np is not None:
            cold_item_indices = [
                serving_context["item2idx"][int(it["item_id"])]
                for it in popular_items
                if int(it["item_id"]) in serving_context["item2idx"]
            ]
            cold_pool = popular_items[: len(cold_item_indices)][:num_candidates]
            cold_item_indices = cold_item_indices[:num_candidates]

            item_embs = _raw_item_embs(cold_item_indices)
            contexts = build_interaction_context(mean_user_emb_np.astype(np.float64), item_embs)
            bonus = bandit.exploration_bonus(contexts)
            # Base = popularity score (already in each popular_items entry). Popularity counts
            # (~hundreds) and the bonus (~0.1) are on wildly different scales, so we blend them on
            # a common [0, 1] scale via combine_explore rather than summing raw values (which would
            # be dominated by the counts, reproducing the static popularity order exactly).
            pop_scores = np.array([it["score"] for it in cold_pool], dtype=np.float64)
            combined = combine_explore(pop_scores, bonus, BANDIT_EXPLORE_WEIGHT)
            pop_order = np.argsort(pop_scores)[::-1]
            combined_order = np.argsort(combined)[::-1]
            pop_rank = {int(i): r for r, i in enumerate(pop_order.tolist())}

            results = []
            for new_rank, j in enumerate(combined_order[:top_k].tolist()):
                it = cold_pool[j]
                results.append({
                    "item_id": it["item_id"],
                    "score": round(float(combined[j]), 4),
                    "dlrm_score": round(float(pop_scores[j]), 4),
                    "exploration_bonus": round(float(bonus[j]), 6),
                    "is_exploratory": pop_rank[int(j)] != new_rank,
                    "title": it["title"],
                    "genres": it["genres"],
                })
            return {
                "user_id": user_id,
                "recommendations": results,
                "pipeline": "bandit-reranked (cold-start)",
                "policy": (
                    f"combine_explore(w={BANDIT_EXPLORE_WEIGHT}): "
                    "(1-w)*norm(popularity) + w*norm(LinUCB bonus) over popularity candidates"
                ),
                "explore_weight": BANDIT_EXPLORE_WEIGHT,
                "cold_start_context": "mean user embedding x item embedding",
                "note": "cold-start: bandit-explored popularity path",
            }
        raise HTTPException(status_code=404, detail=f"Unknown user: {user_id}")

    user_idx = serving_context["user2idx"][user_id]
    user_feats = serving_context["user_features"][user_idx]
    item_features = serving_context.get("item_features", {})
    num_items = len(serving_context["item2idx"])

    num_candidates = min(num_candidates, num_items)

    stage1_start = time.perf_counter()

    # Same candidate generation as v2: query the item-embedding index with the
    # centroid of the user's liked items (shared space), falling back to the raw
    # user embedding for users with no liked history.
    liked = user_history.get(user_idx, [])
    if liked and item_embeddings_np is not None:
        query_vec = item_embeddings_np[liked].mean(axis=0, keepdims=True).astype(np.float32)
        retrieval_method = "item-centroid"
    else:
        query_vec = model.embeddings[0](torch.tensor([user_idx])).detach().numpy().astype(np.float32)
        retrieval_method = "user-embedding-fallback"
    query_vec = (query_vec / (np.linalg.norm(query_vec) or 1.0)).astype(np.float32)

    if FAISS_AVAILABLE:
        _, candidate_indices = faiss_index.search(query_vec, num_candidates)
        candidate_indices = candidate_indices[0].tolist()
    else:
        _, indices = faiss_index.kneighbors(query_vec, n_neighbors=num_candidates)
        candidate_indices = indices[0].tolist()

    stage1_ms = (time.perf_counter() - stage1_start) * 1000

    stage2_start = time.perf_counter()

    cont = _build_user_item_features(user_feats, candidate_indices, item_features, num_continuous_features)
    cat = torch.tensor([[user_idx, idx] for idx in candidate_indices], dtype=torch.int64)

    with torch.no_grad():
        scores = model(cont, cat).squeeze()

    if scores.dim() == 0:
        scores = scores.unsqueeze(0)

    dlrm_scores = scores.detach().numpy().astype(np.float64)
    stage2_ms = (time.perf_counter() - stage2_start) * 1000

    # Stage 3: bandit exploration bonus over the same candidates. Context is the
    # elementwise user x item embedding product (raw DLRM embeddings), matching
    # the interaction term the DLRM itself computes.
    stage3_start = time.perf_counter()
    user_emb = _raw_user_emb(user_idx)
    item_embs = _raw_item_embs(candidate_indices)
    contexts = build_interaction_context(user_emb, item_embs)
    bonus = bandit.exploration_bonus(contexts)
    # Common-scale blend: the DLRM sigmoid (~0.9-1.0) and the bonus (mean ~3.3) are on different
    # scales, so combine_explore min-max normalizes both across the candidate set before the convex
    # combination. A raw dlrm_scores + bonus sum would just rank by the (larger) bonus.
    combined = combine_explore(dlrm_scores, bonus, BANDIT_EXPLORE_WEIGHT)
    stage3_ms = (time.perf_counter() - stage3_start) * 1000

    # Rank change: pure-DLRM order vs combined order. An item is "exploratory"
    # when the bonus moved it to a different rank position.
    dlrm_order = np.argsort(dlrm_scores)[::-1]
    combined_order = np.argsort(combined)[::-1]
    dlrm_rank = {int(i): r for r, i in enumerate(dlrm_order.tolist())}

    top_positions = combined_order[: min(top_k, len(candidate_indices))].tolist()

    results = []
    for new_rank, pos in enumerate(top_positions):
        item_idx = candidate_indices[pos]
        raw_item_id = int(serving_context["idx2item"][item_idx])
        meta = item_metadata.get(raw_item_id, {"title": f"Item {raw_item_id}", "genres": []})
        results.append({
            "item_id": raw_item_id,
            "score": round(float(combined[pos]), 4),
            "dlrm_score": round(float(dlrm_scores[pos]), 4),
            "exploration_bonus": round(float(bonus[pos]), 6),
            "is_exploratory": dlrm_rank[int(pos)] != new_rank,
            "title": meta["title"],
            "genres": meta["genres"],
        })

    return {
        "user_id": user_id,
        "recommendations": results,
        "pipeline": "three-stage (ANN + DLRM rerank + bandit exploration)",
        "policy": (
            f"combine_explore(w={BANDIT_EXPLORE_WEIGHT}): "
            f"(1-w)*norm(DLRM) + w*norm(LinUCB alpha={bandit.alpha} bonus)"
        ),
        "explore_weight": BANDIT_EXPLORE_WEIGHT,
        "retrieval": retrieval_method,
        "num_candidates": num_candidates,
        "latency_ms": {
            "stage1_ann": round(stage1_ms, 2),
            "stage2_rerank": round(stage2_ms, 2),
            "stage3_bandit": round(stage3_ms, 2),
            "total": round(stage1_ms + stage2_ms + stage3_ms, 2),
        },
        "ann_backend": "faiss-cpu" if FAISS_AVAILABLE else "sklearn-NearestNeighbors",
    }


@app.post("/api/v1/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """Online-learning update: apply an observed (user, item, reward) to the bandit.

    Builds the interaction context from the stored DLRM embeddings, calls
    ``bandit.update(context, reward)``, and persists the new state to
    ``bandit_state.pkl``. This is what makes the exploration layer a real online
    learner rather than a static bonus.
    """
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if serving_context is None:
        raise HTTPException(status_code=503, detail="Serving context is not loaded.")

    if bandit is None:
        raise HTTPException(status_code=503, detail="Bandit exploration layer is not available.")

    if not 0.0 <= request.reward <= 1.0:
        raise HTTPException(
            status_code=400,
            detail=f"reward must be in [0, 1], got {request.reward}.",
        )

    if request.item_id not in serving_context["item2idx"]:
        raise HTTPException(status_code=400, detail=f"Unknown item: {request.item_id}.")

    item_idx = serving_context["item2idx"][request.item_id]
    item_emb = _raw_item_embs([item_idx])[0]

    # Known users get their personalised embedding; unknown users fall back to
    # the mean user embedding (same anchor as the cold-start recommend path).
    if request.user_id in serving_context["user2idx"]:
        user_emb = _raw_user_emb(serving_context["user2idx"][request.user_id])
    elif mean_user_emb_np is not None:
        user_emb = mean_user_emb_np.astype(np.float64)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown user: {request.user_id}.")

    context = build_interaction_context(user_emb, item_emb)

    # Serialize the read-modify-write: update the in-memory bandit and atomically persist it under
    # a single lock so concurrent /feedback requests (within this worker) cannot interleave the A /
    # A_inv update or observe a half-written pickle. Multi-worker deployments still race the file
    # (each worker has its own bandit); see EVALUATION.md "single-worker" note.
    with bandit_lock:
        bandit.update(context, request.reward)

        persisted = False
        try:
            save_state(bandit, BANDIT_STATE_PATH)
            persisted = True
        except Exception as exc:
            logger.error("Failed to persist bandit state: %s", exc, exc_info=True)

        mean, bonus = (
            float(bandit.exploit_mean(context)[0]),
            float(bandit.exploration_bonus(context)[0]),
        )

    return {
        "success": True,
        "user_id": request.user_id,
        "item_id": request.item_id,
        "reward": request.reward,
        "updates_applied": 1,
        "exploit_mean": round(mean, 6),
        "exploration_bonus": round(bonus, 6),
        "persisted": persisted,
    }


@app.post("/predict/", response_model=list[RecommendationResponse])
async def predict_legacy(request: PredictionRequest):
    """Legacy predict endpoint -- delegates to /api/v1/predict."""
    return await predict(request)

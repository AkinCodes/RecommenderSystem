from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import requests
import os
import logging
from dotenv import load_dotenv
from models.dlrm import DLRMModel
import uvicorn

# Setup Logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# FastAPI App Setup
app = FastAPI(
    title="CinemaScopeAI 🎬",
    description="AI-powered movie recommendation API using DLRM + TMDB",
    version="1.0.0",
)

# TMDB API Keys
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")


# --- Movie Fetcher Function ---
def fetch_real_movies():
    """Fetch movie data from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/popular?language=en-US&page=1"
    headers = {
        "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
        "accept": "application/json",
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        logging.error(f"TMDB API Error: {response.status_code} {response.text}")
        return []

    try:
        movies = response.json().get("results", [])
    except Exception as e:
        logging.error(f"Error parsing TMDB response: {e}")
        return []

    return [
        {
            "title": movie["title"],
            "genre": "Unknown",  # Replace with actual genre logic if available
            "rating": "N/A",  # Add real rating logic
            "score": movie["vote_average"] / 10,
            "poster_url": (
                f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                if movie.get("poster_path")
                else "https://via.placeholder.com/500"
            ),
            "director": "N/A",  # Add director info if available
            "release_year": (
                int(movie["release_date"].split("-")[0])
                if movie.get("release_date")
                else "Unknown"
            ),
            "summary": movie["overview"],
        }
        for movie in movies
    ]


# --- Request and Response Models ---
class PredictionRequest(BaseModel):
    """Request model for prediction."""

    continuous_features: List[float]
    categorical_features: List[int]

    class Config:
        schema_extra = {
            "example": {
                "continuous_features": [0.6, 0.8],
                "categorical_features": [1, 7],
            }
        }


class RecommendationResponse(BaseModel):
    """Response model for movie recommendations."""

    title: str
    genre: str
    rating: str
    score: float
    poster_url: str
    director: str
    release_year: int
    summary: str


# --- Model Loading ---
num_continuous_features = 2
num_categorical_features = 2
num_genres = 73
embedding_sizes = [2, 18] + [2] * num_genres

model = DLRMModel(
    num_continuous_features=num_continuous_features,
    embedding_sizes=embedding_sizes,
    mlp_layers=[64, 32, 16],
)

# Load the model state
try:
    state_dict = torch.load("trained_model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    raise RuntimeError(f"❌ Failed to load model: {e}")


# --- Routes ---
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"message": "Recommendation API is running!"}


@app.post(
    "/predict/",
    summary="Get Movie Recommendations",
    description=(
        "**Valid Index Ranges:**\n"
        "- `categorical_features[0]`: 0 - 1 (gender)\n"
        "- `categorical_features[1]`: 0 - 17 (age bucket)\n"
        "- You must provide exactly 2 categorical features.\n\n"
        "**Note:** The model will internally pad genre indexes (73 extra dimensions)."
    ),
    response_model=Dict[str, Any],
    tags=["Inference"],
)
async def predict(request: PredictionRequest) -> Dict[str, Any]:
    """Endpoint to get movie recommendations based on user features."""

    # Validate input features
    if len(request.categorical_features) != num_categorical_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {num_categorical_features} categorical features.",
        )

    if request.categorical_features[0] > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Feature 0 index {request.categorical_features[0]} out of range (max allowed is 1).",
        )
    if request.categorical_features[1] > 17:
        raise HTTPException(
            status_code=400,
            detail=f"Feature 1 index {request.categorical_features[1]} out of range (max allowed is 17).",
        )

    # Pad genre features and convert to tensor
    full_categorical_features = request.categorical_features + [0] * num_genres
    continuous_tensor = torch.tensor([request.continuous_features], dtype=torch.float32)
    categorical_tensor = torch.tensor([full_categorical_features], dtype=torch.int64)

    # Predict score using model
    try:
        prediction_score = model(continuous_tensor, categorical_tensor).item()
    except Exception as e:
        logging.error(f"Error during model prediction: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")

    # Fetch real movie data
    MOVIE_DATABASE = fetch_real_movies()
    if not MOVIE_DATABASE:
        raise HTTPException(status_code=500, detail="Failed to fetch movies from TMDB.")

    # Sort and return the top 5 closest recommendations based on predicted score
    recommended_movies = sorted(
        MOVIE_DATABASE, key=lambda x: abs(x["score"] - prediction_score), reverse=True
    )

    return {"recommendations": recommended_movies[:5]}


# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

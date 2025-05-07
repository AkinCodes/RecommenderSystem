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

logging.basicConfig(level=logging.INFO)
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
                else "Unknown"
            ),
            "summary": movie["overview"],
        }
        for movie in movies
    ]


# --- Request Model for Human Input ---
class UserMovieInput(BaseModel):
    """User-facing input for movie recommendations."""

    release_year: int
    duration_text: str  # e.g. "2 Seasons", "90 min"
    type: str  # "Movie" or "TV Show"
    rating: str  # "PG", "R", "TV-MA", etc.

    class Config:
        schema_extra = {
            "example": {
                "release_year": 1999,
                "duration_text": "2 Seasons",
                "type": "TV Show",
                "rating": "PG-13",
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
embedding_sizes = [2, 18]  # Match training exactly

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


from scripts.preprocessing import load_and_apply_scaler, load_encoders


def parse_duration(val: str) -> int:
    try:
        return int(val.split(" ")[0])
    except:
        return 0


@app.post("/predict/", tags=["Inference"])
async def predict_user_input(input: UserMovieInput):
    try:
        type_encoder, rating_encoder = load_encoders()
    except Exception as e:
        logging.error(f"Failed to load encoders: {e}")
        raise HTTPException(status_code=500, detail="Failed to load encoders")

    duration = parse_duration(input.duration_text)
    try:
        norm_cont = load_and_apply_scaler(input.release_year, duration)
    except Exception as e:
        logging.error(f"Scaler load/transform error: {e}")
        raise HTTPException(status_code=500, detail="Scaler error")

    continuous_tensor = torch.tensor([norm_cont], dtype=torch.float32)

    try:
        type_index = int(type_encoder.transform([input.type])[0])
        rating_index = int(rating_encoder.transform([input.rating])[0])
    except Exception as e:
        logging.error(f"Encoding error: {e}")
        raise HTTPException(status_code=400, detail="Invalid 'type' or 'rating' input.")

    full_cat = [type_index, rating_index]
    categorical_tensor = torch.tensor([full_cat], dtype=torch.int64)

    try:
        prediction_score = model(continuous_tensor, categorical_tensor).item()
    except Exception as e:
        logging.error(f"Model inference error: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")

    # Get movies and return results
    movies = fetch_real_movies()
    if not movies:
        raise HTTPException(status_code=500, detail="Failed to fetch movies.")

    recommendations = sorted(
        movies, key=lambda x: abs(x["score"] - prediction_score), reverse=True
    )
    return {"recommendations": recommendations[:5]}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any
import torch
import numpy as np
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
    title="CinemaScopeAI",
    description="AI-powered movie recommendation API using DLRM + TMDB",
    version="1.0.0",
)

# TMDB API Keys
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")


# --- Movie Fetcher Function ---
def fetch_real_movies(content_type: str, release_year: int, rating: str):
    """Fetch movie or TV show data using TMDB Discover API, filtered by user input."""
    endpoint = "tv" if content_type.lower() == "tv show" else "movie"
    url = f"https://api.themoviedb.org/3/discover/{endpoint}"

    # Common query parameters
    params = {
        "language": "en-US",
        "sort_by": "popularity.desc",
        "page": 1,
    }

    # Conditional filters
    if endpoint == "movie":
        params["certification_country"] = "US"
        params["certification"] = rating
        params["primary_release_year"] = release_year
    else:
        params["first_air_date_year"] = release_year

    # Headers
    headers = {
        "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
        "accept": "application/json",
    }

    logging.info(f"🟢 TMDB request URL: {url}")
    logging.info(f"🟢 TMDB request params: {params}")
    logging.info(f"🟢 TMDB headers: {headers}")

    # Request
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        logging.error(f"TMDB API Error: {response.status_code} {response.text}")
        return []

    # Parse and transform results
    try:
        raw_results = response.json().get("results", [])
    except Exception as e:
        logging.error(f"Error parsing TMDB response: {e}")
        return []

    movies = [
        {
            "title": movie.get("title") or movie.get("name", "Untitled"),
            "genre": "Unknown",
            "rating": rating,
            "score": movie.get("vote_average", 0) / 10,
            "poster_url": (
                f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                if movie.get("poster_path")
                else "https://via.placeholder.com/500"
            ),
            "director": "N/A",
            "release_year": (
                int(movie.get("release_date", "0000").split("-")[0])
                if endpoint == "movie"
                else int(movie.get("first_air_date", "0000").split("-")[0])
            ),
            "summary": movie.get("overview", "No summary available."),
        }
        for movie in raw_results
    ]

    movies = [m for m in movies if m["score"] > 0]
    return movies


# --- Request Model ---
class UserMovieInput(BaseModel):
    release_year: int
    duration_text: str
    type: str
    rating: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "release_year": 1999,
                "duration_text": "2 Seasons",
                "type": "TV Show",
                "rating": "PG-13",
            }
        }
    )


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
embedding_sizes = [2, 18]

model = DLRMModel(
    num_continuous_features=num_continuous_features,
    embedding_sizes=embedding_sizes,
    mlp_layers=[64, 32, 16],
)


# Load the model state
try:
    state_dict = torch.load(
        "trained_model.pth", map_location=torch.device("cpu"), weights_only=True
    )
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


import re


def parse_duration(val: str) -> int:
    if not val or not isinstance(val, str):
        return 0

    val = val.lower().strip()

    # Handle "X Seasons"
    if "season" in val:
        match = re.search(r"(\d+)", val)
        return int(match.group(1)) * 10 if match else 0

    # Handle "1h 30m" style
    hour_match = re.search(r"(\d+)\s*h", val)
    min_match = re.search(r"(\d+)\s*m", val)

    total_minutes = 0
    if hour_match:
        total_minutes += int(hour_match.group(1)) * 60
    if min_match:
        total_minutes += int(min_match.group(1))

    if total_minutes > 0:
        return total_minutes

    # Handle fallback "90 min" or single number
    match = re.search(r"(\d+)", val)
    return int(match.group(1)) if match else 0


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

    continuous_tensor = torch.from_numpy(norm_cont).unsqueeze(0)

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
    movies = fetch_real_movies(input.type, input.release_year, input.rating)

    if not movies:
        raise HTTPException(status_code=500, detail="Failed to fetch movies.")

    recommendations = sorted(
        movies, key=lambda x: abs(x["score"] - prediction_score), reverse=True
    )
    return {"recommendations": recommendations[:5]}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

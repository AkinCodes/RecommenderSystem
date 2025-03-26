from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import requests
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from models.dlrm import DLRMModel

# Load .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

app = FastAPI(
    title="CinemaScopeAI 🎬",
    description="AI-powered movie recommendation API using DLRM + TMDB",
    version="1.0.0",
)

# TMDB Keys
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")


# --- Movie Fetcher ---
def fetch_real_movies():
    url = f"https://api.themoviedb.org/3/movie/popular?language=en-US&page=1"
    headers = {
        "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
        "accept": "application/json",
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("🛑 TMDB API Error:", response.status_code, response.text)
        return []

    try:
        movies = response.json()["results"]
    except Exception as e:
        print("🛑 Error parsing TMDB response:", e)
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


# --- Request + Response Models ---
class PredictionRequest(BaseModel):
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
    num_features=num_continuous_features,
    embedding_sizes=embedding_sizes,
    mlp_layers=[64, 32, 16],
)

state_dict = torch.load("trained_model.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()


# --- Routes ---
@app.get("/", tags=["Health"])
async def root():
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
    if len(request.categorical_features) != num_categorical_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {num_categorical_features} categorical features.",
        )

    # Validate range
    if request.categorical_features[0] > 1:
        raise HTTPException(
            status_code=400,
            detail="Feature 0 index {} out of range (max allowed is 1).".format(
                request.categorical_features[0]
            ),
        )
    if request.categorical_features[1] > 17:
        raise HTTPException(
            status_code=400,
            detail="Feature 1 index {} out of range (max allowed is 17).".format(
                request.categorical_features[1]
            ),
        )

    # Pad genre
    full_categorical_features = request.categorical_features + [0] * num_genres
    continuous_tensor = torch.tensor([request.continuous_features], dtype=torch.float32)
    categorical_tensor = torch.tensor([full_categorical_features], dtype=torch.int64)

    # Predict score
    prediction_score = model(continuous_tensor, categorical_tensor).item()

    # Get movie database
    MOVIE_DATABASE = fetch_real_movies()
    if not MOVIE_DATABASE:
        raise HTTPException(status_code=500, detail="Failed to fetch movies from TMDB.")

    # Sort movies by closeness to predicted score
    recommended_movies = sorted(
        MOVIE_DATABASE, key=lambda x: abs(x["score"] - prediction_score), reverse=True
    )

    return {"recommendations": recommended_movies[:5]}


class RecommendRequest(BaseModel):
    user_id: int


class Recommendation(BaseModel):
    title: str
    rating: float


@app.post("/recommend", response_model=List[Recommendation])
async def recommend_movies(request: RecommendRequest):
    # Dummy recommendations for now
    return [
        {"title": "Inception", "rating": 9.0},
        {"title": "The Matrix", "rating": 8.7},
        {"title": "Interstellar", "rating": 8.6},
    ]

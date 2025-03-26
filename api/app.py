from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import torch
from models.dlrm import DLRMModel
from typing import List
from dotenv import load_dotenv
from typing import Dict, Any
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


app = FastAPI()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")


# Fetch real movie data from TMDB
def fetch_real_movies():
    url = f"https://api.themoviedb.org/3/movie/popular?language=en-US&page=1"
    headers = {
        "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
        "accept": "application/json",
    }
    # response = requests.get(url, headers=headers)
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("🛑 TMDB API Error:", response.status_code, response.text)
        return []

    try:
        results = response.json()["results"]
        ...
    except Exception as e:
        print("🛑 Error parsing TMDB response:", e)
        return []

    if response.status_code == 200:
        movies = response.json()["results"]
        movie_list = [
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
        return movie_list
    else:
        return []


class PredictionRequest(BaseModel):
    continuous_features: List[float]
    categorical_features: List[int]


class RecommendationResponse(BaseModel):
    title: str
    genre: str
    rating: str
    score: float
    poster_url: str
    director: str
    release_year: int
    summary: str


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


# Prediction Route (Fetch Movie Recommendations)
# @app.post("/predict/")
# async def predict(request: PredictionRequest) -> Dict[str, Any]:
#     if len(request.categorical_features) != num_categorical_features:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Expected {num_categorical_features} categorical features.",
#         )

#     full_categorical_features = request.categorical_features + [0] * num_genres
#     continuous_tensor = torch.tensor([request.continuous_features], dtype=torch.float32)
#     categorical_tensor = torch.tensor([full_categorical_features], dtype=torch.int64)

#     # Generate a prediction score
#     prediction_score = model(continuous_tensor, categorical_tensor).item()

#     # Fetch real movies from TMDB
#     MOVIE_DATABASE = fetch_real_movies()

#     if not MOVIE_DATABASE:
#         raise HTTPException(status_code=500, detail="Failed to fetch movies from TMDB.")

#     recommended_movies = sorted(
#         MOVIE_DATABASE, key=lambda x: abs(x["score"] - prediction_score), reverse=True
#     )
#     return {"recommendations": recommended_movies[:5]}

#     # return recommended_movies[:5]  # Return top 5 movie recommendations


# Prediction Route (Fetch Movie Recommendations)
@app.post("/predict/")
async def predict(request: PredictionRequest) -> Dict[str, Any]:
    if len(request.categorical_features) != num_categorical_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {num_categorical_features} categorical features.",
        )

    # Validate index ranges against embedding_sizes
    max_indices = embedding_sizes[:num_categorical_features]  # e.g. [2, 18]
    for i, val in enumerate(request.categorical_features):
        if val >= max_indices[i]:
            raise HTTPException(
                status_code=400,
                detail=f"Feature {i} index {val} out of range (max allowed is {max_indices[i] - 1}).",
            )

    full_categorical_features = request.categorical_features + [0] * num_genres
    continuous_tensor = torch.tensor([request.continuous_features], dtype=torch.float32)
    categorical_tensor = torch.tensor([full_categorical_features], dtype=torch.int64)

    # Generate a prediction score
    prediction_score = model(continuous_tensor, categorical_tensor).item()

    # Fetch real movies from TMDB
    MOVIE_DATABASE = fetch_real_movies()

    if not MOVIE_DATABASE:
        raise HTTPException(status_code=500, detail="Failed to fetch movies from TMDB.")

    recommended_movies = sorted(
        MOVIE_DATABASE, key=lambda x: abs(x["score"] - prediction_score), reverse=True
    )
    return {"recommendations": recommended_movies[:5]}


# Health Check Endpoint
@app.get("/")
async def root():
    return {"message": "Recommendation API is running!"}

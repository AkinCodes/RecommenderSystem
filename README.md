# 🎬 CinemaScopeAI – AI-Powered Movie Recommendation System

CinemaScopeAI is a **full-stack, production-ready AI movie recommendation platform** that leverages deep learning and scalable backend infrastructure. It combines a powerful **Python-based FastAPI backend** with a **Swift-based iOS frontend**. Built using **Clean Architecture** principles, this project is modular, testable, and cloud-deployable via **Docker** and **Render/AWS ECS**.

---

## Demo
**Live Backend**: Render/AWS ECS deployment

**📱 iOS Frontend Preview:**  

<img src="https://github.com/user-attachments/assets/ba68128d-5340-4e9b-8c09-22376492176f" width="300" />

---

## How It Works
The system uses **collaborative filtering** and **content-based techniques** to recommend movies based on user behavior and content similarity.

### Architecture Overview
```
Xcode App (SwiftUI) 📱
       ↓  REST API
FastAPI (Python)
       ↓  PyTorch
DL Recommendation Model 🎥
       ↓
Deployed via Docker + Render/AWS ECS ☁️
```

---

## Tech Stack

### 🔙 Backend (FastAPI)
- Python 3.10+
- FastAPI
- PyTorch
- scikit-learn
- Uvicorn + Gunicorn
- Dockerized + Deployable to AWS ECR/ECS or Render

### 📱 Frontend (iOS)
- Swift
- SwiftUI
- MVVM
- URLSession networking
- Async API calls to deployed FastAPI

---

## Features
- **Smart Recommendations** – Based on vector embeddings and metadata.
- **API-Driven** – Clean, documented RESTful endpoints.
- **Unit & Integration Tests** – For both backend and frontend.
- **CI/CD Ready** – GitHub Actions + Docker + Render/AWS ECS.
- **Custom Model Training** – Scripted via `train.py` & `inference.py`.

---

## 📊 Dataset
This project leverages the [`netflix_titles.csv`](https://www.kaggle.com/datasets/shivamb/netflix-shows) dataset, sourced from Kaggle. It contains metadata about Netflix content, including **titles, directors, genres, cast, release years, and descriptions**.

While the deployed recommendation engine uses **real-time data from TMDB**, this dataset was vital during experimentation and model development:

- Cold-start simulations
- Embedding training
- Initial data preprocessing & pipeline validation

It laid the groundwork for learning before switching to live TMDB-fetching in production.

---

## 📁 Project Structure
```
RecommenderSystem2/
├── api/                 # FastAPI routes
│   └── app.py
├── scripts/             # Model training + inference logic
│   └── train.py
│   └── inference.py
├── models/              # Saved model artifacts
├── tests/               # Unit tests
├── Dockerfile           # Backend containerization
├── requirements.txt     # Python dependencies
├── .gitignore
└── trust-policy.json    # (legacy AWS setup)
```

---

## 🧩 Running Locally

### 🔙 Backend (FastAPI)
```bash
# 1. Create and activate a virtualenv
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
uvicorn api.app:app --reload
```
Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 📱 Frontend (iOS)
- Open `CinemaScopeAI.xcodeproj` in Xcode
- In `CinemaScopeAIService.swift`, set:
  ```swift
  let baseURL = "https://cinemascope-api.onrender.com"
  ```
- Build & run on iOS Simulator (iOS 16+)
- GitHub iOS Frontend Repo: [CinemaScopeAI (Frontend)](https://github.com/AkinCodes/CinemaScopeAI)

---

## Deployment

### 🐳 Docker
```bash
# Build Docker image
docker build -t cinemascope-recsys .

# Run locally
docker run -d -p 8000:8000 cinemascope-recsys
```

### ☁️ Render (Easy Alternative)
- Create a new Web Service
- Connect GitHub repo
- Set:
  ```bash
  uvicorn api.app:app --host 0.0.0.0 --port 8000
  ```
- Done 

### (Optional) AWS ECS (Fargate)
- Push Docker image to Amazon ECR
- Use ECS CLI or Console to deploy
- Auto-scales and generates public endpoint

---

## ✅ To Do
- [x] Backend API endpoints
- [x] Xcode frontend integration
- [x] Torch model training/inference
- [x] Docker containerization
- [x] GitHub Actions (CI)

---

## Inspiration
Built as a **portfolio-grade project** to demonstrate expertise in:
- **End-to-end ML Systems**
- **iOS Development & App Architecture**
- **DevOps & Scalable Deployments**
- **Modern UX integrated with real-time ML APIs**

---

## Author
**Akin Olusanya**  
🎓 iOS Engineer | ML Enthusiast | Full-Stack Creator  
📧 workwithakin@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/akindeveloper)  
📁 [GitHub](https://github.com/AkinCodes)


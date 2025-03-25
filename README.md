🎬 CinemaScopeAI – AI-Powered Movie Recommendation System
CinemaScopeAI is a full-stack, production-ready AI movie recommendation platform that leverages deep learning and scalable backend infrastructure. It combines a powerful Python-based FastAPI backend with a Swift-based iOS frontend. Built with Clean Architecture principles, this project is modular, testable, and cloud-deployable via Docker and AWS ECS.

🚀 Demo
🔗 Live Backend (Render/AWS ECS)

📱 iOS Frontend Preview:
<img src="screenshot_url" width="300" />

🧠 How It Works
The system uses collaborative filtering and content-based techniques to recommend movies based on user behavior and content similarity.

Architecture Overview:

css
Copy
Edit
Xcode App (SwiftUI) 📱 
       ↓ REST API
FastAPI (Python 🔥) 🚀
       ↓ PyTorch
DL Recommendation Model 🎥
       ↓
Deployed via Docker + Render/AWS ECS ☁️
📦 Tech Stack
🔙 Backend (FastAPI)
Python 3.10+

FastAPI

PyTorch

scikit-learn

Uvicorn + Gunicorn

Dockerized + Deployable to AWS ECR/ECS or Render

📱 Frontend (iOS)
Swift

SwiftUI

MVVM

URLSession networking

Async API calls to deployed FastAPI

🧪 Features
🔍 Smart Recommendations – Based on vector embeddings and metadata.

🌐 API-Driven – Clean, documented RESTful endpoints.

🧪 Unit & Integration Tests – For both backend and frontend.

⚙️ CI/CD Ready – GitHub Actions + Docker + Render/AWS ECS.

💾 Custom Model Training – Scripted via train.py & inference.py.

📁 Project Structure
graphql
Copy
Edit
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
└── trust-policy.json    # AWS ECS IAM Trust policy
🧪 Running Locally
Backend (FastAPI)
bash
Copy
Edit
# 1. Create and activate a virtualenv
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
uvicorn api.app:app --reload
Visit: http://127.0.0.1:8000/docs

Frontend (iOS)
Open CinemaScopeAI.xcodeproj

Update base URL in CinemaScopeAIService.swift

Build & run the app in the simulator

📦 Deployment
🐳 Docker
bash
Copy
Edit
# Build Docker image
docker build -t cinemascope-recsys .

# Run locally
docker run -d -p 8000:8000 cinemascope-recsys
🚀 AWS ECS (Fargate)
Push image to Amazon ECR

Use ECS CLI or Console to deploy to Fargate

Auto-scales and publicly available endpoint

⚡ Render (Easy Alternative)
Create a new Web Service

Connect GitHub repo

Set uvicorn api.app:app --host 0.0.0.0 --port 8000 as start command

Done 🎉

✅ To Do
 Backend API endpoints

 Xcode frontend integration

 Torch model training/inference

 Docker containerization

 GitHub Actions (CI)

 Add Swagger custom docs

 Expand to TV show recommendations

 Integrate Firebase Auth (iOS)

🧠 Inspiration
Built as a portfolio-ready project to showcase modern machine learning, mobile development, and DevOps pipelines — from model training to user experience.

📩 Contact
Akin Olusanya
📧 [your.email@example.com]
🔗 LinkedIn
📁 Portfolio


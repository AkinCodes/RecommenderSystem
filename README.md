# 🎬 CinemaScopeAI – AI-Powered Movie Recommendation System

CinemaScopeAI is a **full-stack, production-ready AI movie recommendation platform** that leverages deep learning and scalable backend infrastructure. It combines a powerful **Python-based FastAPI backend** with a **Swift-based iOS frontend**. Built using **Clean Architecture** principles, this project is modular, testable, and cloud-deployable via **Docker** and **Render**.

As a bonus, the app includes a GPT-powered natural language recommender, allowing users to type freeform prompts like “mind-bending sci-fi thrillers with a twist” and receive smart, tailored movie suggestions using the OpenAI API.

---

## Demo
**Live Backend**: Render deployment

**📱 iOS Frontend Preview:**  

<img src="https://github.com/user-attachments/assets/ba68128d-5340-4e9b-8c09-22376492176f" width="300" />

<img src="https://github.com/user-attachments/assets/ff45edd1-489e-4192-8d1a-27fb90a15fd0" width="300" />

<img src="https://github.com/user-attachments/assets/881c4d17-b156-4f18-b11e-c0c0244134d6" width="300" />

---

## How It Works
The system uses **collaborative filtering** and **content-based techniques** to recommend movies based on user behavior and content similarity.

### Architecture Overview
```
Xcode App (SwiftUI) 
       ↓  REST API
FastAPI (Python)
       ↓  PyTorch
DL Recommendation Model 
       ↓
Deployed via Docker + Render
```

---

## Tech Stack

### Backend (FastAPI)
- Python 3.10+
- FastAPI
- PyTorch
- scikit-learn
- Uvicorn + Gunicorn
- Dockerized + Deployable to AWS ECR/ECS or Render

### Frontend (iOS)
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

## Dataset
This project leverages the [`netflix_titles.csv`](https://www.kaggle.com/datasets/shivamb/netflix-shows) dataset, sourced from Kaggle. It contains metadata about Netflix content, including **titles, directors, genres, cast, release years, and descriptions**.

While the deployed recommendation engine uses **real-time data from TMDB**, this dataset was vital during experimentation and model development:

- Cold-start simulations
- Embedding training
- Initial data preprocessing & pipeline validation

It laid the groundwork for learning before switching to live TMDB-fetching in production.

---

## Project Structure
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

## Running Locally

### Backend (FastAPI)
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


## Training the Model & Visualizing with TensorBoard

CinemaScopeAI includes a **custom training pipeline** powered by **PyTorch Lightning**, with **TensorBoard** support built in for real-time training visualization and performance insights.

---

### Step 1: Train the Model

```bash
# Activate your virtual environment
source venv/bin/activate

# Run the training script
python scripts/train.py
```

This will:

- Train the DLRM-based recommendation model  
- Log training & validation loss to `lightning_logs/`  
- Save model checkpoints for future inference  

---

### Step 2: Launch TensorBoard

```bash
tensorboard --logdir lightning_logs
```

Then open the URL shown in your terminal — typically:

- [`http://localhost:6006`](http://localhost:6006)  
- Or something like [`http://localhost:6008`](http://localhost:6008) if 6006 is in use  

You’ll be able to:

- Visualize **training & validation loss curves**
- Explore **epoch-by-epoch scalar metrics**
- Monitor how your model is learning in real time

### TensorBoard Screenshots Preview: 
<img width="300" height="300" alt="Screenshot 2025-04-10 at 08 20 11" src="https://github.com/user-attachments/assets/37b82d80-e7e0-4d9e-af8c-869ee86a1ff0" />
<img width="300" height="300" alt="Screenshot 2025-04-10 at 08 20 38" src="https://github.com/user-attachments/assets/c7a1fde6-1019-42a2-902a-692779f526d0" />
<img width="300" height="300" alt="Screenshot 2025-04-10 at 08 21 02" src="https://github.com/user-attachments/assets/a94d0f0e-fa3c-4153-9ca9-b580acc8006c" />

---

### Pro Tip: Keep It Clean

All logs and checkpoints are excluded from Git via `.gitignore`.  
You can regenerate them anytime by re-running the training script above.

---

### Netflix SQL Analysis: Country vs Content Type Breakdown

As part of exploring my SQL and data visualization skills, I performed a detailed analysis using **SQLite + Pandas** to uncover how Netflix content varies across countries.

### 🔍 Objective
**Which countries have the most Netflix content, and what's the breakdown between Movies and TV Shows in those countries?**

### SQL Query (SQLite)
```sql
SELECT 
    country,
    type,
    COUNT(*) AS count
FROM netflix_titles
WHERE country IS NOT NULL
GROUP BY country, type
HAVING country IN (
    SELECT country
    FROM netflix_titles
    WHERE country IS NOT NULL
    GROUP BY country
    ORDER BY COUNT(*) DESC
    LIMIT 5
)
ORDER BY country, type;
```

### 📈 Visualization
Using `matplotlib`, I transformed the result into a **stacked bar chart**, showing the split between **TV Shows** and **Movies** per country:

<img width="456" alt="Screenshot 2025-04-10 at 17 56 33" src="https://github.com/user-attachments/assets/fe5ea478-73ca-4c71-8fa7-d6aa8b27c38a" />

<br><br>

**Bonus: Epic Stacked Bar Plot** — Built using Pandas' pivot + Matplotlib for clean, readable comparison across countries.

<img width="1063" alt="Screenshot 2025-04-10 at 17 57 18" src="https://github.com/user-attachments/assets/a921b128-50a5-4d33-8115-f297faef9d40" />



### Skills Demonstrated
- SQL aggregation: `COUNT`, `GROUP BY`, `HAVING`, subqueries
- Data loading, transformation, and analysis with `pandas`
- Database creation and interaction with `sqlite3`
- Clean and professional visualization with `matplotlib`


---


## Deployment

### Docker
```bash
# Build Docker image
docker build -t cinemascope-recsys .

# Run locally
docker run -d -p 8000:8000 cinemascope-recsys
```

### Render
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

## Inspiration
Built as a **portfolio-grade project** to demonstrate expertise in:
- **End-to-end ML Systems**
- **iOS Development & App Architecture**
- **DevOps & Scalable Deployments**
- **Modern UX integrated with real-time ML APIs**

---

## Author
**Akin Olusanya**  
iOS Engineer | ML Engineer | Full-Stack Creator  
workwithakin@gmail.com  
[LinkedIn](https://www.linkedin.com/in/akindeveloper)  
[GitHub](https://github.com/AkinCodes)


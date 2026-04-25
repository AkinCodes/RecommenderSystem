# Use official Python image as the base
FROM python:3.11-slim

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies that uvicorn may need
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential libffi-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy project metadata first for better caching
COPY pyproject.toml .

# Install Python dependencies using uv
RUN uv pip install --system .

# Now copy the rest of the code
COPY . .

# Ensure project root is on Python path so data/models/api are importable
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI app
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

.PHONY: run test docker-build docker-run lint clean install

# Install dependencies using uv
install:
	uv sync --all-extras

# Run the FastAPI development server
run:
	uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Run the test suite
test:
	uv run pytest tests/ -v

# Build the Docker image
docker-build:
	docker build -t recommender-system .

# Run the Docker container
docker-run:
	docker run --rm -p 8000:8000 --env-file .env recommender-system

# Lint the codebase
lint:
	uv run ruff check .
	uv run ruff format --check .

# Remove build artifacts and caches
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info .venv

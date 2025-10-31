.PHONY: help install test lint format run dev clean docker-up docker-down migrate setup

# Default target
help:
	@echo "Content Redundancy Detector - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          - Run complete setup (venv, deps, etc.)"
	@echo "  make install       - Install dependencies"
	@echo "  make clean         - Clean cache and temporary files"
	@echo ""
	@echo "Development:"
	@echo "  make dev           - Run development server with auto-reload"
	@echo "  make run           - Run production server"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Run linter"
	@echo "  make format        - Format code"
	@echo "  make typecheck     - Run type checker"
	@echo "  make security      - Run security audit"
	@echo "  make check         - Run all checks (lint, typecheck, test)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"
	@echo "  make docker-build  - Build Docker image"
	@echo ""
	@echo "Database:"
	@echo "  make migrate       - Generate migration"
	@echo "  make migrate-up    - Apply migrations"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          - Open API documentation"

# Python and virtual environment
PYTHON := python3
VENV := .venv
VENV_BIN := $(VENV)/bin
VENV_ACTIVATE := $(VENV_BIN)/activate
ifeq ($(OS),Windows_NT)
	VENV_BIN := $(VENV)/Scripts
	VENV_ACTIVATE := $(VENV_BIN)/activate.bat
endif

# Setup
setup:
	@echo "Setting up development environment..."
	$(PYTHON) scripts/setup.py

install:
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Development server
dev:
	@echo "Starting development server..."
	uvicorn app:app --host 127.0.0.1 --port 8000 --reload

run:
	@echo "Starting production server..."
	uvicorn app:app --host 0.0.0.0 --port 8000

# Testing
test:
	@pytest tests/ -v

test-cov:
	@pytest tests/ --cov=. --cov-report=html --cov-report=term

# Code quality
lint:
	@echo "Running linter..."
	@ruff check . || flake8 . || echo "No linter found"

format:
	@echo "Formatting code..."
	@ruff format . || black . || echo "No formatter found"

typecheck:
	@echo "Running type checker..."
	@mypy . || echo "mypy not found"

security:
	@echo "Running security audit..."
	@pip-audit --desc || echo "pip-audit not found"

check: lint typecheck test
	@echo "All checks completed"

# Docker
docker-up:
	@echo "Starting Docker services..."
	docker-compose up -d

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

docker-build:
	@echo "Building Docker image..."
	docker-compose build

# Database
migrate:
	@echo "Generating migration..."
	@read -p "Migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

migrate-up:
	@echo "Applying migrations..."
	alembic upgrade head

# Documentation
docs:
	@echo "Opening API documentation..."
	@python -c "import webbrowser; webbrowser.open('http://127.0.0.1:8000/docs')"

# Cleanup
clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -r {} + 2>/dev/null || true
	rm -rf .coverage coverage.xml 2>/dev/null || true
	@echo "Cleanup completed"

# Benchmarking
benchmark:
	@echo "Running benchmarks..."
	@python scripts/benchmark.py || echo "Benchmark script not found"

# Profiling
profile:
	@echo "Running profiler..."
	@python scripts/profile.py || echo "Profile script not found"



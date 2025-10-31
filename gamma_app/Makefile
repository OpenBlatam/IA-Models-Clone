# Gamma App - Makefile
# Development and deployment commands

.PHONY: help install dev test build deploy clean lint format

# Default target
help:
	@echo "Gamma App - Available Commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make dev        - Start development server"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linting"
	@echo "  make format     - Format code"
	@echo "  make build      - Build Docker images"
	@echo "  make deploy     - Deploy to production"
	@echo "  make clean      - Clean up containers and volumes"

# Install dependencies
install:
	pip install -r requirements.txt

# Start development server
dev:
	python scripts/start.py

# Run tests
test:
	python scripts/test.py

# Run linting
lint:
	flake8 gamma_app/ --max-line-length=100 --ignore=E203,W503
	black --check gamma_app/
	isort --check-only gamma_app/
	mypy gamma_app/ --ignore-missing-imports

# Format code
format:
	black gamma_app/
	isort gamma_app/

# Build Docker images
build:
	docker-compose build

# Deploy to production
deploy:
	python scripts/deploy.py

# Clean up
clean:
	docker-compose down -v
	docker system prune -f

# Development with hot reload
dev-docker:
	docker-compose -f docker-compose.yml up --build

# Production deployment
prod:
	docker-compose -f docker-compose.yml --profile production up -d

# Run with monitoring
monitor:
	docker-compose -f docker-compose.yml --profile monitoring up -d

# Database migrations
migrate:
	alembic upgrade head

# Create migration
migration:
	alembic revision --autogenerate -m "$(msg)"

# Backup database
backup:
	docker-compose exec postgres pg_dump -U gamma_user gamma_db > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Restore database
restore:
	docker-compose exec -T postgres psql -U gamma_user gamma_db < $(file)

# View logs
logs:
	docker-compose logs -f gamma_app

# Shell access
shell:
	docker-compose exec gamma_app /bin/bash

# Health check
health:
	curl -f http://localhost:8030/health || exit 1

# API documentation
docs:
	@echo "API Documentation: http://localhost:8030/docs"
	@echo "ReDoc: http://localhost:8030/redoc"

# Metrics
metrics:
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000"

# Full setup
setup: install
	@echo "Setting up Gamma App..."
	@echo "1. Copy env.example to .env and configure"
	@echo "2. Run 'make dev' to start development server"
	@echo "3. Visit http://localhost:8000/docs for API documentation"




























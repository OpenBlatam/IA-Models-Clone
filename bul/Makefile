# BUL Enhanced API - Makefile
# ===========================
# Comprehensive build, test, and deployment automation

.PHONY: help install dev test lint format security deploy clean docker-build docker-run docker-stop

# Default target
help: ## Show this help message
	@echo "BUL Enhanced API - Available Commands:"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development Setup
install: ## Install all dependencies
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

install-prod: ## Install production dependencies
	@echo "Installing production dependencies..."
	pip install -r requirements-prod.txt

dev: ## Run development server
	@echo "Starting development server..."
	python main.py

dev-reload: ## Run development server with auto-reload
	@echo "Starting development server with auto-reload..."
	uvicorn api.enhanced_api:app --host 0.0.0.0 --port 8000 --reload

# Testing
test: ## Run all tests
	@echo "Running all tests..."
	pytest tests/ -v --cov=api --cov=core --cov=agents --cov=database --cov=security --cov=utils

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	pytest tests/ -v -m "unit" --cov=api --cov=core --cov=agents --cov=database --cov=security --cov=utils

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	pytest tests/ -v -m "integration" --cov=api --cov=core --cov=agents --cov=database --cov=security --cov=utils

test-performance: ## Run performance tests only
	@echo "Running performance tests..."
	pytest tests/ -v -m "performance" --cov=api --cov=core --cov=agents --cov=database --cov=security --cov=utils

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=api --cov=core --cov=agents --cov=database --cov=security --cov=utils --cov-report=html --cov-report=term

test-watch: ## Run tests in watch mode
	@echo "Running tests in watch mode..."
	pytest-watch tests/ -v

# Code Quality
lint: ## Run linting
	@echo "Running linting..."
	flake8 api/ core/ agents/ database/ security/ utils/ tests/
	pylint api/ core/ agents/ database/ security/ utils/ tests/
	mypy api/ core/ agents/ database/ security/ utils/

format: ## Format code
	@echo "Formatting code..."
	black api/ core/ agents/ database/ security/ utils/ tests/
	isort api/ core/ agents/ database/ security/ utils/ tests/

format-check: ## Check code formatting
	@echo "Checking code formatting..."
	black --check api/ core/ agents/ database/ security/ utils/ tests/
	isort --check-only api/ core/ agents/ database/ security/ utils/ tests/

# Security
security: ## Run security checks
	@echo "Running security checks..."
	bandit -r api/ core/ agents/ database/ security/ utils/
	safety check
	pip-audit

security-scan: ## Run comprehensive security scan
	@echo "Running comprehensive security scan..."
	bandit -r api/ core/ agents/ database/ security/ utils/ -f json -o security-report.json
	safety check --json --output security-dependencies.json
	pip-audit --format=json --output security-audit.json

# Database
db-init: ## Initialize database
	@echo "Initializing database..."
	python -c "from database.enhanced_database import create_tables; import asyncio; asyncio.run(create_tables())"

db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	alembic upgrade head

db-migrate-create: ## Create new migration
	@echo "Creating new migration..."
	@read -p "Enter migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

db-reset: ## Reset database
	@echo "Resetting database..."
	python -c "from database.enhanced_database import drop_tables, create_tables; import asyncio; asyncio.run(drop_tables()); asyncio.run(create_tables())"

# Docker
docker-build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t bul-api:latest .

docker-run: ## Run Docker container
	@echo "Running Docker container..."
	docker-compose up -d

docker-stop: ## Stop Docker containers
	@echo "Stopping Docker containers..."
	docker-compose down

docker-logs: ## Show Docker logs
	@echo "Showing Docker logs..."
	docker-compose logs -f

docker-shell: ## Open shell in Docker container
	@echo "Opening shell in Docker container..."
	docker-compose exec api bash

# Production
prod-build: ## Build production image
	@echo "Building production image..."
	docker build -f Dockerfile.prod -t bul-api:prod .

prod-run: ## Run production container
	@echo "Running production container..."
	docker run -d --name bul-api-prod -p 8000:8000 bul-api:prod

prod-stop: ## Stop production container
	@echo "Stopping production container..."
	docker stop bul-api-prod && docker rm bul-api-prod

# Deployment
deploy-dev: ## Deploy to development
	@echo "Deploying to development..."
	@echo "Development deployment not implemented yet"

deploy-staging: ## Deploy to staging
	@echo "Deploying to staging..."
	@echo "Staging deployment not implemented yet"

deploy-prod: ## Deploy to production
	@echo "Deploying to production..."
	@echo "Production deployment not implemented yet"

# Kubernetes
k8s-apply: ## Apply Kubernetes manifests
	@echo "Applying Kubernetes manifests..."
	kubectl apply -f k8s/

k8s-delete: ## Delete Kubernetes resources
	@echo "Deleting Kubernetes resources..."
	kubectl delete -f k8s/

k8s-logs: ## Show Kubernetes logs
	@echo "Showing Kubernetes logs..."
	kubectl logs -f deployment/bul-api

k8s-shell: ## Open shell in Kubernetes pod
	@echo "Opening shell in Kubernetes pod..."
	kubectl exec -it deployment/bul-api -- bash

# Monitoring
monitor: ## Start monitoring
	@echo "Starting monitoring..."
	@echo "Monitoring not implemented yet"

logs: ## Show application logs
	@echo "Showing application logs..."
	tail -f logs/app.log

logs-error: ## Show error logs
	@echo "Showing error logs..."
	tail -f logs/error.log

# Performance
benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	pytest tests/ -v -m "performance" --benchmark-only

load-test: ## Run load tests
	@echo "Running load tests..."
	locust -f tests/load_test.py --host=http://localhost:8000

stress-test: ## Run stress tests
	@echo "Running stress tests..."
	@echo "Stress testing not implemented yet"

# Backup
backup: ## Create backup
	@echo "Creating backup..."
	@echo "Backup not implemented yet"

restore: ## Restore from backup
	@echo "Restoring from backup..."
	@echo "Restore not implemented yet"

# Cleanup
clean: ## Clean up temporary files
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/

clean-docker: ## Clean up Docker resources
	@echo "Cleaning up Docker resources..."
	docker system prune -f
	docker volume prune -f

clean-all: clean clean-docker ## Clean up everything
	@echo "Cleaning up everything..."

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	mkdocs build

docs-serve: ## Serve documentation
	@echo "Serving documentation..."
	mkdocs serve

docs-deploy: ## Deploy documentation
	@echo "Deploying documentation..."
	mkdocs gh-deploy

# Health Checks
health: ## Check application health
	@echo "Checking application health..."
	curl -f http://localhost:8000/health || echo "Health check failed"

health-db: ## Check database health
	@echo "Checking database health..."
	@echo "Database health check not implemented yet"

health-redis: ## Check Redis health
	@echo "Checking Redis health..."
	@echo "Redis health check not implemented yet"

# Configuration
config-check: ## Check configuration
	@echo "Checking configuration..."
	python -c "from config import validate_config; validate_config()"

config-reload: ## Reload configuration
	@echo "Reloading configuration..."
	python -c "from config import reload_config; reload_config()"

# Environment
env-dev: ## Set development environment
	@echo "Setting development environment..."
	export ENVIRONMENT=development
	export DEBUG=true
	export LOG_LEVEL=DEBUG

env-prod: ## Set production environment
	@echo "Setting production environment..."
	export ENVIRONMENT=production
	export DEBUG=false
	export LOG_LEVEL=INFO

# SSL/TLS
ssl-generate: ## Generate SSL certificates
	@echo "Generating SSL certificates..."
	@echo "SSL certificate generation not implemented yet"

ssl-renew: ## Renew SSL certificates
	@echo "Renewing SSL certificates..."
	@echo "SSL certificate renewal not implemented yet"

# Backup and Recovery
backup-db: ## Backup database
	@echo "Backing up database..."
	pg_dump -h localhost -U bul_user bul_db > backup_$(shell date +%Y%m%d_%H%M%S).sql

restore-db: ## Restore database
	@echo "Restoring database..."
	@echo "Database restore not implemented yet"

# Maintenance
maintenance: ## Run maintenance tasks
	@echo "Running maintenance tasks..."
	@echo "Maintenance not implemented yet"

update: ## Update dependencies
	@echo "Updating dependencies..."
	pip install --upgrade -r requirements.txt
	pip install --upgrade -r requirements-dev.txt

# Development Tools
shell: ## Open Python shell
	@echo "Opening Python shell..."
	python -c "from api.enhanced_api import app; import asyncio; print('BUL API shell ready')"

debug: ## Start debug mode
	@echo "Starting debug mode..."
	python -m pdb main.py

profile: ## Profile application
	@echo "Profiling application..."
	python -m cProfile -o profile.stats main.py

# CI/CD
ci-test: ## Run CI tests
	@echo "Running CI tests..."
	pytest tests/ -v --cov=api --cov=core --cov=agents --cov=database --cov=security --cov=utils --cov-report=xml

ci-lint: ## Run CI linting
	@echo "Running CI linting..."
	flake8 api/ core/ agents/ database/ security/ utils/ tests/
	pylint api/ core/ agents/ database/ security/ utils/ tests/
	mypy api/ core/ agents/ database/ security/ utils/

ci-security: ## Run CI security checks
	@echo "Running CI security checks..."
	bandit -r api/ core/ agents/ database/ security/ utils/
	safety check

ci-build: ## Run CI build
	@echo "Running CI build..."
	docker build -t bul-api:ci .

ci-deploy: ## Run CI deployment
	@echo "Running CI deployment..."
	@echo "CI deployment not implemented yet"

# Quick Commands
quick-start: install dev ## Quick start development
	@echo "Quick start completed!"

quick-test: test lint format ## Quick test and lint
	@echo "Quick test and lint completed!"

quick-deploy: docker-build docker-run ## Quick deploy with Docker
	@echo "Quick deploy completed!"

# Help
help-all: ## Show all available commands
	@echo "All available commands:"
	@echo "======================"
	@make help

# Default target
.DEFAULT_GOAL := help
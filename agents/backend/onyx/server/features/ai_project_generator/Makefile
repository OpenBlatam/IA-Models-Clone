.PHONY: help install test lint format security build deploy clean ci cd

# Default target
help:
	@echo "AI Project Generator - Makefile Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make security     - Run security checks"
	@echo ""
	@echo "CI/CD:"
	@echo "  make ci           - Run CI pipeline locally"
	@echo "  make cd           - Run CD pipeline locally"
	@echo "  make build        - Build Docker image"
	@echo "  make deploy       - Deploy to AWS"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        - Clean temporary files"
	@echo ""
	@echo "Variables:"
	@echo "  ENVIRONMENT       - Deployment environment (default: staging)"
	@echo "  IMAGE_TAG         - Docker image tag (default: latest)"

ENVIRONMENT ?= staging
IMAGE_TAG ?= latest

# Development commands
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt 2>/dev/null || true
	pre-commit install

test:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v

lint:
	ruff check .
	mypy . --ignore-missing-imports || true
	pylint **/*.py --disable=all --enable=E,F || true

format:
	black .
	ruff check --fix .

security:
	bandit -r . -f json -o bandit-report.json || true
	safety check || true

# CI/CD commands
ci: lint security test
	@echo "✅ CI pipeline completed"

cd: build deploy
	@echo "✅ CD pipeline completed"

build:
	./scripts/ci_cd/build.sh docker

deploy:
	./scripts/ci_cd/deploy.sh deploy

# Utilities
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info
	rm -f .coverage coverage.xml htmlcov/
	rm -f bandit-report.json safety-report.json

# Pre-commit
pre-commit:
	pre-commit run --all-files

# Docker commands
docker-build:
	docker build -t ai-project-generator:$(IMAGE_TAG) .

docker-test:
	docker run --rm ai-project-generator:$(IMAGE_TAG) pytest tests/ -v

docker-run:
	docker run -p 8020:8020 ai-project-generator:$(IMAGE_TAG)

# AWS commands
aws-deploy:
	cd aws/terraform && terraform apply -var="environment=$(ENVIRONMENT)"
	cd aws/ansible && ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml -e "environment=$(ENVIRONMENT)"

aws-destroy:
	cd aws/terraform && terraform destroy -var="environment=$(ENVIRONMENT)"


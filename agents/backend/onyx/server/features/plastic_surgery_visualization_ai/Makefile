.PHONY: help install install-dev test lint format clean run setup validate-config generate-docs cleanup-storage

help:
	@echo "Available commands:"
	@echo "  make install         - Install production dependencies"
	@echo "  make install-dev     - Install development dependencies"
	@echo "  make setup           - Setup storage directories"
	@echo "  make test            - Run tests"
	@echo "  make lint            - Run linters"
	@echo "  make format          - Format code"
	@echo "  make clean           - Clean temporary files"
	@echo "  make run             - Run the server"
	@echo "  make validate-config - Validate configuration"
	@echo "  make generate-docs   - Generate API documentation"
	@echo "  make cleanup-storage - Cleanup old storage files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

setup:
	python scripts/setup_storage.py

test:
	pytest tests/ -v

lint:
	flake8 . --exclude=venv,env,.git
	mypy . --ignore-missing-imports

format:
	black .
	isort .

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

run:
	python main.py

check-deps:
	python scripts/check_dependencies.py

check-health:
	python scripts/check_health.py

validate-config:
	python scripts/validate_config.py

generate-docs:
	python scripts/generate_docs.py

cleanup-storage:
	python scripts/cleanup_storage.py --days 30


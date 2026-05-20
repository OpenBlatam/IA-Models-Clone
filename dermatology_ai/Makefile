# ============================================================================
# Dermatology AI - Makefile
# Convenient commands for development and deployment
# ============================================================================

.PHONY: help install install-dev install-opt install-min install-gpu install-docker update outdated check compile clean test lint format type-check docker-build docker-run docker-test docs serve

# Default target
help:
	@echo "Dermatology AI - Available Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  make install          - Install production dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make install-opt      - Install optimized dependencies"
	@echo "  make install-min      - Install minimal dependencies"
	@echo "  make install-gpu      - Install GPU-enabled dependencies"
	@echo "  make install-docker   - Install Docker-optimized dependencies"
	@echo ""
	@echo "Dependency Management:"
	@echo "  make update           - Update all dependencies"
	@echo "  make outdated         - Show outdated packages"
	@echo "  make check            - Check for vulnerabilities"
	@echo "  make compile          - Generate requirements-lock.txt"
	@echo "  make clean            - Clean pip cache"
	@echo ""
	@echo "Development:"
	@echo "  make test             - Run tests"
	@echo "  make test-cov         - Run tests with coverage"
	@echo "  make lint             - Run linters"
	@echo "  make format           - Format code"
	@echo "  make type-check       - Run type checker"
	@echo "  make docs             - Generate documentation"
	@echo "  make serve            - Start development server"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"
	@echo "  make docker-test      - Run tests in Docker"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

install-opt:
	pip install -r requirements-optimized.txt

install-min:
	pip install -r requirements-minimal.txt

install-gpu:
	pip install -r requirements-gpu.txt

install-docker:
	pip install -r requirements-docker.txt

# Dependency Management
update:
	pip install --upgrade -r requirements.txt

outdated:
	pip list --outdated

check:
	@if command -v safety >/dev/null 2>&1; then \
		safety check -r requirements.txt; \
	else \
		echo "Installing safety..."; \
		pip install safety; \
		safety check -r requirements.txt; \
	fi
	@if command -v pip-audit >/dev/null 2>&1; then \
		pip-audit -r requirements.txt; \
	fi

compile:
	@if command -v pip-compile >/dev/null 2>&1; then \
		pip-compile requirements.txt -o requirements-lock.txt; \
	else \
		echo "Installing pip-tools..."; \
		pip install pip-tools; \
		pip-compile requirements.txt -o requirements-lock.txt; \
	fi

clean:
	pip cache purge
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true

# Development
test:
	pytest

test-cov:
	pytest --cov=. --cov-report=html --cov-report=term-missing

lint:
	ruff check .
	bandit -r . -f json -o bandit-report.json || true

format:
	black .
	ruff check --fix .

type-check:
	mypy .

docs:
	@if command -v mkdocs >/dev/null 2>&1; then \
		mkdocs build; \
	else \
		echo "Installing mkdocs..."; \
		pip install mkdocs mkdocs-material; \
		mkdocs build; \
	fi

serve:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run API with debugging
run-api:
	bash scripts/run-api.sh || python scripts/run-api-debug.py

run-api-debug:
	python scripts/run-api-debug.py

run-api-windows:
	scripts\run-api.bat

# Docker
docker-build:
	docker build -t dermatology-ai:latest .

docker-run:
	docker run -p 8000:8000 dermatology-ai:latest

docker-test:
	docker build -t dermatology-ai:test -f Dockerfile.test .
	docker run dermatology-ai:test

# Quick checks
quick-check: lint type-check test

# Full check before commit
pre-commit: format lint type-check test

# Validate requirements
validate-reqs:
	python scripts/validate-requirements.py requirements.txt

# Run all dependency checks
check-all:
	./scripts/all-checks.sh

# Benchmark installation
benchmark-install:
	./scripts/benchmark-install.sh

# Dependency tree
dep-tree:
	python scripts/dependency-tree.py requirements.txt

# Monitor dependencies
monitor:
	./scripts/monitor-dependencies.sh

# Backup requirements
backup:
	./scripts/backup-requirements.sh

# Restore requirements
restore:
	./scripts/restore-requirements.sh

# Dashboard
dashboard:
	python scripts/dependency-dashboard.py

# Cleanup
cleanup:
	./scripts/cleanup-requirements.sh

# Health score
health:
	python scripts/requirements-health-score.py

# Statistics
stats:
	./scripts/requirements-stats.sh

# Diff between files
diff:
	@echo "Usage: python scripts/requirements-diff.py <file1> <file2>"
	@echo "Example: python scripts/requirements-diff.py requirements.txt requirements-optimized.txt"

# Audit
audit:
	./scripts/requirements-audit.sh

# Sync
sync:
	./scripts/requirements-sync.sh

# Compatibility check
compat:
	python scripts/requirements-check-compatibility.py

# Export
export:
	./scripts/requirements-export.sh

# Notify
notify:
	./scripts/requirements-notify.sh

# Dependency graph
graph:
	python scripts/requirements-deps-graph.py

# Version check
version-check:
	./scripts/requirements-version-check.sh

# Auto-fix
auto-fix:
	./scripts/requirements-auto-fix.sh

# Integration test
integration-test:
	./scripts/requirements-integration-test.sh

# Conflict detection
conflicts:
	python scripts/requirements-dependency-conflicts.py

# Size analysis
size:
	./scripts/requirements-size-analysis.sh

# Quick scan
quick-scan:
	./scripts/requirements-quick-scan.sh

# Master check (all checks)
master-check:
	./scripts/requirements-master-check.sh

# ============================================================================
# Refactored Commands (using organized structure)
# ============================================================================

# Analysis commands
analyze:
	python scripts/requirements/analysis/analyze-dependencies.py || python scripts/analyze-dependencies.py

visualize:
	python scripts/requirements/analysis/visualize-dependencies.py || python scripts/visualize-dependencies.py

# Validation commands (aliases to existing)
validate-reqs-refactored:
	python scripts/requirements/validation/validate-requirements.py requirements.txt || python scripts/validate-requirements.py requirements.txt

# Run organized scripts
run-req:
	@echo "Usage: make run-req <category> <script>"
	@echo "Example: make run-req analysis analyze-dependencies.py"
	@bash scripts/requirements/run.sh $(ARGS)

# Migrate scripts to new structure
migrate-scripts:
	bash scripts/requirements/migrate-scripts.sh

# Refactoring commands
refactor-docs:
	bash scripts/refactor/move-docs.sh

refactor-config:
	bash scripts/refactor/organize-config.sh

refactor-all: refactor-docs refactor-config
	@echo "Refactoring completed"

# Analysis and checking
analyze-structure:
	python scripts/refactor/analyze-structure.py

check-refactoring:
	bash scripts/refactor/check-refactoring.sh

create-docs-index:
	python scripts/refactor/create-index.py

# Organize code
organize-services:
	bash scripts/refactor/organize-services.sh

organize-utils:
	bash scripts/refactor/organize-utils.sh

create-import-map:
	python scripts/refactor/create-import-map.py

organize-code: organize-services organize-utils create-import-map
	@echo "Code organization completed"

# Update and validate
update-imports:
	python scripts/refactor/update-imports.py

update-imports-dry-run:
	python scripts/refactor/update-imports.py --dry-run

validate-refactoring:
	bash scripts/refactor/validate-refactoring.sh

generate-report:
	python scripts/refactor/generate-refactoring-report.py

# Complete refactoring workflow
refactor-complete: refactor-all organize-code validate-refactoring generate-report
	@echo "Complete refactoring workflow finished"

# Monitoring
monitor-refactoring:
	bash scripts/refactor/monitor-refactoring.sh

# Cleanup
cleanup-old-files:
	bash scripts/refactor/cleanup-old-files.sh

# Visualization and export
visualize-structure:
	python scripts/refactor/visualize-structure.py

compare-structure:
	bash scripts/refactor/compare-structure.sh

export-structure:
	python scripts/refactor/export-structure.py


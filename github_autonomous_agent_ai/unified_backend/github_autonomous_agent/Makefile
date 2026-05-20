# ============================================================================
# GitHub Autonomous Agent - Makefile
# ============================================================================
# Comandos útiles para desarrollo
# ============================================================================

.PHONY: help install install-dev install-prod setup clean test lint format type-check run run-dev docker-build docker-up docker-down migrate

# Variables
PYTHON := python3
PIP := pip
VENV := venv
PYTEST := pytest

help: ## Mostrar esta ayuda
	@echo "Comandos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instalar dependencias base
	$(PIP) install -r requirements.txt

install-dev: install ## Instalar dependencias de desarrollo
	$(PIP) install -r requirements-dev.txt

install-prod: install ## Instalar dependencias de producción
	$(PIP) install -r requirements-prod.txt

setup: ## Setup completo (crear venv e instalar)
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
		echo "✅ Entorno virtual creado"; \
	fi
	@echo "📦 Instalando dependencias..."
	@$(MAKE) install-dev

clean: ## Limpiar archivos temporales
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info
	@echo "✅ Limpieza completada"

test: ## Ejecutar tests
	$(PYTEST) -v

test-cov: ## Ejecutar tests con coverage
	$(PYTEST) --cov=. --cov-report=html --cov-report=term

lint: ## Ejecutar linters
	ruff check .
	ruff format --check .

format: ## Formatear código
	ruff format .
	black . 2>/dev/null || true

type-check: ## Verificar tipos
	mypy . --ignore-missing-imports || true

check-deps: ## Verificar dependencias
	$(PYTHON) scripts/check-dependencies.py

validate-env: ## Validar archivo .env
	$(PYTHON) scripts/validate-env.py

update-deps: ## Actualizar dependencias (dry-run)
	$(PYTHON) scripts/update-dependencies.py --dry-run

update-deps-force: ## Actualizar dependencias (real)
	$(PYTHON) scripts/update-dependencies.py

security-check: ## Verificar seguridad de dependencias
	$(PYTHON) scripts/security-check.py

requirements-lock: ## Generar requirements-lock.txt
	$(PYTHON) scripts/requirements-lock.py

run: ## Ejecutar aplicación
	$(PYTHON) main.py

run-dev: ## Ejecutar en modo desarrollo
	uvicorn main:app --reload --host 0.0.0.0 --port 8030

run-prod: ## Ejecutar en modo producción
	uvicorn main:app --host 0.0.0.0 --port 8030 --workers 4

docker-build: ## Construir imagen Docker
	docker build -t github-autonomous-agent:latest .

docker-up: ## Iniciar servicios con Docker Compose
	docker-compose up -d

docker-down: ## Detener servicios de Docker Compose
	docker-compose down

docker-logs: ## Ver logs de Docker Compose
	docker-compose logs -f

migrate: ## Ejecutar migraciones
	alembic upgrade head

migrate-create: ## Crear nueva migración
	@read -p "Nombre de la migración: " name; \
	alembic revision --autogenerate -m "$$name"

migrate-downgrade: ## Revertir última migración
	alembic downgrade -1

migrate-history: ## Ver historial de migraciones
	alembic history

start-services: ## Iniciar servicios (Redis, etc.)
	@bash scripts/start-services.sh

health-check: ## Verificar estado de todos los servicios
	$(PYTHON) scripts/health-check.py

backup: ## Crear backup de base de datos y archivos
	$(PYTHON) scripts/backup.py

cleanup: ## Limpiar archivos temporales (dry-run)
	$(PYTHON) scripts/cleanup.py

cleanup-force: ## Limpiar archivos temporales (real)
	$(PYTHON) scripts/cleanup.py --force

generate-secret: ## Generar SECRET_KEY seguro
	$(PYTHON) scripts/generate-secret.py --type key

init-project: ## Inicializar estructura del proyecto
	$(PYTHON) scripts/init-project.py

check: lint type-check test ## Ejecutar todas las verificaciones

check-all: check-deps validate-env security-check check ## Ejecutar todas las verificaciones incluyendo seguridad

ci: clean install-dev check-all ## Ejecutar pipeline CI localmente

docker-rebuild: ## Rebuild y restart Docker Compose
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d

docker-restart: ## Restart servicios Docker Compose
	docker-compose restart

docker-ps: ## Ver estado de contenedores Docker Compose
	docker-compose ps

docker-exec: ## Ejecutar comando en contenedor (uso: make docker-exec CMD="bash")
	docker-compose exec app $(CMD)

logs: ## Ver logs de la aplicación
	tail -f storage/logs/app.log

logs-docker: ## Ver logs de Docker Compose
	docker-compose logs -f app

version: ## Mostrar versión del proyecto
	@python -c "import tomli; data = tomli.load(open('pyproject.toml', 'rb')); print(data['project']['version'])" 2>/dev/null || echo "1.0.0"

info: ## Mostrar información del proyecto
	@echo "📦 GitHub Autonomous Agent"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Python: $$(python --version 2>/dev/null || echo 'No disponible')"
	@echo "Pip: $$(pip --version 2>/dev/null || echo 'No disponible')"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'No disponible')"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📚 Documentación: Ver README.md"
	@echo "🚀 Quick Start: make setup && make run-dev"


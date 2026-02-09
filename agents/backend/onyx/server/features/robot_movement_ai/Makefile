# Makefile para Robot Movement AI v2.0
# Simplifica comandos comunes de desarrollo y deployment

.PHONY: help setup test lint format clean docker-build docker-up docker-down deploy

# Variables
PYTHON := python3
PIP := pip3
VENV := venv
PYTEST := pytest

help: ## Mostrar esta ayuda
	@echo "Robot Movement AI v2.0 - Comandos Disponibles"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Configurar entorno de desarrollo
	@echo "🚀 Configurando entorno de desarrollo..."
	bash scripts/dev_setup.sh

install: ## Instalar dependencias
	@echo "📦 Instalando dependencias..."
	$(PIP) install -r requirements.txt

test: ## Ejecutar tests
	@echo "🧪 Ejecutando tests..."
	bash scripts/run_tests.sh

test-coverage: ## Ejecutar tests con cobertura
	@echo "🧪 Ejecutando tests con cobertura..."
	$(PYTEST) tests/ -v --cov=core --cov=api --cov-report=html --cov-report=term

test-integration: ## Ejecutar tests de integración
	@echo "🔗 Ejecutando tests de integración..."
	$(PYTEST) tests/integration/ -v

lint: ## Ejecutar linters
	@echo "🔍 Ejecutando linters..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	pylint core/ api/ --exit-zero || true

format: ## Formatear código con Black
	@echo "✨ Formateando código..."
	black .

format-check: ## Verificar formato sin modificar
	@echo "🔍 Verificando formato..."
	black --check .

type-check: ## Verificar tipos con MyPy
	@echo "🔍 Verificando tipos..."
	mypy core/ api/ --ignore-missing-imports || true

security: ## Ejecutar análisis de seguridad
	@echo "🔒 Ejecutando análisis de seguridad..."
	bandit -r core/ api/ -f json -o bandit-report.json || true
	safety check || true

clean: ## Limpiar archivos temporales
	@echo "🧹 Limpiando archivos temporales..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

docker-build: ## Construir imagen Docker
	@echo "🐳 Construyendo imagen Docker..."
	docker build -t robot-movement-ai:latest .

docker-up: ## Iniciar servicios con Docker Compose
	@echo "🚀 Iniciando servicios..."
	docker-compose up -d

docker-down: ## Detener servicios con Docker Compose
	@echo "🛑 Deteniendo servicios..."
	docker-compose down

docker-logs: ## Ver logs de Docker Compose
	@echo "📋 Mostrando logs..."
	docker-compose logs -f

docker-restart: ## Reiniciar servicios
	@echo "🔄 Reiniciando servicios..."
	docker-compose restart

docker-clean: ## Limpiar contenedores e imágenes
	@echo "🧹 Limpiando Docker..."
	docker-compose down -v
	docker system prune -f

deploy: ## Deployar aplicación
	@echo "🚀 Deployando aplicación..."
	bash scripts/deploy.sh

deploy-staging: ## Deployar a staging
	@echo "🚀 Deployando a staging..."
	bash scripts/deploy.sh staging

deploy-production: ## Deployar a producción
	@echo "🚀 Deployando a producción..."
	bash scripts/deploy.sh production

run: ## Ejecutar aplicación localmente
	@echo "▶️  Ejecutando aplicación..."
	$(PYTHON) -m robot_movement_ai.main

run-dev: ## Ejecutar en modo desarrollo
	@echo "▶️  Ejecutando en modo desarrollo..."
	DEBUG=true $(PYTHON) -m robot_movement_ai.main

health: ## Verificar health check
	@echo "🏥 Verificando health check..."
	curl -f http://localhost:8010/health || echo "❌ Health check falló"

metrics: ## Ver métricas
	@echo "📊 Mostrando métricas..."
	curl http://localhost:8010/health/metrics

db-migrate: ## Ejecutar migraciones de base de datos
	@echo "🗄️  Ejecutando migraciones..."
	# Agregar comandos de migración aquí

db-backup: ## Crear backup de base de datos
	@echo "💾 Creando backup..."
	@mkdir -p backups
	@if [ -f db/robots.db ]; then \
		cp db/robots.db backups/backup_$$(date +%Y%m%d_%H%M%S).db; \
		echo "✅ Backup creado"; \
	else \
		echo "⚠️  Base de datos no encontrada"; \
	fi

docs: ## Generar documentación
	@echo "📚 Generando documentación..."
	@echo "Documentación disponible en:"
	@echo "  - START_HERE.md"
	@echo "  - MASTER_ARCHITECTURE_GUIDE.md"
	@echo "  - DEPLOYMENT_GUIDE.md"

all: clean setup test lint format ## Ejecutar todo: limpiar, setup, tests, lint, format

ci: test lint format-check type-check security ## Ejecutar pipeline CI completo





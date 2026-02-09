# Makefile para TruthGPT Test Suite
# Facilita la ejecución de comandos comunes

.PHONY: help test test-unit test-integration test-all test-analyzers test-coverage test-html clean lint format

# Variables
PYTHON := python
PYTEST := pytest
COVERAGE := coverage

# Colores para output
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Mostrar esta ayuda
	@echo "$(GREEN)TruthGPT Test Suite - Comandos disponibles:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

test: ## Ejecutar todos los tests
	@echo "$(GREEN)Ejecutando todos los tests...$(NC)"
	$(PYTHON) run_tests.py all

test-unit: ## Ejecutar solo tests unitarios
	@echo "$(GREEN)Ejecutando tests unitarios...$(NC)"
	$(PYTHON) run_tests.py unit -v

test-integration: ## Ejecutar solo tests de integración
	@echo "$(GREEN)Ejecutando tests de integración...$(NC)"
	$(PYTHON) run_tests.py integration -v

test-all: ## Ejecutar todos los tests con verbose
	@echo "$(GREEN)Ejecutando todos los tests (verbose)...$(NC)"
	$(PYTEST) core/ -v

test-analyzers: ## Ejecutar todos los analizadores
	@echo "$(GREEN)Ejecutando tests de analizadores...$(NC)"
	$(PYTHON) run_tests.py analyzers -v

test-performance: ## Ejecutar tests de rendimiento
	@echo "$(GREEN)Ejecutando tests de rendimiento...$(NC)"
	$(PYTHON) run_tests.py performance -v

test-quality: ## Ejecutar tests de calidad
	@echo "$(GREEN)Ejecutando tests de calidad...$(NC)"
	$(PYTHON) run_tests.py quality -v

test-coverage: ## Ejecutar tests con cobertura
	@echo "$(GREEN)Ejecutando tests con cobertura...$(NC)"
	$(PYTEST) core/ --cov=.. --cov-report=term-missing --cov-report=html

test-html: ## Generar reporte HTML de tests
	@echo "$(GREEN)Generando reporte HTML...$(NC)"
	$(PYTEST) core/ --html=report.html --self-contained-html

test-fast: ## Ejecutar tests rápidos (excluir slow)
	@echo "$(GREEN)Ejecutando tests rápidos...$(NC)"
	$(PYTEST) core/ -v -m "not slow"

test-slow: ## Ejecutar solo tests lentos
	@echo "$(GREEN)Ejecutando tests lentos...$(NC)"
	$(PYTEST) core/ -v -m "slow"

lint: ## Ejecutar linter
	@echo "$(GREEN)Ejecutando linter...$(NC)"
	@if command -v flake8 > /dev/null; then \
		flake8 core/ analyzers/ systems/ --max-line-length=120 --ignore=E501,W503; \
	else \
		echo "$(YELLOW)flake8 no está instalado. Instala con: pip install flake8$(NC)"; \
	fi

format: ## Formatear código con black
	@echo "$(GREEN)Formateando código...$(NC)"
	@if command -v black > /dev/null; then \
		black core/ analyzers/ systems/ reporters/ exporters/ utilities/; \
	else \
		echo "$(YELLOW)black no está instalado. Instala con: pip install black$(NC)"; \
	fi

format-check: ## Verificar formato sin modificar
	@echo "$(GREEN)Verificando formato...$(NC)"
	@if command -v black > /dev/null; then \
		black --check core/ analyzers/ systems/ reporters/ exporters/ utilities/; \
	else \
		echo "$(YELLOW)black no está instalado. Instala con: pip install black$(NC)"; \
	fi

validate: ## Validar estructura de tests
	@echo "$(GREEN)Validando estructura...$(NC)"
	$(PYTHON) validate_structure.py

clean: ## Limpiar archivos temporales
	@echo "$(GREEN)Limpiando archivos temporales...$(NC)"
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	find . -type f -name "report.html" -delete
	@echo "$(GREEN)¡Limpieza completada!$(NC)"

install: ## Instalar dependencias de tests
	@echo "$(GREEN)Instalando dependencias...$(NC)"
	pip install -r requirements-test.txt

stats: ## Mostrar estadísticas de tests
	@echo "$(GREEN)Estadísticas de tests:$(NC)"
	@echo "  Tests unitarios: $$(find core/unit -name 'test_*.py' | wc -l)"
	@echo "  Tests integración: $$(find core/integration -name 'test_*.py' | wc -l)"
	@echo "  Analizadores: $$(find analyzers -name '*.py' -not -name '__init__.py' | wc -l)"
	@echo "  Sistemas: $$(find systems -name '*.py' -not -name '__init__.py' | wc -l)"
	@echo "  Total archivos Python: $$(find . -name '*.py' | wc -l)"

analyze: ## Analizar estructura de tests
	@echo "$(GREEN)Analizando tests...$(NC)"
	$(PYTHON) analyze_tests.py

benchmark: ## Ejecutar benchmarks de tests
	@echo "$(GREEN)Ejecutando benchmarks...$(NC)"
	$(PYTHON) benchmark_tests.py

report: ## Generar reporte completo
	@echo "$(GREEN)Generando reporte...$(NC)"
	$(PYTHON) generate_report.py --output test-results

docker-test: ## Ejecutar tests en Docker
	@echo "$(GREEN)Ejecutando tests en Docker...$(NC)"
	docker-compose -f docker-compose.test.yml up tests

docker-clean: ## Limpiar contenedores Docker de tests
	@echo "$(GREEN)Limpiando contenedores Docker...$(NC)"
	docker-compose -f docker-compose.test.yml down -v

monitor: ## Monitorear tests en tiempo real
	@echo "$(GREEN)Iniciando monitor...$(NC)"
	$(PYTHON) monitor_tests.py

visualize: ## Visualizar resultados
	@echo "$(GREEN)Generando visualización...$(NC)"
	@if [ -f "monitor_stats.json" ]; then \
		$(PYTHON) visualize_results.py --input monitor_stats.json --output dashboard.html; \
	else \
		echo "$(YELLOW)No se encontró monitor_stats.json. Ejecuta 'make monitor' primero.$(NC)"; \
	fi

debug: ## Debuggear tests
	@echo "$(GREEN)Debuggeando tests...$(NC)"
	$(PYTHON) debug_tests.py

compare: ## Comparar resultados (requiere 2 archivos)
	@echo "$(GREEN)Comparando resultados...$(NC)"
	@echo "$(YELLOW)Uso: make compare FILE1=archivo1.json FILE2=archivo2.json$(NC)"

profile: ## Profilear tests
	@echo "$(GREEN)Profileando tests...$(NC)"
	$(PYTHON) profile_tests.py

optimize: ## Optimizar tests
	@echo "$(GREEN)Analizando optimizaciones...$(NC)"
	$(PYTHON) optimize_tests.py

export: ## Exportar resultados
	@echo "$(GREEN)Exportando resultados...$(NC)"
	@echo "$(YELLOW)Uso: make export INPUT=archivo.json FORMAT=all$(NC)"

run-advanced: ## Ejecutar tests con características avanzadas
	@echo "$(GREEN)Ejecutando tests avanzados...$(NC)"
	$(PYTHON) test_runner_advanced.py --parallel --workers 4

notify: ## Enviar notificaciones de resultados
	@echo "$(GREEN)Enviando notificaciones...$(NC)"
	@if [ -f "monitor_stats.json" ]; then \
		$(PYTHON) notify_results.py --input monitor_stats.json; \
	else \
		echo "$(YELLOW)No se encontró monitor_stats.json$(NC)"; \
	fi

metrics: ## Recopilar métricas avanzadas
	@echo "$(GREEN)Recopilando métricas...$(NC)"
	$(PYTHON) metrics_collector.py

dashboard: ## Iniciar servidor de dashboard
	@echo "$(GREEN)Iniciando dashboard...$(NC)"
	@echo "$(YELLOW)Dashboard disponible en http://localhost:8080$(NC)"
	$(PYTHON) dashboard_server.py --port 8080 --data monitor_stats.json

list: ## Listar todos los tests disponibles
	@echo "$(GREEN)Tests disponibles:$(NC)"
	@find core -name "test_*.py" | sort

setup: ## Configurar entorno de tests
	@echo "$(GREEN)Configurando entorno...$(NC)"
	$(PYTHON) setup.py

version: ## Mostrar versión y estadísticas
	@echo "$(GREEN)TruthGPT Test Suite v2.0.0$(NC)"
	@echo ""
	@echo "📊 Estadísticas:"
	@echo "  Scripts: 20"
	@echo "  Comandos Make: 30+"
	@echo "  Documentación: 9 archivos"
	@echo "  Categorías: 11"
	@echo ""
	@echo "📚 Documentación:"
	@echo "  - README.md"
	@echo "  - QUICK_START.md"
	@echo "  - TOOLS.md"
	@echo "  - ADVANCED_FEATURES.md"
	@echo "  - INTEGRATIONS.md"
	@echo "  - EXECUTIVE_SUMMARY.md"
	@echo "  - CHANGELOG.md"

backup: ## Crear backup de resultados y configuración
	@echo "$(GREEN)Creando backup...$(NC)"
	$(PYTHON) backup_tests.py --all

backup-results: ## Backup solo de resultados
	@echo "$(GREEN)Creando backup de resultados...$(NC)"
	$(PYTHON) backup_tests.py --results

backup-config: ## Backup solo de configuración
	@echo "$(GREEN)Creando backup de configuración...$(NC)"
	$(PYTHON) backup_tests.py --config

backup-list: ## Listar backups disponibles
	@echo "$(GREEN)Listando backups...$(NC)"
	$(PYTHON) backup_tests.py --list

health: ## Health check completo del sistema
	@echo "$(GREEN)Ejecutando health check...$(NC)"
	$(PYTHON) health_check.py

cleanup: ## Limpieza completa de archivos temporales
	@echo "$(GREEN)Limpiando archivos temporales...$(NC)"
	$(PYTHON) cleanup_tests.py --all

cleanup-cache: ## Limpiar solo cachés
	@echo "$(GREEN)Limpiando cachés...$(NC)"
	$(PYTHON) cleanup_tests.py --cache

cleanup-results: ## Limpiar resultados antiguos
	@echo "$(GREEN)Limpiando resultados antiguos...$(NC)"
	$(PYTHON) cleanup_tests.py --results 7

stats-dashboard: ## Generar dashboard de estadísticas
	@echo "$(GREEN)Generando dashboard de estadísticas...$(NC)"
	$(PYTHON) stats_dashboard.py --output stats_dashboard.html

team-report: ## Generar reporte para el equipo
	@echo "$(GREEN)Generando reporte para el equipo...$(NC)"
	$(PYTHON) collaboration_tools.py --report

contrib-guide: ## Generar guía de contribución
	@echo "$(GREEN)Generando guía de contribución...$(NC)"
	$(PYTHON) collaboration_tools.py --guide --output CONTRIBUTING.md

migrate: ## Escanear archivos para migración
	@echo "$(GREEN)Escaneando archivos para migración...$(NC)"
	$(PYTHON) migrate_tests.py --scan

migrate-apply: ## Aplicar migraciones
	@echo "$(GREEN)Aplicando migraciones...$(NC)"
	$(PYTHON) migrate_tests.py --apply

list-analyzers: ## Listar todos los analizadores
	@echo "$(GREEN)Analizadores disponibles:$(NC)"
	@find analyzers -name "*.py" -not -name "__init__.py" | sort

all: ## Ejecutar todas las verificaciones
	@echo "$(GREEN)Ejecutando todas las verificaciones...$(NC)"
	@echo ""
	@echo "1. Health check..."
	@$(MAKE) health
	@echo ""
	@echo "2. Validación de estructura..."
	@$(MAKE) validate
	@echo ""
	@echo "3. Tests unitarios..."
	@$(MAKE) test-unit
	@echo ""
	@echo "✅ Todas las verificaciones completadas"

discover: ## Descubrir y catalogar todos los tests
	@echo "$(GREEN)Descubriendo tests...$(NC)"
	$(PYTHON) test_discovery.py

coverage-analysis: ## Análisis avanzado de cobertura
	@echo "$(GREEN)Analizando cobertura...$(NC)"
	$(PYTHON) test_coverage_analyzer.py

trends: ## Analizar tendencias de tests
	@echo "$(GREEN)Analizando tendencias...$(NC)"
	$(PYTHON) test_trend_analyzer.py

quality-gates: ## Ejecutar quality gates
	@echo "$(GREEN)Ejecutando quality gates...$(NC)"
	$(PYTHON) test_quality_gates.py

schedule: ## Programar ejecución de tests
	@echo "$(GREEN)Programando tests...$(NC)"
	@echo "$(YELLOW)Uso: python test_scheduler.py --daily 02:00 --category all$(NC)"

api: ## Iniciar API REST de resultados
	@echo "$(GREEN)Iniciando API REST...$(NC)"
	$(PYTHON) test_result_api.py --host 127.0.0.1 --port 5000

api-start: ## Iniciar API REST en background
	@echo "$(GREEN)Iniciando API REST en background...$(NC)"
	@$(PYTHON) test_result_api.py --host 127.0.0.1 --port 5000 &

impact: ## Analizar impacto de cambios en tests
	@echo "$(GREEN)Analizando impacto de cambios...$(NC)"
	$(PYTHON) test_impact_analyzer.py --generate-command

impact-report: ## Generar reporte de impacto
	@echo "$(GREEN)Generando reporte de impacto...$(NC)"
	$(PYTHON) test_impact_analyzer.py --output test_impact_report.json

cache-stats: ## Mostrar estadísticas de caché
	@echo "$(GREEN)Estadísticas de caché...$(NC)"
	$(PYTHON) test_cache_manager.py --stats

cache-clear: ## Limpiar caché de tests
	@echo "$(GREEN)Limpiando caché...$(NC)"
	$(PYTHON) test_cache_manager.py --clear

cache-invalidate: ## Invalidar caché de test específico
	@echo "$(GREEN)Invalidando caché...$(NC)"
	@echo "$(YELLOW)Uso: make cache-invalidate TEST=path/to/test.py$(NC)"

aggregate: ## Agregar resultados de múltiples fuentes
	@echo "$(GREEN)Agregando resultados...$(NC)"
	@echo "$(YELLOW)Uso: python result_aggregator.py source1.json source2.json$(NC)"

predict: ## Entrenar modelo de predicción de fallos
	@echo "$(GREEN)Entrenando modelo de predicción...$(NC)"
	$(PYTHON) test_failure_predictor.py --train

predict-failures: ## Predecir tests que pueden fallar
	@echo "$(GREEN)Predicting test failures...$(NC)"
	$(PYTHON) test_failure_predictor.py --predict

anomalies: ## Detectar anomalías en resultados
	@echo "$(GREEN)Detectando anomalías...$(NC)"
	$(PYTHON) test_anomaly_detector.py --detect

anomalies-report: ## Generar reporte de anomalías
	@echo "$(GREEN)Generando reporte de anomalías...$(NC)"
	$(PYTHON) test_anomaly_detector.py --report anomalies_report.txt

dependency-graph: ## Generar grafo de dependencias
	@echo "$(GREEN)Generando grafo de dependencias...$(NC)"
	$(PYTHON) test_dependency_graph.py --export-json dependency_graph.json --export-dot dependency_graph.dot

dependency-stats: ## Estadísticas del grafo de dependencias
	@echo "$(GREEN)Estadísticas de dependencias...$(NC)"
	$(PYTHON) test_dependency_graph.py --stats

optimize-suggestions: ## Generar sugerencias de optimización
	@echo "$(GREEN)Generando sugerencias de optimización...$(NC)"
	$(PYTHON) test_optimization_suggestions.py --analyze

optimize-report: ## Generar reporte de optimización
	@echo "$(GREEN)Generando reporte de optimización...$(NC)"
	$(PYTHON) test_optimization_suggestions.py --report optimization_report.txt

stream: ## Iniciar streamer de resultados en tiempo real
	@echo "$(GREEN)Iniciando streamer...$(NC)"
	@echo "$(YELLOW)Conectar a: ws://localhost:8765$(NC)"
	$(PYTHON) test_result_streamer.py --port 8765

stream-tests: ## Ejecutar tests y transmitir resultados
	@echo "$(GREEN)Ejecutando tests con streaming...$(NC)"
	$(PYTHON) test_result_streamer.py --run-tests --port 8765

db-enhanced: ## Usar base de datos mejorada
	@echo "$(GREEN)Base de datos mejorada disponible$(NC)"
	@echo "$(YELLOW)Uso: python test_result_database_enhanced.py --search <query>$(NC)"

compare-advanced: ## Comparación avanzada de resultados
	@echo "$(GREEN)Comparación avanzada...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_comparator_advanced.py run1.json run2.json --report$(NC)"

performance-regression: ## Detectar regresiones de rendimiento
	@echo "$(GREEN)Detectando regresiones de rendimiento...$(NC)"
	$(PYTHON) performance_regression_detector.py --detect

performance-report: ## Reporte de regresiones de rendimiento
	@echo "$(GREEN)Generando reporte de rendimiento...$(NC)"
	$(PYTHON) performance_regression_detector.py --report performance_regression_report.txt

plan-execution: ## Planificar ejecución de tests
	@echo "$(GREEN)Planificando ejecución...$(NC)"
	@echo "$(YELLOW)Uso: python test_execution_planner.py --plan test1.py test2.py --strategy balanced$(NC)"

optimize-execution: ## Optimizar selección de tests
	@echo "$(GREEN)Optimizando selección...$(NC)"
	@echo "$(YELLOW)Uso: python test_execution_planner.py --optimize test1.py test2.py --target-time 300$(NC)"

cluster-results: ## Agrupar resultados por patrones
	@echo "$(GREEN)Agrupando resultados...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_clusterer.py --results results.json --by-error$(NC)"

search-advanced: ## Búsqueda avanzada de resultados
	@echo "$(GREEN)Búsqueda avanzada...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_search_advanced.py --build-index --search <query>$(NC)"

alerts: ## Ver alertas recientes
	@echo "$(GREEN)Mostrando alertas...$(NC)"
	$(PYTHON) test_alert_system.py --recent 24

alerts-summary: ## Resumen de alertas
	@echo "$(GREEN)Generando resumen de alertas...$(NC)"
	$(PYTHON) test_alert_system.py --summary

visualize-advanced: ## Visualización avanzada
	@echo "$(GREEN)Generando visualizaciones...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_visualizer_advanced.py --results results.json --dashboard$(NC)"

custom-metrics: ## Sistema de métricas personalizadas
	@echo "$(GREEN)Métricas personalizadas...$(NC)"
	@echo "$(YELLOW)Uso: python custom_metrics_system.py --list$(NC)"

archive-results: ## Archivar resultados antiguos
	@echo "$(GREEN)Archivando resultados...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_archiver.py --archive 30$(NC)"

archive-list: ## Listar archivos archivados
	@echo "$(GREEN)Listando archivos archivados...$(NC)"
	$(PYTHON) test_result_archiver.py --list

archive-stats: ## Estadísticas de archivos archivados
	@echo "$(GREEN)Estadísticas de archivos...$(NC)"
	$(PYTHON) test_result_archiver.py --stats

export-advanced: ## Exportar resultados a múltiples formatos
	@echo "$(GREEN)Exportación avanzada...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_exporter_advanced.py --results results.json --format all$(NC)"

validate-results: ## Validar archivos de resultados
	@echo "$(GREEN)Validando resultados...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_validator.py --file results.json$(NC)"

validate-directory: ## Validar directorio de resultados
	@echo "$(GREEN)Validando directorio...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_validator.py --directory test_results/$(NC)"

integrations: ## Configurar integraciones externas
	@echo "$(GREEN)Integraciones disponibles$(NC)"
	@echo "$(YELLOW)Uso: python test_result_integrations.py --configure-jira URL EMAIL TOKEN PROJECT$(NC)"

encrypt-results: ## Encriptar resultados sensibles
	@echo "$(GREEN)Encriptación de resultados...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_encryption.py --encrypt results.json --password <pass>$(NC)"

decrypt-results: ## Desencriptar resultados
	@echo "$(GREEN)Desencriptación...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_encryption.py --decrypt results.enc --password <pass>$(NC)"

backup-advanced: ## Sistema de backup avanzado
	@echo "$(GREEN)Backup avanzado...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_backup_advanced.py --backup$(NC)"

backup-list: ## Listar backups
	@echo "$(GREEN)Listando backups...$(NC)"
	$(PYTHON) test_result_backup_advanced.py --list

backup-restore: ## Restaurar desde backup
	@echo "$(GREEN)Restaurando backup...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_backup_advanced.py --restore <backup_name>$(NC)"

migrate-results: ## Migrar resultados entre versiones
	@echo "$(GREEN)Migración de resultados...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_migrator.py --migrate-file results.json --to-version 2.0$(NC)"

deduplicate-results: ## Eliminar resultados duplicados
	@echo "$(GREEN)Deduplicación...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_deduplicator.py --deduplicate-dir test_results/$(NC)"

analytics-advanced: ## Analytics avanzado
	@echo "$(GREEN)Analytics avanzado...$(NC)"
	@echo "$(YELLOW)Uso: python test_result_analytics_advanced.py --results results.json --report$(NC)"


# TruthGPT Test Suite

## Descripción
Suite completa de tests para TruthGPT con 189+ tests organizados en una estructura modular y categorizada.

## Estructura Organizada

```
tests/
├── __init__.py
├── conftest.py              # Configuración pytest
├── README.md                # Este archivo
│
├── core/                    # Tests unitarios e integración
│   ├── test_core.py         # Tests de componentes core
│   ├── test_models.py       # Tests de modelos
│   ├── test_training.py     # Tests de entrenamiento
│   ├── test_inference.py    # Tests de inferencia
│   ├── test_integration.py  # Tests de integración
│   ├── test_performance.py # Tests de rendimiento
│   ├── test_security.py     # Tests de seguridad
│   └── ...                  # Otros tests core
│
├── analyzers/               # Módulos de análisis
│   ├── cost/                # Análisis de costos
│   ├── performance/         # Análisis de rendimiento
│   ├── quality/             # Análisis de calidad
│   ├── security/            # Análisis de seguridad
│   ├── compliance/          # Verificación de cumplimiento
│   ├── coverage/            # Análisis de cobertura
│   ├── dependency/          # Análisis de dependencias
│   ├── trend/               # Análisis de tendencias
│   ├── flakiness/           # Análisis de flakiness
│   ├── regression/          # Análisis de regresión
│   └── optimization/        # Análisis de optimización
│
├── systems/                 # Sistemas y servicios
│   ├── prediction_system_advanced.py
│   ├── metrics_system.py
│   ├── business_metrics.py
│   ├── anomaly_detector.py
│   ├── recommendation_engine.py
│   └── ...                  # Otros sistemas
│
├── reporters/               # Módulos de reportes
│   ├── executive_report.py
│   ├── executive_dashboard.py
│   ├── comprehensive_reporter.py
│   └── ...                  # Otros reportes
│
├── exporters/               # Utilidades de exportación
│   ├── junit_exporter.py
│   ├── pdf_exporter.py
│   ├── universal_exporter.py
│   └── ...                  # Otros exportadores
│
└── utilities/               # Utilidades
    ├── integration/         # Integraciones (Slack, dashboards, alerts)
    └── results/             # Procesamiento de resultados
```

## Organización por Categorías

### Core Tests (`core/`)
Tests unitarios e integración del sistema principal.
- `test_core.py` - Componentes core
- `test_models.py` - Gestión de modelos
- `test_training.py` - Sistema de entrenamiento
- `test_inference.py` - Motor de inferencia
- `test_integration.py` - Tests end-to-end
- `test_performance.py` - Rendimiento
- `test_security.py` - Seguridad

### Analyzers (`analyzers/`)
Módulos de análisis organizados por tipo:

#### Cost Analysis (`analyzers/cost/`)
- Análisis de costos de ejecución
- Cálculo de ROI
- Optimización de costos

#### Performance Analysis (`analyzers/performance/`)
- Análisis de rendimiento
- Detección de regresiones
- Optimización de recursos

#### Quality Analysis (`analyzers/quality/`)
- Análisis de calidad
- Quality gates
- Evaluación de calidad

#### Security Analysis (`analyzers/security/`)
- Análisis de seguridad
- Detección de vulnerabilidades

#### Compliance (`analyzers/compliance/`)
- Verificación de cumplimiento
- Auditorías

#### Coverage (`analyzers/coverage/`)
- Análisis de cobertura de tests

#### Dependency (`analyzers/dependency/`)
- Análisis de dependencias
- Visualización de dependencias

#### Trend Analysis (`analyzers/trend/`)
- Análisis de tendencias
- Predicción de tendencias
- Forecasting

#### Flakiness (`analyzers/flakiness/`)
- Detección de tests flaky

#### Regression (`analyzers/regression/`)
- Análisis de regresiones

#### Optimization (`analyzers/optimization/`)
- Análisis de optimización
- Impact analysis

### Systems (`systems/`)
Sistemas y servicios de alto nivel:
- Sistemas de predicción
- Sistemas de métricas
- Sistemas de recomendación
- Detección de anomalías

### Reporters (`reporters/`)
Módulos de generación de reportes:
- Reportes ejecutivos
- Dashboards
- Reportes comprehensivos

### Exporters (`exporters/`)
Utilidades de exportación:
- Exportación JUnit
- Exportación PDF
- Exportación universal

### Utilities (`utilities/`)
Utilidades organizadas en:
- `integration/` - Integraciones (Slack, dashboards, alerts)
- `results/` - Procesamiento de resultados

## Ejecutar Tests

### Usando el Script de Utilidad (`run_tests.py`)

El script `run_tests.py` facilita la ejecución de tests por categoría:

#### Todos los Tests
```bash
python run_tests.py all
```

#### Tests por Categoría
```bash
# Tests unitarios
python run_tests.py unit

# Tests de integración
python run_tests.py integration

# Todos los analizadores
python run_tests.py analyzers

# Analizador específico
python run_tests.py performance
python run_tests.py quality
python run_tests.py cost
```

#### Con Opciones Avanzadas
```bash
# Modo verbose
python run_tests.py unit -v

# Ejecutar tests que coincidan con un patrón
python run_tests.py performance -k "test_performance"

# Con cobertura
python run_tests.py all --coverage

# Con reporte HTML
python run_tests.py all --html

# Combinar opciones
python run_tests.py analyzers -v --coverage --html
```

#### Ver Ayuda
```bash
python run_tests.py --help
```

### Ejecución Directa con pytest

También puedes ejecutar tests directamente con pytest:

```bash
# Todos los tests
pytest core/ -v

# Tests unitarios
pytest core/unit/ -v

# Tests de integración
pytest core/integration/ -v

# Analizador específico
pytest analyzers/performance/ -v

# Sistema específico
pytest systems/ -v
```

## Utilidades

### Fixtures y Helpers (`core/fixtures/`)

#### test_utils.py
Funciones utilitarias compartidas:
- `create_test_model()` - Crear modelos de prueba
- `create_test_dataset()` - Crear datasets de prueba
- `create_test_tokenizer()` - Crear tokenizers de prueba
- `assert_model_valid()` - Validar modelos
- `TestTimer` - Medir tiempo de ejecución

#### test_helpers.py
Decoradores y helpers:
- `@retry_on_failure` - Reintentar tests
- `@skip_if_no_cuda` - Saltar si no hay CUDA
- `@performance_test` - Validar rendimiento
- `@memory_profiler` - Perfilar memoria

#### test_fixtures.py
Fixtures de pytest para configuración compartida

### Importar Utilidades

```python
# Desde tests core
from core.fixtures.test_utils import create_test_model, assert_model_valid
from core.fixtures.test_helpers import retry_on_failure, performance_test

# Desde analyzers
from analyzers.performance import PerformanceAnalyzer
from analyzers.quality import QualityAnalyzer

# Desde systems
from systems import PredictionSystem, MetricsSystem
```

## Estadísticas

- **Total de Tests**: 189+
- **Archivos de Test**: 150+
- **Categorías de Analyzers**: 11
- **Tests Core**: 40+
- **Sistemas**: 25+
- **Utilidades**: 30+
- **Cobertura**: Alta

### Distribución por Categoría

- **Core Tests**: 40+ archivos
  - Unit Tests: 6 archivos
  - Integration Tests: 2 archivos
  - Fixtures: 3 archivos
  - Otros: 29+ archivos
- **Analyzers**: 50+ archivos en 11 subcategorías
- **Systems**: 25+ archivos
- **Reporters**: 7 archivos
- **Exporters**: 5 archivos
- **Utilities**: 24+ archivos

## Mejores Prácticas

1. **Usar utilidades compartidas**: Usa `test_utils` para código común
2. **Decoradores útiles**: Aprovecha los decoradores de `test_helpers`
3. **Validaciones robustas**: Usa `assert_model_valid()` y similares
4. **Logging**: Usa `logger.info()` para mensajes informativos
5. **Cleanup**: Limpia recursos en `tearDown()` si es necesario

## Troubleshooting

### Import Errors
Asegúrate de ejecutar desde el directorio `TruthGPT-main`:
```bash
cd TruthGPT-main
python run_unified_tests.py
```

### Dependencies Missing
Instala dependencias:
```bash
pip install torch numpy psutil
```

### CUDA Tests Failing
Algunos tests requieren CUDA. Se saltan automáticamente si no está disponible.

## Herramientas Avanzadas

### Análisis de Tests
```bash
# Analizar estructura y generar estadísticas
python analyze_tests.py
make analyze
```

### Benchmarking
```bash
# Ejecutar benchmarks de rendimiento
python benchmark_tests.py --iterations 5
make benchmark
```

### Generación de Reportes
```bash
# Generar reporte completo con visualización
python generate_report.py --output test-results
make report
```

### Docker
```bash
# Ejecutar tests en contenedor Docker
docker-compose -f docker-compose.test.yml up tests
make docker-test

# Limpiar contenedores
make docker-clean
```

### CI/CD
- **GitHub Actions**: Configuración en `.github/workflows/tests.yml`
- Ejecuta tests automáticamente en push/PR
- Soporta múltiples OS y versiones de Python
- Genera reportes de cobertura automáticamente

### Características Avanzadas
- **Ejecución Paralela**: `test_runner_advanced.py --parallel`
- **Detección de Flaky Tests**: Automática con múltiples ejecuciones
- **Optimización Automática**: Análisis y sugerencias de mejora
- **Exportación Multi-formato**: JSON, CSV, XML, Markdown
- Ver `ADVANCED_FEATURES.md` para más detalles

## Navegación Rápida

### Buscar Archivos
- **Índice completo**: Ver `INDEX.md` para listado detallado de todos los archivos
- **Por funcionalidad**: Usa la estructura de carpetas para encontrar módulos relacionados
- **Por tipo**: 
  - Tests → `core/`
  - Analizadores → `analyzers/`
  - Sistemas → `systems/`

### Estructura de Carpetas Mejorada

```
tests/
├── core/
│   ├── unit/          # Tests unitarios
│   ├── integration/   # Tests de integración
│   └── fixtures/      # Fixtures y utilidades compartidas
├── analyzers/
│   ├── general/       # Analizadores generales
│   └── [categoría]/   # Analizadores por tipo
└── ...
```

## Setup Inicial

### Configuración Rápida

```bash
# 1. Instalar dependencias
pip install -r requirements-test.txt

# 2. Configurar entorno
python setup.py
# o
make setup

# 3. Validar estructura
python validate_structure.py
# o
make validate

# 4. Ejecutar tests
make test
```

### Verificar Instalación

```bash
# Ver versión y estadísticas
make version

# Ver todos los comandos disponibles
make help
```

## Contribuir

Al agregar nuevos tests:
1. **Ubicación correcta**: Coloca tests en `core/unit/` o `core/integration/` según corresponda
2. **Usa utilidades compartidas**: Importa desde `core/fixtures/test_utils.py`
3. **Sigue el formato**: Mantén consistencia con tests existentes
4. **Agrega logging**: Usa `logger.info()` para mensajes informativos
5. **Actualiza documentación**: Actualiza `README.md` e `INDEX.md` si agregas nuevas categorías
6. **Ejecuta tests**: Verifica que los tests pasen antes de commitear

### Guía de Organización

- **Nuevos tests unitarios** → `core/unit/`
- **Nuevos tests de integración** → `core/integration/`
- **Nuevos analizadores** → `analyzers/[categoría]/` o `analyzers/general/`
- **Nuevos sistemas** → `systems/`
- **Nuevos reportes** → `reporters/`
- **Nuevos exportadores** → `exporters/`
- **Nuevas utilidades** → `utilities/[tipo]/`

## Changelog

Ver `CHANGELOG.md` para historial completo de cambios.

## Mantenimiento

Ver `MAINTENANCE.md` para guía de mantenimiento, backups y limpieza.

## Resumen Ejecutivo

Ver `EXECUTIVE_SUMMARY.md` para resumen completo de características y capacidades.

## Comandos de Mantenimiento

```bash
make health          # Health check completo
make backup          # Backup de resultados y configuración
make cleanup         # Limpieza de archivos temporales
make setup           # Configurar entorno
make version         # Ver versión y estadísticas
```

## Guía Completa

Para una guía completa y exhaustiva de todas las características, ver `COMPLETE_GUIDE.md`.

## Comandos Principales

### Ver Todos los Comandos
```bash
make help            # Lista completa de comandos
make version         # Versión y estadísticas
```

### Ejecución Rápida
```bash
make test            # Todos los tests
make test-unit       # Tests unitarios
make all             # Todas las verificaciones
```

### Análisis y Reportes
```bash
make analyze         # Analizar estructura
make report          # Generar reportes
make stats-dashboard # Dashboard de estadísticas
make discover        # Descubrir todos los tests
```

### Monitoreo
```bash
make monitor         # Monitor en tiempo real
make dashboard       # Dashboard web
make health          # Health check
```

### Calidad y Tendencias
```bash
make quality-gates   # Ejecutar quality gates
make trends          # Analizar tendencias
make schedule        # Programar tests
```

### API y Acceso Programático
```bash
make api             # Iniciar API REST de resultados
make api-start       # Iniciar API en background
```

### Análisis de Impacto
```bash
make impact          # Analizar qué tests ejecutar según cambios
make impact-report   # Generar reporte de impacto
```

### Caché Inteligente
```bash
make cache-stats     # Ver estadísticas de caché
make cache-clear     # Limpiar caché completo
make cache-invalidate # Invalidar caché de test específico
```

### Agregación de Resultados
```bash
make aggregate       # Agregar resultados de múltiples fuentes/entornos
```

### Machine Learning y Predicción
```bash
make predict         # Entrenar modelo de predicción de fallos
make predict-failures # Predecir qué tests pueden fallar
```

### Detección de Anomalías
```bash
make anomalies       # Detectar anomalías en resultados
make anomalies-report # Generar reporte de anomalías
```

### Dependencias y Optimización
```bash
make dependency-graph # Generar grafo de dependencias
make dependency-stats  # Estadísticas de dependencias
make optimize-suggestions # Sugerencias de optimización
make optimize-report     # Reporte de optimización
```

### Streaming en Tiempo Real
```bash
make stream         # Iniciar streamer de resultados
make stream-tests   # Ejecutar tests con streaming
```

## Nuevas Características Avanzadas

### 🚀 API REST de Resultados
Acceso programático a resultados de tests mediante API REST:
- Consultar runs históricos
- Obtener estadísticas
- Buscar tests
- Comparar runs
- Ver historial de tests específicos

**Uso:**
```bash
# Iniciar API
make api

# Consultar desde otro proceso
curl http://localhost:5000/api/runs
curl http://localhost:5000/api/statistics
curl http://localhost:5000/api/tests/failing
```

### 🔍 Análisis de Impacto
Determina automáticamente qué tests ejecutar basándose en cambios de código:
- Analiza cambios en git
- Mapea archivos fuente a tests
- Genera comandos pytest optimizados
- Reduce tiempo de ejecución

**Uso:**
```bash
# Analizar cambios desde HEAD
make impact

# Comparar dos commits
python test_impact_analyzer.py --base HEAD~5 --target HEAD

# Generar reporte
make impact-report
```

### 💾 Caché Inteligente
Sistema de caché que evita ejecutar tests sin cambios:
- Detecta cambios en código y dependencias
- Cachea resultados válidos por 24h (configurable)
- Reduce tiempo de ejecución significativamente
- Invalida automáticamente cuando hay cambios

**Uso:**
```bash
# Ver estadísticas
make cache-stats

# Limpiar todo
make cache-clear

# Invalidar test específico
python test_cache_manager.py --invalidate core/test_models.py
```

### 📊 Agregación Multi-Entorno
Agrega resultados de múltiples ejecuciones y entornos:
- Combina resultados de dev, staging, prod
- Calcula estadísticas agregadas
- Identifica diferencias entre entornos
- Genera reportes comparativos

**Uso:**
```bash
# Agregar múltiples fuentes
python result_aggregator.py results/dev/*.json results/staging/*.json

# Con Make
make aggregate
```

### 🤖 Predicción de Fallos con ML
Sistema de machine learning que predice qué tests pueden fallar:
- Entrena modelo con historial de resultados
- Calcula probabilidad de fallo por test
- Prioriza tests de alto riesgo
- Reduce tiempo de ejecución ejecutando tests críticos primero

**Uso:**
```bash
# Entrenar modelo
make predict

# Predecir fallos
make predict-failures

# Obtener orden prioritario
python test_failure_predictor.py --prioritize test1.py test2.py test3.py
```

### 🔍 Detección de Anomalías
Detecta patrones anómalos en resultados de tests:
- Anomalías en duración de ejecución
- Picos de fallos inesperados
- Caídas en tasa de éxito
- Alertas automáticas

**Uso:**
```bash
# Detectar todas las anomalías
make anomalies

# Generar reporte
make anomalies-report
```

### 📊 Visualización de Dependencias
Crea y visualiza el grafo de dependencias entre tests y código:
- Mapea tests a archivos fuente
- Identifica qué tests afecta cada cambio
- Exporta en formato JSON y DOT (Graphviz)
- Estadísticas de cobertura de tests

**Uso:**
```bash
# Generar grafo
make dependency-graph

# Ver estadísticas
make dependency-stats

# Ver dependencias de un test
python test_dependency_graph.py --test core/test_models.py

# Ver tests afectados por un archivo
python test_dependency_graph.py --source models/trainer.py
```

### 💡 Sugerencias de Optimización
Analiza la suite de tests y proporciona sugerencias:
- Tests lentos que necesitan optimización
- Tests flaky que requieren atención
- Duplicados potenciales
- Problemas de organización
- Gaps de cobertura

**Uso:**
```bash
# Analizar suite
make optimize-suggestions

# Generar reporte
make optimize-report
```

### 📡 Streaming en Tiempo Real
Streaming de resultados de tests en tiempo real vía WebSocket:
- Resultados en tiempo real mientras se ejecutan tests
- Múltiples clientes pueden conectarse
- Útil para dashboards y monitoreo
- Buffer de últimos 100 resultados

**Uso:**
```bash
# Iniciar streamer
make stream

# Ejecutar tests y transmitir
make stream-tests

# Conectar desde cliente (JavaScript)
# const ws = new WebSocket('ws://localhost:8765');
# ws.onmessage = (event) => console.log(JSON.parse(event.data));
```

## Ver Todas las Características

Para ver todas las características disponibles, ver `ALL_FEATURES.md`.









# 🛠️ Herramientas y Utilidades - TruthGPT Test Suite

## 🛠️ Scripts Disponibles

### Herramientas Básicas

### 1. `run_tests.py` - Ejecutor de Tests
Ejecuta tests organizados por categoría.

```bash
python run_tests.py [categoria] [opciones]
python run_tests.py --help
```

**Categorías disponibles:**
- `all` - Todos los tests
- `unit` - Tests unitarios
- `integration` - Tests de integración
- `analyzers` - Todos los analizadores
- `performance`, `quality`, `cost`, etc. - Categorías específicas

### 2. `validate_structure.py` - Validador de Estructura
Valida que la estructura de tests esté correctamente organizada.

```bash
python validate_structure.py
make validate
```

**Verifica:**
- ✅ Estructura de directorios
- ✅ Archivos requeridos
- ✅ Organización de archivos Python
- ✅ Ubicación correcta de tests

### 3. `analyze_tests.py` - Analizador de Tests
Analiza la suite de tests y genera estadísticas detalladas.

```bash
python analyze_tests.py
python analyze_tests.py --output report.json
make analyze
```

**Genera:**
- 📊 Estadísticas de tests
- 📦 Análisis de imports
- 🎨 Uso de decoradores
- 📝 Listado de archivos de test

### 4. `benchmark_tests.py` - Benchmark de Tests
Ejecuta benchmarks de rendimiento para tests.

```bash
python benchmark_tests.py
python benchmark_tests.py --categories unit integration --iterations 5
make benchmark
```

**Mide:**
- ⏱️ Tiempo de ejecución
- 📈 Promedios y rangos
- 🏆 Ranking de categorías

### 5. `generate_report.py` - Generador de Reportes
Genera reportes completos con visualizaciones HTML.

```bash
python generate_report.py
python generate_report.py --output test-results
make report
```

**Genera:**
- 📊 Reporte JSON con datos
- 🌐 Reporte HTML visual
- 📈 Estadísticas de cobertura

## Comandos Make

### Tests
```bash
make test              # Ejecutar todos los tests
make test-unit         # Solo tests unitarios
make test-integration  # Solo tests de integración
make test-coverage     # Con cobertura
make test-html         # Generar reporte HTML
```

### Análisis
```bash
make analyze           # Analizar estructura
make benchmark         # Ejecutar benchmarks
make report            # Generar reporte
make stats             # Mostrar estadísticas
```

### Calidad de Código
```bash
make lint              # Ejecutar linter
make format             # Formatear código
make format-check       # Verificar formato
```

### Docker
```bash
make docker-test       # Ejecutar tests en Docker
make docker-clean      # Limpiar contenedores
```

### Utilidades
```bash
make clean             # Limpiar archivos temporales
make install           # Instalar dependencias
make list              # Listar tests
make help              # Ver todos los comandos
```

## Docker

### Construir Imagen
```bash
docker build -f Dockerfile.test -t truthgpt-tests .
```

### Ejecutar Tests
```bash
# Todos los tests
docker run --rm truthgpt-tests

# Tests específicos
docker run --rm truthgpt-tests pytest tests/core/unit/ -v

# Con volúmenes
docker run --rm -v $(pwd):/app truthgpt-tests
```

### Docker Compose
```bash
# Ejecutar todos los tests
docker-compose -f docker-compose.test.yml up tests

# Solo tests unitarios
docker-compose -f docker-compose.test.yml up tests-unit

# Con cobertura
docker-compose -f docker-compose.test.yml up tests-coverage

# Limpiar
docker-compose -f docker-compose.test.yml down
```

## CI/CD

### GitHub Actions
La configuración en `.github/workflows/tests.yml` incluye:

- ✅ Tests en múltiples OS (Linux, Windows, macOS)
- ✅ Múltiples versiones de Python (3.8-3.11)
- ✅ Validación de estructura
- ✅ Linting y formateo
- ✅ Cobertura de código
- ✅ Reportes automáticos

**Se ejecuta automáticamente en:**
- Push a `main` o `develop`
- Pull requests
- Diariamente a las 2 AM UTC (schedule)

## Pre-commit Hooks

### Instalación
```bash
pip install pre-commit
pre-commit install
```

### Hooks Configurados
- ✅ Formateo automático (black, isort)
- ✅ Linting (flake8)
- ✅ Validación de estructura
- ✅ Tests rápidos (pre-push)

### Ejecutar Manualmente
```bash
pre-commit run --all-files
```

## Workflow Recomendado

### Desarrollo Local
```bash
# 1. Instalar dependencias
make install

# 2. Validar estructura
make validate

# 3. Ejecutar tests
make test-unit

# 4. Formatear código
make format

# 5. Verificar antes de commit
pre-commit run --all-files
```

### Antes de Push
```bash
# 1. Ejecutar todos los tests
make test

# 2. Verificar cobertura
make test-coverage

# 3. Generar reporte
make report

# 4. Analizar estructura
make analyze
```

### CI/CD
- Los tests se ejecutan automáticamente en GitHub Actions
- Los reportes se generan y suben como artifacts
- La cobertura se sube a Codecov

## Troubleshooting

### Tests Failing en CI pero no localmente
```bash
# Ejecutar en Docker para reproducir
make docker-test
```

### Problemas de Dependencias
```bash
# Reinstalar dependencias
pip install --upgrade -r requirements-test.txt
```

### Validación Falla
```bash
# Ver detalles del error
python validate_structure.py
```

### Docker No Funciona
```bash
# Verificar Docker
docker --version
docker-compose --version

# Limpiar y reconstruir
make docker-clean
docker-compose -f docker-compose.test.yml build --no-cache
```

### 6. `monitor_tests.py` - Monitor de Tests
Monitorea ejecuciones de tests en tiempo real y genera alertas.

```bash
python monitor_tests.py
python monitor_tests.py --interval 30 --duration 3600
make monitor
```

**Características:**
- 🔍 Monitoreo continuo
- ⚠️ Alertas automáticas
- 📊 Estadísticas en tiempo real
- 🏥 Verificación de salud

### 7. `visualize_results.py` - Visualizador de Resultados
Genera visualizaciones interactivas de resultados de tests.

```bash
python visualize_results.py --input monitor_stats.json --output dashboard.html
make visualize
```

**Genera:**
- 📊 Dashboard HTML interactivo
- 📈 Gráficos de tendencias
- 🕐 Timeline de ejecuciones
- 📉 Métricas visuales

### 8. `debug_tests.py` - Debugger de Tests
Ayuda a identificar y resolver problemas en tests.

```bash
python debug_tests.py --category unit
python debug_tests.py --test test_specific_name
make debug
```

**Funciones:**
- 🔍 Encuentra tests fallidos
- 📋 Analiza patrones de errores
- 💡 Sugiere fixes
- 🐛 Debugging detallado

### 9. `compare_results.py` - Comparador de Resultados
Compara resultados entre diferentes ejecuciones.

```bash
python compare_results.py results1.json results2.json
python compare_results.py results1.json results2.json --output comparison.json
```

**Compara:**
- 📊 Estadísticas
- ⏱️ Rendimiento
- ⚠️ Regresiones
- 📈 Mejoras

### 10. `profile_tests.py` - Profiler de Tests
Analiza el rendimiento de tests y genera perfiles.

```bash
python profile_tests.py --test test_name
python profile_tests.py --threshold 2.0
make profile
```

**Analiza:**
- 🔬 Profiling detallado
- ⏱️ Tests lentos
- 📊 Top funciones
- 🎯 Optimizaciones

## Comandos Make Avanzados

### Monitoreo y Visualización
```bash
make monitor          # Monitorear tests en tiempo real
make visualize        # Generar visualización de resultados
```

### Debugging y Análisis
```bash
make debug            # Debuggear tests
make profile          # Profilear tests
make compare          # Comparar resultados
```

## Recursos Adicionales

- **README.md**: Documentación completa
- **QUICK_START.md**: Guía de inicio rápido
- **INDEX.md**: Índice de archivos
- **TOOLS.md**: Esta guía de herramientas
- **templates/**: Templates para nuevos tests


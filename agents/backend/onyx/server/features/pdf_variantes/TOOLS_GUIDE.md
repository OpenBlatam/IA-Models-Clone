# Guía Completa de Herramientas

## Herramientas Disponibles

### 1. 🐛 Debug Tool (`debug_api.py`)

Herramienta interactiva para probar endpoints de la API.

**Uso:**
```bash
python debug_api.py
# o desde el menú principal: opción 1
```

**Características:**
- Prueba interactiva de endpoints
- Historial de requests
- Guardado de historial
- Comandos: health, upload, variant, topics, preview, test, history, save, summary, clear

### 2. 📊 API Monitor (`api_monitor.py`)

Monitoreo en tiempo real de la API.

**Uso:**
```bash
python api_monitor.py
# o desde el menú principal: opción 2
```

**Características:**
- Monitoreo continuo
- Estadísticas en tiempo real
- Cálculo de error rate y availability
- Métricas de response time
- Guardado de métricas

**Comandos:**
- `check` - Verificar salud una vez
- `start [interval]` - Iniciar monitoreo continuo
- `stop` - Detener monitoreo
- `stats` - Mostrar estadísticas
- `save <file>` - Guardar métricas

### 3. 📈 API Profiler (`api_profiler.py`)

Profiling de performance de endpoints.

**Uso:**
```bash
python api_profiler.py health
python api_profiler.py endpoint GET /health
# o desde el menú principal: opción 3
```

**Características:**
- Profiling de endpoints
- Análisis de funciones
- Guardado de perfiles
- Estadísticas detalladas

### 4. 📊 API Dashboard (`api_dashboard.py`) - NUEVO

Dashboard en tiempo real con actualización automática.

**Uso:**
```bash
python api_dashboard.py
# o desde el menú principal: opción 4
```

**Características:**
- Dashboard en tiempo real
- Actualización automática cada 2 segundos
- Métricas completas:
  - Status y availability
  - Requests y rate
  - Response time (min, max, avg, median, p95, p99)
  - Status codes
  - Errors
  - Uptime
- Exportación de datos

**Opciones:**
```bash
python api_dashboard.py --url http://localhost:8000
python api_dashboard.py --interval 1.0  # Refresh cada 1 segundo
python api_dashboard.py --export data.json  # Exportar datos
```

### 5. 🧪 API Test Suite (`api_test_suite.py`)

Suite automatizada de tests para la API.

**Uso:**
```bash
# Tests básicos
python api_test_suite.py

# Suite personalizada
python api_test_suite.py --suite test_suite_example.json

# Exportar resultados
python api_test_suite.py --export results.json
# o desde el menú principal: opción 5
```

**Características:**
- Tests automatizados
- Múltiples tipos de tests:
  - Health check
  - File upload
  - Endpoints genéricos
- Resumen de resultados
- Exportación de resultados
- Suite configurable via JSON

**Formato de Suite JSON:**
```json
{
  "name": "My Test Suite",
  "tests": [
    {
      "type": "health",
      "name": "Health Check"
    },
    {
      "type": "endpoint",
      "name": "Test Endpoint",
      "method": "GET",
      "endpoint": "/health",
      "expected_status": 200
    },
    {
      "type": "upload",
      "name": "Upload Test",
      "file": "test.pdf"
    }
  ]
}
```

### 6. 🔍 API Health Checker (`api_health_checker.py`) - NUEVO

Herramienta comprehensiva para health checks.

**Uso:**
```bash
python api_health_checker.py
# o desde el menú principal: opción 6
```

**Características:**
- Health check comprehensivo
- Verificación de múltiples endpoints
- Diagnóstico detallado
- Clasificación de status (healthy, degraded, unhealthy)
- Exportación de resultados

**Opciones:**
```bash
# Health check completo
python api_health_checker.py

# Endpoint específico
python api_health_checker.py --endpoint /health

# Exportar resultados
python api_health_checker.py --export health_results.json
```

### 7. 📝 API Logger (`api_logger.py`) - NUEVO

Herramienta avanzada para logging de requests y responses.

**Uso:**
```bash
python api_logger.py
# o desde el menú principal: opción 7
```

**Características:**
- Logging automático de requests/responses
- Captura de headers, body, tiempos
- Estadísticas de logging
- Exportación de logs
- Auto-save opcional

**Opciones:**
```bash
# Logger con auto-save
python api_logger.py --log-file api.log

# Ver estadísticas
python api_logger.py --stats

# Ver logs recientes
python api_logger.py --recent 20

# Exportar logs
python api_logger.py --export logs.json
```

## Herramienta Unificada

### `start_api_and_debug.py`

Inicia la API y proporciona acceso a todas las herramientas.

**Uso básico:**
```bash
python start_api_and_debug.py
```

**Opciones:**
```bash
# Abrir herramienta específica
python start_api_and_debug.py --tool dashboard
python start_api_and_debug.py --tool test
python start_api_and_debug.py --tool health
python start_api_and_debug.py --tool logger

# Opciones avanzadas
python start_api_and_debug.py --port 8001
python start_api_and_debug.py --quiet
python start_api_and_debug.py --no-menu
```

## Flujos de Trabajo

### Desarrollo Local

1. **Iniciar API y herramientas:**
   ```bash
   python start_api_and_debug.py
   ```

2. **Abrir dashboard para monitoreo:**
   - Seleccionar opción 4 (Dashboard)
   - Ver métricas en tiempo real

3. **Probar endpoints:**
   - Seleccionar opción 1 (Debug Tool)
   - Probar endpoints interactivamente

### Testing

1. **Ejecutar test suite:**
   ```bash
   python start_api_and_debug.py --tool test
   ```

2. **Suite personalizada:**
   ```bash
   python api_test_suite.py --suite my_tests.json --export results.json
   ```

### Monitoreo en Producción

1. **Dashboard continuo:**
   ```bash
   python api_dashboard.py --url https://api.example.com
   ```

2. **Exportar datos:**
   ```bash
   python api_dashboard.py --export metrics.json
   ```

## Integración

### Con CI/CD

```yaml
# .github/workflows/test.yml
- name: Run API Tests
  run: |
    python start_api_and_debug.py --no-menu &
    sleep 5
    python api_test_suite.py --export test_results.json
```

### Con Scripts Personalizados

```python
from api_dashboard import APIDashboard
from api_test_suite import APITestSuite

# Dashboard
dashboard = APIDashboard(base_url="http://localhost:8000")
dashboard.monitor_continuous(interval=1.0)
time.sleep(60)
dashboard.stop_monitoring()
dashboard.export_data(Path("metrics.json"))

# Test Suite
suite = APITestSuite(base_url="http://localhost:8000")
results = suite.run_suite([
    {"type": "health"},
    {"type": "endpoint", "name": "Test", "method": "GET", "endpoint": "/health"}
])
suite.print_summary()
suite.export_results(Path("results.json"))
```

## Mejores Prácticas

1. **Usar dashboard para monitoreo continuo**: Visualiza métricas en tiempo real
2. **Ejecutar test suite regularmente**: Asegura que la API funciona correctamente
3. **Exportar datos**: Guarda métricas y resultados para análisis
4. **Combinar herramientas**: Usa múltiples herramientas para análisis completo
5. **Automatizar en CI/CD**: Integra herramientas en pipelines

## Troubleshooting

### Dashboard no actualiza

Verificar que la API esté corriendo:
```bash
python debug_api.py health
```

### Test suite falla

Verificar configuración de suite JSON:
```bash
python -m json.tool test_suite_example.json
```

### Herramientas no encuentran API

Verificar URL:
```bash
python api_dashboard.py --url http://localhost:8000
```


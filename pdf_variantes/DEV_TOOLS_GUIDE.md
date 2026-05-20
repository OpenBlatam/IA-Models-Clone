# Guía de Herramientas de Desarrollo

## Herramientas Disponibles

### 1. API Monitor (`api_monitor.py`)

Monitoreo en tiempo real de la API.

#### Uso Interactivo

```bash
python api_monitor.py
```

#### Comandos Disponibles

- `check` - Verificar salud una vez
- `start [interval]` - Iniciar monitoreo continuo
- `stop` - Detener monitoreo
- `stats` - Mostrar estadísticas
- `save <file>` - Guardar métricas en archivo
- `quit/exit` - Salir

#### Ejemplo

```bash
$ python api_monitor.py

📊 API Monitor - Real-time Monitoring
============================================================
Commands:
  check          - Check health once
  start          - Start continuous monitoring
  stop           - Stop monitoring
  stats          - Show statistics
  save <file>     - Save metrics to file
  quit/exit       - Exit
============================================================

monitor> start 1
✅ Monitoring started (interval: 1.0s)
   Use 'stats' to view statistics, 'stop' to stop

monitor> stats

============================================================
📊 API Monitoring Statistics
============================================================
Total Requests: 50
Total Errors: 2
Error Rate: 4.00%
Availability: 96.00%
Uptime: 50.0s

Response Time:
  Current: 45.23ms
  Average: 42.15ms
  Min: 38.12ms
  Max: 125.45ms

Status Codes:
  200: 48
  500: 2
============================================================

monitor> save metrics.json
✅ Metrics saved to metrics.json
```

#### Uso No Interactivo

```bash
# Verificar salud una vez
python api_monitor.py check

# Mostrar estadísticas
python api_monitor.py stats

# Monitoreo continuo
python api_monitor.py monitor 1
```

### 2. API Profiler (`api_profiler.py`)

Profiling de performance de endpoints.

#### Uso Básico

```bash
# Profilar health endpoint
python api_profiler.py health

# Profilar endpoint específico
python api_profiler.py endpoint GET /health
python api_profiler.py endpoint POST /pdf/upload
```

#### Uso en Código

```python
from api_profiler import APIProfiler

profiler = APIProfiler()

# Profilar un endpoint
with profiler.profile_endpoint("upload", "POST", "/pdf/upload", files=files):
    pass

# Ver estadísticas
print(profiler.get_profile_stats("upload"))

# Guardar perfil
profiler.save_profile("upload", Path("upload_profile.prof"))

# Guardar resultados
profiler.save_results(Path("profiling_results.json"))
```

### 3. Development Tools (`dev_tools.py`)

Herramienta unificada para todas las herramientas de desarrollo.

#### Uso Interactivo

```bash
python dev_tools.py
```

#### Menú

```
🛠️  Development Tools
============================================================
1. Run API with debugging
2. Run debug tool (interactive)
3. Run API monitor
4. Run API profiler
5. Run tests
6. Run tests with debugging
7. Run linter
8. Exit
============================================================
```

#### Uso Directo

```bash
# Ejecutar herramienta específica
python dev_tools.py api        # Run API with debugging
python dev_tools.py debug      # Run debug tool
python dev_tools.py monitor    # Run API monitor
python dev_tools.py profiler   # Run API profiler
python dev_tools.py tests      # Run tests
python dev_tools.py test-debug # Run tests with debugging
python dev_tools.py lint       # Run linter
```

### 4. Debug API (`debug_api.py`)

Herramienta interactiva para debugging de API.

Ver `DEBUG_GUIDE.md` para más detalles.

### 5. Run API Debug (`run_api_debug.py`)

Ejecutar API con debugging habilitado.

Ver `DEBUG_GUIDE.md` para más detalles.

## Flujos de Trabajo Comunes

### Desarrollo Local

1. **Iniciar API con debugging:**
   ```bash
   python dev_tools.py api
   # o
   python run_api_debug.py
   ```

2. **En otra terminal, monitorear:**
   ```bash
   python dev_tools.py monitor
   # o
   python api_monitor.py
   ```

3. **Probar endpoints:**
   ```bash
   python dev_tools.py debug
   # o
   python debug_api.py
   ```

### Debugging de Problemas

1. **Iniciar monitoreo:**
   ```bash
   python api_monitor.py
   monitor> start 0.5  # Monitoreo cada 0.5s
   ```

2. **Reproducir problema**

3. **Ver estadísticas:**
   ```bash
   monitor> stats
   ```

4. **Guardar métricas:**
   ```bash
   monitor> save debug_metrics.json
   ```

### Profiling de Performance

1. **Profilar endpoint:**
   ```bash
   python api_profiler.py endpoint GET /health
   ```

2. **Ver resultados:**
   - Se muestran las top 20 funciones
   - Tiempo de ejecución
   - Estadísticas detalladas

3. **Guardar perfil:**
   ```python
   from api_profiler import APIProfiler
   
   profiler = APIProfiler()
   with profiler.profile_endpoint("test", "GET", "/health"):
       pass
   
   profiler.save_profile("test", Path("profile.prof"))
   ```

### Testing

1. **Ejecutar tests:**
   ```bash
   python dev_tools.py tests
   ```

2. **Tests con debugging:**
   ```bash
   python dev_tools.py test-debug
   ```

3. **Tests específicos:**
   ```bash
   pytest tests/test_api_endpoints.py -v
   ```

## Integración

### Con CI/CD

```yaml
# .github/workflows/test.yml
- name: Run API Monitor
  run: python api_monitor.py check

- name: Run Tests
  run: python dev_tools.py tests
```

### Con Scripts Personalizados

```python
from api_monitor import APIMonitor
from api_profiler import APIProfiler

# Monitor
monitor = APIMonitor()
monitor.monitor_continuous(interval=1.0)
time.sleep(60)  # Monitor for 1 minute
monitor.stop_monitoring()
monitor.print_statistics()

# Profiler
profiler = APIProfiler()
with profiler.profile_endpoint("test", "GET", "/health"):
    pass
profiler.print_summary()
```

## Mejores Prácticas

1. **Usar monitor durante desarrollo**: Detecta problemas temprano
2. **Profilar endpoints lentos**: Identifica cuellos de botella
3. **Guardar métricas**: Útil para análisis posterior
4. **Combinar herramientas**: Usa monitor + profiler para análisis completo
5. **Automatizar en CI/CD**: Integra herramientas en pipelines

## Troubleshooting

### Monitor no conecta

Verificar que la API esté corriendo:
```bash
python debug_api.py health
```

### Profiler muestra errores

Verificar que el endpoint existe y es accesible:
```bash
curl http://localhost:8000/health
```

### Herramientas no encuentran módulos

Asegurarse de estar en el directorio correcto:
```bash
cd agents/backend/onyx/server/features/pdf_variantes
python dev_tools.py
```




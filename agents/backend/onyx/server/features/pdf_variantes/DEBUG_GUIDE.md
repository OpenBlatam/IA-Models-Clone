# Guía de Debugging de la API

## Scripts Disponibles

### 1. Ejecutar API con Debugging

#### Windows (PowerShell)
```powershell
.\run_with_debug.ps1
```

#### Linux/Mac (Bash)
```bash
chmod +x run_with_debug.sh
./run_with_debug.sh
```

#### Python Directo
```bash
python run_api_debug.py
```

### 2. Herramienta de Debugging Interactiva

```bash
python debug_api.py
```

## Características de Debugging

### Variables de Entorno Activadas

Cuando ejecutas con debugging, se activan automáticamente:

- `DEBUG=true` - Modo debug
- `LOG_LEVEL=debug` - Nivel de log detallado
- `DETAILED_ERRORS=true` - Errores detallados
- `LOG_REQUESTS=true` - Log de requests
- `LOG_RESPONSES=true` - Log de responses
- `ENABLE_METRICS=true` - Métricas habilitadas
- `ENABLE_PROFILING=true` - Profiling habilitado

### Herramienta Interactiva de Debugging

La herramienta `debug_api.py` proporciona una interfaz interactiva para probar la API:

#### Comandos Disponibles

```
health          - Verificar salud de la API
upload <file>    - Subir un archivo PDF
variant <id>     - Generar variante para un archivo
topics <id>      - Extraer topics de un archivo
preview <id>     - Obtener preview de un archivo
test <method> <endpoint> [data] - Probar endpoint personalizado
history         - Mostrar historial de requests
save <file>      - Guardar historial en archivo
summary         - Mostrar resumen
quit/exit        - Salir
```

#### Ejemplo de Uso

```bash
$ python debug_api.py

🐛 API Debugging Tool
============================================================
Commands:
  health          - Check API health
  upload <file>    - Upload a PDF file
  variant <id>     - Generate variant for file
  topics <id>      - Extract topics from file
  preview <id>     - Get preview of file
  test <method> <endpoint> [data] - Test custom endpoint
  history         - Show request history
  save <file>      - Save history to file
  summary         - Show summary
  quit/exit        - Exit
============================================================

Checking API health...
✅ API is healthy

debug> health
{
  "status": 200,
  "data": {
    "status": "healthy",
    "version": "1.0.0"
  },
  "headers": {...},
  "timestamp": "2024-01-01T12:00:00"
}

debug> upload test.pdf
{
  "method": "POST",
  "endpoint": "/pdf/upload",
  "status": 200,
  "data": {
    "file_id": "abc123",
    "filename": "test.pdf"
  }
}

debug> variant abc123
{
  "method": "POST",
  "endpoint": "/pdf/abc123/variants",
  "status": 200,
  "data": {...}
}

debug> summary
============================================================
📊 Request Summary
============================================================
1. ✅ POST /pdf/upload - Status: 200
2. ✅ POST /pdf/abc123/variants - Status: 200
============================================================

debug> save debug_history.json
✅ History saved to debug_history.json

debug> quit
👋 Goodbye!
```

## Modo No Interactivo

También puedes usar la herramienta en modo no interactivo:

```bash
# Verificar salud
python debug_api.py health

# Probar endpoint
python debug_api.py test GET /health
python debug_api.py test POST /pdf/upload '{"file": "test.pdf"}'
```

## Integración con Tests

Puedes usar `APIDebugger` en tus tests:

```python
from debug_api import APIDebugger

def test_api_with_debug():
    debugger = APIDebugger(base_url="http://localhost:8000")
    
    # Check health
    health = debugger.health_check()
    assert health.get("status") == 200
    
    # Test endpoint
    result = debugger.test_endpoint("GET", "/health")
    assert result.get("status") == 200
    
    # Get history
    history = debugger.get_request_history()
    assert len(history) > 0
```

## Debugging Avanzado

### Usando con Playwright Debugger

Puedes combinar con las herramientas de debugging de Playwright:

```python
from debug_api import APIDebugger
from playwright_debug import create_debugger

# API Debugger
api_debugger = APIDebugger()

# Playwright Debugger
playwright_debugger = create_debugger(page)

# Combinar ambos
api_result = api_debugger.test_endpoint("GET", "/health")
playwright_debugger.save_debug_info("api_test")
```

### Guardar Historial

El historial de requests se guarda automáticamente y puede ser útil para:

- Análisis de problemas
- Documentación de flujos
- Reproducción de bugs
- Análisis de performance

```python
debugger = APIDebugger()
# ... hacer requests ...
debugger.save_history(Path("debug_history.json"))
```

## Troubleshooting

### API no responde

1. Verificar que la API esté corriendo:
   ```bash
   python debug_api.py health
   ```

2. Verificar el puerto:
   ```bash
   # Cambiar puerto en debug_api.py
   debugger = APIDebugger(base_url="http://localhost:8001")
   ```

### Errores de autenticación

Actualizar el token en `debug_api.py`:

```python
self.session.headers.update({
    "Authorization": "Bearer tu_token_aqui"
})
```

### Errores de conexión

Verificar que la API esté accesible:

```bash
curl http://localhost:8000/health
```

## Herramientas Adicionales

### API Monitor

Monitoreo en tiempo real:
```bash
python api_monitor.py
```

### API Profiler

Profiling de performance:
```bash
python api_profiler.py health
python api_profiler.py endpoint GET /health
```

### Development Tools

Herramienta unificada:
```bash
python dev_tools.py
```

Ver `DEV_TOOLS_GUIDE.md` para más detalles.

## Mejores Prácticas

1. **Usar en desarrollo**: Las herramientas de debugging están diseñadas para desarrollo
2. **Guardar historial**: Guarda el historial cuando encuentres problemas
3. **Combinar herramientas**: Usa API Debugger con Playwright Debugger para debugging completo
4. **Documentar problemas**: Usa el historial para documentar problemas encontrados
5. **Tests automatizados**: Usa `APIDebugger` en tests para debugging automatizado
6. **Monitoreo continuo**: Usa API Monitor para detectar problemas en tiempo real
7. **Profiling regular**: Usa API Profiler para identificar cuellos de botella


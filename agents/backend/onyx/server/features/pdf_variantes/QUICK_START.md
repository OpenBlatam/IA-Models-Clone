# Quick Start - API and Debug

## Inicio Rápido

### Opción 1: Todo en Uno (Recomendado)

```bash
# Windows (PowerShell)
python start_api_and_debug.py

# Linux/Mac
python start_api_and_debug.py
```

Esto:
1. ✅ Inicia la API con debugging habilitado
2. ✅ Verifica que el puerto esté disponible
3. ✅ Espera a que la API esté lista (con indicador de progreso)
4. ✅ Muestra información de la API (URLs, docs, etc.)
5. ✅ Muestra un menú interactivo con herramientas de debugging
6. ✅ Guarda información de sesión en `debug_session.json`

### Opción 2: Herramienta Específica

```bash
# Abrir directamente el debug tool
python start_api_and_debug.py --tool debug

# Abrir directamente el monitor
python start_api_and_debug.py --tool monitor

# Abrir directamente el profiler
python start_api_and_debug.py --tool profiler
```

### Opción 3: Opciones Avanzadas

```bash
# Usar puerto diferente
python start_api_and_debug.py --port 8001

# Inicio silencioso (sin logs de API)
python start_api_and_debug.py --quiet

# Solo iniciar API sin menú
python start_api_and_debug.py --no-menu

# No esperar a que API esté lista
python start_api_and_debug.py --no-wait
```

### Opción 3: Scripts Rápidos

```bash
# Windows
.\start_and_debug.ps1

# Linux/Mac
chmod +x start_and_debug.sh
./start_and_debug.sh
```

## Menú Interactivo

Cuando ejecutas `start_api_and_debug.py`, verás:

```
✅ API Server Running
============================================================
📍 URL: http://localhost:8000
📚 Docs: http://localhost:8000/docs
🔍 ReDoc: http://localhost:8000/redoc
============================================================

🛠️  Debugging Tools Menu
============================================================
API is running at http://localhost:8000

Available tools:
  1. 🐛 Debug Tool (Interactive API testing)
  2. 📊 API Monitor (Real-time monitoring)
  3. 📈 API Profiler (Performance profiling)
  4. 📊 API Dashboard (Live dashboard)
  5. 🧪 API Test Suite (Automated tests)
  6. 🌐 Open API Docs (Browser)
  7. 📖 Open ReDoc (Browser)
  8. 💚 Check Health
  9. 📋 Show API Info
 10. 🛑 Exit (Stop API)
============================================================
```

## Uso del Debug Tool

Una vez que seleccionas opción 1, puedes usar:

```
debug> health
{
  "status": 200,
  "data": {"status": "healthy"},
  ...
}

debug> upload test.pdf
{
  "method": "POST",
  "endpoint": "/pdf/upload",
  "status": 200,
  ...
}

debug> summary
📊 Request Summary
1. ✅ GET /health - Status: 200
2. ✅ POST /pdf/upload - Status: 200
```

## Uso del Monitor

Selecciona opción 2 para monitoreo en tiempo real:

```
monitor> start 1
✅ Monitoring started (interval: 1.0s)

monitor> stats
📊 API Monitoring Statistics
Total Requests: 50
Error Rate: 0.00%
Availability: 100.00%
...
```

## Detener la API

- Presiona `Ctrl+C` en la terminal
- O selecciona opción 6 en el menú

## Troubleshooting

### API no inicia

Verifica que el puerto 8000 esté libre:
```bash
# Windows
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000
```

### Herramienta no se abre

Ejecuta manualmente:
```bash
python debug_api.py
```

### API no responde

Verifica la salud:
```bash
python debug_api.py health
```

## Más Información

- `DEBUG_GUIDE.md` - Guía completa de debugging
- `DEV_TOOLS_GUIDE.md` - Guía de herramientas de desarrollo

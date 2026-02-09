# AI Project Generator - Implementación Fácil

Sistema modular para generar proyectos de IA con arquitectura de microservicios.

## ⚡ Inicio Rápido (30 segundos)

```bash
# 1. Instalar
pip install -r requirements.txt

# 2. Ejecutar
python main.py
```

¡Listo! Tu API está en `http://localhost:8020`

## 📝 Uso Básico

### Crear Aplicación

```python
from core.easy_setup import quick_start

app = quick_start()
```

### Crear Proyecto

```python
import requests

response = requests.post(
    "http://localhost:8020/api/v1/projects",
    json={
        "description": "Un chat con IA",
        "project_name": "chat_ai",
        "author": "Blatam Academy"
    }
)
```

## 🎯 Opciones de Configuración

### 1. Inicio Rápido (Recomendado)

```python
from core.easy_setup import quick_start
app = quick_start()
```

### 2. Personalizado

```python
from core.easy_setup import create_app_easy

app = create_app_easy(
    enable_cache=True,
    enable_metrics=True,
    redis_url="redis://localhost:6379"  # Opcional
)
```

### 3. Presets

```python
from core.easy_setup import (
    create_app_development,   # Desarrollo
    create_app_production,     # Producción
    create_app_serverless      # Serverless
)

app = create_app_development()
```

## 📚 Documentación

- **`QUICK_START.md`** - Guía rápida paso a paso
- **`EASY_IMPLEMENTATION.md`** - Guía de implementación fácil
- **`MODULAR_ARCHITECTURE.md`** - Arquitectura modular
- **`MICROSERVICES_GUIDE.md`** - Características avanzadas

## 🎨 Ejemplos

Ver carpeta `examples/`:
- `simple_example.py` - Ejemplo mínimo
- `custom_example.py` - Configuración personalizada
- `production_example.py` - Producción
- `serverless_example.py` - Serverless

## ✨ Características

- ✅ **Fácil de usar**: Una línea para empezar
- ✅ **Auto-configurable**: Detecta servicios automáticamente
- ✅ **Modular**: Arquitectura de microservicios
- ✅ **Escalable**: Listo para producción
- ✅ **Serverless-ready**: Optimizado para Lambda, Azure Functions

## 🔧 Endpoints Principales

- `POST /api/v1/projects` - Crear proyecto
- `GET /api/v1/projects/{id}` - Obtener proyecto
- `POST /api/v1/generate` - Generar proyecto
- `GET /health` - Health check
- `GET /metrics` - Métricas Prometheus

## 💡 Tips

1. **Empieza simple**: Usa `quick_start()`
2. **Agrega gradualmente**: Habilita features según necesites
3. **Usa presets**: Están optimizados para cada caso
4. **Auto-detección**: El sistema detecta Redis automáticamente

## 🆘 Ayuda

- Ver `QUICK_START.md` para guía detallada
- Ver `EASY_IMPLEMENTATION.md` para troubleshooting
- Ver ejemplos en carpeta `examples/`

¡Listo para generar proyectos de IA! 🚀
















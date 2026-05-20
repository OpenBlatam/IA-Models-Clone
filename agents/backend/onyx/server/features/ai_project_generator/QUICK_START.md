# Quick Start - Inicio Rápido

Guía rápida para empezar a usar el AI Project Generator con arquitectura modular.

## 🚀 Inicio en 3 Pasos

### 1. Instalación

```bash
pip install -r requirements.txt
```

### 2. Crear Aplicación (3 opciones)

#### Opción A: Inicio Rápido (Más Fácil)

```python
from core.easy_setup import quick_start

app = quick_start()
```

#### Opción B: Con Configuración Personalizada

```python
from core.easy_setup import create_app_easy

app = create_app_easy(
    enable_cache=True,      # Cache habilitado
    enable_metrics=True,    # Métricas Prometheus
    enable_workers=False,   # Sin workers (más simple)
    enable_events=False    # Sin eventos (más simple)
)
```

#### Opción C: Presets Pre-configurados

```python
from core.easy_setup import (
    create_app_development,
    create_app_production,
    create_app_serverless
)

# Desarrollo
app = create_app_development()

# Producción (requiere Redis)
app = create_app_production(redis_url="redis://localhost:6379")

# Serverless
app = create_app_serverless()
```

### 3. Ejecutar

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)
```

O usar el archivo main.py:

```bash
python main.py
```

## 📝 Ejemplo Completo

```python
# main.py
from core.easy_setup import quick_start
import uvicorn

app = quick_start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)
```

¡Eso es todo! Tu API está lista en 4 líneas.

## 🎯 Endpoints Disponibles

Una vez iniciado, tendrás acceso a:

- `POST /api/v1/projects` - Crear proyecto
- `GET /api/v1/projects/{project_id}` - Obtener proyecto
- `GET /api/v1/projects` - Listar proyectos
- `POST /api/v1/generate` - Generar proyecto
- `GET /health` - Health check
- `GET /metrics` - Métricas Prometheus (si está habilitado)

## ⚙️ Configuración Avanzada (Opcional)

Si necesitas más control, puedes configurar variables de entorno:

```bash
# Cache
export MICROSERVICES_CACHE_BACKEND=redis
export MICROSERVICES_CACHE_URL=redis://localhost:6379

# Workers
export MICROSERVICES_WORKER_BACKEND=celery
export MICROSERVICES_WORKER_BROKER_URL=redis://localhost:6379/0

# Metrics
export MICROSERVICES_PROMETHEUS_ENABLED=true
```

O usar la configuración programática:

```python
from core.microservices_config import MicroservicesConfig
from api.app_factory import create_app

# Configurar
config = MicroservicesConfig(
    cache_backend="redis",
    cache_url="redis://localhost:6379",
    prometheus_enabled=True
)

app = create_app()
```

## 🔧 Servicios Opcionales

### Redis (Opcional pero Recomendado)

```bash
# Instalar Redis
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Docker
docker run -d -p 6379:6379 redis
```

Si Redis está disponible, se usará automáticamente. Si no, se usa cache en memoria.

### Workers (Opcional)

Para tareas asíncronas, necesitas un broker (Redis o RabbitMQ):

```python
app = create_app_easy(
    enable_workers=True,
    redis_url="redis://localhost:6379"
)
```

### Events (Opcional)

Para arquitectura event-driven:

```python
app = create_app_easy(
    enable_events=True,
    redis_url="redis://localhost:6379"
)
```

## 📊 Monitoreo

### Métricas Prometheus

Si `enable_metrics=True` (por defecto), las métricas están en:

```
GET /metrics
```

### Health Check

```
GET /health
```

## 🐛 Troubleshooting

### Error: Redis no disponible

**Solución**: El sistema automáticamente usa cache en memoria. No es necesario Redis para empezar.

### Error: Workers no funcionan

**Solución**: Asegúrate de tener Redis o RabbitMQ configurado, o deshabilita workers:

```python
app = create_app_easy(enable_workers=False)
```

### Error: Métricas no aparecen

**Solución**: Verifica que `prometheus-client` esté instalado:

```bash
pip install prometheus-client
```

## 🎓 Siguiente Paso

Una vez que tengas la aplicación funcionando, consulta:

- `MODULAR_ARCHITECTURE.md` - Para entender la arquitectura
- `MICROSERVICES_GUIDE.md` - Para características avanzadas
- `README.md` - Para documentación completa

## 💡 Tips

1. **Empieza Simple**: Usa `quick_start()` primero
2. **Agrega Complejidad Gradualmente**: Habilita features según necesites
3. **Usa Presets**: Los presets están optimizados para cada caso de uso
4. **Auto-detección**: El sistema detecta automáticamente servicios disponibles

## ✅ Checklist de Inicio

- [ ] Instaladas dependencias (`pip install -r requirements.txt`)
- [ ] Creada aplicación con `quick_start()` o `create_app_easy()`
- [ ] Ejecutada aplicación (`python main.py`)
- [ ] Verificado health check (`GET /health`)
- [ ] (Opcional) Configurado Redis para mejor performance
- [ ] (Opcional) Habilitado workers si necesitas tareas asíncronas

¡Listo para generar proyectos de IA! 🚀
















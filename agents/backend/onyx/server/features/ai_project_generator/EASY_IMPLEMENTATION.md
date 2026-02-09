# Implementación Fácil

Guía para implementar el AI Project Generator de forma fácil y rápida.

## 🎯 Filosofía de Diseño

El sistema está diseñado para ser:
- **Fácil de empezar**: Funciona con configuración mínima
- **Progresivo**: Agrega complejidad solo cuando la necesitas
- **Auto-configurable**: Detecta automáticamente servicios disponibles
- **Con valores por defecto sensatos**: Funciona bien sin configuración

## ⚡ Inicio Rápido (30 segundos)

### 1. Instalar

```bash
pip install -r requirements.txt
```

### 2. Ejecutar

```bash
python main.py
```

¡Eso es todo! La aplicación está corriendo en `http://localhost:8020`

## 📝 Opciones de Configuración

### Opción 1: Inicio Rápido (Recomendado)

```python
from core.easy_setup import quick_start

app = quick_start()
```

**Características:**
- ✅ Cache en memoria (o Redis si está disponible)
- ✅ Métricas Prometheus
- ✅ Middleware avanzado
- ✅ Sin workers (más simple)
- ✅ Sin eventos (más simple)

### Opción 2: Configuración Personalizada

```python
from core.easy_setup import create_app_easy

app = create_app_easy(
    enable_cache=True,      # Cache habilitado
    enable_metrics=True,    # Métricas
    enable_workers=False,   # Sin workers
    enable_events=False,    # Sin eventos
    redis_url="redis://localhost:6379"  # Opcional
)
```

### Opción 3: Presets Pre-configurados

```python
from core.easy_setup import (
    create_app_development,  # Desarrollo
    create_app_production,    # Producción
    create_app_serverless    # Serverless
)

# Desarrollo (todo habilitado, in-memory si no hay Redis)
app = create_app_development()

# Producción (requiere Redis)
app = create_app_production(redis_url="redis://localhost:6379")

# Serverless (mínimo)
app = create_app_serverless()
```

## 🔧 Configuración por Entorno

### Desarrollo

```python
# main.py
from core.easy_setup import create_app_development

app = create_app_development()
```

**Características:**
- Cache en memoria (o Redis si está disponible)
- Métricas habilitadas
- Logging detallado
- Sin workers (más simple para desarrollo)

### Producción

```python
# main.py
from core.easy_setup import create_app_production

app = create_app_production(
    redis_url="redis://your-redis-url:6379"
)
```

**Características:**
- Redis para cache y workers
- Métricas Prometheus
- Eventos habilitados
- Workers asíncronos
- Optimizaciones de producción

### Serverless

```python
# handler.py
from core.easy_setup import create_app_serverless
from mangum import Mangum

app = create_app_serverless()
handler = Mangum(app)
```

**Características:**
- Sin cache persistente
- Métricas habilitadas
- Optimizado para cold start
- Sin workers (usa async de FastAPI)

## 🎨 Uso de Servicios

### Forma Simple

```python
from helpers import create_project_simple, generate_project_simple

# Crear proyecto
result = await create_project_simple(
    description="Un chat con IA",
    project_name="chat_ai"
)

# Generar proyecto
result = await generate_project_simple(
    description="Un analizador de imágenes",
    project_name="image_analyzer"
)
```

### Forma Avanzada

```python
from services.project_service import ProjectService
from infrastructure.dependencies import get_project_service

# Obtener servicio
service = get_project_service()

# Usar servicio
project = await service.get_project("project-id")
projects = await service.list_projects(status="completed")
```

## 📊 Endpoints Principales

Una vez iniciada la aplicación:

```python
# Health check
GET /health

# Crear proyecto
POST /api/v1/projects
{
    "description": "Un chat con IA",
    "project_name": "chat_ai",
    "author": "Blatam Academy"
}

# Obtener proyecto
GET /api/v1/projects/{project_id}

# Generar proyecto
POST /api/v1/generate
{
    "description": "Un analizador de imágenes",
    "project_name": "image_analyzer"
}

# Métricas
GET /metrics
```

## 🚀 Despliegue

### Local

```bash
python main.py
```

### Docker

```dockerfile
FROM python:3.11

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

### Serverless (AWS Lambda)

```python
# handler.py
from core.easy_setup import create_app_serverless
from mangum import Mangum

app = create_app_serverless()
handler = Mangum(app)
```

## 🔍 Auto-detección

El sistema detecta automáticamente:

- **Redis**: Si está disponible, se usa para cache. Si no, usa in-memory.
- **Workers**: Se habilitan solo si hay broker disponible.
- **Events**: Se habilitan solo si hay message broker disponible.

**No necesitas configurar nada** - el sistema funciona automáticamente.

## 📦 Dependencias Opcionales

### Mínimas (Siempre Necesarias)

```txt
fastapi
uvicorn
pydantic
```

### Recomendadas (Para Mejor Performance)

```txt
redis          # Para cache y workers
prometheus-client  # Para métricas
```

### Opcionales (Solo si Necesitas)

```txt
celery         # Para workers avanzados
pika           # Para RabbitMQ
kafka-python   # Para Kafka
mangum         # Para AWS Lambda
```

## 🎓 Ejemplos

Ver la carpeta `examples/`:

- `simple_example.py` - Ejemplo mínimo
- `custom_example.py` - Configuración personalizada
- `production_example.py` - Configuración de producción
- `serverless_example.py` - Configuración serverless
- `api_usage_example.py` - Cómo usar la API

## ✅ Checklist de Implementación

### Básico (5 minutos)

- [ ] Instalar dependencias: `pip install -r requirements.txt`
- [ ] Ejecutar: `python main.py`
- [ ] Verificar: `curl http://localhost:8020/health`

### Intermedio (10 minutos)

- [ ] Configurar Redis (opcional)
- [ ] Habilitar workers (opcional)
- [ ] Configurar métricas Prometheus
- [ ] Probar endpoints principales

### Avanzado (30 minutos)

- [ ] Configurar message broker
- [ ] Habilitar eventos
- [ ] Configurar API Gateway
- [ ] Configurar OAuth2
- [ ] Desplegar a producción

## 💡 Tips

1. **Empieza Simple**: Usa `quick_start()` primero
2. **Agrega Gradualmente**: Habilita features según necesites
3. **Usa Presets**: Están optimizados para cada caso
4. **Auto-detección**: Confía en la auto-detección
5. **Documentación**: Consulta `QUICK_START.md` para más detalles

## 🆘 Troubleshooting

### No inicia

**Solución**: Verifica que las dependencias estén instaladas:
```bash
pip install -r requirements.txt
```

### Redis no funciona

**Solución**: No es necesario. El sistema usa cache en memoria automáticamente.

### Workers no funcionan

**Solución**: Deshabilita workers o configura Redis:
```python
app = create_app_easy(enable_workers=False)
```

### Métricas no aparecen

**Solución**: Instala prometheus-client:
```bash
pip install prometheus-client
```

## 🎯 Siguiente Paso

Una vez que tengas la aplicación funcionando:

1. Lee `QUICK_START.md` para uso básico
2. Lee `MODULAR_ARCHITECTURE.md` para entender la arquitectura
3. Lee `MICROSERVICES_GUIDE.md` para características avanzadas

¡Listo para generar proyectos de IA! 🚀
















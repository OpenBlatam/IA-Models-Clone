# 🚀 Bulk TruthGPT API - Quick Start Guide

## Instalación y Ejecución

### Opción 1: Local (Desarrollo)

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno (opcional)
export API_KEY=tu_clave_secreta
export REDIS_URL=redis://localhost:6379/0
export CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Ejecutar servidor
uvicorn bulk_truthgpt.main:app --host 0.0.0.0 --port 8000 --reload
```

### Opción 2: Docker Compose (Producción)

```bash
# Construir y levantar servicios
docker compose up -d --build

# Ver logs
docker compose logs -f bulk_truthgpt

# Detener servicios
docker compose down
```

## Endpoints Principales

### Health Checks

```bash
# Health general
curl http://localhost:8000/health

# Health de Redis
curl http://localhost:8000/health/redis

# Readiness completo (verifica todas las dependencias)
curl http://localhost:8000/readiness
```

### Generación Masiva

```bash
# Iniciar generación (con API Key si está configurada)
curl -X POST http://localhost:8000/api/v1/bulk/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: tu_clave_secreta" \
  -d '{
    "query": "Generar contenido sobre inteligencia artificial",
    "config": {
      "max_documents": 10,
      "max_tokens": 2000,
      "temperature": 0.7
    }
  }'

# Consultar estado de tarea
curl http://localhost:8000/api/v1/bulk/status/{task_id}

# Historial de eventos de una tarea
curl http://localhost:8000/api/v1/bulk/tasks/{task_id}/history

# Listar todas las tareas
curl http://localhost:8000/api/v1/bulk/tasks

# Obtener documentos generados
curl http://localhost:8000/api/v1/bulk/documents/{task_id}?limit=10&offset=0
```

### Métricas y Monitoreo

```bash
# Métricas Prometheus (protegido por NGINX)
curl http://localhost:8000/metrics

# Resumen de métricas JSON
curl http://localhost:8000/api/v1/bulk/metrics
```

### Optimización

```bash
# Stats de optimización de velocidad
curl http://localhost:8000/api/v1/optimization/speed/stats

# Trigger warmup del sistema
curl -X POST http://localhost:8000/api/v1/optimization/speed/warmup
```

## Documentación Interactiva

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Características Clave

✅ **Cache Redis**: Respuestas cacheadas para mejor performance  
✅ **Rate Limiting**: Protección contra abuso (10 req/min en generate, 60/min en status)  
✅ **Tracking de Eventos**: Historial completo de cada tarea  
✅ **Métricas Prometheus**: Observabilidad completa  
✅ **Seguridad**: Headers de seguridad, API Key opcional  
✅ **Resiliencia**: Circuit breaker y retries automáticos  

## Variables de Entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `API_KEY` | Clave API para endpoints sensibles | None (opcional) |
| `REDIS_URL` | URL de conexión a Redis | `redis://localhost:6379/0` |
| `CORS_ORIGINS` | Orígenes permitidos (separados por coma) | `*` |
| `API_HOST` | Host del servidor | `0.0.0.0` |
| `API_PORT` | Puerto del servidor | `8000` |

## Ejemplos de Uso

### Python Client

```python
import httpx

async with httpx.AsyncClient() as client:
    # Iniciar generación
    response = await client.post(
        "http://localhost:8000/api/v1/bulk/generate",
        json={
            "query": "Explorar las aplicaciones de AI",
            "config": {"max_documents": 5}
        },
        headers={"X-API-Key": "tu_clave"}  # Si está configurada
    )
    task = response.json()
    task_id = task["task_id"]
    
    # Consultar estado
    status = await client.get(f"http://localhost:8000/api/v1/bulk/status/{task_id}")
    print(status.json())
```

## Troubleshooting

### Error: Redis no disponible
- Verifica que Redis esté corriendo: `docker compose ps redis`
- Revisa `REDIS_URL` en variables de entorno

### Error: Rate limit excedido
- Reduce la frecuencia de requests
- Los límites están configurados por endpoint (ver `/docs`)

### Error: Componente no inicializado
- Revisa logs del servidor
- Verifica `/readiness` para diagnóstico detallado

## Arquitectura

- **API Layer**: FastAPI con routers modulares
- **Service Layer**: Lógica de negocio separada
- **Repository Layer**: Acceso a datos
- **Cache Layer**: Redis para performance
- **Worker Layer**: Tareas asíncronas en background



# Guía de Despliegue - Piel Mejorador AI SAM3

## Despliegue con Docker

### Construcción de la Imagen

```bash
docker build -t piel-mejorador-ai-sam3:latest .
```

### Ejecución con Docker Compose

```bash
# Configurar variables de entorno
export OPENROUTER_API_KEY="tu-api-key"
export TRUTHGPT_ENDPOINT="opcional-endpoint"

# Iniciar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

### Ejecución Manual

```bash
docker run -d \
  --name piel-mejorador \
  -p 8000:8000 \
  -e OPENROUTER_API_KEY="tu-api-key" \
  -e TRUTHGPT_ENDPOINT="opcional-endpoint" \
  -v $(pwd)/output:/app/piel_mejorador_output \
  piel-mejorador-ai-sam3:latest
```

## Despliegue en Producción

### Requisitos del Sistema

- Python 3.11+
- 4GB RAM mínimo (8GB recomendado)
- 10GB espacio en disco
- CPU: 2+ cores

### Variables de Entorno

```bash
# Requeridas
OPENROUTER_API_KEY=tu-api-key

# Opcionales
TRUTHGPT_ENDPOINT=http://endpoint
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_TIMEOUT=120.0
PIEL_MEJORADOR_MAX_PARALLEL_TASKS=5
PIEL_MEJORADOR_OUTPUT_DIR=/app/output
PIEL_MEJORADOR_DEBUG=false
```

### Instalación

```bash
# Clonar repositorio
git clone <repo-url>
cd piel_mejorador_ai_sam3

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
export OPENROUTER_API_KEY="tu-api-key"

# Ejecutar
python -m piel_mejorador_ai_sam3.main
```

### Ejecutar API

```bash
uvicorn piel_mejorador_ai_sam3.api.piel_mejorador_api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

## Monitoreo

### Health Check

```bash
curl http://localhost:8000/health
```

### Métricas Prometheus

```bash
curl http://localhost:8000/metrics
```

### Estadísticas

```bash
curl http://localhost:8000/stats
```

## Escalabilidad

### Horizontal Scaling

Para escalar horizontalmente:

1. Usar un load balancer (nginx, traefik)
2. Múltiples instancias de la API
3. Base de datos compartida para tareas (opcional)
4. Sistema de colas compartido (Redis, RabbitMQ)

### Vertical Scaling

- Aumentar `max_parallel_tasks` según CPU
- Aumentar límites de memoria según RAM
- Ajustar timeouts según carga

## Seguridad

### Rate Limiting

Configurar límites apropiados:

```python
RateLimitConfig(
    requests_per_second=10.0,
    burst_size=20
)
```

### Webhooks

Usar secretos para verificar webhooks:

```python
agent.register_webhook(
    url="https://example.com/webhook",
    events=[...],
    secret="strong-secret-key"
)
```

### Variables de Entorno

Nunca commitear API keys. Usar:
- `.env` files (no en git)
- Secret managers (AWS Secrets Manager, etc.)
- Variables de entorno del sistema

## Troubleshooting

### Problemas Comunes

1. **Memoria alta**: Reducir `max_parallel_tasks` o procesar en lotes más pequeños
2. **Rate limiting**: Ajustar límites en configuración
3. **Timeouts**: Aumentar `OPENROUTER_TIMEOUT`
4. **Caché lleno**: Ejecutar limpieza periódica

### Logs

```bash
# Ver logs de aplicación
tail -f piel_mejorador_ai_sam3.log

# Logs de Docker
docker-compose logs -f piel-mejorador-api
```

## CI/CD

El proyecto incluye GitHub Actions para:

- Tests automáticos
- Linting
- Build de Docker

Ver `.github/workflows/ci.yml` para detalles.





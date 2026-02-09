# AI Video FastAPI Microservice (Enterprise Ready)

## Endpoints principales

- `POST /api/v1/video` — Solicita la generación de un video AI (asíncrono, requiere JWT)
- `GET /api/v1/video/{request_id}/status` — Estado del job
- `GET /api/v1/video/{request_id}/logs` — Logs del job (paginado, filtrado)
- `POST /api/v1/video/{request_id}/cancel` — Cancelar job
- `POST /api/v1/video/{request_id}/retry` — Reintentar job
- `POST /api/v1/video/{request_id}/pause` — Pausar job
- `POST /api/v1/video/{request_id}/resume` — Reanudar job
- `DELETE /api/v1/video/{request_id}` — Eliminar job
- `GET /api/v1/jobs/search` — Buscar jobs por usuario, estado, fecha, etc.
- `GET /api/v1/logs/search` — Buscar logs por filtros avanzados
- `GET /api/v1/audit` — Auditoría de accesos (solo admin)
- `GET /api/v1/webhook_failures` — Fallos de webhooks
- `POST /api/v1/token/refresh` — Refresh token
- `POST /api/v1/token/revoke` — Revocar token
- `/metrics` — Métricas Prometheus (solo admin)
- `/health` — Healthcheck extendido
- `/docs` — Documentación OpenAPI enriquecida

## Persistencia
- Jobs, logs, auditoría y tokens revocados se almacenan en SQLite (fácil de migrar a PostgreSQL).
- Fallback automático a in-memory si la DB no está disponible.

## Seguridad
- JWT con scopes, expiración y revocación persistente.
- Rate limiting configurable por usuario y endpoint.
- Protección de endpoints sensibles (audit, metrics, jobs/search) solo para admin.

## Observabilidad
- Métricas Prometheus por usuario, endpoint, estado, errores, reintentos, cancelaciones.
- Propagación de trace_id/span_id a todos los logs, Celery y webhooks.

## Auditoría
- Todos los accesos a endpoints sensibles quedan registrados con usuario, IP, scope, timestamp y trace_id.
- Endpoint de consulta de auditoría solo para admin, con paginación y filtros.

## Webhooks
- Soporte para múltiples webhooks por job.
- Notificación en todos los cambios de estado.
- Fallos de webhooks quedan registrados y consultables.

## Ejemplo de request

```bash
curl -X POST "http://localhost:8000/api/v1/video" \
  -H "accept: application/json" \
  -H "Authorization: Bearer supersecrettoken" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Crea un video demo", "user_id": "user1", "webhook_url": "http://requestbin.net/r/xxxx"}'
```

## Variables de entorno
- `DB_URL` — URL de la base de datos (por defecto: `sqlite:///./ai_video.db`)
- `CELERY_BROKER_URL` — URL de Redis (por defecto: `redis://localhost:6379/0`)
- `JWT_SECRET` — Secreto para firmar JWT
- `ALLOWED_ORIGINS` — Orígenes permitidos para CORS

## Notas
- El procesamiento es simulado (5s). Puedes conectar tu lógica real en `process_video_task`.
- Si envías `webhook_url`, recibirás notificación al finalizar el procesamiento y en cada cambio de estado.
- Si la base de datos no está disponible, el sistema sigue funcionando en modo demo (in-memory).
- Tracing OpenTelemetry y métricas Prometheus instrumentados.

## Autenticación
- Usa OAuth2/JWT (token de ejemplo: `supersecrettoken`)
- Header: `Authorization: Bearer supersecrettoken`

## Despliegue local (recomendado)

```bash
docker-compose up --build
```
Esto levanta:
- API FastAPI en `localhost:8000`
- Redis en `localhost:6379`
- Worker Celery

```bash
curl -X POST "http://localhost:8000/api/v1/video" \
  -H "accept: application/json" \
  -H "Authorization: Bearer supersecrettoken" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Crea un video demo", "user_id": "user1", "webhook_url": "http://requestbin.net/r/xxxx"}'
```

## Notas
- El procesamiento es simulado (5s). Puedes conectar tu lógica real en `process_video_task`.
- Si envías `webhook_url`, recibirás notificación al finalizar el procesamiento.
- Métricas Prometheus expuestas en `/metrics`.
- Tracing OpenTelemetry ya instrumentado.

# Recomendaciones de Despliegue y Performance para AI Video Microservice

## 1. Servidor ASGI (Uvicorn/Gunicorn)
- Usa Uvicorn con `--loop uvloop` para máxima velocidad:
  ```bash
  uvicorn fastapi_microservice:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop
  ```
- O Gunicorn con múltiples workers:
  ```bash
  gunicorn -k uvicorn.workers.UvicornWorker fastapi_microservice:app --workers 4 --bind 0.0.0.0:8000
  ```
- Ajusta el número de workers según núcleos de CPU y carga esperada.

## 2. Cache asíncrono para batch
- Si despliegas en varios procesos/nodos, usa un cache asíncrono compartido (ej: Redis con `aiocache` o `async_lru`).
- Ajusta el TTL del cache para jobs completados/fallidos según el patrón de uso.

## 3. Pools asíncronos para HTTP y DB
- Usa `httpx.AsyncClient` con pool para llamadas a Onyx u otros servicios externos.
- Usa `asyncpg.Pool` o similar para base de datos asíncrona.

## 4. Serialización rápida
- Si usas Pydantic v2, usa `.model_dump(exclude_unset=True)` en vez de `.dict()`.
- Configura FastAPI para usar `orjson` como backend de serialización JSON:
  ```python
  from fastapi import FastAPI
  import orjson
  app = FastAPI(default_response_class=ORJSONResponse)
  ```

## 5. Ajuste de max_concurrency
- El parámetro `max_concurrency` en endpoints batch controla el paralelismo real (default 10, máximo 50).
- Ajusta según el hardware, la carga y los límites de APIs externas.

## 6. Logging eficiente
- Usa logging asíncrono o bufferizado para no bloquear el event loop.
- Loggea solo eventos críticos en el hot path.

## 7. Monitoreo y métricas
- Expón métricas Prometheus (latencia, errores, recuento de peticiones) para todos los endpoints.
- Usa tracing distribuido (OpenTelemetry) para correlacionar logs y traces entre servicios.

## 8. Ejemplo de comando de despliegue óptimo
```bash
uvicorn fastapi_microservice:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop --log-level info
```

---

**Sigue estas recomendaciones para máxima velocidad, robustez y observabilidad en producción.**

# Troubleshooting y Tuning Avanzado

## 1. Identificación de cuellos de botella
- **CPU:** Usa `htop`, `top` o dashboards de cloud para ver saturación de workers.
- **IO/Red:** Monitorea latencia de disco y red, especialmente si usas almacenamiento externo o APIs remotas.
- **Onyx API:** Si la latencia de Onyx es alta, ajusta `max_concurrency` y usa timeouts/retries.
- **Base de datos:** Usa EXPLAIN y métricas de slow queries para optimizar índices y consultas.

## 2. Herramientas de profiling y benchmarking
- **Carga:** Usa [locust](https://locust.io/) o [wrk](https://github.com/wg/wrk) para simular usuarios concurrentes y medir throughput/latencia.
- **Profiling Python:** Usa [py-spy](https://github.com/benfred/py-spy) para identificar funciones lentas o bloqueantes.
- **Prometheus:** Exporta métricas y crea alertas para latencia, errores y saturación de recursos.

## 3. Tuning de max_concurrency y cache
- Comienza con `max_concurrency=10` y aumenta gradualmente mientras monitoreas CPU, latencia y errores.
- Si ves saturación de CPU o timeouts, reduce `max_concurrency`.
- Ajusta el tamaño y TTL del cache según el patrón de acceso y la memoria disponible.

## 4. Escalado horizontal y coherencia de cache
- Usa un cache compartido (ej: Redis) si corres varios pods/nodos.
- Considera invalidar el cache globalmente cuando un job cambia de estado.
- Usa un balanceador de carga con sticky sessions si es necesario.

## 5. Dashboard Prometheus/Grafana recomendado
- Métricas clave: latencia p95/p99, throughput, errores 5xx/4xx, uso de CPU/memoria, jobs en batch, tiempo de respuesta de Onyx.
- Ejemplo de panel: "Batch Job Latency", "Onyx API Response Time", "Cache Hit Ratio".

## 6. Logging distribuido y correlación de trace_id
- Asegúrate de propagar `trace_id` en todos los logs y llamadas entre servicios.
- Usa una solución de logging centralizado (ELK, Loki, Datadog) para buscar por `trace_id` y reconstruir flujos de usuario.

---

**Con estas prácticas, podrás identificar, diagnosticar y resolver cuellos de botella en producción, y ajustar el microservicio para máxima performance real.**

# 📡 API Endpoints Completos - Bulk TruthGPT

## 🎯 Endpoints Principales

### Health & Monitoring

#### `GET /health`
Health check completo con métricas.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:00:00Z",
  "version": "1.0.0",
  "components": {
    "truthgpt_engine": "initialized",
    "document_generator": "initialized",
    ...
  },
  "performance": {
    "cache": {
      "hit_rate_percent": 85.5,
      "entries": 1250,
      "size_mb": 45.2
    },
    "batch_processing": {
      "current_batch_size": 15,
      "avg_latency_ms": 450.5,
      "avg_throughput": 33.3
    },
    "system_metrics": {
      "cpu": {"percent": 45.2},
      "memory": {"rss_mb": 512.3, "percent": 60.5},
      "disk": {"percent": 45.0}
    },
    "memory": {
      "rss_mb": 512.3,
      "peak_mb": 580.1,
      "available_mb": 2048.0
    },
    "monitor": {
      "overall": "healthy",
      "checks": {...},
      "suggestions": [...]
    }
  }
}
```

#### `GET /health/redis`
Health check específico de Redis.

#### `GET /readiness`
Verificación completa de readiness.

#### `GET /metrics`
Métricas Prometheus.

---

### Generación de Documentos

#### `POST /api/v1/bulk/generate`
Iniciar generación masiva de documentos.

**Request:**
```json
{
  "query": "Explicar inteligencia artificial",
  "config": {
    "max_documents": 10,
    "max_tokens": 2000,
    "temperature": 0.7,
    "model": "gpt-3.5-turbo"
  },
  "priority": 1
}
```

**Response:**
```json
{
  "task_id": "abc123-def456",
  "status": "started",
  "message": "Bulk generation process started",
  "estimated_documents": 10,
  "estimated_duration": 5
}
```

#### `GET /api/v1/bulk/status/{task_id}`
Estado de una tarea de generación.

**Response:**
```json
{
  "task_id": "abc123",
  "status": "processing",
  "progress": 60,
  "documents_generated": 6,
  "total_documents": 10,
  "started_at": "2024-01-15T10:00:00Z",
  "estimated_completion": "2024-01-15T10:05:00Z"
}
```

#### `GET /api/v1/bulk/documents/{task_id}`
Obtener documentos generados.

**Query Parameters:**
- `limit` (int): Máximo de documentos (default: 100)
- `offset` (int): Offset para paginación (default: 0)

**Response:**
```json
{
  "task_id": "abc123",
  "total_documents": 10,
  "documents": [
    {
      "id": "doc_1",
      "content": "Contenido generado...",
      "timestamp": "2024-01-15T10:00:30Z",
      "model_used": "gpt-3.5-turbo",
      "quality_score": 0.85,
      "metadata": {...}
    },
    ...
  ],
  "source": "storage_service"
}
```

#### `GET /api/v1/bulk/tasks/{task_id}/history`
Historial de eventos de una tarea.

#### `GET /api/v1/bulk/tasks`
Listar todas las tareas.

#### `POST /api/v1/bulk/stop/{task_id}`
Detener generación de una tarea.

---

### Bulk AI System

#### `POST /api/v1/bulk-ai/process-query`
Procesar query con generación continua.

**Request:**
```json
{
  "query": "Historia de la programación",
  "max_documents": 100,
  "enable_continuous": true
}
```

**Response:**
```json
{
  "task_id": "task_xyz789",
  "status": "generating",
  "query": "Historia de la programación",
  "max_documents": 100,
  "message": "Continuous generation started"
}
```

#### `GET /api/v1/bulk-ai/status`
Estado del sistema Bulk AI.

#### `POST /api/v1/bulk-ai/stop-generation`
Detener generación continua.

#### `GET /api/v1/bulk-ai/performance`
Métricas de performance del Bulk AI.

---

### Enhanced Bulk AI

#### `POST /api/v1/enhanced-bulk-ai/process-query`
Versión mejorada con optimizaciones avanzadas.

#### `GET /api/v1/enhanced-bulk-ai/status`
Estado del sistema mejorado.

#### `GET /api/v1/enhanced-bulk-ai/benchmark`
Benchmark del sistema.

#### `GET /api/v1/enhanced-bulk-ai/models`
Modelos disponibles.

---

### Performance & Optimization

#### `GET /api/v1/performance/stats`
Estadísticas de optimización de performance.

#### `POST /api/v1/performance/optimize`
Trigger de optimización.

#### `GET /api/v1/cache/stats`
Estadísticas de cache.

#### `POST /api/v1/cache/clear`
Limpiar cache.

#### `GET /api/v1/batch/stats`
Estadísticas de batch processing.

#### `GET /api/v1/compression/stats`
Estadísticas de compresión.

#### `GET /api/v1/speed/stats`
Estadísticas de optimización de velocidad.

#### `POST /api/v1/speed/warmup`
Warmup del sistema.

#### `GET /api/v1/gpu/stats`
Estadísticas de GPU.

#### `GET /api/v1/memory/stats`
Estadísticas de memoria (NUEVO).

---

### AI & ML

#### `POST /api/v1/ai/process-text`
Procesar texto con AI.

#### `POST /api/v1/ai/reason`
Razonamiento con AI.

#### `POST /api/v1/ai/plan`
Planificación con AI.

#### `POST /api/v1/ml/train`
Entrenar modelo ML.

#### `POST /api/v1/ml/predict`
Predicción con ML.

---

### Security

#### `POST /api/v1/security/encrypt`
Encriptar datos (NUEVO).

#### `POST /api/v1/security/decrypt`
Desencriptar datos (NUEVO).

#### `POST /api/v1/security/hash-password`
Hash de contraseña.

#### `POST /api/v1/security/verify-password`
Verificar contraseña.

---

### Monitoring & Analytics

#### `GET /api/v1/monitoring/stats`
Estadísticas de monitoreo.

#### `GET /api/v1/monitoring/metrics`
Métricas detalladas.

#### `GET /api/v1/monitoring/alerts`
Alertas activas.

---

## 📊 Nuevos Endpoints Disponibles

### Memory Management
- `GET /api/v1/memory/stats` - Estadísticas de memoria
- `POST /api/v1/memory/optimize` - Optimizar memoria

### Event Bus
- `GET /api/v1/events/stats` - Estadísticas del event bus
- `GET /api/v1/events/history` - Historial de eventos

### Compression
- `POST /api/v1/compression/compress` - Comprimir datos
- `POST /api/v1/compression/decompress` - Descomprimir datos

### Connection Pool
- `GET /api/v1/pool/stats` - Estadísticas de connection pools

---

## 🔐 Seguridad

### API Key
Muchos endpoints requieren `X-API-Key` header si está configurado.

### Rate Limiting
- `/api/v1/bulk/generate`: 10 req/min
- Otros endpoints: 100 req/min

---

## 📖 Documentación Interactiva

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 🎯 Ejemplos de Uso

Ver `EJEMPLOS_GENERACION.md` para ejemplos completos.

































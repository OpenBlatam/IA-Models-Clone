# 📝 Ejemplos de Generación - Bulk TruthGPT

## 🎯 ¿Qué Genera el Sistema?

El sistema Bulk TruthGPT puede generar:

1. **Documentos de texto** - Artículos, ensayos, análisis
2. **Contenido continuo** - Generación masiva de documentos
3. **Análisis y procesamiento** - Análisis de texto, imágenes, audio
4. **Respuestas inteligentes** - Respuestas basadas en conocimiento
5. **Planificación y razonamiento** - Planes y razonamiento estructurado

## 🚀 Endpoints Principales de Generación

### 1. Generación Masiva Básica

**Endpoint:** `POST /api/v1/bulk/generate`

```bash
curl -X POST http://localhost:8000/api/v1/bulk/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explicar las ventajas de la inteligencia artificial en la medicina moderna",
    "config": {
      "max_documents": 10,
      "max_tokens": 2000,
      "temperature": 0.7,
      "format": "markdown"
    }
  }'
```

**Respuesta:**
```json
{
  "task_id": "abc123-def456-...",
  "status": "processing",
  "message": "Generación iniciada",
  "estimated_documents": 10
}
```

### 2. Generación Continua (Bulk AI)

**Endpoint:** `POST /api/v1/bulk-ai/process-query`

Genera documentos continuamente basados en una query.

```bash
curl -X POST http://localhost:8000/api/v1/bulk-ai/process-query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Historia de la programación desde sus inicios",
    "max_documents": 100,
    "enable_continuous": true,
    "config": {
      "temperature": 0.8,
      "diversity": "high"
    }
  }'
```

**Respuesta:**
```json
{
  "task_id": "task_xyz789",
  "status": "generating",
  "query": "Historia de la programación desde sus inicios",
  "generated_documents": 0,
  "target_documents": 100,
  "estimated_time_remaining": "15 minutos"
}
```

### 3. Generación Mejorada (Enhanced Bulk AI)

**Endpoint:** `POST /api/v1/enhanced-bulk-ai/process-query`

Versión mejorada con optimizaciones avanzadas.

```bash
curl -X POST http://localhost:8000/api/v1/enhanced-bulk-ai/process-query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Análisis profundo de machine learning",
    "max_documents": 50,
    "enable_ultra_optimization": true,
    "enable_quantum_optimization": true,
    "enable_edge_computing": true
  }'
```

### 4. Iniciar Generación Continua

**Endpoint:** `POST /api/v1/bulk-ai/start-continuous`

Inicia generación continua ilimitada.

```bash
curl -X POST http://localhost:8000/api/v1/bulk-ai/start-continuous \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tendencias de tecnología en 2024",
    "max_documents": 1000,
    "generation_interval": 0.1,
    "enable_real_time_monitoring": true
  }'
```

## 📊 Consultar Estado y Resultados

### Ver Estado de una Tarea

```bash
curl http://localhost:8000/api/v1/bulk/status/{task_id}
```

**Respuesta:**
```json
{
  "task_id": "abc123",
  "status": "completed",
  "progress": 100,
  "documents_generated": 10,
  "total_documents": 10,
  "started_at": "2024-01-15T10:00:00Z",
  "completed_at": "2024-01-15T10:05:00Z"
}
```

### Obtener Documentos Generados

```bash
curl http://localhost:8000/api/v1/bulk/documents/{task_id}?limit=10&offset=0
```

**Respuesta:**
```json
{
  "task_id": "abc123",
  "total_documents": 10,
  "documents": [
    {
      "id": "doc_1",
      "title": "Documento 1",
      "content": "Contenido generado...",
      "generated_at": "2024-01-15T10:00:30Z",
      "model_used": "truthgpt-v1",
      "quality_score": 0.85,
      "format": "markdown"
    },
    ...
  ]
}
```

### Historial de Eventos

```bash
curl http://localhost:8000/api/v1/bulk/tasks/{task_id}/history
```

## 🎨 Ejemplos de Queries

### 1. Artículos Educativos

```json
{
  "query": "Explicar cómo funciona el aprendizaje profundo paso a paso",
  "config": {
    "max_documents": 5,
    "format": "markdown",
    "style": "educational"
  }
}
```

### 2. Análisis Técnicos

```json
{
  "query": "Comparar arquitecturas de transformers: GPT vs BERT vs T5",
  "config": {
    "max_documents": 10,
    "format": "technical",
    "depth": "detailed"
  }
}
```

### 3. Contenido Creativo

```json
{
  "query": "Escribir historias cortas sobre inteligencia artificial",
  "config": {
    "max_documents": 20,
    "format": "narrative",
    "creativity": "high"
  }
}
```

### 4. Documentación Técnica

```json
{
  "query": "Crear guía de implementación de API REST con FastAPI",
  "config": {
    "max_documents": 3,
    "format": "documentation",
    "include_code": true
  }
}
```

## 🔍 Procesamiento Avanzado

### Procesar Texto

```bash
curl -X POST http://localhost:8000/api/v1/ai/process-text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Texto a procesar...",
    "operations": ["summarize", "analyze", "extract_keywords"]
  }'
```

### Razonamiento

```bash
curl -X POST http://localhost:8000/api/v1/ai/reason \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cuál es la mejor estrategia para escalar un sistema?",
    "context": "Sistema con 1M usuarios...",
    "reasoning_depth": "deep"
  }'
```

### Planificación

```bash
curl -X POST http://localhost:8000/api/v1/ai/plan \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Implementar sistema de recomendaciones",
    "constraints": ["Budget limitado", "Time: 3 meses"],
    "detail_level": "step_by_step"
  }'
```

## 📈 Monitoreo de Generación

### Ver Métricas de Rendimiento

```bash
curl http://localhost:8000/api/v1/bulk-ai/performance
```

### Estadísticas del Sistema

```bash
curl http://localhost:8000/api/v1/bulk/metrics
```

## 🛑 Detener Generación

```bash
curl -X POST http://localhost:8000/api/v1/bulk-ai/stop-generation \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "abc123"
  }'
```

## 💡 Tips de Uso

1. **Para generar pocos documentos**: Usa `max_documents: 5-10`
2. **Para generación masiva**: Usa `max_documents: 100-1000` con generación continua
3. **Para contenido técnico**: Aumenta `temperature` a 0.7-0.9
4. **Para contenido preciso**: Reduce `temperature` a 0.3-0.5
5. **Para monitoreo**: Usa `enable_real_time_monitoring: true`

## 🔗 Ver en Acción

1. Inicia el servidor: `python start.py`
2. Abre Swagger UI: http://localhost:8000/docs
3. Prueba los endpoints directamente desde la interfaz
4. Verifica los documentos en `storage/`

---

**¿Listo para generar?** ¡Inicia el servidor y prueba estos endpoints!

































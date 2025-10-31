# Ultimate TruthGPT API Documentation

## 🚀 API Reference - Final Definitive Version

Esta documentación describe la API completa de Ultimate TruthGPT, el sistema definitivo de generación masiva de documentos con IA avanzada.

## 📋 Tabla de Contenidos

- [Información General](#información-general)
- [Autenticación](#autenticación)
- [Endpoints Principales](#endpoints-principales)
- [Endpoints de Analytics](#endpoints-de-analytics)
- [Endpoints de AI History](#endpoints-de-ai-history)
- [Endpoints de Prompt Evolution](#endpoints-de-prompt-evolution)
- [Endpoints de Clustering](#endpoints-de-clustering)
- [Endpoints de Sentiment Analysis](#endpoints-de-sentiment-analysis)
- [Endpoints de Content Metrics](#endpoints-de-content-metrics)
- [Endpoints de Sistema](#endpoints-de-sistema)
- [Modelos de Datos](#modelos-de-datos)
- [Códigos de Error](#códigos-de-error)
- [Ejemplos de Uso](#ejemplos-de-uso)

## 📖 Información General

### Base URL
```
http://localhost:8000
```

### Versión
```
1.0.0-ultimate
```

### Formato de Respuesta
Todas las respuestas están en formato JSON.

### Rate Limiting
- **Límite**: 100 requests por minuto
- **Burst**: 200 requests
- **Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

## 🔐 Autenticación

Actualmente no se requiere autenticación, pero se recomienda configurar API keys para producción.

### Headers Opcionales
```http
X-API-Key: your-api-key
X-Client-ID: your-client-id
```

## 🎯 Endpoints Principales

### POST /generate/bulk

Genera múltiples documentos con todas las características avanzadas habilitadas.

#### Request Body
```json
{
  "query": "string (required)",
  "document_count": "integer (1-1000, default: 10)",
  "business_area": "string (default: 'general')",
  "enable_workflow": "boolean (default: true)",
  "enable_redundancy_detection": "boolean (default: true)",
  "enable_analytics": "boolean (default: true)",
  "enable_ai_history": "boolean (default: true)",
  "enable_prompt_evolution": "boolean (default: true)",
  "enable_clustering": "boolean (default: true)",
  "enable_sentiment_analysis": "boolean (default: true)",
  "enable_content_metrics": "boolean (default: true)",
  "streaming": "boolean (default: false)",
  "metadata": "object (optional)"
}
```

#### Response
```json
{
  "request_id": "string",
  "total_documents": "integer",
  "generated_documents": [
    {
      "document_id": "string",
      "content": "string",
      "quality_score": "float",
      "performance_score": "float",
      "sentiment_score": "float",
      "metrics": "object",
      "generated_at": "datetime",
      "processing_time": "float"
    }
  ],
  "analytics": "object",
  "clustering_results": "object",
  "sentiment_analysis": "object",
  "content_metrics": "object",
  "processing_time": "float",
  "created_at": "datetime"
}
```

#### Ejemplo de Request
```bash
curl -X POST "http://localhost:8000/generate/bulk" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Inteligencia artificial en el futuro",
    "document_count": 50,
    "business_area": "technology",
    "enable_workflow": true,
    "enable_analytics": true,
    "enable_clustering": true,
    "enable_sentiment_analysis": true,
    "enable_content_metrics": true
  }'
```

### POST /generate/bulk/stream

Genera documentos con respuesta en streaming.

#### Request Body
Mismo formato que `/generate/bulk`

#### Response
Stream de eventos Server-Sent Events (SSE):

```
data: {"type": "start", "request_id": "bulk_1234567890"}

data: {"type": "document", "document_id": "doc_1", "content": "...", "analysis": {...}}

data: {"type": "document", "document_id": "doc_2", "content": "...", "analysis": {...}}

data: {"type": "complete"}
```

#### Ejemplo de Request
```bash
curl -X POST "http://localhost:8000/generate/bulk/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Futuro de la tecnología",
    "document_count": 10,
    "business_area": "technology"
  }'
```

## 📊 Endpoints de Analytics

### GET /analytics/report

Obtiene el reporte completo de analytics.

#### Response
```json
{
  "report_generated_at": "datetime",
  "data_summary": {
    "total_analyses": "integer",
    "average_quality": "float",
    "average_performance": "float",
    "total_throughput": "float"
  },
  "quality_analysis": "object",
  "performance_analysis": "object",
  "throughput_analysis": "object",
  "trend_analysis": "object",
  "correlation_analysis": "object",
  "predictions": "object",
  "recommendations": "array"
}
```

### GET /analytics/predictions

Obtiene predicciones de analytics.

#### Response
```json
{
  "predictions_generated_at": "datetime",
  "quality_predictions": [
    {
      "timestamp": "datetime",
      "predicted_quality": "float",
      "confidence": "float"
    }
  ],
  "performance_predictions": [
    {
      "timestamp": "datetime",
      "predicted_performance": "float",
      "confidence": "float"
    }
  ],
  "throughput_predictions": [
    {
      "timestamp": "datetime",
      "predicted_throughput": "float",
      "confidence": "float"
    }
  ]
}
```

## 🤖 Endpoints de AI History

### GET /ai-history/analysis

Obtiene análisis de evolución de IA.

#### Response
```json
{
  "analysis_generated_at": "datetime",
  "total_models": "integer",
  "model_evolution": [
    {
      "model_name": "string",
      "version": "string",
      "performance_score": "float",
      "quality_score": "float",
      "timestamp": "datetime"
    }
  ],
  "evolution_trends": "object",
  "performance_comparison": "object",
  "recommendations": "array"
}
```

## 🧬 Endpoints de Prompt Evolution

### GET /prompt-evolution/analysis

Obtiene análisis de evolución de prompts.

#### Response
```json
{
  "analysis_generated_at": "datetime",
  "total_prompts": "integer",
  "evolution_cycles": "integer",
  "prompt_evolution": [
    {
      "prompt_id": "string",
      "version": "string",
      "effectiveness_score": "float",
      "optimization_score": "float",
      "timestamp": "datetime"
    }
  ],
  "optimization_strategies": "array",
  "evolution_trends": "object",
  "recommendations": "array"
}
```

## 🔗 Endpoints de Clustering

### GET /clustering/analysis

Obtiene análisis de clustering.

#### Response
```json
{
  "analysis_generated_at": "datetime",
  "total_clusters": "integer",
  "clustering_methods": [
    {
      "method": "string",
      "clusters_found": "integer",
      "silhouette_score": "float",
      "effectiveness": "float"
    }
  ],
  "similarity_analysis": "object",
  "cluster_analysis": "object",
  "recommendations": "array"
}
```

## 😊 Endpoints de Sentiment Analysis

### GET /sentiment/analysis

Obtiene reporte de análisis de sentimientos.

#### Response
```json
{
  "analysis_generated_at": "datetime",
  "total_documents": "integer",
  "sentiment_analysis": {
    "overall_sentiment": "float",
    "sentiment_distribution": "object",
    "sentiment_trends": "array"
  },
  "emotion_analysis": {
    "emotion_distribution": "object",
    "dominant_emotions": "array",
    "emotion_trends": "array"
  },
  "tone_analysis": "object",
  "attitude_analysis": "object",
  "sarcasm_analysis": "object",
  "irony_analysis": "object",
  "recommendations": "array"
}
```

## 📏 Endpoints de Content Metrics

### GET /content-metrics/report

Obtiene reporte de métricas de contenido.

#### Response
```json
{
  "report_generated_at": "datetime",
  "data_summary": {
    "total_metrics_analyzed": "integer",
    "recent_metrics": "integer",
    "trend_analyses": "integer",
    "optimizations_applied": "integer"
  },
  "average_metrics": {
    "quality_score": "float",
    "performance_score": "float",
    "readability_score": "float",
    "engagement_score": "float",
    "clarity_score": "float",
    "coherence_score": "float",
    "complexity_score": "float",
    "structure_score": "float"
  },
  "recent_trends": "array",
  "recent_optimizations": "array",
  "analysis_statistics": "object",
  "recommendations": "array"
}
```

## 🖥️ Endpoints de Sistema

### GET /health

Verificación de salud del sistema.

#### Response
```json
{
  "status": "string (healthy|degraded|unhealthy)",
  "components": {
    "orchestrator": "object",
    "redundancy_detector": "object",
    "analytics": "object",
    "ai_history_analyzer": "object",
    "prompt_evolver": "object",
    "clusterer": "object",
    "sentiment_analyzer": "object",
    "content_metrics": "object"
  },
  "statistics": {
    "total_components": "integer",
    "healthy_components": "integer",
    "system_uptime": "string",
    "total_requests": "integer",
    "active_workflows": "integer"
  },
  "health_checks": {
    "orchestrator": "boolean",
    "redundancy_detector": "boolean",
    "analytics": "boolean",
    "ai_history_analyzer": "boolean",
    "prompt_evolver": "boolean",
    "clusterer": "boolean",
    "sentiment_analyzer": "boolean",
    "content_metrics": "boolean",
    "redis": "boolean"
  },
  "timestamp": "datetime"
}
```

### GET /system/info

Información del sistema.

#### Response
```json
{
  "application": "string",
  "version": "string",
  "description": "string",
  "components": {
    "workflow_orchestrator": "string",
    "redundancy_detector": "string",
    "advanced_analytics": "string",
    "ai_history_analyzer": "string",
    "prompt_evolver": "string",
    "smart_clusterer": "string",
    "sentiment_analyzer": "string",
    "content_metrics": "string"
  },
  "features": "array",
  "capabilities": {
    "max_documents_per_request": "integer",
    "streaming_support": "boolean",
    "real_time_analysis": "boolean",
    "ml_predictions": "boolean",
    "advanced_caching": "boolean",
    "multi_modal_analysis": "boolean"
  }
}
```

## 📋 Modelos de Datos

### BulkGenerationRequest
```json
{
  "query": "string",
  "document_count": "integer",
  "business_area": "string",
  "enable_workflow": "boolean",
  "enable_redundancy_detection": "boolean",
  "enable_analytics": "boolean",
  "enable_ai_history": "boolean",
  "enable_prompt_evolution": "boolean",
  "enable_clustering": "boolean",
  "enable_sentiment_analysis": "boolean",
  "enable_content_metrics": "boolean",
  "streaming": "boolean",
  "metadata": "object"
}
```

### DocumentGenerationResponse
```json
{
  "document_id": "string",
  "content": "string",
  "quality_score": "float",
  "performance_score": "float",
  "sentiment_score": "float",
  "metrics": "object",
  "generated_at": "datetime",
  "processing_time": "float"
}
```

### BulkGenerationResponse
```json
{
  "request_id": "string",
  "total_documents": "integer",
  "generated_documents": "array",
  "analytics": "object",
  "clustering_results": "object",
  "sentiment_analysis": "object",
  "content_metrics": "object",
  "processing_time": "float",
  "created_at": "datetime"
}
```

## ❌ Códigos de Error

### 400 Bad Request
```json
{
  "detail": "Invalid request parameters"
}
```

### 422 Unprocessable Entity
```json
{
  "detail": [
    {
      "loc": ["body", "document_count"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error occurred"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Service temporarily unavailable"
}
```

## 💡 Ejemplos de Uso

### Ejemplo 1: Generación Básica
```python
import requests

response = requests.post("http://localhost:8000/generate/bulk", json={
    "query": "Inteligencia artificial",
    "document_count": 10,
    "business_area": "technology"
})

result = response.json()
print(f"Generados {result['total_documents']} documentos")
```

### Ejemplo 2: Generación con Todas las Características
```python
import requests

response = requests.post("http://localhost:8000/generate/bulk", json={
    "query": "Futuro de la tecnología",
    "document_count": 50,
    "business_area": "technology",
    "enable_workflow": True,
    "enable_analytics": True,
    "enable_clustering": True,
    "enable_sentiment_analysis": True,
    "enable_content_metrics": True,
    "metadata": {
        "client": "demo",
        "priority": "high"
    }
})

result = response.json()
print(f"Calidad promedio: {result['analytics']['average_quality']}")
```

### Ejemplo 3: Generación con Streaming
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/generate/bulk/stream",
    json={
        "query": "Innovación tecnológica",
        "document_count": 20,
        "business_area": "technology"
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8').replace('data: ', ''))
        if data['type'] == 'document':
            print(f"Documento: {data['document_id']}")
```

### Ejemplo 4: Obtener Analytics
```python
import requests

# Obtener reporte de analytics
response = requests.get("http://localhost:8000/analytics/report")
analytics = response.json()
print(f"Análisis totales: {analytics['data_summary']['total_analyses']}")

# Obtener predicciones
response = requests.get("http://localhost:8000/analytics/predictions")
predictions = response.json()
print(f"Predicciones de calidad: {len(predictions['quality_predictions'])}")
```

### Ejemplo 5: Verificar Salud del Sistema
```python
import requests

response = requests.get("http://localhost:8000/health")
health = response.json()

print(f"Estado: {health['status']}")
print(f"Componentes saludables: {health['statistics']['healthy_components']}")
print(f"Componentes totales: {health['statistics']['total_components']}")
```

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
# API Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1
LOG_LEVEL=info

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# AI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
DEFAULT_MODEL=gpt-3.5-turbo
TEMPERATURE=0.7
MAX_TOKENS=4000

# Performance Configuration
MAX_DOCUMENTS_PER_REQUEST=1000
MAX_CONCURRENT_REQUESTS=50
CACHE_TTL=3600
BATCH_SIZE=10

# Feature Flags
ENABLE_ANALYTICS=true
ENABLE_CLUSTERING=true
ENABLE_SENTIMENT_ANALYSIS=true
ENABLE_CONTENT_METRICS=true
ENABLE_AI_HISTORY=true
ENABLE_PROMPT_EVOLUTION=true
```

### Configuración de CORS
```python
# Para desarrollo
CORS_ORIGINS = ["*"]

# Para producción
CORS_ORIGINS = ["https://yourdomain.com", "https://app.yourdomain.com"]
```

## 📈 Monitoreo y Métricas

### Métricas Disponibles
- **Rendimiento**: Tiempo de procesamiento, throughput, latencia
- **Calidad**: Puntuaciones de calidad, coherencia, claridad
- **Analytics**: Predicciones, tendencias, correlaciones
- **Clustering**: Similitud, agrupaciones, patrones
- **Sentiment**: Sentimientos, emociones, actitudes
- **Métricas de Contenido**: Legibilidad, estructura, completitud

### Health Checks
```bash
# Verificar salud del sistema
curl http://localhost:8000/health

# Verificar información del sistema
curl http://localhost:8000/system/info
```

## 🚀 Mejores Prácticas

### 1. Optimización de Requests
- Use `streaming=true` para requests grandes
- Configure `document_count` apropiadamente
- Use `metadata` para tracking

### 2. Manejo de Errores
- Implemente retry logic
- Maneje rate limiting
- Verifique health checks

### 3. Monitoreo
- Monitoree métricas de rendimiento
- Configure alertas
- Use health checks

### 4. Seguridad
- Use HTTPS en producción
- Configure CORS apropiadamente
- Implemente rate limiting

## 📞 Soporte

Para soporte y preguntas:
- **Documentación**: [GitHub Docs](https://github.com/your-repo/docs)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@yourdomain.com

---

**Ultimate TruthGPT API** - La API definitiva para generación masiva de documentos con IA avanzada. 🚀


























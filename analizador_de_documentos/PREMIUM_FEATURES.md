# 💎 Funcionalidades Premium - Document Analyzer

## ✨ Nuevas Funcionalidades Premium Implementadas

### 1. 🌐 API REST Completa
- **API REST completa**: Endpoints para todas las funcionalidades
- **FastAPI**: Framework moderno y rápido
- **Documentación automática**: Swagger UI integrado
- **Múltiples endpoints**: Análisis, calidad, gramática, versiones, batch, etc.
- **Autenticación**: Soporte para autenticación (HTTPBearer)

```python
from analizador_de_documentos.core.document_analyzer import DocumentAnalyzer
from analizador_de_documentos.core.document_api import DocumentAPIServer

analyzer = DocumentAnalyzer()

# Crear servidor API
api_server = analyzer.create_api_server(host="0.0.0.0", port=8000)

# Ejecutar servidor
api_server.run()

# O usar directamente
server = DocumentAPIServer(analyzer, host="0.0.0.0", port=8000)
server.run()
```

**Endpoints disponibles:**
- `POST /api/v1/analyze` - Analizar documento
- `POST /api/v1/analyze/quality` - Analizar calidad
- `POST /api/v1/analyze/grammar` - Analizar gramática
- `POST /api/v1/versions/add` - Agregar versión
- `GET /api/v1/versions/{doc_id}/compare` - Comparar versiones
- `POST /api/v1/batch/analyze` - Análisis batch
- `POST /api/v1/recommendations/generate` - Generar recomendaciones
- `GET /api/v1/collaboration/{doc_id}` - Análisis de colaboración
- `GET /api/v1/metrics/dashboard` - Dashboard de métricas
- `GET /api/v1/metrics/statistics` - Estadísticas
- `GET /api/v1/health` - Health check

### 2. 🔔 Sistema de Webhooks
- **Notificaciones en tiempo real**: Webhooks para eventos
- **Múltiples eventos**: Diferentes tipos de eventos
- **Retry automático**: Reintentos con exponential backoff
- **Autenticación**: Soporte para secrets
- **Historial**: Registro de eventos enviados

```python
from analizador_de_documentos.core.document_webhooks import DocumentEvents

analyzer = DocumentAnalyzer()

# Registrar webhook
analyzer.register_webhook(
    url="https://api.example.com/webhooks/document",
    events=[DocumentEvents.ANALYSIS_COMPLETE, DocumentEvents.QUALITY_CHECK],
    secret="your-secret-key",
    timeout=30
)

# Disparar evento automáticamente después de análisis
result = await analyzer.analyze_document(document_content="...")
await analyzer.trigger_webhook_event(
    DocumentEvents.ANALYSIS_COMPLETE,
    document_id="doc_123",
    data={"quality_score": 85.5, "processing_time": 2.3}
)

# Eventos disponibles
# - DocumentEvents.ANALYSIS_COMPLETE
# - DocumentEvents.QUALITY_CHECK
# - DocumentEvents.GRAMMAR_CHECK
# - DocumentEvents.VERSION_ADDED
# - DocumentEvents.VERSION_COMPARED
# - DocumentEvents.COLLABORATION_ANALYZED
# - DocumentEvents.RECOMMENDATIONS_GENERATED
# - DocumentEvents.BATCH_COMPLETE
# - DocumentEvents.ERROR
```

### 3. 🤖 Machine Learning y Predicciones
- **Predicción de calidad**: Predecir calidad antes de análisis completo
- **Predicción de tiempo**: Estimar tiempo de procesamiento
- **Tamaño óptimo de batch**: Predecir tamaño óptimo
- **Análisis de tendencias**: Identificar tendencias en métricas
- **Datos de entrenamiento**: Recolección automática

```python
# Predecir calidad de documento
features = {
    "word_count": 1500,
    "sentence_count": 50,
    "paragraph_count": 10,
    "has_structure": True,
    "has_keywords": True
}

prediction = await analyzer.predict_document_quality(features)
print(f"Calidad predicha: {prediction.prediction:.1f}/100")
print(f"Confianza: {prediction.confidence:.1%}")

# Predecir tiempo de procesamiento
time_prediction = await analyzer.predict_processing_time(
    document_length=5000,
    tasks_count=3
)
print(f"Tiempo predicho: {time_prediction.prediction:.2f}s")
print(f"Confianza: {time_prediction.confidence:.1%}")

# Analizar tendencias
metrics_data = [
    {"date": "2024-01-01", "quality_score": 75, "processing_time": 2.1},
    {"date": "2024-01-02", "quality_score": 78, "processing_time": 2.0},
    {"date": "2024-01-03", "quality_score": 82, "processing_time": 1.9},
]

trends = await analyzer.analyze_trends(metrics_data, period_days=30)
print(f"Tendencia calidad: {trends['trends']['quality']['direction']}")
print(f"Cambio: {trends['trends']['quality']['change']:.1f}")
print(f"Predicción próxima: {trends['predictions']['next_quality']:.1f}")
```

## 🎯 Ejemplos Completos

### Ejemplo 1: API REST Completa
```python
from analizador_de_documentos.core.document_analyzer import DocumentAnalyzer

analyzer = DocumentAnalyzer()

# Crear y ejecutar servidor API
api_server = analyzer.create_api_server(host="0.0.0.0", port=8000)

# El servidor estará disponible en:
# http://localhost:8000
# Documentación: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc

# Ejecutar
api_server.run()
```

**Uso desde cliente:**
```python
import requests

# Analizar documento
response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json={
        "document_content": "Contenido del documento...",
        "tasks": ["classification", "summarization"]
    }
)

result = response.json()
print(f"Success: {result['success']}")
print(f"Data: {result['data']}")
```

### Ejemplo 2: Webhooks con Eventos
```python
analyzer = DocumentAnalyzer()

# Registrar múltiples webhooks
analyzer.register_webhook(
    url="https://api.example.com/webhooks/quality",
    events=[DocumentEvents.QUALITY_CHECK],
    secret="secret1"
)

analyzer.register_webhook(
    url="https://api.example.com/webhooks/analysis",
    events=[DocumentEvents.ANALYSIS_COMPLETE],
    secret="secret2"
)

# Analizar documento (dispara webhooks automáticamente)
result = await analyzer.analyze_document(document_content="...")
quality = await analyzer.analyze_quality(result.content)

# Disparar eventos manualmente
await analyzer.trigger_webhook_event(
    DocumentEvents.ANALYSIS_COMPLETE,
    document_id="doc_123",
    data={
        "classification": result.classification,
        "summary": result.summary
    }
)

await analyzer.trigger_webhook_event(
    DocumentEvents.QUALITY_CHECK,
    document_id="doc_123",
    data={
        "quality_score": quality.overall_score,
        "readability": quality.readability_score
    }
)

# Ver estadísticas de webhooks
if analyzer.webhook_manager:
    stats = analyzer.webhook_manager.get_webhook_stats()
    print(f"Total webhooks: {stats['total_webhooks']}")
    print(f"Eventos enviados: {stats['total_events']}")
```

### Ejemplo 3: ML y Predicciones
```python
analyzer = DocumentAnalyzer()

# Predecir antes de analizar
features = {
    "word_count": len(document_content.split()),
    "sentence_count": len(document_content.split('.')),
    "paragraph_count": len(document_content.split('\n\n')),
    "has_structure": bool(re.search(r'^#+\s', document_content, re.MULTILINE)),
    "has_keywords": True
}

quality_prediction = await analyzer.predict_document_quality(features)
print(f"Calidad predicha: {quality_prediction.prediction:.1f}/100")

# Predecir tiempo de procesamiento
time_prediction = await analyzer.predict_processing_time(
    document_length=len(document_content),
    tasks_count=3
)
print(f"Tiempo estimado: {time_prediction.prediction:.2f}s")

# Analizar documento real
import time
start = time.time()
result = await analyzer.analyze_document(document_content)
actual_time = time.time() - start
quality = await analyzer.analyze_quality(result.content)

# Registrar para entrenamiento
analyzer.ml_predictor.record_training_data(
    features=features,
    actual_quality=quality.overall_score,
    actual_processing_time=actual_time
)

# Analizar tendencias
dashboard = await analyzer.generate_metrics_dashboard(days=30)
metrics_data = [
    {
        "date": str(dashboard.start_date.date()),
        "quality_score": dashboard.average_quality_score,
        "processing_time": dashboard.average_processing_time
    }
]

trends = await analyzer.analyze_trends(metrics_data, period_days=30)
print(f"Tendencia: {trends['trends']['quality']['direction']}")
print(f"Predicción: {trends['predictions']['next_quality']:.1f}")
```

## 📊 Integración Completa

### Workflow Premium Completo
```python
analyzer = DocumentAnalyzer()

# 1. Configurar webhooks
analyzer.register_webhook(
    url="https://api.example.com/webhooks",
    events=["*"],
    secret="secret-key"
)

# 2. Predecir calidad
features = extract_features(document_content)
prediction = await analyzer.predict_document_quality(features)

# 3. Analizar documento
result = await analyzer.analyze_document(document_content)
quality = await analyzer.analyze_quality(result.content)
grammar = await analyzer.analyze_grammar(result.content)

# 4. Disparar webhooks
await analyzer.trigger_webhook_event(
    DocumentEvents.ANALYSIS_COMPLETE,
    document_id="doc_123",
    data={"result": result, "quality": quality}
)

# 5. Registrar métricas
analyzer.record_analysis_for_metrics(
    "doc_123", result, processing_time,
    quality.overall_score, grammar.overall_score
)

# 6. Generar recomendaciones
recommendations = await analyzer.generate_recommendations(
    result, quality, grammar
)

# 7. Exportar
analyzer.export_results({
    "prediction": prediction,
    "analysis": result,
    "quality": quality,
    "grammar": grammar,
    "recommendations": recommendations
}, "premium_report.json")
```

## 🚀 Ventajas Premium

- **API REST**: Acceso externo completo
- **Webhooks**: Notificaciones en tiempo real
- **ML Predictions**: Predicciones inteligentes
- **Trend Analysis**: Análisis de tendencias
- **Automatización**: Procesos completamente automatizados
- **Escalabilidad**: Listo para producción enterprise

---

**Estado**: ✅ **Todas las Funcionalidades Premium Implementadas**

















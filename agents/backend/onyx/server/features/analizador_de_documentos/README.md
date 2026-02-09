# Analizador de Documentos Inteligente

Sistema avanzado de análisis de documentos con capacidades de fine-tuning y aprendizaje adaptativo. Proporciona análisis completo de documentos incluyendo clasificación, resumen, extracción de información, análisis de sentimiento y más.

## 🚀 Características

- **Análisis Multi-Tarea**: Clasificación, resumen, extracción de keywords, análisis de sentimiento, reconocimiento de entidades, modelado de temas
- **Fine-Tuning**: Sistema completo para entrenar modelos personalizados en tus propios datos
- **Multi-Formato**: Soporta PDF, DOCX, TXT, HTML, Markdown, JSON, XML, CSV
- **API REST**: Endpoints completos para integración fácil
- **Embeddings**: Generación de embeddings para búsqueda semántica y comparación
- **Question-Answering**: Respuesta a preguntas sobre documentos
- **Sistema de Caché**: Caché inteligente con múltiples backends (memoria, disco, Redis)
- **Procesamiento por Lotes**: Procesamiento paralelo optimizado para múltiples documentos
- **Rate Limiting**: Protección contra abuso con rate limiting configurable
- **Métricas y Monitoring**: Sistema completo de métricas de rendimiento y estadísticas
- **Optimizaciones de Rendimiento**: Procesamiento paralelo, caching, y optimizaciones de memoria
- **Comparación de Documentos**: Comparación semántica y detección de similitud
- **Extracción Estructurada**: Extracción de información según schemas personalizados
- **Análisis de Estilo**: Análisis de legibilidad, complejidad y calidad de escritura
- **Exportación Multi-formato**: Exporta resultados en JSON, CSV, Markdown, HTML
- **Búsqueda Semántica**: Encuentra documentos similares usando embeddings
- **Detección de Plagio**: Detecta posible plagio comparando con corpus de referencia
- **Motor de Búsqueda Semántica Avanzada**: Búsqueda híbrida con índices vectoriales
- **Automatización de Workflows**: Workflows personalizables para análisis automatizados
- **Bases de Datos Vectoriales**: Integración con Pinecone, Weaviate, Chroma, Qdrant, Milvus
- **Detección de Anomalías**: Detección automática de anomalías e inconsistencias
- **Análisis Predictivo**: Forecasting y predicciones basadas en tendencias históricas
- **Análisis de Imágenes**: Detección de objetos, OCR, análisis de colores en imágenes
- **Sistema de Alertas**: Alertas configurables con reglas personalizadas
- **Auditoría Completa**: Registro de todas las acciones del sistema
- **WebSockets**: Updates en tiempo real vía WebSocket

## 📋 Requisitos

- Python 3.8+
- CUDA (opcional, para GPU)
- 8GB+ RAM recomendado
- 10GB+ espacio en disco para modelos

## 🛠️ Instalación

### 1. Instalar dependencias

```bash
cd analizador_de_documentos
pip install -r requirements.txt
```

### 2. Configurar variables de entorno (opcional)

Crear archivo `.env`:

```env
HOST=0.0.0.0
PORT=8000
MODEL_NAME=bert-base-multilingual-cased
DEVICE=cuda  # o cpu
```

## 🚀 Uso Rápido

### Iniciar servidor

```bash
python main.py
```

El servidor estará disponible en `http://localhost:8000`

### Documentación API

Visita `http://localhost:8000/docs` para la documentación interactiva de la API.

## 📖 Uso de la API

### Analizar un documento

```bash
curl -X POST "http://localhost:8000/api/analizador-documentos/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "document_content": "Este es un documento sobre inteligencia artificial...",
    "tasks": ["classification", "summarization", "keywords"]
  }'
```

### Clasificar texto

```bash
curl -X POST "http://localhost:8000/api/analizador-documentos/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Este documento trata sobre tecnología"
  }'
```

### Generar resumen

```bash
curl -X POST "http://localhost:8000/api/analizador-documentos/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Texto largo a resumir...",
    "max_length": 150,
    "min_length": 30
  }'
```

### Subir y analizar archivo

```bash
curl -X POST "http://localhost:8000/api/analizador-documentos/analyze/upload" \
  -F "file=@documento.pdf" \
  -F "tasks=classification,summarization"
```

## 🔧 Fine-Tuning

### Preparar datos de entrenamiento

Los datos deben estar en formato JSON:

```json
[
  {"text": "Texto ejemplo 1", "label": 0},
  {"text": "Texto ejemplo 2", "label": 1},
  ...
]
```

### Entrenar modelo

```bash
python training/train_model.py \
  --data datos_entrenamiento.json \
  --num-labels 3 \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 2e-5
```

### Crear datos de ejemplo

```bash
python training/train_model.py \
  --create-sample \
  --data datos_ejemplo.json \
  --sample-size 100
```

### Usar modelo fine-tuned

```python
from core.document_analyzer import DocumentAnalyzer

analyzer = DocumentAnalyzer(
    fine_tuned_model_path="./models/fine_tuned/model_entrenado"
)

result = await analyzer.analyze_document(
    document_content="Texto a analizar..."
)
```

## 📚 Estructura del Proyecto

```
analizador_de_documentos/
├── core/
│   ├── document_analyzer.py      # Analizador principal
│   ├── fine_tuning_model.py       # Sistema de fine-tuning
│   ├── document_processor.py      # Procesador de documentos
│   └── embedding_generator.py     # Generador de embeddings
├── api/
│   └── routes.py                  # Endpoints REST API
├── training/
│   └── train_model.py             # Script de entrenamiento
├── config/
│   └── config.yaml                # Configuración
├── models/                         # Modelos guardados
│   ├── cache/                      # Cache de modelos
│   └── fine_tuned/                 # Modelos fine-tuned
├── main.py                         # Aplicación principal
├── requirements.txt                # Dependencias
└── README.md                       # Esta documentación
```

## 🎯 Tareas de Análisis Disponibles

- **classification**: Clasificar documento en categorías
- **summarization**: Generar resumen del documento
- **keyword_extraction**: Extraer palabras clave
- **sentiment**: Análisis de sentimiento
- **entity_recognition**: Reconocimiento de entidades nombradas
- **topic_modeling**: Extracción de temas
- **question_answering**: Responder preguntas sobre el documento

## 🔌 Integración con Python

```python
from core.document_analyzer import DocumentAnalyzer, AnalysisTask

# Inicializar analizador
analyzer = DocumentAnalyzer()

# Analizar documento
result = await analyzer.analyze_document(
    document_content="Texto del documento...",
    tasks=[
        AnalysisTask.CLASSIFICATION,
        AnalysisTask.SUMMARIZATION,
        AnalysisTask.KEYWORD_EXTRACTION
    ]
)

print(f"Resumen: {result.summary}")
print(f"Keywords: {result.keywords}")
print(f"Clasificación: {result.classification}")
```

## 🚀 Nuevas Características Mejoradas

### Sistema de Caché

El sistema ahora incluye caché inteligente que mejora significativamente el rendimiento:

```python
# Configurar backend de caché (memoria, disco, redis, auto)
import os
os.environ["CACHE_BACKEND"] = "redis"  # o "memory", "disk", "auto"

# El caché se usa automáticamente en todas las operaciones
```

### Procesamiento por Lotes

Procesa múltiples documentos en paralelo:

```bash
curl -X POST "http://localhost:8000/api/analizador-documentos/batch/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"content": "Documento 1...", "document_id": "doc1"},
      {"content": "Documento 2...", "document_id": "doc2"}
    ],
    "tasks": ["classification", "summarization"],
    "max_workers": 10,
    "batch_size": 100
  }'
```

### Métricas y Monitoring

Accede a métricas de rendimiento en tiempo real:

```bash
# Ver todas las métricas
curl http://localhost:8000/api/analizador-documentos/metrics/

# Ver estadísticas de rendimiento
curl http://localhost:8000/api/analizador-documentos/metrics/performance

# Health check detallado
curl http://localhost:8000/api/analizador-documentos/metrics/health
```

### Rate Limiting

Protección automática contra abuso:

- Límite por defecto: 50 peticiones por minuto por IP
- Configurable por endpoint
- Headers de rate limit en respuestas

## 📊 Modelos Soportados

- `bert-base-multilingual-cased` (por defecto)
- `distilbert-base-multilingual-cased`
- `xlm-roberta-base`
- Cualquier modelo compatible con HuggingFace Transformers

## 🎓 Fine-Tuning Avanzado

### Configuración personalizada

```python
from core.fine_tuning_model import FineTuningModel, FineTuningConfig

config = FineTuningConfig(
    model_name="bert-base-multilingual-cased",
    num_labels=5,
    max_length=512,
    batch_size=32,
    learning_rate=3e-5,
    num_epochs=10,
    output_dir="./mi_modelo_personalizado"
)

model = FineTuningModel(config=config)

# Preparar datos
train_dataset, eval_dataset = model.prepare_dataset(texts, labels)

# Entrenar
results = model.train(train_dataset, eval_dataset)
```

## 🐛 Troubleshooting

### Error: CUDA out of memory

Reducir `batch_size` en la configuración o usar CPU:

```python
analyzer = DocumentAnalyzer(device="cpu")
```

### Error: Modelo no encontrado

Los modelos se descargan automáticamente la primera vez. Si hay problemas, descargar manualmente:

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModel.from_pretrained("model-name")
```

### Documentos muy largos

El sistema procesa documentos grandes automáticamente dividiéndolos en chunks. Para documentos muy largos (>10MB), considerar pre-procesamiento.

## 📝 Ejemplos

Ver carpeta `examples/` para ejemplos completos de uso.

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es parte de Blatam Academy.

## 🔗 Referencias

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📧 Soporte

Para soporte y preguntas, contacta al equipo de Blatam Academy.

---

**Versión**: 1.0.0  
**Última actualización**: 2024


# Advanced AI Document Processor

Un sistema avanzado de procesamiento de documentos con capacidades de IA/ML, OCR, análisis de contenido y procesamiento en lote.

## 🚀 Características Principales

### 📄 Procesamiento de Documentos
- **Soporte Multi-formato**: PDF, DOCX, DOC, TXT, RTF, ODT, PPTX, XLSX, CSV
- **OCR Avanzado**: Reconocimiento óptico de caracteres con EasyOCR y Tesseract
- **Extracción de Metadatos**: Información completa del documento
- **Procesamiento en Lote**: Manejo eficiente de múltiples documentos

### 🤖 Análisis con IA/ML
- **Clasificación de Documentos**: Categorización automática usando transformers
- **Extracción de Entidades**: Identificación de personas, lugares, organizaciones
- **Análisis de Sentimientos**: Evaluación del tono y emociones del contenido
- **Modelado de Temas**: Identificación de temas principales (LDA)
- **Resumen Automático**: Generación de resúmenes usando BART
- **Extracción de Palabras Clave**: Identificación de términos importantes
- **Análisis de Contenido**: Evaluación de calidad y legibilidad

### 🔍 Búsqueda y Comparación
- **Búsqueda Semántica**: Búsqueda inteligente basada en significado
- **Comparación de Documentos**: Análisis de similitud y detección de plagio
- **Base de Datos Vectorial**: Almacenamiento eficiente de embeddings

### ⚡ Características Avanzadas
- **Procesamiento Asíncrono**: Manejo eficiente de múltiples tareas
- **WebSockets**: Actualizaciones en tiempo real
- **Caché Inteligente**: Optimización de rendimiento con Redis
- **Exportación**: Múltiples formatos de salida (JSON, CSV, XLSX, PDF, DOCX)
- **API RESTful**: Interfaz completa y bien documentada

## 🛠️ Tecnologías Utilizadas

### Core Framework
- **FastAPI**: Framework web moderno y rápido
- **Pydantic**: Validación de datos y configuración
- **Uvicorn**: Servidor ASGI de alto rendimiento

### IA/ML y NLP
- **Transformers**: Modelos de lenguaje pre-entrenados
- **PyTorch**: Framework de deep learning
- **spaCy**: Procesamiento de lenguaje natural
- **NLTK**: Herramientas de NLP
- **scikit-learn**: Machine learning
- **sentence-transformers**: Embeddings semánticos

### Procesamiento de Documentos
- **PyPDF2/pdfplumber**: Procesamiento de PDFs
- **python-docx**: Procesamiento de documentos Word
- **openpyxl**: Procesamiento de Excel
- **python-pptx**: Procesamiento de PowerPoint
- **EasyOCR/Tesseract**: Reconocimiento óptico de caracteres

### Base de Datos y Almacenamiento
- **SQLAlchemy**: ORM para bases de datos
- **Redis**: Caché en memoria
- **ChromaDB**: Base de datos vectorial
- **FAISS**: Búsqueda de similitud

### Monitoreo y Logging
- **Prometheus**: Métricas del sistema
- **Sentry**: Monitoreo de errores
- **structlog**: Logging estructurado

## 📦 Instalación

### Requisitos Previos
- Python 3.8+
- pip
- Redis (opcional, para caché)
- Base de datos SQL (opcional)

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone <repository-url>
cd ai-document-processor

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelos de spaCy
python -m spacy download en_core_web_sm

# Descargar datos de NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

### Configuración

1. Copiar el archivo de configuración:
```bash
cp env.example .env
```

2. Editar `.env` con tus configuraciones:
```bash
# Configuración básica
APP_NAME="Advanced AI Document Processor"
DEBUG=true
HOST=0.0.0.0
PORT=8001

# Base de datos (opcional)
DATABASE_URL=sqlite:///./document_processor.db

# Redis (opcional)
REDIS_URL=redis://localhost:6379

# Configuración de IA/ML
EMBEDDING_MODEL=all-MiniLM-L6-v2
CLASSIFICATION_MODEL=distilbert-base-uncased
SUMMARIZATION_MODEL=facebook/bart-large-cnn
```

## 🚀 Uso

### Iniciar el Servidor

```bash
# Desarrollo
python app.py

# Producción
uvicorn app:app --host 0.0.0.0 --port 8001 --workers 4
```

### Acceder a la Documentación

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## 📚 API Endpoints

### Documentos

#### Subir Documento
```bash
curl -X POST "http://localhost:8001/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "analysis_types=content_analysis,classification,ocr" \
  -F "language=en"
```

#### Subir Lote de Documentos
```bash
curl -X POST "http://localhost:8001/api/v1/documents/upload/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.docx" \
  -F "batch_name=my_batch" \
  -F "analysis_types=content_analysis,classification"
```

#### Obtener Resultado
```bash
curl -X GET "http://localhost:8001/api/v1/documents/{document_id}"
```

### Análisis Específicos

#### Análisis OCR
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/ocr"
```

#### Clasificación
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/classification"
```

#### Extracción de Entidades
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/entities"
```

#### Análisis de Sentimientos
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/sentiment"
```

#### Modelado de Temas
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/topics"
```

#### Resumen
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/summary"
```

#### Palabras Clave
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/keywords"
```

#### Análisis de Contenido
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/content"
```

### Búsqueda y Comparación

#### Búsqueda Semántica
```bash
curl -X POST "http://localhost:8001/api/v1/documents/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "documentos sobre inteligencia artificial",
    "search_type": "semantic",
    "limit": 10
  }'
```

#### Comparar Documentos
```bash
curl -X POST "http://localhost:8001/api/v1/documents/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": ["doc1_id", "doc2_id"],
    "comparison_type": "similarity",
    "threshold": 0.8
  }'
```

### Exportación

#### Exportar Resultados
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/export" \
  -F "format=json"
```

### Sistema

#### Estado del Sistema
```bash
curl -X GET "http://localhost:8001/health"
```

#### Estadísticas
```bash
curl -X GET "http://localhost:8001/api/v1/documents/stats"
```

## 🔧 Configuración Avanzada

### Tipos de Análisis Disponibles

- `ocr`: Reconocimiento óptico de caracteres
- `classification`: Clasificación de documentos
- `entity_extraction`: Extracción de entidades
- `sentiment_analysis`: Análisis de sentimientos
- `topic_modeling`: Modelado de temas
- `summarization`: Resumen automático
- `keyword_extraction`: Extracción de palabras clave
- `semantic_search`: Búsqueda semántica
- `plagiarism_detection`: Detección de plagio
- `content_analysis`: Análisis de contenido

### Formatos Soportados

- **PDF**: Documentos PDF
- **DOCX/DOC**: Documentos de Microsoft Word
- **TXT**: Archivos de texto plano
- **RTF**: Rich Text Format
- **ODT**: OpenDocument Text
- **PPTX**: Presentaciones de PowerPoint
- **XLSX**: Hojas de cálculo de Excel
- **CSV**: Archivos CSV
- **Imágenes**: PNG, JPG, JPEG, TIFF, BMP (con OCR)

### Idiomas OCR Soportados

- Inglés (en)
- Español (es)
- Francés (fr)
- Alemán (de)
- Italiano (it)
- Portugués (pt)

## 🐳 Docker

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  document-processor:
    build: .
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/documents
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./uploads:/app/uploads
      - ./processed:/app/processed

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=documents
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## 🧪 Testing

```bash
# Ejecutar tests
pytest

# Tests con cobertura
pytest --cov=.

# Tests específicos
pytest tests/test_document_processor.py
```

## 📊 Monitoreo

### Métricas Prometheus
- Endpoint: http://localhost:9091/metrics
- Métricas disponibles:
  - `documents_processed_total`
  - `document_processing_duration_seconds`
  - `active_processing_jobs`
  - `error_rate`

### Logs Estructurados
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "logger": "document_processor",
  "message": "Document processed successfully",
  "document_id": "uuid",
  "processing_time": 2.5,
  "analysis_types": ["ocr", "classification"]
}
```

## 🔒 Seguridad

### Configuración de Seguridad
- Validación de tipos de archivo
- Límites de tamaño de archivo
- Rate limiting
- CORS configurable
- Autenticación JWT (opcional)

### Variables de Entorno Sensibles
```bash
SECRET_KEY=your-secret-key-change-in-production
SENTRY_DSN=your-sentry-dsn
DATABASE_URL=your-database-url
REDIS_URL=your-redis-url
```

## 🚀 Despliegue en Producción

### Requisitos del Sistema
- **CPU**: 4+ cores recomendados
- **RAM**: 8GB+ recomendados
- **Almacenamiento**: SSD recomendado
- **GPU**: Opcional, para aceleración de modelos

### Configuración de Producción
```bash
# Variables de entorno para producción
ENVIRONMENT=production
DEBUG=false
WORKERS=4
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    client_max_body_size 100M;
}
```

## 🤝 Contribución

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Soporte

- **Documentación**: [Wiki del proyecto]
- **Issues**: [GitHub Issues]
- **Discusiones**: [GitHub Discussions]
- **Email**: support@example.com

## 🎯 Roadmap

### Próximas Características
- [ ] Integración con servicios en la nube (AWS, GCP, Azure)
- [ ] Procesamiento de audio y video
- [ ] Análisis de documentos en tiempo real
- [ ] Interfaz web para usuarios
- [ ] API GraphQL
- [ ] Soporte para más idiomas
- [ ] Modelos personalizados
- [ ] Análisis de documentos históricos
- [ ] Integración con sistemas de gestión documental
- [ ] Análisis de tendencias temporales

---

**Desarrollado con ❤️ usando FastAPI, Transformers y tecnologías de vanguardia en IA/ML.**
# Advanced Content Redundancy Detector - Sistema AI/ML Avanzado

Sistema funcional, escalable y optimizado para detectar redundancia en contenido de texto con capacidades avanzadas de IA/ML, siguiendo las mejores prácticas de Python y FastAPI.

## 🚀 Características Principales

### 📊 Análisis de Contenido Básico
- **Análisis de redundancia**: Detecta repetición de palabras y frases
- **Comparación de similitud**: Compara dos textos y calcula su similitud
- **Evaluación de calidad**: Evalúa la legibilidad y calidad del contenido

### 🤖 Análisis AI/ML Avanzado
- **Análisis de sentimiento**: Detecta emociones y polaridad en el texto
- **Detección de idioma**: Identifica automáticamente el idioma del contenido
- **Extracción de temas**: Descubre temas principales usando LDA
- **Similitud semántica**: Compara textos usando embeddings avanzados
- **Detección de plagio**: Identifica contenido potencialmente plagiado
- **Extracción de entidades**: Encuentra nombres, lugares, organizaciones
- **Resumen automático**: Genera resúmenes usando modelos BART
- **Análisis de legibilidad**: Evaluación avanzada de complejidad del texto
- **Análisis integral**: Combina todas las características en una sola llamada
- **Procesamiento por lotes**: Analiza múltiples textos eficientemente

### ⚡ Performance y Escalabilidad
- **Sistema de caché**: Caché en memoria con TTL para optimizar respuestas
- **Rate limiting**: Control de velocidad por IP y endpoint
- **Métricas avanzadas**: Monitoreo en tiempo real de performance
- **Operaciones asíncronas**: Optimizado para alta concurrencia

### 🛡️ Seguridad y Robustez
- **Manejo de errores**: Sistema robusto con guard clauses
- **Validación avanzada**: Validación de entrada con Pydantic
- **Headers de seguridad**: Protección automática contra ataques
- **Logging estructurado**: Monitoreo y debugging avanzado

### 🔧 Arquitectura
- **Programación funcional**: Código limpio y mantenible
- **Patrón RORO**: Receive an Object, Return an Object
- **Middleware optimizado**: Stack de middleware especializado
- **Lifespan management**: Gestión del ciclo de vida de la aplicación

## 📁 Estructura del Proyecto

```
content_redundancy_detector/
├── app.py              # Aplicación principal con lifespan context manager
├── config.py           # Configuración centralizada
├── types.py            # Modelos Pydantic y tipos (RORO pattern)
├── utils.py            # Funciones puras utilitarias
├── services.py         # Servicios funcionales con caché
├── middleware.py       # Middleware optimizado (logging, rate limiting, security)
├── routers.py          # Handlers de rutas funcionales
├── cache.py            # Sistema de caché en memoria
├── metrics.py          # Sistema de métricas y monitoreo
├── rate_limiter.py     # Sistema de rate limiting
├── tests_functional.py # Tests funcionales
├── requirements.txt    # Dependencias mínimas
├── env.example         # Variables de entorno
└── README.md           # Documentación completa
```

## 🛠️ Instalación

1. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

2. **Configurar variables de entorno (opcional):**
```bash
cp env.example .env
# Editar .env según sea necesario
```

3. **Ejecutar la aplicación:**
```bash
python app.py
```

La aplicación estará disponible en `http://localhost:8000`

## 📖 Uso

### Análisis de contenido básico
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"content": "Este es un texto de ejemplo para analizar"}'
```

### Comparación de similitud
```bash
curl -X POST "http://localhost:8000/similarity" \
     -H "Content-Type: application/json" \
     -d '{"text1": "Texto uno", "text2": "Texto dos", "threshold": 0.8}'
```

### Evaluación de calidad
```bash
curl -X POST "http://localhost:8000/quality" \
     -H "Content-Type: application/json" \
     -d '{"content": "Texto para evaluar su calidad"}'
```

### Análisis de sentimiento
```bash
curl -X POST "http://localhost:8000/ai/sentiment" \
     -H "Content-Type: application/json" \
     -d '{"content": "Me encanta este producto, es fantástico!"}'
```

### Detección de idioma
```bash
curl -X POST "http://localhost:8000/ai/language" \
     -H "Content-Type: application/json" \
     -d '{"content": "This is an English text for language detection"}'
```

### Extracción de temas
```bash
curl -X POST "http://localhost:8000/ai/topics" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Texto sobre tecnología", "Texto sobre deportes", "Texto sobre cocina"], "num_topics": 3}'
```

### Similitud semántica
```bash
curl -X POST "http://localhost:8000/ai/semantic-similarity" \
     -H "Content-Type: application/json" \
     -d '{"text1": "El gato está durmiendo", "text2": "El felino descansa", "threshold": 0.8}'
```

### Detección de plagio
```bash
curl -X POST "http://localhost:8000/ai/plagiarism" \
     -H "Content-Type: application/json" \
     -d '{"content": "Texto a verificar", "reference_texts": ["Texto de referencia 1", "Texto de referencia 2"], "threshold": 0.8}'
```

### Extracción de entidades
```bash
curl -X POST "http://localhost:8000/ai/entities" \
     -H "Content-Type: application/json" \
     -d '{"content": "Juan Pérez trabaja en Google en California"}'
```

### Resumen automático
```bash
curl -X POST "http://localhost:8000/ai/summary" \
     -H "Content-Type: application/json" \
     -d '{"content": "Texto largo para resumir...", "max_length": 150}'
```

### Análisis integral
```bash
curl -X POST "http://localhost:8000/ai/comprehensive" \
     -H "Content-Type: application/json" \
     -d '{"content": "Texto para análisis completo con todas las características"}'
```

### Procesamiento por lotes
```bash
curl -X POST "http://localhost:8000/ai/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Texto 1", "Texto 2", "Texto 3"]}'
```

## 🔗 Endpoints

### Endpoints Básicos
- `GET /` - Información básica del sistema
- `GET /health` - Health check del sistema
- `POST /analyze` - Analizar contenido para redundancia
- `POST /similarity` - Comparar similitud entre textos
- `POST /quality` - Evaluar calidad del contenido
- `GET /stats` - Estadísticas del sistema

### Endpoints AI/ML Avanzados
- `POST /ai/sentiment` - Análisis de sentimiento
- `POST /ai/language` - Detección de idioma
- `POST /ai/topics` - Extracción de temas
- `POST /ai/semantic-similarity` - Similitud semántica
- `POST /ai/plagiarism` - Detección de plagio
- `POST /ai/entities` - Extracción de entidades
- `POST /ai/summary` - Resumen automático
- `POST /ai/readability` - Análisis de legibilidad avanzado
- `POST /ai/comprehensive` - Análisis integral
- `POST /ai/batch` - Procesamiento por lotes

## 📚 Documentación

La documentación interactiva está disponible en:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🧪 Testing

Ejecutar los tests:
```bash
python -m pytest tests.py -v
```

## ⚙️ Configuración

El sistema se puede configurar mediante variables de entorno:

- `APP_NAME`: Nombre de la aplicación
- `APP_VERSION`: Versión de la aplicación
- `DEBUG`: Modo debug (true/false)
- `HOST`: Host del servidor
- `PORT`: Puerto del servidor
- `LOG_LEVEL`: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
- `MAX_CONTENT_LENGTH`: Longitud máxima del contenido
- `MIN_CONTENT_LENGTH`: Longitud mínima del contenido

## 🏗️ Arquitectura

### Separación de Responsabilidades

- **`app.py`**: Aplicación FastAPI y endpoints
- **`config.py`**: Configuración centralizada
- **`models.py`**: Modelos de datos y validación
- **`services.py`**: Lógica de negocio y análisis
- **`utils.py`**: Utilidades y helpers
- **`tests.py`**: Tests del sistema

### Servicios

- **`ContentAnalyzer`**: Análisis de contenido y redundancia
- **`SimilarityDetector`**: Detección de similitud entre textos
- **`QualityAssessor`**: Evaluación de calidad y legibilidad

## 🔧 Tecnologías

### Framework y Core
- **FastAPI**: Framework web moderno y rápido
- **Pydantic**: Validación de datos y configuración
- **Uvicorn**: Servidor ASGI de alto rendimiento
- **Python 3.8+**: Lenguaje de programación

### AI/ML Libraries
- **Transformers**: Modelos de lenguaje pre-entrenados
- **Sentence-Transformers**: Embeddings semánticos
- **spaCy**: Procesamiento de lenguaje natural
- **NLTK**: Herramientas de NLP
- **scikit-learn**: Algoritmos de machine learning
- **Gensim**: Modelado de temas

### Base de Datos y Caché
- **Redis**: Sistema de caché en memoria
- **SQLAlchemy**: ORM para bases de datos
- **AsyncPG**: Driver asíncrono para PostgreSQL

### Monitoreo y Logging
- **Prometheus**: Métricas y monitoreo
- **Structlog**: Logging estructurado
- **Sentry**: Monitoreo de errores

### Testing
- **Pytest**: Framework de testing
- **pytest-asyncio**: Testing asíncrono
- **pytest-cov**: Cobertura de código

## 📈 Mejoras Implementadas

### Mejoras Básicas
- ✅ **Separación de responsabilidades**: Código organizado en módulos
- ✅ **Manejo de errores**: Sistema robusto de manejo de excepciones
- ✅ **Logging**: Sistema de logging estructurado
- ✅ **Configuración**: Sistema de configuración flexible
- ✅ **Validación**: Validación robusta de datos de entrada
- ✅ **Tests**: Suite de tests básicos
- ✅ **Documentación**: Documentación clara y completa
- ✅ **CORS**: Soporte para CORS
- ✅ **Performance**: Optimizaciones de rendimiento

### Mejoras AI/ML Avanzadas
- ✅ **Análisis de sentimiento**: Detección de emociones con modelos transformer
- ✅ **Detección de idioma**: Identificación automática de idiomas
- ✅ **Extracción de temas**: Modelado de temas con LDA
- ✅ **Similitud semántica**: Comparación usando embeddings avanzados
- ✅ **Detección de plagio**: Identificación de contenido duplicado
- ✅ **Extracción de entidades**: Reconocimiento de entidades nombradas
- ✅ **Resumen automático**: Generación de resúmenes con BART
- ✅ **Análisis de legibilidad**: Evaluación avanzada de complejidad
- ✅ **Procesamiento asíncrono**: Operaciones paralelas para mejor rendimiento
- ✅ **Procesamiento por lotes**: Análisis eficiente de múltiples textos
- ✅ **Caché inteligente**: Optimización de respuestas con caché
- ✅ **Configuración avanzada**: Feature flags y configuración granular
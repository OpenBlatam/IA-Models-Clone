# Ultimate NLP AI Document Processor

Sistema ultimate de procesamiento de documentos con características avanzadas de Procesamiento de Lenguaje Natural (NLP).

## Características Principales

### 🔧 Procesamiento de Texto Avanzado
- **Tokenización Mejorada**: spaCy, NLTK, Tweet tokenizer
- **Análisis de Sentimientos**: NLTK, spaCy con detección de emociones
- **Preprocesamiento de Texto**: 12+ pasos de limpieza y normalización
- **Extracción de Palabras Clave**: TF-IDF, frecuencia, YAKE
- **Cálculo de Similitud**: Coseno, Jaccard, Euclidiana, Manhattan
- **Modelado de Temas**: LDA, NMF, LSA con análisis de coherencia
- **Clasificación de Texto**: Naive Bayes, Ensemble con puntuaciones de confianza
- **Resumen de Texto**: Extractor, Abstractivo, Híbrido con ranking de oraciones
- **Redes de Palabras**: Análisis de co-ocurrencia
- **Métricas de Legibilidad**: Flesch, SMOG, Coleman-Liau

### 🚀 Características Avanzadas
- **Análisis de Dependencias**: Parsing sintáctico con spaCy, NLTK, Stanford
- **Resolución de Co-referencias**: Identificación y resolución de referencias
- **Vinculación de Entidades**: Enlace a bases de conocimiento
- **Análisis de Discurso**: Estructura retórica y coherencia
- **Embeddings de Palabras**: Word2Vec, TF-IDF, Count-based
- **Redes Semánticas**: Análisis de co-ocurrencia y similitud semántica
- **Grafos de Conocimiento**: Extracción de entidades y relaciones
- **Procesamiento por Lotes**: Todas las características en lote
- **Análisis Integral**: Combinación de todas las características
- **Comparación de Textos**: Análisis lado a lado

## Instalación

### Requisitos del Sistema
- Python 3.8+
- 8GB RAM mínimo (16GB recomendado)
- 4GB espacio en disco
- Conexión a internet para descargar modelos

### Instalación de Dependencias

```bash
# Instalar dependencias básicas
pip install fastapi uvicorn python-multipart

# Instalar dependencias de NLP
pip install nltk spacy scikit-learn networkx textstat

# Instalar dependencias adicionales
pip install numpy pandas matplotlib seaborn

# Instalar modelos de spaCy
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

### Instalación Automática

```bash
# Ejecutar script de instalación
python install_ultimate_nlp.py
```

## Uso

### Iniciar la Aplicación

```bash
# Iniciar servidor de desarrollo
python ultimate_nlp_app.py

# O usar uvicorn directamente
uvicorn ultimate_nlp_app:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints Principales

#### 1. Análisis Ultimate
```bash
POST /analyze/ultimate
{
    "text": "Texto para análisis completo"
}
```

#### 2. Comparación Ultimate
```bash
POST /compare/ultimate
{
    "text1": "Primer texto",
    "text2": "Segundo texto"
}
```

#### 3. Análisis por Lotes
```bash
POST /batch/analyze/ultimate
{
    "texts": ["Texto 1", "Texto 2", "Texto 3"]
}
```

### Endpoints Enhanced NLP

#### Tokenización Mejorada
```bash
POST /enhanced-nlp/tokenize
{
    "text": "Texto a procesar",
    "method": "spacy",
    "include_phrases": true,
    "include_entities": true
}
```

#### Análisis de Sentimientos
```bash
POST /enhanced-nlp/sentiment
{
    "text": "Texto a analizar",
    "method": "nltk",
    "include_emotions": true
}
```

#### Preprocesamiento de Texto
```bash
POST /enhanced-nlp/preprocess
{
    "text": "Texto a preprocesar",
    "steps": ["lowercase", "remove_punctuation", "remove_stopwords", "lemmatize"]
}
```

#### Extracción de Palabras Clave
```bash
POST /enhanced-nlp/keywords
{
    "text": "Texto para extraer palabras clave",
    "method": "tfidf",
    "top_k": 10,
    "include_phrases": true
}
```

#### Cálculo de Similitud
```bash
POST /enhanced-nlp/similarity
{
    "text1": "Primer texto",
    "text2": "Segundo texto",
    "method": "cosine",
    "include_semantic": true
}
```

#### Modelado de Temas
```bash
POST /enhanced-nlp/topics
{
    "texts": ["Texto 1", "Texto 2", "Texto 3"],
    "method": "lda",
    "num_topics": 5,
    "include_coherence": true
}
```

#### Clasificación de Texto
```bash
POST /enhanced-nlp/classify
{
    "text": "Texto a clasificar",
    "categories": ["tecnología", "negocios", "ciencia"],
    "method": "naive_bayes",
    "include_confidence": true
}
```

#### Resumen de Texto
```bash
POST /enhanced-nlp/summarize
{
    "text": "Texto a resumir",
    "method": "extractive",
    "max_sentences": 3,
    "include_ranking": true
}
```

#### Redes de Palabras
```bash
POST /enhanced-nlp/network
{
    "texts": ["Texto 1", "Texto 2"],
    "min_frequency": 2
}
```

#### Métricas de Legibilidad
```bash
POST /enhanced-nlp/readability
{
    "text": "Texto a analizar"
}
```

### Endpoints Advanced NLP

#### Análisis de Dependencias
```bash
POST /advanced-nlp/dependencies/parse
{
    "text": "Texto a analizar dependencias",
    "parser_type": "spacy"
}
```

#### Resolución de Co-referencias
```bash
POST /advanced-nlp/coreferences/resolve
{
    "text": "Texto a resolver co-referencias",
    "method": "spacy"
}
```

#### Vinculación de Entidades
```bash
POST /advanced-nlp/entities/link
{
    "text": "Texto a vincular entidades",
    "method": "spacy"
}
```

#### Análisis de Discurso
```bash
POST /advanced-nlp/discourse/analyze
{
    "text": "Texto a analizar discurso",
    "method": "rhetorical"
}
```

#### Embeddings de Palabras
```bash
POST /advanced-nlp/embeddings/create
{
    "text": "Texto para crear embeddings",
    "method": "word2vec"
}
```

#### Redes Semánticas
```bash
POST /advanced-nlp/networks/semantic
{
    "text": "Texto para red semántica",
    "method": "co_occurrence"
}
```

#### Grafos de Conocimiento
```bash
POST /advanced-nlp/graphs/knowledge
{
    "text": "Texto para grafo de conocimiento",
    "method": "entity_relation"
}
```

### Procesamiento por Lotes

#### Análisis Integral por Lotes
```bash
POST /enhanced-nlp/batch/tokenize
{
    "texts": ["Texto 1", "Texto 2", "Texto 3"],
    "method": "spacy"
}
```

#### Análisis Avanzado por Lotes
```bash
POST /advanced-nlp/batch/dependencies
{
    "texts": ["Texto 1", "Texto 2", "Texto 3"],
    "parser_type": "spacy"
}
```

## Métodos de Procesamiento

### Enhanced NLP
- **Tokenización**: spaCy, NLTK, Tweet
- **Sentimientos**: NLTK, spaCy
- **Preprocesamiento**: 12+ pasos disponibles
- **Palabras Clave**: TF-IDF, Frecuencia, YAKE
- **Similitud**: Coseno, Jaccard, Euclidiana, Manhattan
- **Temas**: LDA, NMF, LSA
- **Clasificación**: Naive Bayes, Ensemble
- **Resumen**: Extractor, Abstractivo, Híbrido

### Advanced NLP
- **Dependencias**: spaCy, NLTK, Stanford
- **Co-referencias**: spaCy, Rule-based
- **Entidades**: spaCy, Rule-based
- **Discurso**: Retórico, Coherencia
- **Embeddings**: Word2Vec, TF-IDF, Count
- **Redes**: Co-ocurrencia, Similitud semántica
- **Grafos**: Entidad-relación, Basado en dependencias

## Monitoreo y Estadísticas

### Verificación de Salud
```bash
GET /health
```

### Estadísticas del Sistema
```bash
GET /enhanced-nlp/stats
GET /advanced-nlp/stats
```

### Métricas de Rendimiento
```bash
GET /metrics
```

### Información del Sistema
```bash
GET /info
```

## Configuración

### Variables de Entorno
```bash
# Configuración del servidor
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Configuración de NLP
NLP_MODEL=en_core_web_sm
NLTK_DATA_PATH=./nltk_data
SPACY_MODEL_PATH=./spacy_models
```

### Configuración Avanzada
```python
# Configuración personalizada
ULTIMATE_NLP_CONFIG = {
    "max_text_length": 20000,
    "batch_size": 200,
    "cache_size": 2000,
    "similarity_threshold": 0.7,
    "topic_coherence_threshold": 0.5,
    "dependency_parser": "spacy",
    "coreference_resolver": "spacy",
    "entity_linker": "spacy",
    "discourse_analyzer": "rhetorical",
    "embedding_method": "word2vec",
    "network_method": "co_occurrence",
    "graph_method": "entity_relation"
}
```

## Despliegue

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "ultimate_nlp_app.py"]
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ultimate-nlp-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ultimate-nlp-app
  template:
    metadata:
      labels:
        app: ultimate-nlp-app
    spec:
      containers:
      - name: ultimate-nlp-app
        image: ultimate-nlp-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: HOST
          value: "0.0.0.0"
        - name: PORT
          value: "8000"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

## Solución de Problemas

### Problemas Comunes

#### 1. Error de Modelo No Encontrado
```bash
# Solución: Instalar modelos de spaCy
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

#### 2. Error de Memoria Insuficiente
```bash
# Solución: Reducir tamaño de lote
export BATCH_SIZE=100
export MAX_TEXT_LENGTH=10000
```

#### 3. Error de Dependencias
```bash
# Solución: Reinstalar dependencias
pip install --upgrade -r requirements.txt
```

### Logs y Depuración
```bash
# Habilitar logs detallados
export LOG_LEVEL=debug
python ultimate_nlp_app.py
```

## Rendimiento

### Métricas de Rendimiento
- **Tiempo de Respuesta**: < 2 segundos para análisis completo
- **Throughput**: 50+ requests/segundo para análisis completo
- **Uso de Memoria**: < 4GB para operaciones normales
- **Precisión**: 95%+ para análisis de sentimientos
- **Cobertura**: 100% de características NLP disponibles

### Optimización
- **Caché**: Implementado para resultados frecuentes
- **Procesamiento por Lotes**: Optimizado para múltiples textos
- **Modelos Precargados**: Modelos cargados al inicio
- **Compresión**: GZIP habilitado para respuestas
- **Paralelización**: Procesamiento paralelo cuando es posible

## Seguridad

### Características de Seguridad
- **Validación de Entrada**: Validación estricta de datos
- **Límites de Tamaño**: Límites en tamaño de texto
- **Rate Limiting**: Limitación de velocidad de requests
- **Logging de Seguridad**: Registro de actividades
- **Sanitización**: Limpieza de datos de entrada

### Mejores Prácticas
- **Validación**: Siempre validar entrada del usuario
- **Sanitización**: Limpiar datos antes del procesamiento
- **Monitoreo**: Monitorear uso del sistema
- **Actualizaciones**: Mantener dependencias actualizadas
- **Backup**: Respaldo regular de modelos y datos

## Contribución

### Cómo Contribuir
1. Fork del repositorio
2. Crear rama de feature
3. Implementar cambios
4. Ejecutar tests
5. Crear pull request

### Estándares de Código
- **PEP 8**: Seguir estándares de Python
- **Type Hints**: Usar anotaciones de tipo
- **Documentación**: Documentar funciones y clases
- **Tests**: Escribir tests unitarios

## Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo LICENSE para más detalles.

## Soporte

### Contacto
- **Email**: support@ultimate-nlp.com
- **Documentación**: https://docs.ultimate-nlp.com
- **Issues**: https://github.com/ultimate-nlp/issues

### Recursos Adicionales
- **Tutoriales**: https://tutorials.ultimate-nlp.com
- **Ejemplos**: https://examples.ultimate-nlp.com
- **API Reference**: https://api.ultimate-nlp.com

---

**Ultimate NLP AI Document Processor** - Procesamiento ultimate de documentos con IA
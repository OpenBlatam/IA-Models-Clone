# Enhanced NLP AI Document Processor

Sistema avanzado de procesamiento de documentos con características mejoradas de Procesamiento de Lenguaje Natural (NLP).

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
- **Procesamiento por Lotes**: Todas las características en lote
- **Análisis Integral**: Combinación de todas las características
- **Comparación de Textos**: Análisis lado a lado
- **Métricas de Rendimiento**: Estadísticas detalladas del sistema
- **Monitoreo de Salud**: Verificación del estado del sistema

## Instalación

### Requisitos del Sistema
- Python 3.8+
- 4GB RAM mínimo (8GB recomendado)
- 2GB espacio en disco
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
python install_enhanced_nlp.py
```

## Uso

### Iniciar la Aplicación

```bash
# Iniciar servidor de desarrollo
python enhanced_nlp_app.py

# O usar uvicorn directamente
uvicorn enhanced_nlp_app:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints Principales

#### 1. Tokenización Mejorada
```bash
POST /enhanced-nlp/tokenize
{
    "text": "Texto a procesar",
    "method": "spacy",
    "include_phrases": true,
    "include_entities": true
}
```

#### 2. Análisis de Sentimientos
```bash
POST /enhanced-nlp/sentiment
{
    "text": "Texto a analizar",
    "method": "nltk",
    "include_emotions": true
}
```

#### 3. Preprocesamiento de Texto
```bash
POST /enhanced-nlp/preprocess
{
    "text": "Texto a preprocesar",
    "steps": ["lowercase", "remove_punctuation", "remove_stopwords", "lemmatize"]
}
```

#### 4. Extracción de Palabras Clave
```bash
POST /enhanced-nlp/keywords
{
    "text": "Texto para extraer palabras clave",
    "method": "tfidf",
    "top_k": 10,
    "include_phrases": true
}
```

#### 5. Cálculo de Similitud
```bash
POST /enhanced-nlp/similarity
{
    "text1": "Primer texto",
    "text2": "Segundo texto",
    "method": "cosine",
    "include_semantic": true
}
```

#### 6. Modelado de Temas
```bash
POST /enhanced-nlp/topics
{
    "texts": ["Texto 1", "Texto 2", "Texto 3"],
    "method": "lda",
    "num_topics": 5,
    "include_coherence": true
}
```

#### 7. Clasificación de Texto
```bash
POST /enhanced-nlp/classify
{
    "text": "Texto a clasificar",
    "categories": ["tecnología", "negocios", "ciencia"],
    "method": "naive_bayes",
    "include_confidence": true
}
```

#### 8. Resumen de Texto
```bash
POST /enhanced-nlp/summarize
{
    "text": "Texto a resumir",
    "method": "extractive",
    "max_sentences": 3,
    "include_ranking": true
}
```

#### 9. Redes de Palabras
```bash
POST /enhanced-nlp/network
{
    "texts": ["Texto 1", "Texto 2"],
    "min_frequency": 2
}
```

#### 10. Métricas de Legibilidad
```bash
POST /enhanced-nlp/readability
{
    "text": "Texto a analizar"
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

#### Análisis de Sentimientos por Lotes
```bash
POST /enhanced-nlp/batch/sentiment
{
    "texts": ["Texto 1", "Texto 2", "Texto 3"],
    "method": "nltk"
}
```

### Análisis Integral

#### Análisis Completo
```bash
POST /enhanced-nlp/analyze/comprehensive
{
    "text": "Texto para análisis completo"
}
```

#### Comparación de Textos
```bash
POST /enhanced-nlp/analyze/comparison
{
    "text1": "Primer texto",
    "text2": "Segundo texto"
}
```

## Métodos de Procesamiento

### Tokenización
- **spacy**: Procesamiento avanzado con spaCy
- **nltk**: Tokenización con NLTK
- **tweet**: Tokenización especializada para tweets

### Análisis de Sentimientos
- **nltk**: Análisis con NLTK VADER
- **spacy**: Análisis con spaCy

### Preprocesamiento
- **lowercase**: Convertir a minúsculas
- **remove_punctuation**: Eliminar puntuación
- **remove_numbers**: Eliminar números
- **remove_stopwords**: Eliminar palabras vacías
- **remove_stopwords_advanced**: Eliminación avanzada de palabras vacías
- **lemmatize**: Lematización
- **stem**: Stemming
- **lancaster_stem**: Stemming Lancaster
- **snowball_stem**: Stemming Snowball
- **remove_extra_whitespace**: Eliminar espacios extra
- **remove_urls**: Eliminar URLs
- **remove_emails**: Eliminar emails

### Extracción de Palabras Clave
- **tfidf**: TF-IDF
- **frequency**: Frecuencia de palabras
- **yake**: Algoritmo YAKE

### Cálculo de Similitud
- **cosine**: Similitud coseno
- **jaccard**: Similitud Jaccard
- **euclidean**: Distancia euclidiana
- **manhattan**: Distancia Manhattan

### Modelado de Temas
- **lda**: Latent Dirichlet Allocation
- **nmf**: Non-negative Matrix Factorization
- **lsa**: Latent Semantic Analysis

### Clasificación
- **naive_bayes**: Naive Bayes
- **ensemble**: Clasificación por conjunto

### Resumen
- **extractive**: Resumen extractivo
- **abstractive**: Resumen abstractivo
- **hybrid**: Resumen híbrido

## Monitoreo y Estadísticas

### Verificación de Salud
```bash
GET /health
```

### Estadísticas del Sistema
```bash
GET /enhanced-nlp/stats
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
ENHANCED_NLP_CONFIG = {
    "max_text_length": 10000,
    "batch_size": 100,
    "cache_size": 1000,
    "similarity_threshold": 0.7,
    "topic_coherence_threshold": 0.5
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

CMD ["python", "enhanced_nlp_app.py"]
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: enhanced-nlp-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: enhanced-nlp-app
  template:
    metadata:
      labels:
        app: enhanced-nlp-app
    spec:
      containers:
      - name: enhanced-nlp-app
        image: enhanced-nlp-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: HOST
          value: "0.0.0.0"
        - name: PORT
          value: "8000"
```

## Solución de Problemas

### Problemas Comunes

#### 1. Error de Modelo No Encontrado
```bash
# Solución: Instalar modelos de spaCy
python -m spacy download en_core_web_sm
```

#### 2. Error de Memoria Insuficiente
```bash
# Solución: Reducir tamaño de lote
export BATCH_SIZE=50
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
python enhanced_nlp_app.py
```

## Rendimiento

### Métricas de Rendimiento
- **Tiempo de Respuesta**: < 1 segundo para textos cortos
- **Throughput**: 100+ requests/segundo
- **Uso de Memoria**: < 2GB para operaciones normales
- **Precisión**: 95%+ para análisis de sentimientos

### Optimización
- **Caché**: Implementado para resultados frecuentes
- **Procesamiento por Lotes**: Optimizado para múltiples textos
- **Modelos Precargados**: Modelos cargados al inicio
- **Compresión**: GZIP habilitado para respuestas

## Seguridad

### Características de Seguridad
- **Validación de Entrada**: Validación estricta de datos
- **Límites de Tamaño**: Límites en tamaño de texto
- **Rate Limiting**: Limitación de velocidad de requests
- **Logging de Seguridad**: Registro de actividades

### Mejores Prácticas
- **Validación**: Siempre validar entrada del usuario
- **Sanitización**: Limpiar datos antes del procesamiento
- **Monitoreo**: Monitorear uso del sistema
- **Actualizaciones**: Mantener dependencias actualizadas

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
- **Email**: support@enhanced-nlp.com
- **Documentación**: https://docs.enhanced-nlp.com
- **Issues**: https://github.com/enhanced-nlp/issues

### Recursos Adicionales
- **Tutoriales**: https://tutorials.enhanced-nlp.com
- **Ejemplos**: https://examples.enhanced-nlp.com
- **API Reference**: https://api.enhanced-nlp.com

---

**Enhanced NLP AI Document Processor** - Procesamiento avanzado de documentos con IA













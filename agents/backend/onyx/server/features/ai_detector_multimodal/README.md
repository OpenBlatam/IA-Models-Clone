# AI Detector Multimodal

Detector de contenido generado por IA con análisis forense multimodal.

## Características

- ✅ Detección de texto generado por IA
- ✅ Detección de imágenes generadas por IA
- ✅ Detección de audio generado por IA
- ✅ Detección de video generado por IA
- ✅ Análisis forense de prompts
- ✅ Identificación de modelos de IA (GPT, Claude, Gemini, LLaMA, etc.)
- ✅ Porcentaje de contenido generado por IA
- ✅ Análisis de parámetros de generación

## Endpoints

### POST `/ai-detector/detect`
Detecta si un contenido fue generado por IA.

**Request:**
```json
{
  "content": "Texto a analizar...",
  "content_type": "text",
  "metadata": {}
}
```

**Response:**
```json
{
  "is_ai_generated": true,
  "ai_percentage": 85.5,
  "detected_models": [
    {
      "model_name": "gpt-4",
      "confidence": 0.85,
      "provider": "OpenAI"
    }
  ],
  "primary_model": {
    "model_name": "gpt-4",
    "confidence": 0.85,
    "provider": "OpenAI"
  },
  "forensic_analysis": {
    "estimated_prompt": "Explain or describe: ...",
    "prompt_confidence": 0.6,
    "prompt_patterns": ["explanatory"],
    "generation_parameters": {
      "estimated_max_tokens": 500,
      "temperature": 0.7
    }
  },
  "confidence_score": 0.85,
  "detection_methods": ["pattern_matching", "statistical_analysis"],
  "processing_time": 0.15,
  "timestamp": 1234567890.123,
  "quality_info": {
    "writing_quality": 0.85,
    "paraphrase_likelihood": 0.15,
    "risk_score": 0.82,
    "reliability": "high"
  },
  "from_cache": false
}
```

### POST `/ai-detector/batch`
Detección en batch de múltiples contenidos.

### GET `/ai-detector/health`
Health check del servicio.

### GET `/ai-detector/models`
Lista los modelos de IA que el detector puede identificar.

### POST `/ai-detector/cache/clear`
Limpia el cache de detecciones.

### GET `/ai-detector/cache/stats`
Obtiene estadísticas del cache.

### GET `/ai-detector/stats`
Obtiene estadísticas generales del detector.

### GET `/ai-detector/history?limit=50`
Obtiene el historial de detecciones recientes.

### GET `/ai-detector/statistics`
Obtiene estadísticas detalladas de las detecciones (tasa de detección, modelo más común, etc.).

## Uso

```python
from ai_detector_multimodal.api.router import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
```

## Instalación

```bash
pip install -r requirements.txt
```

## Métodos de Detección (MEJORADOS - V2)

1. **Pattern Matching** (Peso: 35%): Detecta patrones específicos de modelos de IA
   - Soporta: GPT-3.5, GPT-4, Claude, Gemini, LLaMA, Mistral, Cohere, PaLM, Jurassic, Groq, OpenRouter
   - Detecta versiones específicas de modelos
   - Análisis de múltiples ocurrencias de patrones

2. **Statistical Analysis** (Peso: 25%): Análisis estadístico avanzado
   - Burstiness (variación en longitud de oraciones)
   - Diversidad de vocabulario (type-token ratio)
   - Ratio de palabras funcionales vs contenido
   - Consistencia en puntuación
   - Análisis de repetición mejorado

3. **Structure Analysis** (Peso: 15%): Análisis de estructura y organización
   - Detección de estructuras organizadas
   - Análisis de párrafos bien formados
   - Coherencia temática

4. **Style Analysis** (Peso: 15%): Análisis de estilo y formalidad
   - Detección de formalidad
   - Uso de vocabulario sofisticado
   - Análisis de errores comunes

5. **Entropy Analysis** (Peso: 10%): Análisis de entropía y n-gramas
   - Entropía de caracteres
   - Análisis de bigramas
   - Detección de patrones predecibles

6. **Semantic Coherence** (Peso: 8%): Análisis de coherencia semántica (NUEVO)
   - Palabras de transición
   - Referencias pronominales
   - Repetición de conceptos clave
   - Estructura lógica

7. **Syntactic Complexity** (Peso: 7%): Análisis de complejidad sintáctica (NUEVO)
   - Oraciones compuestas y complejas
   - Uso de gerundios y participios
   - Preposiciones complejas
   - Voz pasiva
   - Longitud de oraciones

8. **Citation Analysis** (Peso: 5%): Análisis de citas y referencias
   - Citas directas
   - Referencias académicas
   - Números de página
   - URLs y enlaces
   - Notas al pie

9. **Temporal Analysis** (Peso: 4%): Análisis de consistencia temporal (NUEVO)
   - Cambios de estilo a lo largo del texto
   - Consistencia de vocabulario
   - Cambios abruptos de tono
   - Variación en longitud de oraciones

10. **Watermark Detection** (Peso: 3%): Detección de watermarks (NUEVO)
    - Patrones de watermark conocidos
    - Caracteres Unicode especiales
    - Espaciado inusual
    - Hashes o identificadores ocultos

11. **Edit Detection** (Peso: 2%): Detección de ediciones/parches (NUEVO)
    - Marcadores de edición explícitos
    - Cambios de formato abruptos
    - Paréntesis o corchetes de aclaración
    - Cambios de estilo dentro del texto

12. **Sentiment Analysis** (Peso: 2%): Análisis de patrones de sentimientos
    - Detección de emociones
    - Uso de emojis
    - Análisis de polaridad
    - Expresiones de incertidumbre

13. **Contextual Analysis** (Peso: 3%): Análisis de coherencia contextual (NUEVO)
    - Coherencia temática avanzada
    - Progresión lógica
    - Referencias cruzadas
    - Coherencia entre párrafos

14. **Translation Detection** (Peso: 2%): Detección de traducción automática (NUEVO)
    - Patrones de traducción automática
    - Orden de palabras inusual
    - Uso excesivo de palabras literales
    - Falta de modismos naturales

15. **Generation Patterns** (Peso: 2%): Análisis de patrones de generación
    - Patrones de inicio típicos
    - Estructura repetitiva
    - Frases de transición excesivas
    - Patrones de cierre típicos

16. **Writing Quality** (Peso: 2%): Análisis de calidad de escritura (NUEVO)
    - Detección de errores gramaticales
    - Análisis de puntuación
    - Análisis de vocabulario
    - Estructura de párrafos

17. **Paraphrase Detection** (Peso: 2%): Detección de parafraseo (NUEVO)
    - Sinónimos excesivos
    - Cambios de estructura
    - Vocabulario formal excesivo
    - Cambios de voz

18. **Risk Analysis** (Peso: 1%): Análisis de riesgo y confiabilidad (NUEVO)
    - Confianza basada en métodos
    - Consistencia entre métodos
    - Análisis de modelos detectados
    - Evaluación de confiabilidad

19. **Metadata Analysis** (Peso: 1%): Análisis de metadatos y contexto (NUEVO)
    - Análisis de fuente
    - Análisis de timestamp
    - Detección de user agent o aplicación
    - Análisis de idioma en metadatos vs texto
    - Análisis de referrer u origen

20. **Language Pattern Analysis** (Peso: 1%): Análisis de patrones de idioma (NUEVO)
    - Detección de mezcla de idiomas
    - Análisis de caracteres especiales por idioma
    - Análisis de orden de palabras (SVO vs otros)
    - Análisis de expresiones idiomáticas

21. **Semantic Similarity Analysis** (Peso: 1%): Análisis de similitud semántica (NUEVO)
    - Similitud entre oraciones (Jaccard)
    - Repetición de conceptos clave
    - Coherencia temática
    - Análisis de palabras relacionadas

22. **Keyword Frequency Analysis** (Peso: 1%): Análisis de frecuencia de palabras clave (NUEVO)
    - Detección de palabras clave típicas de IA
    - Análisis de frases características
    - Ratio de conectores lógicos
    - Patrones de lenguaje formal

23. **Response Pattern Detection** (Peso: 1%): Detección de patrones de respuesta (NUEVO)
    - Patrones de inicio típicos
    - Estructura organizada (intro-desarrollo-conclusión)
    - Uso excesivo de conectores lógicos
    - Formato de listas numeradas

24. **Narrative Coherence Analysis** (Peso: 1%): Análisis de coherencia narrativa (NUEVO)
    - Referencias pronominales
    - Progresión temática
    - Variación en longitud de oraciones
    - Transiciones entre párrafos

25. **Adaptive Weighting System** (Dinámico): Sistema de pesos adaptativos (NUEVO)
    - Ajusta pesos según longitud del texto
    - Optimiza pesos basado en modelos detectados
    - Aprende del historial de detecciones
    - Mejora precisión según contexto

26. **Historical Context Analysis** (Peso: 1%): Análisis de contexto histórico (NUEVO)
    - Compara con detecciones anteriores
    - Detecta patrones y tendencias
    - Análisis de consistencia de modelos
    - Evaluación de confianza histórica

27. **Advanced N-gram Analysis** (Peso: 1%): Análisis avanzado de n-gramas (NUEVO)
    - Análisis de trigramas
    - Análisis de 4-gramas (frases comunes de IA)
    - Detección de secuencias repetitivas
    - Análisis de distribución de n-gramas

28. **Comparative Similarity Analysis** (Peso: 1%): Análisis comparativo (NUEVO)
    - Compara con textos conocidos de IA
    - Compara con textos conocidos humanos
    - Análisis de similitud de vocabulario
    - Detección de patrones comunes

29. **Machine Learning Pattern Analysis** (Peso: 1%): Análisis con patrones ML (NUEVO)
    - Análisis de combinación de características
    - Análisis de consistencia entre métodos
    - Evaluación de modelos detectados
    - Análisis de características combinadas

30. **Model Signature Analysis** (Peso: 1%): Análisis de firmas de modelos (NUEVO)
    - Firmas específicas por modelo (GPT-4, Claude, Gemini, etc.)
    - Frases características de cada modelo
    - Análisis de estructura y formalidad
    - Verificación de combinación de características

31. **Semantic Embedding Analysis** (Peso: 1%): Análisis de embeddings semánticos (NUEVO)
    - Análisis de clusters semánticos básico
    - Análisis de co-ocurrencia de palabras
    - Análisis de densidad semántica
    - Análisis de distribución de palabras
    - Análisis de contexto semántico

32. **Temporal Pattern Analysis** (Peso: 1%): Análisis de patrones temporales (NUEVO)
    - Análisis de timestamp en metadatos
    - Detección de referencias temporales en el texto
    - Análisis de referencias a eventos recientes
    - Identificación de contenido generado recientemente

33. **Hybrid Model Detection** (Peso: 1%): Detección de modelos híbridos (NUEVO)
    - Detección de múltiples modelos con confianza similar
    - Identificación de modelos de diferentes proveedores
    - Análisis de patrones en diferentes partes del texto
    - Detección de estilo mixto

34. **Advanced Frequency Analysis** (Peso: 1%): Análisis avanzado de frecuencia (NUEVO)
    - Análisis de distribución de Zipf
    - Análisis de palabras raras vs comunes
    - Análisis de frecuencia de palabras funcionales específicas
    - Análisis de frecuencia de palabras de contenido
    - Detección de palabras de alta frecuencia inusual

## Sistema de Alertas

El detector ahora incluye un sistema de alertas automático que notifica sobre:

- **Alta confianza de IA**: Cuando el porcentaje supera el umbral (80% por defecto)
- **Modelo específico detectado**: Cuando se detecta un modelo con alta confianza (>80%)
- **Múltiples modelos detectados**: Indica posible contenido parafraseado
- **Confianza muy alta**: Cuando la confianza supera el 90%
- **Detección inconsistente**: Cuando hay discrepancias entre porcentaje y confianza

## Optimizaciones

- ✅ **Sistema de Cache**: Cache inteligente para resultados repetidos
- ✅ **Procesamiento optimizado**: Análisis eficiente y rápido
- ✅ **Gestión de memoria**: Cache con límite automático (FIFO)
- ✅ **Información de calidad**: Análisis de calidad y confiabilidad en resultados
- ✅ **Historial de detecciones**: Sistema de historial para análisis de tendencias
- ✅ **Estadísticas avanzadas**: Métricas detalladas de uso y precisión

## Análisis Forense (MEJORADO)

El detector intenta inferir con mayor precisión:

### Detección de Prompts
- **Tipos de prompts detectados**:
  - Explicatorio/Descriptivo
  - Generación de listas
  - Análisis/Evaluación
  - Resumen
  - Comparación
  - Escritura creativa
  - Pregunta-Respuesta

- **Estilos de prompt**:
  - Solicitud educada
  - Comando directo
  - Solicitud informacional
  - General

### Parámetros de Generación Estimados
- `max_tokens`: Basado en longitud del texto
- `temperature`: Estimado según variabilidad (burstiness)
- `top_p`: Estimación común
- `presence_penalty` y `frequency_penalty`: Detectados

### Evidencia Forense
- Longitud del texto (corto/medio/largo)
- Tipo de estructura
- Estructura de oraciones (uniforme/variada)
- Confianza basada en modelos detectados
- Instrucciones estimadas del prompt


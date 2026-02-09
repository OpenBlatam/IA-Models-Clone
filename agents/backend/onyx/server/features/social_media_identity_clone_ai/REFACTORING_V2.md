# Refactoring v2 - Mejoras con Nuevas Librerías

## Resumen

Refactorización del proyecto para usar las mejores librerías disponibles y mejorar la calidad del código.

## Mejoras Implementadas

### 1. TextProcessor Refactorizado ✅

**Archivo:** `utils/text_processor.py`

**Mejoras:**
- ✅ Integración con VaderSentiment para análisis de sentimiento optimizado para redes sociales
- ✅ Soporte para Spacy para NLP avanzado (tokenización, NER, dependency parsing)
- ✅ Análisis de emojis con librería `emoji`
- ✅ Estadísticas de texto con `textstat` (legibilidad, sílabas, etc.)
- ✅ Fallback graceful si librerías no están disponibles
- ✅ Mejor detección de tono (formal, casual, humorístico)
- ✅ Análisis de entidades nombradas (NER) con Spacy
- ✅ Métodos adicionales: `get_readability_score()`, `get_text_statistics()`

**Antes:**
```python
# Análisis básico con palabras hardcodeadas
positive_words = ['bueno', 'genial', ...]
```

**Después:**
```python
# Análisis avanzado con VaderSentiment
scores = self._vader_analyzer.polarity_scores(text)
# Soporta emojis, slang, y contexto de redes sociales
```

**Uso:**
```python
processor = TextProcessor(spacy_model='es_core_news_sm')
analysis = processor.analyze_basic(text)
stats = processor.get_text_statistics(text)
readability = processor.get_readability_score(text)
```

### 2. TikTokConnector Refactorizado ✅

**Archivo:** `connectors/tiktok_connector.py`

**Mejoras:**
- ✅ Soporte para yt-dlp para extracción de transcripciones
- ✅ Mejor manejo de errores
- ✅ Type hints completos
- ✅ Logging estructurado

**Nuevo:**
```python
# Soporte para yt-dlp
async def get_video_transcript(self, video_id: str) -> Optional[str]:
    # Usa yt-dlp para extraer subtítulos automáticos
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'es'],
    }
```

### 3. InstagramConnector Refactorizado ✅

**Archivo:** `connectors/instagram_connector.py`

**Mejoras:**
- ✅ Integración con Instaloader para scraping de Instagram
- ✅ Soporte dual: API oficial (Graph API) y scraping (Instaloader)
- ✅ Mejor manejo de perfiles privados y errores
- ✅ Extracción completa de posts y comentarios
- ✅ Type hints completos

**Nuevo:**
```python
# Inicialización con scraping opcional
connector = InstagramConnector(
    api_key=api_key,
    use_scraping=True  # Usa Instaloader
)

# Obtiene perfil con instaloader
profile = await connector.get_profile(username)
posts = await connector.get_posts(username, limit=100)
comments = await connector.get_post_comments(post_id, limit=50)
```

**Características:**
- Extrae información completa: followers, following, posts, bio
- Maneja perfiles privados con errores apropiados
- Extrae posts con metadata completa
- Extrae comentarios con autor y likes

### 4. YouTubeConnector Refactorizado ✅

**Archivo:** `connectors/youtube_connector.py`

**Mejoras:**
- ✅ Integración completa con YouTube Data API v3
- ✅ Soporte para yt-dlp para transcripciones
- ✅ Paginación automática para obtener múltiples videos
- ✅ Extracción completa de metadata del canal
- ✅ Mejor manejo de errores HTTP

**Nuevo:**
```python
# Inicialización con API key
connector = YouTubeConnector(api_key=api_key)

# Obtiene canal completo
channel = await connector.get_channel(channel_id)
# Retorna: subscribers, videos_count, views_count, etc.

# Obtiene videos con paginación
videos = await connector.get_videos(channel_id, limit=100)

# Obtiene transcripción con yt-dlp
transcript = await connector.get_video_transcript(video_id)
```

**Características:**
- Extrae estadísticas completas del canal
- Paginación automática para listas grandes
- Soporte para múltiples idiomas en transcripciones
- Fallback graceful si API no está disponible

## Dependencias Nuevas Utilizadas

### TextProcessor
- `vaderSentiment>=3.1.6` - Análisis de sentimiento
- `spacy>=3.7.0` - NLP avanzado
- `emoji>=2.10.0` - Manejo de emojis
- `textstat>=0.7.3` - Estadísticas de texto

### Conectores
- `yt-dlp>=2024.8.0` - Extracción de videos y transcripciones
- `instaloader>=4.12.0` - Scraping de Instagram
- `google-api-python-client>=2.150.0` - YouTube Data API v3

## Compatibilidad y Fallbacks

Todos los componentes tienen fallbacks graceful:

1. **TextProcessor:**
   - Si VaderSentiment no está disponible → usa método básico
   - Si Spacy no está disponible → omite análisis avanzado
   - Si emoji no está disponible → omite análisis de emojis

2. **Conectores:**
   - Si yt-dlp no está disponible → retorna None con warning
   - Si Instaloader no está disponible → usa API oficial o estructura de ejemplo
   - Si YouTube API no está disponible → retorna estructura de ejemplo

## Mejoras de Performance

1. **TextProcessor:**
   - VaderSentiment es más rápido y preciso que análisis básico
   - Spacy puede cachear modelos para reutilización
   - Análisis de emojis es muy rápido

2. **Conectores:**
   - yt-dlp es más rápido y confiable que youtube-dl
   - Instaloader es eficiente para scraping de Instagram
   - YouTube API v3 es oficial y confiable

## Próximos Pasos

### Pendientes:
- [ ] Implementar parsing completo de subtítulos de yt-dlp
- [ ] Agregar soporte para más idiomas en transcripciones
- [ ] Implementar caché para perfiles de Instagram (Instaloader)
- [ ] Agregar tests para nuevos componentes
- [ ] Documentar configuración de Spacy models

### Mejoras Futuras:
- [ ] Usar ChromaDB para almacenar embeddings de identidades
- [ ] Integrar Celery para tareas asíncronas de extracción
- [ ] Agregar rate limiting con slowapi
- [ ] Implementar monitoreo con Sentry

## Instalación de Dependencias

```bash
# Instalar todas las dependencias
pip install -r requirements.txt

# O instalar solo las nuevas
pip install vaderSentiment spacy emoji textstat yt-dlp instaloader

# Descargar modelos de Spacy
python -m spacy download es_core_news_sm  # Para español
python -m spacy download en_core_web_sm    # Para inglés
```

## Ejemplos de Uso

### TextProcessor Mejorado

```python
from utils.text_processor import TextProcessor

# Inicializar con modelo de Spacy
processor = TextProcessor(spacy_model='es_core_news_sm')

# Análisis completo
text = "¡Este video es increíble! 😍 #amazing #viral"
analysis = processor.analyze_basic(text)

print(analysis.sentiment_analysis)
# {'positive': 0.8, 'negative': 0.0, 'neutral': 0.2, 'compound': 0.8}

print(analysis.language_patterns)
# {'emoji_count': 1, 'emojis': ['😍'], 'entities': [...]}

# Estadísticas
stats = processor.get_text_statistics(text)
print(stats['readability'])  # Score de legibilidad
```

### InstagramConnector con Scraping

```python
from connectors.instagram_connector import InstagramConnector

# Inicializar con scraping
connector = InstagramConnector(use_scraping=True)

# Extraer perfil completo
profile = await connector.get_profile("username")
print(profile['followers_count'])
print(profile['bio'])

# Extraer posts
posts = await connector.get_posts("username", limit=50)
for post in posts:
    print(post['caption'])
    print(post['likes'])
```

### YouTubeConnector Completo

```python
from connectors.youtube_connector import YouTubeConnector

connector = YouTubeConnector(api_key="YOUR_API_KEY")

# Canal completo
channel = await connector.get_channel("UC...")
print(f"Subscribers: {channel['subscribers_count']}")

# Videos
videos = await connector.get_videos("UC...", limit=100)

# Transcripción
transcript = await connector.get_video_transcript("video_id")
```

## Notas de Migración

1. **TextProcessor:** Compatible hacia atrás, pero mejor usar nuevos métodos
2. **Conectores:** Agregar parámetros opcionales, no rompe código existente
3. **Dependencias:** Instalar nuevas librerías según necesidades

## Testing

```bash
# Ejecutar tests
pytest tests/

# Tests específicos
pytest tests/test_text_processor.py
pytest tests/test_connectors.py
```

## Conclusión

Esta refactorización mejora significativamente:
- ✅ Calidad del análisis de texto
- ✅ Capacidades de extracción de datos
- ✅ Manejo de errores y fallbacks
- ✅ Type safety y documentación
- ✅ Performance y eficiencia

El código es más robusto, mantenible y utiliza las mejores prácticas de la industria.


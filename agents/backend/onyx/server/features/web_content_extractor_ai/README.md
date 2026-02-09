# Web Content Extractor AI

Sistema avanzado para extraer información completa de páginas web usando OpenRouter y múltiples métodos de scraping.

## Características

### ✨ Extracción Avanzada
- **Múltiples métodos de extracción**: Trafilatura, Readability, Newspaper3k, BeautifulSoup
- **Scraping avanzado**: Soporte para JavaScript con Playwright
- **Extracción de tablas**: Tablas HTML estructuradas
- **Extracción de videos**: HTML5, embeds (YouTube, Vimeo, etc.)
- **Extracción de citas**: Blockquotes y citas inline
- **Extracción de código**: Bloques de código con detección de lenguaje
- **Extracción de formularios**: Campos y estructura de formularios
- **Feeds RSS/Atom**: Detección y extracción de feeds

### 🧠 Análisis Inteligente
- **Procesamiento con IA**: OpenRouter para análisis y estructuración de contenido
- **Detección de idioma**: Detección avanzada con múltiples idiomas
- **Análisis de calidad**: Métricas de legibilidad (Flesch, Flesch-Kincaid, etc.)
- **Metadatos enriquecidos**: Autor, fecha, keywords, Open Graph, Twitter Cards, JSON-LD
- **Datos estructurados**: Microdata, Schema.org

### ⚡ Rendimiento
- **Cache inteligente**: Sistema de cache con TTL para optimizar rendimiento
- **Batch scraping**: Procesamiento paralelo de múltiples URLs
- **Retry automático**: Reintentos con backoff exponencial
- **Rate limiting**: Control de velocidad de requests
- **User agents rotativos**: Evita bloqueos

### 🔌 API REST
- **Endpoints documentados**: FastAPI con Swagger/OpenAPI
- **Validación de datos**: Pydantic para validación robusta
- **Manejo de errores**: Respuestas de error estructuradas

## Instalación

```bash
# Instalar dependencias esenciales
pip install -r requirements.txt

# Instalar dependencias opcionales (para funcionalidades avanzadas)
pip install -r requirements-optional.txt

# Instalar navegadores para Playwright (opcional, solo si usas JavaScript rendering)
playwright install chromium
```

## Configuración

Crea un archivo `.env` basado en `.env.example`:

```bash
OPENROUTER_API_KEY=tu_api_key_aqui
HOST=0.0.0.0
PORT=8000
CACHE_MAX_SIZE=1000
CACHE_TTL=3600
```

## Inicio Rápido

```bash
# Linux/Mac
chmod +x scripts/start.sh
./scripts/start.sh

# Windows
scripts\start.bat

# O manualmente
python main.py
```

## Uso

### Iniciar servidor

```bash
python main.py
```

O con uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

#### Extraer contenido

```bash
POST /api/v1/extract
Content-Type: application/json

{
  "url": "https://example.com",
  "model": "anthropic/claude-3.5-sonnet",
  "max_tokens": 4000
}
```

#### Estadísticas de cache

```bash
GET /api/v1/extract/cache/stats
```

#### Limpiar cache

```bash
DELETE /api/v1/extract/cache
```

#### Extracción en batch (múltiples URLs)

```bash
POST /api/v1/extract/batch
Content-Type: application/json

{
  "urls": [
    "https://example.com",
    "https://example.org",
    "https://example.net"
  ],
  "max_concurrent": 5,
  "extract_strategy": "auto"
}
```

### Ejemplo con curl

```bash
# Extraer contenido
curl -X POST "http://localhost:8000/api/v1/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "model": "anthropic/claude-3.5-sonnet"
  }'

# Ver estadísticas de cache
curl "http://localhost:8000/api/v1/extract/cache/stats"

# Limpiar cache
curl -X DELETE "http://localhost:8000/api/v1/extract/cache"

# Extracción en batch
curl -X POST "http://localhost:8000/api/v1/extract/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com", "https://example.org"],
    "max_concurrent": 5
  }'
```

### Ejemplo con Python

```python
import asyncio
from example_usage import example_extract, example_cache_stats

# Extraer contenido
asyncio.run(example_extract())

# Ver estadísticas
asyncio.run(example_cache_stats())
```

## Modelos disponibles

Puedes usar cualquier modelo de OpenRouter. Recomendados:

- `anthropic/claude-3.5-sonnet` (por defecto) - Mejor calidad
- `anthropic/claude-3-haiku` - Más rápido y económico
- `openai/gpt-4-turbo` - Excelente para análisis
- `google/gemini-pro` - Buen balance calidad/precio

## Métodos de extracción

El sistema intenta automáticamente en este orden:

1. **Trafilatura** - Mejor para artículos y contenido estructurado
2. **Readability** - Extrae contenido principal limpio
3. **Newspaper3k** - Ideal para noticias y artículos
4. **BeautifulSoup** - Fallback siempre disponible

## Contenido extraído

El scraper extrae:

- ✅ **Texto principal** - Contenido limpio y estructurado
- ✅ **Metadatos** - Título, descripción, autor, fecha, keywords
- ✅ **Enlaces** - Todos los enlaces con texto y URLs normalizadas
- ✅ **Imágenes** - Con alt text, dimensiones y URLs
- ✅ **Tablas** - Estructura completa con headers y filas
- ✅ **Videos** - HTML5, embeds (YouTube, Vimeo, etc.)
- ✅ **Citas** - Blockquotes y citas inline con autores
- ✅ **Código** - Bloques de código con detección de lenguaje
- ✅ **Formularios** - Campos y estructura completa
- ✅ **Feeds** - RSS/Atom feeds detectados
- ✅ **Datos estructurados** - JSON-LD, Microdata, Open Graph
- ✅ **Análisis de calidad** - Métricas de legibilidad
- ✅ **Detección de idioma** - Con nivel de confianza

## Estructura del proyecto

```
web_content_extractor_ai/
├── main.py                          # Servidor FastAPI
├── config.py                        # Configuración
├── example_usage.py                 # Ejemplos de uso
├── infrastructure/
│   ├── openrouter/
│   │   └── client.py               # Cliente OpenRouter
│   ├── web_scraper/
│   │   └── scraper.py              # Scraper multi-método
│   └── cache/
│       └── content_cache.py        # Sistema de cache
├── application/
│   └── use_cases/
│       └── extract_content_use_case.py
├── api/
│   └── v1/
│       ├── controllers/
│       │   └── extract_controller.py
│       ├── schemas/
│       │   ├── requests.py
│       │   └── responses.py
│       └── routes.py
└── requirements.txt
```

## Respuesta de ejemplo

```json
{
  "success": true,
  "url": "https://example.com",
  "raw_data": {
    "title": "Example Domain",
    "description": "...",
    "links_count": 5,
    "images_count": 2,
    "extraction_method": "trafilatura"
  },
  "extracted_info": "{\"titulo\": \"...\", \"contenido\": \"...\"}",
  "processing_metadata": {
    "model_used": "anthropic/claude-3.5-sonnet",
    "tokens_used": 1234
  },
  "message": "Contenido extraído exitosamente"
}
```

## Parámetros de configuración

- `use_cache`: Usar cache (default: true)
- `use_javascript`: Renderizar JavaScript con Playwright (default: false, más lento)
- `extract_strategy`: Forzar método específico ("auto", "trafilatura", "readability", "newspaper", "beautifulsoup")

## Documentación API

Una vez iniciado el servidor, visita:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker

### Construir y ejecutar

```bash
# Construir imagen
docker build -t web-content-extractor-ai .

# Ejecutar
docker run -p 8000:8000 -e OPENROUTER_API_KEY=tu_key web-content-extractor-ai

# O con docker-compose
docker-compose up
```

## Testing

```bash
# Ejecutar tests
pytest tests/

# Con coverage
pytest tests/ --cov=. --cov-report=html
```

## Notas

- El cache tiene TTL de 1 hora por defecto
- Playwright requiere más recursos pero maneja mejor páginas con JavaScript
- Trafilatura es generalmente el método más efectivo para artículos
- El sistema detecta automáticamente el encoding del contenido
- Rate limiting: 100 requests por minuto por IP

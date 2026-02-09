# Guía Rápida de API - Music Analyzer AI

## 🚀 Inicio Rápido

### Base URL

```
http://localhost:8010
```

### Autenticación

La mayoría de endpoints no requieren autenticación. Algunos endpoints avanzados pueden requerir tokens de Spotify.

## 📖 Endpoints Principales

### 1. Buscar Canciones

Busca canciones en Spotify usando una query de texto.

**Endpoint**: `POST /music/search`

**Request Body**:
```json
{
  "query": "Bohemian Rhapsody Queen",
  "limit": 5
}
```

**Response**:
```json
{
  "success": true,
  "query": "Bohemian Rhapsody Queen",
  "results": [
    {
      "id": "4uLU6hMCjMI75M1A2tKUQC",
      "name": "Bohemian Rhapsody",
      "artists": ["Queen"],
      "album": "A Night At The Opera",
      "duration_ms": 355000,
      "preview_url": "https://...",
      "popularity": 85
    }
  ],
  "total": 1
}
```

**Ejemplo con cURL**:
```bash
curl -X POST http://localhost:8010/music/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Bohemian Rhapsody", "limit": 5}'
```

### 2. Analizar Canción

Analiza una canción completa con información musical detallada.

**Endpoint**: `POST /music/analyze`

**Request Body**:
```json
{
  "track_id": "4uLU6hMCjMI75M1A2tKUQC",
  "include_coaching": true
}
```

**Response**:
```json
{
  "success": true,
  "track_id": "4uLU6hMCjMI75M1A2tKUQC",
  "track_name": "Bohemian Rhapsody",
  "artists": ["Queen"],
  "album": "A Night At The Opera",
  "duration_seconds": 355,
  "analysis": {
    "key": "Bb",
    "mode": "major",
    "tempo": 72,
    "time_signature": 4,
    "energy": 0.7,
    "danceability": 0.5,
    "valence": 0.6
  },
  "coaching": {
    "learning_path": [...],
    "exercises": [...],
    "tips": [...]
  }
}
```

**Ejemplo con cURL**:
```bash
curl -X POST http://localhost:8010/music/analyze \
  -H "Content-Type: application/json" \
  -d '{"track_id": "4uLU6hMCjMI75M1A2tKUQC", "include_coaching": true}'
```

### 3. Analizar por ID (GET)

Obtiene el análisis de una canción usando su ID de Spotify.

**Endpoint**: `GET /music/analyze/{track_id}`

**Query Parameters**:
- `include_coaching` (opcional): `true` o `false` (default: `false`)

**Ejemplo**:
```bash
curl http://localhost:8010/music/analyze/4uLU6hMCjMI75M1A2tKUQC?include_coaching=true
```

### 4. Obtener Recomendaciones

Obtiene canciones similares basadas en una canción.

**Endpoint**: `GET /music/recommendations/{track_id}`

**Query Parameters**:
- `limit` (opcional): Número de recomendaciones (default: 10, max: 50)

**Ejemplo**:
```bash
curl http://localhost:8010/music/recommendations/4uLU6hMCjMI75M1A2tKUQC?limit=5
```

### 5. Coaching Musical

Obtiene coaching personalizado para una canción.

**Endpoint**: `POST /music/coaching`

**Request Body**:
```json
{
  "track_id": "4uLU6hMCjMI75M1A2tKUQC",
  "skill_level": "intermediate",
  "instrument": "piano"
}
```

**Response**:
```json
{
  "success": true,
  "track_id": "4uLU6hMCjMI75M1A2tKUQC",
  "coaching": {
    "learning_path": [
      {
        "step": 1,
        "title": "Aprender la estructura",
        "description": "..."
      }
    ],
    "exercises": [...],
    "tips": [...]
  }
}
```

## 🔍 API v1 (Nueva Arquitectura)

### Analizar Track

**Endpoint**: `POST /v1/music/analyze`

Similar al endpoint legacy pero con mejor estructura y manejo de errores.

**Request Body**:
```json
{
  "track_id": "4uLU6hMCjMI75M1A2tKUQC",
  "track_name": null,
  "include_coaching": true
}
```

**Nota**: Puedes usar `track_id` o `track_name`. Si usas `track_name`, el sistema buscará la canción primero.

## 📊 Health Check

**Endpoint**: `GET /health`

Verifica el estado del servidor.

**Response**:
```json
{
  "status": "healthy",
  "version": "2.21.0",
  "timestamp": "2025-01-XXT..."
}
```

## 🐍 Ejemplos en Python

### Usando requests

```python
import requests

BASE_URL = "http://localhost:8010"

# Buscar canciones
response = requests.post(
    f"{BASE_URL}/music/search",
    json={"query": "Bohemian Rhapsody", "limit": 5}
)
results = response.json()

# Analizar canción
response = requests.post(
    f"{BASE_URL}/music/analyze",
    json={"track_id": "4uLU6hMCjMI75M1A2tKUQC", "include_coaching": True}
)
analysis = response.json()
```

### Usando httpx (async)

```python
import httpx
import asyncio

BASE_URL = "http://localhost:8010"

async def analyze_track(track_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/music/analyze",
            json={"track_id": track_id, "include_coaching": True}
        )
        return response.json()

# Uso
result = asyncio.run(analyze_track("4uLU6hMCjMI75M1A2tKUQC"))
```

## 📚 Documentación Interactiva

Una vez que el servidor esté corriendo, puedes acceder a:

- **Swagger UI**: http://localhost:8010/docs
- **ReDoc**: http://localhost:8010/redoc

Estas interfaces te permiten:
- Ver todos los endpoints disponibles
- Probar endpoints directamente desde el navegador
- Ver esquemas de request/response
- Ver ejemplos de uso

## 🚨 Manejo de Errores

### Errores Comunes

#### 404 - Track Not Found

```json
{
  "detail": "Track not found: 4uLU6hMCjMI75M1A2tKUQC"
}
```

**Solución**: Verificar que el track_id es correcto.

#### 401 - Spotify Authentication Failed

```json
{
  "detail": "Spotify authentication failed"
}
```

**Solución**: Verificar credenciales de Spotify en `.env`.

#### 429 - Rate Limit Exceeded

```json
{
  "detail": "Rate limit exceeded. Please try again later."
}
```

**Solución**: Esperar antes de hacer más requests.

#### 500 - Internal Server Error

```json
{
  "detail": "Internal server error"
}
```

**Solución**: Revisar logs del servidor para más detalles.

## 🔗 Recursos Adicionales

- **Documentación Completa**: Ver `README.md`
- **Arquitectura**: Ver `ARCHITECTURE_QUICK_START.md`
- **Referencia Rápida**: Ver `QUICK_REFERENCE.md`
- **Spotify API Docs**: https://developer.spotify.com/documentation/web-api/

---

**Última actualización**: 2025  
**Versión**: 2.21.0







# 🔄 Guía de Migración - Music Analyzer AI

Esta guía te ayudará a migrar entre versiones de Music Analyzer AI o desde otras soluciones.

## 📋 Índice

- [Migración entre Versiones](#migración-entre-versiones)
- [Migración desde Spotify API Directa](#migración-desde-spotify-api-directa)
- [Migración desde Otras Soluciones](#migración-desde-otras-soluciones)
- [Actualización de Código](#actualización-de-código)
- [Migración de Datos](#migración-de-datos)

## 🔄 Migración entre Versiones

### De 2.0.x a 2.21.0

#### Cambios en Endpoints

**Antes (2.0.x):**
```python
POST /music/analyze
{
  "track_id": "4uLU6hMCjMI75M1A2tKUQC"
}
```

**Después (2.21.0):**
```python
POST /music/analyze
{
  "track_id": "4uLU6hMCjMI75M1A2tKUQC",
  "include_coaching": false  # Nuevo parámetro opcional
}
```

#### Cambios en Respuestas

**Antes:**
```json
{
  "track_id": "...",
  "analysis": {...}
}
```

**Después:**
```json
{
  "success": true,
  "track_id": "...",
  "analysis": {...},
  "coaching": {...}  // Si include_coaching=true
}
```

#### Migración de Código

```python
# Antes
response = requests.post(
    f"{BASE_URL}/music/analyze",
    json={"track_id": track_id}
)
analysis = response.json()["analysis"]

# Después
response = requests.post(
    f"{BASE_URL}/music/analyze",
    json={"track_id": track_id, "include_coaching": True}
)
data = response.json()
analysis = data["analysis"]
coaching = data.get("coaching")  # Opcional
```

### De 1.x a 2.0.x

#### Cambios Arquitectónicos

**Antes (1.x):**
```python
from services.spotify_service import SpotifyService

spotify = SpotifyService()
track = spotify.get_track(track_id)
```

**Después (2.0.x):**
```python
from api.dependencies import get_spotify_service

# En endpoints
@router.post("/analyze")
async def analyze(
    spotify_service: ISpotifyService = Depends(get_spotify_service)
):
    track = await spotify_service.get_track(track_id)
```

#### Migración de Servicios

1. **Actualizar imports:**
```python
# Antes
from services.spotify_service import SpotifyService

# Después
from domain.interfaces.spotify import ISpotifyService
from api.dependencies import get_spotify_service
```

2. **Actualizar uso:**
```python
# Antes
spotify = SpotifyService()
track = spotify.get_track(track_id)

# Después
spotify_service = get_spotify_service()
track = await spotify_service.get_track(track_id)
```

## 🎵 Migración desde Spotify API Directa

### Paso 1: Instalar Music Analyzer AI

```bash
cd music_analyzer_ai
pip install -r requirements.txt
```

### Paso 2: Configurar Credenciales

```env
# .env
SPOTIFY_CLIENT_ID=tu_client_id
SPOTIFY_CLIENT_SECRET=tu_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8010/callback
```

### Paso 3: Actualizar Código

#### Antes (Spotify API Directa)

```python
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
)

# Búsqueda
results = sp.search(q="Bohemian Rhapsody", type="track", limit=5)

# Análisis
track = sp.track("4uLU6hMCjMI75M1A2tKUQC")
audio_features = sp.audio_features("4uLU6hMCjMI75M1A2tKUQC")
```

#### Después (Music Analyzer AI)

```python
import requests

BASE_URL = "http://localhost:8010"

# Búsqueda
response = requests.post(
    f"{BASE_URL}/music/search",
    json={"query": "Bohemian Rhapsody", "limit": 5}
)
results = response.json()

# Análisis (con análisis avanzado incluido)
response = requests.post(
    f"{BASE_URL}/music/analyze",
    json={"track_id": "4uLU6hMCjMI75M1A2tKUQC", "include_coaching": True}
)
analysis = response.json()
```

### Ventajas de la Migración

- ✅ Análisis musical avanzado incluido
- ✅ Coaching musical automático
- ✅ Recomendaciones inteligentes
- ✅ Estructura de datos enriquecida
- ✅ Caché automático
- ✅ API más simple

## 🔄 Migración desde Otras Soluciones

### Desde Last.fm API

#### Diferencias Clave

| Last.fm | Music Analyzer AI |
|---------|-------------------|
| `track.getInfo()` | `POST /music/analyze` |
| Datos de scrobbling | Análisis técnico |
| Basado en usuarios | Basado en audio |

#### Migración

```python
# Antes (Last.fm)
import pylast

network = pylast.LastFMNetwork(api_key, api_secret)
track = network.get_track("Queen", "Bohemian Rhapsody")
info = track.get_info()

# Después (Music Analyzer AI)
response = requests.post(
    f"{BASE_URL}/music/search",
    json={"query": "Bohemian Rhapsody Queen", "limit": 1}
)
track_id = response.json()["results"][0]["id"]

response = requests.post(
    f"{BASE_URL}/music/analyze",
    json={"track_id": track_id}
)
analysis = response.json()
```

### Desde APIs Comerciales

#### Migración Genérica

1. **Identificar endpoints equivalentes:**
   - Búsqueda → `POST /music/search`
   - Análisis → `POST /music/analyze`
   - Recomendaciones → `GET /music/recommendations/{track_id}`

2. **Actualizar autenticación:**
   ```python
   # Antes (API comercial)
   headers = {"Authorization": f"Bearer {api_key}"}
   
   # Después (Music Analyzer AI - self-hosted)
   # No requiere autenticación para endpoints básicos
   headers = {"Content-Type": "application/json"}
   ```

3. **Actualizar formato de requests:**
   ```python
   # Adaptar formato de request a Music Analyzer AI
   # Ver documentación de API para formato exacto
   ```

## 💻 Actualización de Código

### Cliente Python

#### Versión Antigua

```python
class MusicClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.commercial.com"
    
    def analyze(self, track_id):
        response = requests.get(
            f"{self.base_url}/analyze/{track_id}",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()
```

#### Versión Nueva

```python
class MusicAnalyzerClient:
    def __init__(self, base_url="http://localhost:8010"):
        self.base_url = base_url
    
    def analyze(self, track_id, include_coaching=False):
        response = requests.post(
            f"{self.base_url}/music/analyze",
            json={
                "track_id": track_id,
                "include_coaching": include_coaching
            }
        )
        return response.json()
```

### Cliente JavaScript

#### Versión Antigua

```javascript
class MusicClient {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.baseUrl = 'https://api.commercial.com';
  }
  
  async analyze(trackId) {
    const response = await fetch(
      `${this.baseUrl}/analyze/${trackId}`,
      {
        headers: { 'Authorization': `Bearer ${this.apiKey}` }
      }
    );
    return await response.json();
  }
}
```

#### Versión Nueva

```javascript
class MusicAnalyzerClient {
  constructor(baseUrl = 'http://localhost:8010') {
    this.baseUrl = baseUrl;
  }
  
  async analyze(trackId, includeCoaching = false) {
    const response = await fetch(`${this.baseUrl}/music/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        track_id: trackId,
        include_coaching: includeCoaching
      })
    });
    return await response.json();
  }
}
```

## 🗄️ Migración de Datos

### Migración de Base de Datos

#### SQLite a PostgreSQL

```python
import sqlite3
import psycopg2

# Conectar a SQLite
sqlite_conn = sqlite3.connect('music_analyzer.db')
sqlite_cursor = sqlite_conn.cursor()

# Conectar a PostgreSQL
pg_conn = psycopg2.connect(DATABASE_URL)
pg_cursor = pg_conn.cursor()

# Migrar datos
sqlite_cursor.execute("SELECT * FROM analyses")
analyses = sqlite_cursor.fetchall()

for analysis in analyses:
    pg_cursor.execute(
        "INSERT INTO analyses (track_id, analysis_data, created_at) VALUES (%s, %s, %s)",
        (analysis[1], analysis[2], analysis[3])
    )

pg_conn.commit()
```

### Migración de Caché

```python
# Exportar de caché antigua
old_cache = {}
# ... cargar datos

# Importar a Redis
import redis
redis_client = redis.Redis.from_url(REDIS_URL)

for key, value in old_cache.items():
    redis_client.setex(key, 3600, json.dumps(value))
```

## ✅ Checklist de Migración

### Pre-Migración

- [ ] Backup de datos existentes
- [ ] Documentar configuración actual
- [ ] Identificar dependencias
- [ ] Plan de rollback

### Durante Migración

- [ ] Instalar nueva versión
- [ ] Configurar variables de entorno
- [ ] Actualizar código
- [ ] Migrar datos
- [ ] Probar funcionalidad

### Post-Migración

- [ ] Verificar todos los endpoints
- [ ] Verificar rendimiento
- [ ] Monitorear errores
- [ ] Actualizar documentación
- [ ] Notificar usuarios

## 🚨 Problemas Comunes

### Error: Endpoint no encontrado

**Solución:**
- Verificar que estás usando la versión correcta de la API
- Revisar [CHANGELOG.md](CHANGELOG.md) para cambios

### Error: Formato de request incorrecto

**Solución:**
- Revisar [API_QUICK_START.md](API_QUICK_START.md)
- Verificar schemas en `/docs`

### Error: Datos no migrados

**Solución:**
- Verificar scripts de migración
- Revisar logs de migración
- Hacer backup antes de reintentar

## 📚 Recursos

- [CHANGELOG.md](CHANGELOG.md) - Cambios entre versiones
- [API_QUICK_START.md](API_QUICK_START.md) - Formato de API actual
- [COMPARISON.md](COMPARISON.md) - Comparación con otras soluciones

---

**Última actualización**: 2025  
**Versión**: 2.21.0







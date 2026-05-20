# ✨ Mejores Prácticas - Music Analyzer AI

## 🏗️ Arquitectura y Código

### 1. Usa Dependency Injection

✅ **Bien:**
```python
@router.post("/analyze")
async def analyze_track(
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    result = await use_case.execute(...)
    return result
```

❌ **Mal:**
```python
@router.post("/analyze")
async def analyze_track(track_id: str):
    spotify = SpotifyService()  # Instanciación directa
    analyzer = MusicAnalyzer()
    # ...
```

### 2. Separa Responsabilidades

✅ **Bien:**
- Controllers: Solo manejan HTTP
- Use Cases: Lógica de negocio
- Services: Operaciones específicas
- Repositories: Acceso a datos

❌ **Mal:**
- Todo mezclado en un solo archivo
- Lógica de negocio en controllers
- Acceso directo a APIs externas desde controllers

### 3. Usa Type Hints

✅ **Bien:**
```python
async def analyze_track(
    track_id: str,
    include_coaching: bool = False
) -> AnalysisResponse:
    # ...
```

❌ **Mal:**
```python
async def analyze_track(track_id, include_coaching=False):
    # ...
```

### 4. Maneja Errores Apropiadamente

✅ **Bien:**
```python
try:
    result = await use_case.execute(track_id)
except TrackNotFoundException:
    raise HTTPException(status_code=404, detail="Track not found")
except AnalysisException as e:
    logger.error(f"Analysis failed: {e}")
    raise HTTPException(status_code=500, detail="Analysis failed")
```

❌ **Mal:**
```python
result = await use_case.execute(track_id)  # Sin manejo de errores
```

## 🔐 Seguridad

### 1. Nunca Commitees Credenciales

✅ **Bien:**
```bash
# .env (en .gitignore)
SPOTIFY_CLIENT_ID=tu_id
SPOTIFY_CLIENT_SECRET=tu_secret
```

❌ **Mal:**
```python
# main.py
SPOTIFY_CLIENT_ID = "hardcoded_id"  # NUNCA hacer esto
```

### 2. Usa Variables de Entorno

✅ **Bien:**
```python
from config.settings import settings

client_id = settings.SPOTIFY_CLIENT_ID
```

❌ **Mal:**
```python
client_id = "hardcoded_value"
```

### 3. Valida Inputs

✅ **Bien:**
```python
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    track_id: str = Field(..., min_length=1, max_length=50)
    include_coaching: bool = False
```

❌ **Mal:**
```python
@router.post("/analyze")
async def analyze(track_id: str):  # Sin validación
    # ...
```

### 4. Implementa Rate Limiting

✅ **Bien:**
```env
RATE_LIMIT_ENABLED=True
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60
```

## 📊 Rendimiento

### 1. Usa Caché

✅ **Bien:**
```python
# Configurar Redis
REDIS_URL=redis://localhost:6379/0
CACHE_ENABLED=True
CACHE_TTL=3600
```

❌ **Mal:**
```python
# Sin caché, cada request hace llamada a Spotify
```

### 2. Usa Async/Await

✅ **Bien:**
```python
async def get_track(track_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

❌ **Mal:**
```python
def get_track(track_id: str):  # Síncrono bloquea el servidor
    response = requests.get(url)
    return response.json()
```

### 3. Procesa en Lotes cuando sea Posible

✅ **Bien:**
```python
async def analyze_multiple_tracks(track_ids: List[str]):
    tasks = [analyze_track(tid) for tid in track_ids]
    return await asyncio.gather(*tasks)
```

❌ **Mal:**
```python
def analyze_multiple_tracks(track_ids: List[str]):
    results = []
    for tid in track_ids:  # Secuencial, lento
        results.append(analyze_track(tid))
    return results
```

## 🧪 Testing

### 1. Escribe Tests

✅ **Bien:**
```python
@pytest.mark.asyncio
async def test_analyze_track():
    use_case = AnalyzeTrackUseCase(
        spotify_service=mock_spotify,
        analysis_service=mock_analyzer
    )
    result = await use_case.execute("track_id")
    assert result.track_id == "track_id"
```

### 2. Usa Mocks

✅ **Bien:**
```python
from unittest.mock import Mock

mock_spotify = Mock(spec=ISpotifyService)
mock_spotify.get_track.return_value = Track(...)
```

### 3. Testea Casos Edge

✅ **Bien:**
```python
def test_analyze_track_not_found():
    # Test cuando track no existe
    pass

def test_analyze_track_invalid_id():
    # Test con ID inválido
    pass
```

## 📝 Logging

### 1. Usa Logging Estructurado

✅ **Bien:**
```python
import structlog

logger = structlog.get_logger()
logger.info("track_analyzed", track_id=track_id, duration=duration)
```

❌ **Mal:**
```python
print(f"Track {track_id} analyzed")  # No usar print
```

### 2. Niveles Apropiados

✅ **Bien:**
```python
logger.debug("Detailed debug info")  # Para desarrollo
logger.info("Track analyzed")         # Información general
logger.warning("Rate limit approaching")  # Advertencias
logger.error("Analysis failed", exc_info=True)  # Errores
```

### 3. No Logs Sensibles

✅ **Bien:**
```python
logger.info("Authentication successful", user_id=user_id)
```

❌ **Mal:**
```python
logger.info(f"Auth: {client_secret}")  # NUNCA loggear secrets
```

## 🔄 Versionado

### 1. Versiona tu API

✅ **Bien:**
```python
# API v1
router_v1 = APIRouter(prefix="/v1")

# API v2 (nueva versión)
router_v2 = APIRouter(prefix="/v2")
```

### 2. Documenta Cambios

✅ **Bien:**
```markdown
# CHANGELOG.md
## [2.21.0] - 2025-01-XX
### Added
- Nuevo endpoint de recomendaciones
- Soporte para análisis en lote
```

## 🚀 Despliegue

### 1. Usa Variables de Entorno por Entorno

✅ **Bien:**
```env
# .env.development
DEBUG=True
LOG_LEVEL=DEBUG

# .env.production
DEBUG=False
LOG_LEVEL=WARNING
```

### 2. Health Checks

✅ **Bien:**
```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "2.21.0",
        "checks": {
            "database": await check_database(),
            "redis": await check_redis(),
            "spotify": await check_spotify()
        }
    }
```

### 3. Monitoreo

✅ **Bien:**
```env
PROMETHEUS_ENABLED=True
OTEL_ENABLED=True
```

## 📚 Documentación

### 1. Documenta Endpoints

✅ **Bien:**
```python
@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_track(
    request: AnalyzeTrackRequest,
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    """
    Analiza una canción de Spotify.
    
    - **track_id**: ID de la canción en Spotify
    - **include_coaching**: Incluir coaching musical
    """
    # ...
```

### 2. Mantén README Actualizado

✅ **Bien:**
- README con instrucciones claras
- Ejemplos de uso
- Información de configuración
- Enlaces a documentación

## 🎯 Resumen

### ✅ Haz Esto

- ✅ Usa Dependency Injection
- ✅ Separa responsabilidades
- ✅ Escribe tests
- ✅ Usa type hints
- ✅ Maneja errores apropiadamente
- ✅ Usa variables de entorno
- ✅ Implementa caché
- ✅ Usa async/await
- ✅ Documenta tu código
- ✅ Versiona tu API

### ❌ Evita Esto

- ❌ Hardcodear credenciales
- ❌ Mezclar responsabilidades
- ❌ Ignorar errores
- ❌ Usar código síncrono cuando async es mejor
- ❌ Committear archivos sensibles
- ❌ Loggear información sensible
- ❌ Sin tests
- ❌ Sin documentación

---

**Última actualización**: 2025  
**Versión**: 2.21.0







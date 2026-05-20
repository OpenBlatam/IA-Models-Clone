# Guía Rápida de Referencia - Music Analyzer AI

## 🚀 Inicio Rápido

### Estructura de Archivos Clave

```
music_analyzer_ai/
├── api/
│   ├── v1/                    # Nueva API (use cases)
│   │   ├── controllers/      # Controllers refactorizados
│   │   ├── schemas/          # Validación Pydantic
│   │   └── routes.py         # Router principal v1
│   ├── routes/                # Routers modulares (legacy refactorizado)
│   ├── factories.py          # Factory functions para servicios
│   ├── base_router.py        # BaseRouter con DI
│   └── music_api.py          # API legacy (refactorizada)
│
├── application/
│   ├── use_cases/            # Lógica de negocio
│   │   ├── analysis/
│   │   └── recommendations/
│   └── dto/                  # Data Transfer Objects
│
├── domain/
│   └── interfaces/           # Contratos de dominio
│
├── infrastructure/
│   ├── repositories/         # Implementaciones de repositorios
│   └── adapters/             # Adaptadores para servicios
│
├── core/
│   └── di/                   # Dependency Injection
│
├── services/
│   ├── spotify_service.py    # Servicio de Spotify
│   └── music_coach.py        # Coaching musical
│
└── config/
    └── di_setup.py           # Configuración de DI
```

## 📝 Patrones de Uso

### Obtener Servicios en Controllers

```python
from ...dependencies import get_analyze_track_use_case

@router.post("/analyze")
async def analyze_track(
    request: AnalyzeTrackRequest,
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    result = await use_case.execute(request.track_id)
    return result
```

### Obtener Servicios en Routers Legacy

```python
from ..base_router import BaseRouter

class MyRouter(BaseRouter):
    def my_endpoint(self):
        # Usa DI automáticamente
        spotify_service = self.get_service("spotify_service")
        music_analyzer = self.get_service("music_analyzer")
        
        # O múltiples a la vez
        spotify, analyzer = self.get_services("spotify_service", "music_analyzer")
```

### Obtener Servicios en Factories

```python
from ..factories import get_spotify_service, get_music_analyzer

def my_function():
    spotify = get_spotify_service()
    analyzer = get_music_analyzer()
    # Usar servicios...
```

## 🎯 Endpoints Principales

### API v1 (Nueva Arquitectura)
- `POST /v1/music/analyze` - Analizar track
- `GET /v1/music/analyze/{track_id}` - Analizar por ID
- `POST /v1/music/search` - Buscar tracks
- `GET /v1/music/recommendations/track/{track_id}` - Recomendaciones

### API Legacy (Refactorizada)
- `POST /music/analyze` - Analizar (usa DI)
- `POST /music/search` - Buscar (usa DI)
- `POST /music/coaching` - Coaching musical
- `GET /music/recommendations/{track_id}` - Recomendaciones
- Todos los demás endpoints en `/music/*`

## 🔧 Configuración

### DI Setup
```python
# main.py
from config.di_setup import setup_dependencies

setup_dependencies()  # Configura todos los servicios
```

### Agregar Nuevo Servicio
```python
# config/di_setup.py
register_service(
    "my_service",
    MyService,
    singleton=True,
    dependencies=["other_service"]
)
```

### Variables de Entorno

```env
# Spotify API (REQUERIDO)
SPOTIFY_CLIENT_ID=tu_client_id
SPOTIFY_CLIENT_SECRET=tu_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8010/callback

# Server
HOST=0.0.0.0
PORT=8010
LOG_LEVEL=INFO

# Redis (opcional)
REDIS_URL=redis://localhost:6379/0
```

## 📚 Interfaces Principales

### Repositorios
- `ITrackRepository` - Acceso a tracks
- `IUserRepository` - Acceso a usuarios
- `IPlaylistRepository` - Acceso a playlists

### Servicios
- `ISpotifyService` - Integración con Spotify
- `IAnalysisService` - Análisis musical
- `ICoachingService` - Coaching musical
- `IRecommendationService` - Recomendaciones
- `ICacheService` - Caching

## 🎵 Ejemplos de Uso

### Buscar Canciones

```python
POST /music/search
{
  "query": "Bohemian Rhapsody Queen",
  "limit": 5
}
```

### Analizar Canción

```python
POST /music/analyze
{
  "track_id": "4uLU6hMCjMI75M1A2tKUQC",
  "include_coaching": true
}
```

### Obtener Recomendaciones

```python
GET /music/recommendations/4uLU6hMCjMI75M1A2tKUQC
```

## 🧪 Testing

### Mockear Servicios

```python
from unittest.mock import Mock
from domain.interfaces.spotify import ISpotifyService

mock_spotify = Mock(spec=ISpotifyService)
mock_spotify.get_track.return_value = {...}
```

### Test Use Case

```python
@pytest.mark.asyncio
async def test_analyze_track():
    use_case = AnalyzeTrackUseCase(
        spotify_service=mock_spotify,
        track_repository=mock_repo,
        analysis_service=mock_analyzer
    )
    result = await use_case.execute("track_id")
    assert result.track_id == "track_id"
```

## 🚨 Troubleshooting

### Servicio no encontrado
```python
# Verificar que está registrado en di_setup.py
# Verificar que setup_dependencies() fue llamado
```

### Cache no funciona
```python
# Verificar que cache_service está registrado
# Verificar que se pasa al repositorio
```

### Spotify API no responde
```bash
# Verificar credenciales en .env
# Verificar que la app de Spotify está activa
# Verificar rate limits
```

### Async no funciona
```python
# Verificar que adaptadores usan _run_sync
# Verificar ThreadPoolExecutor configurado
```

## 📖 Documentación Completa

- `START.md` - Inicio rápido
- `INSTALLATION_GUIDE.md` - Instalación detallada
- `ARCHITECTURE_QUICK_START.md` - Arquitectura
- `API_QUICK_START.md` - Uso de API
- `ARCHITECTURE_MIGRATION_SUMMARY.md` - Resumen arquitectura
- `REFACTORING_FINAL_STATUS.md` - Estado de refactorización

## 🔗 Enlaces Útiles

- **API Docs**: http://localhost:8010/docs
- **Health Check**: http://localhost:8010/health
- **ReDoc**: http://localhost:8010/redoc
- **Spotify Developer**: https://developer.spotify.com/

---

**Última actualización**: 2025  
**Versión**: 2.21.0







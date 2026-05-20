# Architecture - Community Manager AI

## 🏗️ Arquitectura del Sistema

### Estructura General

```
community_manager_ai/
├── core/                    # Lógica de negocio principal
│   ├── community_manager.py    # Orquestador principal
│   ├── scheduler.py            # Programador de publicaciones
│   └── calendar.py             # Calendario de contenido
│
├── services/                # Servicios especializados
│   ├── meme_manager.py         # Gestión de memes
│   ├── social_media_connector.py  # Conexiones a redes sociales
│   ├── content_generator.py      # Generación de contenido
│   ├── analytics_service.py       # Analytics y métricas
│   ├── template_manager.py        # Gestión de plantillas
│   └── notification_service.py    # Sistema de notificaciones
│
├── integrations/            # Integraciones con plataformas
│   ├── base_platform.py         # Clase base para plataformas
│   ├── factory.py               # Factory de plataformas
│   ├── facebook.py              # Integración Facebook
│   ├── instagram.py             # Integración Instagram
│   ├── twitter.py               # Integración Twitter/X
│   ├── linkedin.py              # Integración LinkedIn
│   ├── tiktok.py                # Integración TikTok
│   └── youtube.py               # Integración YouTube
│
├── api/                     # API REST
│   ├── app.py                   # Aplicación FastAPI
│   └── routes/                  # Endpoints
│       ├── posts.py              # Endpoints de posts
│       ├── memes.py              # Endpoints de memes
│       ├── calendar.py           # Endpoints de calendario
│       ├── platforms.py          # Endpoints de plataformas
│       ├── analytics.py          # Endpoints de analytics
│       └── templates.py          # Endpoints de plantillas
│
├── scripts/                 # Scripts de automatización
│   ├── auto_post.py             # Auto-poster
│   ├── content_analyzer.py      # Analizador de contenido
│   └── engagement_tracker.py     # Rastreador de engagement
│
├── config/                  # Configuración
│   └── settings.py             # Configuración centralizada
│
├── utils/                   # Utilidades
│   ├── validators.py            # Validadores
│   ├── helpers.py               # Funciones auxiliares
│   ├── rate_limiter.py           # Limitador de tasa
│   └── content_optimizer.py     # Optimizador de contenido
│
└── tests/                   # Tests
    ├── test_scheduler.py        # Tests del scheduler
    └── test_validators.py       # Tests de validadores
```

## 🔄 Flujo de Datos

### 1. Programación de Post

```
Usuario → API → CommunityManager → Scheduler → Calendar
                                      ↓
                                   Post almacenado
```

### 2. Publicación Automática

```
Auto-Poster → Scheduler (posts pendientes) → SocialMediaConnector
                                                    ↓
                                              Plataformas Sociales
```

### 3. Analytics

```
Plataformas → AnalyticsService → Métricas almacenadas
                                    ↓
                              Reportes y tendencias
```

## 🎯 Principios de Diseño

### 1. Separación de Responsabilidades

- **Core**: Lógica de negocio principal
- **Services**: Servicios especializados y reutilizables
- **Integrations**: Abstracción de APIs externas
- **API**: Capa de presentación

### 2. Patrón Factory

Las plataformas sociales se crean usando el patrón Factory:

```python
platform = get_platform_handler("facebook")
platform.connect(credentials)
```

### 3. Inyección de Dependencias

Los servicios se inyectan como dependencias en FastAPI:

```python
def get_community_manager():
    return CommunityManager()

@router.post("/")
async def create_post(manager = Depends(get_community_manager)):
    ...
```

### 4. Validación en Múltiples Capas

- Validación en la capa de API (Pydantic)
- Validación en la capa de negocio (validators)
- Validación en integraciones (plataformas)

## 🔌 Integraciones

### Patrón de Integración

Todas las plataformas implementan la interfaz `SocialPlatform`:

```python
class SocialPlatform(ABC):
    @abstractmethod
    def connect(credentials) -> bool
    
    @abstractmethod
    def publish(content, media_paths) -> Dict
    
    @abstractmethod
    def get_analytics(post_id) -> Dict
```

### Plataformas Soportadas

- Facebook (Graph API v18.0)
- Instagram (Graph API v18.0)
- Twitter/X (API v2)
- LinkedIn (API v2)
- TikTok (API v1.3)
- YouTube (Data API v3)

## 📊 Servicios Principales

### AnalyticsService

- Registro de métricas de engagement
- Cálculo de engagement rates
- Tendencias y reportes
- Posts con mejor performance

### TemplateManager

- Creación y gestión de plantillas
- Variables en plantillas
- Renderizado dinámico
- Búsqueda y categorización

### NotificationService

- Sistema de notificaciones por tipo
- Handlers personalizables
- Historial de notificaciones
- Limpieza automática

## 🛡️ Seguridad y Validación

### Validaciones Implementadas

1. **Validación de Plataformas**: Verificar plataformas soportadas
2. **Validación de Contenido**: Longitud según plataforma
3. **Validación de Fechas**: Fechas programadas en el futuro
4. **Validación de Medios**: Verificar existencia y formato
5. **Sanitización**: Limpieza de contenido

### Rate Limiting

Sistema de rate limiting para proteger APIs:

```python
rate_limiter = RateLimiter(requests_per_minute=60)
if rate_limiter.is_allowed("facebook"):
    # Publicar
```

## 🧪 Testing

### Estructura de Tests

- **Unit Tests**: Tests de componentes individuales
- **Integration Tests**: Tests de integración entre componentes
- **API Tests**: Tests de endpoints (pendiente)

### Ejecutar Tests

```bash
pytest tests/
```

## 🚀 Escalabilidad

### Mejoras Futuras

1. **Base de Datos**: Persistencia en PostgreSQL/MongoDB
2. **Cache**: Redis para cache de métricas
3. **Queue**: Celery para tareas asíncronas
4. **WebSockets**: Notificaciones en tiempo real
5. **Microservicios**: Separar en servicios independientes

## 📈 Monitoreo

### Logging

Sistema de logging estructurado:

```python
logger.info("Post programado: {post_id}")
logger.error("Error publicando: {error}")
```

### Métricas

- Tiempo de respuesta de APIs
- Tasa de éxito de publicaciones
- Engagement rates por plataforma
- Uso de recursos





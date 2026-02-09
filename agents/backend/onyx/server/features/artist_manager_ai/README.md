# Artist Manager AI

Sistema de IA para gestión integral de artistas que ayuda a cumplir rutinas, calendarios, protocolos y proporciona recomendaciones de vestimenta.

## Características

- 📅 **Gestión de Calendarios**: Organización completa de eventos y compromisos
- 🔄 **Gestión de Rutinas**: Seguimiento de rutinas diarias y semanales
- 📋 **Protocolos de Comportamiento**: Verificación de cumplimiento de protocolos
- 👔 **Gestión de Vestimenta**: Recomendaciones inteligentes de outfit basadas en eventos
- 🤖 **IA con OpenRouter**: Generación inteligente de recomendaciones y resúmenes
- 💾 **Persistencia de Datos**: Base de datos SQLite para guardar información
- 🔔 **Sistema de Notificaciones**: Recordatorios automáticos de eventos y rutinas
- 📊 **Analytics y Métricas**: Seguimiento de estadísticas y rendimiento
- ⚡ **Cache Inteligente**: Optimización de rendimiento con sistema de cache
- ✅ **Validación Robusta**: Validación de datos y manejo de errores mejorado
- 🧠 **Machine Learning**: Predicciones inteligentes de duración y completación
- 🔍 **Búsqueda Avanzada**: Búsqueda difusa con scoring de relevancia
- 🚨 **Sistema de Alertas**: Detección automática de conflictos y problemas
- 🔄 **Sincronización Automática**: Integración con Google Calendar y Outlook
- 📡 **Webhooks**: Notificaciones externas en tiempo real
- 📦 **Backup y Restore**: Sistema completo de respaldo
- 📝 **Plantillas**: Creación rápida desde plantillas predefinidas
- 📊 **Reportes**: Generación de reportes de actividad y cumplimiento

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

Crear archivo `.env`:

```env
OPENROUTER_API_KEY=tu_api_key_aqui
OPENROUTER_MODEL=anthropic/claude-3-haiku
LOG_LEVEL=INFO
```

## Estructura del Proyecto

```
artist_manager_ai/
├── core/                    # Módulos principales
│   ├── artist_manager.py    # Gestor principal
│   ├── calendar_manager.py  # Gestión de calendarios
│   ├── routine_manager.py   # Gestión de rutinas
│   ├── protocol_manager.py # Gestión de protocolos
│   └── wardrobe_manager.py # Gestión de vestimenta
├── infrastructure/          # Infraestructura
│   └── openrouter_client.py # Cliente OpenRouter
├── api/                     # API REST
│   └── routes/              # Rutas de la API
├── mcp_server/              # Servidor MCP
├── config/                  # Configuración
└── requirements.txt         # Dependencias
```

## Uso

### API REST

```python
from fastapi import FastAPI
from artist_manager_ai.api.routes import router

app = FastAPI()
app.include_router(router)
```

### Uso Directo

```python
from artist_manager_ai import ArtistManager
from artist_manager_ai.core.calendar_manager import CalendarEvent, EventType
from datetime import datetime, timedelta

async with ArtistManager(
    artist_id="artist_123",
    openrouter_api_key="...",
    enable_persistence=True,  # Habilitar persistencia en BD
    enable_notifications=True,  # Habilitar notificaciones
    enable_analytics=True  # Habilitar analytics
) as manager:
    # Crear evento con recordatorios automáticos
    event = CalendarEvent(
        id="event_001",
        title="Concierto",
        description="Concierto principal",
        event_type=EventType.CONCERT,
        start_time=datetime.now() + timedelta(days=3),
        end_time=datetime.now() + timedelta(days=3, hours=3)
    )
    manager.create_event_with_reminders(event, reminder_minutes=[60, 30, 15])
    
    # Obtener dashboard
    dashboard = manager.get_dashboard_data()
    
    # Generar resumen diario (con cache)
    summary = await manager.generate_daily_summary()
    
    # Obtener recomendación de vestimenta (mejorada con IA)
    recommendation = await manager.generate_wardrobe_recommendation(event_id="event_001")
    
    # Verificar cumplimiento de protocolos (con IA mejorada)
    compliance = await manager.check_protocol_compliance(event_id="event_001")
    
    # Obtener estadísticas
    stats = manager.get_statistics(days=30)
    
    # Los datos se guardan automáticamente en BD al cerrar
```

## Endpoints API

### Dashboard
- `GET /artist-manager/dashboard/{artist_id}` - Obtener dashboard
- `GET /artist-manager/dashboard/{artist_id}/daily-summary` - Resumen diario con IA

### Calendario
- `POST /artist-manager/calendar/{artist_id}/events` - Crear evento
- `GET /artist-manager/calendar/{artist_id}/events` - Listar eventos
- `GET /artist-manager/calendar/{artist_id}/events/{event_id}` - Obtener evento
- `PUT /artist-manager/calendar/{artist_id}/events/{event_id}` - Actualizar evento
- `DELETE /artist-manager/calendar/{artist_id}/events/{event_id}` - Eliminar evento
- `GET /artist-manager/calendar/{artist_id}/events/{event_id}/wardrobe-recommendation` - Recomendación de vestimenta

### Rutinas
- `POST /artist-manager/routines/{artist_id}/tasks` - Crear rutina
- `GET /artist-manager/routines/{artist_id}/tasks` - Listar rutinas
- `POST /artist-manager/routines/{artist_id}/tasks/{task_id}/complete` - Completar rutina
- `GET /artist-manager/routines/{artist_id}/pending` - Rutinas pendientes

### Protocolos
- `POST /artist-manager/protocols/{artist_id}` - Crear protocolo
- `GET /artist-manager/protocols/{artist_id}` - Listar protocolos
- `POST /artist-manager/protocols/{artist_id}/events/{event_id}/check-compliance` - Verificar cumplimiento

### Vestimenta
- `POST /artist-manager/wardrobe/{artist_id}/items` - Agregar item
- `GET /artist-manager/wardrobe/{artist_id}/items` - Listar items
- `POST /artist-manager/wardrobe/{artist_id}/outfits` - Crear outfit
- `GET /artist-manager/wardrobe/{artist_id}/outfits` - Listar outfits

## Funcionalidades de IA

### Resumen Diario
Genera un resumen inteligente del día con:
- Resumen de eventos programados
- Recordatorios de rutinas importantes
- Recomendaciones generales
- Motivación positiva
- **Cache automático** para optimizar rendimiento

### Recomendaciones de Vestimenta
Analiza el evento y genera recomendaciones basadas en:
- Tipo de evento
- Protocolos aplicables
- Items disponibles en el guardarropa
- Consideraciones del clima
- **Prompts mejorados** para mejores resultados
- **Parsing robusto** de respuestas de IA

### Verificación de Protocolos
Verifica automáticamente el cumplimiento de protocolos usando IA:
- Análisis de protocolos aplicables
- Detección de violaciones
- Recomendaciones de mejora
- **Auditoría detallada** por protocolo

## Nuevas Funcionalidades

### Sistema de Notificaciones
- Recordatorios automáticos de eventos
- Alertas de rutinas pendientes
- Notificaciones de protocolos
- Sistema de prioridades (low, normal, high, urgent)

### Persistencia de Datos
- Base de datos SQLite integrada
- Guardado automático de eventos y rutinas
- Carga automática al inicializar
- Índices optimizados para rendimiento

### Analytics y Métricas
- Seguimiento de métricas personalizadas
- Estadísticas por artista
- Promedios y sumas de métricas
- Análisis de tendencias

### Cache Inteligente
- Cache automático de resúmenes diarios
- TTL configurable
- Limpieza automática de entradas expiradas
- Estadísticas de uso de cache

### Validación Mejorada
- Validación de IDs de artista
- Validación de rangos de tiempo
- Validación de prioridades
- Validación de URLs y emails

## Modelos de Datos

### CalendarEvent
- `id`: ID único
- `title`: Título del evento
- `description`: Descripción
- `event_type`: Tipo (concert, interview, photoshoot, etc.)
- `start_time`: Hora de inicio
- `end_time`: Hora de fin
- `location`: Ubicación
- `protocol_requirements`: Requisitos de protocolo
- `wardrobe_requirements`: Requisitos de vestimenta

### RoutineTask
- `id`: ID único
- `title`: Título de la rutina
- `description`: Descripción
- `routine_type`: Tipo (morning, afternoon, evening, etc.)
- `scheduled_time`: Hora programada
- `duration_minutes`: Duración
- `priority`: Prioridad (1-10)
- `days_of_week`: Días de la semana

### Protocol
- `id`: ID único
- `title`: Título del protocolo
- `description`: Descripción
- `category`: Categoría (social_media, interview, etc.)
- `priority`: Prioridad (critical, high, medium, low)
- `rules`: Lista de reglas
- `do_s`: Cosas a hacer
- `dont_s`: Cosas a evitar

### WardrobeItem
- `id`: ID único
- `name`: Nombre del item
- `category`: Categoría (shirt, pants, shoes, etc.)
- `color`: Color
- `dress_codes`: Códigos de vestimenta aplicables
- `season`: Estación

## Licencia

Propietaria - Blatam Academy

## Autor

Blatam Academy


# Imagen Video Enhancer AI

Sistema de mejoramiento de imágenes y videos con arquitectura SAM3, integrado con OpenRouter y TruthGPT. Similar a Krea AI, permite subir imágenes/videos y mejorarlos con inteligencia artificial.

## Características

- ✅ Arquitectura SAM3 para procesamiento paralelo y continuo
- ✅ Integración con OpenRouter para LLM de alta calidad
- ✅ Integración con TruthGPT para optimización avanzada
- ✅ **API REST completa con FastAPI** para subir archivos
- ✅ **Soporte para vision models** - análisis real de imágenes
- ✅ **Validación de archivos** - imágenes y videos
- ✅ Operación continua 24/7
- ✅ Ejecución paralela de tareas
- ✅ Gestión automática de tareas con cola de prioridades
- ✅ Servicios de mejoramiento:
  - Mejora de calidad de imágenes (con análisis visual)
  - Mejora de calidad de videos
  - Upscaling inteligente
  - Reducción de ruido
  - Mejora de colores y contraste
  - Restauración de imágenes antiguas

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

Configura las variables de entorno:

```bash
export OPENROUTER_API_KEY="tu-api-key"
export TRUTHGPT_ENDPOINT="opcional-endpoint"  # Opcional
```

## Uso Básico

### Uso con Python API

```python
import asyncio
from imagen_video_enhancer_ai import EnhancerAgent, EnhancerConfig

async def main():
    # Crear configuración
    config = EnhancerConfig()
    
    # Crear agente
    agent = EnhancerAgent(config=config)
    
    # Iniciar agente (modo 24/7)
    # await agent.start()  # En producción
    
    # O usar métodos directos
    task_id = await agent.enhance_image(
        file_path="path/to/image.jpg",
        enhancement_type="general",
        options={"quality": "high"}
    )
    
    # Esperar resultado
    import time
    while True:
        status = await agent.get_task_status(task_id)
        if status["status"] == "completed":
            result = await agent.get_task_result(task_id)
            print(result)
            break
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
```

### Uso con API REST

Inicia el servidor API:

```bash
uvicorn imagen_video_enhancer_ai.api.enhancer_api:app --host 0.0.0.0 --port 8000
```

Luego puedes usar los endpoints:

**Subir y mejorar imagen:**
```bash
curl -X POST "http://localhost:8000/upload-image" \
  -F "file=@image.jpg" \
  -F "enhancement_type=general" \
  -F "priority=5"
```

**Mejorar imagen existente:**
```bash
curl -X POST "http://localhost:8000/enhance-image" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/image.jpg",
    "enhancement_type": "general",
    "options": {"quality": "high"},
    "priority": 5
  }'
```

**Consultar estado de tarea:**
```bash
curl "http://localhost:8000/task/{task_id}/status"
```

**Obtener resultado:**
```bash
curl "http://localhost:8000/task/{task_id}/result"
```

**Procesar múltiples archivos en batch:**
```bash
curl -X POST "http://localhost:8000/batch-process" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "file_path": "/path/to/image1.jpg",
        "service_type": "enhance_image",
        "enhancement_type": "general"
      },
      {
        "file_path": "/path/to/image2.jpg",
        "service_type": "upscale",
        "options": {"scale_factor": 2}
      }
    ]
  }'
```

**Registrar webhook:**
```bash
curl -X POST "http://localhost:8000/webhooks/register" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/webhook",
    "events": ["task.completed", "task.failed"],
    "secret": "your-secret-key"
  }'
```

**Ver estadísticas:**
```bash
curl "http://localhost:8000/stats"
```

La documentación interactiva está disponible en: `http://localhost:8000/docs`

## Servicios Disponibles

### Mejora de Imágenes
```python
task_id = await agent.enhance_image(
    file_path="image.jpg",
    enhancement_type="general",  # general, sharpness, colors, etc.
    options={"quality": "high"}
)
```

### Mejora de Videos
```python
task_id = await agent.enhance_video(
    file_path="video.mp4",
    enhancement_type="general",
    options={"fps": 60}
)
```

### Upscaling
```python
task_id = await agent.upscale(
    file_path="image.jpg",
    scale_factor=2,  # 2x, 4x, etc.
    options={"method": "ai"}
)
```

### Reducción de Ruido
```python
task_id = await agent.denoise(
    file_path="image.jpg",
    noise_level="medium",  # low, medium, high
    options={"preserve_details": True}
)
```

### Restauración
```python
task_id = await agent.restore(
    file_path="old_image.jpg",
    damage_type="scratches",
    options={"preserve_style": True}
)
```

### Corrección de Colores
```python
task_id = await agent.color_correction(
    file_path="image.jpg",
    correction_type="auto",
    options={"vibrance": 1.2}
)
```

## Arquitectura

El sistema sigue la arquitectura SAM3 con base classes y abstracciones comunes:
- **Base Models**: Modelos base con funcionalidad común (`BaseModel`, `TimestampedModel`, `IdentifiedModel`, `StatusModel`)
- **Repository Pattern**: Base repository con operaciones comunes (`BaseRepository`, `RepositoryMixin`)
- **Manager Base**: Base manager con lifecycle y estadísticas (`BaseManager`, `ManagerRegistry`)
- **Base HTTP Client**: Cliente HTTP base con pooling y retries (`BaseHTTPClient`)
- **Config Manager**: Gestión centralizada de configuración (`ConfigManager`)
- **Lifecycle Management**: Sistema de gestión de ciclo de vida (`LifecycleManager`, `LifecycleComponent`)
- **Dependency Injection**: Contenedor de inyección de dependencias (`DependencyContainer`)
- **Component Registry**: Registro de componentes (`ComponentRegistry`)
- **TaskManager**: Gestión de tareas con persistencia (hereda de base classes)
- **ParallelExecutor**: Ejecución paralela de tareas
- **ServiceHandler**: Manejo de diferentes servicios de mejoramiento
- **OpenRouterClient**: Integración con modelos LLM (hereda de `BaseHTTPClient`)
- **TruthGPTClient**: Optimización avanzada
- **Validation Helpers**: Validaciones reutilizables (`ValidationRule`, `ValidationChain`)

## Estructura del Proyecto

```
imagen_video_enhancer_ai/
├── config/          # Configuración
├── core/            # Lógica principal
│   ├── base_models.py      # Modelos base (BaseModel, TimestampedModel, etc.)
│   ├── repository_base.py # Base repository y mixins
│   ├── manager_base.py     # Base manager y registry
│   ├── task_manager.py     # Gestión de tareas
│   ├── enhancer_agent.py   # Agente principal
│   └── services/           # Handlers de servicios
├── infrastructure/  # Clientes OpenRouter y TruthGPT
├── api/             # Endpoints REST con FastAPI
│   ├── routes/      # Rutas organizadas por funcionalidad
│   ├── models.py    # Modelos Pydantic
│   ├── dependencies.py # Dependencias compartidas
│   └── middleware.py # Middleware (CORS, rate limiting)
├── utils/           # Utilidades
│   ├── validation_helpers.py # Helpers de validación
│   ├── formatters.py        # Formateo de datos
│   └── ...          # Otras utilidades
├── tests/           # Tests
├── examples/        # Ejemplos de uso
└── main.py          # Punto de entrada
```

## Endpoints de la API

### Procesamiento Individual
- `POST /upload-image` - Subir y mejorar imagen
- `POST /upload-video` - Subir y mejorar video
- `POST /enhance-image` - Mejorar imagen existente
- `POST /enhance-video` - Mejorar video existente
- `POST /upscale` - Hacer upscaling
- `POST /denoise` - Reducir ruido
- `POST /restore` - Restaurar imagen
- `POST /color-correction` - Corrección de colores

### Procesamiento por Lotes
- `POST /batch-process` - Procesar múltiples archivos en batch

### Consultas
- `GET /task/{task_id}/status` - Estado de tarea
- `GET /task/{task_id}/result` - Resultado de tarea
- `GET /stats` - Estadísticas del agente
- `GET /health` - Health check
- `GET /docs` - Documentación interactiva (Swagger)

### Webhooks
- `POST /webhooks/register` - Registrar webhook
- `DELETE /webhooks/unregister` - Desregistrar webhook

### Análisis
- `POST /analyze` - Analizar archivo (imagen o video)

### Exportación
- `POST /export-results` - Exportar resultados a varios formatos

### Dashboard y Monitoreo
- `GET /dashboard/metrics` - Métricas del dashboard
- `GET /dashboard/health` - Estado de salud del sistema
- `GET /dashboard/trends` - Tendencias de rendimiento
- `GET /memory/usage` - Uso de memoria
- `POST /memory/optimize` - Optimizar memoria

### Autenticación
- `POST /auth/generate-key` - Generar API key
- `GET /auth/keys` - Listar API keys

### Notificaciones
- `POST /notifications/send` - Enviar notificación
- `GET /notifications/stats` - Estadísticas de notificaciones

### Configuración
- `POST /config/validate` - Validar configuración

### Métricas Avanzadas
- `GET /metrics/{metric_name}` - Obtener datos de métrica
- `GET /metrics/{metric_name}/stats` - Estadísticas de métrica
- `GET /metrics/{histogram_name}/percentiles` - Percentiles de histograma

### Eventos
- `GET /events/history` - Historial de eventos

## Mejoras Implementadas

### ✅ API REST Completa
- Endpoints para subir archivos (imágenes y videos)
- Endpoints para todos los servicios de mejoramiento
- Validación automática de archivos
- Manejo de errores robusto

### ✅ Soporte para Vision Models
- Análisis real de imágenes usando modelos de visión
- Integración con OpenRouter vision API
- Análisis automático de calidad y problemas
- Guías detalladas basadas en el contenido real

### ✅ Validación de Archivos
- Validación de formato (extensiones permitidas)
- Validación de tamaño (límites configurables)
- Validación de parámetros (tipos, rangos, etc.)
- Mensajes de error claros y descriptivos

### ✅ Utilidades Mejoradas
- Helpers para manejo de archivos
- Detección automática de tipo de archivo
- Generación de nombres únicos
- Detección de MIME types

### ✅ Batch Processing
- Procesamiento paralelo de múltiples archivos
- Seguimiento de progreso
- Manejo de errores por item
- Agregación de resultados

### ✅ Cache Management
- Sistema de caché inteligente con TTL
- Invalidación automática por modificación de archivos
- Estadísticas de caché (hits, misses, hit rate)
- Limpieza automática de entradas expiradas

### ✅ Rate Limiting
- Token bucket rate limiter
- Límites por cliente (IP)
- Soporte para bursts
- Configuración flexible

### ✅ Estadísticas y Métricas
- Estadísticas del ejecutor paralelo
- Estadísticas de caché
- Métricas de rendimiento
- Endpoint `/stats` para consulta

### ✅ Webhooks
- Notificaciones automáticas de eventos
- Múltiples endpoints configurable
- Firma de seguridad (HMAC)
- Reintentos automáticos
- Eventos: task.created, task.started, task.completed, task.failed, batch.completed

### ✅ Logging Mejorado
- Configuración centralizada
- Niveles configurables
- Soporte para archivos de log
- Formato estructurado

### ✅ Tests
- Tests básicos para componentes principales
- Tests de validadores
- Fixtures para configuración
- Tests de integración

### ✅ Procesamiento de Videos Mejorado
- Análisis de videos con OpenCV
- Detección de problemas de calidad
- Análisis de frames
- Recomendaciones automáticas
- Extracción de frames de muestra

### ✅ Utilidades de Imágenes
- Información detallada de imágenes
- Validación de dimensiones
- Estimación de tiempo de procesamiento
- Análisis de propiedades

### ✅ Manejo de Errores Mejorado
- Excepciones personalizadas
- Formato consistente de errores
- Contexto en errores
- Decoradores para manejo automático

### ✅ Sistema de Retry Automático
- Reintentos automáticos para tareas fallidas
- Estrategias configurables (exponential backoff, fixed delay, etc.)
- Clasificación de errores retryables
- Historial de reintentos
- Estadísticas de retry

### ✅ Exportación de Resultados
- Exportación a múltiples formatos (JSON, Markdown, CSV, HTML)
- Exportación de tareas individuales o todas
- Reportes HTML con formato profesional
- Exportación a CSV para análisis
- Exportación a Markdown para documentación

### ✅ Compresión
- Compresión de resultados JSON
- Compresión de archivos
- Niveles de compresión configurables
- Descompresión automática
- Estadísticas de compresión

### ✅ Dashboard de Monitoreo
- Métricas en tiempo real
- Estado de salud del sistema
- Tendencias de rendimiento
- Historial de métricas
- Score de salud automático
- Análisis de tendencias

### ✅ Optimización de Memoria
- Monitoreo de uso de memoria
- Limpieza automática de caché
- Recolector de basura optimizado
- Recomendaciones de optimización
- Estadísticas de memoria
- Limpieza agresiva opcional

### ✅ Sistema de Plugins
- Arquitectura extensible
- Registro de plugins personalizados
- Carga automática desde directorio
- Validación de plugins
- Ejecución de plugins
- Gestión de plugins (habilitar/deshabilitar)

### ✅ Autenticación y Autorización
- Sistema de API keys
- Permisos granulares
- Expiración de keys
- Revocación de keys
- Validación de permisos
- Integración con FastAPI

### ✅ Sistema de Notificaciones
- Múltiples canales (email, SMS, push, webhook, Slack, Discord)
- Prioridades configurables
- Historial de notificaciones
- Estadísticas de notificaciones
- Handlers personalizables
- Retry automático

### ✅ Validación de Configuración
- Validación avanzada de config
- Validación de paths y permisos
- Validación de entorno
- Recomendaciones automáticas
- Validación de dependencias
- Mensajes de error claros

### ✅ Utilidades Adicionales
- Helpers para operaciones comunes
- Formateo de archivos y duraciones
- Generación de IDs únicos
- Cálculo de hashes de archivos
- Sanitización de nombres de archivo
- Manejo seguro de JSON
- Decoradores de retry y throttle
- Cache de resultados con TTL

### ✅ Logging Mejorado
- Rotación automática de logs
- Configuración flexible
- Niveles configurables
- Soporte para archivos de log
- Formato estructurado
- Limpieza automática de logs antiguos

### ✅ Optimizaciones
- Cache de resultados con TTL
- Throttling de llamadas
- Decoradores de optimización
- Limpieza automática de cache
- Gestión eficiente de memoria

### ✅ Sistema de Métricas Avanzadas
- Time-series metrics
- Contadores y gauges
- Histogramas con percentiles
- Cálculo de tasas
- Filtrado por tags
- Estadísticas agregadas
- Análisis de tendencias

### ✅ Sistema de Eventos
- Event bus pub/sub
- Múltiples tipos de eventos
- Handlers async
- Historial de eventos
- Filtrado de eventos
- Subscripciones wildcard
- Integración automática con el agente

### ✅ Integración Completa
- Métricas integradas en el agente
- Eventos publicados automáticamente
- Webhooks con eventos
- Monitoreo comprehensivo
- Utilidades de integración
- Ejemplos avanzados de uso

### ✅ Sistema de Backup y Recuperación
- Backups completos e incrementales
- Compresión de backups
- Restauración de backups
- Listado de backups
- Eliminación de backups
- Scripts de utilidad

### ✅ Scripts de Utilidad
- `backup_tasks.py` - Backup de tareas y resultados
- `cleanup.py` - Limpieza de archivos antiguos
- `export_stats.py` - Exportación de estadísticas
- `dev_setup.py` - Configuración del entorno de desarrollo
- Scripts configurables
- Soporte para dry-run

### ✅ Herramientas de Desarrollo
- Utilidades de desarrollo (`dev_helpers.py`)
- Decoradores de timing y logging
- Performance profiler
- Error reporter avanzado
- Validadores avanzados
- Tests unitarios adicionales

### ✅ Documentación Completa
- Quick Start Guide
- API Documentation
- Best Practices
- Plugin System
- Deployment Guide
- Development Guide
- Refactoring Guide
- Architecture Guide
- Contributing Guide
- Ejemplos avanzados

### ✅ Utilidades Avanzadas
- Config loader con múltiples fuentes
- Type checker para validación en runtime
- Helpers de testing mejorados
- Soporte para YAML y JSON en configuración
- Merge de configuraciones
- Serialización (JSON, Pickle, Base64)
- Response builder para respuestas consistentes
- Data transformers (dict, list, datetime)
- Error context para mejor tracking
- Utilidades de transformación de datos
- Schema validator para validación de esquemas
- API versioning system
- Formatters para formateo de datos
- Structured logging avanzado
- Documentación helpers para generación de docs
- Health check system para monitoreo de servicios
- Test helpers avanzados para testing
- Sistema de migraciones para cambios de esquema
- Performance profiler para análisis de rendimiento
- Feature flags system para rollouts graduales
- Circuit breaker pattern para resiliencia
- Advanced rate limiter con múltiples estrategias
- Distributed cache con múltiples backends
- Distributed tracing para observability
- Observability system completo para monitoreo
- Sistema de alertas basado en condiciones
- Advanced metrics con agregación y percentiles
- Module system para organización de módulos
- Agent builder pattern para construcción flexible
- Service registry para gestión de servicios
- Middleware system avanzado para procesamiento de requests
- Advanced configuration system con validación
- Advanced event system con routing y filtrado
- Request validator para validación de requests
- Data transformer para transformación de datos
- Advanced serialization system con múltiples formatos
- Infrastructure consolidation con exports centralizados
- Service providers para servicios externos
- Initialization system modular con fases
- Consolidated imports para mejor organización
- Security system con encriptación y hashing
- Audit system para auditoría de acciones
- Advanced throttling con múltiples estrategias
- Queue management con prioridades
- Resource pooling para gestión de recursos
- Strategy system consolidado para retry, cache y validación
- Service configuration consolidado
- Validation manager centralizado
- Context managers para operaciones comunes
- Benchmark system para testing de performance
- Performance optimizer con auto-optimización
- Documentation generator automático
- Dynamic configuration con hot-reloading
- Advanced health checks con dependencias
- Manager registry consolidado
- System integrator para coordinación de componentes
- Error recovery system con múltiples estrategias
- Async utilities para operaciones comunes
- Testing helpers avanzados
- CI/CD helpers para automatización
- Analytics system para análisis de uso
- Reporting system para generación de reportes
- Advanced data validator con schemas
- Registry base pattern para todos los registries
- Executor base pattern para ejecución de operaciones
- Storage base pattern para almacenamiento
- Workflow system para definición y ejecución de workflows
- Pipeline system para procesamiento de datos
- Orchestrator system para orquestación de servicios
- State management para gestión de estado de aplicación
- Advanced cache con múltiples estrategias (LRU, LFU, FIFO, TTL)
- Service base pattern para todos los servicios
- Handler base pattern para todos los handlers
- Processor base pattern para todos los procesadores
- Coordinator para coordinación de componentes
- Integration system para integración de servicios externos
- Data pipeline para transformación de datos
- Serializer avanzado con múltiples formatos
- Structured logging para logging estructurado
- Config builder para construcción de configuración
- Final utilities para utilidades finales
- Agent component system para arquitectura de componentes
- Event handler system para manejo de eventos
- Factory base pattern para creación de objetos
- Middleware base pattern para procesamiento de requests/responses
- Batch operations system para operaciones en lote
- Scheduler system para programación de tareas
- Advanced queue con prioridades y scheduling
- Result aggregator para análisis de resultados
- Performance tuner para optimización automática
- Resource manager para gestión de recursos del sistema
- Route decorators para decoradores comunes de rutas
- Response formatter para formato consistente de respuestas
- Request validator para validación de requests
- Middleware helpers para helpers de middleware
- Route builder para construcción de rutas con builder pattern
- Data transformer para transformación avanzada de datos
- Cache utils para utilidades de caché
- Compression utils para compresión de datos
- Encryption utils para encriptación y hashing
- File utils advanced para operaciones avanzadas de archivos
- Network utils para utilidades de red
- Advanced service base para servicios avanzados
- Execution context para gestión de contexto de ejecución
- Advanced error handler para manejo avanzado de errores
- Test utils para utilidades de testing
- Test fixtures para fixtures avanzados de pytest
- Advanced assertions para aserciones avanzadas
- Test runner para ejecución de tests con reporting
- Client base avanzado para clientes API con funcionalidad común
- Response handler para manejo de respuestas HTTP
- Advanced logging para logging estructurado y performance tracking
- Advanced monitoring para métricas del sistema y health checks
- Config base para gestión de configuración con múltiples fuentes
- Advanced config validator para validación avanzada de configuración
- Code generator system para generación automática de código
- Seeds system para seeding de datos iniciales
- Automatic backup system con scheduling y retención
- Deployment utilities para deployment y environment checks
- Migrations system mejorado con soporte async
- API versioning system con múltiples estrategias y deprecation
- Distributed cache system con múltiples backends y consistencia
- Advanced logging system con rotación, filtrado y formato estructurado
- Advanced testing system con fixtures, mocks y utilidades de testing
- Advanced request validation system con schemas y reglas
- Advanced data transformation system con pipelines y transformadores
- Advanced middleware system para FastAPI con procesamiento avanzado
- Advanced rate limiting system con múltiples estrategias y límites por usuario
- Advanced circuit breaker system con estados avanzados y gestión de fallos
- Telemetry system para recolección y análisis de datos del sistema
- Advanced performance profiler con análisis detallado de rendimiento
- Real-time metrics system para métricas en tiempo real y agregación
- Advanced permissions system con RBAC y permisos granulares
- Advanced encryption system con múltiples algoritmos y gestión de claves
- Security validator system para validación de seguridad y sanitización
- Advanced audit system con tracking detallado y compliance
- Advanced health monitoring system con dependency checks y status aggregation
- Advanced retry system con múltiples estrategias y exponential backoff
- Advanced queue system con prioridades, scheduling y persistence
- Advanced event bus system con pub/sub, filtering y event history
- Advanced cache strategy system con múltiples políticas de evicción y TTL
- Advanced validation system con schemas, rules y custom validators

## Notas

- El sistema analiza imágenes reales usando vision models de OpenRouter
- Proporciona guías detalladas basadas en el análisis visual
- Similar a Krea AI en funcionalidad pero con arquitectura SAM3 y TruthGPT
- Para procesamiento real de imágenes/videos, se requiere integración con herramientas adicionales de procesamiento


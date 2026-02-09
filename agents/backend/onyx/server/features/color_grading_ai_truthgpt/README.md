# Color Grading AI TruthGPT

Sistema completo de color grading automático con arquitectura enterprise, integrado con OpenRouter y TruthGPT. Similar a DaVinci Resolve pero completamente automático.

## 🚀 Características Principales

### Procesamiento
- ✅ Procesamiento de video e imágenes
- ✅ Análisis de color avanzado
- ✅ Color matching desde referencias
- ✅ Análisis de calidad de video

### Gestión
- ✅ Templates predefinidos (cinematic, vintage, modern, etc.)
- ✅ Presets personalizados
- ✅ LUTs profesionales
- ✅ Historial completo
- ✅ Versionado de parámetros
- ✅ Backup y restauración

### Infraestructura
- ✅ Event bus (pub/sub)
- ✅ Security manager
- ✅ Telemetry service
- ✅ Task queue unificada
- ✅ Cloud integration (S3)

### Analytics
- ✅ Métricas y estadísticas
- ✅ Performance monitoring
- ✅ Analytics service
- ✅ Dashboard en tiempo real

### Inteligencia
- ✅ Recommendation engine
- ✅ ML optimizer
- ✅ Optimization engine

### Colaboración
- ✅ Webhooks
- ✅ Notifications (Email, Slack, Discord, Telegram)
- ✅ Collaboration manager
- ✅ Workflow manager

### Resiliencia
- ✅ Circuit breaker
- ✅ Retry manager
- ✅ Load balancer
- ✅ Feature flags

### Control de Tráfico
- ✅ Rate limiter
- ✅ Throttle manager
- ✅ Backpressure manager

### Gestión de Ciclo de Vida
- ✅ Health monitor
- ✅ Graceful shutdown
- ✅ Lifecycle manager

### Compliance y Auditoría
- ✅ Audit logger
- ✅ Compliance manager (GDPR, CCPA, HIPAA, SOC2, ISO27001)

## 📊 Estadísticas del Proyecto

- **Servicios totales**: 63+
- **Categorías**: 11
- **Patrones de diseño**: 8+
- **Componentes base**: 5
- **Utilidades**: 10+
- **Decoradores**: 4

## 🏗️ Arquitectura

Ver [ARCHITECTURE.md](ARCHITECTURE.md) para detalles completos de la arquitectura.

### Estructura Principal
```
core/              # Componentes base y agentes
services/          # 63+ servicios organizados
infrastructure/    # Clientes (OpenRouter, TruthGPT)
api/              # API REST completa
config/           # Configuración
```

### Categorías de Servicios
1. **Processing** (5): Video, Image, Color Analysis, Color Matching, Quality
2. **Management** (7): Templates, Presets, LUTs, Cache, History, Version, Backup
3. **Infrastructure** (5): EventBus, Security, Telemetry, Queue, Cloud
4. **Analytics** (4): Metrics, Performance Monitor, Performance Optimizer, Analytics
5. **Intelligence** (3): Recommendations, ML Optimizer, Optimization Engine
6. **Collaboration** (4): Webhooks, Notifications, Collaboration, Workflow
7. **Resilience** (4): Circuit Breaker, Retry, Load Balancer, Feature Flags
8. **Traffic Control** (3): Rate Limiter, Throttle, Backpressure
9. **Lifecycle** (3): Health Monitor, Graceful Shutdown, Lifecycle Manager
10. **Compliance** (2): Audit Logger, Compliance Manager
11. **Support** (23+): Batch, Comparison, Export, y más...

## 📦 Instalación

```bash
pip install -r requirements.txt
```

## ⚙️ Configuración

### Variables de Entorno

```bash
export OPENROUTER_API_KEY="tu-api-key"
export TRUTHGPT_ENDPOINT="opcional-endpoint"  # Opcional
export FFMPEG_PATH="/path/to/ffmpeg"  # Opcional, defaults to "ffmpeg"
```

### Configuración Programática

```python
from color_grading_ai_truthgpt import ColorGradingConfig

config = ColorGradingConfig()
config.openrouter.api_key = "tu-api-key"
config.max_parallel_tasks = 10
config.enable_cache = True
config.cache_ttl = 3600
```

## 🎯 Uso Básico

### Python API

```python
import asyncio
from color_grading_ai_truthgpt import UnifiedColorGradingAgent, ColorGradingConfig

async def main():
    # Crear configuración
    config = ColorGradingConfig()
    
    # Crear agente unificado (recomendado)
    agent = UnifiedColorGradingAgent(config=config)
    
    # Aplicar template a video
    result = await agent.grade_video(
        video_path="input.mp4",
        template_name="cinematic"
    )
    print(f"Video procesado: {result['output_path']}")
    
    # Aplicar color matching desde imagen de referencia
    result = await agent.grade_image(
        image_path="input.jpg",
        reference_image="reference.jpg"
    )
    print(f"Imagen procesada: {result['output_path']}")
    
    # Analizar color
    analysis = await agent.analyze_media("input.mp4")
    print(f"Análisis: {analysis}")

asyncio.run(main())
```

### REST API

```bash
# Iniciar servidor
uvicorn color_grading_ai_truthgpt.api.color_grading_api:app --reload

# Aplicar color grading
curl -X POST "http://localhost:8000/api/v1/grade/video" \
  -F "file=@input.mp4" \
  -F "template_name=cinematic"

# Listar templates
curl "http://localhost:8000/api/v1/templates"
```

## 📚 Documentación

Ver [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) para índice completo de documentación.

### Documentos Principales
- [ARCHITECTURE.md](ARCHITECTURE.md) - Arquitectura completa
- [REFACTORING_FINAL_COMPLETE.md](REFACTORING_FINAL_COMPLETE.md) - Refactorización final
- [RESILIENCE_PATTERNS.md](RESILIENCE_PATTERNS.md) - Patrones de resiliencia
- [TRAFFIC_CONTROL.md](TRAFFIC_CONTROL.md) - Control de tráfico
- [LIFECYCLE_MANAGEMENT.md](LIFECYCLE_MANAGEMENT.md) - Gestión de ciclo de vida
- [COMPLIANCE_AUDIT.md](COMPLIANCE_AUDIT.md) - Compliance y auditoría
- [DEPLOYMENT.md](DEPLOYMENT.md) - Guías de deployment

## 🔧 Componentes Clave

### Agentes
- **UnifiedColorGradingAgent** ⭐ (Recomendado)
- ColorGradingAgent (Original, compatible)
- RefactoredColorGradingAgent (Refactorizado, compatible)

### Factories
- **RefactoredServiceFactory** ⭐ (Recomendado)
- ServiceFactory (Original, compatible)

### Componentes Base
- BaseService - Base para todos los servicios
- FileManagerBase - Base para managers de archivos
- ConfigManager - Gestión de configuración
- ServiceGroups - Agrupación lógica
- ServiceAccessor - Acceso unificado

## 🎨 Ejemplos

### Template Predefinido
```python
result = await agent.grade_video(
    video_path="input.mp4",
    template_name="vintage"
)
```

### Color Matching
```python
result = await agent.grade_video(
    video_path="input.mp4",
    reference_video="reference.mp4"
)
```

### Descripción de Texto
```python
result = await agent.grade_image(
    image_path="input.jpg",
    description="warm sunset colors with high contrast"
)
```

### Parámetros Personalizados
```python
result = await agent.grade_video(
    video_path="input.mp4",
    color_params={
        "brightness": 1.1,
        "contrast": 1.2,
        "saturation": 1.15,
        "temperature": 5500
    }
)
```

## 🚀 Deployment

Ver [DEPLOYMENT.md](DEPLOYMENT.md) para guías completas.

### Docker
```bash
docker-compose up -d
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

## 📈 Características Enterprise

- ✅ **Resiliencia**: Circuit breaker, retry, load balancing
- ✅ **Observabilidad**: Health monitoring, performance tracking, telemetry
- ✅ **Seguridad**: Security manager, input validation, threat detection
- ✅ **Escalabilidad**: Load balancing, resource pooling, cloud integration
- ✅ **Compliance**: GDPR, CCPA, HIPAA, SOC2, ISO27001
- ✅ **Auditoría**: Audit logging completo

## 🤝 Contribuir

El proyecto está completamente funcional y listo para producción. Para contribuir:

1. Revisar [ARCHITECTURE.md](ARCHITECTURE.md)
2. Seguir los patrones establecidos
3. Usar los componentes base disponibles
4. Agregar tests para nuevas funcionalidades

## 📄 Licencia

[Especificar licencia]

## 🙏 Agradecimientos

- OpenRouter por la integración LLM
- TruthGPT por la optimización avanzada
- Comunidad open source

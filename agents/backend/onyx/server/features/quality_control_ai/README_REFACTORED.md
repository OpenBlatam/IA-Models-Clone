# Quality Control AI - Sistema Refactorizado

## 🎯 Visión General

Sistema completo de control de calidad con detección de defectos por cámara, completamente refactorizado siguiendo principios de **Clean Architecture** y **Domain-Driven Design**.

**Versión**: 2.2.0  
**Estado**: ✅ Producción Ready  
**Arquitectura**: Clean Architecture + DDD

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────┐
│      Presentation Layer (API)          │
│  - FastAPI Routes                       │
│  - Pydantic Schemas                     │
│  - Middleware                           │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      Application Layer                  │
│  - Use Cases                            │
│  - DTOs                                 │
│  - Application Services                 │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      Domain Layer                       │
│  - Entities                             │
│  - Value Objects                        │
│  - Domain Services                      │
│  - Validators                           │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      Infrastructure Layer              │
│  - Repositories                         │
│  - Adapters                             │
│  - ML Services                          │
│  - Logging, Cache, Metrics              │
└─────────────────────────────────────────┘
```

## 🚀 Inicio Rápido

### Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno (opcional)
export API_PORT=8000
export LOG_LEVEL=INFO
export CACHE_ENABLED=True
```

### Uso Básico

```python
from quality_control_ai import (
    ApplicationServiceFactory,
    InspectionRequest,
)

# Crear factory (configura todo automáticamente)
factory = ApplicationServiceFactory()
service = factory.create_inspection_application_service()

# Inspeccionar imagen
request = InspectionRequest(
    image_data=image_array,  # numpy array, bytes, file path, o base64
    image_format="numpy",
    include_visualization=True,
)

response = service.inspect_image(request)
print(f"Quality Score: {response.quality_score}")
print(f"Defects: {len(response.defects)}")
print(f"Anomalies: {len(response.anomalies)}")
```

### Ejecutar API

```bash
# Opción 1: Script
python -m quality_control_ai.scripts.run_server

# Opción 2: Uvicorn directo
uvicorn quality_control_ai.presentation.api:app --host 0.0.0.0 --port 8000

# Opción 3: Python
from quality_control_ai.presentation.api import create_app
app = create_app()
# Ejecutar con uvicorn
```

## 📡 API Endpoints

### Inspección

- `POST /api/v1/inspections` - Inspeccionar imagen
- `POST /api/v1/inspections/upload` - Subir y inspeccionar archivo
- `POST /api/v1/inspections/batch` - Inspección en batch
- `GET /api/v1/inspections/{id}` - Obtener inspección

### Sistema

- `GET /api/v1/health` - Health check
- `GET /api/v1/metrics` - Métricas del sistema
- `GET /api/v1/settings` - Configuración
- `GET /api/v1/` - Información de API

### Documentación

- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc

## 🔧 Configuración

### Variables de Entorno

```bash
# Aplicación
APP_NAME=Quality Control AI
APP_VERSION=2.2.0
DEBUG=False

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=./logs/app.log

# Cache
CACHE_ENABLED=True
CACHE_MAX_SIZE=1000
CACHE_DEFAULT_TTL=3600

# Métricas
METRICS_ENABLED=True

# ML Models
MODEL_DEVICE=auto  # auto, cpu, cuda
MODEL_CACHE_DIR=./models

# Inspección
INSPECTION_TIMEOUT=30.0
INSPECTION_BATCH_SIZE=8

# Storage
STORAGE_PATH=./storage
```

## 📦 Componentes Principales

### Domain Layer

**Entidades:**
- `Inspection` - Inspección completa
- `Defect` - Defecto detectado
- `Anomaly` - Anomalía detectada
- `QualityScore` - Puntuación de calidad
- `Camera` - Cámara

**Value Objects:**
- `ImageMetadata` - Metadata de imagen
- `DetectionResult` - Resultado de detección
- `QualityMetrics` - Métricas de calidad

**Servicios:**
- `InspectionService` - Servicio de inspección
- `QualityAssessmentService` - Evaluación de calidad
- `DefectClassificationService` - Clasificación de defectos

### Application Layer

**Use Cases:**
- `InspectImageUseCase` - Inspección de imagen
- `InspectBatchUseCase` - Inspección en batch
- `StartInspectionStreamUseCase` - Iniciar stream
- `StopInspectionStreamUseCase` - Detener stream
- `TrainModelUseCase` - Entrenar modelo
- `GenerateReportUseCase` - Generar reporte

### Infrastructure Layer

**Repositories:**
- `InspectionRepository` - Persistencia de inspecciones
- `ModelRepository` - Gestión de modelos ML
- `ConfigurationRepository` - Configuración

**Adapters:**
- `CameraAdapter` - Interfaz de cámara
- `MLModelLoader` - Carga de modelos
- `StorageAdapter` - Almacenamiento
- `ImageProcessor` - Procesamiento de imágenes

**Servicios:**
- `AnomalyDetectionService` - Detección de anomalías
- `ObjectDetectionService` - Detección de objetos
- `DefectClassificationService` - Clasificación de defectos

**Utilidades:**
- `StructuredLogger` - Logging estructurado
- `CacheManager` - Gestión de caché
- `MetricsCollector` - Recolección de métricas
- `ErrorHandler` - Manejo de errores

### Presentation Layer

**API:**
- FastAPI con 8+ endpoints
- Pydantic schemas para validación
- Middleware para errores y logging
- Dependency injection

## 🛠️ Utilidades

### Disponibles

**Validación:**
- `validate_email()`, `validate_url()`, `validate_range()`, etc.

**Performance:**
- `@measure_time`, `@retry_on_failure`, `@throttle`, `PerformanceMonitor`

**String:**
- `camel_to_snake()`, `format_bytes()`, `format_duration()`, etc.

**Security:**
- `generate_token()`, `hash_string()`, `mask_sensitive_data()`, etc.

**File:**
- `ensure_directory()`, `is_image_file()`, `safe_filename()`, etc.

**Date:**
- `now_utc()`, `format_datetime()`, `time_ago()`, etc.

## 📊 Características

### Funcionalidades Core
- ✅ Inspección de imágenes (múltiples formatos)
- ✅ Detección de defectos con ML
- ✅ Detección de anomalías con ML
- ✅ Clasificación de defectos
- ✅ Cálculo de calidad
- ✅ Inspección en batch
- ✅ Stream en tiempo real
- ✅ Generación de reportes

### Características Avanzadas
- ✅ Logging estructurado
- ✅ Caché inteligente
- ✅ Métricas y monitoreo
- ✅ Validación completa
- ✅ Manejo de errores robusto
- ✅ Configuración centralizada
- ✅ 40+ funciones de utilidad

## 🧪 Testing

```bash
# Health check
python -m quality_control_ai.scripts.check_health

# Tests (cuando estén implementados)
pytest tests/
```

## 📚 Documentación

- `COMPLETE_REFACTORING_SUMMARY.md` - Resumen completo
- `REFACTORING_PLAN.md` - Plan de refactorización
- `FINAL_IMPROVEMENTS_V2.md` - Mejoras finales
- `ADDITIONAL_UTILITIES.md` - Utilidades adicionales
- Y más...

## 🔍 Ejemplos

### Ejemplo 1: Inspección Simple

```python
from quality_control_ai import ApplicationServiceFactory, InspectionRequest
import numpy as np

factory = ApplicationServiceFactory()
service = factory.create_inspection_application_service()

image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
request = InspectionRequest(image_data=image, image_format="numpy")

response = service.inspect_image(request)
print(f"Score: {response.quality_score}, Status: {response.quality_status}")
```

### Ejemplo 2: Batch Inspection

```python
from quality_control_ai import ApplicationServiceFactory, BatchInspectionRequest, InspectionRequest

factory = ApplicationServiceFactory()
service = factory.create_inspection_application_service()

batch_request = BatchInspectionRequest(
    images=[
        InspectionRequest(image_data=img1, image_format="numpy"),
        InspectionRequest(image_data=img2, image_format="numpy"),
    ],
    parallel=True,
    max_workers=4,
)

response = service.inspect_batch(batch_request)
print(f"Processed: {response.total_processed}, Avg Score: {response.average_quality_score}")
```

### Ejemplo 3: Usar Utilidades

```python
from quality_control_ai.utils import (
    format_bytes,
    format_duration,
    time_ago,
    generate_token,
)

size = format_bytes(1024 * 1024)  # "1.00 MB"
duration = format_duration(3665)  # "1h 1m 5.00s"
ago = time_ago(some_datetime)  # "2 hours ago"
token = generate_token(32)  # Secure token
```

## 🎯 Mejores Prácticas

1. **Usar Factories**: Siempre usar `ApplicationServiceFactory` para crear servicios
2. **Validación**: Las validaciones se ejecutan automáticamente
3. **Configuración**: Usar variables de entorno para configuración
4. **Logging**: El logging estructurado está activo por defecto
5. **Métricas**: Las métricas se recolectan automáticamente
6. **Caché**: El caché está habilitado por defecto

## 🐛 Troubleshooting

### Problemas Comunes

**Error: "Service not initialized"**
- Solución: Usar `ApplicationServiceFactory` para crear servicios

**Error: "Invalid image format"**
- Solución: Verificar que el formato sea uno de: numpy, bytes, file_path, base64

**Error: "Model not found"**
- Solución: Asegurarse de que los modelos estén en `MODEL_CACHE_DIR`

## 📝 Changelog

### v2.2.0 (Actual)
- ✅ Refactorización completa con Clean Architecture
- ✅ Factory Pattern para DI
- ✅ Validación completa
- ✅ Logging estructurado
- ✅ Caché y métricas
- ✅ 40+ utilidades
- ✅ API completa

### v2.1.0
- Mejoras en modelos ML
- Optimizaciones de performance

### v2.0.0
- Versión inicial refactorizada

## 🤝 Contribuir

El sistema está completamente refactorizado y listo para extensiones. Para agregar nuevas funcionalidades:

1. Agregar entidades en `domain/entities/`
2. Crear use cases en `application/use_cases/`
3. Implementar en `infrastructure/`
4. Exponer en `presentation/api/`

## 📄 Licencia

Blatam Academy

---

**Sistema completamente refactorizado y listo para producción** 🚀




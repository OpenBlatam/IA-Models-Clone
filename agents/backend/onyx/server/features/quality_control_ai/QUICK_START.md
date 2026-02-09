# Quality Control AI - Quick Start Guide

## 🚀 Inicio Rápido en 5 Minutos

### 1. Instalación

```bash
# Instalar dependencias
pip install torch torchvision opencv-python numpy pillow fastapi uvicorn pydantic
```

### 2. Uso Básico - Código

```python
from quality_control_ai import (
    ApplicationServiceFactory,
    InspectionRequest,
)

# Crear servicio (todo configurado automáticamente)
factory = ApplicationServiceFactory()
service = factory.create_inspection_application_service()

# Inspeccionar imagen
import numpy as np
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

request = InspectionRequest(
    image_data=image,
    image_format="numpy",
    include_visualization=False,
)

response = service.inspect_image(request)

print(f"Quality Score: {response.quality_score}")
print(f"Status: {response.quality_status}")
print(f"Defects: {len(response.defects)}")
print(f"Anomalies: {len(response.anomalies)}")
print(f"Recommendation: {response.recommendation}")
```

### 3. Ejecutar API

```bash
# Opción 1: Script
python -m quality_control_ai.scripts.run_server

# Opción 2: Uvicorn
uvicorn quality_control_ai.presentation.api:app --host 0.0.0.0 --port 8000

# Acceder a:
# - API: http://localhost:8000/api/v1/
# - Docs: http://localhost:8000/docs
# - Health: http://localhost:8000/api/v1/health
```

### 4. Usar API

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Inspeccionar imagen (base64)
curl -X POST http://localhost:8000/api/v1/inspections \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "base64_encoded_image",
    "image_format": "base64",
    "include_visualization": false
  }'

# Subir archivo
curl -X POST http://localhost:8000/api/v1/inspections/upload \
  -F "file=@image.jpg" \
  -F "include_visualization=false"
```

## 📚 Ejemplos Completos

### Ejemplo 1: Inspección Simple

```python
from quality_control_ai import ApplicationServiceFactory, InspectionRequest
import numpy as np

# Setup
factory = ApplicationServiceFactory()
service = factory.create_inspection_application_service()

# Crear imagen de prueba
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Inspeccionar
request = InspectionRequest(
    image_data=image,
    image_format="numpy",
)
response = service.inspect_image(request)

# Resultados
print(f"Score: {response.quality_score:.2f}")
print(f"Status: {response.quality_status}")
print(f"Acceptable: {response.is_acceptable}")
```

### Ejemplo 2: Batch Inspection

```python
from quality_control_ai import (
    ApplicationServiceFactory,
    BatchInspectionRequest,
    InspectionRequest,
)
import numpy as np

factory = ApplicationServiceFactory()
service = factory.create_inspection_application_service()

# Crear múltiples imágenes
images = [
    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(5)
]

# Crear batch request
batch_request = BatchInspectionRequest(
    images=[
        InspectionRequest(image_data=img, image_format="numpy")
        for img in images
    ],
    parallel=True,
    max_workers=4,
)

# Inspeccionar batch
response = service.inspect_batch(batch_request)

print(f"Processed: {response.total_processed}")
print(f"Succeeded: {response.total_succeeded}")
print(f"Failed: {response.total_failed}")
print(f"Average Score: {response.average_quality_score:.2f}")
```

### Ejemplo 3: Usar Utilidades

```python
from quality_control_ai.utils import (
    format_bytes,
    format_duration,
    time_ago,
    generate_token,
    ensure_directory,
)

# Formatear
size = format_bytes(1024 * 1024)  # "1.00 MB"
duration = format_duration(3665)  # "1h 1m 5.00s"
ago = time_ago(some_datetime)  # "2 hours ago"

# Security
token = generate_token(32)

# File operations
dir_path = ensure_directory("./storage")
```

### Ejemplo 4: Health Check

```python
from quality_control_ai.infrastructure.health import get_health_checker

health = get_health_checker()
status = health.check_all()

print(f"Status: {status['status']}")
for check_name, check_result in status['checks'].items():
    print(f"  {check_name}: {check_result['status']} - {check_result['message']}")
```

### Ejemplo 5: Métricas

```python
from quality_control_ai.infrastructure.metrics import get_metrics_collector

metrics = get_metrics_collector()
data = metrics.get_metrics()

print(f"Total Inspections: {data['counters']['inspections.total']}")
print(f"Success Rate: {100 - data['errors']['error_rate']:.2f}%")
print(f"Uptime: {data['uptime_seconds']:.0f} seconds")
```

## 🔧 Configuración Rápida

### Variables de Entorno Mínimas

```bash
# .env file
API_PORT=8000
LOG_LEVEL=INFO
CACHE_ENABLED=True
```

### Configuración en Código

```python
from quality_control_ai.config.app_settings import get_settings

settings = get_settings()
settings.api_port = 8000
settings.log_level = "INFO"
```

## 📖 Más Información

- `README_REFACTORED.md` - Documentación completa
- `COMPLETE_REFACTORING_SUMMARY.md` - Resumen del refactor
- `/docs` endpoint - Documentación interactiva de API

## 🎯 Próximos Pasos

1. **Explorar API**: Visita http://localhost:8000/docs
2. **Ver Ejemplos**: Revisa `examples/usage_example.py`
3. **Configurar**: Ajusta variables de entorno según necesidad
4. **Integrar**: Usa el sistema en tu aplicación

---

**¡Listo para usar!** 🚀




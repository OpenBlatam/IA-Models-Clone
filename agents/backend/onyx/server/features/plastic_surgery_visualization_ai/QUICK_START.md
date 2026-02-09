# Quick Start Guide

Guía rápida para empezar con el proyecto.

## Instalación Rápida

```bash
# 1. Instalar dependencias
make install-dev

# 2. Configurar storage
make setup

# 3. Validar configuración
make validate-config

# 4. Ejecutar servidor
make run
```

## Configuración Mínima

Crea un archivo `.env`:

```env
HOST=0.0.0.0
PORT=8025
LOG_LEVEL=INFO
UPLOAD_DIR=./storage/uploads
OUTPUT_DIR=./storage/outputs
```

## Primer Request

```bash
# Health check
curl http://localhost:8025/health

# Crear visualización
curl -X POST http://localhost:8025/api/v1/visualize \
  -F "image=@photo.jpg" \
  -F "surgery_type=rhinoplasty" \
  -F "intensity=0.5"
```

## Estructura Básica

```
api/          # Endpoints HTTP
core/         # Interfaces y configuración
domain/       # Lógica de negocio
services/     # Servicios de aplicación
utils/        # Utilidades (60+)
```

## Comandos Útiles

```bash
# Testing
make test

# Linting
make lint

# Formatear código
make format

# Generar documentación
make generate-docs

# Limpiar storage
make cleanup-storage
```

## Ejemplo de Uso

```python
from services.visualization_service import VisualizationService
from api.schemas.visualization import VisualizationRequest, SurgeryType

service = VisualizationService()

request = VisualizationRequest(
    surgery_type=SurgeryType.RHINOPLASTY,
    intensity=0.5,
    image_data=image_bytes
)

result = await service.create_visualization(request)
print(f"Visualization ID: {result.visualization_id}")
```

## Recursos

- **README.md**: Documentación completa
- **ARCHITECTURE_IMPROVEMENTS.md**: Arquitectura detallada
- **UTILITIES_GUIDE.md**: Guía de utilidades
- **TESTING_GUIDE.md**: Guía de testing

## Soporte

Para más información, consulta la documentación completa en `README.md`.


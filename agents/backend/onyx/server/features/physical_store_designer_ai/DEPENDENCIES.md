# Dependencias del Proyecto

Este documento describe las dependencias del proyecto y sus propósitos.

## Instalación

### Instalación Completa (Recomendado)
```bash
pip install -r requirements.txt
```

### Instalación Mínima (Solo Core)
```bash
pip install -r requirements-minimal.txt
```

### Instalación para Desarrollo
```bash
pip install -r requirements-dev.txt
```

## Dependencias Principales

### Core Framework
- **fastapi**: Framework web moderno y rápido
- **uvicorn**: Servidor ASGI de alto rendimiento
- **pydantic**: Validación de datos con type hints
- **pydantic-settings**: Configuración basada en Pydantic

### HTTP & API
- **httpx**: Cliente HTTP async moderno (recomendado para async)
- **requests**: Cliente HTTP sync (compatibilidad)
- **aiohttp**: Alternativa async HTTP client

### Seguridad
- **PyJWT**: Tokens JWT
- **python-jose**: JWT con algoritmos adicionales
- **passlib**: Hashing de contraseñas
- **cryptography**: Criptografía de bajo nivel

### AI & ML
- **openai**: API de OpenAI
- **anthropic**: API de Claude (Anthropic)

### Procesamiento de Datos
- **numpy**: Computación numérica
- **pandas**: Análisis de datos
- **python-dateutil**: Utilidades de fecha/hora

### Utilidades
- **psutil**: Monitoreo del sistema
- **python-dotenv**: Variables de entorno
- **click/typer**: CLI frameworks
- **rich**: Output en terminal mejorado

### Validación
- **email-validator**: Validación de emails
- **marshmallow**: Serialización/deserialización
- **cerberus**: Validación de esquemas
- **jsonschema**: Validación JSON Schema

### Archivos
- **aiofiles**: Operaciones de archivo async
- **Pillow**: Procesamiento de imágenes
- **openpyxl**: Archivos Excel
- **python-docx**: Documentos Word
- **reportlab**: Generación de PDFs

### Caché
- **redis**: Caché distribuido
- **cachetools**: Caché en memoria

### Monitoreo
- **prometheus-client**: Métricas Prometheus
- **sentry-sdk**: Error tracking

### Logging
- **structlog**: Logging estructurado
- **python-json-logger**: Formato JSON para logs

### Testing
- **pytest**: Framework de testing
- **pytest-asyncio**: Soporte async en tests
- **pytest-cov**: Cobertura de código
- **faker**: Datos falsos para tests

### Calidad de Código
- **black**: Formateo de código
- **flake8**: Linting
- **mypy**: Type checking
- **isort**: Organización de imports
- **pre-commit**: Git hooks

## Dependencias Opcionales

### Base de Datos
- **sqlalchemy**: ORM
- **alembic**: Migraciones
- **asyncpg**: PostgreSQL async
- **pymongo**: MongoDB

### Task Queue
- **celery**: Procesamiento de tareas en background

### Deep Learning
- **torch**: PyTorch
- **transformers**: Modelos transformer
- **diffusers**: Modelos de difusión
- **gradio**: Interfaces interactivas

## Mejoras Aplicadas

### Versiones Actualizadas
- Todas las dependencias actualizadas a versiones más recientes y estables
- Compatibilidad con Python 3.10+

### Duplicados Eliminados
- `httpx` (aparecía 2 veces)
- `numpy` (aparecía 2 veces)
- `onnx` y `onnxruntime` (aparecían 2 veces)
- `redis` (aparecía 2 veces)

### Dependencias Removidas
- `python-cors>=1.7.0` - No existe, FastAPI ya incluye CORS
- `swagger-ui-bundle` - FastAPI incluye Swagger UI

### Organización Mejorada
- Secciones claramente separadas
- Comentarios descriptivos
- Dependencias opcionales claramente marcadas

### Archivos Adicionales
- `requirements-dev.txt` - Dependencias de desarrollo
- `requirements-minimal.txt` - Solo dependencias esenciales
- `DEPENDENCIES.md` - Documentación completa

## Recomendaciones

1. **Usar httpx para async**: Preferir `httpx` sobre `requests` para operaciones async
2. **Redis para producción**: Usar Redis para caché en producción
3. **Sentry para monitoreo**: Configurar Sentry para tracking de errores
4. **Type checking**: Ejecutar `mypy` regularmente
5. **Security scanning**: Usar `safety` para verificar vulnerabilidades

## Actualización de Dependencias

Para actualizar dependencias:

```bash
# Verificar dependencias desactualizadas
pip list --outdated

# Actualizar requirements.txt manualmente
# Luego reinstalar
pip install -r requirements.txt --upgrade
```

## Seguridad

Revisar vulnerabilidades conocidas:

```bash
pip install safety
safety check -r requirements.txt
```









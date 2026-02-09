# Guía de Librerías - Plastic Surgery Visualization AI

## Resumen

Este documento describe todas las librerías utilizadas en el proyecto, organizadas por categoría y propósito.

## Instalación

### Producción (Recomendado)
```bash
pip install -r requirements.txt
```

### Desarrollo
```bash
pip install -r requirements-dev.txt
```

### Mínimo (Solo funcionalidad básica)
```bash
pip install -r requirements-minimal.txt
```

### Opcionales (Features avanzadas)
```bash
pip install -r requirements-optional.txt
```

## Categorías de Librerías

### 1. Core FastAPI Stack

**fastapi** (>=0.104.0)
- Framework web moderno y rápido
- Soporte completo para async/await
- Generación automática de documentación OpenAPI

**uvicorn[standard]** (>=0.24.0)
- Servidor ASGI de alto rendimiento
- Soporte para WebSockets
- Hot reload en desarrollo

**pydantic** (>=2.5.0)
- Validación de datos con type hints
- Serialización/deserialización automática
- Validación de schemas

**starlette** (>=0.27.0)
- Framework base de FastAPI
- Middleware y routing

### 2. Async & HTTP

**aiofiles** (>=23.2.1)
- Operaciones de archivo asíncronas
- No bloquea el event loop

**aiohttp** (>=3.9.0)
- Cliente HTTP asíncrono
- Fetching de imágenes desde URLs

**httpx** (>=0.25.0)
- Cliente HTTP moderno
- Soporte para testing async
- Mejor que requests para async

### 3. Image Processing & Computer Vision

**Pillow** (>=10.1.0)
- Procesamiento básico de imágenes
- Conversión de formatos
- Operaciones de transformación

**numpy** (>=1.24.0)
- Operaciones numéricas
- Arrays multidimensionales
- Base para otras librerías de CV

**opencv-python-headless** (>=4.8.0)
- Computer vision avanzado
- Detección de características
- Transformaciones de imagen
- Headless (sin GUI) para servidores

**scikit-image** (>=0.21.0)
- Procesamiento de imágenes científico
- Filtros y transformaciones
- Segmentación

**imageio** (>=2.31.0)
- Lectura/escritura de imágenes
- Soporte para múltiples formatos
- Optimización de imágenes

### 4. Face Detection & Landmarks (Opcional)

**dlib** (>=19.24.0)
- Detección facial
- Landmark detection
- Requiere compilación C++

**face-recognition** (>=1.3.0)
- Wrapper fácil de usar sobre dlib
- Reconocimiento facial
- Comparación de caras

**mediapipe** (>=0.10.0)
- Framework de ML de Google
- Detección facial en tiempo real
- Landmark detection
- Más ligero que dlib

### 5. Machine Learning (Opcional)

**torch** (>=2.0.0)
- PyTorch para deep learning
- Modelos de transformación de imágenes
- Requiere GPU para mejor performance

**tensorflow** (>=2.13.0)
- TensorFlow para ML
- Modelos pre-entrenados
- Alternativa a PyTorch

**transformers** (>=4.30.0)
- Modelos de HuggingFace
- Vision transformers
- Modelos pre-entrenados

### 6. Validation & Utilities

**python-dotenv** (>=1.0.0)
- Carga de variables de entorno
- Configuración desde .env

**validators** (>=0.22.0)
- Validación de URLs, emails, etc.
- Validadores reutilizables

**email-validator** (>=2.0.0)
- Validación de emails
- Más robusto que regex

### 7. Monitoring & Observability

**prometheus-client** (>=0.18.0)
- Métricas Prometheus
- Exportación de métricas
- Integración con Grafana

**sentry-sdk** (>=1.32.0)
- Error tracking
- Performance monitoring
- Alertas automáticas

### 8. Testing

**pytest** (>=7.4.0)
- Framework de testing
- Fixtures y parametrización

**pytest-asyncio** (>=0.21.0)
- Soporte async para pytest
- Testing de funciones async

**pytest-cov** (>=4.1.0)
- Coverage de código
- Reportes de cobertura

**pytest-mock** (>=3.11.0)
- Mocking en tests
- Patches y spies

### 9. Code Quality

**black** (>=23.7.0)
- Formateo automático de código
- Estilo consistente

**flake8** (>=6.1.0)
- Linting de código
- Detección de errores

**mypy** (>=1.5.0)
- Type checking estático
- Validación de tipos

**isort** (>=5.12.0)
- Ordenamiento de imports
- Organización de código

### 10. Async Utilities

**tenacity** (>=8.2.0)
- Lógica de retry
- Backoff exponencial
- Manejo de errores transitorios

**asyncio-throttle** (>=1.0.2)
- Rate limiting utilities
- Throttling de requests
- Control de concurrencia

### 11. Security

**cryptography** (>=41.0.0)
- Primitivas criptográficas
- Hashing seguro
- Encriptación

**python-jose[cryptography]** (>=3.3.0)
- JWT tokens
- OAuth2
- Autenticación

### 12. Caching & Performance

**redis** (>=5.0.0)
- Caché distribuido
- Session storage
- Pub/sub

**hiredis** (>=2.2.0)
- Cliente Redis más rápido
- Parsing optimizado en C

## Uso por Feature

### Procesamiento Básico de Imágenes
- Pillow
- numpy
- imageio

### Computer Vision Avanzado
- opencv-python-headless
- scikit-image
- numpy

### Detección Facial (Futuro)
- dlib o mediapipe
- face-recognition (opcional)

### Machine Learning (Futuro)
- torch o tensorflow
- transformers

### Testing
- pytest + pytest-asyncio
- httpx (para testing HTTP)
- pytest-cov (coverage)

### Monitoreo
- prometheus-client
- sentry-sdk

## Recomendaciones de Instalación

### Para Desarrollo Local
```bash
pip install -r requirements-dev.txt
```

### Para Producción
```bash
pip install -r requirements.txt
```

### Para Servidores con Recursos Limitados
```bash
pip install -r requirements-minimal.txt
```

### Para Features Avanzadas
```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

## Notas Importantes

1. **dlib** requiere compilación C++ y puede ser difícil de instalar. Considera usar **mediapipe** como alternativa más fácil.

2. **opencv-python-headless** es más ligero que **opencv-python** (sin GUI) y es mejor para servidores.

3. **torch** y **tensorflow** son grandes. Solo instálalos si planeas usar modelos locales.

4. **redis** es opcional pero recomendado para producción con múltiples instancias.

5. Para desarrollo, instala **requirements-dev.txt** que incluye herramientas de calidad de código.

## Troubleshooting

### Error al instalar dlib
```bash
# En Ubuntu/Debian
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev

# Luego instalar dlib
pip install dlib
```

### Error con opencv
```bash
# Usa la versión headless
pip install opencv-python-headless
```

### Error con torch (tamaño)
```bash
# Instala solo CPU version (más pequeña)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Nuevas Librerías Agregadas

### Image Processing Avanzado
- **scipy** - Computación científica para procesamiento avanzado
- **matplotlib** - Visualización y plotting
- **wand** - Binding de ImageMagick
- **pyvips** - Procesamiento rápido de imágenes

### Data Processing
- **pandas** - Manipulación y análisis de datos
- **orjson** - JSON rápido (más rápido que json estándar)
- **ujson** - Encoder/decoder JSON ultra-rápido

### Logging Estructurado
- **structlog** - Logging estructurado avanzado
- **python-json-logger** - Formatter JSON para logging

### Caching Avanzado
- **cachetools** - Utilidades avanzadas de caché
- **diskcache** - Caché basado en disco

### Utilidades
- **tqdm** - Progress bars
- **rich** - Formato de texto rico y hermoso
- **colorama** - Texto coloreado en terminal
- **python-dateutil** - Manipulación de fechas
- **pytz** - Soporte de timezones
- **pyyaml** - Parsing de YAML
- **toml** - Parsing de TOML
- **click** - Framework CLI

### Compression
- **brotli** - Compresión Brotli

## Archivos de Requirements Adicionales

1. **requirements-gpu.txt** - Dependencias para GPU (CUDA)
2. **requirements-production.txt** - Dependencias optimizadas para producción

## Actualización de Dependencias

```bash
# Ver dependencias desactualizadas
pip list --outdated

# Actualizar todas
pip install --upgrade -r requirements.txt

# Actualizar una específica
pip install --upgrade package-name

# Verificar vulnerabilidades
safety check
```


# Librerías y Dependencias - Burnout Prevention AI

## 📦 Estructura de Requirements

El proyecto incluye tres archivos de requirements para diferentes casos de uso:

### `requirements.txt` - Producción
Dependencias esenciales para producción con optimizaciones de rendimiento.

### `requirements-dev.txt` - Desarrollo
Incluye herramientas de testing, linting y desarrollo.

### `requirements-minimal.txt` - Mínimo
Solo las dependencias core necesarias para funcionar.

## 🎯 Librerías Principales

### Core Framework
- **FastAPI 0.115+**: Framework web moderno con validación automática
- **Uvicorn**: Servidor ASGI con soporte HTTP/2
- **Pydantic 2.9+**: Validación de datos con mejoras de rendimiento v2
- **Pydantic Settings**: Gestión de configuración desde variables de entorno

### HTTP & Networking
- **httpx 0.27+**: Cliente HTTP asíncrono moderno (reemplaza requests)
- **httpcore**: Biblioteca HTTP core usada por httpx

### Rendimiento
- **orjson 3.10+**: JSON 2-3x más rápido que stdlib (escrito en Rust)
- **uvloop 0.20+**: Event loop ultra-rápido (solo Linux/macOS)

### Logging & Observabilidad
- **structlog 24.4+**: Logging estructurado con JSON
- **python-json-logger**: Formateador JSON para logs
- **prometheus-client**: Exportador de métricas Prometheus

### Resiliencia
- **tenacity 9.0+**: Reintentos con backoff exponencial

### Rate Limiting
- **slowapi 0.1.9+**: Rate limiting para FastAPI

### Testing
- **pytest 8.3+**: Framework de testing
- **pytest-asyncio**: Soporte para tests asíncronos
- **pytest-cov**: Reporte de cobertura

### Desarrollo
- **black**: Formateador de código
- **ruff**: Linter rápido (reemplaza flake8, isort)
- **mypy**: Type checking estático

## 🔄 Mejoras vs Versión Anterior

### Antes
```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
httpx>=0.25.0
```

### Ahora
- Versiones más recientes (FastAPI 0.115, Pydantic 2.9)
- orjson para JSON más rápido
- structlog para logging estructurado
- uvloop para mejor rendimiento
- tenacity para reintentos inteligentes
- prometheus-client para métricas
- slowapi para rate limiting

## 📊 Comparación de Rendimiento

| Librería | Antes | Ahora | Mejora |
|----------|-------|-------|--------|
| JSON | stdlib | orjson | 2-3x más rápido |
| Event Loop | asyncio | uvloop | ~2x más rápido |
| Logging | basic | structlog | Mejor estructuración |
| HTTP | httpx 0.25 | httpx 0.27 | Mejoras de rendimiento |

## 🚀 Uso Recomendado

### Producción
```bash
pip install -r requirements.txt
```

### Desarrollo
```bash
pip install -r requirements-dev.txt
```

### Mínimo (solo core)
```bash
pip install -r requirements-minimal.txt
```

## 🔧 Configuración Opcional

### uvloop (Linux/macOS)
Se activa automáticamente si está disponible. Mejora significativamente el rendimiento.

### Rate Limiting
Puede configurarse agregando slowapi middleware en `main.py` si es necesario.

### Prometheus
Las métricas están disponibles en `/metrics` si prometheus-client está instalado.


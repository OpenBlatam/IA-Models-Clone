# Guía de Librerías - Dermatology AI

## Resumen

Este documento describe las mejores librerías seleccionadas para el proyecto, organizadas por categoría y uso.

## Estructura de Requirements

- **`requirements.txt`** - Dependencias de producción completas
- **`requirements-dev.txt`** - Dependencias adicionales para desarrollo
- **`requirements-minimal.txt`** - Dependencias mínimas para serverless

## Categorías de Librerías

### 1. Core FastAPI Stack

#### FastAPI (>=0.115.0)
- Framework web moderno y rápido
- Validación automática con Pydantic
- Documentación automática (Swagger/OpenAPI)
- Soporte async nativo

#### Uvicorn (>=0.32.0)
- Servidor ASGI de alto rendimiento
- Soporte para HTTP/2, WebSockets
- Workers múltiples
- Hot reload para desarrollo

#### Pydantic (>=2.9.0)
- Validación de datos con type hints
- Serialización rápida (Rust-based en v2)
- Settings management
- Mejoras de performance vs v1

### 2. Autenticación & Seguridad

#### python-jose (>=3.3.0)
- **Mejor opción** para JWT/OAuth2 en Python
- Soporte para múltiples algoritmos
- Integración con FastAPI

#### passlib (>=1.7.4)
- **Estándar de la industria** para password hashing
- Múltiples algoritmos (bcrypt, argon2, etc.)
- Actualizaciones automáticas de hash

#### slowapi (>=0.1.9)
- Rate limiting con Redis
- Más ligero que alternatives
- Compatible con FastAPI

### 3. Base de Datos

#### SQLAlchemy (>=2.0.36)
- **ORM más popular** en Python
- Soporte async completo en v2.0+
- Type hints mejorados
- Performance optimizado

#### asyncpg (>=0.30.0)
- **Driver más rápido** para PostgreSQL
- Implementado en C (muy rápido)
- Soporte async nativo
- Mejor que psycopg2 para async

#### Motor (>=3.6.0)
- Driver async para MongoDB
- Basado en PyMongo
- Performance optimizado

### 4. Caching

#### Redis (>=5.2.0) con hiredis
- **Estándar de la industria** para caching distribuido
- hiredis: parser C (muy rápido)
- Soporte para pub/sub, streams, etc.

#### aiocache (>=0.12.2)
- Framework de caching async
- Múltiples backends (Redis, Memcached, etc.)
- Decorators fáciles de usar

### 5. Message Brokers

#### aio-pika (>=9.4.0)
- **Mejor opción** para RabbitMQ async
- Performance optimizado
- Soporte completo de AMQP

#### aiokafka (>=0.11.0)
- **Mejor opción** para Kafka async
- Consumer groups, transactions
- Performance optimizado

#### Celery (>=5.4.0)
- **Estándar** para task queues
- Múltiples brokers (Redis, RabbitMQ, etc.)
- Monitoring con Flower

### 6. HTTP Clients

#### httpx (>=0.27.2)
- **Recomendado sobre aiohttp** para nuevos proyectos
- API más moderna
- HTTP/2 support
- Mejor type hints
- Client de testing integrado

#### aiohttp (>=3.11.0)
- Alternativa estable y madura
- Más features (WebSockets, etc.)
- Buena para casos específicos

### 7. Observabilidad

#### OpenTelemetry (>=1.27.0)
- **Estándar de la industria** para tracing
- Vendor-agnostic
- Auto-instrumentation para FastAPI
- Exporters para múltiples backends

#### Prometheus Client (>=0.21.0)
- **Estándar** para métricas
- Integración con Grafana
- Métricas personalizadas fáciles

#### structlog (>=24.4.0)
- **Mejor que logging estándar** para producción
- Structured logging (JSON)
- Context propagation
- Performance optimizado

### 8. Performance

#### uvloop (>=0.20.0)
- **Event loop más rápido** (2-4x que asyncio)
- Drop-in replacement
- Solo Linux/macOS

#### orjson (>=3.10.10)
- **JSON más rápido** (Rust-based)
- 2-3x más rápido que json estándar
- Soporte para más tipos

#### msgpack (>=1.1.0)
- Serialización binaria
- Más rápido y compacto que JSON
- Ideal para caching

### 9. Testing

#### pytest (>=8.3.3)
- **Framework de testing más popular**
- Fixtures, parametrización
- Plugins extensos

#### httpx (para testing)
- Client de testing async
- Mejor que TestClient de FastAPI para casos avanzados

### 10. Code Quality

#### ruff (>=0.7.0)
- **Linter más rápido** (Rust-based)
- Reemplaza: flake8, isort, pyupgrade, etc.
- 10-100x más rápido

#### black (>=24.10.0)
- **Formatter estándar** de la industria
- Sin configuración necesaria
- Consistencia garantizada

#### mypy (>=1.11.0)
- **Type checker estándar**
- Type hints validation
- Mejora calidad del código

### 11. Serverless

#### mangum (>=0.18.1)
- **Mejor opción** para AWS Lambda
- ASGI adapter
- Soporte para lifespan events

#### aws-lambda-powertools (>=2.40.0)
- Best practices para Lambda
- Logging, tracing, métricas
- Utilities comunes

## Comparación de Alternativas

### JSON Libraries
- **orjson**: Más rápido (Rust), mejor para producción
- **ujson**: Rápido (C), buena alternativa
- **json estándar**: Más lento, pero built-in

### HTTP Clients
- **httpx**: Moderno, HTTP/2, mejor para nuevos proyectos
- **aiohttp**: Maduro, más features, buena para casos específicos
- **requests**: Sync, solo para casos legacy

### Logging
- **structlog**: Structured logging, mejor para producción
- **logging estándar**: Simple, suficiente para desarrollo

### Linters
- **ruff**: Más rápido, reemplaza múltiples herramientas
- **flake8**: Tradicional, más lento
- **pylint**: Más estricto, más lento

## Mejores Prácticas

### 1. Version Pinning
```bash
# Para producción, usar exact versions
fastapi==0.115.0
pydantic==2.9.0

# O usar pip-tools para generar lock file
pip-compile requirements.txt
```

### 2. Security Updates
```bash
# Revisar vulnerabilidades regularmente
safety check
pip-audit
```

### 3. Dependency Management
```bash
# Usar poetry o pip-tools
poetry add fastapi
pip-compile requirements.in
```

### 4. Minimal Dependencies
- Para serverless, usar `requirements-minimal.txt`
- Agregar dependencias solo cuando se necesiten
- Evitar dependencias pesadas (ML libraries) si no se usan

## Actualización de Versiones

### Proceso Recomendado
1. Revisar changelogs de librerías principales
2. Actualizar en entorno de desarrollo
3. Ejecutar tests completos
4. Revisar breaking changes
5. Actualizar en producción

### Frecuencia
- **Críticas (seguridad)**: Inmediato
- **Mayores**: Cada 3-6 meses
- **Menores/Parches**: Mensual

## Recursos

- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Dependency Security](https://pyup.io/safety/)
- [Python Performance](https://docs.python.org/3/library/profile.html)
















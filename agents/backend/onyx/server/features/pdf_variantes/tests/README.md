# PDF Variantes - Test Suite

Suite completa de tests para el módulo PDF Variantes.

## Estructura de Tests

```
tests/
├── __init__.py              # Inicialización del módulo de tests
├── conftest.py              # Configuración y fixtures compartidos
├── test_api_endpoints.py   # Tests para endpoints de API
├── test_variant_generator.py # Tests para generador de variantes
├── test_services.py         # Tests para servicios
├── test_integration.py      # Tests de integración end-to-end
├── test_utils.py            # Tests unitarios para utilidades
├── test_exceptions.py       # Tests unitarios para excepciones
├── test_models.py           # Tests unitarios para modelos
├── test_schemas.py          # Tests unitarios para schemas
├── test_dependencies.py     # Tests unitarios para dependencias
├── test_performance.py      # Tests de rendimiento
├── test_edge_cases.py       # Tests de casos límite y edge cases
├── test_pdf_processor.py    # Tests para procesamiento de PDFs
├── test_analytics.py        # Tests para analytics y estadísticas
├── test_security.py         # Tests para seguridad y autenticación
├── test_topic_extractor.py  # Tests para extracción de topics
├── test_cache.py           # Tests para gestión de cache
├── test_config.py          # Tests para configuración
├── test_middleware.py      # Tests para middleware
├── test_e2e.py            # Tests end-to-end completos
├── test_e2e_scenarios.py  # Tests e2e de escenarios reales
├── test_playwright.py     # Tests con Playwright (automatización de navegador)
├── test_playwright_scenarios.py # Tests Playwright de escenarios
├── test_playwright_advanced.py # Tests avanzados de Playwright
├── test_playwright_api.py     # Tests específicos de API con Playwright
├── test_playwright_ui.py      # Tests de UI/interfaz con Playwright
├── test_playwright_load.py    # Tests de carga y stress con Playwright
├── test_playwright_comprehensive.py # Tests comprehensivos de Playwright
├── test_playwright_security.py # Tests de seguridad con Playwright
├── test_playwright_workflows.py # Tests de workflows complejos
├── test_playwright_regression.py # Tests de regresión
├── test_playwright_improved.py # Tests mejorados con helpers avanzados
├── test_playwright_integration.py # Tests de integración
├── test_playwright_parallel.py # Tests de ejecución paralela
├── test_playwright_quality.py # Tests de calidad y mejores prácticas
├── test_playwright_monitoring.py # Tests de monitoreo y observabilidad
├── test_playwright_compliance.py # Tests de cumplimiento y estándares
├── test_playwright_chaos.py # Tests de chaos engineering
├── test_playwright_visual.py # Tests de regresión visual
├── test_playwright_accessibility.py # Tests de accesibilidad
├── test_playwright_api_contract.py # Tests de contrato de API
├── test_playwright_smoke.py # Tests de smoke (rápidos)
├── test_playwright_ci.py # Tests optimizados para CI/CD
├── test_playwright_benchmark.py # Tests de benchmarking
├── test_playwright_exploratory.py # Tests exploratorios
├── test_playwright_refactored.py # Tests refactorizados con clases base
├── playwright_base.py # Clases base para tests
├── playwright_pages.py # Page Object Model (POM)
├── playwright_test_runner.py # Test runner y reportes
├── test_playwright_pom.py # Tests usando POM
├── conftest_playwright.py     # Configuración de Playwright
├── playwright.config.py        # Configuración adicional de Playwright
├── playwright_helpers.py       # Funciones helper para Playwright
├── fixtures_common.py          # Fixtures comunes centralizadas
├── POM_GUIDE.md                # Guía de Page Object Model
├── TEST_RUNNER_GUIDE.md        # Guía de Test Runner
├── playwright_debug.py         # Utilidades de debugging
├── playwright_analytics.py     # Analytics y métricas
├── playwright_comparison.py   # Utilidades de comparación
├── test_playwright_debug.py    # Tests de debugging
└── run_tests_advanced.py       # Script avanzado de ejecución
├── test_pdf_variantes.py    # Tests unitarios existentes
└── test_system.py           # Tests del sistema existentes
```

## Ejecutar Tests

### Ejecutar todos los tests

```bash
pytest
```

### Ejecutar tests específicos

```bash
# Tests de API
pytest tests/test_api_endpoints.py

# Tests de generador de variantes
pytest tests/test_variant_generator.py

# Tests de servicios
pytest tests/test_services.py

# Tests de integración
pytest tests/test_integration.py

# Tests unitarios de utilidades
pytest tests/test_utils.py

# Tests unitarios de excepciones
pytest tests/test_exceptions.py

# Tests unitarios de modelos
pytest tests/test_models.py

# Tests unitarios de schemas
pytest tests/test_schemas.py

# Tests unitarios de dependencias
pytest tests/test_dependencies.py

# Tests de rendimiento
pytest tests/test_performance.py

# Tests de casos límite
pytest tests/test_edge_cases.py

# Tests de procesamiento de PDFs
pytest tests/test_pdf_processor.py

# Tests de analytics
pytest tests/test_analytics.py

# Tests de seguridad
pytest tests/test_security.py

# Tests de extracción de topics
pytest tests/test_topic_extractor.py

# Tests de cache
pytest tests/test_cache.py

# Tests de configuración
pytest tests/test_config.py

# Tests de middleware
pytest tests/test_middleware.py

# Tests end-to-end
pytest tests/test_e2e.py -v

# Tests e2e de escenarios
pytest tests/test_e2e_scenarios.py -v

# Tests con Playwright
pytest tests/test_playwright.py -v

# Tests Playwright de escenarios
pytest tests/test_playwright_scenarios.py -v

# Todos los tests de Playwright
pytest -m playwright -v

# Playwright con navegador visible
pytest tests/test_playwright.py --headless=false -v

# Playwright con navegador específico
pytest tests/test_playwright.py --browser=firefox -v

# Tests avanzados de Playwright
pytest tests/test_playwright_advanced.py -v

# Todos los tests de Playwright (básicos + avanzados)
pytest tests/test_playwright.py tests/test_playwright_advanced.py -v

# Tests con helpers
pytest tests/test_playwright.py -v -k "helper"

# Tests específicos de API
pytest tests/test_playwright_api.py -v

# Tests de UI
pytest tests/test_playwright_ui.py -v

# Tests de carga (pueden ser lentos)
pytest tests/test_playwright_load.py -v -m load

# Tests de stress
pytest tests/test_playwright_load.py -v -m stress

# Todos los tests de Playwright
pytest tests/test_playwright*.py -v

# Tests comprehensivos
pytest tests/test_playwright_comprehensive.py -v

# Tests de seguridad
pytest tests/test_playwright_security.py -v

# Tests de seguridad y carga
pytest tests/test_playwright_security.py tests/test_playwright_load.py -v

# Tests de workflows
pytest tests/test_playwright_workflows.py -v -m workflow

# Tests de regresión
pytest tests/test_playwright_regression.py -v -m regression

# Tests de integración
pytest tests/test_playwright_integration.py -v -m integration

# Tests paralelos
pytest tests/test_playwright_parallel.py -v -m parallel

# Tests de calidad
pytest tests/test_playwright_quality.py -v -m quality

# Tests de monitoreo
pytest tests/test_playwright_monitoring.py -v -m monitoring

# Tests de cumplimiento
pytest tests/test_playwright_compliance.py -v -m compliance

# Tests de chaos engineering
pytest tests/test_playwright_chaos.py -v -m chaos

# Tests visuales
pytest tests/test_playwright_visual.py -v -m visual

# Tests de accesibilidad
pytest tests/test_playwright_accessibility.py -v -m accessibility

# Tests de contrato de API
pytest tests/test_playwright_api_contract.py -v -m contract

# Tests de smoke (rápidos)
pytest tests/test_playwright_smoke.py -v -m smoke

# Tests de CI/CD
pytest tests/test_playwright_ci.py -v -m ci

# Tests de benchmarking
pytest tests/test_playwright_benchmark.py -v -m benchmark

# Tests exploratorios
pytest tests/test_playwright_exploratory.py -v -m exploratory

# Tests refactorizados
pytest tests/test_playwright_refactored.py -v -m refactored

# Tests con utilidades
pytest tests/test_playwright_refactored_utils.py -v -m refactored

# Tests con decoradores
pytest tests/test_playwright_refactored_decorators.py -v -m refactored

# Tests con Page Object Model
pytest tests/test_playwright_pom.py -v -m pom

# Todos los tests de Playwright
pytest tests/test_playwright*.py -v

# Solo smoke tests (muy rápidos)
pytest -m smoke -v

# Solo tests rápidos para CI
pytest -m "ci and fast" -v

# Solo tests críticos
pytest -m critical -v

# Todos los tests e2e
pytest -m e2e -v

# Tests e2e de escenarios
pytest -m scenario -v

# Tests e2e lentos (incluye performance)
pytest -m "e2e and slow" -v
```

### Ejecutar por marcadores

```bash
# Solo tests unitarios
pytest -m unit

# Solo tests de integración
pytest -m integration

# Tests de API
pytest -m api

# Tests unitarios solamente
pytest -m unit

# Excluir tests lentos
pytest -m "not slow"

# Ejecutar todos los tests unitarios
pytest tests/test_utils.py tests/test_exceptions.py tests/test_models.py tests/test_schemas.py tests/test_dependencies.py

# Ejecutar tests de rendimiento (pueden ser más lentos)
pytest tests/test_performance.py -m performance

# Ejecutar tests de casos límite
pytest tests/test_edge_cases.py -m edge_case

# Ejecutar todos los tests excepto los de rendimiento
pytest -m "not performance"
```

### Ejecutar con cobertura

```bash
pytest --cov=. --cov-report=html --cov-report=term
```

### Ejecutar con verbose

```bash
pytest -v
```

### Ejecutar tests en paralelo (requiere pytest-xdist)

```bash
pytest -n auto
```

## Tipos de Tests

### 1. Tests de API Endpoints (`test_api_endpoints.py`)

Tests para todos los endpoints de la API:
- Upload de PDFs
- Generación de variantes
- Extracción de topics
- Preview de páginas
- Eliminación de PDFs
- Health checks

**Ejemplo:**
```python
def test_upload_pdf_success(client, mock_pdf_service, sample_pdf_content):
    """Test successful PDF upload."""
    files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
    response = client.post("/pdf/upload", files=files)
    assert response.status_code == 200
```

### 2. Tests de Variant Generator (`test_variant_generator.py`)

Tests para el generador de variantes:
- Generación de diferentes tipos de variantes (summary, outline, highlights, etc.)
- Validación de opciones
- Manejo de errores

**Ejemplo:**
```python
async def test_generate_summary(generator, sample_pdf_file, default_options):
    """Test summary generation."""
    result = await generator.generate(
        file=sample_pdf_file,
        variant_type=VariantType.SUMMARY,
        options=default_options
    )
    assert result["variant_type"] == "summary"
```

### 3. Tests de Servicios (`test_services.py`)

Tests para los servicios del sistema:
- PDFVariantesService
- CollaborationService
- MonitoringSystem
- AnalyticsService
- HealthService
- CacheService
- SecurityService

### 4. Tests de Integración (`test_integration.py`)

Tests end-to-end que verifican flujos completos:
- Upload → Generate Variant
- Upload → Preview → Extract Topics
- Operaciones concurrentes
- Persistencia de datos

### 5. Tests Unitarios de Utilidades (`test_utils.py`)

Tests para funciones de utilidad:
- Validación de nombres de archivo
- Validación de extensiones
- Validación de rangos numéricos
- Validación de emails y UUIDs
- Helpers de archivos
- Validadores de archivos

### 6. Tests Unitarios de Excepciones (`test_exceptions.py`)

Tests para excepciones personalizadas:
- PDFVariantesError (base)
- PDFNotFoundError
- InvalidFileError
- FileSizeError
- ProcessingError
- VariantGenerationError
- Manejo de detalles y códigos de error

### 7. Tests Unitarios de Modelos (`test_models.py`)

Tests para modelos Pydantic:
- Enums (VariantStatus, PDFProcessingStatus, VariantType)
- PDFMetadata
- PDFVariant
- VariantConfiguration
- VariantGenerateRequest/Response
- Serialización y validación

### 8. Tests Unitarios de Schemas (`test_schemas.py`)

Tests para schemas de API:
- PDFUploadSchema
- VariantGenerateSchema
- TopicExtractSchema
- PDFEditSchema
- Validación y serialización

### 9. Tests Unitarios de Dependencias (`test_dependencies.py`)

Tests para dependencias de FastAPI:
- get_pdf_service
- get_current_user
- validate_file_size
- get_db_session
- get_cache_service
- Inyección de dependencias

### 10. Tests de Rendimiento (`test_performance.py`)

Tests para verificar que se cumplen los requisitos de rendimiento:
- Validación rápida de nombres de archivo (< 1ms)
- Validación de extensiones (< 1ms)
- Validación de rangos numéricos (< 1ms)
- Validación de emails y UUIDs (< 1ms)
- Validación de PDFs (< 100ms)
- Extracción de metadata (< 500ms)
- Operaciones concurrentes
- Uso de memoria
- Escalabilidad

### 11. Tests de Casos Límite (`test_edge_cases.py`)

Tests para casos límite y condiciones de borde:
- Strings vacíos y solo espacios en blanco
- Caracteres Unicode y emojis
- Strings muy largos
- Valores en los límites
- Valores None
- Caracteres especiales
- Sensibilidad a mayúsculas/minúsculas
- Acceso concurrente
- Eficiencia de memoria
- Claridad de mensajes de error

### 12. Tests de Procesamiento de PDFs (`test_pdf_processor.py`)

Tests para funciones de procesamiento de PDFs:
- Extracción de contenido de PDFs
- Procesamiento con reintentos
- Extracción asíncrona de topics
- Generación asíncrona de variantes
- Análisis de calidad de contenido
- Pipeline de procesamiento
- Procesamiento completo
- Procesamiento por lotes
- Generación de IDs de archivo
- Extracción paralela de features
- Análisis de contenido
- Procesamiento con métricas

### 13. Tests de Analytics (`test_analytics.py`)

Tests para sistema de analytics:
- UsageStats (estadísticas de uso)
- UserActivity (actividad de usuario)
- AnalyticsEngine (motor de analytics)
- Tracking de eventos
- Estadísticas de uso
- Actividad de usuarios
- Usuarios más activos
- Variantes populares
- Estadísticas diarias
- Estadísticas por rango de tiempo
- Agregación de métricas
- Exportación de analytics

### 14. Tests de Seguridad (`test_security.py`)

Tests para sistema de seguridad:
- PermissionType (tipos de permisos)
- AccessToken (tokens de acceso)
- AuditLog (logs de auditoría)
- SecurityManager (gestor de seguridad)
- Generación de tokens
- Validación de tokens
- Verificación de permisos
- Logging de auditoría
- Revocación de tokens
- Encriptación/desencriptación
- Hash y verificación de contraseñas
- Ciclo de vida de tokens

### 15. Tests de Extracción de Topics (`test_topic_extractor.py`)

Tests para extracción de topics:
- Topic dataclass
- PDFTopicExtractor
- Extracción de topics desde texto
- Filtrado por relevancia mínima
- Límite de topics máximos
- Manejo de contenido vacío
- Scores de relevancia
- Ordenamiento por relevancia
- Categorización de topics
- Casos edge (texto corto/largo, Unicode, caracteres especiales)

### 16. Tests de Cache (`test_cache.py`)

Tests para gestión de cache:
- CachePolicy enum
- CacheEntry class
- CacheManager class
- Operaciones get/set/delete
- TTL y expiración
- Límites de tamaño
- Patrón get_or_set
- Estadísticas de cache
- Acceso concurrente

### 17. Tests de Configuración (`test_config.py`)

Tests para sistema de configuración:
- Settings base class
- DevelopmentSettings
- ProductionSettings
- TestingSettings
- get_settings function
- get_settings_by_env function
- FeatureFlags
- AIConfig
- SecurityConfig
- PerformanceConfig
- ExportConfig
- Variables de entorno

### 18. Tests de Middleware (`test_middleware.py`)

Tests para middleware personalizado:
- LoggingMiddleware
- RateLimitMiddleware
- ErrorHandlingMiddleware
- PerformanceMiddleware
- Logging de requests/responses
- Rate limiting
- Manejo de errores
- Tracking de performance
- Encadenamiento de middlewares
- Headers de respuesta

### 19. Tests End-to-End (`test_e2e.py`)

Tests end-to-end completos que prueban flujos reales:
- Health check e2e
- Journey completo de usuario (Upload -> Preview -> Topics -> Variants)
- Generación de múltiples variantes
- Workflow de eliminación
- Escenarios de error e2e
- Operaciones concurrentes
- Tests de rendimiento e2e
- Persistencia de datos
- Compatibilidad de API
- Rate limiting e2e
- Ciclo de vida completo de documentos

### 20. Tests E2E de Escenarios (`test_e2e_scenarios.py`)

Tests e2e basados en escenarios reales:
- **Estudiante**: Upload notas -> Generar resumen + quiz
- **Investigador**: Análisis por lotes de múltiples papers
- **Creador de contenido**: Generar múltiples variaciones
- **Colaboración**: Múltiples usuarios en el mismo documento
- **Recuperación de errores**: Sistema después de errores
- **Degradación elegante**: Cuando servicios no están disponibles
- **Carga de trabajo**: Operaciones secuenciales y mixtas

### 21. Tests con Playwright (`test_playwright.py`)

Tests de automatización de navegador con Playwright:
- Requests HTTP a la API
- Navegación del navegador
- Envío de formularios
- Documentación de API (Swagger/OpenAPI)
- Manejo de errores
- Tests de performance
- Captura de screenshots
- Monitoreo de red
- Headers de respuesta
- CORS y seguridad

### 22. Tests Playwright de Escenarios (`test_playwright_scenarios.py`)

Escenarios reales con Playwright:
- Journey completo de usuario via browser
- Interacción con documentación de API
- Escenarios de error en navegador
- Monitoreo de performance
- Verificación de headers de seguridad
- Tests de latencia y concurrencia

### 23. Tests Avanzados de Playwright (`test_playwright_advanced.py`)

Tests avanzados con funcionalidades mejoradas:
- Descubrimiento automático de endpoints API
- Validación de estructuras de datos
- Manejo de cookies y storage
- Grabación de video y traces
- Tests cross-browser (Chromium, Firefox, WebKit)
- Emulación de dispositivos (iPhone, iPad)
- Condiciones de red (slow 3G, offline)
- Geolocalización y permisos
- Tests de accesibilidad
- Tests responsive (mobile, tablet, desktop)

### 24. Helpers de Playwright (`playwright_helpers.py`)

Funciones utilitarias para tests de Playwright:
- wait_for_api_response: Espera respuestas específicas
- retry_request: Reintentos con exponential backoff
- assert_json_response: Validación de JSON
- measure_performance: Métricas de performance
- check_accessibility: Verificación de accesibilidad
- mock_api_response: Mocking de respuestas
- extract_api_endpoints_from_openapi: Descubrimiento de endpoints
- Y más funciones helper

### 25. Tests Playwright de API (`test_playwright_api.py`)

Tests específicos de API con Playwright:
- Upload de PDFs (multipart, con opciones, archivos grandes)
- Generación de variantes (todos los tipos, opciones personalizadas, async)
- Extracción de topics (básica, con filtros, paginación)
- Preview de PDFs (múltiples páginas, opciones)
- Gestión de PDFs (listar, metadata, actualizar, eliminar)
- Operaciones por lotes (batch upload, batch variants)
- Búsqueda y filtrado (por texto, fecha, tags)
- Webhooks (registro, listado)
- Rate limiting (detección, headers)
- Versionado de API

### 26. Tests Playwright de UI (`test_playwright_ui.py`)

Tests de interfaz de usuario con Playwright:
- Navegación (home, docs, back/forward)
- Elementos UI (botones, links, formularios, inputs)
- Interacciones (click, typing, scrolling)
- Formularios (text inputs, dropdowns, checkboxes)
- Scrolling (top, bottom)
- Modales (alerts, confirms)
- Tabs/windows (abrir, cambiar)
- Teclado (typing, shortcuts)
- Imágenes (carga, alt text)

### 27. Tests Playwright de Carga (`test_playwright_load.py`)

Tests de carga y stress con Playwright:
- Carga concurrente (health checks, sustained load, ramp-up)
- Stress testing (max connections, rapid-fire)
- Memory leaks (uso de memoria en el tiempo)
- Connection pooling (reutilización de conexiones)
- Timeout handling (bajo carga)
- Resource limits (tamaños máximos de archivo)

### 28. Tests Comprehensivos de Playwright (`test_playwright_comprehensive.py`)

Tests comprehensivos con mejores prácticas:
- Validación de requests (campos requeridos, JSON, content-type)
- Validación de responses (estructura, formato)
- Caching (headers, conditional requests)
- Compresión (GZIP, DEFLATE)
- Streaming responses
- Paginación (links, consistencia)
- Sorting y ordering
- Field selection (sparse fieldsets)
- Content negotiation (Accept headers)
- Idempotency
- Concurrency control (ETags, optimistic locking)
- HATEOAS (links en responses)
- Versioning (URL, headers)
- Deprecation headers
- Metrics y monitoring
- WebSocket connections
- GraphQL endpoints
- OAuth endpoints
- File download
- Export functionality
- Bulk operations
- Notifications

### 29. Tests de Seguridad con Playwright (`test_playwright_security.py`)

Tests de seguridad con Playwright:
- Autenticación (missing, invalid, expired tokens)
- Autorización (unauthorized access, RBAC)
- Input sanitization (XSS, SQL injection, path traversal)
- CSRF protection
- Rate limiting security
- Data exposure prevention
- HTTPS enforcement
- Security headers
- Session security (HttpOnly, Secure cookies)
- API key security

### 30. Tests de Workflows con Playwright (`test_playwright_workflows.py`)

Tests de workflows complejos:
- Workflow completo (Upload -> Process -> Variants -> Topics -> Export)
- Workflow colaborativo (múltiples usuarios)
- Error recovery workflow
- Data flow consistency
- State transitions
- Async operations (polling, job status)
- Chain operations (variants en cadena)
- Dependent operations
- Rollback scenarios
- Atomic operations
- Optimistic locking
- Event-driven workflows

### 31. Tests de Regresión con Playwright (`test_playwright_regression.py`)

Tests de regresión para prevenir bugs:
- File ID consistency
- No duplicate file IDs
- Error message consistency
- Memory leak prevention
- Timeout handling consistency
- Data integrity (no corruption on retry)
- Metadata preservation
- Backward compatibility
- Deprecated endpoints
- Race conditions (concurrent uploads/deletes)
- Edge cases (empty strings, long IDs, special chars)
- Performance regressions (response time, memory growth)

### 32. Tests Mejorados de Playwright (`test_playwright_improved.py`)

Tests mejorados con helpers y fixtures avanzados:
- Helpers mejorados (retry, validation, schema, PDF creation)
- Fixtures avanzados (tracing, video, HAR, slow network, offline)
- Error handling mejorado
- Performance monitoring mejorado
- Network logging
- Response comparison
- Batch requests
- Performance metrics (min, max, avg, median, p95, p99)

### 33. Tests de Integración con Playwright (`test_playwright_integration.py`)

Tests de integración combinando múltiples features:
- Integración multi-feature (upload, preview, topics, variants)
- Integración de búsqueda, filtrado y ordenamiento
- Integración de operaciones por lotes
- Integración de cache (headers, invalidación)
- Integración de autenticación (token refresh, RBAC)
- Integración de webhooks (event flow, status)
- Integración de notificaciones
- Integración de export (múltiples formatos)
- Integración de búsqueda (filtros, paginación)
- Integración de versionado (compatibilidad, negociación)
- Integración de error handling (recovery chain, cascading errors)

### 34. Tests Paralelos con Playwright (`test_playwright_parallel.py`)

Tests de ejecución paralela y concurrencia:
- Ejecución paralela (health checks, uploads, variants)
- Acceso concurrente (reads, updates)
- Race conditions (upload/delete, create/update)
- Load balancing (distributed load)
- Resource contention (operaciones concurrentes)
- Concurrencia de múltiples operaciones
- Tests de escalabilidad
- Tests de throughput

### 35. Tests de Calidad con Playwright (`test_playwright_quality.py`)

Tests de calidad y mejores prácticas:
- Calidad de código (consistencia, idempotencia, statelessness)
- Calidad de datos (validación, completitud, precisión)
- Confiabilidad (disponibilidad, tolerancia a fallos)
- Mantenibilidad (versionado, deprecación, compatibilidad)
- Documentación (OpenAPI, disponibilidad)
- Estándares (HTTP methods, status codes, content types)
- Mejores prácticas (RESTful design, error handling, security)

### 36. Tests de Monitoreo con Playwright (`test_playwright_monitoring.py`)

Tests de monitoreo y observabilidad:
- Métricas (endpoint, Prometheus, health metrics)
- Logging (request, error, access logs)
- Tracing (trace ID, propagación)
- Alerting (health checks, error rate)
- Performance monitoring (response time, throughput)
- Resource monitoring (memory, connection pool)
- Business metrics (user activity, feature usage)

### 37. Tests de Cumplimiento con Playwright (`test_playwright_compliance.py`)

Tests de cumplimiento y estándares:
- GDPR compliance (data deletion, portability, access rights, privacy by design)
- Security compliance (OWASP Top 10, HTTPS enforcement, security headers)
- API compliance (REST, JSON:API, OpenAPI)
- Accessibility compliance (WCAG, ARIA)
- Performance compliance (response time, availability)
- Data compliance (encryption, validation, sanitization)

### 38. Tests de Chaos Engineering con Playwright (`test_playwright_chaos.py`)

Tests de chaos engineering para resiliencia:
- Network chaos (latency, packet loss, timeouts)
- Load chaos (sudden spikes, sustained high load)
- Error chaos (error injection, malformed requests)
- Data chaos (large payloads, rapid changes)
- Concurrency chaos (concurrent conflicts)
- Recovery chaos (graceful degradation, recovery after failure)

### 39. Tests Visuales con Playwright (`test_playwright_visual.py`)

Tests de regresión visual y screenshots:
- Screenshots de páginas (homepage, documentación)
- Screenshots responsive (mobile, tablet, desktop)
- Comparación de screenshots
- Screenshots de elementos específicos
- Screenshots en diferentes viewports
- Tests de accesibilidad visual (contraste, legibilidad)

### 40. Tests de Accesibilidad con Playwright (`test_playwright_accessibility.py`)

Tests comprehensivos de accesibilidad:
- Accesibilidad básica (título, lang, headings, alt text, labels)
- Navegación por teclado (tab navigation, shortcuts, focus indicators)
- ARIA (labels, roles, live regions)
- Screen reader (semantic HTML, skip links)
- Color y contraste (contraste, independencia de color)
- WCAG compliance
- Tests de accesibilidad en diferentes dispositivos

### 41. Tests de Contrato de API con Playwright (`test_playwright_api_contract.py`)

Tests de contrato de API:
- Contrato de endpoints (health, upload, variants, errors)
- Contrato de esquemas (consistencia, validación)
- Contrato de versiones (versionado, compatibilidad)
- Contrato de headers (requeridos, CORS)
- Validación de contratos
- Mantenimiento de contratos

### 42. Tests de Smoke con Playwright (`test_playwright_smoke.py`)

Tests de smoke (rápidos para verificación básica):
- Smoke tests básicos (health check, accesibilidad, tiempo de respuesta)
- Smoke tests de endpoints (upload, preview, variants, topics)
- Smoke tests de workflows (workflow básico)
- Smoke tests de error handling (404, 400)
- Smoke tests de performance (tiempo de respuesta, requests concurrentes)
- Tests rápidos para CI/CD

### 43. Tests de CI/CD con Playwright (`test_playwright_ci.py`)

Tests optimizados para pipelines de CI/CD:
- Tests rápidos para CI (health check, accesibilidad, tiempo de respuesta)
- Tests de rutas críticas (endpoints críticos, error handling)
- Tests paralelos para CI (health checks paralelos)
- Tests de reporting (JUnit XML, coverage)
- Configuración de timeouts para CI
- Variables de entorno para CI

### 44. Tests de Benchmarking con Playwright (`test_playwright_benchmark.py`)

Tests de benchmarking y performance:
- Benchmark de tiempo de respuesta (min, max, mean, median, p95, p99)
- Benchmark de throughput (req/s)
- Benchmark de throughput concurrente
- Benchmark de uso de memoria
- Comparación con baseline
- Detección de regresiones de performance
- Estadísticas detalladas de performance

### 45. Tests Exploratorios con Playwright (`test_playwright_exploratory.py`)

Tests exploratorios para descubrir funcionalidades:
- Exploración de endpoints disponibles
- Exploración de formatos de respuesta
- Exploración de formatos de error
- Exploración de headers
- Exploración de query parameters
- Exploración de métodos HTTP
- Exploración de estructuras de datos
- Exploración de recursos anidados

### 46. Fixtures Centralizadas (`fixtures_common.py`)

Fixtures comunes para evitar duplicación:
- `api_base_url`: URL base de API (con variables de entorno)
- `auth_headers`: Headers de autenticación estándar
- `sample_pdf`: Contenido PDF de prueba
- `test_data`: Datos de prueba estructurados
- `ci_timeout`: Timeout para CI
- `performance_thresholds`: Umbrales de performance

### 47. Clases Base (`base_playwright_test.py`)

Clases base para reducir duplicación de código:
- `BasePlaywrightTest`: Clase base con métodos comunes
- `BaseAPITest`: Helpers para tests de API (upload, variants, topics, preview)
- `BaseUITest`: Helpers para tests de UI (navegación, formularios)
- `BaseLoadTest`: Helpers para tests de carga
- `BaseSecurityTest`: Helpers para tests de seguridad

### 48. Ejemplo de Tests Refactorizados (`test_playwright_refactored_example.py`)

Ejemplo de cómo usar las clases base y fixtures centralizadas:
- Tests de API usando `BaseAPITest`
- Tests de UI usando `BaseUITest`
- Workflows completos usando helpers de base
- Mejores prácticas de refactorización

### 46. Fixtures Centralizadas (`fixtures_common.py`)

Fixtures comunes para evitar duplicación:
- `api_base_url`: URL base de API (con variables de entorno)
- `auth_headers`: Headers de autenticación estándar
- `sample_pdf`: Contenido PDF de prueba
- `test_data`: Datos de prueba estructurados
- `ci_timeout`: Timeout para CI
- `performance_thresholds`: Umbrales de performance

### 47. Clases Base (`base_playwright_test.py`)

Clases base para reducir duplicación de código:
- `BasePlaywrightTest`: Clase base con métodos comunes (assertions, helpers)
- `BaseAPITest`: Helpers para tests de API (upload, variants, topics, preview)
- `BaseUITest`: Helpers para tests de UI (navegación, formularios)
- `BaseLoadTest`: Helpers para tests de carga (concurrent requests)
- `BaseSecurityTest`: Helpers para tests de seguridad (injection, auth)

### 48. Ejemplo de Tests Refactorizados (`test_playwright_refactored_example.py`)

Ejemplo de cómo usar las clases base y fixtures centralizadas:
- Tests de API usando `BaseAPITest`
- Tests de UI usando `BaseUITest`
- Workflows completos usando helpers de base
- Mejores prácticas de refactorización
- BaseSecurityTest: Para tests de seguridad
- BaseWorkflowTest: Para tests de workflows
- Métodos helper comunes
- Configuración compartida

### 49. Page Object Model (`playwright_pages.py`)

Page Object Model pattern para mejor organización:
- `BasePage`: Clase base para páginas
- `HealthPage`: Page object para health checks
- `UploadPage`: Page object para uploads
- `VariantPage`: Page object para variantes
- `TopicPage`: Page object para topics
- `PreviewPage`: Page object para previews
- `PDFManagementPage`: Page object para gestión de PDFs
- `SearchPage`: Page object para búsqueda
- `APIPage`: Page object principal que combina todo
- Ver `POM_GUIDE.md` para más detalles

### 50. Test Runner (`playwright_test_runner.py`)

Utilidades para ejecutar y gestionar tests:
- `PlaywrightTestRunner`: Ejecutar tests y generar reportes
- `TestResult`: Estructura de datos para resultados
- `TestSuiteResult`: Estructura de datos para suite completa
- `PlaywrightTestFilter`: Filtrar tests por criterios
- `PlaywrightTestExecutor`: Ejecutor con opciones predefinidas
- Generación de reportes HTML
- Guardado de resultados en JSON
- Ver `TEST_RUNNER_GUIDE.md` para más detalles

### 51. Tests con Page Object Model (`test_playwright_pom.py`)

Tests usando POM pattern:
- Tests de health usando `HealthPage`
- Tests de upload usando `UploadPage`
- Tests de variants usando `VariantPage`
- Tests de topics usando `TopicPage`
- Tests de preview usando `PreviewPage`
- Tests de management usando `PDFManagementPage`
- Tests de search usando `SearchPage`
- Workflow completo usando `APIPage`

## Fixtures Disponibles

### Fixtures de Configuración

- `test_settings`: Configuración de testing
- `temp_dir`: Directorio temporal
- `test_config`: Configuración completa de prueba

### Fixtures de Datos

- `sample_pdf_content`: Contenido de PDF de ejemplo
- `sample_text_content`: Contenido de texto de ejemplo
- `sample_document_data`: Datos de documento de ejemplo
- `sample_variant_data`: Datos de variante de ejemplo
- `sample_user_data`: Datos de usuario de ejemplo
- `test_data`: Conjunto completo de datos de prueba

### Fixtures de Servicios

- `mock_pdf_service`: Servicio PDF mockeado
- `mock_database`: Base de datos mock
- `mock_redis`: Redis mock
- `mock_external_services`: Servicios externos mockeados

### Fixtures de API

- `app`: Aplicación FastAPI para testing
- `client`: Cliente de test para la API
- `mock_pdf_file`: Archivo PDF mock
- `mock_file_id`: ID de archivo mock
- `mock_user_id`: ID de usuario mock

## Configuración

### Variables de Entorno

Los tests configuran automáticamente las siguientes variables de entorno:

```python
ENVIRONMENT=testing
DATABASE_URL=sqlite:///test.db
REDIS_URL=redis://localhost:6379/1
ENABLE_CACHE=false
ENABLE_METRICS=false
ENABLE_ANALYTICS=false
LOG_LEVEL=WARNING
```

### Configuración de pytest

Ver `pytest.ini` para la configuración completa de pytest.

## Marcadores de Tests

Los tests pueden ser marcados con:

- `@pytest.mark.unit`: Tests unitarios rápidos
- `@pytest.mark.integration`: Tests de integración
- `@pytest.mark.slow`: Tests lentos
- `@pytest.mark.ai`: Tests que requieren IA
- `@pytest.mark.api`: Tests de API
- `@pytest.mark.services`: Tests de servicios
- `@pytest.mark.generator`: Tests del generador

## Utilidades y Refactorización

### Utilidades de Playwright (`playwright_utils.py`)

Utilidades avanzadas para tests:
- **Request Builder**: Fluent API para construir requests
  ```python
  response = (
      create_request_builder(page, api_base_url)
      .post("/pdf/upload")
      .with_headers(auth_headers)
      .with_multipart(files)
      .execute()
  )
  ```

- **Response Validator**: Validación encadenada
  ```python
  validate_response(response)
      .assert_status(200)
      .assert_has_keys("file_id")
      .assert_json()
  ```

- **Test Data Factory**: Factory para crear datos
  ```python
  factory = create_test_data()
  pdf_file = factory.create_pdf_file("test.pdf", size_kb=100)
  variant_request = factory.create_variant_request("summary")
  ```

### Decoradores de Playwright (`playwright_decorators.py`)

Decoradores para agregar funcionalidad:
- `@retry_on_failure`: Reintentos automáticos
- `@measure_performance`: Medición de performance
- `@capture_screenshot_on_failure`: Screenshots en fallos
- `@validate_response_time`: Validación de tiempo
- `@skip_if_api_unavailable`: Skip condicional
- `@log_test_execution`: Logging
- `@require_auth`: Validación de auth

### Configuración Centralizada (`playwright_config.py`)

Configuración accesible globalmente:
- `PlaywrightConfig`: Configuración general
- `APIConfig`: Configuración de API
- `PerformanceConfig`: Configuración de performance
- `SecurityConfig`: Configuración de seguridad
- `TestDataConfig`: Configuración de test data

### Page Object Model (`playwright_pages.py`)

Page Object Model pattern para mejor organización:
- `BasePage`: Clase base para páginas
- `HealthPage`: Page object para health checks
- `UploadPage`: Page object para uploads
- `VariantPage`: Page object para variantes
- `TopicPage`: Page object para topics
- `PreviewPage`: Page object para previews
- `PDFManagementPage`: Page object para gestión de PDFs
- `SearchPage`: Page object para búsqueda
- `APIPage`: Page object principal que combina todo

### Test Runner (`playwright_test_runner.py`)

Utilidades para ejecutar y gestionar tests:
- `PlaywrightTestRunner`: Ejecutar tests y generar reportes
- `TestResult`: Estructura de datos para resultados
- `TestSuiteResult`: Estructura de datos para suite completa
- `PlaywrightTestFilter`: Filtrar tests por criterios
- `PlaywrightTestExecutor`: Ejecutor con opciones predefinidas
- Generación de reportes HTML
- Guardado de resultados en JSON

### Tests con Page Object Model (`test_playwright_pom.py`)

Tests usando POM pattern:
- Tests de health usando `HealthPage`
- Tests de upload usando `UploadPage`
- Tests de variants usando `VariantPage`
- Tests de topics usando `TopicPage`
- Tests de preview usando `PreviewPage`
- Tests de management usando `PDFManagementPage`
- Tests de search usando `SearchPage`
- Workflow completo usando `APIPage`

### 52. Utilidades de Debugging (`playwright_debug.py`)

Utilidades para debugging y troubleshooting:
- `PlaywrightDebugger`: Captura de screenshots, network logs, console logs
- `PlaywrightTroubleshooter`: Diagnóstico de timeouts, performance, errores
- `create_debugger()`: Crear instancia de debugger
- `troubleshoot_timeout()`: Diagnóstico rápido de timeouts
- `troubleshoot_performance()`: Diagnóstico rápido de performance
- Captura automática de información de debug
- Análisis de performance de página
- Comparación de respuestas

### 53. Analytics y Métricas (`playwright_analytics.py`)

Utilidades para análisis y métricas:
- `PlaywrightAnalytics`: Análisis de resultados de tests
- `TestMetrics`: Métricas de un test individual
- `SuiteMetrics`: Métricas de toda la suite
- Generación de reportes JSON y HTML
- Comparación con baseline
- Identificación de tests lentos
- Identificación de tests flaky
- Generación de tendencias históricas

### 54. Script Avanzado de Ejecución (`run_tests_advanced.py`)

Script avanzado para ejecutar tests:
- Filtrado por markers
- Ejecución paralela
- Generación de coverage
- Generación de reportes HTML
- Generación de analytics
- Opciones para smoke, critical, fast tests
- Modo quiet/verbose

### 55. Tests de Debugging (`test_playwright_debug.py`)

Tests demostrando utilidades de debugging:
- Captura de screenshots
- Captura de network logs
- Captura de console logs
- Guardado de información de debug
- Análisis de performance
- Troubleshooting de timeouts
- Troubleshooting de performance

### 56. Utilidades de Comparación (`playwright_comparison.py`)

Utilidades para comparar resultados y respuestas:
- `PlaywrightComparator`: Comparación de respuestas, resultados, métricas
- `compare_responses()`: Comparar dos respuestas API
- `compare_test_results()`: Comparar resultados de tests
- `compare_performance_metrics()`: Comparar métricas de performance
- `compare_json_structures()`: Comparar estructuras JSON
- `compare_file_contents()`: Comparar contenidos de archivos
- Detección de diferencias detalladas

### 57. Sistema de Mixins (`playwright_mixins.py`)

Sistema de mixins modulares para funcionalidad reutilizable:
- `RequestMixin`: Funcionalidad de requests HTTP
- `AssertionMixin`: Assertions comunes
- `APIOperationsMixin`: Operaciones de API (upload, variants, topics, preview)
- `PerformanceMixin`: Medición de performance
- `DebuggingMixin`: Funcionalidad de debugging
- `AnalyticsMixin`: Funcionalidad de analytics
- `WorkflowMixin`: Combina Request, API Operations y Assertions
- Permite combinación flexible de funcionalidades

### 58. Clases Base Unificadas (`playwright_base_unified.py`)

Clases base que usan mixins:
- `BasePlaywrightTest`: Base con Request y Assertion
- `BaseAPITest`: Base + API Operations
- `BasePerformanceTest`: Base + Performance
- `BaseSecurityTest`: Base con métodos de seguridad
- `BaseWorkflowTest`: Base + Workflow (completo)
- `BaseDebugTest`: Base + Debugging
- `BaseAnalyticsTest`: Base + Analytics
- `BaseComprehensiveTest`: Base con TODOS los mixins
- Ver `REFACTORING_V2.md` para más detalles

### 59. Tests Refactorizados V2 (`test_playwright_refactored_v2.py`)

Tests usando el nuevo sistema de mixins:
- Tests simples usando `BasePlaywrightTest`
- Tests de API usando `BaseAPITest`
- Tests de performance usando `BasePerformanceTest`
- Tests de workflow usando `BaseWorkflowTest`
- Tests con debugging usando `BaseDebugTest`
- Tests comprehensivos usando `BaseComprehensiveTest`

### 60. Herramientas de Debugging de API

Scripts y herramientas para debugging:
- `run_api_debug.py`: Ejecutar API con debugging habilitado
- `debug_api.py`: Herramienta interactiva de debugging
- `run_with_debug.sh`: Script bash para ejecutar con debug
- `run_with_debug.ps1`: Script PowerShell para ejecutar con debug
- `debug_api_test.py`: Tests para herramientas de debugging
- `DEBUG_GUIDE.md`: Guía completa de debugging

**Funcionalidades:**
- Modo debug con logging detallado
- Herramienta interactiva para probar endpoints
- Historial de requests
- Guardado de historial para análisis
- Integración con tests

## Mejores Prácticas

1. **Aislamiento**: Cada test debe ser independiente
2. **Mocking**: Usar mocks para servicios externos
3. **Fixtures**: Reutilizar fixtures para datos comunes (usar `fixtures_common.py`)
4. **Clases Base**: Usar clases base para reducir duplicación
5. **Utilidades**: Usar Request Builder y Response Validator
6. **Decoradores**: Usar decoradores para funcionalidad común
7. **Configuración**: Usar configuración centralizada
8. **Nombres descriptivos**: Usar nombres claros para tests
9. **Documentación**: Documentar qué prueba cada test
10. **Assertions claras**: Usar assertions específicas

## Troubleshooting

### Tests fallan por imports

Asegúrate de que todas las dependencias estén instaladas:

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-mock pytest-cov
```

### Tests fallan por servicios externos

Los tests usan mocks por defecto. Si necesitas servicios reales, configura las variables de entorno apropiadas.

### Tests lentos

Usa marcadores para excluir tests lentos:

```bash
pytest -m "not slow"
```

## Contribuir

Al agregar nuevos tests:

1. Sigue la estructura existente
2. Usa fixtures apropiadas
3. Agrega documentación
4. Marca los tests apropiadamente
5. Asegúrate de que pasen todos los tests

## Cobertura de Código

El objetivo es mantener una cobertura de código superior al 80%. Ejecuta:

```bash
pytest --cov=. --cov-report=html
```

Y revisa el reporte en `htmlcov/index.html`.


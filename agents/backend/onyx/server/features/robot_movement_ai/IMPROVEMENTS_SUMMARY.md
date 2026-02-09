# Robot Movement AI - Mejoras Implementadas

## 📋 Resumen

Este documento describe las mejoras implementadas en el sistema Robot Movement AI para mejorar la calidad del código, manejo de errores, validación y robustez general.

## ✅ Mejoras Implementadas

### 1. Sistema de Excepciones Mejorado

#### Cambios Realizados:
- **Nueva clase base `BaseRobotException`**: Todas las excepciones ahora heredan de esta clase base que incluye:
  - `error_code`: Código único de error para identificación
  - `details`: Diccionario con información adicional del error
  - `cause`: Excepción original que causó el error
  - `traceback`: Stack trace completo
  - `to_dict()`: Método para serialización JSON

#### Nuevas Excepciones de Seguridad:
- `SafetyError`: Base para errores de seguridad
- `CollisionDetectedError`: Cuando se detecta una colisión
- `SafetyLimitExceededError`: Cuando se excede un límite de seguridad
- `EmergencyStopError`: Cuando se activa parada de emergencia

#### Beneficios:
- Mejor debugging con contexto completo
- Serialización JSON para APIs
- Trazabilidad completa de errores
- Códigos de error únicos para integración

### 2. Manejo de Errores en API

#### Cambios Realizados:
- **Global Exception Handlers**: Manejadores globales para todas las excepciones
  - `BaseRobotException`: Maneja todas las excepciones del sistema
  - `Exception`: Captura excepciones no manejadas
- **Status Codes Apropiados**:
  - `503`: Robot no conectado/inicializado
  - `409`: Movimiento en progreso
  - `400`: Errores de validación/trayectoria
  - `423`: Errores de seguridad (locked)
  - `500`: Errores internos

#### Respuestas de Error Mejoradas:
```json
{
  "success": false,
  "error": {
    "error_type": "ValidationError",
    "error_code": "INVALID_POSITION",
    "message": "Position coordinates must be between -10.0 and 10.0 meters",
    "details": {"x": 15.0, "y": 5.0, "z": 2.0},
    "traceback": "..."
  },
  "timestamp": 1234567890.123
}
```

### 3. Validación de Entrada

#### Validación en Pydantic Models:
- **MoveToRequest**:
  - Coordenadas entre -10.0 y 10.0 metros
  - Validación de quaternion (4 elementos, normalizado)
- **ChatMessage**:
  - Longitud mínima: 1 carácter
  - Longitud máxima: 10,000 caracteres
  - Trim automático de espacios
- **PathRequest**:
  - Mínimo 2 waypoints
  - Máximo 100 waypoints
- **ObstaclesRequest**:
  - Formato validado: [min_x, min_y, min_z, max_x, max_y, max_z]
  - Validación de límites (max >= min)

#### Nuevo Módulo de Validación:
- `utils/input_validators.py`: Funciones de validación reutilizables
  - `validate_position()`: Validar coordenadas
  - `validate_quaternion()`: Validar y normalizar quaternions
  - `validate_waypoints()`: Validar lista de waypoints
  - `validate_message()`: Validar mensajes de chat
  - `validate_obstacles()`: Validar obstáculos

### 4. Mejoras en Endpoints

#### `/api/v1/move/to`:
- ✅ Validación de coordenadas antes de procesar
- ✅ Validación de quaternion con normalización
- ✅ Manejo de errores mejorado con contexto
- ✅ Timestamp en respuesta

#### `/api/v1/chat`:
- ✅ Validación de mensaje vacío
- ✅ Validación de longitud máxima
- ✅ Preview del mensaje en errores
- ✅ Manejo robusto de excepciones

## 📊 Impacto de las Mejoras

### Antes:
- Excepciones genéricas sin contexto
- Errores difíciles de debuggear
- Validación inconsistente
- Respuestas de error poco informativas

### Después:
- ✅ Excepciones con contexto completo
- ✅ Debugging más fácil con códigos de error
- ✅ Validación consistente en todos los endpoints
- ✅ Respuestas de error informativas y estructuradas
- ✅ Mejor experiencia para desarrolladores que integran la API

## 🔄 Compatibilidad

Todas las mejoras son **backward compatible**:
- Las excepciones existentes siguen funcionando
- Los endpoints mantienen la misma interfaz
- Solo se agrega información adicional, no se elimina funcionalidad

## ✅ Mejoras Adicionales Implementadas

### 5. Logging Estructurado Mejorado

#### Cambios Realizados:
- **Context Variables**: Uso de `ContextVar` para request ID y operation name
- **Correlación de Logs**: Todos los logs incluyen request ID automáticamente
- **Contexto Global**: Sistema de contexto global para logs
- **Stack Traces**: Opción para incluir stack traces en errores
- **Métricas de Performance**: Métodos especializados para log de performance
- **Decoradores**: `@with_request_id` y `@with_operation_name` para contexto automático

#### Beneficios:
- Trazabilidad completa de requests
- Correlación de logs entre servicios
- Debugging más fácil con request IDs
- Métricas de performance integradas

### 6. Middleware de API

#### Nuevos Middlewares:
- **RequestLoggingMiddleware**: 
  - Logging automático de todos los requests
  - Request ID en headers de respuesta
  - Métricas de duración
  - Logging de errores con contexto
  
- **PerformanceMonitoringMiddleware**:
  - Detección de requests lentos
  - Estadísticas de performance (avg, min, max, p95, p99)
  - Headers de tiempo de respuesta
  - Métricas agregadas

#### Headers Agregados:
- `X-Request-ID`: ID único del request
- `X-Response-Time`: Tiempo de respuesta en ms

### 7. Sistema de Caché

#### Nuevo Módulo `utils/cache.py`:
- **TTLCache**: Cache thread-safe con TTL
- **LRU Eviction**: Eliminación automática de entradas menos usadas
- **Decoradores**: `@cached` y `@async_cached` para funciones
- **Estadísticas**: Métricas de uso del cache
- **Limpieza Automática**: Limpieza de entradas expiradas

#### Características:
- Thread-safe para uso concurrente
- TTL configurable por entrada
- Límite de tamaño con evicción LRU
- Estadísticas de hit/miss rate
- Soporte para funciones sync y async

## 📝 Próximos Pasos Sugeridos

1. **Type Hints Completos**: Agregar type hints a todos los módulos core
2. **Performance**: Optimizar rutas críticas identificadas
3. **Testing**: Agregar tests para validaciones y manejo de errores
4. **Documentación API**: Actualizar documentación OpenAPI con ejemplos de errores
5. **Redis Cache**: Integrar Redis para cache distribuido

### 8. Middleware de Seguridad

#### Nuevos Middlewares de Seguridad:
- **RateLimitMiddleware**: 
  - Rate limiting por minuto y por hora
  - Límites configurables
  - Headers informativos (X-RateLimit-*)
  - Limpieza automática de registros antiguos
  - Soporte para burst size
  
- **SecurityHeadersMiddleware**:
  - Headers de seguridad estándar
  - X-Content-Type-Options: nosniff
  - X-Frame-Options: DENY
  - X-XSS-Protection
  - Strict-Transport-Security
  - Content-Security-Policy
  
- **RequestSizeLimitMiddleware**:
  - Validación de tamaño de request
  - Límite configurable (default: 10MB)
  - Prevención de DoS por requests grandes

#### Headers de Seguridad Agregados:
- `X-RateLimit-Limit-Minute`: Límite por minuto
- `X-RateLimit-Remaining-Minute`: Requests restantes por minuto
- `X-RateLimit-Limit-Hour`: Límite por hora
- `X-RateLimit-Remaining-Hour`: Requests restantes por hora
- Headers de seguridad estándar

### 9. Configuración Mejorada

#### Mejoras en RobotConfig:
- **to_dict()**: Método mejorado con todos los campos
- **from_dict()**: Crear configuración desde diccionario
- **update()**: Actualizar configuración inmutably
- Validación mejorada de todos los campos

#### Beneficios:
- Configuración más flexible
- Serialización/deserialización completa
- Actualización inmutable de configuración
- Mejor integración con sistemas externos

### 10. Utilidades de Performance

#### Nuevo Módulo `utils/performance.py`:
- **@timeit**: Decorador para medir tiempo de ejecución
- **timer()**: Context manager para medir operaciones
- **PerformanceMonitor**: Monitor de performance con estadísticas
  - Promedio, min, max
  - Percentiles (p50, p95, p99)
  - Conteo de operaciones
  - Estadísticas agregadas

#### Características:
- Soporte para funciones sync y async
- Logging automático de tiempos
- Estadísticas detalladas
- Monitor global reutilizable

### 11. Documentación OpenAPI Mejorada

#### Mejoras en FastAPI App:
- Tags organizados por categoría
- URLs de documentación configuradas
- OpenAPI schema mejorado
- Mejor organización de endpoints

### 12. Dependency Injection System

#### Nuevo Módulo `utils/dependency_injection.py`:
- **DependencyContainer**: Contenedor de dependencias
  - Registro de servicios (instancias, factories)
  - Soporte para singletons
  - Resolución automática de dependencias
  
- **@inject**: Decorador para inyección automática
  - Soporte para funciones sync y async
  - Resolución automática basada en type hints
  - Integración con contenedor global

#### Beneficios:
- Facilita testing con mocks
- Reduce acoplamiento entre módulos
- Mejora modularidad del código
- Permite intercambiar implementaciones fácilmente

### 13. Retry Utilities Mejoradas

#### Nuevo Módulo `utils/retry.py`:
- **@retry**: Decorador para reintentos automáticos
  - Múltiples estrategias (fixed, exponential, linear)
  - Configuración flexible
  - Callbacks personalizados
  - Soporte para excepciones específicas
  
- **RetryConfig**: Configuración de reintentos
  - Max attempts, delays, multipliers
  - Estrategias configurables
  
- **retry_async()**: Función helper para reintentos async

#### Características:
- Exponential backoff por defecto
- Logging automático de reintentos
- Soporte para sync y async
- Callbacks personalizables

### 14. Circuit Breaker Pattern

#### Nuevo Módulo `utils/circuit_breaker.py`:
- **CircuitBreaker**: Implementación del patrón circuit breaker
  - Estados: CLOSED, OPEN, HALF_OPEN
  - Thresholds configurables
  - Timeouts automáticos
  - Estadísticas detalladas
  
- **@circuit_breaker**: Decorador para usar circuit breakers
- **CircuitBreakerManager**: Gestor global de circuit breakers

#### Características:
- Protección automática de servicios externos
- Recuperación automática (half-open state)
- Estadísticas y monitoreo
- Integración fácil con decoradores

### 15. Async Utilities

#### Nuevo Módulo `utils/async_utils.py`:
- **gather_with_limit()**: Ejecutar coroutines con límite de concurrencia
- **timeout_after()**: Ejecutar coroutines con timeout
- **async_to_sync()**: Convertir funciones async a sync
- **sync_to_async()**: Convertir funciones sync a async
- **AsyncRateLimiter**: Rate limiter asíncrono con context manager

#### Beneficios:
- Control de concurrencia
- Timeouts automáticos
- Conversión entre sync/async
- Rate limiting asíncrono

### 16. Helpers Mejorados

#### Mejoras en `utils/helpers.py`:
- **deep_get()**: Obtener valores anidados de diccionarios usando paths
- **deep_set()**: Establecer valores anidados en diccionarios
- **hash_string()**: Generar hash de strings
- **hash_dict()**: Generar hash de diccionarios
- **retry_on_exception()**: Decorador para reintentos
- **batch_process()**: Procesar items en batches
- **remove_none_values()**: Limpiar diccionarios
- **merge_dicts()**: Fusionar múltiples diccionarios
- **get_nested_keys()**: Obtener todas las keys anidadas
- **flatten()**: Aplanar listas anidadas

#### Beneficios:
- Funciones helper más completas
- Mejor manejo de estructuras de datos anidadas
- Utilidades de hash y serialización
- Procesamiento en batches

### 17. Sistema de Métricas Mejorado

#### Mejoras en `tracing/metrics.py`:
- **MetricData**: Dataclass para datos de métricas
- **MetricSummary**: Resumen estadístico con:
  - Count, sum, avg, min, max
  - Median, p95, p99
  - Standard deviation
  - Tags y metadata
  
- **Métodos mejorados**:
  - `get_summary()`: Resumen estadístico de métricas
  - `get_all_summaries()`: Resúmenes de todas las métricas
  - `get_metrics_in_range()`: Métricas en rango de tiempo
  - `clear_metric()`: Limpiar métricas específicas
  - Límite de métricas por nombre (evita memory leaks)

#### Beneficios:
- Estadísticas detalladas automáticas
- Mejor análisis de performance
- Prevención de memory leaks
- Métricas con timestamps y tags

### 18. Utilidades de Serialización

#### Nuevo Módulo `utils/serialization.py`:
- **JSONEncoder**: Encoder extendido para JSON
  - Soporte para datetime, date, Enum
  - Serialización automática de objetos
  
- **to_json() / from_json()**: Serialización JSON mejorada
- **to_pickle() / from_pickle()**: Serialización pickle
- **to_base64() / from_base64()**: Codificación base64
- **serialize_dict() / deserialize_dict()**: Serialización de diccionarios
- **safe_json_loads()**: Carga segura de JSON con defaults

#### Características:
- Soporte para múltiples formatos
- Serialización segura con manejo de errores
- Soporte para tipos complejos
- Codificación base64 para transferencia

### 19. Formatters Mejorados

#### Mejoras en `utils/formatters.py`:
- **format()**: Método principal mejorado con múltiples formatos
- **to_json()**: Conversión a JSON con opciones
- **to_yaml()**: Conversión a YAML
- **to_xml()**: Conversión a XML
- **to_csv()**: Conversión a CSV con headers
- **to_table()**: Conversión a tabla formateada
- **format_number()**: Formateo de números con separadores
- **format_bytes()**: Formateo de bytes a formato legible (KB, MB, GB)
- **format_duration()**: Formateo de duraciones (h, m, s)

#### Características:
- Soporte para múltiples formatos de salida
- Formateo de números y bytes
- Tablas formateadas con ancho configurable
- Manejo de tipos complejos

### 20. Utilidades de DateTime

#### Nuevo Módulo `utils/datetime_utils.py`:
- **now()**: Obtener fecha/hora actual con timezone
- **parse_datetime()**: Parsear strings a datetime con múltiples formatos
- **format_datetime()**: Formatear datetime a string
- **to_timestamp() / from_timestamp()**: Conversión timestamp
- **add_time()**: Agregar tiempo a datetime
- **time_ago()**: String "hace X tiempo"
- **is_business_day()**: Verificar día laboral
- **next_business_day()**: Obtener siguiente día laboral
- **time_range()**: Generar rango de fechas

#### Beneficios:
- Manejo robusto de fechas y horas
- Parsing automático de múltiples formatos
- Utilidades para días laborales
- Formateo legible de tiempos

### 21. Validators Mejorados

#### Mejoras en `utils/validators.py`:
- **validate()**: Validación mejorada contra esquemas
- **validate_email()**: Validación de email con regex
- **validate_url()**: Validación de URL con esquemas configurables
- **validate_ip()**: Validación de IP (IPv4/IPv6)
- **validate_phone()**: Validación de teléfono
- **validate_regex()**: Validación contra regex
- **validate_length()**: Validación de longitud de texto
- **validate_range()**: Validación de rango numérico
- **validate_required()**: Validación de campos requeridos

#### Beneficios:
- Validación robusta de múltiples tipos de datos
- Validación de formatos comunes (email, URL, IP, teléfono)
- Validación de esquemas complejos
- Reutilizable en toda la aplicación

### 22. Utilidades de Archivos

#### Nuevo Módulo `utils/file_utils.py`:
- **ensure_dir()**: Asegurar que directorio existe
- **safe_remove()**: Eliminar archivo/directorio de forma segura
- **get_file_hash()**: Calcular hash de archivo
- **get_file_size()**: Obtener tamaño de archivo
- **list_files()**: Listar archivos con patrones
- **copy_file()**: Copiar archivo
- **move_file()**: Mover archivo
- **read_text_file()**: Leer archivo de texto
- **write_text_file()**: Escribir archivo de texto
- **get_file_extension()**: Obtener extensión
- **get_file_name()**: Obtener nombre de archivo

#### Características:
- Operaciones seguras con manejo de errores
- Soporte para Path y strings
- Operaciones de hash para verificación
- Utilidades para manipulación de archivos

### 23. EnvLoader Mejorado

#### Mejoras en `configs/env_loader.py`:
- **load()**: Carga mejorada con prefijos y case sensitivity
- **get()**: Obtener variable con parsing automático y tipos
- **get_bool()**: Obtener variable booleana
- **get_int()**: Obtener variable entera
- **get_float()**: Obtener variable flotante
- **get_list()**: Obtener variable como lista
- **load_from_file()**: Cargar desde archivo .env
- **set()**: Establecer variable
- **unset()**: Eliminar variable
- **has()**: Verificar existencia

#### Características:
- Parsing automático de tipos
- Soporte para archivos .env
- Validación de variables requeridas
- Case sensitivity configurable

### 24. Utilidades de Entorno

#### Nuevo Módulo `utils/env_utils.py`:
- **get_env()**: Obtener variable con validación
- **get_env_bool()**: Obtener booleano
- **get_env_int()**: Obtener entero
- **get_env_float()**: Obtener flotante
- **get_env_list()**: Obtener lista
- **set_env()**: Establecer variable
- **unset_env()**: Eliminar variable
- **has_env()**: Verificar existencia
- **load_env_file()**: Cargar desde .env
- **get_env_prefix()**: Obtener variables con prefijo

#### Beneficios:
- Funciones helper para variables de entorno
- Parsing automático de tipos
- Validación de variables requeridas
- Carga desde archivos .env

### 25. Utilidades de Strings

#### Nuevo Módulo `utils/string_utils.py`:
- **slugify()**: Convertir texto a slug
- **camel_to_snake()**: Convertir camelCase a snake_case
- **snake_to_camel()**: Convertir snake_case a camelCase
- **truncate()**: Truncar texto a longitud máxima
- **remove_whitespace()**: Remover espacios múltiples
- **extract_emails()**: Extraer emails de texto
- **extract_urls()**: Extraer URLs de texto
- **mask_sensitive()**: Enmascarar texto sensible
- **normalize_whitespace()**: Normalizar espacios
- **word_count()**: Contar palabras
- **contains_any()**: Verificar si contiene alguna keyword
- **contains_all()**: Verificar si contiene todas las keywords
- **sanitize_filename()**: Sanitizar nombre de archivo
- **url_encode() / url_decode()**: Codificación/decodificación URL
- **split_camel_case()**: Dividir camelCase en palabras
- **pluralize()**: Pluralizar palabras

#### Beneficios:
- Manipulación robusta de strings
- Conversión entre formatos (camelCase, snake_case)
- Extracción de datos (emails, URLs)
- Sanitización y seguridad

### 26. Utilidades de Decoradores

#### Nuevo Módulo `utils/decorators.py`:
- **@memoize**: Memoización simple
- **@throttle**: Throttling (limitar frecuencia)
- **@debounce**: Debounce (esperar antes de ejecutar)
- **@singleton**: Clase singleton
- **@deprecated**: Marcar funciones como deprecated
- **@retry_on_failure**: Reintentar en caso de fallo
- **@timeout**: Timeout en funciones
- **@log_calls**: Loggear llamadas a funciones
- **@validate_args**: Validar argumentos
- **@cache_result**: Cachear resultado con TTL

#### Características:
- Decoradores para funciones sync y async
- Memoización y caching
- Control de frecuencia (throttle/debounce)
- Validación y logging automático

### 27. Utilidades Matemáticas y Geométricas

#### Nuevo Módulo `utils/math_utils.py`:
- **Distancias**:
  - `euclidean_distance()`: Distancia euclidiana
  - `manhattan_distance()`: Distancia Manhattan
  
- **Vectores**:
  - `normalize_vector()`: Normalizar vector
  - `dot_product()`: Producto punto
  - `cross_product()`: Producto cruz
  - `angle_between_vectors()`: Ángulo entre vectores
  
- **Ángulos**:
  - `radians_to_degrees()`: Convertir radianes a grados
  - `degrees_to_radians()`: Convertir grados a radianes
  - `normalize_angle()`: Normalizar ángulo
  
- **Quaternions**:
  - `quaternion_multiply()`: Multiplicar quaternions
  - `quaternion_conjugate()`: Conjugado de quaternion
  - `quaternion_normalize()`: Normalizar quaternion
  - `quaternion_to_euler()`: Convertir a ángulos de Euler
  - `euler_to_quaternion()`: Convertir de Euler a quaternion
  - `rotate_vector_by_quaternion()`: Rotar vector por quaternion
  
- **Geometría**:
  - `point_in_circle()`: Verificar punto en círculo
  - `point_in_sphere()`: Verificar punto en esfera
  - `point_in_box()`: Verificar punto en bounding box
  
- **Interpolación**:
  - `lerp()`: Interpolación lineal
  - `slerp()`: Interpolación esférica (SLERP) para quaternions
  
- **Utilidades**:
  - `clamp()`: Limitar valor a rango
  - `smooth_step()`: Función smooth step

#### Beneficios:
- Cálculos geométricos precisos para robots
- Manejo robusto de quaternions y rotaciones
- Utilidades de interpolación para trayectorias suaves
- Detección de colisiones y geometría espacial

### 28. Utilidades de Colecciones Avanzadas

#### Nuevo Módulo `utils/collections_utils.py`:
- **Agrupación y Partición**:
  - `group_by()`: Agrupar items por función clave
  - `partition()`: Dividir lista según predicado
  - `count_by()`: Contar items por clave
  
- **Búsqueda y Filtrado**:
  - `find()`: Encontrar primer item que cumple predicado
  - `find_all()`: Encontrar todos los items que cumplen
  - `filter_dict()`: Filtrar diccionario por predicado
  
- **Transformación**:
  - `unique()`: Obtener items únicos preservando orden
  - `sort_by()`: Ordenar por función clave
  - `map_dict()`: Aplicar función a diccionario
  - `invert_dict()`: Invertir diccionario
  
- **Fusión y Diferencias**:
  - `deep_merge()`: Fusionar diccionarios recursivamente
  - `dict_diff()`: Calcular diferencia entre diccionarios
  - `set_operations()`: Operaciones de sets (union, intersection, etc.)
  
- **Conversión**:
  - `dict_to_list()`: Convertir diccionario a lista
  - `list_to_dict()`: Convertir lista a diccionario
  - `flatten_dict()`: Aplanar diccionario anidado
  - `unflatten_dict()`: Desaplanar diccionario
  
- **Iteración**:
  - `chunk_by()`: Dividir en chunks (generador)
  - `zip_dicts()`: Hacer zip de múltiples diccionarios
  - `take()` / `drop()`: Tomar/omitir items
  - `take_while()` / `drop_while()`: Tomar/omitir mientras se cumple condición
  
- **Reducción**:
  - `reduce_dict()`: Reducir diccionario
  - `batch_map()`: Aplicar función en batches

#### Beneficios:
- Manipulación avanzada de colecciones
- Operaciones funcionales sobre datos
- Transformaciones eficientes de estructuras
- Utilidades para procesamiento de datos

### 29. Testing Infrastructure Mejorada

#### Nuevos Tests:
- **test_api.py**: Tests de integración para endpoints de API
  - Tests para `/api/v1/move/to`
  - Tests para `/api/v1/chat`
  - Tests para `/api/v1/move/path`
  - Tests para health checks
  - Tests para manejo de errores
  
- **test_validators.py**: Tests para validadores
  - Tests de validación de posición
  - Tests de validación de quaternions
  - Tests de validación de waypoints
  - Tests de validación de mensajes
  - Tests de validación de obstáculos
  
- **test_exceptions.py**: Tests para sistema de excepciones
  - Tests de BaseRobotException
  - Tests de excepciones de movimiento
  - Tests de excepciones de seguridad
  - Tests de ValidationError

#### Fixtures Mejoradas:
- `api_client`: Cliente de test para API
- `async_api_client`: Cliente async para tests
- `mock_robot_connection`: Mock de conexión de robot
- `sample_move_request`: Request de movimiento de prueba
- `sample_chat_message`: Mensaje de chat de prueba
- `sample_path_request`: Request de path de prueba

#### Beneficios:
- Cobertura de tests mejorada
- Tests automatizados para validación
- Fixtures reutilizables
- Tests de integración para API

## 🎯 Archivos Modificados/Creados

1. `core/exceptions.py` - Sistema de excepciones mejorado
2. `api/robot_api.py` - Manejo de errores, validación y documentación mejorada
3. `utils/input_validators.py` - Nuevo módulo de validación
4. `tracing/logger.py` - Logger estructurado mejorado
5. `api/middleware.py` - Nuevo módulo de middleware
6. `api/security_middleware.py` - Middleware de seguridad
7. `utils/cache.py` - Nuevo sistema de caché
8. `utils/performance.py` - Utilidades de performance
9. `config/robot_config.py` - Configuración mejorada
10. `utils/dependency_injection.py` - Sistema de dependency injection
11. `utils/retry.py` - Utilidades de reintento
12. `utils/circuit_breaker.py` - Circuit breaker pattern
13. `utils/async_utils.py` - Utilidades asíncronas
14. `utils/helpers.py` - Helpers mejorados
15. `utils/serialization.py` - Utilidades de serialización
16. `utils/formatters.py` - Formatters mejorados
17. `utils/datetime_utils.py` - Utilidades de fecha y hora
18. `utils/validators.py` - Validators mejorados
19. `utils/file_utils.py` - Utilidades de archivos
20. `configs/env_loader.py` - EnvLoader mejorado
21. `utils/env_utils.py` - Utilidades de entorno
22. `utils/string_utils.py` - Utilidades de strings
23. `utils/decorators.py` - Utilidades de decoradores
24. `utils/math_utils.py` - Utilidades matemáticas y geométricas
25. `utils/collections_utils.py` - Utilidades de colecciones avanzadas
26. `tracing/metrics.py` - Sistema de métricas mejorado
27. `tests/conftest.py` - Fixtures mejoradas
28. `tests/test_api.py` - Tests de integración para API
29. `tests/test_validators.py` - Tests para validadores
30. `tests/test_exceptions.py` - Tests para excepciones

## 📚 Referencias

- [FastAPI Exception Handling](https://fastapi.tiangolo.com/tutorial/handling-errors/)
- [Pydantic Validation](https://docs.pydantic.dev/latest/concepts/validators/)
- [Python Exception Best Practices](https://docs.python.org/3/tutorial/errors.html)


# Guía de Utilidades Avanzadas - MCP Server

## Resumen

Utilidades avanzadas para operaciones comunes, decoradores útiles,
y funciones de optimización.

## Decoradores

### `@retry_on_failure`

Reintentar función en caso de fallo.

```python
from mcp_server.utils.advanced_utils import retry_on_failure

@retry_on_failure(max_attempts=3, delay=1.0, backoff=2.0)
def risky_operation():
    # Operación que puede fallar
    ...
```

**Parámetros:**
- `max_attempts`: Número máximo de intentos (default: 3)
- `delay`: Delay inicial en segundos (default: 1.0)
- `backoff`: Factor de backoff exponencial (default: 2.0)
- `exceptions`: Tupla de excepciones a capturar (default: Exception)
- `on_failure`: Función a llamar en cada fallo (opcional)

### `@timed_operation`

Medir tiempo de ejecución de una operación.

```python
from mcp_server.utils.advanced_utils import timed_operation

@timed_operation("database_query")
def query_database():
    # Operación a medir
    ...
```

**Parámetros:**
- `operation_name`: Nombre de la operación (default: nombre de función)

### `@cache_result`

Cachear resultados de función.

```python
from mcp_server.utils.advanced_utils import cache_result

@cache_result(ttl=3600, max_size=128)
def expensive_computation(x):
    # Cálculo costoso
    ...
```

**Parámetros:**
- `ttl`: Time to live en segundos (None = sin expiración)
- `max_size`: Tamaño máximo del cache (default: 128)

## Context Managers

### `performance_context`

Medir performance de un bloque de código.

```python
from mcp_server.utils.advanced_utils import performance_context

with performance_context("data_processing"):
    process_data()
    transform_data()
```

## Funciones de Utilidad

### `safe_execute`

Ejecutar función de forma segura con manejo de errores.

```python
from mcp_server.utils.advanced_utils import safe_execute

result = safe_execute(
    risky_function,
    arg1, arg2,
    default="error",
    on_error=lambda e: f"Error: {e}"
)
```

**Parámetros:**
- `func`: Función a ejecutar
- `*args`: Argumentos posicionales
- `default`: Valor por defecto si falla
- `on_error`: Función a llamar en caso de error
- `**kwargs`: Argumentos nombrados

### `batch_process`

Procesar items en lotes.

```python
from mcp_server.utils.advanced_utils import batch_process

results = batch_process(
    items,
    batch_size=50,
    processor=process_item
)
```

### `merge_dicts`

Fusionar múltiples diccionarios.

```python
from mcp_server.utils.advanced_utils import merge_dicts

merged = merge_dicts(dict1, dict2, dict3, deep=True)
```

**Parámetros:**
- `*dicts`: Diccionarios a fusionar
- `deep`: Si True, hace merge profundo (default: True)

### `flatten_dict`

Aplanar diccionario anidado.

```python
from mcp_server.utils.advanced_utils import flatten_dict

flat = flatten_dict({'a': {'b': 1, 'c': 2}}, separator='_')
# {'a_b': 1, 'a_c': 2}
```

### `group_by`

Agrupar items por función de clave.

```python
from mcp_server.utils.advanced_utils import group_by

grouped = group_by(users, key_func=lambda u: u.age)
```

### `chunk_list`

Dividir lista en chunks.

```python
from mcp_server.utils.advanced_utils import chunk_list

chunks = chunk_list(range(10), chunk_size=3)
# [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
```

### `format_bytes`

Formatear bytes en formato legible.

```python
from mcp_server.utils.advanced_utils import format_bytes

format_bytes(1024)      # "1.0 KB"
format_bytes(1048576)   # "1.0 MB"
```

### `format_duration`

Formatear duración en formato legible.

```python
from mcp_server.utils.advanced_utils import format_duration

format_duration(3661)   # "1h 1m 1s"
format_duration(0.5)    # "500ms"
```

### `validate_not_none`

Validar que un valor no sea None.

```python
from mcp_server.utils.advanced_utils import validate_not_none

validate_not_none(value, name="config")
```

### `validate_not_empty`

Validar que un valor no esté vacío.

```python
from mcp_server.utils.advanced_utils import validate_not_empty

validate_not_empty(value, name="data")
```

## Rate Limiter

### `RateLimiter`

Rate limiter simple basado en tiempo.

```python
from mcp_server.utils.advanced_utils import RateLimiter

limiter = RateLimiter(max_calls=10, time_window=60.0)

if limiter.is_allowed():
    make_api_call()
else:
    wait_time = limiter.wait_time()
    print(f"Rate limit exceeded. Wait {wait_time}s")
```

**Métodos:**
- `is_allowed()`: Verificar si se permite la llamada
- `wait_time()`: Obtener tiempo de espera hasta la próxima llamada

## Ejemplos de Uso

### Operación con Retry

```python
from mcp_server.utils.advanced_utils import retry_on_failure

@retry_on_failure(max_attempts=3, delay=1.0)
def fetch_data():
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
```

### Operación con Cache

```python
from mcp_server.utils.advanced_utils import cache_result

@cache_result(ttl=3600)
def get_config():
    # Cargar configuración costosa
    ...
```

### Medición de Performance

```python
from mcp_server.utils.advanced_utils import timed_operation, performance_context

@timed_operation("data_processing")
def process_data():
    with performance_context("step1"):
        step1()
    with performance_context("step2"):
        step2()
```

### Procesamiento en Lotes

```python
from mcp_server.utils.advanced_utils import batch_process

def process_item(item):
    # Procesar item individual
    ...

results = batch_process(items, batch_size=100, processor=process_item)
```

### Rate Limiting

```python
from mcp_server.utils.advanced_utils import RateLimiter

limiter = RateLimiter(max_calls=100, time_window=60.0)

for request in requests:
    if limiter.is_allowed():
        process_request(request)
    else:
        time.sleep(limiter.wait_time())
```

## Próximos Pasos

1. Agregar más decoradores útiles
2. Agregar más funciones de utilidad
3. Optimizar funciones existentes
4. Agregar tests unitarios
5. Documentar patrones de uso


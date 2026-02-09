# Referencia Completa de Utilidades

Referencia rápida de todas las utilidades disponibles.

## Decoradores Avanzados

### `@timeout(seconds)`
Agrega timeout a funciones async.

```python
from utils.decorators_advanced import timeout

@timeout(5.0)
async def slow_operation():
    # Se cancela después de 5 segundos
    pass
```

### `@cache_result(ttl_seconds, max_size)`
Cachea resultados de funciones.

```python
from utils.decorators_advanced import cache_result

@cache_result(ttl_seconds=300, max_size=128)
async def expensive_operation(arg):
    # Resultado cacheado por 5 minutos
    pass
```

### `@throttle(calls, period)`
Limita número de llamadas en un período.

```python
from utils.decorators_advanced import throttle

@throttle(calls=10, period=60.0)
async def api_call():
    # Máximo 10 llamadas por minuto
    pass
```

### `@track_performance(metric_name)`
Rastrea performance de funciones.

```python
from utils.decorators_advanced import track_performance

@track_performance("my_function")
async def my_function():
    # Métricas automáticas
    pass
```

### `@validate_input(validator)`
Valida inputs de funciones.

```python
from utils.decorators_advanced import validate_input

def validate_args(x, y):
    if x < 0 or y < 0:
        raise ValueError("Must be positive")

@validate_input(validate_args)
async def operation(x, y):
    pass
```

### `@log_execution(log_args, log_result)`
Registra ejecución de funciones.

```python
from utils.decorators_advanced import log_execution

@log_execution(log_args=True, log_result=False)
async def important_function(arg1, arg2):
    pass
```

### `@singleton`
Convierte clase en singleton.

```python
from utils.decorators_advanced import singleton

@singleton
class MyService:
    pass

# Siempre retorna la misma instancia
```

### `@memoize(ttl_seconds)`
Memoiza resultados (síncrono).

```python
from utils.decorators_advanced import memoize

@memoize(ttl_seconds=60)
def expensive_calculation(x):
    # Resultado memoizado
    pass
```

## Utilidades de Seguridad

### Generación de Tokens

```python
from utils.security_utils import generate_token, generate_api_key

token = generate_token(length=32)
api_key = generate_api_key(prefix="psk")
```

### Hash de Passwords

```python
from utils.security_utils import hash_password, verify_password

hashed, salt = hash_password("mypassword")
is_valid = verify_password("mypassword", hashed, salt)
```

### Sanitización

```python
from utils.security_utils import (
    sanitize_filename,
    generate_secure_filename,
    mask_sensitive_data
)

safe_name = sanitize_filename("../../../etc/passwd")
secure_name = generate_secure_filename("image.jpg")
masked = mask_sensitive_data("sk_live_1234567890abcdef")
```

### Validación

```python
from utils.security_utils import (
    validate_file_extension,
    check_file_size,
    constant_time_compare
)

is_valid = validate_file_extension("file.jpg", [".jpg", ".png"])
within_limit = check_file_size(1024 * 1024, max_size_mb=10)
is_equal = constant_time_compare(token1, token2)
```

## Utilidades de Strings

### Transformación

```python
from utils.string_utils import (
    slugify,
    truncate,
    camel_to_snake,
    snake_to_camel
)

slug = slugify("Hello World!")  # "hello-world"
short = truncate("Long text", 10)  # "Long te..."
snake = camel_to_snake("camelCase")  # "camel_case"
camel = snake_to_camel("snake_case")  # "snakeCase"
```

### Extracción

```python
from utils.string_utils import (
    extract_emails,
    extract_urls,
    is_valid_url
)

emails = extract_emails("Contact: user@example.com")
urls = extract_urls("Visit https://example.com")
is_valid = is_valid_url("https://example.com")
```

### Normalización

```python
from utils.string_utils import (
    normalize_whitespace,
    normalize_url,
    remove_html_tags,
    escape_html
)

clean = normalize_whitespace("  multiple   spaces  ")
url = normalize_url("/path", "https://example.com")
text = remove_html_tags("<p>Hello</p>")
safe = escape_html("<script>alert('xss')</script>")
```

## Utilidades de Fechas

### Fechas Actuales

```python
from utils.date_utils import now_utc, now_local

utc_now = now_utc()
local_now = now_local("America/New_York")
```

### Parsing y Formato

```python
from utils.date_utils import parse_datetime, format_datetime

dt = parse_datetime("2024-01-01T12:00:00")
formatted = format_datetime(dt, "%Y-%m-%d")
```

### Utilidades

```python
from utils.date_utils import (
    time_ago,
    is_expired,
    add_time,
    get_start_of_day,
    get_end_of_day
)

ago = time_ago(datetime(2024, 1, 1))  # "2 months ago"
expired = is_expired(dt, ttl_seconds=3600)
future = add_time(dt, days=7, hours=2)
start = get_start_of_day(dt)
end = get_end_of_day(dt)
```

## Ejemplos de Uso Combinado

### Endpoint con Timeout y Cache

```python
from fastapi import APIRouter
from utils.decorators_advanced import timeout, cache_result

router = APIRouter()

@router.get("/data")
@timeout(5.0)
@cache_result(ttl_seconds=300)
async def get_data():
    # Timeout de 5s, cacheado por 5 minutos
    return await fetch_data()
```

### Service con Throttle y Performance Tracking

```python
from utils.decorators_advanced import throttle, track_performance

class ExternalAPIService:
    @throttle(calls=10, period=60.0)
    @track_performance("external_api")
    async def call_api(self):
        # Máximo 10 llamadas/minuto, métricas automáticas
        pass
```

### Validación y Sanitización

```python
from utils.security_utils import sanitize_filename, validate_file_extension
from utils.decorators_advanced import validate_input

def validate_file(filename):
    if not validate_file_extension(filename, [".jpg", ".png"]):
        raise ValueError("Invalid extension")
    return sanitize_filename(filename)

@validate_input(validate_file)
async def upload_file(filename):
    # Filename validado y sanitizado
    pass
```

### Logging y Observabilidad

```python
from utils.decorators_advanced import log_execution
from utils.observability import get_tracer

@log_execution(log_args=True)
async def process_image(image_id):
    tracer = get_tracer()
    
    async with tracer.span("process_image") as span:
        span.set_tag("image_id", image_id)
        # Procesamiento
        pass
```

## Mejores Prácticas

1. **Decoradores**: Usar decoradores para cross-cutting concerns
2. **Seguridad**: Siempre sanitizar inputs del usuario
3. **Validación**: Validar temprano, fallar rápido
4. **Caching**: Cachear operaciones costosas
5. **Throttling**: Limitar llamadas a APIs externas
6. **Logging**: Registrar operaciones importantes
7. **Tracing**: Agregar spans en operaciones críticas


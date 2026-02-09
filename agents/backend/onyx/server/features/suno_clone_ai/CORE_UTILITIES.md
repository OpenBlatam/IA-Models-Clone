# Utilidades Core - Suno Clone AI

Este documento describe las utilidades y helpers disponibles en el módulo `core`.

## 🔧 Service Locator

Acceso global a servicios sin acoplamiento directo.

### Uso Básico

```python
from core import get_service, resolve_service

# Obtener servicio por nombre
generator = get_service("music_generator")

# Resolver por tipo
from core.interfaces import IMusicGenerator
generator = resolve_service(IMusicGenerator)
```

## ✅ Validators

Validadores reutilizables para datos comunes.

### Validación de UUID

```python
from core import Validator

is_valid = Validator.validate_uuid("550e8400-e29b-41d4-a716-446655440000")
```

### Validación de Email

```python
is_valid = Validator.validate_email("user@example.com")
```

### Validación de URL

```python
is_valid = Validator.validate_url("https://example.com")
```

### Validación de Audio

```python
is_valid = Validator.validate_audio_format("song.wav", ["wav", "mp3"])
```

### Validación de Prompt

```python
is_valid = Validator.validate_prompt("happy music", min_length=1, max_length=1000)
```

### Validación con Excepción

```python
from core import validate_and_raise, Validator

try:
    validate_and_raise(
        Validator.validate_uuid,
        "invalid-uuid",
        "Invalid UUID format"
    )
except ValidationError as e:
    print(e)
```

## 🛠️ Helpers

Funciones helper comunes para operaciones frecuentes.

### Generación de IDs

```python
from core import generate_id

# ID simple
id = generate_id()

# ID con prefijo
song_id = generate_id("song")
```

### Hashing

```python
from core import hash_string

hash_value = hash_string("my string", algorithm="sha256")
```

### JSON Seguro

```python
from core import safe_json_loads, safe_json_dumps

# Cargar JSON
data = safe_json_loads('{"key": "value"}', default={})

# Serializar JSON
json_str = safe_json_dumps({"key": "value"}, default="{}")
```

### Formateo

```python
from core import format_duration, format_file_size

# Formatear duración
duration = format_duration(225.5)  # "3:45"

# Formatear tamaño
size = format_file_size(1572864)  # "1.50 MB"
```

### Directorios

```python
from core import ensure_directory

# Asegurar que existe
path = ensure_directory("storage/audio")
```

### Listas

```python
from core import chunk_list

# Dividir en chunks
items = [1, 2, 3, 4, 5, 6, 7, 8, 9]
chunks = chunk_list(items, 3)  # [[1,2,3], [4,5,6], [7,8,9]]
```

### Diccionarios

```python
from core import merge_dicts, get_nested_value, set_nested_value

# Fusionar diccionarios
merged = merge_dicts({"a": 1}, {"b": 2}, {"c": 3})

# Obtener valor anidado
value = get_nested_value({"user": {"profile": {"name": "John"}}}, "user.profile.name")

# Establecer valor anidado
data = {}
set_nested_value(data, "user.profile.name", "John")
```

### Archivos

```python
from core import sanitize_filename

# Sanitizar nombre de archivo
safe_name = sanitize_filename("my<>file?.txt")  # "my__file_.txt"
```

### Reintentos

```python
from core import retry_on_failure

@retry_on_failure(max_retries=3, delay=1.0)
async def my_function():
    # Código que puede fallar
    pass
```

## 🚀 Sistema de Inicialización

Inicialización ordenada de todos los componentes.

### Inicialización Automática

```python
from core import initialize_system

# Inicializar con configuración
config = {
    "storage_type": "local",
    "cache_type": "memory",
    "generator_type": "fast"
}
results = await initialize_system(config)
```

### Inicialización Manual

```python
from core import get_system_initializer

initializer = get_system_initializer()
results = await initializer.initialize_all(config)

# Obtener estadísticas
stats = initializer.get_initialization_stats()
print(f"Total time: {stats['total_time']}s")
```

### Shutdown

```python
await initializer.shutdown()
```

## 📋 Ejemplos Completos

### Ejemplo 1: Validar y Procesar

```python
from core import Validator, generate_id, format_duration

# Validar entrada
if not Validator.validate_prompt(prompt):
    raise ValueError("Invalid prompt")

# Generar ID
song_id = generate_id("song")

# Formatear duración
duration_str = format_duration(180.5)  # "3:00"
```

### Ejemplo 2: Manejo de Errores con Reintentos

```python
from core import retry_on_failure

@retry_on_failure(max_retries=3, delay=2.0)
async def generate_with_retry(prompt):
    generator = get_service("music_generator")
    return await generator.generate(prompt)
```

### Ejemplo 3: Inicialización Completa

```python
from core import initialize_system, get_service

# Inicializar sistema
config = {
    "storage_type": "local",
    "storage_path": "storage",
    "cache_type": "distributed",
    "generator_type": "optimized"
}
await initialize_system(config)

# Usar servicios
generator = get_service("music_generator")
cache = get_service("cache")
storage = get_service("storage")
```

## 🎯 Casos de Uso

### Validación de Entrada de Usuario

```python
from core import Validator, validate_and_raise

def validate_user_input(data):
    validate_and_raise(
        Validator.validate_email,
        data.get("email"),
        "Invalid email"
    )
    validate_and_raise(
        Validator.validate_prompt,
        data.get("prompt"),
        "Invalid prompt"
    )
```

### Procesamiento en Lotes

```python
from core import chunk_list

async def process_songs(songs):
    chunks = chunk_list(songs, 10)
    for chunk in chunks:
        await process_chunk(chunk)
```

### Configuración Anidada

```python
from core import get_nested_value, set_nested_value

# Leer configuración
api_key = get_nested_value(config, "api.keys.openai", default="")

# Escribir configuración
set_nested_value(config, "api.keys.openai", "new-key")
```

## 📚 Referencia Completa

Todas las utilidades están disponibles desde `core`:

```python
from core import (
    # Service Locator
    get_service,
    resolve_service,
    # Validators
    Validator,
    ValidationError,
    validate_and_raise,
    # Helpers
    generate_id,
    hash_string,
    safe_json_loads,
    safe_json_dumps,
    format_duration,
    format_file_size,
    ensure_directory,
    chunk_list,
    merge_dicts,
    get_nested_value,
    set_nested_value,
    sanitize_filename,
    retry_on_failure,
    # Initialization
    initialize_system,
    get_system_initializer
)
```


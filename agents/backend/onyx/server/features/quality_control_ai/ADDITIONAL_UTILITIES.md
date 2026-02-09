# Quality Control AI - Utilidades Adicionales ✅

## 🚀 Nuevas Utilidades Implementadas

### 1. Security Utilities ✅

**Archivo Creado:**
- `utils/security_utils.py`

**Funciones:**
- ✅ `generate_token()` - Generar tokens seguros
- ✅ `hash_string()` - Hash de strings
- ✅ `verify_hash()` - Verificar hashes
- ✅ `sanitize_input()` - Sanitizar input de usuario
- ✅ `encode_base64()` / `decode_base64()` - Codificación base64
- ✅ `mask_sensitive_data()` - Enmascarar datos sensibles

**Uso:**
```python
from quality_control_ai.utils import (
    generate_token,
    hash_string,
    mask_sensitive_data
)

# Generar token seguro
token = generate_token(32)

# Hash de string
hash_value = hash_string("password", "sha256")

# Enmascarar datos sensibles
masked = mask_sensitive_data("sk-1234567890abcdef", visible_chars=4)
# Resultado: "sk-1***********cdef"
```

### 2. File Utilities ✅

**Archivo Creado:**
- `utils/file_utils.py`

**Funciones:**
- ✅ `ensure_directory()` - Crear directorio si no existe
- ✅ `get_file_size()` - Obtener tamaño de archivo
- ✅ `get_file_extension()` - Obtener extensión
- ✅ `is_image_file()` - Verificar si es imagen
- ✅ `get_mime_type()` - Obtener MIME type
- ✅ `list_files()` - Listar archivos
- ✅ `safe_filename()` - Crear nombre de archivo seguro

**Uso:**
```python
from quality_control_ai.utils import (
    ensure_directory,
    is_image_file,
    safe_filename
)

# Crear directorio
dir_path = ensure_directory("./storage/images")

# Verificar si es imagen
if is_image_file("photo.jpg"):
    # Procesar imagen
    pass

# Nombre seguro
safe = safe_filename("my file (1).jpg")
```

### 3. Date Utilities ✅

**Archivo Creado:**
- `utils/date_utils.py`

**Funciones:**
- ✅ `now_utc()` / `now_local()` - Fechas actuales
- ✅ `to_utc()` - Convertir a UTC
- ✅ `format_datetime()` - Formatear fecha
- ✅ `parse_datetime()` - Parsear fecha
- ✅ `time_ago()` - Tiempo relativo ("2 hours ago")
- ✅ `is_within_timeframe()` - Verificar si está en timeframe

**Uso:**
```python
from quality_control_ai.utils import (
    now_utc,
    format_datetime,
    time_ago,
    is_within_timeframe
)
from datetime import timedelta

# Fecha actual UTC
now = now_utc()

# Formatear
formatted = format_datetime(now, "%Y-%m-%d %H:%M:%S")

# Tiempo relativo
ago = time_ago(some_datetime)  # "2 hours ago"

# Verificar timeframe
is_recent = is_within_timeframe(some_datetime, timedelta(hours=1))
```

### 4. Scripts de Utilidad ✅

**Archivos Creados:**
- `scripts/run_server.py` - Script para ejecutar servidor
- `scripts/check_health.py` - Script para health check

**Uso:**
```bash
# Ejecutar servidor
python -m quality_control_ai.scripts.run_server

# Health check
python -m quality_control_ai.scripts.check_health
```

### 5. Utils Module Actualizado ✅

**Archivo Mejorado:**
- `utils/__init__.py`

**Mejoras:**
- ✅ Todas las utilidades exportadas
- ✅ Organización clara
- ✅ Fácil importación

## 📊 Utilidades Disponibles

### Por Categoría

**Validación:**
- validate_email, validate_url, validate_positive_number
- validate_range, validate_not_empty, validate_required_fields

**Performance:**
- measure_time, retry_on_failure, throttle
- PerformanceMonitor

**String:**
- camel_to_snake, snake_to_camel, truncate
- sanitize_filename, format_bytes, format_duration

**Security:**
- generate_token, hash_string, verify_hash
- sanitize_input, encode_base64, decode_base64
- mask_sensitive_data

**File:**
- ensure_directory, get_file_size, get_file_extension
- is_image_file, get_mime_type, list_files
- safe_filename

**Date:**
- now_utc, now_local, to_utc
- format_datetime, parse_datetime, time_ago
- is_within_timeframe

## 🎯 Ejemplos de Uso

### Ejemplo Completo
```python
from quality_control_ai.utils import (
    # Security
    generate_token, mask_sensitive_data,
    # File
    ensure_directory, is_image_file,
    # Date
    now_utc, format_datetime, time_ago,
    # String
    format_bytes, format_duration,
)

# Security
api_key = generate_token(32)
masked_key = mask_sensitive_data(api_key)

# File operations
storage_dir = ensure_directory("./storage")
if is_image_file("photo.jpg"):
    # Process image
    pass

# Date formatting
timestamp = now_utc()
formatted = format_datetime(timestamp)
relative = time_ago(some_past_time)

# Formatting
size_str = format_bytes(1024 * 1024)  # "1.00 MB"
duration_str = format_duration(3665)  # "1h 1m 5.00s"
```

## ✅ Estado Final

- ✅ Security Utilities implementado
- ✅ File Utilities implementado
- ✅ Date Utilities implementado
- ✅ Scripts de utilidad creados
- ✅ Utils module actualizado
- ✅ Sin errores de linting
- ✅ Type hints completos
- ✅ Documentación completa

## 📚 Archivos Creados

**Nuevos:**
- `utils/security_utils.py`
- `utils/file_utils.py`
- `utils/date_utils.py`
- `scripts/run_server.py`
- `scripts/check_health.py`
- `scripts/__init__.py`
- `ADDITIONAL_UTILITIES.md`

**Mejorados:**
- `utils/__init__.py`

---

**Versión**: 2.2.0
**Estado**: ✅ UTILIDADES COMPLETAS Y LISTAS PARA USO




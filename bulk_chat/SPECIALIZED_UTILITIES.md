# Specialized Utilities - Utilidades Especializadas
## Utilidades Especializadas para Operaciones Bulk

Este documento describe utilidades especializadas para seguridad, strings, fechas, configuración, testing, validación y sanitización.

## 🛠️ Nuevas Utilidades Especializadas

### 1. BulkSecurityManager - Gestor de Seguridad

Gestor de seguridad con encriptación, hashing y tokens.

```python
from bulk_chat.core.bulk_operations_performance import BulkSecurityManager

security = BulkSecurityManager()

# Encriptar datos
encrypted = await security.encrypt("sensitive_data", key="secret_key")
decrypted = await security.decrypt(encrypted, key="secret_key")

# Hash de contraseña
hashed = await security.hash_password("password123")
is_valid = await security.verify_password("password123", hashed)

# Generar token
token = await security.generate_token(length=32)
```

**Características:**
- Encriptación/desencriptación
- Hashing de contraseñas
- Generación de tokens
- **Nota:** Usar bibliotecas de criptografía robustas en producción

### 2. BulkStringProcessor - Procesador de Strings

Procesador avanzado de strings con múltiples utilidades.

```python
from bulk_chat.core.bulk_operations_performance import BulkStringProcessor

processor = BulkStringProcessor()

# Slugify
slug = processor.slugify("Hello World! 123")
# "hello-world-123"

# Truncar
truncated = processor.truncate("Long text here", max_length=10)
# "Long text..."

# Extraer emails
emails = processor.extract_emails("Contact: user@example.com")
# ["user@example.com"]

# Extraer URLs
urls = processor.extract_urls("Visit https://example.com")
# ["https://example.com"]

# Normalizar espacios
normalized = processor.normalize_whitespace("Hello    World")
# "Hello World"

# Remover HTML
clean = processor.remove_html_tags("<p>Hello</p>")
# "Hello"

# Conversiones
snake = processor.camel_to_snake("camelCase")
# "camel_case"

camel = processor.snake_to_camel("snake_case")
# "snakeCase"
```

**Características:**
- Slugify
- Truncado
- Extracción de emails/URLs
- Normalización
- Conversiones de formato
- **Mejora:** Procesamiento eficiente de strings

### 3. BulkDateTimeProcessor - Procesador de Fechas

Procesador avanzado de fechas y horas.

```python
from bulk_chat.core.bulk_operations_performance import BulkDateTimeProcessor

dt_processor = BulkDateTimeProcessor()

# Parsear fecha
timestamp = dt_processor.parse_date("2024-01-15")
timestamp = dt_processor.parse_date("15/01/2024")
timestamp = dt_processor.parse_date("2024-01-15 10:30:00", format_str="%Y-%m-%d %H:%M:%S")

# Formatear fecha
formatted = dt_processor.format_date(timestamp, format_str="%Y-%m-%d")
# "2024-01-15"

# Operaciones con fechas
future = dt_processor.add_days(timestamp, 7)
diff = dt_processor.diff_days(timestamp, future)
# 7.0

# Verificar fin de semana
is_weekend = dt_processor.is_weekend(timestamp)

# Timezone
offset = dt_processor.get_timezone_offset("America/New_York")
```

**Características:**
- Parsing flexible de fechas
- Formateo personalizado
- Operaciones con fechas
- Detección de fin de semana
- Manejo de timezones
- **Mejora:** Procesamiento eficiente de fechas

### 4. BulkConfigManager - Gestor de Configuración

Gestor de configuración con valores por defecto.

```python
from bulk_chat.core.bulk_operations_performance import BulkConfigManager

config = BulkConfigManager()

# Establecer configuración
await config.set("database.host", "localhost")
await config.set("database.port", 5432)

# Obtener configuración
host = await config.get("database.host")
port = await config.get("database.port", default=5432)

# Valores por defecto
config.set_default("database.timeout", 30)

# Cargar desde dict
await config.load_from_dict({
    "app.name": "MyApp",
    "app.version": "1.0.0"
})

# Cargar desde JSON
json_config = '{"app.name": "MyApp"}'
await config.load_from_json(json_config)

# Obtener toda la configuración
all_config = await config.get_all()
```

**Características:**
- Valores por defecto
- Carga desde dict/JSON
- Thread-safe
- **Mejora:** Gestión centralizada de configuración

### 5. BulkTestingUtilities - Utilidades de Testing

Utilidades para testing y debugging.

```python
from bulk_chat.core.bulk_operations_performance import BulkTestingUtilities

testing = BulkTestingUtilities()

# Generar datos mock
template = {
    "id": "random_int:1-100",
    "name": "random_string:10",
    "email": "user@example.com"
}
mock_data = testing.generate_mock_data(count=100, template=template)

# Asserts asíncronos
await testing.assert_async(condition=True, message="Test passed")
await testing.assert_equals(actual=5, expected=5)

# Obtener aserciones
assertions = testing.get_assertions()
```

**Características:**
- Generación de datos mock
- Asserts asíncronos
- Tracking de aserciones
- **Mejora:** Testing eficiente

### 6. BulkValidationAdvanced - Validador Avanzado

Validador con reglas predefinidas y personalizables.

```python
from bulk_chat.core.bulk_operations_performance import BulkValidationAdvanced

validator = BulkValidationAdvanced()

# Registrar reglas por defecto
validator.register_default_rules()

# Validar con regla
is_valid, error = await validator.validate("test@example.com", "email")
# (True, None)

is_valid, error = await validator.validate("invalid", "email")
# (False, "Invalid email format")

# Validar con múltiples reglas
results = await validator.validate_multiple(
    "password123",
    ["not_empty", "min_length"],
    min_len=8
)

# Registrar regla personalizada
def custom_rule(value, min_val, max_val):
    return min_val <= value <= max_val

validator.register_rule(
    "range",
    custom_rule,
    error_message="Value out of range"
)

is_valid, error = await validator.validate(50, "range", min_val=0, max_val=100)
```

**Reglas predefinidas:**
- `not_empty`: No vacío
- `min_length`: Longitud mínima
- `max_length`: Longitud máxima
- `email`: Email válido

**Mejora:** Validación robusta y flexible

### 7. BulkDataSanitizer - Sanitizador de Datos

Sanitizador de datos para seguridad.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataSanitizer

sanitizer = BulkDataSanitizer()

# Sanitizar string
clean = sanitizer.sanitize_string("<script>alert('xss')</script>")
# "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"

# Sanitizar HTML
clean_html = sanitizer.sanitize_html("<p>Hello</p>")

# Sanitizar JSON
data = {
    "name": "<script>alert('xss')</script>",
    "email": "user@example.com"
}
sanitized = sanitizer.sanitize_json(data)

# Remover caracteres especiales
clean = sanitizer.remove_special_chars("Hello!@#$World", keep="!")
# "Hello!World"

# Normalizar Unicode
normalized = sanitizer.normalize_unicode("café")
```

**Características:**
- Sanitización de strings
- Sanitización de HTML
- Sanitización de JSON
- Remoción de caracteres especiales
- Normalización Unicode
- **Mejora:** Seguridad en datos de entrada

### 8. BulkResourceTracker - Rastreador de Recursos

Rastreador de recursos del sistema.

```python
from bulk_chat.core.bulk_operations_performance import BulkResourceTracker

tracker = BulkResourceTracker()

# Rastrear recurso
await tracker.track_resource(
    "resource1",
    "database_connection",
    metadata={"host": "localhost"}
)

# Usar recurso...
# ...

# Liberar recurso
await tracker.release_resource("resource1")

# Estadísticas
stats = await tracker.get_resource_stats()
# {
#   "active": 5,
#   "released": 10,
#   "total": 15,
#   "avg_duration": 2.5
# }

# Todos los recursos
all_resources = await tracker.get_all_resources()
```

**Características:**
- Tracking de recursos
- Estadísticas de uso
- Duración promedio
- **Mejora:** Monitoreo de recursos

## 📊 Resumen de Utilidades Especializadas

| Utilidad | Tipo | Mejora |
|----------|------|--------|
| **Security Manager** | Seguridad | Encriptación y hashing |
| **String Processor** | Strings | Procesamiento avanzado |
| **DateTime Processor** | Fechas | Operaciones con fechas |
| **Config Manager** | Configuración | Gestión centralizada |
| **Testing Utilities** | Testing | Testing eficiente |
| **Validation Advanced** | Validación | Reglas flexibles |
| **Data Sanitizer** | Sanitización | Seguridad en datos |
| **Resource Tracker** | Recursos | Monitoreo de recursos |

## 🎯 Casos de Uso Especializados

### Sistema con Seguridad y Validación
```python
security = BulkSecurityManager()
validator = BulkValidationAdvanced()
sanitizer = BulkDataSanitizer()

# Validar y sanitizar entrada
user_input = "<script>alert('xss')</script>user@example.com"
sanitized = sanitizer.sanitize_string(user_input)
is_valid, error = await validator.validate(sanitized, "email")

# Encriptar datos sensibles
encrypted = await security.encrypt(sanitized, key="secret")
```

### Procesamiento de Strings y Fechas
```python
string_proc = BulkStringProcessor()
dt_proc = BulkDateTimeProcessor()

# Procesar texto
text = "Contact: user@example.com Visit https://example.com"
emails = string_proc.extract_emails(text)
urls = string_proc.extract_urls(text)

# Procesar fechas
timestamp = dt_proc.parse_date("2024-01-15")
formatted = dt_proc.format_date(timestamp, "%Y-%m-%d")
```

### Configuración y Testing
```python
config = BulkConfigManager()
testing = BulkTestingUtilities()

# Cargar configuración
await config.load_from_json(json_config)

# Generar datos de prueba
mock_data = testing.generate_mock_data(100, template)

# Validar con configuración
min_length = await config.get("validation.min_length", default=8)
is_valid, error = await validator.validate(
    password,
    "min_length",
    min_len=min_length
)
```

## 📈 Beneficios Totales

1. **Security Manager**: Seguridad básica (usar bibliotecas robustas en producción)
2. **String Processor**: Procesamiento avanzado de strings
3. **DateTime Processor**: Operaciones eficientes con fechas
4. **Config Manager**: Gestión centralizada de configuración
5. **Testing Utilities**: Testing y debugging eficiente
6. **Validation Advanced**: Validación robusta y flexible
7. **Data Sanitizer**: Sanitización para seguridad
8. **Resource Tracker**: Monitoreo de recursos del sistema

## 🚀 Resultados Esperados

Con todas las utilidades especializadas:

- **Seguridad básica** con encriptación y hashing
- **Procesamiento avanzado** de strings
- **Operaciones eficientes** con fechas
- **Gestión centralizada** de configuración
- **Testing eficiente** con datos mock
- **Validación robusta** con reglas flexibles
- **Sanitización** para seguridad en datos
- **Monitoreo** de recursos del sistema

El sistema ahora tiene **85+ optimizaciones, utilidades y componentes** que cubren todos los aspectos posibles de procesamiento masivo, desde optimizaciones de bajo nivel hasta utilidades especializadas de seguridad, strings, fechas, configuración y testing.
















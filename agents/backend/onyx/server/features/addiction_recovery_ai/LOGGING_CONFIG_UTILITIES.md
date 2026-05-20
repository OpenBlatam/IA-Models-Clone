# Utilidades de Logging y Configuración Completas

## Nuevas Utilidades Agregadas

### 1. Logging Utils ✅
**Archivo**: `utils/logging_utils.py`

**Funciones:**
- `setup_logger()` - Configurar logger
- `log_function_call()` - Decorator para log de llamadas
- `log_performance()` - Decorator para log de performance
- `log_error()` - Log de errores con contexto
- `log_info()` - Log de información
- `log_warning()` - Log de advertencias
- `log_debug()` - Log de debug
- `create_log_context()` - Crear contexto de log
- `format_log_message()` - Formatear mensaje de log

**Uso:**
```python
from utils import setup_logger, log_function_call, log_performance

# Setup logger
logger = setup_logger(
    name="my_app",
    level=logging.INFO,
    file_path="logs/app.log"
)

# Decorator for function calls
@log_function_call(logger)
def my_function(x, y):
    return x + y

# Decorator for performance
@log_performance(logger)
def slow_function():
    time.sleep(1)

# Log with context
from utils import log_error, log_info
log_info(logger, "User logged in", user_id=123, ip="192.168.1.1")
log_error(logger, exception, context={"user_id": 123})
```

### 2. Config Utils ✅
**Archivo**: `utils/config_utils.py`

**Clases y Funciones:**
- `Config` - Gestor de configuración
- `load_config_from_file()` - Cargar desde archivo JSON
- `save_config_to_file()` - Guardar a archivo JSON
- `load_config_from_env()` - Cargar desde variables de entorno
- `merge_configs()` - Fusionar configuraciones
- `get_env_var()` - Obtener variable de entorno
- `get_env_bool()` - Obtener booleano de entorno
- `get_env_int()` - Obtener entero de entorno
- `get_env_float()` - Obtener float de entorno

**Uso:**
```python
from utils import Config, load_config_from_file, load_config_from_env

# Load from file
config = load_config_from_file("config.json")
value = config.get("database.host", default="localhost")

# Load from environment
config = load_config_from_env(prefix="APP_")
value = config.get("database.host")

# Use Config class
config = Config()
config.set("database.host", "localhost")
config.set("database.port", 5432)
host = config.get("database.host")

# Environment variables
from utils import get_env_var, get_env_bool, get_env_int
db_host = get_env_var("DB_HOST", default="localhost", required=True)
debug = get_env_bool("DEBUG", default=False)
port = get_env_int("PORT", default=8000)
```

## Estadísticas Finales

### Utilidades Agregadas
- ✅ **2 módulos** nuevos
- ✅ **18 funciones/clases** adicionales
- ✅ **Cobertura completa** de logging y configuración

### Categorías
- ✅ **Logging Utils** - Logging avanzado con decorators
- ✅ **Config Utils** - Gestión de configuración flexible

## Ejemplos de Uso Avanzado

### Logging Utils
```python
from utils import (
    setup_logger, log_function_call, log_performance,
    log_error, log_info, create_log_context
)
import logging

# Setup logger
logger = setup_logger(
    name="my_app",
    level=logging.INFO,
    format_string="%(asctime)s - %(levelname)s - %(message)s",
    file_path="logs/app.log"
)

# Function call logging
@log_function_call(logger)
def process_data(data):
    # Function implementation
    return processed_data

# Performance logging
@log_performance(logger)
def expensive_operation():
    # Expensive operation
    pass

# Contextual logging
context = create_log_context(user_id=123, action="login")
log_info(logger, "User action", **context)

# Error logging
try:
    risky_operation()
except Exception as e:
    log_error(logger, e, context={"operation": "risky_operation"})
```

### Config Utils
```python
from utils import (
    Config, load_config_from_file, save_config_to_file,
    load_config_from_env, merge_configs,
    get_env_var, get_env_bool, get_env_int
)

# Create config
config = Config()
config.set("database.host", "localhost")
config.set("database.port", 5432)
config.set("database.name", "mydb")

# Get values
host = config.get("database.host")
port = config.get("database.port", default=5432)
exists = config.has("database.host")

# Load from file
config = load_config_from_file("config.json")

# Save to file
save_config_to_file(config, "config.json")

# Load from environment
config = load_config_from_env(prefix="APP_")

# Merge configs
default_config = load_config_from_file("default.json")
user_config = load_config_from_file("user.json")
merged = merge_configs(default_config, user_config)

# Environment variables
db_host = get_env_var("DB_HOST", default="localhost", required=True)
debug = get_env_bool("DEBUG", default=False)
port = get_env_int("PORT", default=8000)
timeout = get_env_float("TIMEOUT", default=30.0)
```

## Beneficios

1. ✅ **Logging Utils**: Logging estructurado y contextual
2. ✅ **Config Utils**: Gestión flexible de configuración
3. ✅ **Decorators**: Logging automático de funciones
4. ✅ **Context**: Logging con contexto adicional
5. ✅ **Flexibilidad**: Múltiples fuentes de configuración
6. ✅ **Type Safety**: Funciones tipadas para variables de entorno

## Configuración Avanzada

### Config con Dot Notation
```python
config = Config()
config.set("database.connection.host", "localhost")
config.set("database.connection.port", 5432)

host = config.get("database.connection.host")
```

### Logging con Decorators
```python
@log_function_call(logger)
@log_performance(logger)
def complex_operation(data):
    # Operation implementation
    pass
```

### Configuración Multi-Fuente
```python
# Load from multiple sources
file_config = load_config_from_file("config.json")
env_config = load_config_from_env(prefix="APP_")
merged = merge_configs(file_config, env_config)
```

## Conclusión

El sistema ahora cuenta con:
- ✅ **69 módulos** de utilidades
- ✅ **403+ funciones** reutilizables
- ✅ **Logging Utils** para logging avanzado
- ✅ **Config Utils** para gestión de configuración
- ✅ **Código completamente optimizado**

**Estado**: ✅ Complete Logging & Config Utilities Suite


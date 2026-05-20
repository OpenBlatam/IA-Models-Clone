# Guía de Serialización y Logging Estructurado - MCP Server

## Resumen

Utilidades avanzadas para serialización de datos y logging estructurado con contexto.

## Serialización

### Formatos Soportados

- **JSON**: Formato estándar, legible por humanos
- **YAML**: Formato legible, requiere PyYAML
- **MessagePack**: Formato binario eficiente, requiere msgpack
- **Pickle**: Serialización nativa de Python
- **Base64**: Codificación de bytes

### Uso Básico

```python
from mcp_server.utils.serialization_utils import (
    serialize_json, deserialize_json,
    serialize_yaml, deserialize_yaml,
    Serializer
)

# Serializar a JSON
data = {"key": "value", "number": 42}
json_str = serialize_json(data, indent=2)

# Deserializar JSON
obj = deserialize_json(json_str)

# Usar Serializer genérico
serialized = Serializer.serialize(data, format="json")
obj = Serializer.deserialize(serialized, format="json")
```

### JSONEncoder Personalizado

```python
from mcp_server.utils.serialization_utils import JSONEncoder, serialize_json
from datetime import datetime
from decimal import Decimal

# Maneja automáticamente tipos especiales
data = {
    "timestamp": datetime.now(),
    "amount": Decimal("123.45"),
    "path": Path("/tmp/file.txt")
}

json_str = serialize_json(data)  # Convierte automáticamente
```

### Serialización de Objetos

```python
from mcp_server.utils.serialization_utils import to_dict, from_dict

class MyClass:
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

obj = MyClass("test", 42)

# Convertir a diccionario
d = to_dict(obj)

# Convertir de diccionario
obj2 = from_dict(d, MyClass)
```

## Logging Estructurado

### Configuración Básica

```python
from mcp_server.utils.structured_logging import (
    setup_structured_logging,
    create_logger
)

# Configurar logging estructurado
logger = setup_structured_logging(
    level="DEBUG",
    format_type="json",
    include_context=True
)

# Crear logger con contexto
logger = create_logger("my_module", level="DEBUG")
```

### Context Logger

```python
from mcp_server.utils.structured_logging import ContextLogger, create_logger

logger = create_logger("api")

# Establecer contexto
logger.set_context(
    user_id="123",
    request_id="abc-123",
    component="api"
)

# Los logs incluirán el contexto automáticamente
logger.info("Processing request")
logger.error("Request failed")

# Actualizar contexto
logger.update_context(step="validation")

# Limpiar contexto
logger.clear_context()
```

### Decorador de Contexto

```python
from mcp_server.utils.structured_logging import log_with_context

@log_with_context(component="api", version="1.0")
def process_request():
    logger = create_logger("api")
    logger.info("Processing")  # Incluye contexto automáticamente
```

### Formato de Logs

Los logs estructurados incluyen:

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "level": "INFO",
  "logger": "mcp_server.api",
  "message": "Processing request",
  "module": "api",
  "function": "process_request",
  "line": 42,
  "context": {
    "user_id": "123",
    "request_id": "abc-123"
  }
}
```

## Ejemplos de Uso

### Serialización de Configuración

```python
from mcp_server.utils.serialization_utils import Serializer

# Guardar configuración
config = {"server": {"port": 8080}}
with open("config.json", "w") as f:
    f.write(Serializer.serialize(config, format="json", indent=2))

# Cargar configuración
with open("config.json", "r") as f:
    config = Serializer.deserialize(f.read(), format="json")
```

### Logging con Contexto de Request

```python
from mcp_server.utils.structured_logging import create_logger

logger = create_logger("api")

def handle_request(request_id: str, user_id: str):
    logger.set_context(request_id=request_id, user_id=user_id)
    
    try:
        logger.info("Starting request processing")
        # ... procesamiento ...
        logger.info("Request completed successfully")
    except Exception as e:
        logger.error(f"Request failed: {e}")
    finally:
        logger.clear_context()
```

### Serialización de Datos Complejos

```python
from mcp_server.utils.serialization_utils import (
    serialize_json, to_dict
)
from datetime import datetime

class User:
    def __init__(self, name: str, created_at: datetime):
        self.name = name
        self.created_at = created_at

user = User("John", datetime.now())

# Serializar objeto complejo
user_dict = to_dict(user)
json_str = serialize_json(user_dict)
```

## Próximos Pasos

1. Agregar más formatos de serialización
2. Mejorar manejo de tipos personalizados
3. Agregar compresión a serialización
4. Mejorar contexto de logging
5. Agregar métricas a logs


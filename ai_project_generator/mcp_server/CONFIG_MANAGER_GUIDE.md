# Guía de Config Manager - MCP Server

## Resumen

Gestor avanzado de configuración con validación, caché, y utilidades para gestión de archivos de configuración.

## ConfigManager

### Inicialización

```python
from mcp_server.utils.config_manager import ConfigManager

# Con ruta de archivo
manager = ConfigManager("config.json")

# Sin ruta (se especifica al cargar)
manager = ConfigManager()
```

### Cargar Configuración

```python
# Cargar desde archivo
config = manager.load("config.json")

# Cargar desde ruta inicializada
config = manager.load()

# Formatos soportados: JSON, YAML
```

### Guardar Configuración

```python
# Guardar en archivo
manager.save("config.json", format="json")
manager.save("config.yaml", format="yaml")

# Guardar en ruta inicializada
manager.save()
```

### Obtener Valores

```python
# Usando notación de punto
host = manager.get("server.host", default="localhost")
port = manager.get("server.port", default=8020)

# Obtener sección completa
server_config = manager.get_section("server")
```

### Establecer Valores

```python
# Usando notación de punto
manager.set("server.host", "0.0.0.0")
manager.set("server.port", 8080)

# Actualizar múltiples valores
manager.update({
    "server": {
        "host": "0.0.0.0",
        "port": 8080
    }
}, deep=True)
```

### Validar Configuración

```python
# Validación básica
errors = manager.validate()

# Validación con esquema
schema = {
    "server.port": {
        "type": int,
        "required": True,
        "min": 1,
        "max": 65535
    }
}
errors = manager.validate(schema)

if errors:
    print("Validation errors:", errors)
```

### Utilidades

```python
# Listar secciones
sections = manager.list_sections()
# ['server', 'security', 'rate_limiting', ...]

# Verificar sección
if manager.has_section("server"):
    server_config = manager.get_section("server")

# Obtener como diccionario
config_dict = manager.to_dict()

# Limpiar caché
manager.clear_cache()
```

## Funciones de Utilidad

### `get_default_config()`

Obtener configuración por defecto.

```python
from mcp_server.utils.config_manager import get_default_config

default = get_default_config()
print(default["server"]["port"])  # 8020
```

### `create_config_template()`

Crear plantilla de configuración.

```python
from mcp_server.utils.config_manager import create_config_template

# Crear plantilla JSON
create_config_template("config.json", format="json")

# Crear plantilla YAML
create_config_template("config.yaml", format="yaml")
```

## Ejemplos de Uso

### Cargar y Modificar Configuración

```python
from mcp_server.utils.config_manager import ConfigManager

manager = ConfigManager("config.json")
manager.load()

# Modificar valores
manager.set("server.port", 8080)
manager.set("server.debug", True)

# Guardar cambios
manager.save()
```

### Validar Configuración

```python
from mcp_server.utils.config_manager import ConfigManager

manager = ConfigManager("config.json")
manager.load()

schema = {
    "server": {
        "port": {"type": int, "required": True, "min": 1, "max": 65535},
        "host": {"type": str, "required": True},
    },
    "security": {
        "secret_key": {"type": str, "required": True},
    }
}

errors = manager.validate(schema)
if errors:
    for error in errors:
        print(f"Error: {error}")
else:
    print("Configuration is valid")
```

### Crear Configuración desde Plantilla

```python
from mcp_server.utils.config_manager import (
    ConfigManager, create_config_template
)

# Crear plantilla
create_config_template("config.json")

# Cargar y personalizar
manager = ConfigManager("config.json")
manager.load()
manager.set("server.port", 8080)
manager.set("security.secret_key", "my-secret-key")
manager.save()
```

## Integración con CLI

El ConfigManager está integrado con el CLI:

```bash
# Mostrar configuración
python -m mcp_server.cli config show --path config.json

# Validar
python -m mcp_server.cli config validate --path config.json

# Generar plantilla
python -m mcp_server.cli config template --output config.json

# Obtener valor
python -m mcp_server.cli config get --path config.json "server.port"

# Establecer valor
python -m mcp_server.cli config set --path config.json "server.port" "8080"
```

## Formatos Soportados

### JSON

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8020
  },
  "security": {
    "secret_key": "change-me"
  }
}
```

### YAML

```yaml
server:
  host: 0.0.0.0
  port: 8020

security:
  secret_key: change-me
```

## Características

- ✅ Carga desde JSON y YAML
- ✅ Guardado en múltiples formatos
- ✅ Notación de punto para acceso
- ✅ Validación con esquemas
- ✅ Merge profundo de configuraciones
- ✅ Plantillas de configuración
- ✅ Integración con CLI

## Próximos Pasos

1. Agregar más validadores
2. Soporte para variables de entorno
3. Cifrado de valores sensibles
4. Historial de cambios
5. Validación de tipos avanzada


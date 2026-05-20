# Guía de Diagnóstico - MCP Server

## Resumen

Se han agregado utilidades de diagnóstico completas para el módulo MCP Server, permitiendo verificar el estado del sistema, diagnosticar problemas y obtener información detallada para debugging.

## Funciones Disponibles

### 1. Funciones Públicas del Módulo

#### `get_diagnostics() -> Dict[str, Any]`
Obtiene diagnóstico completo del módulo incluyendo:
- Información del sistema
- Estado de imports
- Características disponibles
- Información del módulo

```python
from mcp_server import get_diagnostics

diagnostics = get_diagnostics()
print(diagnostics)
```

#### `check_health() -> Dict[str, Any]`
Verifica la salud del módulo:
- Estado general (healthy/degraded/unhealthy)
- Checks individuales (imports, core components)
- Métricas de disponibilidad

```python
from mcp_server import check_health

health = check_health()
if health["status"] == "healthy":
    print("Module is healthy")
```

#### `validate_setup() -> Tuple[bool, List[str]]`
Valida la configuración del módulo:
- Verifica imports core
- Verifica ImportManager
- Verifica utilidades

```python
from mcp_server import validate_setup

is_valid, errors = validate_setup()
if not is_valid:
    print(f"Setup errors: {errors}")
```

### 2. Funciones de Utilidades

#### `get_system_info() -> Dict[str, Any]`
Obtiene información del sistema:
- Plataforma
- Versión de Python
- Ruta de ejecución
- Path de Python

```python
from mcp_server.utils.diagnostics import get_system_info

info = get_system_info()
print(f"Platform: {info['platform']}")
print(f"Python: {info['python_version']}")
```

#### `get_module_diagnostics() -> Dict[str, Any]`
Obtiene diagnóstico completo del módulo.

#### `check_module_health() -> Dict[str, Any]`
Verifica salud del módulo con checks detallados.

#### `get_dependency_tree() -> Dict[str, Any]`
Obtiene árbol de dependencias del módulo:
- Grupos de imports
- Módulos por grupo
- Símbolos por módulo

```python
from mcp_server.utils.diagnostics import get_dependency_tree

tree = get_dependency_tree()
print(f"Total groups: {tree['total_groups']}")
print(f"Total modules: {tree['total_modules']}")
```

#### `validate_module_setup() -> Tuple[bool, List[str]]`
Valida configuración del módulo.

#### `get_performance_metrics() -> Dict[str, Any]`
Obtiene métricas de performance:
- Uso de memoria (RSS, VMS, porcentaje)
- Uso de CPU (porcentaje, threads)
- Timestamp

```python
from mcp_server.utils.diagnostics import get_performance_metrics

metrics = get_performance_metrics()
print(f"Memory: {metrics['memory']['rss_mb']:.2f} MB")
print(f"CPU: {metrics['cpu']['percent']:.2f}%")
```

#### `generate_diagnostic_report() -> str`
Genera reporte de diagnóstico completo formateado.

```python
from mcp_server.utils.diagnostics import generate_diagnostic_report

report = generate_diagnostic_report()
print(report)
```

#### `print_diagnostic_report() -> None`
Imprime reporte de diagnóstico en consola.

```python
from mcp_server.utils.diagnostics import print_diagnostic_report

print_diagnostic_report()
```

## Ejemplos de Uso

### Verificar Salud del Módulo

```python
from mcp_server import check_health

health = check_health()
print(f"Status: {health['status']}")

for check_name, check_info in health['checks'].items():
    print(f"{check_name}: {check_info['status']}")
```

### Obtener Diagnóstico Completo

```python
from mcp_server import get_diagnostics

diagnostics = get_diagnostics()
print(f"Version: {diagnostics['version']}")
print(f"System: {diagnostics['system_info']['platform']}")
print(f"Import availability: {diagnostics['imports']['status']}")
```

### Validar Configuración

```python
from mcp_server import validate_setup

is_valid, errors = validate_setup()
if is_valid:
    print("✓ Module setup is valid")
else:
    print("✗ Module setup has errors:")
    for error in errors:
        print(f"  - {error}")
```

### Generar Reporte Completo

```python
from mcp_server.utils.diagnostics import print_diagnostic_report

# Imprime reporte completo en consola
print_diagnostic_report()
```

### Monitorear Performance

```python
from mcp_server.utils.diagnostics import get_performance_metrics
import time

while True:
    metrics = get_performance_metrics()
    print(f"Memory: {metrics['memory']['rss_mb']:.2f} MB")
    print(f"CPU: {metrics['cpu']['percent']:.2f}%")
    time.sleep(5)
```

## Estructura de Respuestas

### Health Check Response

```python
{
    "status": "healthy" | "degraded" | "unhealthy",
    "checks": {
        "imports": {
            "status": "ok" | "degraded" | "error",
            "availability_rate": 95.5,
            "total": 100,
            "available": 95
        },
        "core_components": {
            "status": "ok" | "error",
            "components": {
                "MCPServer": True,
                "ConnectorRegistry": True,
                "MCPSecurityManager": True
            }
        }
    },
    "timestamp": "2024-01-01T12:00:00"
}
```

### Diagnostics Response

```python
{
    "timestamp": "2024-01-01T12:00:00",
    "version": "2.2.0",
    "system_info": {
        "platform": "...",
        "python_version": "...",
        ...
    },
    "imports": {
        "status": {...},
        "missing": [...],
        "available_features": {...}
    },
    "module_info": {...}
}
```

## Casos de Uso

1. **Debugging**: Usar `get_diagnostics()` para obtener información completa
2. **Monitoring**: Usar `check_health()` en endpoints de health check
3. **Validation**: Usar `validate_setup()` en scripts de inicialización
4. **Performance**: Usar `get_performance_metrics()` para monitoreo
5. **Reporting**: Usar `generate_diagnostic_report()` para reportes

## Integración con Health Checks

```python
from fastapi import APIRouter
from mcp_server import check_health

router = APIRouter()

@router.get("/health")
async def health_endpoint():
    health = check_health()
    status_code = 200 if health["status"] == "healthy" else 503
    return health, status_code
```

## Próximos Pasos

1. Agregar más checks de salud
2. Integrar con sistemas de monitoreo
3. Agregar alertas automáticas
4. Crear dashboard de diagnóstico
5. Agregar métricas históricas


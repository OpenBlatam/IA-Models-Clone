# Tools Package - Sistema Refactorizado

## Visión General

Paquete refactorizado que proporciona una arquitectura modular y extensible para todas las herramientas de API.

## Estructura

```
tools/
├── __init__.py                    # Exports
├── base.py                        # BaseAPITool, ToolResult
├── config.py                      # ToolConfig
├── utils.py                       # Utilidades
├── registry.py                    # ToolRegistry
├── manager.py                     # ToolManager
├── refactored_health_checker.py   # Health Checker
└── refactored_benchmark.py        # Benchmark
```

## Uso Rápido

### Crear Nueva Herramienta

```python
from tools.base import BaseAPITool, ToolResult
from tools.registry import register_tool

@register_tool("my_tool")
class MyTool(BaseAPITool):
    def run(self, **kwargs) -> ToolResult:
        response = self.make_request("GET", "/endpoint")
        return ToolResult(
            success=response.status_code == 200,
            message="Success",
            data={"status": response.status_code}
        )
```

### Usar Tool Manager

```python
from tools.manager import ToolManager

manager = ToolManager()
result = manager.run_tool("health")
```

### CLI

```bash
python -m tools.manager --list
python -m tools.manager --tool health
```

## Ventajas

- ✅ Eliminación de duplicación
- ✅ Estructura consistente
- ✅ Fácil de extender
- ✅ Configuración centralizada
- ✅ Mejor mantenibilidad




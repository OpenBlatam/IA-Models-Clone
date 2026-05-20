# Guía de Integración - Tools Package

## Integración con Sistema Principal

El sistema de tools refactorizado está integrado con `start_api_and_debug.py`.

### Nuevas Opciones en el Menú

```
17. 🔧 Tool Manager (Refactored tools system)
18. ⛓️  Tool Chain (Execute tool chains)
19. ⚡ Tool Executor (Advanced executor)
```

### Uso desde CLI

```bash
# Tool Manager
python start_api_and_debug.py --tool tool-manager

# Tool Chain
python start_api_and_debug.py --tool tool-chain

# Tool Executor
python start_api_and_debug.py --tool tool-executor
```

### Uso desde Menú Interactivo

1. Ejecutar `python start_api_and_debug.py`
2. Seleccionar opción 17, 18, o 19
3. Las herramientas se ejecutarán automáticamente

## Ejemplos de Integración

### Tool Manager

```python
from tools.manager import ToolManager

manager = ToolManager()
manager.print_tools_list()
result = manager.run_tool("health")
```

### Tool Chain

```python
from tools.chain import create_chain

chain = create_chain()
chain.add_tool("health", endpoints=["/health"])
chain.add_tool("benchmark", endpoint="/health", iterations=10)
results = chain.execute()
chain.print_summary()
```

### Tool Executor

```python
from tools.executor import ToolExecutor

executor = ToolExecutor()
result = executor.execute("health", endpoints=["/health"])
```

## Beneficios de la Integración

1. **Acceso Unificado**: Todas las herramientas desde un solo menú
2. **Consistencia**: Mismo sistema de configuración
3. **Facilidad de Uso**: No necesita conocer rutas de archivos
4. **Extensibilidad**: Fácil agregar nuevas herramientas




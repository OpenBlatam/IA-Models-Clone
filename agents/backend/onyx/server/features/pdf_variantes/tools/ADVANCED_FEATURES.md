# Características Avanzadas - Tools Package

## Nuevas Características

### 1. Tool Factory (`factory.py`)

Factory pattern para creación de herramientas con dependency injection.

**Características:**
- Creación de herramientas con configuración
- Custom creators
- Dependency injection
- Configuración por herramienta

**Uso:**
```python
from tools.factory import get_factory

factory = get_factory()
tool = factory.create("health", base_url="http://localhost:8000")
```

### 2. Tool Chain (`chain.py`)

Ejecución secuencial de múltiples herramientas.

**Características:**
- Chain de herramientas
- Ejecución secuencial
- Stop on error opcional
- Resumen de ejecución

**Uso:**
```python
from tools.chain import create_chain

chain = create_chain()
chain.add_tool("health", endpoints=["/health"])
chain.add_tool("benchmark", endpoint="/health", iterations=10)
results = chain.execute()
chain.print_summary()
```

### 3. Plugin System (`plugins.py`)

Sistema de plugins para extender funcionalidad.

**Plugins incluidos:**
- `LoggingPlugin`: Logging automático
- `MetricsPlugin`: Recopilación de métricas
- `CachingPlugin`: Caché de resultados

**Uso:**
```python
from tools.plugins import get_plugin_manager, MetricsPlugin

plugin_manager = get_plugin_manager()
plugin_manager.register(MetricsPlugin())
```

### 4. Tool Executor (`executor.py`)

Ejecutor avanzado con plugins, chaining y ejecución paralela.

**Características:**
- Ejecución con plugins
- Ejecución paralela
- Ejecución en cadena
- Métricas automáticas

**Uso:**
```python
from tools.executor import ToolExecutor

executor = ToolExecutor()

# Ejecución simple
result = executor.execute("health")

# Ejecución paralela
results = executor.execute_parallel([
    {"name": "health", "kwargs": {}},
    {"name": "benchmark", "kwargs": {"endpoint": "/health"}}
])

# Ejecución en cadena
results = executor.execute_chain([
    {"name": "health", "kwargs": {}},
    {"name": "test", "kwargs": {}}
])
```

## Ejemplos Avanzados

### Workflow Completo con Chain

```python
from tools.chain import create_chain

chain = create_chain()
chain.add_tool("health", endpoints=["/health", "/"])
chain.add_tool("test", tests=[{"name": "Test", "method": "GET", "endpoint": "/health"}])
chain.add_tool("benchmark", endpoint="/health", iterations=100)

results = chain.execute()
chain.print_summary()
```

### Ejecución Paralela

```python
from tools.executor import ToolExecutor

executor = ToolExecutor()

tools = [
    {"name": "health", "kwargs": {"endpoints": ["/health"]}},
    {"name": "health", "kwargs": {"endpoints": ["/"]}},
    {"name": "health", "kwargs": {"endpoints": ["/docs"]}}
]

results = executor.execute_parallel(tools, max_workers=3)
```

### Con Plugins Personalizados

```python
from tools.plugins import ToolPlugin
from tools.executor import ToolExecutor

class CustomPlugin(ToolPlugin):
    def before_run(self, tool, **kwargs):
        print(f"Starting {tool.__class__.__name__}")
        return kwargs
    
    def after_run(self, tool, result, **kwargs):
        print(f"Completed: {result.success}")
        return result

executor = ToolExecutor()
executor.plugin_manager.register(CustomPlugin())
result = executor.execute("health")
```

## Beneficios

1. **Flexibilidad**: Múltiples formas de ejecutar herramientas
2. **Extensibilidad**: Sistema de plugins
3. **Performance**: Ejecución paralela
4. **Workflows**: Chaining para flujos complejos
5. **Métricas**: Recopilación automática




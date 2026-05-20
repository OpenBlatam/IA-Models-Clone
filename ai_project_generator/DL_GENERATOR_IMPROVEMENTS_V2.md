# Mejoras del Deep Learning Generator V2 - Funcionalidades Avanzadas

## Resumen

Se han agregado funcionalidades avanzadas al generador de Deep Learning: sistema de templates personalizados, plugins, validación de código, y mejor integración.

## Nuevas Funcionalidades

### 1. Template Manager (`deep_learning/template_manager.py`)

Sistema de templates personalizados usando Jinja2.

#### Características:

✅ **Templates Personalizados**
- Templates basados en Jinja2
- Variables configurables
- Categorización de templates
- Persistencia en JSON

✅ **Renderizado Dinámico**
- Renderizado con contexto personalizado
- Soporte para variables complejas
- Templates reutilizables

#### Uso:

```python
from core.deep_learning.template_manager import get_template_manager

manager = get_template_manager()

# Registrar template
manager.register_template(
    name="custom_model",
    content="""
class {{ model_name }}(nn.Module):
    def __init__(self, {{ params }}):
        super().__init__()
        # ...
    """,
    variables=["model_name", "params"],
    description="Template para modelo personalizado",
    category="models"
)

# Renderizar template
code = manager.render_template(
    "custom_model",
    context={
        "model_name": "MyTransformer",
        "params": "hidden_size=512, num_layers=6"
    }
)
```

### 2. Plugin System (`deep_learning/plugin_system.py`)

Sistema de plugins para extender el generador.

#### Características:

✅ **Hooks Personalizados**
- `generation_start`: Al inicio de generación
- `generation_end`: Al final de generación
- `file_generated`: Cuando se genera un archivo

✅ **Plugins Reutilizables**
- Metadatos de plugins
- Registro/desregistro dinámico
- Múltiples plugins simultáneos

#### Uso:

```python
from core.deep_learning.plugin_system import Plugin, PluginMetadata, get_plugin_manager

class MyPlugin(Plugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            description="Plugin personalizado",
            hooks=["generation_start", "generation_end"]
        )
    
    def on_generation_start(self, generator_key, project_dir, keywords, project_info):
        print(f"Iniciando generación: {generator_key}")
    
    def on_generation_end(self, generator_key, project_dir, success, files_generated):
        print(f"Generación completada: {generator_key}, {len(files_generated)} archivos")

# Registrar plugin
manager = get_plugin_manager()
manager.register_plugin(MyPlugin())
```

### 3. Code Validator (`deep_learning/code_validator.py`)

Validador de código Python generado.

#### Características:

✅ **Validaciones Automáticas**
- Sintaxis Python
- Imports no usados
- Docstrings faltantes
- Type hints
- Convenciones de nombres (PEP 8)

✅ **Reglas Personalizables**
- Agregar reglas personalizadas
- Validación por archivo o directorio
- Resultados detallados

#### Uso:

```python
from core.deep_learning.code_validator import get_validator

validator = get_validator()

# Validar archivo
result = validator.validate_file(Path("app/models/model.py"))
if not result.is_valid:
    print(f"Errores: {result.errors}")
print(f"Warnings: {result.warnings}")

# Validar directorio
results = validator.validate_directory(Path("app"), recursive=True)

# Agregar regla personalizada
def check_custom_rule(content: str, file_path: Path) -> List[str]:
    issues = []
    if "TODO" in content:
        issues.append("Archivo contiene TODOs")
    return issues

validator.add_rule(check_custom_rule)
```

### 4. Integración Completa

✅ **Integración Automática**
- Templates se cargan automáticamente
- Plugins se ejecutan automáticamente
- Validación se ejecuta después de generar
- Cache funciona transparentemente

✅ **Configuración Flexible**
- Habilitar/deshabilitar funcionalidades
- Configuración por instancia
- Fallbacks si componentes no están disponibles

## Ejemplo Completo

```python
from pathlib import Path
from core.deep_learning_generator import DeepLearningGenerator
from core.deep_learning.template_manager import get_template_manager
from core.deep_learning.plugin_system import Plugin, PluginMetadata

# 1. Crear plugin personalizado
class LoggingPlugin(Plugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="logging_plugin",
            version="1.0.0",
            hooks=["generation_start", "generation_end"]
        )
    
    def on_generation_start(self, generator_key, project_dir, keywords, project_info):
        logger.info(f"🚀 Iniciando: {generator_key}")
    
    def on_generation_end(self, generator_key, project_dir, success, files_generated):
        logger.info(f"✅ Completado: {generator_key} ({len(files_generated)} archivos)")

# 2. Configurar generador
generator = DeepLearningGenerator(
    enable_cache=True,
    enable_validation=True
)

# 3. Registrar plugin
plugin_manager = generator.get_plugin_manager()
if plugin_manager:
    plugin_manager.register_plugin(LoggingPlugin())

# 4. Registrar template personalizado
template_manager = generator.get_template_manager()
if template_manager:
    template_manager.register_template(
        name="custom_layer",
        content="""
class {{ layer_name }}(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)
        """,
        variables=["layer_name"]
    )

# 5. Generar proyecto
project_dir = Path("my_project")
keywords = {"requires_training": True, "framework": "pytorch"}
project_info = {"name": "my_dl_project"}

stats = generator.generate_all(project_dir, keywords, project_info)

# 6. Validar código generado
validation = generator.validate_generated_code(project_dir)
print(f"Archivos válidos: {validation['valid_files']}/{validation['total_files']}")
print(f"Warnings: {validation['total_warnings']}")

# 7. Obtener estadísticas
print(f"Generación completada: {stats.get_summary()}")
```

## Beneficios

1. **Extensibilidad**: Plugins permiten agregar funcionalidades sin modificar código base
2. **Personalización**: Templates permiten código personalizado
3. **Calidad**: Validación automática asegura código correcto
4. **Performance**: Cache reduce tiempo de regeneración
5. **Observabilidad**: Plugins permiten logging y monitoreo personalizado

## Estado

✅ **Completado**

Todas las funcionalidades avanzadas están implementadas y funcionando.


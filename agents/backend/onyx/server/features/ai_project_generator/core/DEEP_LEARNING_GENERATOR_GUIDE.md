# Guía Completa - Deep Learning Generator

## 📚 Visión General

El `DeepLearningGenerator` es un generador especializado para crear código de proyectos de Deep Learning usando PyTorch, TensorFlow, JAX y otros frameworks.

## 🚀 Uso Básico

### Crear un Generador

```python
from core.deep_learning_generator import create_generator, DeepLearningGenerator

# Método 1: Usar función helper
generator = create_generator(
    framework="pytorch",
    model_type="transformer"
)

# Método 2: Crear directamente
generator = DeepLearningGenerator(
    framework="pytorch",
    model_type="transformer"
)
```

### Obtener Información

```python
from core.deep_learning_generator import (
    get_supported_frameworks,
    get_supported_model_types,
    get_generator_info
)

# Frameworks soportados
frameworks = get_supported_frameworks()
# ['pytorch', 'tensorflow', 'jax', 'onnx']

# Tipos de modelos
model_types = get_supported_model_types()
# ['transformer', 'cnn', 'rnn', 'lstm', 'gru', 'gan', 'vae', 'diffusion', 'llm', 'vision_transformer']

# Información del generador
info = get_generator_info()
```

## 🔍 Detección Automática

### Detectar Framework y Tipo de Modelo

```python
from core.deep_learning_generator import (
    detect_framework_from_code,
    detect_model_type_from_code,
    analyze_code_file
)

# Detectar desde código
code = """
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(...)
"""

framework = detect_framework_from_code(code)  # "pytorch"
model_type = detect_model_type_from_code(code)  # "transformer"

# Analizar archivo
from pathlib import Path
analysis = analyze_code_file(Path("model.py"))
```

### Sugerir Configuración

```python
from core.deep_learning_generator import suggest_generator_config
from pathlib import Path

# Analizar proyecto y sugerir configuración
config = suggest_generator_config(project_path=Path("."))
# {'framework': 'pytorch', 'model_type': 'transformer'}

# Crear generador con configuración sugerida
generator = create_generator(**config)
```

## 📊 Información de Frameworks y Modelos

### Información de Framework

```python
from core.deep_learning_generator import get_framework_info

info = get_framework_info("pytorch")
# {
#     "name": "PyTorch",
#     "description": "Deep learning framework by Facebook",
#     "common_imports": ["torch", "torch.nn", "torch.optim"],
#     "version": "2.0+",
#     "use_cases": ["Research", "Production", "Prototyping"]
# }
```

### Información de Tipo de Modelo

```python
from core.deep_learning_generator import get_model_type_info

info = get_model_type_info("transformer")
# {
#     "name": "Transformer",
#     "description": "Attention-based architecture",
#     "use_cases": ["NLP", "Vision", "Multimodal"],
#     "common_layers": ["MultiHeadAttention", "FeedForward", "LayerNorm"]
# }
```

## ✅ Validación

### Validar Configuración

```python
from core.deep_learning_generator import validate_generator_config

config = {
    "framework": "pytorch",
    "model_type": "transformer"
}

is_valid, error = validate_generator_config(config)

if is_valid:
    generator = create_generator(**config)
else:
    print(f"Invalid config: {error}")
```

## 🎨 Plantillas de Configuración

### Generar Plantilla

```python
from core.deep_learning_generator import generate_config_template

template = generate_config_template(
    framework="pytorch",
    model_type="transformer"
)

# Usar plantilla
generator = create_generator(**template)
```

## 📝 Ejemplos Completos

### Ejemplo 1: Generador Básico

```python
from core.deep_learning_generator import create_generator

# Crear generador
generator = create_generator(
    framework="pytorch",
    model_type="transformer"
)

# Generar código (ejemplo)
# code = generator.generate_model_code(config)
```

### Ejemplo 2: Detección Automática

```python
from core.deep_learning_generator import (
    suggest_generator_config,
    create_generator
)
from pathlib import Path

# Analizar proyecto existente
project_path = Path("existing_project")
config = suggest_generator_config(project_path=project_path)

# Crear generador con configuración detectada
generator = create_generator(**config)
```

### Ejemplo 3: Validación y Creación

```python
from core.deep_learning_generator import (
    validate_generator_config,
    create_generator,
    get_framework_info
)

# Validar antes de crear
config = {
    "framework": "pytorch",
    "model_type": "llm"
}

is_valid, error = validate_generator_config(config)
if is_valid:
    # Obtener información del framework
    framework_info = get_framework_info(config["framework"])
    print(f"Using {framework_info['name']}")
    
    # Crear generador
    generator = create_generator(**config)
else:
    print(f"Error: {error}")
```

## 🛠️ Funciones Disponibles

### Funciones Principales
- `create_generator()`: Crea instancia del generador
- `get_supported_frameworks()`: Lista de frameworks
- `get_supported_model_types()`: Lista de tipos de modelos
- `validate_generator_config()`: Valida configuración
- `get_generator_info()`: Información del generador

### Funciones Helper
- `detect_framework_from_code()`: Detecta framework en código
- `detect_model_type_from_code()`: Detecta tipo de modelo
- `analyze_code_file()`: Analiza archivo de código
- `suggest_generator_config()`: Sugiere configuración
- `get_framework_info()`: Información de framework
- `get_model_type_info()`: Información de tipo de modelo
- `generate_config_template()`: Genera plantilla

## 📚 Recursos

- Módulo principal: `core.deep_learning_generator`
- Helpers: `core.deep_learning_generator_helpers`
- Core: `core.deep_learning.core`

---

**Última actualización**: 2024


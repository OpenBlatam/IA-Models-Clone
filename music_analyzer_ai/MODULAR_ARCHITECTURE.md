# Modular Architecture - Music Analyzer AI

## Estructura Modular

El código ha sido refactorizado para seguir principios de diseño modular:

### 1. **Interfaces** (`interfaces/`)
Define contratos claros para todos los componentes:
- `IMusicModel` - Contrato para modelos
- `IMusicClassifier` - Contrato para clasificadores
- `IMusicEncoder` - Contrato para encoders
- `ITrainer` - Contrato para trainers
- `IMusicAnalyzer` - Contrato para analizadores
- `IInferenceEngine` - Contrato para inferencia

### 2. **Factories** (`factories/`)
Patrón Factory para crear instancias:
- `ModelFactory` - Crear modelos
- `TrainerFactory` - Crear trainers
- `AnalyzerFactory` - Crear analizadores
- `ConfigFactory` - Crear configuraciones

### 3. **Base Classes** (`base/`)
Implementaciones base compartidas:
- `BaseMusicModel` - Funcionalidad común para modelos
- `BaseTrainer` - Funcionalidad común para trainers
- `BaseMusicAnalyzer` - Funcionalidad común para analizadores

## Uso Modular

### Crear Modelo
```python
from music_analyzer_ai import create_model, create_config

# Crear configuración
config = create_config("model", {
    "model_type": "music_classifier",
    "input_dim": 169,
    "hidden_dims": [512, 256, 128],
    "output_dim": 10
})

# Crear modelo
model = create_model("music_classifier", config)
```

### Crear Trainer
```python
from music_analyzer_ai import create_trainer

# Crear trainer
trainer = create_trainer(
    trainer_type="fast",
    model=model,
    config={
        "learning_rate": 1e-4,
        "compile_model": True
    }
)
```

### Crear Analyzer
```python
from music_analyzer_ai import create_analyzer

# Crear analyzer
analyzer = create_analyzer(
    analyzer_type="deep",
    config={"device": "cuda", "compile_models": True}
)
```

## Ventajas de la Arquitectura Modular

1. **Separación de Responsabilidades**: Cada módulo tiene una responsabilidad clara
2. **Intercambiabilidad**: Fácil cambiar implementaciones
3. **Testabilidad**: Interfaces permiten mocking fácil
4. **Extensibilidad**: Fácil agregar nuevos tipos
5. **Mantenibilidad**: Código más organizado y fácil de mantener

## Extender el Sistema

### Registrar Nuevo Modelo
```python
from music_analyzer_ai import ModelFactory
from music_analyzer_ai.interfaces import IMusicModel

class MyCustomModel(IMusicModel):
    # Implementación
    pass

ModelFactory.register("my_custom_model", MyCustomModel)
```

### Registrar Nuevo Trainer
```python
from music_analyzer_ai import TrainerFactory
from music_analyzer_ai.interfaces import ITrainer

class MyCustomTrainer(ITrainer):
    # Implementación
    pass

TrainerFactory.register("my_custom_trainer", MyCustomTrainer)
```

## Principios Aplicados

- **SOLID**: Single Responsibility, Open/Closed, Liskov Substitution
- **Dependency Injection**: Factories permiten inyección de dependencias
- **Interface Segregation**: Interfaces específicas y pequeñas
- **Factory Pattern**: Creación centralizada de instancias
- **Template Method**: Base classes con métodos template














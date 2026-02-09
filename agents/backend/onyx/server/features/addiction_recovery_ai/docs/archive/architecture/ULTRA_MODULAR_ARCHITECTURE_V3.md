# Ultra Modular Architecture V3 - Layered Design

## 🏗️ Arquitectura Ultra-Modular por Capas

Esta versión implementa una arquitectura ultra-modular con separación clara de responsabilidades en capas distintas.

## 📐 Estructura de Capas

```
┌─────────────────────────────────────────────────────────┐
│  Layer 6: Interface Layer                                │
│  - API Handlers, Request Processing, Response Formatting │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 5: Service Layer                                  │
│  - Business Logic, Service Orchestration                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 4: Inference Layer                                │
│  - Inference Engines, Predictors, Batch Processing        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Training Layer                                 │
│  - Trainers, Optimizers, Schedulers                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Model Layer                                    │
│  - Model Definitions, Builders, Configurations           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Data Layer                                     │
│  - Data Processing, Validation, Pipelines               │
└─────────────────────────────────────────────────────────┘
```

## 🔧 Componentes por Capa

### Layer 1: Data Layer (`core/layers/data_layer.py`)

**Responsabilidades:**
- Procesamiento de datos
- Validación de datos
- Pipelines de transformación
- Carga de datasets

**Componentes:**
- `DataProcessor` - Procesador base con chaining
- `NormalizationProcessor` - Normalización de datos
- `TokenizationProcessor` - Tokenización de texto
- `PaddingProcessor` - Padding de secuencias
- `DataValidator` - Validación de datos
- `DatasetFactory` - Factory para datasets
- `DataPipeline` - Pipeline composable
- `DataLoaderFactory` - Factory para data loaders

**Ejemplo:**
```python
from addiction_recovery_ai.core.layers import DataPipeline, NormalizationProcessor

# Crear pipeline
pipeline = DataPipeline()
pipeline.add_processor(NormalizationProcessor())
pipeline.add_processor(PaddingProcessor(max_length=128))

# Procesar datos
processed = pipeline.process(data)
```

### Layer 2: Model Layer (`core/layers/model_layer.py`)

**Responsabilidades:**
- Definición de modelos
- Configuración de modelos
- Construcción de modelos
- Carga de modelos

**Componentes:**
- `ModelConfig` - Configuración de modelos
- `ModelRegistry` - Registro centralizado
- `ModelBuilder` - Builder pattern fluido
- `ModelFactory` - Factory simple
- `ModelLoader` - Carga de modelos

**Ejemplo:**
```python
from addiction_recovery_ai.core.layers import ModelBuilder, ModelRegistry

# Registrar modelo
ModelRegistry.register_model("RecoveryPredictor", RecoveryProgressPredictor)

# Construir modelo
model = (ModelBuilder()
    .with_config(input_size=10, hidden_size=128)
    .with_device(torch.device("cuda"))
    .with_mixed_precision(True)
    .build("RecoveryPredictor"))
```

### Layer 3: Training Layer (`core/layers/training_layer.py`)

**Responsabilidades:**
- Loops de entrenamiento
- Optimizadores
- Schedulers
- Configuración de entrenamiento

**Componentes:**
- `TrainingConfig` - Configuración de entrenamiento
- `OptimizerFactory` - Factory de optimizadores
- `SchedulerFactory` - Factory de schedulers
- `TrainingPipeline` - Pipeline de entrenamiento
- `TrainerFactory` - Factory de trainers

**Ejemplo:**
```python
from addiction_recovery_ai.core.layers import TrainingPipeline, TrainingConfig

# Crear configuración
config = TrainingConfig(
    num_epochs=50,
    learning_rate=1e-3,
    use_mixed_precision=True
)

# Crear pipeline
pipeline = TrainingPipeline(model, config)
pipeline.set_criterion(nn.MSELoss())
pipeline.set_scheduler("reduce_on_plateau", patience=5)

# Entrenar
pipeline.train_epoch(train_loader)
```

### Layer 4: Inference Layer (`core/layers/inference_layer.py`)

**Responsabilidades:**
- Inferencia de modelos
- Procesamiento por lotes
- Optimizaciones de inferencia
- Pipelines de predicción

**Componentes:**
- `InferenceEngine` - Motor de inferencia
- `BatchProcessor` - Procesador de lotes
- `PredictorFactory` - Factory de predictores
- `InferencePipeline` - Pipeline de inferencia

**Ejemplo:**
```python
from addiction_recovery_ai.core.layers import InferenceEngine, InferencePipeline

# Crear engine
engine = InferenceEngine(model, use_mixed_precision=True)

# Crear pipeline
pipeline = InferencePipeline(engine)
pipeline.add_preprocessor(normalize_data)
pipeline.add_postprocessor(format_output)

# Inferir
result = pipeline.process(inputs)
```

### Layer 5: Service Layer (`core/layers/service_layer.py`)

**Responsabilidades:**
- Lógica de negocio
- Orquestación de servicios
- Dependency injection
- Registro de servicios

**Componentes:**
- `ServiceConfig` - Configuración de servicios
- `ServiceRegistry` - Registro de servicios
- `ServiceContainer` - Contenedor DI
- `ServiceFactory` - Factory de servicios

**Ejemplo:**
```python
from addiction_recovery_ai.core.layers import ServiceContainer, ServiceFactory

# Registrar servicio
container = ServiceContainer()
container.register("SentimentService", SentimentAnalyzerService, singleton=True)

# Crear servicio
factory = ServiceFactory(container)
service = factory.create("SentimentService")
```

### Layer 6: Interface Layer (`core/layers/interface_layer.py`)

**Responsabilidades:**
- Manejo de APIs
- Procesamiento de requests
- Formateo de responses
- Validación de entrada

**Componentes:**
- `RequestProcessor` - Procesador de requests
- `ResponseFormatter` - Formateador de responses
- `APIHandler` - Handler de API
- `InterfaceFactory` - Factory de interfaces

**Ejemplo:**
```python
from addiction_recovery_ai.core.layers import InterfaceFactory

# Crear handler
handler = InterfaceFactory.create_handler(
    service=sentiment_service,
    validate_requests=True,
    response_format="json"
)

# Manejar request
response = handler.handle({"text": "I'm feeling great!"})
```

## 🔌 Dependency Injection (`core/layers/dependency_injection.py`)

**Sistema de Inyección de Dependencias:**

```python
from addiction_recovery_ai.core.layers import (
    DependencyContainer,
    inject_dependencies,
    register_service
)

# Registrar servicios
container = DependencyContainer()
container.register_singleton("model", my_model)
container.register_factory("processor", create_processor)

# Usar decorator
@inject_dependencies
def predict(model: nn.Module, processor: DataProcessor, text: str):
    processed = processor.process(text)
    return model(processed)

# Registrar servicio con decorator
@register_service("SentimentService", singleton=True)
class SentimentAnalyzerService:
    def __init__(self):
        self.model = load_model()
```

## 📋 Interfaces y Protocolos (`core/layers/interfaces.py`)

**Protocolos Type-Safe:**

```python
from addiction_recovery_ai.core.layers.interfaces import (
    IModel,
    IPredictor,
    IService,
    IAPIHandler
)

# Los protocolos definen contratos sin implementación
# Permiten type checking y dependency injection
```

## 🎯 Ventajas de esta Arquitectura

1. **Separación de Responsabilidades**: Cada capa tiene una responsabilidad clara
2. **Testabilidad**: Componentes pueden ser testeados independientemente
3. **Reutilización**: Componentes pueden ser reutilizados en diferentes contextos
4. **Flexibilidad**: Fácil cambiar implementaciones sin afectar otras capas
5. **Escalabilidad**: Fácil agregar nuevas funcionalidades
6. **Mantenibilidad**: Código más fácil de entender y mantener

## 📦 Uso Completo

```python
from addiction_recovery_ai.core.layers import (
    DataPipeline,
    ModelBuilder,
    TrainingPipeline,
    InferenceEngine,
    ServiceContainer,
    InterfaceFactory
)

# 1. Data Layer
pipeline = DataPipeline()
pipeline.add_processor(NormalizationProcessor())

# 2. Model Layer
model = ModelBuilder().with_config(...).build("RecoveryPredictor")

# 3. Training Layer
trainer = TrainingPipeline(model, config)

# 4. Inference Layer
engine = InferenceEngine(model)

# 5. Service Layer
container = ServiceContainer()
container.register("Engine", engine)

# 6. Interface Layer
handler = InterfaceFactory.create_handler(service)
```

## 🔄 Migración desde Arquitectura Anterior

La nueva arquitectura es compatible con la anterior pero más modular:

```python
# Antes
from addiction_recovery_ai import create_sentiment_analyzer
analyzer = create_sentiment_analyzer()

# Ahora (más modular)
from addiction_recovery_ai.core.layers import ModelBuilder
model = ModelBuilder().build("SentimentAnalyzer")
```

---

**Version**: 3.6.0  
**Status**: Ultra Modular ✅  
**Last Updated**: 2025




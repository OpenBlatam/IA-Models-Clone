# 🎯 Mejores Prácticas de Deep Learning

Este documento describe las mejores prácticas implementadas en el sistema de Deep Learning.

## 📋 Tabla de Contenidos

1. [Arquitectura del Código](#arquitectura-del-código)
2. [Manejo de Dispositivos](#manejo-de-dispositivos)
3. [Mixed Precision Training](#mixed-precision-training)
4. [Inicialización de Pesos](#inicialización-de-pesos)
5. [Optimizadores y Schedulers](#optimizadores-y-schedulers)
6. [Gradient Management](#gradient-management)
7. [Early Stopping](#early-stopping)
8. [Checkpointing](#checkpointing)
9. [Error Handling](#error-handling)
10. [Logging](#logging)

## 🏗️ Arquitectura del Código

### Estructura Modular

```
core/
├── base_model_service.py      # Clase base para servicios
├── utils/
│   ├── model_utils.py         # Utilidades de modelos
│   └── training_utils.py       # Utilidades de entrenamiento
├── config/
│   └── model_config.py        # Configuraciones centralizadas
└── [servicios específicos]
```

### Principios

1. **Separación de Responsabilidades**: Cada módulo tiene una responsabilidad única
2. **Reutilización**: Utilidades comunes en módulos compartidos
3. **Configuración Centralizada**: Configuraciones en archivos dedicados
4. **Clases Base**: Funcionalidad común en clases base

## 💻 Manejo de Dispositivos

### Configuración Automática

```python
from core.base_model_service import DeviceConfig, BaseModelService

# Configuración automática
device_config = DeviceConfig(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_mixed_precision=True,
    torch_dtype=torch.float16,
)
```

### Optimizaciones de GPU

- **cuDNN Benchmark**: Habilitado automáticamente para convoluciones más rápidas
- **Memory Pinning**: Para transferencias más rápidas CPU→GPU
- **Device Mapping**: Automático con `device_map="auto"`

## 🚀 Mixed Precision Training

### Uso Correcto

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# En el loop de entrenamiento
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Beneficios

- **2x más rápido** en GPUs modernas (V100, A100)
- **Menor uso de memoria** (float16 vs float32)
- **Sin pérdida de precisión** significativa

## 🎲 Inicialización de Pesos

### Métodos Disponibles

```python
from core.utils.model_utils import initialize_weights

# Xavier (para activaciones lineales/tanh)
initialize_weights(model, method="xavier_uniform")

# Kaiming (para ReLU)
initialize_weights(model, method="kaiming_normal")

# Ortogonal (para RNNs)
initialize_weights(model, method="orthogonal")
```

### Reglas de Oro

- **Xavier**: Para activaciones lineales, tanh, sigmoid
- **Kaiming**: Para ReLU, LeakyReLU
- **Ortogonal**: Para RNNs, LSTMs
- **Normal pequeño**: Para embeddings

## ⚙️ Optimizadores y Schedulers

### Creación Estándar

```python
from core.utils.training_utils import create_optimizer, create_scheduler

# Optimizador
optimizer = create_optimizer(
    model,
    optimizer_type="adamw",
    learning_rate=1e-3,
    weight_decay=1e-5
)

# Scheduler
scheduler = create_scheduler(
    optimizer,
    scheduler_type="cosine",
    T_max=num_epochs
)
```

### Recomendaciones

- **AdamW**: Generalmente mejor que Adam (weight decay correcto)
- **Cosine Annealing**: Excelente para fine-tuning
- **ReduceLROnPlateau**: Útil cuando el loss se estanca

## 📊 Gradient Management

### Gradient Clipping

```python
from core.utils.model_utils import clip_gradients

# Clipping por norma
grad_norm = clip_gradients(model, max_norm=1.0)

# Verificar gradientes
from core.utils.model_utils import get_gradient_norm
grad_norm = get_gradient_norm(model)
```

### Gradient Accumulation

```python
# Para batches grandes sin más memoria
gradient_accumulation_steps = 4

for batch_idx, batch in enumerate(dataloader):
    loss = loss / gradient_accumulation_steps
    loss.backward()
    
    if (batch_idx + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 🛑 Early Stopping

### Implementación

```python
from core.utils.training_utils import EarlyStopping

early_stopping = EarlyStopping(
    patience=5,
    min_delta=0.001,
    mode="min",
    restore_best_weights=True
)

# En el loop de entrenamiento
if early_stopping(val_loss, model):
    print("Early stopping triggered")
    break
```

## 💾 Checkpointing

### Guardar Checkpoints

```python
from core.base_model_service import BaseModelService

service = BaseModelService()

service.save_model_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    loss=loss,
    filepath=f"checkpoint_epoch_{epoch}.pt"
)
```

### Cargar Checkpoints

```python
result = service.load_model_checkpoint(
    model=model,
    optimizer=optimizer,
    filepath="checkpoint.pt"
)
```

## 🐛 Error Handling

### Detección de Anomalías

```python
# Habilitar durante debugging
service.enable_anomaly_detection_mode()

# Deshabilitar en producción
service.disable_anomaly_detection_mode()
```

### Verificación de NaN/Inf

```python
from core.utils.model_utils import check_for_nan_inf

result = check_for_nan_inf(model)
if result["has_nan"]:
    print(f"NaN found in: {result['nan_params']}")
```

## 📝 Logging

### Mejores Prácticas

```python
import logging

logger = logging.getLogger(__name__)

# Info para eventos importantes
logger.info("Model loaded successfully")

# Warning para situaciones no críticas
logger.warning("CUDA not available, using CPU")

# Error para excepciones
logger.error(f"Error loading model: {e}", exc_info=True)
```

## 🔧 Configuración Recomendada

### Para Entrenamiento Estándar

```python
from core.config.model_config import TrainingConfig, OptimizerConfig

config = TrainingConfig(
    num_epochs=10,
    batch_size=32,
    optimizer=OptimizerConfig(
        type="adamw",
        learning_rate=2e-5,
        weight_decay=1e-5
    ),
    use_mixed_precision=True,
    max_grad_norm=1.0,
    early_stopping_patience=5,
)
```

### Para Fine-tuning

```python
config = TrainingConfig(
    num_epochs=3,
    batch_size=8,
    gradient_accumulation_steps=4,  # Batch efectivo de 32
    optimizer=OptimizerConfig(
        type="adamw",
        learning_rate=5e-5,
    ),
    scheduler=SchedulerConfig(
        type="cosine",
        T_max=3
    ),
)
```

## 🔍 Validación y Debugging

### Validación de Configuraciones

```python
from core.utils.validation_utils import (
    validate_model_config,
    validate_training_config,
    validate_model_output
)

# Validar configuración de modelo
is_valid, errors = validate_model_config({
    "input_size": 784,
    "output_size": 10,
    "dropout": 0.2
})

# Validar configuración de entrenamiento
is_valid, errors = validate_training_config({
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate": 1e-3
})

# Validar salida del modelo
is_valid, error, output = validate_model_output(
    model,
    input_shape=(784,),
    expected_output_shape=(10,)
)
```

### Validación de Gradientes

```python
from core.utils.validation_utils import validate_gradients

# Después de backward()
is_valid, grad_info = validate_gradients(model)
if not is_valid:
    print(f"Gradient issues: {grad_info}")
```

## 📊 Performance y Profiling

### Medir Tiempo de Ejecución

```python
from core.utils.performance_utils import timer

with timer("Training epoch"):
    train_one_epoch(...)
```

### Perfilar Modelo

```python
from core.utils.performance_utils import profile_model

stats = profile_model(
    model,
    input_shape=(784,),
    num_runs=100
)
print(f"Throughput: {stats['throughput']} samples/sec")
```

### Monitorear Memoria

```python
from core.utils.performance_utils import get_memory_usage, clear_cache

# Obtener uso de memoria
memory = get_memory_usage(device)
print(f"GPU Memory: {memory['allocated_gb']:.2f} GB")

# Limpiar caché
clear_cache(device)
```

## 📦 Manejo de Datos

### Crear DataLoaders Optimizados

```python
from core.utils.data_utils import create_dataloader, create_data_splits

# Dividir dataset
train_ds, val_ds, test_ds = create_data_splits(
    dataset,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Crear DataLoader optimizado
train_loader = create_dataloader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # Más rápido para GPU
)
```

### Balancear Dataset

```python
from core.utils.data_utils import balance_dataset, get_class_weights

# Balancear dataset
balanced_ds = balance_dataset(dataset, labels, method="undersample")

# Obtener pesos de clases
class_weights = get_class_weights(labels)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

## 📚 Referencias

- [PyTorch Best Practices](https://pytorch.org/docs/stable/notes/best_practices.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)


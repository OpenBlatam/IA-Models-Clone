# 🏗️ Arquitectura Modular - TruthGPT Optimization Core

Este documento describe la nueva arquitectura modular del sistema, que separa responsabilidades en módulos independientes y extensibles.

## 📋 Visión General

La arquitectura modular divide el código en módulos independientes con responsabilidades claras:

```
optimization_core/
├── core/                    # Módulos centrales
│   ├── config.py           # Gestión de configuración
│   └── interfaces.py       # Interfaces base (ABCs)
├── data/                    # Gestión de datos
│   ├── dataset_manager.py  # Carga de datasets
│   ├── data_loader_factory.py  # Creación de DataLoaders
│   └── collators.py        # Funciones de collation
├── models/                  # Gestión de modelos
│   ├── model_manager.py    # Carga/guardado de modelos
│   └── model_builder.py     # Builder para modelos
├── training/                # Componentes de entrenamiento
│   ├── evaluator.py        # Evaluación de modelos
│   ├── checkpoint_manager.py  # Gestión de checkpoints
│   ├── ema_manager.py      # Exponential Moving Average
│   └── training_loop.py    # Loop de entrenamiento
└── inference/              # Inferencia (próximamente)
```

## 🎯 Principios de Diseño

### 1. Separación de Responsabilidades
Cada módulo tiene una responsabilidad única y bien definida:
- **Config**: Carga y validación de configuración
- **Data**: Gestión de datasets y DataLoaders
- **Models**: Gestión del ciclo de vida de modelos
- **Training**: Componentes de entrenamiento
- **Inference**: Componentes de inferencia (separado)

### 2. Interfaces Base (ABCs)
Todas las funcionalidades principales implementan interfaces base:
- `BaseModelManager`: Contrato para gestión de modelos
- `BaseDataLoader`: Contrato para carga de datos
- `BaseEvaluator`: Contrato para evaluación
- `BaseCheckpointManager`: Contrato para checkpoints
- `BaseTrainer`: Contrato para entrenamiento

### 3. Patrón Builder
Los módulos complejos usan el patrón Builder para configuración fluida:
```python
from models.model_builder import ModelBuilder

model = (ModelBuilder()
    .with_model_name("gpt2")
    .with_lora(enabled=True, r=16)
    .with_multi_gpu(enabled=True)
    .build())
```

### 4. Factory Pattern
Fábricas para crear objetos complejos:
```python
from data.data_loader_factory import DataLoaderFactory

loader = DataLoaderFactory.create_train_loader(
    texts=train_texts,
    tokenizer=tokenizer,
    max_length=512,
    batch_size=8,
    bucket_by_length=True,
)
```

## 📦 Módulos Detallados

### Core Module (`core/`)

#### `config.py`
- `TrainerConfig`: Configuración completa tipo-safe
- `ConfigManager`: Carga y validación de YAML
- Sub-configs: `ModelConfig`, `TrainingConfig`, `DataConfig`, etc.

**Uso:**
```python
from core.config import ConfigManager

config = ConfigManager.load_config("configs/llm_default.yaml")
print(config.model.name_or_path)
print(config.training.learning_rate)
```

#### `interfaces.py`
Define contratos (ABCs) que todas las implementaciones deben seguir.

**Interfaces:**
- `BaseModelManager`: `load_model()`, `save_model()`, `get_model_device()`
- `BaseDataLoader`: `create_train_loader()`, `create_val_loader()`
- `BaseEvaluator`: `evaluate()`, `compute_metrics()`
- `BaseCheckpointManager`: `save_checkpoint()`, `load_checkpoint()`
- `BaseTrainer`: `train_step()`, `train_epoch()`, `should_stop_early()`

### Data Module (`data/`)

#### `dataset_manager.py`
Gestiona la carga de datasets desde diferentes fuentes.

**Fuentes soportadas:**
- HuggingFace (`hf`)
- JSONL files (`jsonl`)
- Text files (`text`)

**Uso:**
```python
from data.dataset_manager import DatasetManager

train_texts, val_texts = DatasetManager.load_dataset(
    source="hf",
    dataset_name="wikitext",
    subset="wikitext-2-raw-v1",
    text_field="text",
)
```

#### `data_loader_factory.py`
Factory para crear DataLoaders optimizados.

**Características:**
- Length bucketing automático
- Configuración de workers
- Dynamic padding
- Builder pattern

**Uso:**
```python
from data.data_loader_factory import DataLoaderBuilder

loader = (DataLoaderBuilder()
    .with_texts(train_texts)
    .with_tokenizer(tokenizer)
    .with_max_length(512)
    .with_batch_size(8)
    .with_length_bucketing(enabled=True, bins=[64, 128, 256, 512])
    .build_train())
```

### Models Module (`models/`)

#### `model_manager.py`
Gestiona el ciclo de vida completo de modelos.

**Funcionalidades:**
- Carga de modelos (HuggingFace Hub o local)
- Guardado de modelos
- Soporte para LoRA
- Multi-GPU (DataParallel)
- torch.compile
- Configuración de device settings (TF32, SDPA)

**Uso:**
```python
from models.model_manager import ModelManager

manager = ModelManager()
model = manager.load_model(
    model_name="gpt2",
    torch_dtype=torch.bfloat16,
    gradient_checkpointing=True,
    lora_config={"enabled": True, "r": 16, "alpha": 32},
)
```

#### `model_builder.py`
Builder pattern para construir modelos con configuración fluida.

**Uso:**
```python
from models.model_builder import ModelBuilder

model = (ModelBuilder()
    .with_model_name("gpt2")
    .with_dtype(torch.bfloat16)
    .with_gradient_checkpointing(enabled=True)
    .with_lora(enabled=True, r=16, alpha=32)
    .with_multi_gpu(enabled=True)
    .with_torch_compile(enabled=True, mode="max-autotune")
    .with_device_settings(allow_tf32=True)
    .build())
```

### Training Module (`training/`)

#### `evaluator.py`
Evalúa modelos en datasets de validación.

**Características:**
- Soporte para AMP
- Manejo de DataParallel
- Cálculo de perplexity
- Manejo robusto de errores

**Uso:**
```python
from training.evaluator import Evaluator

evaluator = Evaluator(use_amp=True, amp_dtype=torch.bfloat16)
metrics = evaluator.evaluate(model, val_loader, device)
print(f"Loss: {metrics['loss']}, Perplexity: {metrics['perplexity']}")
```

#### `checkpoint_manager.py`
Gestiona guardado y carga de checkpoints.

**Características:**
- Guardado completo de estado (model, optimizer, scheduler, scaler, EMA)
- Búsqueda de último checkpoint
- Pruning de checkpoints antiguos
- Soporte para SafeTensors

**Uso:**
```python
from training.checkpoint_manager import CheckpointManager

ckpt_manager = CheckpointManager(output_dir="runs/experiment")
ckpt_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    step=global_step,
    path="runs/experiment/checkpoint_step_1000",
)

# Cargar
state = ckpt_manager.load_checkpoint(
    path="runs/experiment/checkpoint_step_1000",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
)
```

#### `ema_manager.py`
Gestiona Exponential Moving Average de pesos.

**Uso:**
```python
from training.ema_manager import EMAManager

ema = EMAManager(decay=0.999, model=model)

# Durante entrenamiento
ema.update(model)

# Para evaluación
ema.apply_to_model(model)
metrics = evaluator.evaluate(model, val_loader, device)
ema.restore_from_backup(model)
```

#### `training_loop.py`
Implementa el loop de entrenamiento básico.

**Uso:**
```python
from training.training_loop import TrainingLoop

loop = TrainingLoop(
    use_amp=True,
    amp_dtype=torch.bfloat16,
    max_grad_norm=1.0,
    grad_accum_steps=2,
)

epoch_metrics = loop.train_epoch(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    step_callback=lambda step, metrics, lr: print(f"Step {step}: loss={metrics['loss']:.4f}"),
)
```

## 🔄 Integración Modular

### Ejemplo Completo de Entrenamiento

```python
from core.config import ConfigManager
from data.dataset_manager import DatasetManager
from data.data_loader_factory import DataLoaderFactory
from models.model_builder import ModelBuilder
from training.evaluator import Evaluator
from training.checkpoint_manager import CheckpointManager
from training.ema_manager import EMAManager
from training.training_loop import TrainingLoop
from transformers import AutoTokenizer

# 1. Cargar configuración
config = ConfigManager.load_config("configs/llm_default.yaml")

# 2. Cargar datos
train_texts, val_texts = DatasetManager.load_dataset(
    source=config.data.source,
    dataset_name=config.data.dataset,
    subset=config.data.subset,
    text_field=config.data.text_field,
)

# 3. Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)

# 4. Crear DataLoaders
train_loader = DataLoaderFactory.create_train_loader(
    texts=train_texts,
    tokenizer=tokenizer,
    max_length=config.data.max_seq_len,
    batch_size=config.training.train_batch_size,
    collate_type=config.data.collate,
    bucket_by_length=config.data.bucket_by_length,
    bucket_bins=config.data.bucket_bins,
    num_workers=config.data.num_workers,
)

val_loader = DataLoaderFactory.create_val_loader(
    texts=val_texts,
    tokenizer=tokenizer,
    max_length=config.data.max_seq_len,
    batch_size=config.training.eval_batch_size,
    collate_type=config.data.collate,
    num_workers=config.data.num_workers,
)

# 5. Construir modelo
model = (ModelBuilder()
    .with_model_name(config.model.name_or_path)
    .with_dtype(torch.bfloat16 if config.training.mixed_precision == "bf16" else None)
    .with_gradient_checkpointing(config.model.gradient_checkpointing)
    .with_lora(
        enabled=config.model.lora_enabled,
        r=config.model.lora_r,
        alpha=config.model.lora_alpha,
    )
    .with_multi_gpu(config.hardware.multi_gpu)
    .with_device_settings(allow_tf32=config.training.allow_tf32)
    .build())

# 6. Configurar componentes de entrenamiento
evaluator = Evaluator(
    use_amp=config.training.mixed_precision != "none",
    amp_dtype=torch.bfloat16 if config.training.mixed_precision == "bf16" else torch.float16,
)
ckpt_manager = CheckpointManager(config.output_dir)
ema = EMAManager(decay=config.ema.decay, model=model) if config.ema.enabled else None
training_loop = TrainingLoop(
    use_amp=config.training.mixed_precision != "none",
    amp_dtype=torch.bfloat16 if config.training.mixed_precision == "bf16" else None,
    max_grad_norm=config.training.max_grad_norm,
    grad_accum_steps=config.training.grad_accum_steps,
)

# 7. Loop de entrenamiento (simplificado)
for epoch in range(config.training.epochs):
    # Training
    epoch_metrics = training_loop.train_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    
    # Update EMA
    if ema:
        ema.update(model)
    
    # Evaluation
    if ema:
        ema.apply_to_model(model)
    
    val_metrics = evaluator.evaluate(model, val_loader, device)
    
    if ema:
        ema.restore_from_backup(model)
    
    # Save checkpoint
    if epoch % 10 == 0:
        ckpt_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=epoch,
            path=f"{config.output_dir}/checkpoint_epoch_{epoch}",
        )
```

## ✅ Ventajas de la Arquitectura Modular

1. **Separación de Responsabilidades**: Cada módulo tiene un propósito claro
2. **Testabilidad**: Cada módulo puede ser testeado independientemente
3. **Extensibilidad**: Fácil agregar nuevas implementaciones (nuevos datasets, optimizadores, etc.)
4. **Reutilización**: Los módulos pueden ser reutilizados en diferentes contextos
5. **Mantenibilidad**: Cambios en un módulo no afectan otros
6. **Type Safety**: Uso extensivo de type hints y dataclasses
7. **Interfaces Claras**: ABCs definen contratos explícitos

## 🔮 Próximos Pasos

1. **Módulo de Inferencia**: Separar lógica de inferencia/generación
2. **Métricas Personalizadas**: Sistema extensible de métricas
3. **Callbacks Modulares**: Sistema de callbacks más flexible
4. **Testing**: Tests unitarios para cada módulo
5. **Documentación**: Docstrings completos y ejemplos

## 📚 Referencias

- [Python ABC Documentation](https://docs.python.org/3/library/abc.html)
- [Design Patterns: Builder](https://refactoring.guru/design-patterns/builder)
- [Design Patterns: Factory](https://refactoring.guru/design-patterns/factory-method)



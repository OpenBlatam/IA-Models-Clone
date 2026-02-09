# Funcionalidades Avanzadas

## 🚀 Mejoras Implementadas

### 1. Distributed Training
Soporte para entrenamiento multi-GPU usando DataParallel y DistributedDataParallel.

```python
from core.routing_training import DistributedTrainer

distributed_trainer = DistributedTrainer(
    model=model,
    use_ddp=True,  # DistributedDataParallel (recomendado)
    use_dp=False   # DataParallel (más simple)
)
```

**Características:**
- ✅ DistributedDataParallel (DDP) para máxima eficiencia
- ✅ DataParallel (DP) para simplicidad
- ✅ Manejo automático de dispositivos
- ✅ Checkpointing compatible

### 2. LoRA (Low-Rank Adaptation)
Fine-tuning eficiente con adaptación de bajo rango.

```python
from core.routing_models.lora import apply_lora_to_model, count_lora_parameters

# Aplicar LoRA
model = apply_lora_to_model(model, rank=8, alpha=16.0)

# Ver estadísticas
stats = count_lora_parameters(model)
print(f"Parámetros entrenables: {stats['trainable_ratio']:.2%}")
```

**Ventajas:**
- ✅ Reduce parámetros entrenables en ~90%
- ✅ Mantiene rendimiento similar
- ✅ Entrenamiento más rápido
- ✅ Menor uso de memoria

### 3. Hyperparameter Optimization
Optimización automática de hiperparámetros usando Optuna.

```python
from core.routing_training import HyperparameterOptimizer, create_objective_function

optimizer = HyperparameterOptimizer(n_trials=50)
objective = create_objective_function(train_loader, val_loader, model_factory)
study = optimizer.optimize(objective)

print(f"Mejores parámetros: {optimizer.get_best_params()}")
```

**Características:**
- ✅ Búsqueda bayesiana (TPE)
- ✅ Pruning automático
- ✅ Visualización de resultados
- ✅ Soporte para múltiples objetivos

### 4. Model Ensembling
Ensamblaje de múltiples modelos para mejor rendimiento.

```python
from core.routing_models.ensemble import ModelEnsemble

ensemble = ModelEnsemble(
    models=[model1, model2, model3],
    weights=[0.4, 0.3, 0.3],
    voting_method="weighted_average"
)

# Predecir con incertidumbre
mean, std = ensemble.predict_with_uncertainty(input, return_std=True)
```

**Métodos:**
- ✅ Average: Promedio simple
- ✅ Weighted Average: Promedio ponderado
- ✅ Uncertainty Estimation: Estimación de incertidumbre
- ✅ Diversity Metrics: Métricas de diversidad

### 5. Advanced Callbacks

#### Learning Rate Finder
Encuentra el learning rate óptimo automáticamente.

```python
from core.routing_training.advanced_callbacks import LearningRateFinder

lr_finder = LearningRateFinder(min_lr=1e-8, max_lr=1.0, num_iterations=100)
trainer = RouteTrainer(..., callbacks=[lr_finder])
# Después del entrenamiento: lr_finder.best_lr
```

#### Gradient Monitor
Monitorea gradientes para detectar problemas.

```python
from core.routing_training.advanced_callbacks import GradientMonitor

grad_monitor = GradientMonitor(log_frequency=10, max_grad_norm=10.0)
```

#### Model EMA
Exponential Moving Average para pesos del modelo.

```python
from core.routing_training.advanced_callbacks import ModelEMA

ema = ModelEMA(decay=0.9999)
# Mejora estabilidad y generalización
```

#### Profiler
Profiling de rendimiento para optimización.

```python
from core.routing_training.advanced_callbacks import ProfilerCallback

profiler = ProfilerCallback(profile_steps=10, output_dir="./profiles")
```

### 6. Gradient Accumulation
Entrenar con batches grandes usando acumulación.

```python
from core.routing_training import GradientAccumulator

accumulator = GradientAccumulator(accumulation_steps=4)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    
    if accumulator.should_update():
        optimizer.step()
        optimizer.zero_grad()
    
    accumulator.step()
```

## 📊 Comparación de Funcionalidades

| Funcionalidad | Parámetros Reducidos | Velocidad | Mejora Rendimiento |
|--------------|---------------------|-----------|-------------------|
| LoRA | ~90% | ⚡⚡⚡ | ✅ |
| Distributed Training | - | ⚡⚡⚡ | ✅ |
| Ensembling | +200% | ⚡ | ✅✅ |
| Hyperparameter Opt | - | ⚡ | ✅✅ |

## 🎯 Casos de Uso

### Fine-tuning Rápido
```python
# 1. Cargar modelo pre-entrenado
model = load_pretrained_model()

# 2. Aplicar LoRA
model = apply_lora_to_model(model, rank=8)

# 3. Entrenar solo parámetros LoRA
trainer = RouteTrainer(model, ...)
trainer.train()
```

### Entrenamiento a Gran Escala
```python
# 1. Configurar distributed training
distributed = DistributedTrainer(model, use_ddp=True)

# 2. Usar gradient accumulation
accumulator = GradientAccumulator(accumulation_steps=8)

# 3. Entrenar
trainer = RouteTrainer(distributed.get_model(), ...)
```

### Optimización Automática
```python
# 1. Crear optimizador
optimizer = HyperparameterOptimizer(n_trials=100)

# 2. Definir objetivo
objective = create_objective_function(...)

# 3. Optimizar
study = optimizer.optimize(objective)

# 4. Usar mejores parámetros
best_config = optimizer.get_best_params()
```

## 🔧 Configuración Recomendada

### Para Fine-tuning:
- LoRA rank: 8-16
- LoRA alpha: 16-32
- Learning rate: 1e-4 a 1e-3

### Para Entrenamiento desde Cero:
- Distributed training si hay múltiples GPUs
- Gradient accumulation para batches grandes
- Learning rate finder para encontrar LR óptimo

### Para Producción:
- Model ensembling para mejor rendimiento
- Model EMA para estabilidad
- Hyperparameter optimization para mejores resultados

## 📚 Referencias

- LoRA: [LoRA Paper](https://arxiv.org/abs/2106.09685)
- Optuna: [Optuna Documentation](https://optuna.org/)
- Distributed Training: [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

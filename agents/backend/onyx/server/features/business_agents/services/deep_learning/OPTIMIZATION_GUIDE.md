# Performance Optimization Guide

## 🚀 Optimizaciones Implementadas

### 1. Model Compilation (torch.compile)

```python
from deep_learning.utils.optimization import ModelOptimizer

# Compilar modelo para entrenamiento más rápido
model = ModelOptimizer.compile_model(model, mode="reduce-overhead")
```

**Modos disponibles:**
- `default`: Balance entre velocidad y compilación
- `reduce-overhead`: Reduce overhead de Python
- `max-autotune`: Máxima optimización (más lento de compilar)

### 2. Inference Optimization

```python
from deep_learning.utils.optimization import InferenceOptimizer

# Optimizar para inferencia
optimizer = InferenceOptimizer(model, device)
outputs = optimizer.batch_inference(inputs, batch_size=64, use_amp=True)
```

**Características:**
- Batching automático para inputs grandes
- Mixed precision automático
- Caching para inputs repetidos

### 3. Memory Optimization

```python
from deep_learning.utils.optimization import MemoryOptimizer

# Limpiar cache de GPU
MemoryOptimizer.clear_cache()

# Obtener uso de memoria
usage = MemoryOptimizer.get_memory_usage()
print(f"Allocated: {usage['allocated_gb']:.2f} GB")

# Configurar fracción de memoria
MemoryOptimizer.set_memory_fraction(0.95)
```

### 4. Optimized DataLoader

```python
from deep_learning.data import create_optimized_dataloader

# Crear DataLoader optimizado
loader = create_optimized_dataloader(
    dataset,
    batch_size=32,
    num_workers=4,  # Auto-detectado si None
    prefetch_factor=4,  # Más prefetch = más rápido
    persistent_workers=True
)
```

**Optimizaciones automáticas:**
- Auto-detección de num_workers óptimo
- pin_memory habilitado para GPU
- persistent_workers para evitar overhead
- prefetch_factor optimizado

### 5. Optimized Training Manager

```python
from deep_learning.training import OptimizedTrainingManager

# Usar trainer optimizado
trainer = OptimizedTrainingManager(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    compile_model=True,  # Compilar modelo
    optimize_memory=True  # Optimizar memoria
)

history = trainer.train()
```

**Mejoras:**
- Model compilation automático
- Progress bars con tqdm
- Memory cleanup periódico
- TF32 habilitado
- non_blocking transfers

### 6. Production Optimization

```python
from deep_learning.utils.optimization import optimize_model_for_production

# Optimización completa para producción
model = optimize_model_for_production(
    model,
    device=device,
    compile_model=True,
    enable_tf32=True,
    enable_flash_attention=True
)
```

## 📊 Mejoras de Rendimiento

### Entrenamiento
- **Model Compilation**: 1.2x - 2x más rápido
- **TF32**: 1.3x más rápido en GPUs Ampere+
- **Optimized DataLoader**: 1.5x - 2x más rápido data loading
- **Mixed Precision**: 1.5x - 2x más rápido, 50% menos memoria

### Inferencia
- **Model Compilation**: 1.5x - 3x más rápido
- **Batching**: Mejor utilización de GPU
- **Caching**: Instante para inputs repetidos

### Memoria
- **Mixed Precision**: 50% reducción
- **Memory Management**: Limpieza automática
- **Gradient Checkpointing**: Reducción adicional (si se implementa)

## 🎯 Mejores Prácticas

1. **Siempre usar Mixed Precision** en GPU
2. **Compilar modelos** para producción
3. **Optimizar DataLoaders** con prefetch y persistent workers
4. **Limpiar memoria** periódicamente durante entrenamiento
5. **Usar TF32** en GPUs Ampere+
6. **Profile antes de optimizar** para identificar bottlenecks

## 📝 Ejemplo Completo

```python
from deep_learning import DeepLearningService
from deep_learning.utils.optimization import optimize_model_for_production
from deep_learning.data import create_optimized_dataloader

# Inicializar servicio
service = DeepLearningService()

# Crear modelo
model = service.create_model("transformer")

# Optimizar modelo
model = optimize_model_for_production(
    model,
    device=service.device,
    compile_model=True
)

# Crear dataloaders optimizados
train_loader = create_optimized_dataloader(
    train_dataset,
    batch_size=32,
    prefetch_factor=4
)

# Entrenar con trainer optimizado
history = service.train_model(
    model,
    train_loader,
    val_loader,
    use_optimized=True  # Usa OptimizedTrainingManager
)
```




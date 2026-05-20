# Advanced Training Guide - Distributed Training & Performance Optimization

## 🚀 Nuevas Características

### 1. Distributed Training
- **DataParallel**: Para single-node multi-GPU
- **DistributedDataParallel (DDP)**: Para multi-node/multi-GPU
- Sincronización automática de métricas
- Distributed samplers para balanceo de datos

### 2. Gradient Accumulation
- Entrenamiento con batches efectivos más grandes
- Útil cuando la memoria GPU es limitada
- Configurable por número de pasos

### 3. Profiling y Debugging
- Profiling integrado con PyTorch Profiler
- Anomaly detection para detectar NaN/Inf
- Performance monitoring
- GPU utilization tracking

### 4. Optimizaciones de Performance
- DataLoader optimizado con prefetch
- Non-blocking transfers
- Persistent workers
- Memory pinning

## 📖 Uso de Distributed Training

### Single-Node Multi-GPU (DataParallel)

```python
from training import Trainer, create_data_loaders
from models import ViTSkinAnalyzer

# Crear modelo
model = ViTSkinAnalyzer(num_conditions=6, num_metrics=8)

# Crear data loaders
loaders = create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size=32,
    use_distributed=False  # DataParallel se activa automáticamente
)

# Crear trainer (automáticamente usa DataParallel si hay múltiples GPUs)
trainer = Trainer(
    model=model,
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    device="cuda",
    use_ddp=False  # Usa DataParallel en lugar de DDP
)
```

### Multi-Node Multi-GPU (DistributedDataParallel)

```python
import torch
from training import Trainer, setup_distributed, create_data_loaders
from models import ViTSkinAnalyzer

# Setup distributed training
rank, world_size, device = setup_distributed(
    rank=int(os.environ.get('RANK', -1)),
    world_size=int(os.environ.get('WORLD_SIZE', -1))
)

# Crear modelo
model = ViTSkinAnalyzer(num_conditions=6, num_metrics=8)

# Crear data loaders con distributed samplers
loaders = create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size=32,
    use_distributed=True  # Habilita distributed samplers
)

# Crear trainer con DDP
trainer = Trainer(
    model=model,
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    device=device,
    use_ddp=True,  # Usa DistributedDataParallel
    find_unused_parameters=False
)

# Entrenar
trainer.fit(optimizer, num_epochs=100, scheduler=scheduler)
```

### Script de Lanzamiento para DDP

```bash
# Single node, 4 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_script.py

# Multi-node
python -m torch.distributed.launch \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr="192.168.1.1" \
    --master_port=12355 \
    train_script.py
```

## 🔄 Gradient Accumulation

Útil cuando quieres un batch efectivo más grande pero no tienes suficiente memoria:

```python
from training import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    gradient_accumulation_steps=4,  # Acumula gradientes por 4 pasos
    batch_size=32  # Batch efectivo = 32 * 4 = 128
)

# El trainer automáticamente:
# 1. Divide la pérdida por gradient_accumulation_steps
# 2. Acumula gradientes por N pasos
# 3. Hace optimizer.step() cada N pasos
```

**Ejemplo:**
- Batch size: 32
- Gradient accumulation steps: 4
- **Batch efectivo**: 32 × 4 = 128

## 📊 Profiling y Performance Monitoring

### Profiling durante el Entrenamiento

```python
from training import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    enable_profiling=True  # Habilita PyTorch Profiler
)

# Los resultados se guardan en ./logs/profiler
# Visualizar con: tensorboard --logdir=./logs/profiler
```

### Anomaly Detection

```python
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    enable_anomaly_detection=True  # Detecta NaN/Inf automáticamente
)

# Si se detecta NaN/Inf, lanza excepción con stack trace
```

### Performance Monitoring Manual

```python
from utils.profiling import PerformanceMonitor, profile_region, profile_model

# Monitor de performance
monitor = PerformanceMonitor()

# Durante entrenamiento
for epoch in range(num_epochs):
    monitor.start("epoch")
    
    for batch in train_loader:
        monitor.start("forward")
        output = model(batch['image'])
        monitor.end("forward")
        
        monitor.start("backward")
        loss.backward()
        monitor.end("backward")
    
    monitor.end("epoch")
    
    # Ver estadísticas
    stats = monitor.get_all_stats()
    print(f"Epoch stats: {stats}")

# Profilear modelo completo
results = profile_model(
    model,
    input_shape=(1, 3, 224, 224),
    device="cuda",
    num_runs=100
)
print(f"Model performance: {results}")
```

### Context Manager para Profiling

```python
from utils.profiling import profile_region

with profile_region("data_loading"):
    batch = next(iter(train_loader))

with profile_region("forward_pass"):
    output = model(batch['image'])

with profile_region("backward_pass"):
    loss.backward()
```

### GPU Utilization

```python
from utils.profiling import check_gpu_utilization, clear_gpu_cache

# Verificar uso de GPU
gpu_stats = check_gpu_utilization()
print(f"GPU Memory: {gpu_stats['memory_allocated_mb']:.2f}MB / {gpu_stats['memory_reserved_mb']:.2f}MB")

# Limpiar cache
clear_gpu_cache()
```

## ⚡ Optimizaciones de DataLoader

### DataLoader Optimizado

```python
from utils.profiling import optimize_data_loader

# Optimizar data loader existente
optimized_loader = optimize_data_loader(
    train_loader,
    num_workers=4,  # Auto-detect si None
    pin_memory=True,
    prefetch_factor=2
)
```

### Configuración Recomendada

```python
from training import create_data_loaders

loaders = create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size=32,
    num_workers=4,  # Número de workers
    pin_memory=True,  # Pin memory para transferencia rápida
    use_distributed=False
)

# El DataLoader incluye:
# - persistent_workers=True (mantiene workers entre epochs)
# - prefetch_factor=2 (prefetch de batches)
# - non_blocking transfers (para GPU)
```

## 🎯 Ejemplo Completo

### Entrenamiento con Todas las Mejoras

```python
import torch
from training import (
    Trainer,
    create_data_loaders,
    MultiTaskLoss,
    get_optimizer,
    get_scheduler,
    setup_distributed,
    is_main_process
)
from models import ViTSkinAnalyzer
from data import SkinDataset, get_train_transforms, get_val_transforms
from utils.profiling import PerformanceMonitor, check_gpu_utilization

# Setup distributed (opcional)
rank, world_size, device = setup_distributed()

# Verificar GPU
if is_main_process():
    gpu_stats = check_gpu_utilization()
    print(f"GPU Stats: {gpu_stats}")

# Crear modelo
model = ViTSkinAnalyzer(
    num_conditions=6,
    num_metrics=8,
    use_pretrained=True
)

# Crear datasets
train_dataset = SkinDataset(
    images=train_images,
    labels={'conditions': train_conditions, 'metrics': train_metrics},
    transform=get_train_transforms(target_size=(224, 224)),
    cache_images=True
)

val_dataset = SkinDataset(
    images=val_images,
    labels={'conditions': val_conditions, 'metrics': val_metrics},
    transform=get_val_transforms(target_size=(224, 224))
)

# Crear data loaders
loaders = create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    use_distributed=(world_size > 1)
)

# Crear trainer con todas las mejoras
trainer = Trainer(
    model=model,
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    device=device,
    use_mixed_precision=True,
    gradient_clip_val=1.0,
    gradient_accumulation_steps=2,  # Batch efectivo = 32 * 2 = 64
    early_stopping_patience=10,
    use_ddp=(world_size > 1),
    enable_profiling=is_main_process(),  # Solo en proceso principal
    enable_anomaly_detection=False  # Desactivar en producción
)

# Crear loss
loss_fn = MultiTaskLoss(
    condition_weight=1.0,
    metric_weight=1.0
)

# Crear optimizer y scheduler
optimizer = get_optimizer(
    model,
    optimizer_name="adamw",
    learning_rate=1e-4,
    weight_decay=1e-4
)

scheduler = get_scheduler(
    optimizer,
    scheduler_name="cosine",
    num_epochs=100
)

# Entrenar
if is_main_process():
    print("Starting training...")

trainer.fit(
    optimizer=optimizer,
    num_epochs=100,
    scheduler=scheduler,
    criterion=loss_fn,
    checkpoint_dir="./checkpoints"
)

if is_main_process():
    print("Training completed!")
```

## 📈 Mejores Prácticas

### 1. Distributed Training
- Usa DDP para multi-node, DataParallel para single-node
- Asegúrate de usar DistributedSampler
- Sincroniza procesos cuando sea necesario
- Guarda checkpoints solo en proceso principal

### 2. Gradient Accumulation
- Úsalo cuando la memoria GPU sea limitada
- Ajusta el learning rate si cambias el batch efectivo
- Batch efectivo = batch_size × gradient_accumulation_steps

### 3. Profiling
- Habilita profiling solo durante desarrollo
- Usa anomaly detection para debugging
- Monitorea GPU utilization regularmente
- Optimiza DataLoader para mejor throughput

### 4. Performance
- Usa pin_memory=True para GPU
- Configura num_workers apropiadamente (4-8 típicamente)
- Usa persistent_workers=True
- Prefetch batches con prefetch_factor=2

### 5. Memory Management
- Limpia GPU cache periódicamente
- Usa gradient checkpointing para modelos grandes
- Considera mixed precision training
- Monitorea memory leaks

## 🔍 Troubleshooting

### Problema: Out of Memory
**Solución:**
- Reduce batch_size
- Aumenta gradient_accumulation_steps
- Usa mixed precision
- Reduce num_workers

### Problema: Lento Data Loading
**Solución:**
- Aumenta num_workers
- Usa pin_memory=True
- Cache imágenes si es posible
- Optimiza transforms

### Problema: NaN/Inf Loss
**Solución:**
- Habilita anomaly_detection
- Reduce learning rate
- Aumenta gradient_clip_val
- Revisa datos de entrada

### Problema: GPUs No Utilizadas
**Solución:**
- Verifica que use_ddp o DataParallel esté activado
- Asegúrate de tener múltiples GPUs disponibles
- Verifica que batch_size sea suficiente

## 📚 Referencias

- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Gradient Accumulation](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation)

---

**Advanced Training Guide - Optimizado para Performance y Escalabilidad**














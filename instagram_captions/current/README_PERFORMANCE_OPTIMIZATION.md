# Performance Optimization System

Sistema completo de optimización de performance implementando DataParallel, DistributedDataParallel, gradient accumulation, mixed precision training y code profiling con mejores prácticas para workflows de deep learning.

## Características

- **Mixed Precision Training**: Automatic Mixed Precision (AMP) con `torch.cuda.amp`
- **Multi-GPU Training**: DataParallel y DistributedDataParallel automático
- **Gradient Accumulation**: Acumulación de gradientes para batch sizes efectivos grandes
- **Gradient Clipping**: Control automático de gradientes explosivos
- **Gradient Checkpointing**: Optimización de memoria para modelos grandes
- **Torch Compile**: Compilación JIT para optimización de código
- **Code Profiling**: Profiling avanzado con cProfile y torch.profiler
- **Memory Optimization**: Gestión inteligente de memoria GPU y CPU
- **Performance Monitoring**: Seguimiento continuo de métricas de performance

## Instalación

```bash
pip install -r requirements_performance_optimization.txt
```

## Uso Básico

### Configuración de Optimización

```python
from performance_optimization_system import OptimizationConfig, TrainingOptimizer

# Configurar optimizaciones
config = OptimizationConfig(
    enable_mixed_precision=True,      # Habilitar AMP
    enable_gradient_accumulation=True, # Acumulación de gradientes
    enable_data_parallel=True,        # Multi-GPU training
    enable_gradient_checkpointing=True, # Ahorrar memoria
    enable_torch_compile=True,        # Compilación JIT
    accumulation_steps=4,             # Pasos de acumulación
    max_grad_norm=1.0                 # Norma máxima de gradientes
)

# Inicializar optimizador de training
training_optimizer = TrainingOptimizer(config)
```

### Training Optimizado Completo

```python
# Ejecutar training loop optimizado
results = training_optimizer.optimize_training_loop(
    model=your_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=your_criterion,
    optimizer=your_optimizer,
    num_epochs=100,
    device="cuda"
)

# Obtener resultados
print(f"Final Train Loss: {results['training_stats']['train_losses'][-1]:.4f}")
print(f"Final Train Accuracy: {results['training_stats']['train_accuracies'][-1]:.2f}%")
print(f"Average Step Time: {results['optimization_summary']['step_time']['mean']:.4f}s")
```

## Componentes Principales

### PerformanceOptimizer

```python
from performance_optimization_system import PerformanceOptimizer

# Inicializar optimizador
optimizer = PerformanceOptimizer(config)

# Optimizar modelo
optimized_model = optimizer.optimize_model(model, device="cuda")

# Optimizar optimizador
optimized_optimizer = optimizer.optimize_optimizer(optimizer)

# Training step optimizado
metrics = optimizer.training_step(
    model=model,
    data=data,
    target=target,
    criterion=criterion,
    optimizer=optimizer,
    accumulation_step=batch_idx
)

# Obtener resumen de optimización
summary = optimizer.get_optimization_summary()
print(f"Step time promedio: {summary['step_time']['mean']:.4f}s")
print(f"Loss promedio: {summary['loss']['mean']:.4f}")
```

### CodeProfiler

```python
from performance_optimization_system import CodeProfiler

# Inicializar profiler
profiler = CodeProfiler(enable_profiling=True)

# Profilar función específica
with profiler.profile_function("training_step"):
    result = your_training_function()

# Profilar operaciones PyTorch
profiler.profile_torch_operations(model, input_tensor)

# Obtener resumen de profiling
profile_summary = profiler.get_profile_summary()
for func_name, results in profile_summary.items():
    print(f"{func_name}: {results['total_calls']} calls, "
          f"{results['total_time']:.4f}s total")
```

### MemoryOptimizer

```python
from performance_optimization_system import MemoryOptimizer

# Inicializar optimizador de memoria
memory_optimizer = MemoryOptimizer()

# Limpiar caches
memory_optimizer.clear_gpu_cache()
memory_optimizer.clear_cpu_cache()
memory_optimizer.clear_all_caches()

# Obtener uso de memoria
memory_info = memory_optimizer.get_memory_usage()
if 'gpu' in memory_info:
    gpu_mem = memory_info['gpu']
    print(f"GPU Memory: {gpu_mem['allocated'] / 1024**3:.2f} GB allocated")

# Optimizar uso de memoria del modelo
memory_optimizer.optimize_memory_usage(model, input_tensor)
```

## Optimizaciones Avanzadas

### Mixed Precision Training

```python
# El sistema habilita automáticamente AMP cuando está disponible
config = OptimizationConfig(enable_mixed_precision=True)

# El sistema usa automáticamente:
# - autocast() para forward pass
# - GradScaler para backward pass
# - Optimización automática de precisión

# Ejemplo de uso automático
optimizer = PerformanceOptimizer(config)
metrics = optimizer.training_step(model, data, target, criterion, optimizer)

# El scaler se maneja automáticamente
print(f"Loss: {metrics['loss']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.2f}%")
```

### Gradient Accumulation

```python
# Configurar acumulación de gradientes
config = OptimizationConfig(
    enable_gradient_accumulation=True,
    accumulation_steps=8  # Acumular 8 pasos antes de optimizar
)

# El sistema maneja automáticamente:
# - Escalado del loss
# - Acumulación de gradientes
# - Optimización en pasos específicos

# Training loop automático
for batch_idx, (data, target) in enumerate(train_loader):
    metrics = optimizer.training_step(
        model, data, target, criterion, optimizer, batch_idx
    )
    
    # Los gradientes se acumulan automáticamente
    # La optimización ocurre cada 8 pasos
```

### Multi-GPU Training

```python
# DataParallel automático
config = OptimizationConfig(
    enable_data_parallel=True,  # Habilitar si hay múltiples GPUs
    enable_distributed=False     # Para single-node multi-GPU
)

# El sistema detecta automáticamente múltiples GPUs
if torch.cuda.device_count() > 1:
    model = optimizer.optimize_model(model)
    print(f"DataParallel habilitado en {torch.cuda.device_count()} GPUs")

# DistributedDataParallel para multi-node
config = OptimizationConfig(
    enable_distributed=True,     # Para multi-node training
    enable_data_parallel=False
)

# Inicializar distributed training primero
# dist.init_process_group(...)
model = optimizer.optimize_model(model)
```

### Torch Compile

```python
# Habilitar compilación JIT
config = OptimizationConfig(
    enable_torch_compile=True,
    compile_mode="reduce-overhead"  # "reduce-overhead" o "max-autotune"
)

# El sistema compila automáticamente el modelo
model = optimizer.optimize_model(model)

# Modos disponibles:
# - "reduce-overhead": Reducir overhead de Python
# - "max-autotune": Máxima optimización (más lento en primera ejecución)
```

## Monitoreo de Performance

### Métricas de Training

```python
# Obtener estadísticas de optimización
summary = optimizer.get_optimization_summary()

# Métricas de tiempo
step_time = summary['step_time']
print(f"Tiempo promedio por step: {step_time['mean']:.4f}s")
print(f"Tiempo total de training: {step_time['total']:.2f}s")
print(f"Tiempo mínimo: {step_time['min']:.4f}s")
print(f"Tiempo máximo: {step_time['max']:.4f}s")

# Métricas de loss
loss_stats = summary['loss']
print(f"Loss promedio: {loss_stats['mean']:.4f}")
print(f"Loss más reciente: {loss_stats['latest']:.4f}")

# Métricas de accuracy
acc_stats = summary['accuracy']
print(f"Accuracy promedio: {acc_stats['mean']:.2f}%")
print(f"Accuracy más reciente: {acc_stats['latest']:.2f}%")
```

### Información del Sistema

```python
# Obtener información del sistema
system_info = summary['system']

# Información de CUDA
if system_info['cuda_available']:
    print(f"GPU: {system_info['device_name']}")
    print(f"Dispositivos disponibles: {system_info['device_count']}")
    print(f"Dispositivo actual: {system_info['current_device']}")

# Información de memoria GPU
if 'memory_allocated' in system_info:
    print(f"Memoria GPU asignada: {system_info['memory_allocated'] / 1024**3:.2f} GB")
    print(f"Memoria GPU reservada: {system_info['memory_reserved'] / 1024**3:.2f} GB")

# Información de CPU y memoria
print(f"CPU: {system_info['cpu_percent']:.1f}%")
print(f"Memoria: {system_info['memory_percent']:.1f}%")
```

## Profiling Avanzado

### Profiling de Funciones

```python
# Profilar funciones específicas
with profiler.profile_function("data_loading"):
    data = load_large_dataset()

with profiler.profile_function("model_inference"):
    output = model(data)

# Obtener resultados de profiling
profile_summary = profiler.get_profile_summary()
for func_name, results in profile_summary.items():
    print(f"{func_name}:")
    print(f"  Total calls: {results['total_calls']}")
    print(f"  Total time: {results['total_time']:.4f}s")
    print(f"  Avg time per call: {results['avg_time_per_call']:.6f}s")
```

### PyTorch Profiler

```python
# Profilar operaciones PyTorch específicas
profiler.profile_torch_operations(model, input_tensor)

# El profiler genera:
# - trace.json para Chrome DevTools
# - Métricas de operaciones clave
# - Análisis de memoria
# - Tiempos CPU vs CUDA

# Analizar resultados en Chrome DevTools:
# 1. Abrir Chrome DevTools
# 2. Ir a Performance tab
# 3. Cargar trace.json
# 4. Analizar timeline de operaciones
```

## Optimización de Memoria

### Gestión de Cache

```python
# Limpiar caches automáticamente
memory_optimizer.clear_gpu_cache()    # Limpiar GPU
memory_optimizer.clear_cpu_cache()    # Limpiar CPU
memory_optimizer.clear_all_caches()   # Limpiar ambos

# Obtener uso actual de memoria
memory_info = memory_optimizer.get_memory_usage()

# Memoria GPU
if 'gpu' in memory_info:
    gpu_mem = memory_info['gpu']
    print(f"GPU Memory Allocated: {gpu_mem['allocated'] / 1024**3:.2f} GB")
    print(f"GPU Memory Reserved: {gpu_mem['reserved'] / 1024**3:.2f} GB")
    print(f"GPU Memory Cached: {gpu_mem['cached'] / 1024**3:.2f} GB")
    print(f"GPU Memory Peak: {gpu_mem['max_allocated'] / 1024**3:.2f} GB")

# Memoria CPU
if 'cpu' in memory_info:
    cpu_mem = memory_info['cpu']
    print(f"CPU Memory Total: {cpu_mem['total'] / 1024**3:.2f} GB")
    print(f"CPU Memory Available: {cpu_mem['available'] / 1024**3:.2f} GB")
    print(f"CPU Memory Used: {cpu_mem['used'] / 1024**3:.2f} GB")
    print(f"CPU Memory Percent: {cpu_mem['percent']:.1f}%")
```

### Optimizaciones Automáticas

```python
# Aplicar optimizaciones de memoria automáticamente
memory_optimizer.optimize_memory_usage(model, input_tensor)

# El sistema habilita automáticamente:
# - Gradient checkpointing
# - Memory efficient attention (Flash Attention 2)
# - Optimizaciones específicas del modelo
```

## Configuraciones Avanzadas

### Configuración Personalizada

```python
# Configuración completa personalizada
config = OptimizationConfig(
    # Mixed precision
    enable_mixed_precision=True,
    
    # Gradient accumulation
    enable_gradient_accumulation=True,
    accumulation_steps=8,
    
    # Multi-GPU
    enable_data_parallel=True,
    enable_distributed=False,
    
    # Memory optimization
    enable_gradient_checkpointing=True,
    
    # Code optimization
    enable_torch_compile=True,
    compile_mode="max-autotune",
    
    # Training stability
    max_grad_norm=1.0,
    
    # Memory efficiency
    memory_efficient_attention=True
)
```

### Configuración por Escenario

```python
# Configuración para training rápido
fast_config = OptimizationConfig(
    enable_mixed_precision=True,
    enable_gradient_accumulation=False,
    enable_torch_compile=True,
    compile_mode="reduce-overhead"
)

# Configuración para ahorrar memoria
memory_config = OptimizationConfig(
    enable_mixed_precision=True,
    enable_gradient_checkpointing=True,
    enable_gradient_accumulation=True,
    accumulation_steps=16,
    memory_efficient_attention=True
)

# Configuración para máxima precisión
precision_config = OptimizationConfig(
    enable_mixed_precision=False,
    enable_gradient_accumulation=True,
    accumulation_steps=4,
    max_grad_norm=0.5
)
```

## Ejemplos de Uso Completo

### Training Loop Optimizado

```python
from performance_optimization_system import *

# Configuración completa
config = OptimizationConfig(
    enable_mixed_precision=True,
    enable_gradient_accumulation=True,
    enable_data_parallel=True,
    enable_gradient_checkpointing=True,
    enable_torch_compile=True,
    accumulation_steps=4,
    max_grad_norm=1.0
)

# Inicializar optimizador
training_optimizer = TrainingOptimizer(config)

# Modelo y datos
model = YourModel()
train_loader = your_train_loader
val_loader = your_val_loader
criterion = your_criterion
optimizer = your_optimizer

# Training optimizado completo
results = training_optimizer.optimize_training_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=100,
    device="cuda"
)

# Análisis de resultados
training_stats = results['training_stats']
optimization_summary = results['optimization_summary']
profile_summary = results['profile_summary']
memory_summary = results['memory_summary']

# Métricas finales
final_train_loss = training_stats['train_losses'][-1]
final_train_acc = training_stats['train_accuracies'][-1]
final_val_loss = training_stats['val_losses'][-1]
final_val_acc = training_stats['val_accuracies'][-1]

# Performance metrics
avg_step_time = optimization_summary['step_time']['mean']
total_training_time = optimization_summary['step_time']['total']

# Memory usage
gpu_memory_used = memory_summary['gpu']['allocated'] / 1024**3

print(f"Training completado:")
print(f"  Final Train Loss: {final_train_loss:.4f}")
print(f"  Final Train Accuracy: {final_train_acc:.2f}%")
print(f"  Final Val Loss: {final_val_loss:.4f}")
print(f"  Final Val Accuracy: {final_val_acc:.2f}%")
print(f"  Average Step Time: {avg_step_time:.4f}s")
print(f"  Total Training Time: {total_training_time:.2f}s")
print(f"  GPU Memory Used: {gpu_memory_used:.2f} GB")
```

### Optimización Incremental

```python
# Aplicar optimizaciones paso a paso
optimizer = PerformanceOptimizer(config)

# 1. Optimizar modelo
model = optimizer.optimize_model(model, device="cuda")

# 2. Optimizar optimizador
optimizer_opt = optimizer.optimize_optimizer(optimizer)

# 3. Training steps optimizados
for batch_idx, (data, target) in enumerate(train_loader):
    metrics = optimizer.training_step(
        model, data, target, criterion, optimizer_opt, batch_idx
    )
    
    # Log progress
    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx}: Loss={metrics['loss']:.4f}, "
              f"Acc={metrics['accuracy']:.2f}%, "
              f"Time={metrics['step_time']:.4f}s")

# 4. Obtener resumen
summary = optimizer.get_optimization_summary()
print(f"Optimization Summary: {summary}")
```

## Estructura del Proyecto

```
performance_optimization_system.py     # Sistema principal
requirements_performance_optimization.txt  # Dependencias
README_PERFORMANCE_OPTIMIZATION.md     # Documentación
```

## Dependencias Principales

- **torch**: Framework de deep learning
- **psutil**: Monitoreo del sistema
- **numpy**: Computación numérica

## Requisitos del Sistema

- **Python**: 3.8+
- **PyTorch**: 2.0.0+
- **CUDA**: Recomendado para mixed precision y multi-GPU

## Solución de Problemas

### Error de Mixed Precision

```python
# Verificar disponibilidad de CUDA
if not torch.cuda.is_available():
    print("CUDA no disponible, mixed precision deshabilitado")
    config.enable_mixed_precision = False

# Verificar versión de PyTorch
if torch.__version__ < "1.6.0":
    print("PyTorch < 1.6.0, mixed precision no disponible")
    config.enable_mixed_precision = False
```

### Error de Multi-GPU

```python
# Verificar número de GPUs
gpu_count = torch.cuda.device_count()
if gpu_count < 2:
    print(f"Solo {gpu_count} GPU disponible, DataParallel deshabilitado")
    config.enable_data_parallel = False

# Verificar inicialización distributed
if config.enable_distributed and not dist.is_initialized():
    print("Distributed training no inicializado")
    config.enable_distributed = False
```

### Error de Torch Compile

```python
# Verificar versión de PyTorch
if not hasattr(torch, 'compile'):
    print("PyTorch < 2.0.0, torch.compile no disponible")
    config.enable_torch_compile = False

# Verificar disponibilidad de compilador
try:
    test_model = nn.Linear(10, 10)
    compiled_model = torch.compile(test_model)
    print("Torch compile disponible")
except Exception as e:
    print(f"Torch compile no disponible: {e}")
    config.enable_torch_compile = False
```

## Contribución

Para contribuir al sistema:

1. Fork del repositorio
2. Crear rama feature
3. Implementar cambios
4. Ejecutar tests
5. Crear Pull Request

## Licencia

Este proyecto está bajo la licencia MIT.

## Contacto

Para soporte técnico o preguntas, abrir un issue en el repositorio.



# Optimizaciones de Rendimiento

## 🚀 Mejoras de Velocidad Implementadas

### 1. Compilación de Modelos

#### torch.compile (PyTorch 2.0+)
```python
from core.routing_optimization import compile_model

# Compilar modelo
compiled_model = compile_model(model, method="torch_compile")
# Speedup: 2-5x
```

#### TorchScript JIT
```python
compiled_model = compile_model(model, method="torchscript")
# Speedup: 1.5-3x
```

**Características:**
- ✅ Compilación JIT para inferencia rápida
- ✅ Optimización automática de kernels
- ✅ Reducción de overhead de Python
- ✅ Soporte para GPU y CPU

### 2. Cuantización

#### Cuantización Dinámica (Post-training)
```python
from core.routing_optimization import quantize_model

# Cuantizar modelo
quantized = quantize_model(model, method="dynamic")
# Speedup: 2-4x (CPU), Reducción tamaño: ~75%
```

#### Cuantización Estática
```python
quantized = quantize_model(model, method="static", calibration_data=data)
# Speedup: 3-5x (CPU), Mejor precisión
```

**Ventajas:**
- ✅ Reducción de tamaño de modelo ~75%
- ✅ Inferencia más rápida en CPU
- ✅ Menor uso de memoria
- ✅ Compatible con móviles/edge devices

### 3. Optimización de Data Loading

#### FastDataLoader
```python
from core.routing_optimization import FastDataLoader

loader = FastDataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)
# Speedup: 2-4x en carga de datos
```

#### CachedDataset
```python
from core.routing_optimization import CachedDataset

cached_dataset = CachedDataset(dataset, cache_size=1000)
# Acceso instantáneo a datos frecuentes
```

#### PrefetchDataLoader
```python
from core.routing_optimization import PrefetchDataLoader

loader = PrefetchDataLoader(
    dataset,
    batch_size=32,
    prefetch_batches=3
)
# Prefetching asíncrono
```

**Características:**
- ✅ Múltiples workers para paralelización
- ✅ Pin memory para transferencia rápida GPU
- ✅ Prefetching para ocultar latencia
- ✅ Cache LRU para acceso rápido

### 4. Optimización de Inferencia

#### InferenceOptimizer
```python
from core.routing_optimization import InferenceOptimizer

optimizer = InferenceOptimizer(model)
optimized_model = optimizer.optimize(
    use_jit=True,
    use_torch_compile=True,
    use_fusion=True,
    use_cudnn=True
)
```

**Optimizaciones aplicadas:**
- ✅ cuDNN benchmark mode
- ✅ TF32 para kernel fusion
- ✅ JIT compilation
- ✅ Warmup para estabilizar

### 5. Optimización de Memoria

#### MemoryOptimizer
```python
from core.routing_optimization import MemoryOptimizer

# Habilitar flash attention
MemoryOptimizer.enable_memory_efficient_attention()

# Limpiar cache
MemoryOptimizer.clear_cache()

# Monitorear memoria
usage = MemoryOptimizer.get_memory_usage()
```

#### Gradient Checkpointing
```python
from core.routing_optimization import GradientCheckpointing

GradientCheckpointing.enable_checkpointing(model)
# Reduce memoria en entrenamiento ~50%
```

### 6. Benchmarking

```python
from core.routing_optimization import benchmark_model

metrics = benchmark_model(
    model,
    input_shape=(32, 20),
    device="cuda",
    num_runs=100
)

print(f"Tiempo promedio: {metrics['avg_inference_time_ms']:.2f} ms")
print(f"Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")
```

## 📊 Mejoras de Rendimiento Esperadas

| Optimización | Speedup | Reducción Memoria | Caso de Uso |
|-------------|---------|-------------------|-------------|
| torch.compile | 2-5x | - | Inferencia GPU |
| TorchScript | 1.5-3x | - | Inferencia CPU/GPU |
| Cuantización Dinámica | 2-4x | ~75% | CPU, Edge |
| Cuantización Estática | 3-5x | ~75% | CPU, Edge |
| FastDataLoader | 2-4x | - | Entrenamiento |
| CachedDataset | 10-100x* | Cache | Datos repetidos |
| Gradient Checkpointing | - | ~50% | Entrenamiento |

*Para datos en cache

## 🎯 Guía de Uso Rápido

### Para Inferencia Rápida:
```python
# 1. Compilar modelo
model = compile_model(model, method="torch_compile")

# 2. Optimizar para inferencia
model = optimize_for_inference(model, input_shape=(32, 20))

# 3. Cuantizar (si CPU)
quantized = quantize_model(model, method="dynamic")
```

### Para Entrenamiento Rápido:
```python
# 1. FastDataLoader
loader = FastDataLoader(dataset, num_workers=4, pin_memory=True)

# 2. Gradient checkpointing (si memoria limitada)
GradientCheckpointing.enable_checkpointing(model)

# 3. Mixed precision (ya implementado en trainer)
```

### Para Producción:
```python
# Pipeline completo de optimización
from core.routing_optimization import OptimizedDataPipeline

# 1. Compilar modelo
model = compile_model(model, method="both")

# 2. Optimizar data loading
loader = OptimizedDataPipeline.create_fast_loader(
    dataset,
    use_cache=True,
    use_prefetch=True,
    num_workers=4
)

# 3. Benchmark
metrics = benchmark_model(model)
```

## 🔧 Configuración Recomendada

### GPU (NVIDIA):
- `torch.compile` con mode="reduce-overhead"
- `FastDataLoader` con `pin_memory=True`
- `num_workers=4-8`
- TF32 habilitado

### CPU:
- Cuantización dinámica o estática
- TorchScript JIT
- `num_workers=2-4`
- Cache de datos

### Edge/Móvil:
- Cuantización INT8
- Modelo compilado
- Batch size pequeño
- Sin workers (single-threaded)

## 📈 Benchmarks

Ejecutar benchmarks:
```bash
python examples/performance_optimization_example.py
```

## ⚡ Tips de Rendimiento

1. **Warmup**: Siempre hacer warmup antes de medir
2. **Batch Size**: Usar batch size óptimo (32-128)
3. **Workers**: 4-8 workers para data loading
4. **Pin Memory**: Habilitar si hay GPU
5. **Compilación**: Compilar una vez, usar muchas veces
6. **Cuantización**: Solo si precisión aceptable

## 🚨 Advertencias

- Cuantización puede reducir precisión ligeramente
- Compilación tiene overhead inicial
- Cache consume memoria adicional
- Algunas optimizaciones requieren PyTorch 2.0+


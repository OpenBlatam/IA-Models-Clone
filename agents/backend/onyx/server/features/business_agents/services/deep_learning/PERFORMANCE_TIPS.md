# Performance Tips & Best Practices

## 🚀 Optimizaciones de Rendimiento

### 1. Model Compilation

**Siempre compila modelos para producción:**

```python
from deep_learning.utils.optimization import ModelOptimizer

model = ModelOptimizer.compile_model(model, mode="reduce-overhead")
```

**Beneficios:**
- 1.5x - 3x más rápido en inferencia
- 1.2x - 2x más rápido en entrenamiento
- Mejor utilización de GPU

### 2. Mixed Precision Training

**Siempre usa AMP en GPU:**

```python
# Ya está habilitado por defecto en TrainingManager
# Solo asegúrate de que use_mixed_precision=True en config
```

**Beneficios:**
- 1.5x - 2x más rápido
- 50% menos memoria
- Misma precisión

### 3. DataLoader Optimization

**Usa DataLoaders optimizados:**

```python
from deep_learning.data import create_optimized_dataloader

loader = create_optimized_dataloader(
    dataset,
    batch_size=32,
    num_workers=4,  # Auto-detectado
    prefetch_factor=4,
    persistent_workers=True
)
```

**Configuración óptima:**
- `num_workers`: 2-4 por GPU
- `prefetch_factor`: 2-4
- `persistent_workers`: True
- `pin_memory`: True (GPU)

### 4. TF32 para GPUs Ampere+

**Habilitado automáticamente, pero verifica:**

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Beneficios:**
- 1.3x más rápido en GPUs Ampere+
- Sin pérdida de precisión

### 5. Flash Attention

**Habilitado automáticamente si está disponible:**

```python
torch.backends.cuda.enable_flash_sdp(True)
```

**Beneficios:**
- Más rápido para secuencias largas
- Menos memoria

### 6. Inference Optimization

**Usa OptimizedInference para producción:**

```python
from deep_learning.utils.batch_optimization import OptimizedInference

inference = OptimizedInference(
    model=model,
    device=device,
    batch_size=64,
    use_cache=True
)

results = inference(inputs)
```

**Características:**
- Batching automático
- Caching para inputs repetidos
- Mixed precision automático

### 7. Memory Management

**Limpia memoria periódicamente:**

```python
from deep_learning.utils.optimization import MemoryOptimizer

# Durante entrenamiento
if batch_idx % 100 == 0:
    MemoryOptimizer.clear_cache()

# Ver uso de memoria
usage = MemoryOptimizer.get_memory_usage()
```

### 8. Gradient Accumulation

**Para batch sizes grandes:**

```python
# En TrainingConfig
gradient_accumulation_steps = 4  # Simula batch_size * 4
```

**Beneficios:**
- Entrenar con batch sizes efectivos más grandes
- Sin necesidad de más memoria

### 9. Model Optimization Pipeline

**Pipeline completo de optimización:**

```python
from deep_learning.utils.optimization import optimize_model_for_production

model = optimize_model_for_production(
    model,
    device=device,
    compile_model=True,
    enable_tf32=True,
    enable_flash_attention=True
)
```

### 10. Profiling

**Profile antes de optimizar:**

```python
from deep_learning.utils.profiling import profile_inference

results = profile_inference(model, input_data, device)
print(f"Avg time: {results['avg_inference_time']*1000:.2f}ms")
```

## 📊 Mejoras Esperadas

| Optimización | Mejora de Velocidad | Reducción de Memoria |
|-------------|-------------------|---------------------|
| Model Compilation | 1.5x - 3x | - |
| Mixed Precision | 1.5x - 2x | 50% |
| TF32 | 1.3x | - |
| Optimized DataLoader | 1.5x - 2x | - |
| Flash Attention | 1.2x - 1.5x | 30% |
| **Combinado** | **3x - 5x** | **50-70%** |

## ⚠️ Consideraciones

1. **Compilación**: Primera ejecución es más lenta (warmup)
2. **Memoria**: Algunas optimizaciones usan más memoria temporal
3. **Compatibilidad**: Algunas optimizaciones requieren hardware específico
4. **Debugging**: Desactiva optimizaciones para debugging

## 🎯 Checklist de Optimización

- [ ] Model compilation habilitado
- [ ] Mixed precision habilitado
- [ ] DataLoaders optimizados
- [ ] TF32 habilitado (GPUs Ampere+)
- [ ] Flash attention habilitado
- [ ] Memory management configurado
- [ ] Gradient accumulation si es necesario
- [ ] Inference caching para producción
- [ ] Profiling realizado
- [ ] Optimizaciones validadas




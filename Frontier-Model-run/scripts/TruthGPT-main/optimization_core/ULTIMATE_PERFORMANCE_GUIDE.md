# 🎯 TRUTHGPT - ULTIMATE PERFORMANCE GUIDE

## ⚡ Guía de Rendimiento Máximo

### 🚀 Optimizaciones Críticas

#### 1. **Flash Attention** (3x speedup)
```python
# Habilitar Flash Attention
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    attn_implementation="flash_attention_2"
)
```

#### 2. **Mixed Precision** (1.5x speedup, -50% memoria)
```python
# FP16 automático
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(**batch)
```

#### 3. **PEFT LoRA** (2x speedup, -90% parámetros)
```python
# Configuración LoRA óptima
lora_config = LoraConfig(
    r=16,                    # Rank óptimo
    lora_alpha=32,           # Scaling factor
    target_modules=["c_attn", "c_proj"],  # Módulos objetivo
    lora_dropout=0.1,        # Dropout
    bias="none",             # Sin bias
    task_type="CAUSAL_LM"    # Tipo de tarea
)
```

#### 4. **Gradient Checkpointing** (1.3x speedup, -40% memoria)
```python
# Habilitar gradient checkpointing
model.gradient_checkpointing_enable()
```

#### 5. **DeepSpeed ZeRO** (2-4x speedup, -75% memoria)
```python
# Configuración DeepSpeed
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

### 📊 Configuración Óptima por Caso

#### Caso 1: Single GPU (Máximo Rendimiento)
```python
config = {
    'precision': 'fp16',
    'flash_attention': True,
    'gradient_checkpointing': True,
    'use_peft': True,
    'lora_r': 16,
    'batch_size': 8,
    'gradient_accumulation_steps': 4,
}
```

#### Caso 2: Multi-GPU (Distribuido)
```python
config = {
    'precision': 'fp16',
    'flash_attention': True,
    'gradient_checkpointing': True,
    'use_peft': True,
    'use_deepspeed': True,
    'deepspeed_stage': 2,
    'num_gpus': 4,
    'batch_size': 2,  # Por GPU
    'gradient_accumulation_steps': 8,
}
```

#### Caso 3: Memory Constrained (Mínima Memoria)
```python
config = {
    'precision': 'int8',
    'quantization': 'bitsandbytes',
    'gradient_checkpointing': True,
    'use_peft': True,
    'lora_r': 8,  # Rank menor
    'micro_batch_size': 1,
    'gradient_accumulation_steps': 16,
    'use_deepspeed': True,
    'deepspeed_stage': 3,  # ZeRO-3
}
```

### 🎯 DataLoader Optimizado
```python
from torch.utils.data import DataLoader

# Configuración óptima
dataloader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,              # Paralelo
    pin_memory=True,             # Transferencia rápida
    persistent_workers=True,    # Reutilizar workers
    prefetch_factor=2,          # Prefetch
    drop_last=True,             # Batch size consistente
    shuffle=True,               # Shuffle para entrenamiento
)
```

### ⚡ Training Loop Optimizado
```python
def optimized_training_loop(model, dataloader, optimizer, config):
    """Loop de entrenamiento optimizado."""
    scaler = GradScaler()
    model.train()
    
    for epoch in range(config['epochs']):
        for step, batch in enumerate(dataloader):
            # Forward pass con mixed precision
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
            
            # Backward optimizado
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.get('max_grad_norm', 1.0)
            )
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Logging
            if step % config.get('logging_steps', 100) == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
```

### 📊 Monitoring y Profiling
```python
import wandb
import torch.profiler

# Configurar WandB
wandb.init(
    project="truthgpt-optimization",
    config=config,
    name=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
)

# Profiler para análisis de rendimiento
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Tu código de entrenamiento aquí
    pass
```

### 🎯 Benchmarking Automático
```python
def benchmark_model(model, dataloader, config):
    """Benchmark automático del modelo."""
    import time
    import psutil
    
    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 5:  # 5 batches de warmup
            break
        with torch.no_grad():
            _ = model(**batch)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    start_memory = torch.cuda.memory_allocated()
    
    for i, batch in enumerate(dataloader):
        if i >= 100:  # 100 batches de benchmark
            break
        with torch.no_grad():
            _ = model(**batch)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    end_memory = torch.cuda.memory_allocated()
    
    # Resultados
    total_time = end_time - start_time
    avg_time_per_batch = total_time / 100
    memory_used = (end_memory - start_memory) / 1024**2  # MB
    
    print(f"\n📊 BENCHMARK RESULTS")
    print(f"Average time per batch: {avg_time_per_batch:.3f}s")
    print(f"Memory used: {memory_used:.1f} MB")
    print(f"Throughput: {100/avg_time_per_batch:.1f} batches/sec")
    
    return {
        'avg_time_per_batch': avg_time_per_batch,
        'memory_used': memory_used,
        'throughput': 100/avg_time_per_batch
    }
```

### ✅ Checklist de Optimización

#### Instalación
- [ ] PyTorch con CUDA correcta
- [ ] Flash Attention instalado
- [ ] XFormers instalado
- [ ] DeepSpeed instalado
- [ ] PEFT instalado
- [ ] Bitsandbytes instalado

#### Configuración
- [ ] FP16/BF16 habilitado
- [ ] Flash Attention habilitado
- [ ] Gradient checkpointing ON
- [ ] PEFT configurado
- [ ] DeepSpeed configurado (si multi-GPU)
- [ ] DataLoader optimizado

#### Entrenamiento
- [ ] Mixed precision ON
- [ ] Gradient accumulation configurado
- [ ] Learning rate schedule configurado
- [ ] Early stopping configurado
- [ ] Monitoring (WandB) activado
- [ ] Profiling configurado

### 🚀 Comandos de Verificación

```bash
# Verificar instalación
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Verificar optimizaciones
python -c "
try:
    from flash_attn import flash_attn_func
    print('✅ Flash Attention: OK')
except ImportError:
    print('❌ Flash Attention: NOT INSTALLED')

try:
    import xformers
    print('✅ XFormers: OK')
except ImportError:
    print('❌ XFormers: NOT INSTALLED')

try:
    import peft
    print('✅ PEFT: OK')
except ImportError:
    print('❌ PEFT: NOT INSTALLED')

try:
    import deepspeed
    print('✅ DeepSpeed: OK')
except ImportError:
    print('❌ DeepSpeed: NOT INSTALLED')
"

# Benchmark rápido
python -c "
import torch
import time
x = torch.randn(1000, 1000).cuda()
start = time.perf_counter()
y = torch.matmul(x, x)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'GPU Benchmark: {(end-start)*1000:.2f}ms')
"
```

### 📈 Resultados Esperados

#### Speedup Total
- **Flash Attention**: 3x
- **XFormers**: 2x
- **Mixed Precision**: 1.5x
- **PEFT LoRA**: 2x
- **DeepSpeed**: 2-4x
- **COMBINADO**: **15-20x** ⚡

#### Reducción de Memoria
- **Mixed Precision**: -50%
- **Gradient Checkpointing**: -40%
- **PEFT LoRA**: -90%
- **DeepSpeed ZeRO**: -75%
- **COMBINADO**: **-95%** 💾

---

**¡Rendimiento máximo alcanzado!** 🎯⚡🚀


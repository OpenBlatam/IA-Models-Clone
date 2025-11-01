# ⚡ Guía de Profiling - Blatam Academy Features

## 🎯 Introducción al Profiling

### Tipos de Profiling

1. **CPU Profiling**: Identificar funciones que consumen más CPU
2. **Memory Profiling**: Identificar fugas de memoria y uso excesivo
3. **I/O Profiling**: Identificar operaciones de I/O lentas
4. **GPU Profiling**: Analizar uso de GPU y CUDA

## 🔧 Herramientas de Profiling

### py-spy (Sampling Profiler)

```bash
# Instalar
pip install py-spy

# Profiling de proceso en ejecución
py-spy top --pid <PID>

# Generar flame graph
py-spy record -o profile.svg --pid <PID>

# Profiling de script
py-spy record -o profile.svg -- python script.py
```

### cProfile (Built-in)

```python
import cProfile
import pstats

# Profiling básico
profiler = cProfile.Profile()
profiler.enable()

# Tu código aquí
result = process_data()

profiler.disable()

# Análisis
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)

# Guardar para análisis posterior
stats.dump_stats('profile.prof')
```

### Visualización con snakeviz

```bash
# Instalar
pip install snakeviz

# Visualizar profile
snakeviz profile.prof
# Abre en navegador: http://127.0.0.1:8080
```

### Line Profiler

```python
# Instalar: pip install line_profiler

# Decorar función
@profile
def slow_function():
    result = []
    for i in range(1000000):
        result.append(i * 2)
    return result

# Ejecutar: kernprof -l -v script.py
```

## 💾 Memory Profiling

### memory_profiler

```python
from memory_profiler import profile
import tracemalloc

@profile
def memory_intensive():
    data = [i for i in range(1000000)]
    return sum(data)

# Ejecutar: python -m memory_profiler script.py
```

### Tracemalloc (Built-in)

```python
import tracemalloc

# Iniciar tracking
tracemalloc.start()

# Tu código
result = process_data()

# Tomar snapshot
snapshot = tracemalloc.take_snapshot()

# Top 10 allocations
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)

# Comparar snapshots
snapshot1 = tracemalloc.take_snapshot()
# ... código ...
snapshot2 = tracemalloc.take_snapshot()

top_stats = snapshot2.compare_to(snapshot1, 'lineno')
for stat in top_stats[:10]:
    print(stat)
```

### Memory Profiling Avanzado

```python
import tracemalloc
import gc

class MemoryProfiler:
    """Profiler de memoria avanzado."""
    
    def __init__(self):
        self.snapshots = []
    
    def start(self):
        """Iniciar profiling."""
        tracemalloc.start()
        gc.collect()  # Baseline
    
    def snapshot(self, label):
        """Tomar snapshot."""
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((label, snapshot))
    
    def compare(self, label1, label2):
        """Comparar dos snapshots."""
        snap1 = next(s for l, s in self.snapshots if l == label1)[1]
        snap2 = next(s for l, s in self.snapshots if l == label2)[1]
        
        stats = snap2.compare_to(snap1, 'lineno')
        print(f"Memory difference between {label1} and {label2}:")
        for stat in stats[:10]:
            print(stat)
    
    def stop(self):
        """Detener profiling."""
        tracemalloc.stop()

# Uso
profiler = MemoryProfiler()
profiler.start()
profiler.snapshot('start')

# Operación
result = process_data()

profiler.snapshot('end')
profiler.compare('start', 'end')
profiler.stop()
```

## 🚀 GPU Profiling

### CUDA Profiling

```python
import torch

# Habilitar profiling CUDA
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Tu código
    result = model_forward(inputs)

# Ver resultados
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))

# Exportar a Chrome trace
prof.export_chrome_trace("trace.json")
# Abrir en chrome://tracing
```

### GPU Memory Tracking

```python
import torch

def print_gpu_memory(device=0):
    """Imprimir estadísticas de memoria GPU."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        
        print(f"GPU {device} Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Max Allocated: {max_allocated:.2f} GB")

# Resetear contador
torch.cuda.reset_peak_memory_stats()
```

## 📊 Profiling de KV Cache

### Profiling Completo del Cache

```python
from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine, KVCacheConfig
import cProfile
import pstats

# Configurar con profiling habilitado
config = KVCacheConfig(
    max_tokens=8192,
    enable_profiling=True  # Habilitar profiling interno
)

engine = UltraAdaptiveKVCacheEngine(config)

# Profiling externo
profiler = cProfile.Profile()
profiler.enable()

# Operaciones de cache
for i in range(1000):
    result = await engine.process_request(request)

profiler.disable()

# Análisis
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(30)

# Estadísticas del cache
cache_stats = engine.get_statistics()
print(f"Hit Rate:", cache_stats['hit_rate'])
print(f"Latency P95:", cache_stats['latency_p95'])
```

### Profiling de Batch Operations

```python
import time
from contextlib import contextmanager

@contextmanager
def profile_batch(engine, batch_size):
    """Profile batch processing."""
    start_time = time.perf_counter()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    yield
    
    duration = time.perf_counter() - start_time
    end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    memory_delta = (end_memory - start_memory) / 1024**3
    
    print(f"Batch Size: {batch_size}")
    print(f"Duration: {duration:.2f}s")
    print(f"Throughput: {batch_size / duration:.2f} req/s")
    print(f"Memory Delta: {memory_delta:.2f} GB")

# Uso
with profile_batch(engine, batch_size=100):
    results = await engine.process_batch_optimized(requests, batch_size=100)
```

## 📈 Análisis de Resultados

### Flame Graph

```bash
# Generar flame graph con py-spy
py-spy record -o flamegraph.svg --format speedscope -- python script.py

# O con cProfile + flameprof
pip install flameprof
python -m cProfile -o profile.prof script.py
flameprof profile.prof > flamegraph.svg
```

### Timeline Visualization

```python
# Generar timeline con Chrome trace
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
) as prof:
    # Operaciones
    pass

prof.export_chrome_trace("timeline.json")
# Abrir en chrome://tracing
```

## ✅ Checklist de Profiling

### Antes de Profiling
- [ ] Identificar objetivo (CPU, memoria, I/O, GPU)
- [ ] Seleccionar herramienta apropiada
- [ ] Reproducir escenario de carga
- [ ] Baseline establecido

### Durante Profiling
- [ ] Capturar datos suficientes
- [ ] Verificar que profiling no afecta resultados
- [ ] Documentar condiciones de prueba

### Después de Profiling
- [ ] Analizar resultados
- [ ] Identificar bottlenecks
- [ ] Implementar optimizaciones
- [ ] Re-profiling para validar mejoras

---

**Más información:**
- [Performance Tuning](PERFORMANCE_TUNING.md)
- [Advanced Debugging](ADVANCED_DEBUGGING.md)
- [Benchmarking Guide](BENCHMARKING_GUIDE.md)


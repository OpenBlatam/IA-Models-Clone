# 🐛 Debugging Avanzado - Blatam Academy Features

## 🔍 Estrategias de Debugging

### Debugging Interactivo con IPython

```python
import IPython

# Breakpoint con IPython
def debug_function():
    IPython.embed()  # Abre shell interactivo
    # Continúa debugging aquí
```

### Debugging Asíncrono

```python
import asyncio
import ipdb

async def debug_async_function():
    ipdb.set_trace()  # Breakpoint en código async
    result = await some_async_operation()
    return result

# Ejecutar
asyncio.run(debug_async_function())
```

### Debugging de Memory Leaks

```python
import tracemalloc
import gc

# Iniciar tracking de memoria
tracemalloc.start()

# Tu código aquí
result = process_large_dataset()

# Obtener estadísticas
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

# Top 10 memory allocations
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)

# Detener tracking
tracemalloc.stop()

# Forzar garbage collection
gc.collect()
```

## 📊 Profiling de Performance

### cProfile para Performance

```python
import cProfile
import pstats

# Profiling básico
profiler = cProfile.Profile()
profiler.enable()

# Tu código aquí
result = complex_operation()

profiler.disable()

# Generar estadísticas
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 funciones

# Guardar en archivo
stats.dump_stats('profile.prof')

# Visualizar con snakeviz
# snakeviz profile.prof
```

### Line Profiler

```python
# Instalar: pip install line_profiler

# Decorar función para profiling
@profile
def slow_function():
    result = []
    for i in range(1000000):
        result.append(i * 2)
    return result

# Ejecutar: kernprof -l -v script.py
```

### Memory Profiler

```python
# Instalar: pip install memory-profiler

from memory_profiler import profile

@profile
def memory_intensive_function():
    large_list = [i for i in range(1000000)]
    return sum(large_list)

# Ejecutar: python -m memory_profiler script.py
```

## 🔧 Debugging de KV Cache

### Debugging de Cache Misses

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Habilitar debug en KV Cache
config = KVCacheConfig(
    max_tokens=8192,
    enable_profiling=True,  # Habilitar profiling
    log_level='DEBUG'  # Logging detallado
)

engine = UltraAdaptiveKVCacheEngine(config)

# Interceptar cache misses
original_get = engine._get_from_cache

def debug_get(key):
    result = original_get(key)
    if result is None:
        logger.debug(f"Cache MISS for key: {key[:50]}...")
    else:
        logger.debug(f"Cache HIT for key: {key[:50]}...")
    return result

engine._get_from_cache = debug_get
```

### Debugging de GPU Memory

```python
import torch

def print_gpu_memory():
    """Imprimir uso de memoria GPU."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Uso
print_gpu_memory()
```

## 🎯 Debugging de Distributed Systems

### Tracing de Requests

```python
import uuid
import functools

# Context manager para tracing
class RequestTracer:
    def __init__(self, request_id=None):
        self.request_id = request_id or str(uuid.uuid4())
        self.trace = []
    
    def trace_call(self, func_name, **kwargs):
        self.trace.append({
            'timestamp': time.time(),
            'function': func_name,
            'kwargs': kwargs
        })
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        # Log trace completo
        logger.debug(f"Request trace {self.request_id}: {self.trace}")

# Uso
with RequestTracer() as tracer:
    tracer.trace_call('cache_get', key='user:123')
    result = cache.get('user:123')
    tracer.trace_call('cache_result', found=result is not None)
```

### Debugging de Network Issues

```python
import socket
import requests

def debug_connection(host, port):
    """Debug conexión de red."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✓ Connection to {host}:{port} successful")
        else:
            print(f"✗ Connection to {host}:{port} failed: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")

# Uso
debug_connection('localhost', 6379)  # Redis
debug_connection('localhost', 5432)  # PostgreSQL
```

## 🔍 Debugging Tools

### Debugger Visual (PDB++)

```python
# Instalar: pip install pdbpp

import pdb

# Breakpoint avanzado
def complex_function():
    pdb.set_trace()  # Breakpoint con PDB++
    # Usa 'h' para ayuda, 'n' para siguiente línea
```

### Debugging con Logpoints

```python
import logging

class LogpointHandler(logging.Handler):
    """Handler que actúa como breakpoint."""
    
    def emit(self, record):
        if 'DEBUG_BREAK' in record.getMessage():
            import pdb; pdb.set_trace()

logger.addHandler(LogpointHandler())

# Uso - logpoint que actúa como breakpoint
logger.info("DEBUG_BREAK: Checkpoint reached")
```

### Debugging de Excepciones

```python
import sys
import traceback

def exception_handler(exc_type, exc_value, exc_traceback):
    """Handler personalizado de excepciones."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    
    # Guardar traceback completo
    tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    with open('error.log', 'a') as f:
        f.write(tb_str)

sys.excepthook = exception_handler
```

## 📈 Debugging de Performance Issues

### Timing Decorator

```python
import time
import functools

def timing_decorator(func):
    """Decorator para medir tiempo de ejecución."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration = (end - start) * 1000  # ms
        logger.debug(f"{func.__name__} took {duration:.2f}ms")
        return result
    return wrapper

# Uso
@timing_decorator
def slow_operation():
    time.sleep(1)
    return "done"
```

### Debugging de Bottlenecks

```python
from contextlib import contextmanager

@contextmanager
def debug_timing(label):
    """Context manager para timing."""
    start = time.perf_counter()
    logger.debug(f"Starting {label}")
    yield
    duration = (time.perf_counter() - start) * 1000
    logger.debug(f"Completed {label} in {duration:.2f}ms")

# Uso
with debug_timing("cache_operation"):
    result = cache.get(key)
```

## ✅ Checklist de Debugging

### Antes de Debugging
- [ ] Reproducir el problema consistentemente
- [ ] Identificar condiciones que causan el problema
- [ ] Recolectar logs relevantes
- [ ] Verificar versión del código

### Durante Debugging
- [ ] Usar breakpoints estratégicos
- [ ] Revisar estado de variables
- [ ] Verificar llamadas a funciones
- [ ] Revisar stack traces completos

### Después de Debugging
- [ ] Documentar la causa raíz
- [ ] Crear test case para prevenir regresión
- [ ] Implementar fix
- [ ] Verificar que el fix funciona

---

**Más información:**
- [Troubleshooting Guide](TROUBLESHOOTING_BY_SYMPTOM.md)
- [Quick Diagnostics](QUICK_DIAGNOSTICS.md)
- [Performance Tuning](PERFORMANCE_TUNING.md)


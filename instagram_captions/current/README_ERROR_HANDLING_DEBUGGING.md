# Error Handling and Debugging System

Sistema completo de manejo de errores y debugging implementando try-except blocks, logging estructurado, herramientas de debugging de PyTorch y recuperación automática de errores.

## Características

- **Manejo Robusto de Errores**: Try-except blocks automáticos con decoradores
- **Logging Estructurado**: Sistema completo de logging con rotación y niveles
- **PyTorch Debugging**: Herramientas específicas para debugging de modelos
- **Recuperación Automática**: Estrategias automáticas de recuperación de errores
- **Monitoreo de Performance**: Seguimiento continuo de recursos del sistema
- **Detección de Anomalías**: `autograd.detect_anomaly()` automático
- **Gestión de Memoria**: Monitoreo y limpieza automática de memoria GPU
- **Decoradores Inteligentes**: Manejo automático de errores con reintentos

## Instalación

```bash
pip install -r requirements_error_handling_debugging.txt
```

## Uso Básico

### ErrorHandler - Manejo Automático de Errores

```python
from error_handling_debugging_system import ErrorHandler

# Inicializar manejador de errores
error_handler = ErrorHandler(enable_debug=True)

# Manejar error automáticamente
try:
    result = risky_function()
except Exception as e:
    success, recovery_result = error_handler.handle_error(
        e, context="data_processing", max_retries=3
    )
    
    if success:
        print(f"Error recuperado: {recovery_result}")
    else:
        print("Error no pudo ser recuperado")
```

### PyTorchDebugger - Herramientas de Debugging

```python
from error_handling_debugging_system import PyTorchDebugger

# Inicializar debugger con detección de anomalías
debugger = PyTorchDebugger(enable_anomaly_detection=True)

# Verificar parámetros del modelo
param_info = debugger.check_model_parameters(model)
print(f"Parámetros NaN: {param_info['nan_parameters']}")
print(f"Parámetros Inf: {param_info['inf_parameters']}")

# Verificar gradientes
grad_info = debugger.check_gradients(model)
print(f"Gradientes NaN: {grad_info['nan_gradients']}")
print(f"Gradientes Inf: {grad_info['inf_gradients']}")

# Perfilar uso de memoria
memory_info = debugger.profile_memory(model, input_tensor)
print(f"Memoria pico: {memory_info['peak_allocated']} bytes")
```

### DebugDecorator - Manejo Automático con Decoradores

```python
from error_handling_debugging_system import DebugDecorator, ErrorHandler

error_handler = ErrorHandler()

# Aplicar decorador a función
@DebugDecorator(error_handler, max_retries=3)
def training_step(model, data, target):
    # Lógica de entrenamiento
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    return loss.item()

# La función se ejecuta automáticamente con manejo de errores
loss = training_step(model, data, target)
```

## Componentes Principales

### ErrorHandler

```python
# Configuración avanzada
error_handler = ErrorHandler(
    log_file="custom_error_log.txt",
    enable_debug=True
)

# Registrar estrategia de recuperación personalizada
def custom_recovery_strategy(error, context, attempt):
    if "connection timeout" in str(error).lower():
        time.sleep(2 ** attempt)  # Backoff exponencial
        return "retry_connection"
    return False

error_handler.register_recovery_strategy(
    "connection timeout", 
    custom_recovery_strategy
)

# Manejar error con contexto específico
success, result = error_handler.handle_error(
    error=connection_error,
    context="API_call",
    max_retries=5
)
```

### PyTorchDebugger

```python
debugger = PyTorchDebugger(enable_anomaly_detection=True)

# Habilitar gradient checkpointing para ahorrar memoria
debugger.enable_gradient_checkpointing(model)

# Verificar estado completo del modelo
param_info = debugger.check_model_parameters(model)
grad_info = debugger.check_gradients(model)

# Perfilar memoria con tensor específico
input_tensor = torch.randn(1, 784)
memory_profile = debugger.profile_memory(model, input_tensor)

# Deshabilitar gradient checkpointing
debugger.disable_gradient_checkpointing(model)
```

### PerformanceMonitor

```python
from error_handling_debugging_system import PerformanceMonitor

# Monitorear sistema cada 30 segundos
monitor = PerformanceMonitor(log_interval=30)

# Obtener estado actual del sistema
status = monitor.log_system_status()
if status:
    print(f"CPU: {status['cpu_percent']}%")
    print(f"Memoria: {status['memory_percent']}%")
    print(f"GPU: {status['gpu_info']}")

# Obtener resumen de performance
summary = monitor.get_performance_summary()
if summary:
    print(f"CPU promedio: {summary['cpu_stats']['mean']:.2f}%")
    print(f"Memoria máxima: {summary['memory_stats']['max']:.2f}%")
```

### ErrorRecoveryManager

```python
from error_handling_debugging_system import ErrorRecoveryManager

recovery_manager = ErrorRecoveryManager(error_handler)

# Registrar fix automático personalizado
def custom_fix_function(error, context, attempt):
    if "learning rate too high" in str(error).lower():
        optimizer.param_groups[0]['lr'] *= 0.5
        return "reduced_learning_rate"
    return False

recovery_manager.register_automatic_fix(
    "learning rate too high", 
    custom_fix_function
)

# Aplicar fixes comunes automáticamente
fixes = recovery_manager.apply_common_fixes(model, optimizer)
print(f"Fixes aplicados: {fixes}")
```

## Estrategias de Recuperación Automática

### Errores de Memoria GPU

```python
# El sistema maneja automáticamente:
# - CUDA out of memory
# - Limpieza de cache GPU
# - Reducción de batch size
# - Garbage collection forzado

# Configurar estrategia personalizada
def memory_recovery_strategy(error, context, attempt):
    if "CUDA out of memory" in str(error):
        # Limpiar cache
        torch.cuda.empty_cache()
        
        # Reducir batch size
        global current_batch_size
        current_batch_size = max(1, current_batch_size // 2)
        
        # Forzar garbage collection
        gc.collect()
        
        return f"reduced_batch_size_to_{current_batch_size}"
    
    return False

error_handler.register_recovery_strategy(
    "CUDA out of memory", 
    memory_recovery_strategy
)
```

### Errores de Carga de Modelos

```python
def model_loading_recovery_strategy(error, context, attempt):
    if attempt == 0:
        # Intentar con float16
        return "try_float16"
    elif attempt == 1:
        # Intentar con CPU
        return "try_cpu"
    elif attempt == 2:
        # Intentar con modelo más pequeño
        return "try_smaller_model"
    
    return False

error_handler.register_recovery_strategy(
    "model loading", 
    model_loading_recovery_strategy
)
```

### Errores de Carga de Datos

```python
def data_loading_recovery_strategy(error, context, attempt):
    if attempt == 0:
        # Reducir número de workers
        return "reduce_workers"
    elif attempt == 1:
        # Deshabilitar memory pinning
        return "disable_pinning"
    elif attempt == 2:
        # Usar CPU para data loading
        return "use_cpu_loading"
    
    return False

error_handler.register_recovery_strategy(
    "data loading", 
    data_loading_recovery_strategy
)
```

## Debugging Avanzado

### Detección de Anomalías

```python
# Habilitar detección automática de anomalías
debugger = PyTorchDebugger(enable_anomaly_detection=True)

# Esto es equivalente a:
# torch.autograd.set_detect_anomaly(True)

# Ahora PyTorch detectará automáticamente:
# - Gradientes NaN/Inf
# - Operaciones problemáticas
# - Dependencias de gradientes incorrectas
```

### Verificación de Gradientes

```python
# Verificar gradientes después de backward()
loss.backward()

grad_info = debugger.check_gradients(model)

if grad_info["nan_gradients"]:
    print(f"⚠️ Gradientes NaN detectados: {grad_info['nan_gradients']}")
    
if grad_info["inf_gradients"]:
    print(f"⚠️ Gradientes Inf detectados: {grad_info['inf_gradients']}")
    
if grad_info["zero_gradients"]:
    print(f"⚠️ Gradientes cero detectados: {grad_info['zero_gradients']}")

# Verificar normas de gradientes
for name, norm in grad_info["gradient_norms"].items():
    if norm > 10.0:
        print(f"⚠️ Gradiente muy grande en {name}: {norm}")
```

### Perfilado de Memoria

```python
# Perfilar uso de memoria del modelo
input_tensor = torch.randn(1, 784).cuda()
memory_info = debugger.profile_memory(model, input_tensor)

print(f"Memoria inicial: {memory_info['initial_allocated']} bytes")
print(f"Memoria después de forward: {memory_info['forward_allocated']} bytes")
print(f"Memoria pico: {memory_info['peak_allocated']} bytes")
print(f"Memoria final: {memory_info['final_allocated']} bytes")

# Calcular overhead de memoria
memory_overhead = memory_info['forward_allocated'] - memory_info['initial_allocated']
print(f"Overhead de memoria: {memory_overhead} bytes")
```

## Monitoreo de Sistema

### Métricas del Sistema

```python
monitor = PerformanceMonitor(log_interval=10)  # Cada 10 segundos

# Obtener estado del sistema
status = monitor.log_system_status()

if status:
    print("=== Estado del Sistema ===")
    print(f"CPU: {status['cpu_percent']:.1f}%")
    print(f"Memoria: {status['memory_percent']:.1f}%")
    print(f"Disco: {status['disk_percent']:.1f}%")
    
    if status['gpu_info']:
        gpu = status['gpu_info']
        print(f"GPU: {gpu['device_name']}")
        print(f"Memoria GPU: {gpu['memory_allocated'] / 1024**3:.2f} GB")
        print(f"Memoria reservada: {gpu['memory_reserved'] / 1024**3:.2f} GB")
```

### Historial de Performance

```python
# Obtener resumen estadístico
summary = monitor.get_performance_summary()

if summary:
    print("=== Resumen de Performance ===")
    print(f"Total de entradas: {summary['total_entries']}")
    print(f"Período: {summary['time_span']['start']} - {summary['time_span']['end']}")
    
    cpu_stats = summary['cpu_stats']
    print(f"CPU - Promedio: {cpu_stats['mean']:.1f}%, "
          f"Máximo: {cpu_stats['max']:.1f}%, "
          f"Mínimo: {cpu_stats['min']:.1f}%")
    
    mem_stats = summary['memory_stats']
    print(f"Memoria - Promedio: {mem_stats['mean']:.1f}%, "
          f"Máximo: {mem_stats['max']:.1f}%, "
          f"Mínimo: {mem_stats['min']:.1f}%")
```

## Fixes Automáticos

### Fix de Parámetros NaN

```python
# El sistema detecta y repara automáticamente parámetros NaN
fixes = recovery_manager.apply_common_fixes(model, optimizer)

if "nan_parameter_fix" in fixes:
    print("✅ Parámetros NaN reparados automáticamente")
    
    # Verificar que se repararon
    param_info = debugger.check_model_parameters(model)
    if not param_info["nan_parameters"]:
        print("✅ Confirmado: No hay parámetros NaN")
```

### Fix de Gradientes Explosivos

```python
# El sistema aplica automáticamente:
# - Gradient clipping
# - Reducción de learning rate

if "exploding_gradient_fix" in fixes:
    print("✅ Gradientes explosivos controlados")
    
    # Verificar que se controlaron
    grad_info = debugger.check_gradients(model)
    if not grad_info["inf_gradients"]:
        print("✅ Confirmado: No hay gradientes Inf")
```

### Fix de Gradientes Cero

```python
# El sistema ajusta automáticamente:
# - Learning rate
# - Inicialización de parámetros

if "zero_gradient_fix" in fixes:
    print("✅ Gradientes cero corregidos")
    
    # Verificar que se corrigieron
    grad_info = debugger.check_gradients(model)
    if not grad_info["zero_gradients"]:
        print("✅ Confirmado: No hay gradientes cero")
```

## Configuraciones Avanzadas

### Logging Personalizado

```python
import logging

# Configurar logging personalizado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('custom_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# El sistema usa el logger configurado
error_handler = ErrorHandler(enable_debug=True)
```

### Estrategias de Recuperación Personalizadas

```python
# Estrategia compleja de recuperación
def advanced_recovery_strategy(error, context, attempt):
    error_msg = str(error).lower()
    
    if "cuda out of memory" in error_msg:
        if attempt == 0:
            # Limpiar cache
            torch.cuda.empty_cache()
            return "cleared_cache"
        elif attempt == 1:
            # Reducir batch size
            global batch_size
            batch_size = max(1, batch_size // 2)
            return f"reduced_batch_size_to_{batch_size}"
        elif attempt == 2:
            # Cambiar a CPU
            return "switch_to_cpu"
    
    elif "model not found" in error_msg:
        if attempt == 0:
            # Intentar modelo alternativo
            return "try_alternative_model"
        elif attempt == 1:
            # Descargar modelo
            return "download_model"
    
    return False

# Registrar estrategia
error_handler.register_recovery_strategy(
    "advanced recovery", 
    advanced_recovery_strategy
)
```

## Ejemplos de Uso Completo

### Sistema de Entrenamiento con Debugging

```python
from error_handling_debugging_system import *

# Inicializar sistema completo
error_handler = ErrorHandler(enable_debug=True)
debugger = PyTorchDebugger(enable_anomaly_detection=True)
monitor = PerformanceMonitor(log_interval=30)
recovery_manager = ErrorRecoveryManager(error_handler)

# Modelo y optimizador
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Función de entrenamiento con debugging automático
@DebugDecorator(error_handler, max_retries=3)
def training_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        try:
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Verificar gradientes antes de optimizar
            grad_info = debugger.check_gradients(model)
            if grad_info["nan_gradients"] or grad_info["inf_gradients"]:
                raise RuntimeError("Gradientes problemáticos detectados")
            
            # Optimizar
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
        except Exception as e:
            # Manejo automático de errores
            success, result = error_handler.handle_error(
                e, f"batch_{batch_idx}", max_retries=2
            )
            
            if not success:
                print(f"Error en batch {batch_idx}: {e}")
                continue
    
    return total_loss / len(dataloader)

# Entrenamiento con monitoreo
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # Monitorear sistema
    status = monitor.log_system_status()
    
    # Aplicar fixes automáticos
    fixes = recovery_manager.apply_common_fixes(model, optimizer)
    
    # Entrenar con debugging automático
    avg_loss = training_epoch(model, train_loader, optimizer)
    
    print(f"Average loss: {avg_loss:.4f}")
    
    # Verificar estado del modelo
    param_info = debugger.check_model_parameters(model)
    if param_info["nan_parameters"]:
        print("⚠️ Parámetros NaN detectados, aplicando fixes...")
        recovery_manager.apply_common_fixes(model, optimizer)
```

## Estructura del Proyecto

```
error_handling_debugging_system.py     # Sistema principal
requirements_error_handling_debugging.txt  # Dependencias
README_ERROR_HANDLING_DEBUGGING.md     # Documentación
```

## Dependencias Principales

- **torch**: Framework de deep learning
- **psutil**: Monitoreo del sistema
- **logging**: Sistema de logging estándar de Python

## Requisitos del Sistema

- **Python**: 3.8+
- **PyTorch**: 2.0.0+
- **psutil**: 5.8.0+

## Solución de Problemas

### Error de Logging

```python
# Verificar permisos de directorio
import os
os.makedirs("logs", exist_ok=True)

# Verificar que el archivo de log se puede escribir
try:
    with open("logs/test.log", "w") as f:
        f.write("test")
    os.remove("logs/test.log")
except Exception as e:
    print(f"Error de permisos: {e}")
```

### Error de psutil

```python
# Instalar psutil si no está disponible
try:
    import psutil
except ImportError:
    print("Instalando psutil...")
    import subprocess
    subprocess.check_call(["pip", "install", "psutil"])
    import psutil
```

### Error de CUDA

```python
# Verificar disponibilidad de CUDA
if not torch.cuda.is_available():
    print("CUDA no disponible, usando CPU")
    device = "cpu"
else:
    print(f"CUDA disponible: {torch.cuda.get_device_name()}")
    device = "cuda"
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



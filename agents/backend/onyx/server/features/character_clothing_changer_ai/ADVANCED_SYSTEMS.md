# 🚀 Sistemas Avanzados - Character Clothing Changer AI

## ✨ Nuevos Sistemas Implementados

### 1. **Batch Processor** (`batch_processor.py`)

Procesamiento en lote avanzado con soporte para procesamiento paralelo:

- ✅ **Procesamiento secuencial**: Para casos donde se necesita orden
- ✅ **Procesamiento paralelo**: Múltiples workers simultáneos
- ✅ **Progress tracking**: Callbacks para seguimiento de progreso
- ✅ **Error handling**: Manejo robusto de errores por item
- ✅ **Streaming**: Procesamiento desde colas

**Uso:**
```python
from character_clothing_changer_ai.models import BatchProcessor, BatchItem

processor = BatchProcessor(
    model=model,
    max_workers=4,
    batch_size=8,
    enable_progress=True,
)

# Crear batch
items = [
    BatchItem(
        image="char1.jpg",
        clothing_description="red dress",
        metadata={"id": 1}
    ),
    BatchItem(
        image="char2.jpg",
        clothing_description="blue suit",
        metadata={"id": 2}
    ),
]

# Procesar con callback
def progress_callback(idx, result):
    print(f"Item {idx}: {'✓' if result.success else '✗'}")

results = processor.process_batch(items, callback=progress_callback)

# Ver resultados
for result in results:
    if result.success:
        result.result.save(f"output_{result.metadata['id']}.jpg")
```

### 2. **Performance Monitor** (`performance_monitor.py`)

Monitoreo de rendimiento en tiempo real:

- ✅ **Métricas en tiempo real**: CPU, memoria, GPU
- ✅ **Historial de métricas**: Almacenamiento de historial
- ✅ **Promedios**: Cálculo de promedios sobre períodos
- ✅ **Thread-safe**: Monitoreo en background thread
- ✅ **GPU monitoring**: Monitoreo de memoria y utilización GPU

**Uso:**
```python
from character_clothing_changer_ai.models import PerformanceMonitor

monitor = PerformanceMonitor(
    history_size=1000,
    update_interval=1.0,
    enable_gpu_monitoring=True,
)

# Iniciar monitoreo
monitor.start_monitoring()

# ... procesar imágenes ...

# Ver métricas actuales
current = monitor.get_current_metrics()
print(f"CPU: {current.cpu_percent}%")
print(f"Memoria: {current.memory_mb:.2f} MB")
print(f"GPU: {current.gpu_memory_mb:.2f} MB")

# Ver promedios (últimos 60 segundos)
averages = monitor.get_average_metrics(duration=60.0)
print(f"CPU promedio: {averages['avg_cpu_percent']:.2f}%")

# Ver resumen completo
summary = monitor.get_metrics_summary()

# Detener monitoreo
monitor.stop_monitoring()
```

### 3. **Queue Manager** (`queue_manager.py`)

Sistema de colas asíncrono para procesamiento:

- ✅ **Cola asíncrona**: Procesamiento en background
- ✅ **Múltiples workers**: Procesamiento paralelo
- ✅ **Status tracking**: Seguimiento de estado de tareas
- ✅ **Task management**: Cancelación, consulta de estado
- ✅ **Thread-safe**: Operaciones thread-safe

**Uso:**
```python
from character_clothing_changer_ai.models import QueueManager

queue = QueueManager(
    model=model,
    max_queue_size=100,
    max_workers=4,
)

# Iniciar queue
queue.start()

# Enviar tareas
task_id1 = queue.submit_task(
    image="char1.jpg",
    clothing_description="red dress",
    metadata={"user_id": 123}
)

task_id2 = queue.submit_task(
    image="char2.jpg",
    clothing_description="blue suit",
)

# Verificar estado
status = queue.get_task_status(task_id1)
print(f"Estado: {status['status']}")

# Esperar y obtener resultado
import time
while status['status'] != 'completed':
    time.sleep(0.5)
    status = queue.get_task_status(task_id1)

result = queue.get_task_result(task_id1)
result.save("output1.jpg")

# Ver estadísticas de cola
stats = queue.get_queue_stats()
print(f"Tareas pendientes: {stats['tasks']['pending']}")
print(f"Tareas completadas: {stats['tasks']['completed']}")

# Detener queue
queue.stop()
```

### 4. **Quality Analyzer** (`quality_analyzer.py`)

Análisis avanzado de calidad de imágenes generadas:

- ✅ **Múltiples métricas**: Realismo, consistencia, detalle, etc.
- ✅ **SSIM**: Structural Similarity Index
- ✅ **Detección de artefactos**: Identificación de problemas
- ✅ **Análisis de color**: Precisión de color
- ✅ **Score general**: Puntuación combinada

**Uso:**
```python
from character_clothing_changer_ai.models import QualityAnalyzer

analyzer = QualityAnalyzer()

# Analizar calidad
score = analyzer.analyze(
    original=original_image,
    generated=generated_image,
    mask=mask_image,  # Opcional
)

print(f"Score general: {score.overall:.3f}")
print(f"Realismo: {score.realism:.3f}")
print(f"Consistencia: {score.consistency:.3f}")
print(f"Detalle: {score.detail:.3f}")
print(f"Precisión de color: {score.color_accuracy:.3f}")
print(f"Artefactos: {score.artifacts:.3f} (menor es mejor)")
```

## 🔄 Integración Completa

### Sistema Completo con Todos los Componentes

```python
from character_clothing_changer_ai.models import (
    Flux2ClothingChangerModelV2,
    BatchProcessor,
    PerformanceMonitor,
    QueueManager,
    QualityAnalyzer,
)

# Inicializar modelo
model = Flux2ClothingChangerModelV2(
    validate_images=True,
    enhance_images=True,
    max_retries=3,
)

# Inicializar sistemas
monitor = PerformanceMonitor()
monitor.start_monitoring()

queue = QueueManager(model, max_workers=4)
queue.start()

analyzer = QualityAnalyzer()

# Procesar con análisis de calidad
def process_with_quality(image, clothing_desc):
    # Enviar a cola
    task_id = queue.submit_task(image, clothing_desc)
    
    # Esperar resultado
    while True:
        status = queue.get_task_status(task_id)
        if status['status'] == 'completed':
            result = queue.get_task_result(task_id)
            
            # Analizar calidad
            score = analyzer.analyze(original_image, result)
            
            return result, score
        elif status['status'] == 'failed':
            raise RuntimeError(f"Task failed: {status['error']}")
        time.sleep(0.1)

# Ver métricas
metrics = monitor.get_average_metrics(duration=60.0)
print(f"Rendimiento: {metrics}")
```

## 📊 Métricas y Estadísticas

### Performance Monitor Metrics

- **CPU Percent**: Uso de CPU
- **Memory MB**: Uso de memoria RAM
- **GPU Memory MB**: Uso de memoria GPU
- **GPU Utilization**: Utilización GPU (estimada)

### Queue Manager Stats

- **Queue Size**: Tamaño actual de cola
- **Tasks**: Desglose por estado
  - Total
  - Pending
  - Processing
  - Completed
  - Failed

### Quality Scores

- **Overall**: Score general (0.0 a 1.0)
- **Realism**: Realismo de la imagen
- **Consistency**: Consistencia con original
- **Detail**: Nivel de detalle
- **Color Accuracy**: Precisión de color
- **Composition**: Composición
- **Artifacts**: Artefactos detectados (menor es mejor)

## 🎯 Casos de Uso

### 1. Procesamiento en Lote de Múltiples Personajes

```python
processor = BatchProcessor(model, max_workers=4)

items = [
    BatchItem(image=f"char{i}.jpg", clothing_description="red dress")
    for i in range(100)
]

results = processor.process_batch(items)
success_rate = sum(1 for r in results if r.success) / len(results)
print(f"Tasa de éxito: {success_rate:.2%}")
```

### 2. API Asíncrona con Queue

```python
# En endpoint de API
@app.post("/change-clothing")
async def change_clothing(request: ChangeClothingRequest):
    task_id = queue.submit_task(
        image=request.image,
        clothing_description=request.description,
    )
    return {"task_id": task_id}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    status = queue.get_task_status(task_id)
    return status
```

### 3. Monitoreo de Rendimiento en Producción

```python
monitor = PerformanceMonitor()
monitor.start_monitoring()

# En endpoint de métricas
@app.get("/metrics")
async def get_metrics():
    return {
        "current": monitor.get_current_metrics(),
        "averages": monitor.get_average_metrics(duration=300),
        "queue": queue.get_queue_stats(),
    }
```

## 🚀 Ventajas

1. **Escalabilidad**: Procesamiento paralelo y colas
2. **Monitoreo**: Visibilidad completa del sistema
3. **Calidad**: Análisis automático de resultados
4. **Robustez**: Manejo de errores y reintentos
5. **Eficiencia**: Optimización de recursos

## 📈 Mejoras de Rendimiento

- **Batch Processing**: Hasta 4x más rápido con paralelización
- **Queue System**: Procesamiento asíncrono sin bloqueos
- **Performance Monitoring**: Optimización basada en métricas
- **Quality Analysis**: Mejora continua de resultados



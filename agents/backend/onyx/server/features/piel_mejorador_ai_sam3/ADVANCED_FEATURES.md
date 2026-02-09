# Características Avanzadas - Piel Mejorador AI SAM3

## Resumen

Este documento describe las características avanzadas implementadas en el proyecto, incluyendo procesamiento frame-by-frame, caché inteligente, procesamiento en lote, y más.

## 🎬 Procesamiento Frame-by-Frame para Videos

### Descripción

El sistema ahora puede procesar videos frame por frame para mejoras más precisas y consistentes.

**Archivo:** `core/video_processor.py`

### Características

- ✅ Extracción de frames de video
- ✅ Procesamiento individual de frames
- ✅ Reconstrucción de video mejorado
- ✅ Procesamiento en lotes de frames
- ✅ Limpieza automática de archivos temporales
- ✅ Seguimiento de progreso

### Uso

```python
from piel_mejorador_ai_sam3.core.video_processor import VideoProcessor

processor = VideoProcessor()

# Extraer frames
frames = await processor.extract_frames(
    video_path="video.mp4",
    max_frames=100,  # Opcional: limitar frames
    frame_interval=1  # Cada frame
)

# Procesar frames
async def process_frame(frame):
    # Tu lógica de procesamiento
    return {"processed_path": "path/to/processed.jpg"}

results = await processor.process_frames(
    frames=frames,
    process_func=process_frame,
    batch_size=5
)

# Reconstruir video
output_video = await processor.reconstruct_video(
    frames=results,
    output_path="enhanced_video.mp4",
    fps=30.0
)

# Limpiar archivos temporales
processor.cleanup_temp_files(frames)
```

## 💾 Sistema de Caché Inteligente

### Descripción

Sistema de caché que evita reprocesamiento de archivos ya procesados.

**Archivo:** `core/cache_manager.py`

### Características

- ✅ Caché basado en hash de parámetros
- ✅ TTL (Time To Live) configurable
- ✅ Validación de modificación de archivos
- ✅ Limpieza automática de entradas expiradas
- ✅ Estadísticas de caché (hits, misses, tasa de aciertos)

### Uso

```python
from piel_mejorador_ai_sam3.core.cache_manager import CacheManager

cache = CacheManager(cache_dir=Path("cache"), default_ttl_hours=24)

# Obtener del caché
cached_result = await cache.get(
    file_path="image.jpg",
    enhancement_level="high",
    realism_level=0.9
)

if cached_result:
    print("Resultado desde caché!")
else:
    # Procesar y guardar en caché
    result = await process_image(...)
    await cache.set(
        file_path="image.jpg",
        enhancement_level="high",
        result=result,
        realism_level=0.9
    )

# Limpiar entradas expiradas
cleaned = await cache.cleanup_expired()

# Estadísticas
stats = cache.get_stats()
print(f"Tasa de aciertos: {stats['hit_rate']:.2%}")
```

## 📦 Procesamiento en Lote (Batch Processing)

### Descripción

Procesa múltiples archivos simultáneamente con seguimiento de progreso.

**Archivo:** `core/batch_processor.py`

### Características

- ✅ Procesamiento paralelo de múltiples archivos
- ✅ Control de concurrencia
- ✅ Seguimiento de progreso
- ✅ Manejo de errores por item
- ✅ Agregación de resultados
- ✅ Estadísticas de procesamiento

### Uso

```python
from piel_mejorador_ai_sam3.core.batch_processor import BatchItem, BatchProcessor

# Crear items para procesar
items = [
    BatchItem(
        file_path="image1.jpg",
        enhancement_level="high",
        realism_level=0.9
    ),
    BatchItem(
        file_path="image2.jpg",
        enhancement_level="medium",
        realism_level=0.7
    ),
]

# Procesar en lote
processor = BatchProcessor(max_concurrent=5)

async def progress_callback(completed, failed, total):
    print(f"Progreso: {completed}/{total} completados, {failed} fallidos")

result = await processor.process_batch(
    items=items,
    process_func=process_image_function,
    progress_callback=progress_callback
)

print(f"Tasa de éxito: {result.success_rate:.2%}")
print(f"Duración: {result.duration:.2f}s")
```

### Uso desde el Agente

```python
from piel_mejorador_ai_sam3.core.batch_processor import BatchItem

items = [
    BatchItem(file_path="img1.jpg", enhancement_level="high"),
    BatchItem(file_path="img2.jpg", enhancement_level="medium"),
]

result = await agent.process_batch(items)
```

## 📊 Logging Avanzado

### Descripción

Sistema de logging estructurado con seguimiento de performance.

**Archivo:** `core/logging_config.py`

### Características

- ✅ Logging estructurado (JSON)
- ✅ Filtros de performance
- ✅ Logging a archivo y consola
- ✅ Contexto adicional en logs
- ✅ Configuración flexible

### Uso

```python
from piel_mejorador_ai_sam3.core.logging_config import setup_logging, get_logger

# Configurar logging
logger = setup_logging(
    level="INFO",
    log_file=Path("app.log"),
    structured=True,  # JSON logging
    performance_tracking=True
)

# Usar logger
logger.info("Procesando imagen", extra={
    "task_id": "123",
    "file_path": "image.jpg",
    "performance": True
})

# Logger específico
module_logger = get_logger("video_processor")
module_logger.debug("Frame procesado", extra={"frame_number": 42})
```

## 🔌 Nuevos Endpoints de API

### Batch Processing

```bash
POST /batch-process
Content-Type: application/json

{
  "items": [
    {
      "file_path": "image1.jpg",
      "enhancement_level": "high",
      "realism_level": 0.9
    },
    {
      "file_path": "image2.jpg",
      "enhancement_level": "medium"
    }
  ]
}
```

### Cache Management

```bash
# Limpiar caché expirado
POST /cache/cleanup

# Estadísticas de caché
GET /cache/stats
```

### Estadísticas Mejoradas

```bash
GET /stats

# Retorna:
{
  "executor_stats": {...},
  "cache_stats": {
    "hits": 150,
    "misses": 50,
    "hit_rate": 0.75,
    "cache_size": 200
  },
  "running": true,
  "max_parallel_tasks": 5
}
```

## 🚀 Optimizaciones de Rendimiento

### 1. Caché Inteligente

- Evita reprocesamiento de archivos idénticos
- Reduce llamadas a APIs externas
- Mejora tiempos de respuesta

### 2. Procesamiento Paralelo

- Worker pool eficiente
- Control de concurrencia
- Mejor utilización de recursos

### 3. Procesamiento en Lote

- Procesa múltiples archivos simultáneamente
- Optimiza uso de recursos
- Reduce tiempo total de procesamiento

### 4. Limpieza Automática

- Archivos temporales se limpian automáticamente
- Caché expirado se elimina periódicamente
- Gestión eficiente de memoria

## 📈 Métricas y Monitoreo

### Métricas Disponibles

1. **Executor Stats**
   - Tareas totales, completadas, fallidas
   - Tiempo promedio de tareas
   - Tasa de éxito
   - Workers activos

2. **Cache Stats**
   - Hits y misses
   - Tasa de aciertos
   - Tamaño del caché
   - Evicciones

3. **Batch Stats**
   - Items procesados
   - Tasa de éxito
   - Duración total
   - Errores por item

## 🔧 Configuración

### Variables de Entorno

```bash
# Logging
PIEL_MEJORADOR_LOG_LEVEL=INFO
PIEL_MEJORADOR_LOG_FILE=app.log
PIEL_MEJORADOR_STRUCTURED_LOGGING=false

# Cache
PIEL_MEJORADOR_CACHE_TTL_HOURS=24
PIEL_MEJORADOR_CACHE_DIR=cache

# Batch Processing
PIEL_MEJORADOR_MAX_CONCURRENT=5
```

## 📝 Ejemplos Completos

### Procesamiento de Video Completo

```python
from piel_mejorador_ai_sam3 import PielMejoradorAgent, PielMejoradorConfig
from piel_mejorador_ai_sam3.core.video_processor import VideoProcessor

config = PielMejoradorConfig()
agent = PielMejoradorAgent(config=config)

processor = VideoProcessor()

# Extraer frames
frames = await processor.extract_frames("video.mp4", frame_interval=2)

# Procesar cada frame
async def enhance_frame(frame):
    task_id = await agent.mejorar_imagen(
        file_path=frame["image_path"],
        enhancement_level="high"
    )
    # Esperar resultado...
    return {"processed_path": "enhanced_frame.jpg"}

results = await processor.process_frames(frames, enhance_frame)

# Reconstruir
output = await processor.reconstruct_video(results, "enhanced.mp4")
```

### Batch con Caché

```python
items = [BatchItem(file_path=f"img{i}.jpg") for i in range(10)]

result = await agent.process_batch(
    items,
    progress_callback=lambda c, f, t: print(f"{c}/{t}")
)

# El agente automáticamente usa caché para evitar reprocesamiento
```

## 🎯 Mejores Prácticas

1. **Usar Caché**: Siempre que sea posible, el sistema usa caché automáticamente
2. **Batch Processing**: Para múltiples archivos, usa batch processing
3. **Video Processing**: Para videos, usa frame-by-frame para mejor calidad
4. **Monitoreo**: Revisa estadísticas regularmente para optimizar
5. **Limpieza**: Ejecuta limpieza de caché periódicamente

## 📚 Referencias

- Video Processing: `core/video_processor.py`
- Cache Management: `core/cache_manager.py`
- Batch Processing: `core/batch_processor.py`
- Logging: `core/logging_config.py`





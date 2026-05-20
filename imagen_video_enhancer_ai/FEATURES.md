# Características Avanzadas - Imagen Video Enhancer AI

## Batch Processing

Procesa múltiples archivos en paralelo con seguimiento de progreso.

```python
from imagen_video_enhancer_ai.core.batch_processor import BatchItem

batch_items = [
    BatchItem(
        file_path="image1.jpg",
        service_type="enhance_image",
        enhancement_type="general"
    ),
    BatchItem(
        file_path="image2.jpg",
        service_type="upscale",
        options={"scale_factor": 2}
    ),
]

result = await agent.process_batch(batch_items)
print(f"Completed: {result.completed}/{result.total_items}")
```

## Cache Management

Sistema de caché inteligente que evita reprocesar archivos.

- **TTL configurable**: Entradas expiradas automáticamente
- **Invalidación por modificación**: Detecta cambios en archivos
- **Estadísticas**: Hits, misses, hit rate
- **Limpieza automática**: Elimina entradas expiradas

```python
# El cache se usa automáticamente
# Para limpiar manualmente:
await agent.cache_manager.cleanup_expired()

# Ver estadísticas:
stats = agent.cache_manager.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

## Rate Limiting

Protección contra abuso con token bucket rate limiter.

- **Por cliente**: Límites por IP
- **Burst support**: Permite ráfagas controladas
- **Configurable**: Ajusta límites según necesidades

```python
# Configurado automáticamente en la API
# 10 requests/segundo, burst de 20
```

## Estadísticas

Monitorea el rendimiento del sistema.

```python
stats = agent.get_stats()
# Incluye:
# - Estadísticas del ejecutor paralelo
# - Estadísticas de caché
# - Estado del agente
```

## Vision Models

Análisis real de imágenes usando modelos de visión.

- **Análisis automático**: Detecta calidad y problemas
- **Guías personalizadas**: Basadas en el contenido real
- **Fallback inteligente**: Modo texto si falla la validación

## Retry Automático

Sistema de reintentos automáticos para tareas fallidas.

```python
# Configuración de retry
retry_config = RetryConfig(
    max_retries=3,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    initial_delay=1.0,
    max_delay=60.0
)

# Se aplica automáticamente a tareas fallidas
# Errores retryables: timeout, connection, rate_limit, temporary
```

## Exportación de Resultados

Exporta resultados a múltiples formatos.

```python
# Exportar a JSON
await agent.export_results(format="json", output_path="results.json")

# Exportar a Markdown
await agent.export_results(format="markdown", output_path="results.md")

# Exportar a CSV
await agent.export_results(format="csv", output_path="results.csv")

# Exportar a HTML (reporte)
await agent.export_results(format="html", output_path="results.html")

# Con compresión
await agent.export_results(format="json", compress=True)
```

## Compresión

Comprime resultados y archivos para ahorrar espacio.

```python
from imagen_video_enhancer_ai.utils.compression import CompressionManager

# Comprimir JSON
compressed = CompressionManager.compress_json(data, "output.json.gz")

# Descomprimir
data = CompressionManager.decompress_json("output.json.gz")

# Comprimir archivo
compressed = CompressionManager.compress_file("large_file.jpg")

# Descomprimir archivo
decompressed = CompressionManager.decompress_file("large_file.jpg.gz")
```

## Validación Robusta

Validación completa de archivos y parámetros.

- **Formatos permitidos**: Extensiones verificadas
- **Tamaños máximos**: Límites configurables
- **Parámetros**: Tipos, rangos, valores válidos
- **Mensajes claros**: Errores descriptivos


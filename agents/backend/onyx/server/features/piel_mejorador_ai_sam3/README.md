# Piel Mejorador AI SAM3

Sistema de mejoramiento de piel con arquitectura SAM3, integrado con OpenRouter y TruthGPT para procesamiento de imágenes y videos con niveles configurables de mejora y realismo.

## Características

### Core
- ✅ Arquitectura SAM3 para procesamiento paralelo y continuo
- ✅ Integración con OpenRouter para LLM de alta calidad con soporte vision
- ✅ Integración con TruthGPT para optimización avanzada
- ✅ Operación continua 24/7
- ✅ Ejecución paralela de tareas
- ✅ Gestión automática de tareas con cola de prioridades
- ✅ Procesamiento de imágenes y videos
- ✅ Niveles configurables de mejora (low, medium, high, ultra)
- ✅ Niveles configurables de realismo (0.0 a 1.0)
- ✅ Análisis de condición de piel

### Avanzadas
- ✅ Procesamiento frame-by-frame para videos
- ✅ Sistema de caché inteligente
- ✅ Procesamiento en lote (batch processing)
- ✅ Logging avanzado estructurado

### Enterprise
- ✅ Rate limiting con token bucket
- ✅ Sistema de webhooks para notificaciones
- ✅ Optimización automática de memoria
- ✅ Métricas y monitoreo avanzado
- ✅ Health checks y recomendaciones

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

Configura las variables de entorno:

```bash
export OPENROUTER_API_KEY="tu-api-key"
export TRUTHGPT_ENDPOINT="opcional-endpoint"  # Opcional
```

## Uso Básico

### Uso del Agente

```python
import asyncio
from piel_mejorador_ai_sam3 import PielMejoradorAgent, PielMejoradorConfig

async def main():
    # Crear configuración
    config = PielMejoradorConfig()
    
    # Crear agente
    agent = PielMejoradorAgent(config=config)
    
    # Mejorar una imagen
    task_id = await agent.mejorar_imagen(
        file_path="ruta/a/imagen.jpg",
        enhancement_level="medium",
        realism_level=0.8
    )
    
    # Esperar resultado
    import time
    while True:
        status = await agent.get_task_status(task_id)
        if status["status"] == "completed":
            result = await agent.get_task_result(task_id)
            print(result)
            break
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
```

### Uso de la API REST

Inicia el servidor:

```bash
uvicorn piel_mejorador_ai_sam3.api.piel_mejorador_api:app --reload
```

#### Subir y mejorar imagen

```bash
curl -X POST "http://localhost:8000/upload-image" \
  -F "file=@imagen.jpg" \
  -F "enhancement_level=medium" \
  -F "realism_level=0.8"
```

#### Mejorar imagen desde ruta

```bash
curl -X POST "http://localhost:8000/mejorar-imagen" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/ruta/a/imagen.jpg",
    "enhancement_level": "high",
    "realism_level": 0.9
  }'
```

#### Mejorar video

```bash
curl -X POST "http://localhost:8000/mejorar-video" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/ruta/a/video.mp4",
    "enhancement_level": "medium",
    "realism_level": 0.7
  }'
```

#### Analizar piel

```bash
curl -X POST "http://localhost:8000/analizar-piel" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/ruta/a/imagen.jpg",
    "file_type": "image"
  }'
```

#### Consultar estado de tarea

```bash
curl "http://localhost:8000/task/{task_id}/status"
```

#### Obtener resultado

```bash
curl "http://localhost:8000/task/{task_id}/result"
```

## Arquitectura

### Estructura de Directorios

```
piel_mejorador_ai_sam3/
├── core/
│   ├── piel_mejorador_agent.py    # Agente principal (orchestrator)
│   ├── task_manager.py            # Gestión de tareas y cola de prioridades
│   ├── service_handler.py         # Manejo de servicios de mejora
│   ├── prompt_builder.py          # Construcción de prompts
│   ├── system_prompts_builder.py  # Prompts del sistema especializados
│   └── helpers.py                 # Utilidades comunes
├── infrastructure/
│   ├── openrouter_client.py       # Cliente OpenRouter (LLM + Vision)
│   ├── truthgpt_client.py         # Cliente TruthGPT (optimización)
│   └── retry_helpers.py           # Helpers de reintentos con backoff
├── config/
│   └── piel_mejorador_config.py  # Configuración centralizada
├── api/
│   └── piel_mejorador_api.py      # API REST (FastAPI)
├── utils/
│   └── (utilidades adicionales)
├── tests/
│   └── (tests)
├── examples/
│   └── (ejemplos de uso)
└── docs/
    └── (documentación)
```

## Servicios Disponibles

### 1. Mejorar Imagen

Mejora la piel en una imagen estática.

```python
task_id = await agent.mejorar_imagen(
    file_path="imagen.jpg",
    enhancement_level="medium",  # low, medium, high, ultra
    realism_level=0.8,  # 0.0 a 1.0 (opcional)
    custom_instructions="Enfocarse en suavizar textura",
    priority=0
)
```

### 2. Mejorar Video

Mejora la piel en un video manteniendo consistencia entre frames.

```python
task_id = await agent.mejorar_video(
    file_path="video.mp4",
    enhancement_level="high",
    realism_level=0.9,
    custom_instructions="Mantener movimiento natural",
    priority=0
)
```

### 3. Analizar Piel

Analiza la condición de la piel y proporciona recomendaciones.

```python
task_id = await agent.analizar_piel(
    file_path="imagen.jpg",
    file_type="image",  # o "video"
    priority=0
)
```

## Niveles de Mejora

- **low**: Mejoras sutiles y naturales (intensidad: 0.3, realismo: 0.5)
- **medium**: Mejoras moderadas manteniendo realismo (intensidad: 0.6, realismo: 0.7)
- **high**: Mejoras significativas con alto realismo (intensidad: 0.9, realismo: 0.9)
- **ultra**: Mejoras máximas con realismo fotográfico perfecto (intensidad: 1.0, realismo: 1.0)

## Niveles de Realismo

El nivel de realismo puede especificarse como un valor flotante entre 0.0 y 1.0:
- **0.0**: Natural, preserva características originales
- **0.5**: Balanceado entre natural y mejorado
- **1.0**: Realismo fotográfico perfecto

## Características Avanzadas

### Procesamiento Frame-by-Frame para Videos

El sistema puede procesar videos frame por frame para mejoras más precisas:

```python
from piel_mejorador_ai_sam3.core.video_processor import VideoProcessor

processor = VideoProcessor()
frames = await processor.extract_frames("video.mp4")
# Procesar frames...
enhanced_video = await processor.reconstruct_video(frames, "output.mp4")
```

### Sistema de Caché Inteligente

Evita reprocesamiento de archivos ya procesados:

```python
# El caché se usa automáticamente
# Ver estadísticas:
stats = agent.cache_manager.get_stats()
print(f"Tasa de aciertos: {stats['hit_rate']:.2%}")
```

### Procesamiento en Lote

Procesa múltiples archivos simultáneamente:

```python
from piel_mejorador_ai_sam3.core.batch_processor import BatchItem

items = [
    BatchItem(file_path="img1.jpg", enhancement_level="high"),
    BatchItem(file_path="img2.jpg", enhancement_level="medium"),
]

result = await agent.process_batch(items)
print(f"Tasa de éxito: {result.success_rate:.2%}")
```

Ver **[ADVANCED_FEATURES.md](ADVANCED_FEATURES.md)** para más detalles.

### Características Enterprise

#### Rate Limiting
Protección automática contra abuso de API con límites configurables por cliente.

#### Webhooks
Sistema completo de notificaciones asíncronas para eventos de tareas.

#### Optimización de Memoria
Monitoreo y optimización automática de memoria para archivos grandes.

Ver **[ENTERPRISE_FEATURES.md](ENTERPRISE_FEATURES.md)** para más detalles.

### Modo 24/7 Continuo

El agente puede ejecutarse en modo continuo procesando tareas automáticamente:

```python
agent = PielMejoradorAgent(config=config)
await agent.start()  # Ejecuta indefinidamente
```

### Priorización de Tareas

Las tareas pueden tener diferentes prioridades:

```python
# Alta prioridad
task_id = await agent.mejorar_imagen(
    file_path="imagen.jpg",
    enhancement_level="high",
    priority=10  # Mayor prioridad
)
```

### Integración con TruthGPT

El agente optimiza automáticamente las consultas usando TruthGPT cuando está disponible.

## Formatos Soportados

### Imágenes
- JPG/JPEG
- PNG
- WebP

Tamaño máximo: 50MB (configurable)

### Videos
- MP4
- MOV
- AVI
- WebM

Tamaño máximo: 500MB (configurable)

## Requisitos

- Python 3.8+
- OpenRouter API key
- TruthGPT (opcional pero recomendado)

## Licencia

MIT


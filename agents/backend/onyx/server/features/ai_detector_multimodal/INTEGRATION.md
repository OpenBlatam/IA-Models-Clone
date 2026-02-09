# Guía de Integración

## Integración con FastAPI

Para integrar el detector en tu aplicación FastAPI:

```python
from fastapi import FastAPI
from ai_detector_multimodal.api.router import router

app = FastAPI(title="Mi Aplicación")

# Incluir el router del detector
app.include_router(router)

# Tu aplicación ahora tiene los endpoints:
# - POST /ai-detector/detect
# - POST /ai-detector/batch
# - GET /ai-detector/health
# - GET /ai-detector/models
```

## Uso desde código Python

```python
from ai_detector_multimodal.core.detector import MultimodalAIDetector
from ai_detector_multimodal.schemas import ContentType

# Inicializar detector
detector = MultimodalAIDetector()

# Detectar texto
resultado = detector.detect(
    content="Texto a analizar...",
    content_type=ContentType.TEXT.value,
    metadata={"source": "web"}
)

print(f"Es IA: {resultado['is_ai_generated']}")
print(f"Porcentaje: {resultado['ai_percentage']}%")
print(f"Modelo: {resultado['primary_model']['model_name']}")
```

## Integración con otros features

El detector puede ser usado por otros features del sistema:

```python
# En otro feature
from ai_detector_multimodal.core.detector import MultimodalAIDetector

detector = MultimodalAIDetector()

def analizar_contenido_usuario(contenido):
    resultado = detector.detect(contenido, "text")
    
    if resultado['is_ai_generated']:
        # Tomar acción si es contenido generado por IA
        pass
    
    return resultado
```

## Endpoints disponibles

### POST `/ai-detector/detect`
Detecta si un contenido fue generado por IA.

**Body:**
```json
{
  "content": "texto o contenido",
  "content_type": "text|image|audio|video",
  "metadata": {}
}
```

### POST `/ai-detector/batch`
Detección en batch.

**Body:**
```json
{
  "items": [
    {
      "content": "texto 1",
      "content_type": "text"
    },
    {
      "content": "texto 2",
      "content_type": "text"
    }
  ],
  "parallel": true
}
```

### GET `/ai-detector/health`
Health check.

### GET `/ai-detector/models`
Lista modelos detectables.







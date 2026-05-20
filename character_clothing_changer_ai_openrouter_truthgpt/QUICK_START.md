# Quick Start Guide

## 🚀 Inicio Rápido

### Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt
```

### Configuración

1. Configurar variables de entorno en `.env`:
```env
COMFYUI_API_URL=http://localhost:8188
COMFYUI_WORKFLOW_PATH=workflows/flux_fill_clothing_changer.json
OPENROUTER_ENABLED=true
OPENROUTER_API_KEY=your_key_here
TRUTHGPT_ENABLED=true
TRUTHGPT_ENDPOINT=http://localhost:8000
```

### Ejecutar

```bash
# Ejecutar servidor
python main.py
```

O con uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📡 Endpoints Principales

### Clothing Change
```bash
POST /api/v1/clothing/change
{
  "image_url": "https://example.com/image.png",
  "clothing_description": "a red elegant dress"
}
```

### Face Swap
```bash
POST /api/v1/face-swap
{
  "image_url": "https://example.com/image.png",
  "face_url": "https://example.com/face.png"
}
```

### Batch Processing
```bash
POST /api/v1/clothing/batch
{
  "items": [
    {
      "image_url": "https://example.com/image1.png",
      "clothing_description": "red dress"
    },
    {
      "image_url": "https://example.com/image2.png",
      "clothing_description": "blue suit"
    }
  ]
}
```

### Health Check
```bash
GET /api/v1/health
```

## 🔧 Configuración Avanzada

Ver `config/advanced_settings.py` para opciones avanzadas:
- Logging
- Cache
- Rate limiting
- Performance tracking
- Webhooks
- Security

## 📚 Documentación Completa

- `README.md` - Documentación principal
- `ARCHITECTURE.md` - Arquitectura del sistema
- `COMPLETE_FEATURES.md` - Todas las características
- `INDEX.md` - Índice de documentación

## 🎯 Ejemplos

### Python
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/clothing/change",
        json={
            "image_url": "https://example.com/image.png",
            "clothing_description": "a red elegant dress"
        }
    )
    result = response.json()
    print(f"Prompt ID: {result['prompt_id']}")
```

### cURL
```bash
curl -X POST http://localhost:8000/api/v1/clothing/change \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.png",
    "clothing_description": "a red elegant dress"
  }'
```

## ✅ Verificación

1. Health check:
```bash
curl http://localhost:8000/api/v1/health
```

2. Ver documentación:
```
http://localhost:8000/docs
```

3. Ver analytics:
```bash
curl http://localhost:8000/api/v1/clothing/analytics
```

¡Listo para usar!


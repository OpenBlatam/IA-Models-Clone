# Quick Start Guide - Faceless Video AI

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Navegar al directorio
cd agents/backend/onyx/server/features/faceless_video_ai

# Instalar dependencias
pip install -r requirements.txt

# Asegurarse de tener FFmpeg instalado
# Windows: Descargar de https://ffmpeg.org/download.html
# macOS: brew install ffmpeg
# Linux: sudo apt-get install ffmpeg
```

### 2. Configuración (Opcional)

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env con tus API keys (opcional)
# Si no tienes API keys, el sistema usará placeholders
```

### 3. Ejecutar el Servidor

```bash
# Opción 1: Usar el script incluido
python run_api.py

# Opción 2: Usar uvicorn directamente
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Probar la API

Abre tu navegador en: `http://localhost:8000/docs`

O usa curl:

```bash
# Health check
curl http://localhost:8000/health

# Generar un video
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "script": {
      "text": "Bienvenidos a este increíble video sobre inteligencia artificial. Hoy exploraremos las últimas tendencias.",
      "language": "es"
    },
    "video_config": {
      "resolution": "1920x1080",
      "fps": 30,
      "style": "realistic"
    },
    "audio_config": {
      "voice": "neutral",
      "speed": 1.0
    },
    "subtitle_config": {
      "enabled": true,
      "style": "modern"
    }
  }'
```

## 📝 Ejemplo en Python

```python
import requests
import time
from uuid import UUID

# 1. Generar video
response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={
        "script": {
            "text": "Este es un ejemplo de video generado completamente con IA. "
                    "El sistema crea imágenes, audio y subtítulos automáticamente.",
            "language": "es"
        },
        "video_config": {
            "resolution": "1920x1080",
            "fps": 30,
            "style": "realistic"
        },
        "audio_config": {
            "voice": "neutral",
            "speed": 1.0
        },
        "subtitle_config": {
            "enabled": True,
            "style": "modern"
        }
    }
)

video_job = response.json()
video_id = video_job["video_id"]
print(f"Video ID: {video_id}")

# 2. Verificar estado
while True:
    status_response = requests.get(f"http://localhost:8000/api/v1/status/{video_id}")
    status = status_response.json()
    
    print(f"Status: {status['status']} - {status['progress']['progress']:.1f}%")
    print(f"Step: {status['progress']['current_step']}")
    
    if status["status"] == "completed":
        print("Video generado exitosamente!")
        break
    elif status["status"] == "failed":
        print(f"Error: {status.get('error', 'Unknown error')}")
        break
    
    time.sleep(2)

# 3. Descargar video
if status["status"] == "completed":
    video_response = requests.get(f"http://localhost:8000/api/v1/videos/{video_id}/download")
    with open("output_video.mp4", "wb") as f:
        f.write(video_response.content)
    print("Video descargado: output_video.mp4")
```

## 🎯 Características Principales

- ✅ **Procesamiento de Scripts**: Divide automáticamente el texto en segmentos
- ✅ **Generación de Imágenes**: Crea imágenes con IA para cada segmento
- ✅ **Síntesis de Voz**: Convierte texto a voz
- ✅ **Subtítulos Automáticos**: Genera y sincroniza subtítulos
- ✅ **Composición de Video**: Combina todo en un video final

## 🔧 Notas Importantes

1. **FFmpeg es Requerido**: El sistema necesita FFmpeg para composición de video
2. **API Keys Opcionales**: Puedes usar el sistema sin API keys (usará placeholders)
3. **Procesamiento Asíncrono**: Los videos se generan en segundo plano
4. **Almacenamiento Temporal**: Los archivos se guardan en `/tmp/faceless_video` por defecto

## 📚 Documentación Completa

Ver `README.md` para documentación completa y detalles avanzados.


# Faceless Video AI

Sistema completo para generar videos sin rostro completamente con IA a partir de scripts, incluyendo generación de imágenes, audio, subtítulos y composición final.

## 🎬 Características

- **Generación de Videos con IA**: Crea videos completamente generados por IA sin necesidad de personas
- **Procesamiento de Scripts**: Procesa y segmenta scripts automáticamente
- **Generación de Imágenes**: Genera imágenes con IA para cada segmento del video
- **Síntesis de Voz (TTS)**: Convierte texto a voz con múltiples opciones de voz
- **Subtítulos Automáticos**: Genera y embebe subtítulos con múltiples estilos
- **Composición de Video**: Combina imágenes, audio y subtítulos en un video final
- **API REST**: API completa con FastAPI para integración
- **Procesamiento Asíncrono**: Generación de videos en segundo plano

## 📋 Requisitos

- Python 3.10+
- FFmpeg (para composición de video)
- API keys para servicios de IA (opcional, según servicios que uses)

## 🚀 Instalación

### Opción 1: Instalación Local

1. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

2. **Instalar FFmpeg**:
   - **Windows**: Descargar de [ffmpeg.org](https://ffmpeg.org/download.html)
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg`

3. **Configurar variables de entorno** (opcional):
```bash
# Crear archivo .env
OPENAI_API_KEY=tu_api_key
STABILITY_AI_API_KEY=tu_api_key
ELEVENLABS_API_KEY=tu_api_key
```

### Opción 2: Docker (Recomendado)

```bash
# Construir imagen
docker build -t faceless-video-ai .

# Ejecutar con docker-compose (incluye Redis, PostgreSQL, etc.)
docker-compose up -d
```

### Opción 3: Despliegue en AWS

Para despliegue completo en AWS (ECS/Fargate, Lambda, etc.), ver la [guía completa de despliegue en AWS](aws/deployment/AWS_DEPLOYMENT.md).

**Quick Start AWS:**
```bash
cd aws/deployment
make deploy  # Requiere AWS CLI configurado
```

## 🎯 Uso

### Iniciar el servidor API

```bash
python -m faceless_video_ai.api.main
# O usando uvicorn directamente
uvicorn faceless_video_ai.api.main:app --host 0.0.0.0 --port 8000
```

### Generar un video

```python
from faceless_video_ai.core.models import (
    VideoGenerationRequest,
    VideoScript,
    VideoConfig,
    AudioConfig,
    SubtitleConfig,
)

# Crear request
request = VideoGenerationRequest(
    script=VideoScript(
        text="Bienvenidos a este increíble video sobre inteligencia artificial. "
             "Hoy exploraremos las últimas tendencias en IA y cómo está transformando el mundo.",
        language="es"
    ),
    video_config=VideoConfig(
        resolution="1920x1080",
        fps=30,
        style="realistic"
    ),
    audio_config=AudioConfig(
        voice="neutral",
        speed=1.0
    ),
    subtitle_config=SubtitleConfig(
        enabled=True,
        style="modern"
    )
)

# Enviar request a la API
import requests
response = requests.post("http://localhost:8000/api/v1/generate", json=request.dict())
video_job = response.json()
```

### Verificar estado

```python
video_id = video_job["video_id"]
status_response = requests.get(f"http://localhost:8000/api/v1/status/{video_id}")
print(status_response.json())
```

### Descargar video

```python
video_response = requests.get(f"http://localhost:8000/api/v1/videos/{video_id}/download")
with open("output_video.mp4", "wb") as f:
    f.write(video_response.content)
```

## 📁 Estructura del Proyecto

```
faceless_video_ai/
├── __init__.py
├── core/
│   ├── __init__.py
│   └── models.py          # Modelos de datos Pydantic
├── services/
│   ├── __init__.py
│   ├── script_processor.py      # Procesamiento de scripts
│   ├── video_generator.py       # Generación de imágenes/video
│   ├── audio_generator.py       # Generación de audio (TTS)
│   ├── subtitle_generator.py   # Generación de subtítulos
│   ├── video_compositor.py      # Composición final de video
│   ├── video_orchestrator.py    # Orquestador principal
│   ├── storage/                  # Almacenamiento (S3, local)
│   ├── metrics/                  # Métricas Prometheus
│   └── ...                       # Otros servicios
├── api/
│   ├── __init__.py
│   ├── main.py           # Aplicación FastAPI
│   └── routes.py         # Endpoints de la API
├── config/
│   ├── __init__.py
│   └── settings.py       # Configuración (con soporte AWS)
├── aws/
│   └── deployment/       # Configuraciones de despliegue AWS
│       ├── Dockerfile
│       ├── docker-compose.yml
│       ├── cloudformation-template.yaml
│       ├── ecs-task-definition.json
│       ├── serverless.yml
│       └── AWS_DEPLOYMENT.md
├── Dockerfile            # Dockerfile de producción
├── Dockerfile.dev        # Dockerfile de desarrollo
├── docker-compose.yml    # Docker Compose con todos los servicios
├── requirements.txt
└── README.md
```

## 🔧 Configuración

### Servicios de IA

El sistema ahora incluye **integración real** con múltiples servicios de IA:

#### Generación de Imágenes
- ✅ **OpenAI DALL-E**: Configurar `OPENAI_API_KEY` (integración completa)
- ✅ **Stability AI**: Configurar `STABILITY_AI_API_KEY` (integración completa)
- ✅ **Placeholder**: Fallback automático si no hay API keys

#### Text-to-Speech
- ✅ **OpenAI TTS**: Configurar `OPENAI_API_KEY` (alta calidad, integración completa)
- ✅ **Google TTS (gTTS)**: Gratis, sin API key (recomendado, integración completa)
- ✅ **ElevenLabs**: Alta calidad, requiere `ELEVENLABS_API_KEY` (integración completa)
- ✅ **Placeholder**: Fallback automático

**Nota**: El sistema selecciona automáticamente el mejor proveedor disponible según tus API keys configuradas.

### Personalización

Puedes personalizar:
- **Estilos de video**: realistic, animated, abstract, minimalist, dynamic
- **Voces de audio**: male_1, male_2, female_1, female_2, neutral
- **Estilos de subtítulos**: simple, modern, bold, elegant, minimal
- **Resolución y FPS**: Configurables por request

## 📝 API Endpoints

### `POST /api/v1/generate`
Genera un video desde un script.

**Request Body**:
```json
{
  "script": {
    "text": "Tu script aquí...",
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
}
```

### `GET /api/v1/status/{video_id}`
Obtiene el estado de generación de un video.

### `GET /api/v1/videos/{video_id}/download`
Descarga el video generado.

### `POST /api/v1/upload-script`
Sube un archivo de script (texto, markdown, etc.).

### `DELETE /api/v1/videos/{video_id}`
Elimina un video y sus archivos asociados.

## 🔄 Flujo de Generación

1. **Procesamiento de Script**: El script se divide en segmentos con timing
2. **Generación de Imágenes**: Se generan imágenes con IA para cada segmento
3. **Generación de Audio**: Se sintetiza el audio del script completo
4. **Generación de Subtítulos**: Se crean subtítulos sincronizados
5. **Composición**: Se combinan todos los elementos en el video final

## 🛠️ Desarrollo

### Ejecutar tests (cuando estén implementados)
```bash
pytest tests/
```

### Formatear código
```bash
black faceless_video_ai/
isort faceless_video_ai/
```

## 📄 Licencia

Propietaria - Blatam Academy

## 🤝 Contribuciones

Este es un proyecto interno de Blatam Academy.

## 📞 Soporte

Para soporte, contacta al equipo de Blatam Academy.

---

**Versión**: 1.0.0  
**Autor**: Blatam Academy


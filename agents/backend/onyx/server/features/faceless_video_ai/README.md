# Faceless Video AI

> Part of the [Blatam Academy Integrated Platform](../README.md)

Complete system for generating faceless videos entirely with AI from scripts, including image generation, audio, subtitles, and final composition.

## 🎬 Features

- **AI Video Generation** — Creates fully AI-generated videos without people
- **Script Processing** — Automatically processes and segments scripts
- **Image Generation** — Generates AI images for each video segment
- **Text-to-Speech (TTS)** — Converts text to speech with multiple voice options
- **Automatic Subtitles** — Generates and embeds subtitles with multiple styles
- **Video Composition** — Combines images, audio, and subtitles into a final video
- **REST API** — Complete API with FastAPI for integration
- **Async Processing** — Background video generation

## 📋 Requirements

- Python 3.10+
- FFmpeg (for video composition)
- API keys for AI services (optional, depending on services used)

## 🚀 Installation

### Option 1: Local Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Install FFmpeg**:
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg`

3. **Configure environment variables** (optional):
```bash
# Create .env file
OPENAI_API_KEY=your_api_key
STABILITY_AI_API_KEY=your_api_key
ELEVENLABS_API_KEY=your_api_key
```

### Option 2: Docker (Recommended)

```bash
# Build image
docker build -t faceless-video-ai .

# Run with docker-compose (includes Redis, PostgreSQL, etc.)
docker-compose up -d
```

### Option 3: AWS Deployment

For full deployment on AWS (ECS/Fargate, Lambda, etc.), see the [AWS Deployment Guide](aws/deployment/AWS_DEPLOYMENT.md).

**Quick Start AWS:**
```bash
cd aws/deployment
make deploy  # Requires configured AWS CLI
```

## 🎯 Usage

### Start API Server

```bash
python -m faceless_video_ai.api.main
# Or using uvicorn directly
uvicorn faceless_video_ai.api.main:app --host 0.0.0.0 --port 8000
```

### Generate a Video

```python
from faceless_video_ai.core.models import (
    VideoGenerationRequest,
    VideoScript,
    VideoConfig,
    AudioConfig,
    SubtitleConfig,
)

# Create request
request = VideoGenerationRequest(
    script=VideoScript(
        text="Welcome to this amazing video about artificial intelligence. "
             "Today we will explore the latest trends in AI and how it transforms the world.",
        language="en"
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

# Send request to API
import requests
response = requests.post("http://localhost:8000/api/v1/generate", json=request.dict())
video_job = response.json()
```

### Check Status

```python
video_id = video_job["video_id"]
status_response = requests.get(f"http://localhost:8000/api/v1/status/{video_id}")
print(status_response.json())
```

### Download Video

```python
video_response = requests.get(f"http://localhost:8000/api/v1/videos/{video_id}/download")
with open("output_video.mp4", "wb") as f:
    f.write(video_response.content)
```

## 📁 Project Structure

```
faceless_video_ai/
├── __init__.py
├── core/
│   ├── __init__.py
│   └── models.py          # Pydantic data models
├── services/
│   ├── __init__.py
│   ├── script_processor.py      # Script processing
│   ├── video_generator.py       # Image/video generation
│   ├── audio_generator.py       # Audio generation (TTS)
│   ├── subtitle_generator.py    # Subtitle generation
│   ├── video_compositor.py      # Final video composition
│   ├── video_orchestrator.py    # Main orchestrator
│   ├── storage/                 # Storage (S3, local)
│   ├── metrics/                 # Prometheus metrics
│   └── ...                      # Other services
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   └── routes.py         # API endpoints
├── config/
│   ├── __init__.py
│   └── settings.py       # Configuration (with AWS support)
├── aws/
│   └── deployment/       # AWS deployment configs
│       ├── Dockerfile
│       ├── docker-compose.yml
│       ├── cloudformation-template.yaml
│       ├── ecs-task-definition.json
│       ├── serverless.yml
│       └── AWS_DEPLOYMENT.md
├── Dockerfile            # Production Dockerfile
├── Dockerfile.dev        # Development Dockerfile
├── docker-compose.yml    # Docker Compose with all services
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### AI Services

The system includes **real integration** with multiple AI services:

#### Image Generation
- ✅ **OpenAI DALL-E** — Configure `OPENAI_API_KEY` (full integration)
- ✅ **Stability AI** — Configure `STABILITY_AI_API_KEY` (full integration)
- ✅ **Placeholder** — Automatic fallback if no API keys

#### Text-to-Speech
- ✅ **OpenAI TTS** — Configure `OPENAI_API_KEY` (high quality, full integration)
- ✅ **Google TTS (gTTS)** — Free, no API key (recommended, full integration)
- ✅ **ElevenLabs** — High quality, requires `ELEVENLABS_API_KEY` (full integration)
- ✅ **Placeholder** — Automatic fallback

**Note**: The system automatically selects the best available provider based on your configured API keys.

### Customization

You can customize:
- **Video Styles** — realistic, animated, abstract, minimalist, dynamic
- **Audio Voices** — male_1, male_2, female_1, female_2, neutral
- **Subtitle Styles** — simple, modern, bold, elegant, minimal
- **Resolution and FPS** — Configurable per request

## 📝 API Endpoints

### `POST /api/v1/generate`
Generates a video from a script.

### `GET /api/v1/status/{video_id}`
Gets the generation status of a video.

### `GET /api/v1/videos/{video_id}/download`
Downloads the generated video.

### `POST /api/v1/upload-script`
Uploads a script file (text, markdown, etc.).

### `DELETE /api/v1/videos/{video_id}`
Deletes a video and its associated files.

## 🔄 Generation Flow

1. **Script Processing** — Script is divided into segments with timing
2. **Image Generation** — AI images are generated for each segment
3. **Audio Generation** — Audio is synthesized for the full script
4. **Subtitle Generation** — Synchronized subtitles are created
5. **Composition** — All elements are combined into the final video

## 📄 License

Proprietary — Blatam Academy

---

[← Back to Main README](../README.md)

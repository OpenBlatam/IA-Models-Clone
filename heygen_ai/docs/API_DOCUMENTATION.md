# HeyGen AI Equivalent - API Documentation

## Overview

The HeyGen AI Equivalent is a comprehensive AI-powered video generation system that creates talking avatar videos with natural speech synthesis, similar to HeyGen AI. This system provides a complete pipeline for generating professional videos with AI avatars, voice cloning, and script generation.

## Features

### Core Capabilities
- **AI Avatar Generation**: Create and customize AI avatars with realistic facial expressions
- **Voice Synthesis**: High-quality text-to-speech with voice cloning capabilities
- **Video Rendering**: Professional video composition with background integration
- **Script Generation**: AI-powered script creation and optimization
- **Multi-language Support**: Support for multiple languages and accents
- **Batch Processing**: Generate multiple videos simultaneously
- **RESTful API**: Complete API for integration with external systems

### Technical Features
- **Lip-sync Technology**: Advanced lip-sync generation for natural speech
- **Voice Cloning**: Clone voices from audio samples
- **Custom Avatars**: Create avatars from personal images
- **Video Effects**: Professional video post-processing and enhancement
- **Quality Control**: Multiple quality presets and resolution options

## API Endpoints

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
Currently, the API supports optional API key authentication. Set the `X-API-Key` header with your API key.

### Health Check

#### GET /health
Check system health and component status.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "avatar_manager": true,
    "voice_engine": true,
    "video_renderer": true,
    "script_generator": true
  },
  "version": "1.0.0",
  "uptime": 3600.5
}
```

### Video Generation

#### POST /videos/create
Create a new AI-generated video.

**Request Body:**
```json
{
  "script": "Hello! Welcome to our AI-powered video generation system.",
  "avatar_id": "professional_male_01",
  "voice_id": "en_us_01",
  "language": "en",
  "style": "professional",
  "resolution": "1080p",
  "output_format": "mp4",
  "duration": 30,
  "background": "https://example.com/background.jpg",
  "custom_settings": {
    "lighting": "studio",
    "camera_angle": "front"
  }
}
```

**Response:**
```json
{
  "video_id": "video_123456789",
  "status": "completed",
  "output_url": "https://storage.example.com/videos/video_123456789.mp4",
  "duration": 28.5,
  "file_size": 15728640,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:32:00Z",
  "metadata": {
    "avatar_id": "professional_male_01",
    "voice_id": "en_us_01",
    "language": "en",
    "resolution": "1080p"
  }
}
```

#### POST /videos/batch
Create multiple videos in batch.

**Request Body:**
```json
{
  "videos": [
    {
      "script": "First video content",
      "avatar_id": "professional_male_01",
      "voice_id": "en_us_01"
    },
    {
      "script": "Second video content",
      "avatar_id": "professional_female_01",
      "voice_id": "en_us_02"
    }
  ]
}
```

**Response:**
```json
{
  "batch_id": "batch_123456789",
  "videos": [...],
  "total_count": 2,
  "completed_count": 2,
  "failed_count": 0
}
```

#### GET /videos/{video_id}
Get video information by ID.

### Script Generation

#### POST /scripts/generate
Generate an AI script from a topic.

**Request Body:**
```json
{
  "topic": "The Future of Artificial Intelligence",
  "language": "en",
  "style": "educational",
  "duration": "3 minutes",
  "additional_context": "Target audience: university students"
}
```

**Response:**
```json
{
  "script_id": "script_123456789",
  "script": "Artificial Intelligence is transforming our world...",
  "word_count": 450,
  "estimated_duration": 3.0,
  "language": "en",
  "style": "educational",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### POST /scripts/analyze
Analyze a script for various metrics.

**Response:**
```json
{
  "script_id": "analysis_123456789",
  "word_count": 450,
  "estimated_duration": 3.0,
  "readability_score": 75.5,
  "sentiment": {
    "overall": "positive",
    "confidence": 0.85,
    "emotions": ["enthusiastic", "informative"]
  },
  "complexity": {
    "vocabulary_level": "intermediate",
    "sentence_complexity": "medium",
    "overall_complexity": "moderate"
  },
  "suggestions": [
    "Consider adding more engaging opening",
    "Include specific examples"
  ]
}
```

#### POST /scripts/translate
Translate a script to another language.

**Request Body:**
```json
{
  "script": "Hello, welcome to our presentation.",
  "target_language": "es",
  "source_language": "en",
  "preserve_style": true
}
```

**Response:**
```json
{
  "translation_id": "translation_123456789",
  "original_script": "Hello, welcome to our presentation.",
  "translated_script": "Hola, bienvenidos a nuestra presentación.",
  "source_language": "en",
  "target_language": "es",
  "word_count": 6,
  "confidence_score": 0.95
}
```

### Voice Management

#### GET /voices
Get list of available voices.

**Response:**
```json
[
  {
    "voice_id": "en_us_01",
    "name": "American Male",
    "language": "en",
    "accent": "us",
    "gender": "male",
    "style": "professional",
    "sample_rate": 22050,
    "is_cloned": false,
    "characteristics": {
      "pitch": "medium",
      "speed": "normal",
      "clarity": "high"
    }
  }
]
```

#### POST /voices/clone
Clone a voice from audio samples.

**Request Body:**
```json
{
  "voice_name": "My Custom Voice",
  "audio_samples": [
    "https://example.com/audio/sample1.wav",
    "https://example.com/audio/sample2.wav",
    "https://example.com/audio/sample3.wav"
  ],
  "description": "Professional voice for business presentations"
}
```

### Avatar Management

#### GET /avatars
Get list of available avatars.

**Response:**
```json
[
  {
    "avatar_id": "professional_male_01",
    "name": "Professional Male",
    "gender": "male",
    "style": "professional",
    "age_range": "25-35",
    "ethnicity": "caucasian",
    "image_url": "https://example.com/avatars/professional_male_01.jpg",
    "is_custom": false,
    "model_config": {
      "face_structure": "oval",
      "hair_style": "short_business",
      "clothing": "business_suit"
    }
  }
]
```

#### POST /avatars/create
Create a custom avatar from an image.

**Request Body:**
```json
{
  "name": "My Custom Avatar",
  "image_url": "https://example.com/my_photo.jpg",
  "style": "professional",
  "description": "Avatar based on my personal photo"
}
```

## Data Models

### Enums

#### VideoStatus
- `pending`: Video is queued for processing
- `processing`: Video is currently being generated
- `completed`: Video generation completed successfully
- `failed`: Video generation failed

#### LanguageCode
- `en`: English
- `es`: Spanish
- `fr`: French
- `de`: German
- `it`: Italian
- `pt`: Portuguese
- `zh`: Chinese
- `ja`: Japanese
- `ko`: Korean

#### VideoStyle
- `professional`: Professional business style
- `casual`: Casual conversational style
- `educational`: Educational content style
- `marketing`: Marketing and promotional style
- `entertainment`: Entertainment content style

#### Resolution
- `720p`: HD resolution (1280x720)
- `1080p`: Full HD resolution (1920x1080)
- `4k`: Ultra HD resolution (3840x2160)

#### OutputFormat
- `mp4`: MP4 video format
- `mov`: MOV video format
- `avi`: AVI video format
- `webm`: WebM video format

## Error Handling

The API uses standard HTTP status codes and returns detailed error information.

### Error Response Format
```json
{
  "error": "Error message description",
  "error_code": "ERROR_CODE",
  "details": {
    "field": "Additional error details"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes
- `400`: Bad Request - Invalid request parameters
- `404`: Not Found - Resource not found
- `422`: Validation Error - Request validation failed
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Server error

## Rate Limiting

The API implements rate limiting to ensure fair usage:
- **Default**: 100 requests per hour per IP
- **Configurable**: Can be adjusted via environment variables
- **Headers**: Rate limit information is included in response headers

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HEYGEN_API_HOST` | API host address | `0.0.0.0` |
| `HEYGEN_API_PORT` | API port number | `8000` |
| `HEYGEN_DEBUG` | Enable debug mode | `false` |
| `HEYGEN_GPU_ENABLED` | Enable GPU acceleration | `true` |
| `HEYGEN_STORAGE_TYPE` | Storage backend type | `local` |
| `HEYGEN_DATABASE_URL` | Database connection URL | `sqlite:///heygen_ai.db` |
| `OPENAI_API_KEY` | OpenAI API key for script generation | - |
| `HUGGINGFACE_TOKEN` | Hugging Face token for models | - |

## Usage Examples

### Python Client Example
```python
import requests
import json

# Create video
response = requests.post(
    "http://localhost:8000/api/v1/videos/create",
    json={
        "script": "Hello, this is an AI-generated video!",
        "avatar_id": "professional_male_01",
        "voice_id": "en_us_01",
        "language": "en"
    }
)

video_data = response.json()
print(f"Video created: {video_data['video_id']}")
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/api/v1/videos/create" \
  -H "Content-Type: application/json" \
  -d '{
    "script": "Hello, this is an AI-generated video!",
    "avatar_id": "professional_male_01",
    "voice_id": "en_us_01",
    "language": "en"
  }'
```

## Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "heygen_ai.main"]
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python -m heygen_ai.main
```

## Monitoring and Logging

The system includes comprehensive monitoring and logging:
- **Health Checks**: Component health monitoring
- **Metrics**: Performance and usage metrics
- **Logging**: Structured logging with configurable levels
- **Error Tracking**: Integration with Sentry for error tracking

## Security Considerations

- **API Key Authentication**: Optional API key-based authentication
- **Rate Limiting**: Prevents abuse and ensures fair usage
- **Input Validation**: Comprehensive request validation
- **CORS Configuration**: Configurable CORS settings
- **Trusted Hosts**: Host validation for production deployments

## Performance Optimization

- **GPU Acceleration**: CUDA support for faster processing
- **Batch Processing**: Efficient batch video generation
- **Caching**: Model and result caching
- **Async Processing**: Non-blocking video generation
- **Resource Management**: Efficient memory and CPU usage

## Support and Documentation

For additional support and documentation:
- **API Documentation**: Available at `/docs` when debug mode is enabled
- **Examples**: See the `examples/` directory for usage examples
- **Configuration**: Detailed configuration options in `config/settings.py`
- **Testing**: Comprehensive test suite in the `tests/` directory 
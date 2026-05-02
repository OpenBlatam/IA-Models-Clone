# Suno Clone AI — AI Music Generation System

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 🚀 Overview

Suno Clone AI is a complete AI music generation system that allows users to create songs via chat, similar to Suno AI. Users can describe what they want in natural language, and the system generates the corresponding song.

## ✨ Key Features

### 🎵 Music Generation
- **Chat Generation** — Users can write requests in natural language
- **Multiple Models** — Support for MusicGen (small, medium, large)
- **Intelligent Processing** — Automatic extraction of genre, mood, tempo, and instruments
- **AI Enhancement** — Uses OpenAI to improve generation prompts
- **Smart Cache** — Caching system to avoid unnecessary regenerations
- **Automatic Processing** — Automatic normalization and fade in/out
- **🚀 Ultra-Fast** — Advanced optimizations with `torch.compile`, mixed precision, and caching (up to 5–10× faster)

### 💬 Intelligent Chat
- **NLP** — Interprets natural language requests
- **Conversation History** — Maintains context of previous conversations
- **Automatic Extraction** — Automatically identifies:
  - Musical genre
  - Mood
  - Tempo/BPM
  - Instruments
  - Duration

### 🎛️ Advanced Control
- **Customizable Parameters** — Control over duration, temperature, top-k, top-p
- **Multiple Formats** — WAV generation with configurable quality
- **Background Processing** — Asynchronous song generation

### 📊 Song Management
- **Storage** — SQLite database for metadata
- **History** — Saves all generated songs
- **Download** — Endpoint to download audio files
- **Search** — Filtering by user, date, etc.
- **Advanced Editing** — Reverb, EQ, tempo/pitch change
- **Mixing** — Combine multiple songs
- **Analysis** — Detailed audio feature analysis

### 📈 Analytics & Metrics
- **General Stats** — Full system tracking
- **User Metrics** — Individual statistics
- **Performance Tracking** — Generation times and usage
- **Popular Prompts** — Frequent usage analysis

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU)
- FFmpeg (for audio processing)

### Dependency Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8020
DEBUG=False

# OpenAI (optional, to enhance prompts)
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4

# Music Generation
MUSIC_MODEL=facebook/musicgen-medium
USE_GPU=True
MAX_AUDIO_LENGTH=300
DEFAULT_DURATION=30
SAMPLE_RATE=32000

# Storage
AUDIO_STORAGE_PATH=./storage/audio
DATABASE_URL=sqlite:///./suno_clone.db

# Security
SECRET_KEY=your-secret-key-change-in-production
```

## 🎯 Quick Start

### Start the Server

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8020
```

### Usage Examples

#### 1. Create Song via Chat

```bash
curl -X POST "http://localhost:8020/suno/chat/create-song" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want an energetic rock song with guitar and drums, 2 minutes long",
    "user_id": "user123"
  }'
```

#### 2. Generate Song Directly

```bash
curl -X POST "http://localhost:8020/suno/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Upbeat electronic music with synthesizers",
    "duration": 30,
    "genre": "electronic",
    "mood": "energetic"
  }'
```

#### 3. List Songs

```bash
curl "http://localhost:8020/suno/songs?user_id=user123"
```

#### 4. Download Song

```bash
curl "http://localhost:8020/suno/songs/{song_id}/download" --output song.wav
```

## 📚 API Endpoints

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/suno/chat/create-song` | Create song from chat message |
| `GET` | `/suno/chat/history/{user_id}` | Get chat history |

### Songs
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/suno/songs` | List all songs |
| `GET` | `/suno/songs/{song_id}` | Get song info |
| `GET` | `/suno/songs/{song_id}/download` | Download audio file |
| `DELETE` | `/suno/songs/{song_id}` | Delete a song |
| `POST` | `/suno/songs/{song_id}/edit` | Edit song with effects |
| `POST` | `/suno/songs/mix` | Mix multiple songs |
| `GET` | `/suno/songs/{song_id}/analyze` | Analyze audio features |

### Generation
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/suno/generate` | Generate song from prompt |
| `GET` | `/suno/generate/status/{task_id}` | Get generation status |

### Models
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/suno/models` | List available models |
| `GET` | `/suno/models/{model_id}` | Model information |

## 🏗️ Architecture

```
suno_clone_ai/
├── api/                 # API Endpoints
│   └── song_api.py     # Main endpoints
├── core/               # Business logic
│   ├── music_generator.py    # Music generator
│   ├── chat_processor.py     # Chat processor
│   ├── cache_manager.py      # Cache manager
│   ├── audio_processor.py    # Advanced audio processor
│   └── error_handler.py      # Centralized error handling
├── services/            # Services
│   ├── song_service.py       # Song management
│   └── metrics_service.py    # Metrics service
├── config/             # Configuration
│   └── settings.py           # Settings
├── middleware/         # Middleware
│   ├── logging_middleware.py  # Logging
│   └── rate_limiter.py        # Rate limiting
├── utils/             # Utilities
│   └── validators.py          # Reusable validators
├── main.py            # Main server
├── requirements.txt   # Dependencies
├── README.md          # Main documentation
├── QUICK_START.md     # Quick start guide
└── ADVANCED_FEATURES.md # Advanced features
```

## 🔧 Advanced Configuration

### Available Models

- **facebook/musicgen-small**: Small and fast model (~300MB)
- **facebook/musicgen-medium**: Balanced model, default (~1.5GB)
- **facebook/musicgen-large**: Large model with better quality (~3GB)

### Generation Parameters

- `temperature`: Controls creativity (default: 1.0)
- `top_k`: Number of tokens to consider (default: 250)
- `top_p`: Nucleus sampling (default: 0.0)
- `cfg_coef`: Guidance scale (default: 3.0)

## 🚀 Deployment

### Production
For production, use an ASGI server like Gunicorn:

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8020
```

## 📊 Performance

- **Standard Generation**: ~10–30 seconds depending on model and duration
- **Ultra-Fast Generation**: ~2–6 seconds with all optimizations enabled (5–10× faster)
- **GPU**: Significantly accelerates generation (required for max performance)
- **Cache**: Optional Redis for results caching + integrated memory/disk cache
- **Optimizations**: `torch.compile`, mixed precision (FP16), batch processing, async inference

## 🔒 Security

- **Rate Limiting** — Protection against abuse with configurable limits
- **Exhaustive Validation** — Validation of all inputs with Pydantic
- **Sanitization** — Automatic cleaning of dangerous data
- **Error Handling** — Safe error handling without exposing sensitive info
- **Health Checks** — Complete system status monitoring

## 🧪 Testing

```bash
pytest tests/
```

## 📝 License

See LICENSE file.

## 🤝 Contribution

Contributions are welcome! Please see CONTRIBUTING.md.

---

[← Back to Main README](../README.md)

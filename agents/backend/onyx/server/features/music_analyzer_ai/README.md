# Music Analyzer AI — Music Analysis & Coaching System

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 🚀 Quick Start — Single Command

**Start the entire system with Docker:**

```bash
# Windows
deployment\start.bat

# Linux/Mac
./deployment/start.sh

# Python (Cross-platform)
python deployment/start.py

# Make (Linux/Mac)
make -C deployment start
```

**Stop everything:**

```bash
# Windows
deployment\stop.bat

# Linux/Mac
./deployment/stop.sh

# Make
make -C deployment stop
```

See `START.md` or `deployment/README_START.md` for more details.

---

## 🎵 Description

Advanced AI music analysis system that connects to Spotify and provides detailed song analysis, including notes, key, tempo, structure, and personalized music coaching.

## ✨ Key Features

### Music Analysis
- **Key Analysis** — Identifies root note, mode (major/minor), and scale
- **Tempo Analysis** — Detects BPM and categorizes tempo
- **Structure Analysis** — Identifies sections (intro, verse, chorus, bridge, outro)
- **Harmonic Analysis** — Analyzes chord progressions and key changes
- **Technical Analysis** — Energy, danceability, valence, acousticness, etc.

### Music Coaching
- **Learning Paths** — Step-by-step guides to learn songs
- **Practice Exercises** — Personalized exercises based on the song
- **Educational Insights** — Explanations about notes, scales, and chords
- **Recommendations** — Suggestions tailored to skill level
- **Performance Tips** — Advice to improve execution

### Spotify Integration
- Song search
- Audio feature retrieval
- Detailed audio analysis
- Complete track information
- **Similar Song Recommendations** (NEW)
- **Comparative Analysis between multiple songs** (NEW)

### Performance Improvements
- **Cache System** — Intelligent caching to improve performance
- **Error Handling** — Custom exceptions and robust handling
- **Validations** — Input validation and query sanitization
- **Rate Limiting** — API abuse protection

## 📁 Project Structure

```
music_analyzer_ai/
├── api/                    # API Endpoints
│   ├── __init__.py
│   └── music_api.py
├── core/                   # Core logic
│   ├── __init__.py
│   └── music_analyzer.py
├── services/               # Services
│   ├── __init__.py
│   ├── spotify_service.py
│   └── music_coach.py
├── models/                 # Data models
│   ├── __init__.py
│   └── schemas.py
├── utils/                  # Utilities
├── config/                 # Configuration
│   ├── __init__.py
│   └── settings.py
├── tests/                  # Tests
├── main.py                 # Main server
├── requirements.txt
└── README.md
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- pip
- Spotify Developer Account (to obtain Client ID and Client Secret)

### Spotify Configuration

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new application
3. Get your **Client ID** and **Client Secret**
4. Configure credentials in environment variables

### Installation

```bash
cd music_analyzer_ai
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Spotify API
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://localhost:8010/callback

# Server
HOST=0.0.0.0
PORT=8010

# Logging
LOG_LEVEL=INFO
```

## 🚀 Usage

### Start Server

```bash
python main.py
```

The server will be available at `http://localhost:8010`

### API Documentation

Once the server is started, access:
- **Swagger UI**: `http://localhost:8010/docs`
- **ReDoc**: `http://localhost:8010/redoc`

## 📖 Main Endpoints

### 1. Search Songs

```bash
POST /music/search
Content-Type: application/json

{
  "query": "Bohemian Rhapsody Queen",
  "limit": 5
}
```

**Response:**
```json
{
  "success": true,
  "query": "Bohemian Rhapsody Queen",
  "results": [
    {
      "id": "4uLU6hMCjMI75M1A2tKUQC",
      "name": "Bohemian Rhapsody",
      "artists": ["Queen"],
      "album": "A Night At The Opera",
      "duration_ms": 355000,
      "preview_url": "...",
      "popularity": 85
    }
  ],
  "total": 1
}
```

### 2. Analyze Song

```bash
POST /music/analyze
Content-Type: application/json

{
  "track_id": "4uLU6hMCjMI75M1A2tKUQC",
  "include_coaching": true
}
```

## 🔗 Integration

This module integrates with:
- **[Integration System](../integration_system/README.md)** — Service orchestration
- **[Voice Coaching AI](../voice_coaching_ai/README.md)** — Voice analysis integration
- **[Export IA](../export_ia/README.md)** — Analysis report export

---

[← Back to Main README](../README.md)

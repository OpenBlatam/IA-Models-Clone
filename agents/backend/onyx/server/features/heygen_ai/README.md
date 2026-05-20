# HeyGen AI

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 📋 Description

HeyGen AI Clone - Advanced AI Video Generation System. This system allows generating realistic avatars, synthesizing voice, and rendering high-quality videos using deep learning models.

## 🚀 Key Features

- **Avatar Generation**: Create realistic avatars from text descriptions or images.
- **Voice Synthesis**: High-quality text-to-speech with multiple voices and languages.
- **Video Rendering**: Composition and rendering of videos with synchronized lip-sync.
- **Script Generation**: AI-assisted script writing.
- **Face Processing**: Advanced face detection and enhancement.

## 🏗️ Architecture

The system follows clean architecture principles and best practices for deep learning projects.

```
heygen_ai/
├── shared/                    # ✅ Shared types, enums, and configurations
├── core/                      # Core business logic
│   ├── diffusion/             # Diffusion model management
│   ├── face_processing/       # Face detection and enhancement
│   ├── image_processing/      # Image processing utilities
│   ├── avatar_manager.py      # Avatar generation orchestrator
│   ├── voice_engine.py        # Voice synthesis engine
│   ├── video_renderer.py      # Video rendering engine
│   └── script_generator.py    # Script generation engine
├── domain/                    # Domain layer (entities, value objects)
├── application/               # Application layer (use cases)
├── infrastructure/            # Infrastructure layer
├── presentation/              # Presentation layer (API)
├── models/                    # PyTorch model architectures
├── data/                      # Data processing
├── training/                  # Training utilities
├── utils/                     # Shared utilities
└── config/                    # Configuration files
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

```python
from heygen_ai.core.avatar_manager import AvatarManager
from heygen_ai.shared import AvatarGenerationConfig, AvatarStyle

# Initialize manager
manager = AvatarManager()

# Configure generation
config = AvatarGenerationConfig(style=AvatarStyle.REALISTIC)

# Generate avatar
avatar_path = await manager.generate_avatar("professional headshot", config)
```

## 🔗 Integration

This module integrates with:
- **[Integration System](../integration_system/README.md)** — For orchestration
- **[Export IA](../export_ia/README.md)** — For results export

---

[← Back to Main README](../README.md)

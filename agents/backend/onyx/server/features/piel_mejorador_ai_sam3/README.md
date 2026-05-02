# Skin Enhancer AI SAM3

> Part of the [Blatam Academy Integrated Platform](../README.md)

Skin enhancement system with SAM3 architecture, integrated with OpenRouter and TruthGPT for image and video processing with configurable levels of enhancement and realism.

## Features

### Core
- ✅ SAM3 architecture for parallel and continuous processing
- ✅ OpenRouter integration for high-quality LLMs with vision support
- ✅ TruthGPT integration for advanced optimization
- ✅ 24/7 continuous operation
- ✅ Parallel task execution
- ✅ Automatic task management with priority queue
- ✅ Image and video processing
- ✅ Configurable enhancement levels (low, medium, high, ultra)
- ✅ Configurable realism levels (0.0 to 1.0)
- ✅ Skin condition analysis

### Advanced
- ✅ Frame-by-frame processing for videos
- ✅ Intelligent cache system
- ✅ Batch processing
- ✅ Advanced structured logging

### Enterprise
- ✅ Rate limiting with token bucket
- ✅ Webhooks system for notifications
- ✅ Automatic memory optimization
- ✅ Advanced metrics and monitoring
- ✅ Health checks and recommendations

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Configure environment variables:

```bash
export OPENROUTER_API_KEY="your-api-key"
export TRUTHGPT_ENDPOINT="optional-endpoint"  # Optional
```

## Basic Usage

### Agent Usage

```python
import asyncio
from piel_mejorador_ai_sam3 import PielMejoradorAgent, PielMejoradorConfig

async def main():
    # Create configuration
    config = PielMejoradorConfig()
    
    # Create agent
    agent = PielMejoradorAgent(config=config)
    
    # Enhance an image
    task_id = await agent.mejorar_imagen(
        file_path="path/to/image.jpg",
        enhancement_level="medium",
        realism_level=0.8
    )
    
    # Wait for result
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

### REST API Usage

Start the server:

```bash
uvicorn piel_mejorador_ai_sam3.api.piel_mejorador_api:app --reload
```

#### Upload and enhance image

```bash
curl -X POST "http://localhost:8000/upload-image" \
  -F "file=@image.jpg" \
  -F "enhancement_level=medium" \
  -F "realism_level=0.8"
```

#### Enhance image from path

```bash
curl -X POST "http://localhost:8000/mejorar-imagen" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/image.jpg",
    "enhancement_level": "high",
    "realism_level": 0.9
  }'
```

#### Enhance video

```bash
curl -X POST "http://localhost:8000/mejorar-video" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/video.mp4",
    "enhancement_level": "medium",
    "realism_level": 0.7
  }'
```

#### Analyze skin

```bash
curl -X POST "http://localhost:8000/analizar-piel" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/image.jpg",
    "file_type": "image"
  }'
```

#### Check task status

```bash
curl "http://localhost:8000/task/{task_id}/status"
```

#### Get result

```bash
curl "http://localhost:8000/task/{task_id}/result"
```

## Architecture

### Directory Structure

```
piel_mejorador_ai_sam3/
├── core/
│   ├── piel_mejorador_agent.py    # Main agent (orchestrator)
│   ├── task_manager.py            # Task management and priority queue
│   ├── service_handler.py         # Enhancement service handling
│   ├── prompt_builder.py          # Prompt construction
│   ├── system_prompts_builder.py  # Specialized system prompts
│   └── helpers.py                 # Common utilities
├── infrastructure/
│   ├── openrouter_client.py       # OpenRouter client (LLM + Vision)
│   ├── truthgpt_client.py         # TruthGPT client (optimization)
│   └── retry_helpers.py           # Retry helpers with backoff
├── config/
│   └── piel_mejorador_config.py  # Centralized configuration
├── api/
│   └── piel_mejorador_api.py      # REST API (FastAPI)
├── utils/
│   └── (additional utilities)
├── tests/
│   └── (tests)
├── examples/
│   └── (usage examples)
└── docs/
    └── (documentation)
```

## Available Services

### 1. Enhance Image

Enhances skin in a static image.

```python
task_id = await agent.mejorar_imagen(
    file_path="image.jpg",
    enhancement_level="medium",  # low, medium, high, ultra
    realism_level=0.8,  # 0.0 to 1.0 (optional)
    custom_instructions="Focus on smoothing texture",
    priority=0
)
```

### 2. Enhance Video

Enhances skin in a video maintaining consistency between frames.

```python
task_id = await agent.mejorar_video(
    file_path="video.mp4",
    enhancement_level="high",
    realism_level=0.9,
    custom_instructions="Keep movement natural",
    priority=0
)
```

### 3. Analyze Skin

Analyzes skin condition and provides recommendations.

```python
task_id = await agent.analizar_piel(
    file_path="image.jpg",
    file_type="image",  # or "video"
    priority=0
)
```

## Enhancement Levels

- **low**: Subtle and natural enhancements (intensity: 0.3, realism: 0.5)
- **medium**: Moderate enhancements maintaining realism (intensity: 0.6, realism: 0.7)
- **high**: Significant enhancements with high realism (intensity: 0.9, realism: 0.9)
- **ultra**: Maximum enhancements with perfect photographic realism (intensity: 1.0, realism: 1.0)

## Realism Levels

Realism level can be specified as a float between 0.0 and 1.0:
- **0.0**: Natural, preserves original features
- **0.5**: Balanced between natural and enhanced
- **1.0**: Perfect photographic realism

## Advanced Features

### Frame-by-Frame Processing for Videos

The system can process videos frame by frame for more precise enhancements:

```python
from piel_mejorador_ai_sam3.core.video_processor import VideoProcessor

processor = VideoProcessor()
frames = await processor.extract_frames("video.mp4")
# Process frames...
enhanced_video = await processor.reconstruct_video(frames, "output.mp4")
```

### Intelligent Cache System

Avoids reprocessing already processed files:

```python
# Cache is used automatically
# View statistics:
stats = agent.cache_manager.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

### Batch Processing

Processes multiple files simultaneously:

```python
from piel_mejorador_ai_sam3.core.batch_processor import BatchItem

items = [
    BatchItem(file_path="img1.jpg", enhancement_level="high"),
    BatchItem(file_path="img2.jpg", enhancement_level="medium"),
]

result = await agent.process_batch(items)
print(f"Success rate: {result.success_rate:.2%}")
```

See **[ADVANCED_FEATURES.md](ADVANCED_FEATURES.md)** for more details.

### Enterprise Features

#### Rate Limiting
Automatic protection against API abuse with configurable limits per client.

#### Webhooks
Complete asynchronous notification system for task events.

#### Memory Optimization
Automatic monitoring and optimization of memory for large files.

See **[ENTERPRISE_FEATURES.md](ENTERPRISE_FEATURES.md)** for more details.

### 24/7 Continuous Mode

The agent can run in continuous mode processing tasks automatically:

```python
agent = PielMejoradorAgent(config=config)
await agent.start()  # Runs indefinitely
```

### Task Prioritization

Tasks can have different priorities:

```python
# High priority
task_id = await agent.mejorar_imagen(
    file_path="image.jpg",
    enhancement_level="high",
    priority=10  # Higher priority
)
```

### TruthGPT Integration

The agent automatially optimizes queries using TruthGPT when available.

## Supported Formats

### Images
- JPG/JPEG
- PNG
- WebP

Maximum size: 50MB (configurable)

### Videos
- MP4
- MOV
- AVI
- WebM

Maximum size: 500MB (configurable)

## Requirements

- Python 3.8+
- OpenRouter API key
- TruthGPT (optional but recommended)

## License

MIT

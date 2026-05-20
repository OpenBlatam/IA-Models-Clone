# Quick Start Guide - Imagen Video Enhancer AI

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Environment Variables

```bash
# OpenRouter API Key
export OPENROUTER_API_KEY="your-api-key"

# TruthGPT (optional)
export TRUTHGPT_API_KEY="your-truthgpt-key"
export TRUTHGPT_ENABLED=true
```

### Configuration File

Create `config.json`:

```json
{
  "openrouter": {
    "api_key": "your-api-key",
    "base_url": "https://openrouter.ai/api/v1",
    "timeout": 30.0
  },
  "truthgpt": {
    "enabled": false,
    "api_key": "your-key"
  },
  "max_file_size_mb": 100,
  "max_parallel_tasks": 5
}
```

## Basic Usage

### Python API

```python
from imagen_video_enhancer_ai import EnhancerAgent, EnhancerConfig

# Initialize agent
config = EnhancerConfig()
agent = EnhancerAgent(config=config)

# Enhance an image
task_id = await agent.enhance_image(
    file_path="image.jpg",
    enhancement_type="general"
)

# Check status
status = await agent.get_task_status(task_id)
print(status)

# Get result
result = await agent.get_task_result(task_id)
print(result["enhancement_guide"])

# Close agent
await agent.close()
```

### REST API

```bash
# Start API server
python -m imagen_video_enhancer_ai.run_api

# Upload and enhance image
curl -X POST "http://localhost:8000/upload-image" \
  -F "file=@image.jpg" \
  -F "enhancement_type=general"

# Check task status
curl "http://localhost:8000/task/{task_id}/status"

# Get result
curl "http://localhost:8000/task/{task_id}/result"
```

## Common Tasks

### Enhance Image

```python
task_id = await agent.enhance_image(
    file_path="image.jpg",
    enhancement_type="sharpness",
    options={"intensity": "high"},
    priority=5
)
```

### Enhance Video

```python
task_id = await agent.enhance_video(
    file_path="video.mp4",
    enhancement_type="general",
    priority=3
)
```

### Batch Processing

```python
from imagen_video_enhancer_ai.core.batch_processor import BatchItem

items = [
    BatchItem(
        file_path="image1.jpg",
        service_type="enhance_image",
        enhancement_type="general"
    ),
    BatchItem(
        file_path="image2.jpg",
        service_type="upscale",
        options={"scale_factor": 2}
    )
]

result = await agent.process_batch(items)
print(f"Completed: {result.completed}/{result.total_items}")
```

### Export Results

```python
# Export to JSON
exported = await agent.export_results(
    format="json",
    output_path="results.json"
)

# Export to HTML report
exported = await agent.export_results(
    format="html",
    output_path="report.html"
)
```

## API Endpoints

### Upload Files
- `POST /upload-image` - Upload and enhance image
- `POST /upload-video` - Upload and enhance video

### Enhancement Services
- `POST /enhance-image` - Enhance existing image
- `POST /enhance-video` - Enhance existing video
- `POST /upscale` - Upscale image/video
- `POST /denoise` - Reduce noise
- `POST /restore` - Restore image
- `POST /color-correction` - Color correction

### Task Management
- `GET /task/{task_id}/status` - Get task status
- `GET /task/{task_id}/result` - Get task result

### Batch Processing
- `POST /batch-process` - Process multiple files

### Monitoring
- `GET /dashboard/metrics` - Dashboard metrics
- `GET /dashboard/health` - System health
- `GET /stats` - Agent statistics

## Next Steps

1. Read [Best Practices](BEST_PRACTICES.md)
2. Explore [API Documentation](API.md)
3. Check [Plugin System](PLUGINS.md)
4. Review [Features](FEATURES.md)





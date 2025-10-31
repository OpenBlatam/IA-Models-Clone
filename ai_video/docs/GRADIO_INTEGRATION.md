# Gradio Integration for AI Video System

## Overview

The Gradio integration provides a comprehensive web interface for the AI Video system, enabling users to generate, style, optimize, and monitor AI-powered videos through an intuitive web interface.

## Features

### 🎬 Video Generation
- **AI Model Selection**: Choose from Stable Diffusion, Midjourney, DALL-E, or custom models
- **Prompt Engineering**: Advanced text-to-video generation with customizable prompts
- **Duration Control**: Generate videos from 1-30 seconds
- **Resolution Options**: Multiple resolution presets (512x512 to 1920x1080)
- **FPS Control**: Adjustable frame rates (15-60 FPS)
- **Creativity Levels**: Fine-tune generation creativity (0.1-1.0)

### 🎨 Style Transfer
- **Style Presets**: Pre-configured cinematic, vintage, and modern styles
- **Custom Parameters**: Adjust contrast, saturation, brightness, color temperature
- **Film Grain**: Add authentic film grain effects
- **Before/After Comparison**: Visual comparison of original vs styled videos
- **Real-time Preview**: Instant preview of style changes

### ⚡ Performance Optimization
- **GPU Optimization**: Automatic GPU utilization and optimization
- **Mixed Precision**: Enable mixed precision training for faster processing
- **Model Quantization**: Reduce model size and improve inference speed
- **Batch Processing**: Configurable batch sizes for optimal performance
- **Memory Management**: Intelligent memory usage and caching
- **Performance Monitoring**: Real-time performance metrics and charts

### 📊 System Monitoring
- **Resource Usage**: Monitor CPU, GPU, memory, and disk usage
- **Real-time Metrics**: Live system performance tracking
- **Alert System**: Configurable alerts for resource thresholds
- **Performance Charts**: Visual representation of system metrics
- **Queue Management**: Monitor processing queue and active tasks

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- FFmpeg installed on system

### Dependencies
```bash
pip install -r requirements/gradio_requirements.txt
```

### Quick Start
```bash
# Basic launch
python gradio_launcher.py

# Custom configuration
python gradio_launcher.py --port 8080 --share --debug

# Skip system checks
python gradio_launcher.py --skip-checks
```

## Usage

### 1. Video Generation

1. **Select AI Model**: Choose your preferred AI model from the dropdown
2. **Enter Prompt**: Describe the video you want to generate
3. **Configure Settings**:
   - Set duration (1-30 seconds)
   - Choose FPS (15-60)
   - Select resolution
   - Pick style preset
   - Adjust creativity level
4. **Generate**: Click "Generate Video" and wait for processing
5. **Download**: Once complete, download the generated video

### 2. Style Transfer

1. **Upload Video**: Select an input video file
2. **Choose Style**: Pick from available style presets
3. **Adjust Parameters**: Fine-tune style parameters as needed
4. **Apply Style**: Click "Apply Style Transfer"
5. **Compare**: View before/after comparison
6. **Download**: Save the styled video

### 3. Performance Optimization

1. **Enable Features**: Toggle optimization features
2. **Configure Settings**: Set batch size, memory limits, cache size
3. **Apply Optimization**: Click "Apply Optimization"
4. **Monitor Results**: View performance improvements and charts
5. **Save Configuration**: Export optimized settings

### 4. System Monitoring

1. **Enable Monitoring**: Toggle real-time monitoring
2. **Set Alerts**: Configure alert thresholds
3. **Monitor Metrics**: View live system performance
4. **Analyze Charts**: Review performance trends
5. **Handle Alerts**: Respond to system alerts

## Configuration

### Environment Variables
```bash
# Server configuration
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false

# GPU configuration
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=7.5

# Performance configuration
BATCH_SIZE=4
MAX_MEMORY_USAGE=8
CACHE_SIZE=20
```

### Command Line Options
```bash
python gradio_launcher.py [OPTIONS]

Options:
  --host TEXT           Host to bind server to (default: 0.0.0.0)
  --port INTEGER        Port to bind server to (default: 7860)
  --share               Create public link for interface
  --debug               Enable debug mode
  --no-error-display    Disable error display in interface
  --skip-checks         Skip dependency and system checks
```

## Architecture

### Core Components

```
gradio_interface.py          # Main Gradio application
├── GradioAIVideoApp        # Main application class
├── video_generation_interface()    # Video generation UI
├── style_transfer_interface()      # Style transfer UI
├── performance_optimization_interface()  # Optimization UI
└── monitoring_interface()          # Monitoring UI

gradio_launcher.py          # Application launcher
├── GradioLauncher         # Launcher class
├── check_dependencies()    # Dependency verification
├── check_gpu_availability() # GPU detection
├── setup_environment()     # Environment setup
└── run_system_checks()     # System validation
```

### Integration Points

- **Core Video Generator**: Integrates with `VideoGenerator` for AI video generation
- **Style Transfer Engine**: Connects to `StyleTransferEngine` for video styling
- **Performance Optimizer**: Uses `PerformanceOptimizer` for system optimization
- **Error Handler**: Leverages `ErrorHandler` for consistent error management
- **Monitoring System**: Integrates with system monitoring for real-time metrics

## API Integration

### RESTful Endpoints
The Gradio interface can be extended with RESTful API endpoints:

```python
# Example API integration
@app.post("/api/generate-video")
async def generate_video_api(request: VideoRequest):
    return await video_generator.generate(request)

@app.post("/api/apply-style")
async def apply_style_api(request: StyleRequest):
    return await style_engine.apply_style(request)
```

### WebSocket Support
Real-time updates via WebSocket connections:

```python
# WebSocket integration for real-time updates
@app.websocket("/ws/progress")
async def progress_websocket(websocket: WebSocket):
    await websocket.accept()
    while True:
        progress = await get_generation_progress()
        await websocket.send_json(progress)
```

## Performance Optimization

### GPU Utilization
- Automatic GPU detection and utilization
- Mixed precision training for faster processing
- Memory-efficient batch processing
- Dynamic memory allocation

### Caching Strategy
- Model caching for faster inference
- Result caching to avoid regeneration
- Configurable cache size and cleanup
- LRU cache eviction policy

### Async Processing
- Non-blocking video generation
- Background task processing
- Queue management for multiple requests
- Progress tracking and cancellation

## Security Considerations

### Input Validation
- File type validation for uploaded videos
- Size limits and format restrictions
- Prompt sanitization and filtering
- Parameter range validation

### Access Control
- Optional authentication integration
- Rate limiting for API endpoints
- Request throttling and queuing
- Resource usage monitoring

### Data Privacy
- Temporary file cleanup
- Secure file handling
- No persistent storage of user data
- Configurable data retention policies

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check CUDA installation
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size and memory usage
   python gradio_launcher.py --batch-size 2 --max-memory 4
   ```

3. **Port Already in Use**
   ```bash
   # Use different port
   python gradio_launcher.py --port 8080
   ```

4. **Missing Dependencies**
   ```bash
   # Install missing packages
   pip install -r requirements/gradio_requirements.txt
   ```

### Debug Mode
Enable debug mode for detailed logging:
```bash
python gradio_launcher.py --debug
```

### Log Files
Check log files for detailed error information:
- `gradio_app.log`: Application logs
- `error.log`: Error logs
- `performance.log`: Performance metrics

## Development

### Adding New Features

1. **Create Interface Method**:
   ```python
   def new_feature_interface(self):
       with gr.Tab("New Feature"):
           # Add UI components
           pass
   ```

2. **Add to Main App**:
   ```python
   def create_app(self):
       # Add new interface
       self.new_feature_interface()
   ```

3. **Update Documentation**:
   - Add feature description
   - Include usage examples
   - Update configuration options

### Testing
```bash
# Run tests
pytest tests/test_gradio_integration.py

# Run with coverage
pytest --cov=gradio_interface tests/
```

### Contributing
1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include error handling
4. Write unit tests
5. Update documentation

## License

This Gradio integration is part of the AI Video System and follows the same licensing terms.

## Support

For issues and questions:
- Check the troubleshooting section
- Review log files for error details
- Consult the main system documentation
- Open an issue in the project repository 
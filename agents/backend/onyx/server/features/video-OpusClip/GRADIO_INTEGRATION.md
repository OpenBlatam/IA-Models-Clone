# Gradio Integration Guide for Video-OpusClip

## Overview

The Gradio integration provides a comprehensive web interface for the Video-OpusClip AI system, offering easy access to all optimized features through an intuitive UI.

## Features

### 🎥 Video Processing Tab
- **Upload Videos**: Direct file upload support
- **URL Processing**: Process videos from URLs
- **Quality Presets**: Fast, balanced, and quality modes
- **Audio Processing**: Enable/disable audio processing
- **Subtitle Generation**: Automatic subtitle generation
- **Real-time Metrics**: Processing performance metrics

### 🤖 AI Generation Tab
- **Text-to-Video**: Generate videos from text prompts
- **Image-to-Video**: Create videos from static images
- **Advanced Parameters**: Control guidance scale, inference steps, seed
- **Motion Control**: Adjust motion strength for image-to-video

### 📈 Viral Analysis Tab
- **Content Analysis**: Analyze viral potential of content
- **Platform Targeting**: Optimize for specific platforms (TikTok, YouTube, Instagram, Twitter)
- **Engagement Prediction**: Predict engagement metrics
- **Optimization Recommendations**: Get actionable improvement suggestions

### 🏋️ Training Tab
- **Model Training**: Train custom models
- **Real-time Progress**: Monitor training progress
- **Loss Visualization**: View training and validation loss
- **Model Types**: Caption generator, viral predictor, quality assessor

### ⚡ Performance Tab
- **System Metrics**: Real-time CPU, memory, and GPU usage
- **Performance Charts**: Historical performance data
- **Resource Monitoring**: Monitor system resources

### ⚙️ Settings Tab
- **Configuration Management**: Adjust system settings
- **Performance Tuning**: Optimize for your hardware
- **Caching Options**: Configure caching behavior
- **Reset Options**: Restore default settings

## Installation

### Prerequisites
```bash
# Install base requirements
pip install -r requirements_optimized.txt

# Or install Gradio separately
pip install gradio>=4.0.0
```

### Quick Start
```bash
# Launch with default settings
python gradio_launcher.py

# Launch with custom settings
python gradio_launcher.py --host 0.0.0.0 --port 7860 --share --debug

# Use simple interface
python gradio_launcher.py --simple
```

## Usage Examples

### 1. Video Processing
```python
# Upload a video and process it
# 1. Go to "🎥 Video Processing" tab
# 2. Upload your video file
# 3. Set target duration (5-300 seconds)
# 4. Choose quality preset (fast/balanced/quality)
# 5. Enable/disable audio and subtitle processing
# 6. Click "🚀 Process Video"
```

### 2. AI Video Generation
```python
# Generate video from text
# 1. Go to "🤖 AI Generation" tab
# 2. Enter your prompt: "A cat playing with a ball in a sunny garden"
# 3. Set duration (3-30 seconds)
# 4. Adjust guidance scale (1.0-20.0)
# 5. Set inference steps (10-100)
# 6. Click "🎬 Generate Video"

# Generate from image
# 1. Upload an image
# 2. Adjust motion strength (0.1-2.0)
# 3. Click "🎬 Generate from Image"
```

### 3. Viral Analysis
```python
# Analyze content viral potential
# 1. Go to "📈 Viral Analysis" tab
# 2. Enter your content description
# 3. Select content type (video/image/text)
# 4. Choose target platform
# 5. Click "🔍 Analyze Viral Potential"
```

### 4. Model Training
```python
# Train a custom model
# 1. Go to "🏋️ Training" tab
# 2. Upload training data (JSON/CSV/TXT)
# 3. Select model type
# 4. Set training parameters
# 5. Click "🚀 Start Training"
```

## Configuration

### Environment Variables
```bash
# Performance settings
export MAX_WORKERS=8
export BATCH_SIZE=16
export USE_GPU=true
export ENABLE_CACHING=true

# Logging
export LOG_LEVEL=INFO
export ENABLE_STRUCTURED_LOGGING=true

# Memory management
export MAX_MEMORY_MB=8192
export CACHE_SIZE=1000
```

### Configuration File
Create `gradio_config.yaml`:
```yaml
env:
  MAX_WORKERS: 8
  BATCH_SIZE: 16
  USE_GPU: true
  ENABLE_CACHING: true
  LOG_LEVEL: INFO

performance:
  parallel_backend: auto
  max_concurrent_tasks: 100
  enable_mixed_precision: true
  gpu_memory_fraction: 0.8

video:
  max_video_duration: 600.0
  target_fps: 30
  target_resolution: "1080p"
  enable_hardware_acceleration: true
```

## Advanced Features

### Custom CSS Styling
The interface supports custom CSS for branding:
```css
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
```

### API Integration
The Gradio interface integrates with the optimized API:
```python
from gradio_integration import GradioInterface

# Create custom interface
interface = GradioInterface()

# Access underlying components
video_processor = interface.video_processor
api = interface.api
cache = interface.cache
```

### Performance Monitoring
Real-time performance monitoring is built-in:
```python
# Get performance metrics
metrics = interface.performance_monitor.get_metrics()

# Monitor specific operations
with interface.performance_monitor.track_operation("video_processing"):
    result = interface.video_processor.process_video(video_data)
```

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**
   ```bash
   # Reduce batch size
   export BATCH_SIZE=8
   
   # Enable memory optimization
   export ENABLE_MEMORY_OPTIMIZATION=true
   ```

2. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install -r requirements_optimized.txt
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **Performance Issues**
   ```bash
   # Reduce workers
   export MAX_WORKERS=4
   
   # Disable GPU if causing issues
   export USE_GPU=false
   ```

### Debug Mode
```bash
# Enable debug mode for detailed logging
python gradio_launcher.py --debug

# Check logs
tail -f logs/error.log
```

## Performance Optimization

### Hardware Recommendations
- **CPU**: 8+ cores for optimal performance
- **RAM**: 16GB+ for large video processing
- **GPU**: NVIDIA RTX 3080+ for AI generation
- **Storage**: SSD for faster I/O

### Optimization Tips
1. **Use GPU**: Enable GPU acceleration for AI tasks
2. **Adjust Batch Size**: Optimize based on available memory
3. **Enable Caching**: Reduce redundant processing
4. **Monitor Resources**: Use performance tab to identify bottlenecks

## Security Considerations

### Production Deployment
```bash
# Use HTTPS in production
python gradio_launcher.py --host 127.0.0.1 --port 7860

# Behind reverse proxy
# Configure nginx/apache to proxy to Gradio
```

### Access Control
```python
# Implement authentication
import gradio as gr

def auth_fn(username, password):
    return username == "admin" and password == "password"

demo = gr.Interface(
    fn=process_video,
    inputs=[...],
    outputs=[...],
    auth=auth_fn
)
```

## API Reference

### GradioInterface Class
```python
class GradioInterface:
    def __init__(self):
        """Initialize the Gradio interface."""
        
    def _create_interface(self):
        """Create the main interface."""
        
    def _process_video(self, video_input, url_input, ...):
        """Process video with optimized pipeline."""
        
    def _generate_video_from_text(self, prompt, duration, ...):
        """Generate video from text prompt."""
        
    def _analyze_viral_potential(self, content, content_type, platform):
        """Analyze viral potential of content."""
```

### Launch Functions
```python
def create_gradio_interface() -> gr.Blocks:
    """Create and return the Gradio interface."""
    
def launch_gradio(server_name="0.0.0.0", server_port=7860, share=False, debug=False):
    """Launch the Gradio interface."""
```

## Contributing

### Adding New Features
1. Extend the `GradioInterface` class
2. Add new tabs in `_create_interface()`
3. Implement processing functions
4. Update documentation

### Custom Components
```python
# Add custom Gradio components
def custom_video_processor(video, options):
    # Your custom processing logic
    return processed_video

# Register with interface
interface.add_custom_processor(custom_video_processor)
```

## License

This Gradio integration is part of the Video-OpusClip project and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs in `logs/` directory
3. Enable debug mode for detailed error information
4. Consult the main project documentation 
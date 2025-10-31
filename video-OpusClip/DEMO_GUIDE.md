# Interactive Demos Guide for Video-OpusClip

## Overview

The Video-OpusClip interactive demos provide comprehensive visualization and testing capabilities for all AI model inference and analysis features. These demos showcase the system's capabilities through intuitive web interfaces built with Gradio.

## Available Demos

### 🎬 Text-to-Video Generation Demo
**Purpose**: Generate videos from text prompts using advanced AI models

**Features**:
- Multiple model presets (Stable Diffusion, DeepFloyd, Kandinsky)
- Quality presets (fast, balanced, quality)
- Advanced parameter control (guidance scale, inference steps, seed)
- Real-time generation metrics
- Performance visualization

**Usage**:
```bash
python demo_launcher.py --demo text-to-video
```

**Example Prompts**:
- "A majestic dragon flying through a mystical forest at sunset"
- "A futuristic city with flying cars and neon lights"
- "A peaceful underwater scene with colorful coral reefs"

### 🖼️ Image-to-Video Generation Demo
**Purpose**: Transform static images into dynamic videos with motion effects

**Features**:
- Motion strength and direction control
- Style transfer options
- Quality enhancement
- Interpolation frame control
- Motion visualization

**Usage**:
```bash
python demo_launcher.py --demo image-to-video
```

**Motion Types**:
- Horizontal: Left-to-right or right-to-left movement
- Vertical: Up-to-down or down-to-up movement
- Diagonal: Combined horizontal and vertical motion
- Circular: Rotational motion around the center

### 📈 Viral Analysis & Prediction Demo
**Purpose**: Analyze content viral potential across different platforms

**Features**:
- Multi-platform analysis (TikTok, YouTube, Instagram, Twitter)
- Target audience optimization
- Content category classification
- Hashtag analysis
- Sentiment analysis
- Engagement prediction

**Usage**:
```bash
python demo_launcher.py --demo viral-analysis
```

**Platform Optimization**:
- **TikTok**: Short-form, trending content, music integration
- **YouTube**: Longer content, SEO optimization, thumbnail design
- **Instagram**: Visual appeal, story format, hashtag strategy
- **Twitter**: Trending topics, concise messaging, engagement timing

### ⚡ Performance Monitoring Demo
**Purpose**: Real-time system performance visualization and monitoring

**Features**:
- CPU, Memory, and GPU usage tracking
- Real-time performance charts
- System resource analytics
- Performance alerts
- Historical data visualization

**Usage**:
```bash
python demo_launcher.py --demo performance
```

**Metrics Tracked**:
- CPU usage and load
- Memory consumption and availability
- GPU utilization and memory
- Network activity and bandwidth
- Disk usage and I/O

### 🏋️ Training Progress & Metrics Demo
**Purpose**: Interactive training simulation and visualization

**Features**:
- Training loss curves
- Validation metrics
- Accuracy tracking
- Learning rate schedules
- Model performance comparison

**Usage**:
```bash
python demo_launcher.py --demo training
```

**Model Types**:
- Caption Generator: Generate video descriptions
- Viral Predictor: Predict content viral potential
- Quality Assessor: Evaluate video quality
- Style Transfer: Apply artistic styles to videos

### 🎯 All Demos Suite
**Purpose**: Complete demo suite with all features in a unified interface

**Features**:
- All demos accessible through tabs
- Cross-demo data sharing
- Unified configuration
- Comprehensive analytics

**Usage**:
```bash
python demo_launcher.py --demo all
```

## Installation & Setup

### Prerequisites
```bash
# Install base requirements
pip install -r requirements_optimized.txt

# Install demo-specific dependencies
pip install matplotlib seaborn plotly pandas opencv-python pillow
```

### Quick Start
```bash
# Launch all demos
python demo_launcher.py

# Launch specific demo
python demo_launcher.py --demo text-to-video

# Launch with custom settings
python demo_launcher.py --demo all --host 0.0.0.0 --port 7860 --share --debug
```

### Command Line Options
```bash
python demo_launcher.py [OPTIONS]

Options:
  --demo TEXT          Demo to launch (default: all)
  --host TEXT          Host to bind to (default: 127.0.0.1)
  --port INTEGER       Port to bind to (default: 7860)
  --share              Create public link
  --debug              Enable debug mode
  --list               List available demos
  --benchmark          Run performance benchmark
  --help-demo          Show demo help
```

## Demo Features & Capabilities

### Real-time Visualization
All demos include real-time visualization capabilities:
- **Live Charts**: Performance metrics, training progress, viral scores
- **Interactive Plots**: Zoom, pan, hover tooltips
- **Dynamic Updates**: Auto-refreshing data and metrics
- **Multi-format Support**: Matplotlib, Plotly, Seaborn

### Performance Monitoring
Comprehensive system monitoring:
- **Resource Tracking**: CPU, memory, GPU usage
- **Performance Alerts**: Threshold-based notifications
- **Historical Data**: Trend analysis and comparison
- **Optimization Tips**: Performance improvement suggestions

### AI Model Integration
Full integration with optimized AI models:
- **Text-to-Video**: Multiple diffusion models
- **Image Processing**: Advanced computer vision
- **Natural Language**: Text analysis and generation
- **Predictive Analytics**: Viral potential assessment

### Data Analytics
Advanced analytics and insights:
- **Statistical Analysis**: Correlation, trends, patterns
- **Predictive Modeling**: Engagement prediction
- **Comparative Analysis**: Platform performance
- **Optimization Recommendations**: Actionable insights

## Usage Examples

### Example 1: Text-to-Video Generation
```python
# Using the demo interface
1. Navigate to "🎬 Text-to-Video" tab
2. Enter prompt: "A cat playing with a ball in a sunny garden"
3. Set duration: 10 seconds
4. Adjust guidance scale: 7.5
5. Set inference steps: 30
6. Choose model: stable-diffusion
7. Click "🎬 Generate Video"
8. View generated video and metrics
```

### Example 2: Viral Analysis
```python
# Analyze content viral potential
1. Navigate to "📈 Viral Analysis" tab
2. Enter content description
3. Select content type: video
4. Choose target platform: TikTok
5. Set target audience: gen-z
6. Enable hashtag analysis
7. Click "🔍 Analyze Viral Potential"
8. Review viral score and recommendations
```

### Example 3: Performance Monitoring
```python
# Monitor system performance
1. Navigate to "⚡ Performance" tab
2. Set refresh interval: 5 seconds
3. Click "▶️ Start Monitoring"
4. Observe real-time metrics
5. Analyze performance trends
6. Identify bottlenecks
7. Apply optimization recommendations
```

## Advanced Features

### Custom Visualizations
Create custom charts and plots:
```python
from visualization_utils import DemoVisualizer

visualizer = DemoVisualizer()

# Create performance chart
performance_fig = visualizer.create_performance_chart(metrics)

# Create training visualization
loss_fig, acc_fig, lr_fig = visualizer.create_training_visualization(training_data)

# Create viral analysis chart
radar_fig, bar_fig = visualizer.create_viral_analysis_chart(analysis_data)
```

### Data Export
Export demo data for further analysis:
```python
# Export metrics to JSON
import json
with open('demo_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Export charts as images
performance_fig.write_image('performance_chart.png')
```

### API Integration
Use demos programmatically:
```python
from gradio_demos import create_text_to_video_demo

# Create demo interface
demo = create_text_to_video_demo()

# Access demo functions
result = demo.fn("A beautiful sunset", 10, 7.5, 30, -1, "stable-diffusion", "balanced")
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
5. **Update Drivers**: Keep GPU drivers updated

### Benchmark Testing
Run performance benchmarks:
```bash
python demo_launcher.py --benchmark
```

This will test:
- NumPy operations performance
- Matplotlib plotting speed
- Plotly visualization rendering
- Overall system performance

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install -r requirements_optimized.txt
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **GPU Memory Errors**
   ```bash
   # Reduce batch size
   export BATCH_SIZE=8
   
   # Enable memory optimization
   export ENABLE_MEMORY_OPTIMIZATION=true
   ```

3. **Performance Issues**
   ```bash
   # Reduce workers
   export MAX_WORKERS=4
   
   # Disable GPU if causing issues
   export USE_GPU=false
   ```

4. **Visualization Errors**
   ```bash
   # Install visualization dependencies
   pip install matplotlib seaborn plotly
   
   # Set backend
   export MPLBACKEND=Agg
   ```

### Debug Mode
Enable debug mode for detailed error information:
```bash
python demo_launcher.py --debug
```

### Log Files
Check log files for detailed error information:
```bash
tail -f logs/error.log
tail -f logs/demo.log
```

## Customization

### Custom Themes
Modify demo appearance:
```python
# Custom CSS
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    padding: 20px;
}
"""
```

### Custom Visualizations
Add custom charts:
```python
def create_custom_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['x'], y=data['y']))
    fig.update_layout(title="Custom Chart")
    return fig
```

### Custom Metrics
Add custom performance metrics:
```python
def custom_metric_calculator():
    # Your custom metric calculation
    return {
        "custom_metric": value,
        "timestamp": time.time()
    }
```

## Best Practices

### Demo Usage
1. **Start Simple**: Begin with basic demos before advanced features
2. **Monitor Resources**: Use performance monitoring during demos
3. **Save Results**: Export important results and metrics
4. **Iterate**: Use feedback to improve content and settings

### Performance
1. **Optimize Parameters**: Adjust settings based on your hardware
2. **Use Caching**: Enable caching for repeated operations
3. **Monitor Memory**: Watch memory usage during large operations
4. **Batch Processing**: Use batch processing for multiple items

### Development
1. **Version Control**: Track demo configurations and results
2. **Documentation**: Document custom modifications
3. **Testing**: Test demos with different data types
4. **Backup**: Backup important demo results and configurations

## Support & Resources

### Documentation
- Main project documentation: `README_OPTIMIZED.md`
- Optimization guide: `OPTIMIZATION_GUIDE.md`
- Gradio integration: `GRADIO_INTEGRATION.md`

### Examples
- Demo examples: `examples/gradio_example.py`
- Visualization examples: `visualization_utils.py`

### Community
- GitHub Issues: Report bugs and request features
- Discussions: Share tips and best practices
- Contributing: Submit improvements and new demos

## License

The interactive demos are part of the Video-OpusClip project and follow the same licensing terms. 
# Interactive Gradio Demos Guide

## Overview

This guide covers the comprehensive interactive Gradio demos for model inference and visualization. The demos provide real-time, interactive interfaces for exploring AI capabilities, monitoring performance, and visualizing results.

## 🎯 Available Demos

### 1. Main Interactive Demos (`gradio_interactive_demos.py`)
**Port**: 7860
**Description**: Comprehensive demos for text, image, audio, training, and radio

**Features**:
- **Text Generation**: Interactive text generation with parameter control
- **Image Generation**: Diffusion model image generation with prompts
- **Audio Processing**: Real-time audio effects and analysis
- **Training Visualization**: Live training progress and metrics
- **Radio Control**: Integrated radio streaming and control

### 2. Real-time Inference Demo (`realtime_inference_demo.py`)
**Port**: 7861
**Description**: Live model inference with real-time performance monitoring

**Features**:
- **Live Inference**: Real-time text, image, and audio inference
- **Performance Monitoring**: CPU, GPU, memory, and latency tracking
- **Dynamic Updates**: Live performance visualization
- **Multi-modal Processing**: Text, image, and audio processing

### 3. Radio Integration Demo (`radio_integration_demo.py`)
**Port**: 7862
**Description**: Radio streaming and audio processing demos

**Features**:
- **Station Search**: Search for radio stations by genre and country
- **Playback Control**: Play, stop, and volume control
- **Audio Analysis**: Real-time audio feature extraction
- **Playlist Management**: Create and manage custom playlists

## 🚀 Quick Start

### Installation

1. **Install Dependencies**:
```bash
pip install -r requirements_gradio_demos.txt
```

2. **Check Dependencies**:
```bash
python demo_launcher.py --check
```

3. **Launch Demos**:
```bash
# Launch all demos
python demo_launcher.py --all

# Launch specific demo
python demo_launcher.py --demo main
python demo_launcher.py --demo realtime
python demo_launcher.py --demo radio

# Launch with custom port
python demo_launcher.py --demo main --port 8080

# Launch with public sharing
python demo_launcher.py --demo main --share
```

### Demo Launcher Commands

```bash
# List all available demos
python demo_launcher.py --list

# Check dependencies
python demo_launcher.py --check

# Launch specific demo
python demo_launcher.py --demo main

# Launch all demos
python demo_launcher.py --all

# Stop specific demo
python demo_launcher.py --stop main

# Stop all demos
python demo_launcher.py --stop-all

# Show demo status
python demo_launcher.py --status

# Monitor running demos
python demo_launcher.py --monitor

# Create standalone script
python demo_launcher.py --create-script main
```

## 📊 Demo Features

### Text Generation Demo

**Capabilities**:
- Interactive text generation with customizable parameters
- Temperature and top-p sampling control
- Multiple sample generation
- Text analysis and statistics

**Parameters**:
- **Prompt**: Input text for generation
- **Max Length**: Maximum output length
- **Temperature**: Creativity level (0.1-2.0)
- **Top-p**: Nucleus sampling parameter (0.1-1.0)
- **Number of Samples**: Multiple outputs

**Usage**:
1. Enter a text prompt
2. Adjust generation parameters
3. Click "Generate Text"
4. View results and analysis

### Image Generation Demo

**Capabilities**:
- Diffusion model image generation
- Prompt and negative prompt control
- Image quality and size settings
- Real-time image analysis

**Parameters**:
- **Prompt**: Description of desired image
- **Negative Prompt**: What to avoid
- **Number of Steps**: Generation quality
- **Guidance Scale**: Prompt adherence
- **Seed**: Reproducible generation
- **Width/Height**: Image dimensions

**Usage**:
1. Enter image description
2. Set generation parameters
3. Click "Generate Image"
4. View and analyze results

### Audio Processing Demo

**Capabilities**:
- Real-time audio effects processing
- Audio analysis and feature extraction
- Multiple audio operations
- Quality assessment

**Operations**:
- **Noise Reduction**: Remove background noise
- **Equalizer**: Frequency adjustment
- **Reverb**: Add spatial effects
- **Pitch Shift**: Change audio pitch

**Usage**:
1. Upload or record audio
2. Select processing operation
3. Click "Process Audio"
4. Download processed audio

### Training Visualization Demo

**Capabilities**:
- Live training progress monitoring
- Performance metrics visualization
- Interactive parameter adjustment
- Real-time plot updates

**Metrics**:
- **Training Loss**: Model training progress
- **Validation Loss**: Model generalization
- **Accuracy**: Classification performance
- **Learning Rate**: Optimization schedule

**Usage**:
1. Set training parameters
2. Click "Update Training Plot"
3. View live training progress
4. Monitor performance metrics

### Real-time Inference Demo

**Capabilities**:
- Live model inference
- Real-time performance monitoring
- Dynamic parameter adjustment
- Multi-modal processing

**Monitoring**:
- **CPU Usage**: System processor utilization
- **Memory Usage**: RAM consumption
- **GPU Usage**: Graphics processor utilization
- **Inference Latency**: Response time
- **Throughput**: Processing speed

**Usage**:
1. Start performance monitoring
2. Run inference on different modalities
3. View real-time performance metrics
4. Monitor system resources

### Radio Integration Demo

**Capabilities**:
- Radio station search and discovery
- Live audio streaming
- Playlist management
- Audio analysis

**Features**:
- **Station Search**: Find stations by genre/country
- **Playback Control**: Play, stop, volume
- **Audio Analysis**: Real-time feature extraction
- **Playlist Management**: Create and save playlists

**Usage**:
1. Search for radio stations
2. Start playback
3. Control volume and playback
4. Analyze audio features

## 🔧 Configuration

### Demo Configuration

Each demo can be configured with custom settings:

```python
# Demo configuration
config = TrainingConfiguration(
    enable_gradio_demo=True,
    gradio_port=7860,
    gradio_share=False,
    enable_radio_integration=True,
    radio_volume=0.7,
    radio_auto_play=False
)
```

### Port Configuration

Default ports for each demo:
- **Main Demos**: 7860
- **Real-time Demo**: 7861
- **Radio Demo**: 7862

Custom ports can be set:
```bash
python demo_launcher.py --demo main --port 8080
```

### Sharing Configuration

Enable public sharing for demos:
```bash
python demo_launcher.py --demo main --share
```

## 📈 Performance Monitoring

### Real-time Metrics

The real-time demo provides comprehensive performance monitoring:

**System Metrics**:
- CPU utilization percentage
- Memory usage percentage
- GPU utilization (if available)
- GPU memory usage

**Inference Metrics**:
- Inference latency (milliseconds)
- Throughput (inferences per second)
- Model response time
- Processing efficiency

### Visualization

Performance data is visualized using Plotly with:
- Real-time line charts
- Multi-panel dashboards
- Interactive plots
- Historical data tracking

### Monitoring Controls

- **Start Monitoring**: Begin performance tracking
- **Stop Monitoring**: End performance tracking
- **Update Performance**: Refresh visualization
- **Auto-refresh**: Continuous updates

## 🎨 Customization

### Theme Customization

Demos use Gradio themes for consistent styling:

```python
import gradio as gr

# Use different themes
interface = gr.Interface(
    fn=your_function,
    inputs=inputs,
    outputs=outputs,
    theme=gr.themes.Soft()  # or gr.themes.Default(), gr.themes.Monochrome()
)
```

### Layout Customization

Customize demo layouts with Gradio components:

```python
with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column(scale=1):
            # Left column
            pass
        with gr.Column(scale=2):
            # Right column
            pass
    
    with gr.Tabs():
        with gr.TabItem("Tab 1"):
            # Tab content
            pass
```

### Function Integration

Integrate custom functions into demos:

```python
def custom_inference_function(input_data, parameters):
    # Your custom inference logic
    result = process_input(input_data, parameters)
    return result

# Add to demo interface
demo_interface = gr.Interface(
    fn=custom_inference_function,
    inputs=[input_component, parameter_component],
    outputs=output_component
)
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**:
   ```bash
   # Use different port
   python demo_launcher.py --demo main --port 8080
   
   # Stop existing demos
   python demo_launcher.py --stop-all
   ```

2. **Missing Dependencies**:
   ```bash
   # Check dependencies
   python demo_launcher.py --check
   
   # Install missing packages
   pip install -r requirements_gradio_demos.txt
   ```

3. **Audio Issues**:
   ```bash
   # Install audio dependencies
   pip install pyaudio librosa soundfile
   
   # System audio setup (Ubuntu)
   sudo apt-get install portaudio19-dev python3-pyaudio
   ```

4. **GPU Issues**:
   ```bash
   # Install GPU monitoring
   pip install GPUtil
   
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

1. **Reduce Update Frequency**: Lower monitoring frequency for better performance
2. **Limit Data Points**: Keep only recent performance data
3. **Use Lower Quality**: Reduce image/audio quality for faster processing
4. **Close Unused Demos**: Stop demos not in use

## 🚀 Advanced Usage

### Custom Demo Creation

Create custom demos by extending the base classes:

```python
from gradio_interactive_demos import InteractiveGradioDemos

class CustomDemo(InteractiveGradioDemos):
    def create_custom_demo(self):
        # Your custom demo implementation
        pass

# Launch custom demo
demo = CustomDemo()
demo.launch_demos()
```

### Integration with Production Code

Integrate demos with your production models:

```python
from production_code import MultiGPUTrainer, TrainingConfiguration

# Initialize with your models
config = TrainingConfiguration(enable_gradio_demo=True)
trainer = MultiGPUTrainer(config)

# Use trainer in demos
demo = InteractiveGradioDemos()
demo.trainer = trainer
demo.launch_demos()
```

### WebSocket Integration

For real-time updates, consider WebSocket integration:

```python
import asyncio
import websockets

async def websocket_handler(websocket, path):
    while True:
        # Send real-time updates
        data = get_performance_data()
        await websocket.send(json.dumps(data))
        await asyncio.sleep(1)

# Start WebSocket server
start_server = websockets.serve(websocket_handler, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
```

## 📚 API Reference

### Demo Classes

#### InteractiveGradioDemos
Main demo class with comprehensive features.

**Methods**:
- `create_text_generation_demo()`: Text generation interface
- `create_image_generation_demo()`: Image generation interface
- `create_audio_processing_demo()`: Audio processing interface
- `create_training_visualization_demo()`: Training visualization
- `create_radio_control_demo()`: Radio control interface
- `create_comprehensive_demo()`: All-in-one interface
- `launch_demos()`: Launch all demos

#### RealTimeInferenceDemo
Real-time inference with performance monitoring.

**Methods**:
- `start_monitoring()`: Start performance monitoring
- `stop_monitoring()`: Stop performance monitoring
- `run_text_inference()`: Text inference
- `run_image_inference()`: Image inference
- `run_audio_inference()`: Audio inference
- `get_performance_plot()`: Performance visualization
- `get_performance_summary()`: Performance summary

#### DemoLauncher
Unified launcher for all demos.

**Methods**:
- `list_demos()`: List available demos
- `check_dependencies()`: Check required packages
- `launch_demo()`: Launch specific demo
- `launch_all_demos()`: Launch all demos
- `stop_demo()`: Stop specific demo
- `stop_all_demos()`: Stop all demos
- `show_status()`: Show demo status
- `monitor_demos()`: Monitor running demos

### Configuration Options

#### TrainingConfiguration
Demo configuration parameters:

```python
@dataclass
class TrainingConfiguration:
    enable_gradio_demo: bool = True
    gradio_port: int = 7860
    gradio_share: bool = False
    enable_radio_integration: bool = True
    radio_volume: float = 0.7
    radio_auto_play: bool = False
    radio_quality: str = "high"
    radio_buffer_size: int = 1024
    radio_sample_rate: int = 44100
    radio_channels: int = 2
```

## 🎯 Best Practices

### Demo Development

1. **Modular Design**: Create separate demo modules for different features
2. **Error Handling**: Implement robust error handling for all operations
3. **Performance Monitoring**: Include performance tracking in all demos
4. **User Feedback**: Provide clear feedback for user actions
5. **Documentation**: Document all demo features and parameters

### Performance Optimization

1. **Efficient Updates**: Use efficient update mechanisms for real-time data
2. **Memory Management**: Properly manage memory for long-running demos
3. **Resource Monitoring**: Monitor system resources during demo execution
4. **Graceful Degradation**: Handle resource limitations gracefully

### User Experience

1. **Intuitive Interface**: Design user-friendly interfaces
2. **Responsive Design**: Ensure demos work on different screen sizes
3. **Clear Documentation**: Provide clear instructions and help text
4. **Error Messages**: Display helpful error messages
5. **Loading States**: Show loading states for long operations

## 🔮 Future Enhancements

### Planned Features

1. **Voice Control**: Voice commands for demo control
2. **Mobile Support**: Mobile-optimized interfaces
3. **Advanced Analytics**: More detailed performance analytics
4. **Model Comparison**: Side-by-side model comparison
5. **Collaborative Features**: Multi-user collaboration
6. **Cloud Integration**: Cloud-based demo hosting
7. **API Endpoints**: REST API for demo functionality
8. **Plugin System**: Extensible plugin architecture

### Contributing

To contribute to the demo system:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📞 Support

For support and questions:

1. **Documentation**: Check this guide and inline documentation
2. **Issues**: Report issues on the project repository
3. **Examples**: Review example implementations
4. **Community**: Join the community discussions

---

**Happy Demo-ing! 🎉**

For more information, see the main documentation or run:
```bash
python demo_launcher.py --help
``` 
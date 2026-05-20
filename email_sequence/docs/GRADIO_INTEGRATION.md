# Gradio Web Interface

A comprehensive web interface for the Email Sequence AI system built with Gradio, providing an intuitive and user-friendly way to interact with all system features.

## Overview

The Gradio web interface offers a complete solution for:

- **Sequence Generation**: AI-powered email sequence creation
- **Evaluation**: Comprehensive sequence analysis and scoring
- **Training**: Model training with advanced optimization
- **Gradient Management**: Real-time gradient monitoring and control

## Features

### 🎨 Modern Web Interface

- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Intuitive Navigation**: Tab-based interface for easy feature access
- **Real-time Updates**: Live feedback and progress tracking
- **Interactive Charts**: Plotly-powered visualizations

### 📧 Sequence Generation

- **AI Model Selection**: Choose from multiple AI models (GPT-3.5, GPT-4, Claude, Custom)
- **Parameter Control**: Adjust sequence length, creativity, and industry focus
- **Audience Targeting**: Select from predefined subscriber profiles
- **Real-time Preview**: See generated sequences immediately

### 📊 Evaluation Dashboard

- **Multi-metric Analysis**: Content quality, engagement, business impact, technical metrics
- **Configurable Weights**: Adjust importance of different evaluation criteria
- **Visual Reports**: Interactive charts and radar plots
- **Step-by-step Analysis**: Detailed breakdown of each sequence step

### 🚀 Training Interface

- **Advanced Configuration**: Early stopping, learning rate scheduling, gradient management
- **Real-time Monitoring**: Live training progress and metrics
- **Performance Charts**: Loss curves, learning rate schedules, gradient statistics
- **Optimization Reports**: Comprehensive training summaries

### 🔧 Gradient Management

- **Multiple Clipping Strategies**: Norm-based, value-based, and adaptive clipping
- **NaN/Inf Handling**: Automatic detection and fixing of numerical issues
- **Health Monitoring**: Real-time gradient health assessment
- **Visual Analytics**: Gradient statistics and health tracking

## Quick Start

### Installation

```bash
# Install Gradio requirements
pip install -r requirements/gradio_requirements.txt

# Install core system requirements
pip install -r requirements/evaluation_requirements.txt
```

### Launch the Application

```bash
# Run the Gradio app
python gradio_app.py
```

The application will be available at:
- **Local**: http://localhost:7860
- **Public**: A shareable link will be provided

## Interface Guide

### 1. Sequence Generation Tab

#### Configuration Panel

- **AI Model**: Select the AI model for sequence generation
  - GPT-3.5: Fast and cost-effective
  - GPT-4: Higher quality, more expensive
  - Claude: Alternative AI model
  - Custom: Your own model

- **Sequence Length**: Number of emails in the sequence (1-10)
- **Creativity Level**: Controls AI creativity (0.1-1.0)
- **Target Audience**: Predefined subscriber profiles
- **Industry Focus**: Target industry for customization

#### Output Panel

- **Sequence JSON**: Raw sequence data in JSON format
- **Sequence Preview**: Formatted markdown preview of the sequence

#### Usage Example

1. Select "GPT-4" as the AI model
2. Set sequence length to 5
3. Adjust creativity to 0.8
4. Choose "John Doe (Tech Corp)" as target audience
5. Set industry focus to "Technology"
6. Click "Generate Sequence"

### 2. Evaluation Tab

#### Configuration Panel

- **Content Quality Metrics**: Enable/disable content analysis
- **Engagement Metrics**: Enable/disable engagement analysis
- **Business Impact Metrics**: Enable/disable business analysis
- **Technical Metrics**: Enable/disable technical analysis
- **Weight Configuration**: Adjust importance of different metrics

#### Output Panel

- **Evaluation Results**: Detailed JSON results
- **Evaluation Summary**: Human-readable summary
- **Evaluation Charts**: Interactive visualizations

#### Usage Example

1. Enable all metric types
2. Set content quality weight to 0.4
3. Set engagement weight to 0.3
4. Click "Evaluate Sequence"
5. Review results and charts

### 3. Training Tab

#### Configuration Panel

**Early Stopping**
- **Patience**: Number of epochs without improvement
- **Minimum Delta**: Minimum change threshold

**Learning Rate Scheduler**
- **Scheduler Type**: cosine, step, exponential, plateau, onecycle
- **Initial Learning Rate**: Starting learning rate

**Gradient Management**
- **Max Gradient Norm**: Maximum gradient norm for clipping
- **Enable Gradient Clipping**: Toggle gradient clipping
- **Enable NaN/Inf Check**: Toggle numerical stability checks

**Training Parameters**
- **Max Epochs**: Maximum training epochs
- **Batch Size**: Training batch size

#### Output Panel

- **Training Log**: Real-time training progress
- **Training Metrics**: Final training statistics
- **Training Charts**: Loss curves and learning rate schedules

#### Usage Example

1. Set patience to 15 epochs
2. Choose "cosine" scheduler
3. Set initial learning rate to 0.001
4. Enable gradient clipping with max norm 1.0
5. Set max epochs to 100
6. Click "Start Training"

### 4. Gradient Management Tab

#### Configuration Panel

**Gradient Clipping**
- **Max Gradient Norm**: Maximum gradient norm
- **Clipping Type**: norm, value, adaptive

**NaN/Inf Handling**
- **Enable NaN/Inf Check**: Toggle detection
- **Replace NaN With**: Replacement value for NaN
- **Replace Inf With**: Replacement value for Inf

**Monitoring**
- **Enable Gradient Monitoring**: Toggle monitoring
- **Verbose Logging**: Toggle detailed logging

**Adaptive Clipping**
- **Adaptive Clipping**: Toggle adaptive strategy
- **Adaptive Window Size**: History window size

#### Output Panel

- **Gradient Management Log**: Real-time gradient information
- **Gradient Metrics**: Comprehensive gradient statistics
- **Gradient Charts**: Gradient health and statistics

#### Usage Example

1. Set max gradient norm to 1.0
2. Choose "norm" clipping type
3. Enable NaN/Inf checking
4. Enable gradient monitoring
5. Click "Test Gradient Management"

## Advanced Features

### Custom Model Integration

```python
# Add your custom model to the interface
class CustomModel:
    def generate_sequence(self, prompt, **kwargs):
        # Your custom generation logic
        return generated_sequence

# Update the model choices in gradio_app.py
model_choices = ["GPT-3.5", "GPT-4", "Claude", "Custom", "YourModel"]
```

### Custom Evaluation Metrics

```python
# Add custom evaluation metrics
class CustomMetrics:
    def evaluate(self, sequence):
        # Your custom evaluation logic
        return custom_score

# Integrate with the evaluation interface
evaluator.add_custom_metrics(CustomMetrics())
```

### Real-time Monitoring

The interface provides real-time monitoring for:

- **Training Progress**: Live loss curves and metrics
- **Gradient Health**: Real-time gradient statistics
- **System Performance**: Memory usage and processing time
- **Error Handling**: Automatic error detection and reporting

### Export Capabilities

- **JSON Export**: Download evaluation results and training logs
- **Chart Export**: Save visualizations as PNG/PDF
- **Report Generation**: Generate comprehensive PDF reports
- **Data Export**: Export training data and metrics

## Configuration

### Environment Variables

```bash
# Gradio configuration
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=true
GRADIO_DEBUG=true

# AI model API keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# System configuration
LOG_LEVEL=INFO
SAVE_PATH=./outputs
```

### Custom Themes

```python
# Custom Gradio theme
custom_theme = gr.themes.Soft().set(
    body_background_fill="*background_fill_secondary",
    background_fill_primary="*background_fill_primary",
    border_color_accent="*border_color_accent",
    border_color_primary="*border_color_primary",
    color_accent="*color_accent",
    color_accent_soft="*color_accent_soft",
    background_fill_secondary="*background_fill_secondary",
)

# Apply theme to app
app = gr.Blocks(theme=custom_theme)
```

## Performance Optimization

### Memory Management

- **Batch Processing**: Process sequences in batches
- **Lazy Loading**: Load models only when needed
- **Memory Monitoring**: Track memory usage
- **Garbage Collection**: Automatic cleanup

### Caching

- **Result Caching**: Cache evaluation results
- **Model Caching**: Cache loaded models
- **Chart Caching**: Cache generated visualizations
- **Configuration Caching**: Cache user preferences

### Async Processing

- **Non-blocking Operations**: All operations are async
- **Progress Updates**: Real-time progress feedback
- **Error Handling**: Graceful error recovery
- **Timeout Management**: Automatic timeout handling

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port
   python gradio_app.py --port 7861
   ```

2. **Model Loading Errors**
   ```bash
   # Check model paths
   python -c "import torch; print(torch.__version__)"
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size or model size
   # Enable gradient checkpointing
   ```

4. **API Key Issues**
   ```bash
   # Set environment variables
   export OPENAI_API_KEY=your_key
   export ANTHROPIC_API_KEY=your_key
   ```

### Debug Mode

```python
# Enable debug mode
app.launch(
    debug=True,
    show_error=True,
    log_level="DEBUG"
)
```

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Security Considerations

### API Key Management

- **Environment Variables**: Store API keys securely
- **Key Rotation**: Regular key rotation
- **Access Control**: Limit API key access
- **Monitoring**: Monitor API usage

### Input Validation

- **Sanitization**: Sanitize all user inputs
- **Validation**: Validate input parameters
- **Rate Limiting**: Implement rate limiting
- **Error Handling**: Secure error messages

### Data Privacy

- **Local Processing**: Process data locally when possible
- **Data Encryption**: Encrypt sensitive data
- **Access Logs**: Log access for audit trails
- **Data Retention**: Implement data retention policies

## Deployment

### Local Deployment

```bash
# Simple local deployment
python gradio_app.py

# With custom configuration
python gradio_app.py --server-name 0.0.0.0 --server-port 7860 --share
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements/ ./requirements/
RUN pip install -r requirements/gradio_requirements.txt

COPY . .
EXPOSE 7860

CMD ["python", "gradio_app.py"]
```

```bash
# Build and run
docker build -t email-sequence-app .
docker run -p 7860:7860 email-sequence-app
```

### Cloud Deployment

#### Heroku

```bash
# Create Procfile
echo "web: python gradio_app.py" > Procfile

# Deploy
heroku create email-sequence-app
git push heroku main
```

#### AWS/GCP

```bash
# Use container deployment
gcloud run deploy email-sequence-app \
  --image gcr.io/your-project/email-sequence-app \
  --platform managed \
  --allow-unauthenticated
```

## API Integration

### REST API Endpoints

```python
# FastAPI integration
from fastapi import FastAPI
from gradio_app import GradioEmailSequenceApp

app = FastAPI()
gradio_app = GradioEmailSequenceApp()

@app.post("/generate-sequence")
async def generate_sequence(request: dict):
    return await gradio_app.generate_sequence(**request)

@app.post("/evaluate-sequence")
async def evaluate_sequence(request: dict):
    return await gradio_app.evaluate_sequence(**request)
```

### Webhook Integration

```python
# Webhook for external integrations
@app.post("/webhook/sequence-generated")
async def sequence_webhook(sequence: dict):
    # Process generated sequence
    # Send notifications
    # Update databases
    pass
```

## Contributing

### Adding New Features

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Implement Feature**
   - Add new interface components
   - Update backend logic
   - Add tests

3. **Update Documentation**
   - Update this guide
   - Add examples
   - Update requirements

4. **Submit Pull Request**
   - Include tests
   - Update documentation
   - Follow code style

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add comprehensive docstrings
- Include error handling

### Testing

```bash
# Run tests
pytest tests/test_gradio_app.py

# Run with coverage
pytest --cov=gradio_app tests/
```

## Support

### Getting Help

- **Documentation**: Check this guide and other docs
- **Issues**: Open issues on GitHub
- **Discussions**: Use GitHub discussions
- **Email**: Contact the development team

### Community

- **GitHub**: Main repository
- **Discord**: Community server
- **Blog**: Technical blog posts
- **Newsletter**: Monthly updates

## License

This Gradio interface is part of the Email Sequence AI project and follows the same licensing terms.

## Changelog

### Version 1.0.0
- Initial release
- Basic sequence generation
- Evaluation interface
- Training interface
- Gradient management

### Version 1.1.0 (Planned)
- Advanced visualizations
- Custom model support
- API integration
- Performance improvements 
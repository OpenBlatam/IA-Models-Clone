# 🎨 Instagram Captions AI - Gradio Interactive Demos

## Overview

This guide covers the interactive Gradio demos for the Instagram Captions AI system. These demos provide a user-friendly interface for exploring model inference, training, evaluation, and visualization capabilities.

## 📋 Table of Contents1. [Installation](#installation)
2uick Start](#quick-start)3 [Demo Features](#demo-features)
4sage Guide](#usage-guide)5 [API Reference](#api-reference)6 [Customization](#customization)
7. [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.8ip package manager

### Install Dependencies

```bash
# Install Gradio and required packages
pip install -r requirements_gradio.txt

# Or install individually
pip install gradio>=4.0.0pip install plotly>=5.0.0
pip install numpy>=1.21pip install pandas>=1.3.0
pip install matplotlib>=3.5.0ip install seaborn>=0.110erify Installation

```bash
python -c "import gradio; import plotly; print('✅ All packages installed successfully!')"
```

## 🏃‍♂️ Quick Start

### Launch the Demo Server

```bash
# Navigate to the demos directory
cd agents/backend/onyx/server/features/instagram_captions/demos

# Run the basic demo
python basic_gradio_demo.py

# Or run with custom settings
python -c "
from basic_gradio_demo import launch_demo_server
launch_demo_server(host='0.0, port=7860hare=True)
```

### Access the Demo
1 **Local Access**: Open `http://localhost:7860` in your browser
2**Public Access**: The demo will generate a public URL for sharing
3. **Network Access**: Use `http://0.0.0.0:7860` for network access

## 🎯 Demo Features

### 1. Caption Generation Tab

**Purpose**: Generate engaging Instagram captions with AI

**Features**:
- **Input Text**: Describe what you want to caption
- **Max Length**: Control caption length (10-200s)
- **Temperature**: Adjust creativity level (0.1-2 **Style**: Choose from Casual, Professional, Creative, Minimalist, Engaging
- **Hashtags**: Include relevant hashtags automatically
- **Emojis**: Add style-appropriate emojis
- **Real-time Metrics**: View engagement score, word count, hashtag count

**Example Usage**:
```
Input: "A beautiful sunset over the ocean"
Style: Casual
Temperature: 0.8
Result: "✨ A beautiful sunset over the ocean - absolutely amazing! #vibes #life"
```

###2Model Training Tab

**Purpose**: Simulate and visualize model training process

**Features**:
- **Epochs**: Set number of training epochs (1-50
- **Batch Size**: Configure batch size (4
- **Learning Rate**: Adjust learning rate (0.000010.001*Training Progress**: Real-time visualization of loss and accuracy
- **Training Results**: Detailed metrics and statistics

**Visualizations**:
- Training and validation loss curves
- Accuracy progression over epochs
- Final training metrics

### 3. Model Evaluation Tab

**Purpose**: Evaluate model performance with various metrics

**Features**:
- **Test Texts**: Input multiple test cases
- **Reference Captions**: Provide ground truth captions
- **Generated Captions**: Input model-generated captions
- **Comprehensive Metrics**: BLEU, ROUGE, METEOR, BERTScore, etc.
- **Visual Comparisons**: Side-by-side caption analysis

**Metrics Included**:
- BLEU Score (0.3-07)
- ROUGE Score (0.4-00.8)
- METEOR Score (0.350.75)
- BERTScore (0.6-0.9- Engagement Score (0.5-0.85- Creativity Score (0.4
- Relevance Score (0.6-0.9. Batch Processing Tab

**Purpose**: Process multiple inputs efficiently

**Features**:
- **Batch Size**: Configure processing batch size (1-20)
- **Include Metrics**: Generate detailed batch metrics
- **Output Formats**: JSON and CSV output options
- **Processing Summary**: Total processed, batch count, performance stats
- **Batch Visualization**: Engagement scores across batches

## 📖 Usage Guide

### Caption Generation1**Enter Input Text**: Describe the image or content you want to caption
2. **Adjust Parameters**:
   - Set max length based on platform requirements
   - Use temperature to control creativity (higher = more creative)
   - Choose style that matches your brand voice
3. **Configure Options**: Enable/disable hashtags and emojis
4. **Generate**: Click Generate Caption" to create the caption
5. **Review Metrics**: Check engagement score and other metrics
6. **Iterate**: Adjust parameters and regenerate as needed

### Model Training
1raining Parameters**:
   - Start with 10-20 for quick testing
   - Use batch size 16 for most scenarios
   - Learning rate 2e-5 works well for most cases
2. **Start Training**: Click Start Training"
3. **Monitor Progress**: Watch the training curves update
4. **Analyze Results**: Review final metrics and training history

### Model Evaluation
1*Prepare Test Data**:
   - Enter test texts (one per line)
   - Provide reference captions (ground truth)
   - Input generated captions to evaluate
2. **Run Evaluation**: Click Evaluate Model"
3. **Review Metrics**: Check all evaluation scores
4. **Analyze Visualizations**: Compare caption lengths and quality

### Batch Processing

1. **Configure Batch Settings**:
   - Set appropriate batch size for your data
   - Enable metrics for detailed analysis2 **Process Data**: Click Process Batch"
3**Review Output**: Check JSON/CSV output
4. **Analyze Performance**: Review batch processing metrics

## 🔧 API Reference

### InstagramCaptionDemo Class

```python
class InstagramCaptionDemo:
    def __init__(self):
       nitialize the demo with sample data"""
    
    def generate_caption(self, input_text: str, max_length: int, 
                        temperature: float, style: str, 
                        include_hashtags: bool, include_emojis: bool) -> Tuple[str, Dict[str, Any]]:
 Generate Instagram caption with specified parameters"""
    
    def simulate_training(self, epochs: int, batch_size: int, 
                         learning_rate: float) -> Tuple[str, Dict[str, Any], go.Figure]:
       ate model training process"""
    
    def evaluate_model(self, test_texts: str, reference_captions: str, 
                      generated_captions: str) -> Tuple[Dict[str, Any], go.Figure, go.Figure]:
        model performance with various metrics"""
    
    def batch_process(self, batch_size: int, include_metrics: bool) -> Tuple[str, Dict[str, Any], go.Figure]:
        Process a batch of inputs efficiently"""
```

### Key Methods

#### generate_caption()
- **input_text**: Description of content to caption
- **max_length**: Maximum caption length (10-200- **temperature**: Creativity level (0.1-2
- **style**: Caption style (Casual, Professional, Creative, Minimalist, Engaging)
- **include_hashtags**: Whether to add hashtags
- **include_emojis**: Whether to add emojis
- **Returns**: (caption, metrics)

#### simulate_training()
- **epochs**: Number of training epochs
- **batch_size**: Training batch size
- **learning_rate**: Learning rate for optimization
- **Returns**: (status, results, plot)

#### evaluate_model()
- **test_texts**: Test input texts (newline-separated)
- **reference_captions**: Ground truth captions (newline-separated)
- **generated_captions**: Model-generated captions (newline-separated)
- **Returns**: (metrics, metrics_plot, comparison_plot)

## 🎨 Customization

### Adding New Styles

```python
# Add new style to templates
templates["NewStyle"] =  f🎯{input_text} - new style caption #newstyle #custom",
    f"✨ {input_text} - another variation #variation #style"
]

# Add hashtags for new style
hashtag_map["NewStyle"] = #newstyle #custom #variation

# Add emoji for new style
emoji_map["NewStyle] = �"
```

### Custom Metrics

```python
def _calculate_custom_metric(self, caption: str) -> float:
    " custom engagement metric"  # Implement your custom logic
    score = 0# Add your scoring logic
    return score
```

### Custom Visualizations

```python
def _create_custom_plot(self, data: Dict[str, Any]) -> go.Figure:
  stom visualization"""
    fig = go.Figure()
    # Add your custom plot logic
    return fig
```

## 🔧 Configuration

### Server Configuration

```python
# Custom server settings
launch_demo_server(
    host="0.0.00,      # Listen on all interfaces
    port=7860,           # Custom port
    share=True,          # Generate public URL
    debug=False          # Debug mode
)
```

### Demo Configuration

```python
# Custom demo settings
demo = InstagramCaptionDemo()
demo.sample_data = your_custom_data  # Use custom sample data
```

## 🐛 Troubleshooting

### Common Issues

####1 Import Errors
```bash
# Solution: Install missing packages
pip install gradio plotly numpy pandas matplotlib seaborn
```

#### 2ort Already in Use
```bash
# Solution: Use different port
launch_demo_server(port=7861``

####3 Memory Issues
```bash
# Solution: Reduce batch size or use smaller models
# In the demo, use smaller batch sizes
```

#### 4. Slow Performance
```bash
# Solution: Optimize settings
# - Reduce max_length
# - Use smaller batch sizes
# - Disable unnecessary features
```

### Debug Mode

```python
# Enable debug mode for detailed error messages
launch_demo_server(debug=true)
```

### Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Optimization

### Best Practices

1. **Batch Processing**: Use appropriate batch sizes (50 for demos)
2. **Caching**: Enable caching for repeated operations
3. **Async Processing**: Use async functions for I/O operations
4. **Memory Management**: Monitor memory usage during batch processing

### Monitoring

```python
# Monitor performance metrics
import time
import psutil

def monitor_performance():
    start_time = time.time()
    memory_usage = psutil.virtual_memory().percent
    # Your processing code here
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f}s")
    print(f"Memory usage: [object Object]memory_usage:0.1
## 🔗 Integration

### With FastAPI

```python
# Integrate with FastAPI backend
from fastapi import FastAPI
from basic_gradio_demo import InstagramCaptionDemo

app = FastAPI()
demo = InstagramCaptionDemo()

@app.post(/generate_caption")
async def generate_caption_api(request: CaptionRequest):
    caption, metrics = demo.generate_caption(
        request.input_text,
        request.max_length,
        request.temperature,
        request.style,
        request.include_hashtags,
        request.include_emojis
    )
    return {caption": caption, metrics": metrics}
```

### With Streamlit

```python
# Alternative UI with Streamlit
import streamlit as st
from basic_gradio_demo import InstagramCaptionDemo

demo = InstagramCaptionDemo()

st.title("Instagram Captions AI)
input_text = st.text_input(Enter description")
if st.button("Generate Caption"):
    caption, metrics = demo.generate_caption(input_text,10.7asual,trueTrue)
    st.write(caption)
    st.json(metrics)
```

## 📈 Future Enhancements

### Planned Features

1. **Real Model Integration**: Connect to actual trained models2 **Image Upload**: Support for image-based caption generation
3. **Multi-language Support**: Support for multiple languages
4. **Advanced Analytics**: More detailed performance metrics
5. **Export Features**: Export results to various formats
6aborative Features**: Multi-user support and sharing

### Contributing

1. Fork the repository
2. Create a feature branch3ement your changes
4. Add tests and documentation
5. Submit a pull request

## 📞 Support

### Getting Help

- **Documentation**: Check this guide and inline comments
- **Issues**: Report bugs and feature requests
- **Community**: Join discussions and share solutions

### Contact

For technical support or questions about the demos, please refer to the main project documentation or create an issue in the repository.

---

**Note**: This demo is for educational and testing purposes. For production use, ensure proper security measures, error handling, and performance optimization are implemented. 
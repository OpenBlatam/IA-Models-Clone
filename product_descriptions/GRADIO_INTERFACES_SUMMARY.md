# Gradio Interfaces for Cybersecurity Model Showcase

## Overview

This module provides comprehensive, user-friendly Gradio interfaces for showcasing cybersecurity AI models. The interfaces are designed to demonstrate model capabilities through interactive demos, real-time inference, batch analysis, and advanced visualizations.

## Key Features

### 🎯 Real-time Inference Interface
- **Live Threat Analysis**: Input feature values and get instant threat classification
- **Confidence Scoring**: Real-time confidence scores with risk level assessment
- **Probability Distribution**: Detailed class probability breakdown
- **Timestamp Tracking**: Automatic timestamp logging for audit trails

### 📊 Batch Analysis Interface
- **CSV Upload**: Upload datasets for batch processing
- **Confusion Matrix**: Interactive confusion matrix visualization
- **ROC Curves**: Multi-class ROC curve analysis
- **Confidence Distribution**: Box plots showing confidence score distributions
- **Results Summary**: Comprehensive analysis results table

### 🚨 Anomaly Detection Interface
- **Anomaly Score Distribution**: Histogram visualization of anomaly scores
- **Feature Importance**: Top 10 most important features analysis
- **Detection Summary**: Statistical summary of anomaly detection results
- **Configurable Sample Size**: Adjustable demo dataset size

### 📈 Performance Dashboard
- **Real-time Trends**: Prediction confidence trends over time
- **Risk Distribution**: Pie chart showing risk level distribution
- **Performance Metrics**: Comprehensive performance summary table
- **History Management**: Clear prediction history functionality

### 🔬 Model Interpretability Interface
- **Feature Importance**: Gradient-based feature importance analysis
- **Feature Analysis Table**: Detailed feature value and importance breakdown
- **Interactive Visualization**: Bar chart showing feature importance scores
- **Normalized Values**: Feature value normalization for comparison

## Architecture

### Core Components

1. **CybersecurityModelInterface Class**
   - Manages model inference and analysis
   - Handles data sanitization (NaN/Inf values)
   - Maintains prediction history
   - Generates demo data for visualization

2. **Advanced Visualization System**
   - Plotly-based interactive charts
   - Matplotlib/Seaborn for static plots
   - Real-time chart updates
   - Responsive design for different screen sizes

3. **Data Processing Pipeline**
   - Input validation and sanitization
   - Feature preprocessing
   - Output post-processing
   - Error handling and recovery

### Key Methods

```python
# Real-time inference with detailed analysis
real_time_inference(*feature_values) -> Dict[str, Any]

# Batch analysis with comprehensive visualizations
batch_analysis(csv_file) -> Tuple[pd.DataFrame, go.Figure, go.Figure, go.Figure]

# Anomaly detection demo with visualizations
anomaly_detection_demo(num_samples: int) -> Tuple[go.Figure, go.Figure, pd.DataFrame]

# Performance dashboard with real-time data
model_performance_dashboard() -> Tuple[go.Figure, go.Figure, pd.DataFrame]

# Model interpretability analysis
model_interpretability(feature_values: List[float]) -> Tuple[go.Figure, pd.DataFrame]
```

## Usage Guide

### Installation

1. Install required dependencies:
```bash
pip install -r requirements-gradio.txt
```

2. Run the advanced interface:
```bash
python advanced_gradio_interfaces.py
```

3. Access the interface at `http://localhost:7860`

### Basic Usage

#### Real-time Inference
1. Navigate to the "Real-time Inference" tab
2. Enter feature values (20 features for demo model)
3. Click "Analyze Threat" to get instant results
4. View threat type, confidence, risk level, and probabilities

#### Batch Analysis
1. Navigate to the "Batch Analysis" tab
2. Upload a CSV file with features and labels
3. Click "Analyze Batch" to process the data
4. View confusion matrix, ROC curves, and confidence distributions

#### Anomaly Detection
1. Navigate to the "Anomaly Detection" tab
2. Adjust sample size if needed
3. Click "Run Anomaly Detection" to generate demo data
4. View anomaly score distribution and feature importance

#### Performance Dashboard
1. Navigate to the "Performance Dashboard" tab
2. Make some predictions in the real-time interface first
3. Click "Refresh Dashboard" to see trends and metrics
4. Use "Clear History" to reset the dashboard

#### Model Interpretability
1. Navigate to the "Model Interpretability" tab
2. Enter feature values for analysis
3. Click "Analyze Features" to see importance scores
4. View feature importance plot and analysis table

## Configuration

### Model Configuration
- **Input Dimension**: 20 features (configurable)
- **Output Classes**: 4 classes (Normal, Malware, Intrusion, Exfiltration)
- **Model Architecture**: 3-layer neural network with dropout

### Interface Configuration
- **Server Port**: 7860 (configurable)
- **Share Mode**: Enabled for public sharing
- **Error Display**: Enabled for debugging
- **Tips Display**: Enabled for user guidance

### Visualization Configuration
- **Chart Themes**: Modern, responsive design
- **Color Schemes**: Consistent cybersecurity theme
- **Interactive Elements**: Hover tooltips, zoom, pan
- **Export Options**: PNG, SVG, PDF formats

## Best Practices

### Data Handling
1. **Input Validation**: Always validate and sanitize inputs
2. **NaN/Inf Handling**: Use `_sanitize_output()` method for all outputs
3. **Error Recovery**: Graceful error handling with user-friendly messages
4. **Data Privacy**: Don't log sensitive data in production

### Visualization
1. **Responsive Design**: Ensure charts work on different screen sizes
2. **Accessibility**: Use colorblind-friendly color schemes
3. **Performance**: Optimize chart rendering for large datasets
4. **Interactivity**: Provide meaningful hover information

### User Experience
1. **Clear Labels**: Use descriptive labels and titles
2. **Loading States**: Show loading indicators for long operations
3. **Error Messages**: Provide helpful error messages
4. **Documentation**: Include tooltips and help text

## Security Considerations

### Input Validation
- Validate all user inputs
- Sanitize data before processing
- Handle malicious inputs gracefully
- Log security-relevant events

### Model Protection
- Don't expose model internals unnecessarily
- Implement rate limiting for API calls
- Monitor for adversarial inputs
- Regular model updates and validation

### Data Privacy
- Don't store sensitive data in logs
- Implement data retention policies
- Use anonymized data for demos
- Comply with relevant privacy regulations

## Customization

### Adding New Models
1. Replace the `_create_demo_model()` method
2. Update input/output dimensions
3. Modify threat type mappings
4. Adjust visualization parameters

### Adding New Visualizations
1. Create new Plotly/Matplotlib figures
2. Add new tabs or sections
3. Implement corresponding analysis methods
4. Update the interface layout

### Styling Customization
1. Modify the `custom_css` variable
2. Update color schemes and themes
3. Adjust layout and spacing
4. Add custom HTML components

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   - Change the port number in `demo.launch()`
   - Kill existing processes using the port

2. **Model Loading Errors**
   - Check model file paths
   - Verify model architecture compatibility
   - Ensure all dependencies are installed

3. **Visualization Errors**
   - Check data types and formats
   - Verify Plotly/Matplotlib installations
   - Handle empty or invalid data

4. **Performance Issues**
   - Reduce batch sizes for large datasets
   - Optimize model inference
   - Use caching for repeated operations

### Debug Mode
Enable debug mode by setting:
```python
demo.launch(show_error=True, show_tips=True)
```

## Future Enhancements

### Planned Features
1. **Real-time Data Streaming**: Live data feed integration
2. **Advanced Analytics**: Statistical analysis tools
3. **Model Comparison**: Side-by-side model evaluation
4. **Export Functionality**: Save results and visualizations
5. **API Integration**: REST API endpoints
6. **Mobile Optimization**: Responsive mobile interface

### Performance Improvements
1. **Caching**: Implement result caching
2. **Async Processing**: Background task processing
3. **Database Integration**: Persistent storage
4. **Load Balancing**: Multiple server instances

## Dependencies

### Core Dependencies
- `gradio>=4.0.0`: Web interface framework
- `torch>=2.0.0`: Deep learning framework
- `pandas>=2.0.0`: Data manipulation
- `numpy>=1.24.0`: Numerical computing
- `plotly>=5.15.0`: Interactive visualizations
- `matplotlib>=3.7.0`: Static plotting
- `seaborn>=0.12.0`: Statistical visualizations

### Optional Dependencies
- `bokeh>=3.0.0`: Advanced visualizations
- `altair>=5.0.0`: Declarative visualizations
- `kaleido>=0.2.1`: Static plot export

## Conclusion

The Gradio interfaces provide a comprehensive showcase for cybersecurity AI models, offering both technical depth and user-friendly interaction. The modular design allows for easy customization and extension, while the robust error handling and data sanitization ensure reliable operation in production environments.

For questions or contributions, please refer to the project documentation or contact the development team. 
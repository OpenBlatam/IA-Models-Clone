# Quality Control AI

> Part of the [Blatam Academy Integrated Platform](../README.md)

Quality Control System with Camera-Based Defect Detection

## Description

Complete quality control system that uses a camera to detect objects, identify anomalies, and classify product defects. The system integrates multiple computer vision and machine learning techniques for automatic quality inspection.

## Features

- **Camera Control**: Complete camera management with flexible configuration
- **Object Detection**: Object detection using deep learning models (YOLOv8, Faster R-CNN, SSD)
- **Anomaly Detection**: Multiple anomaly detection methods (statistical, autoencoder, edges, color)
- **Defect Classification**: Automatic defect classification into multiple categories
- **Quality Analysis**: Detailed analysis and quality scoring
- **Real-Time Inspection**: Continuous inspection from live camera
- **Video Analysis**: Complete analysis of video files and streams
- **Advanced Visualization**: Result visualization with charts and statistics
- **Alert System**: Automatic alerts based on quality thresholds
- **Report Generation**: Reports in JSON, CSV, and HTML
- **Performance Optimization**: Caching and optimizations for better performance

## Structure

```
quality_control_ai/
├── __init__.py
├── core/
│   ├── camera_controller.py      # Camera controller
│   ├── object_detector.py        # Object detector
│   ├── anomaly_detector.py       # Anomaly detector
│   └── defect_classifier.py     # Defect classifier
├── config/
│   ├── camera_config.py          # Camera configuration
│   └── detection_config.py       # Detection configuration
├── services/
│   ├── quality_inspector.py     # Main quality inspector
│   └── defect_analyzer.py        # Defect analyzer
└── utils/
    ├── image_utils.py           # Image utilities
    └── detection_utils.py        # Detection utilities
```

## Basic Usage

### Initialize Inspector

```python
from quality_control_ai import QualityInspector, CameraConfig, DetectionConfig

# Configure camera
camera_config = CameraConfig()
camera_config.update_settings(
    camera_index=0,
    resolution_width=1920,
    resolution_height=1080,
    fps=30
)

# Configure detection
detection_config = DetectionConfig()
detection_config.update_settings(
    confidence_threshold=0.5,
    anomaly_threshold=0.7
)

# Create inspector
inspector = QualityInspector(camera_config, detection_config)
```

### Camera Inspection

```python
# Initialize camera
inspector.start_inspection()

# Inspect frame
result = inspector.inspect_frame()

print(f"Quality: {result['quality_score']}/100")
print(f"Defects detected: {result['defects_detected']}")
print(f"Status: {result['summary']['status']}")

# Stop inspection
inspector.stop_inspection()
inspector.release()
```

### Image Inspection

```python
# Inspect image from file
result = inspector.inspect_frame("path/to/image.jpg")

# View results
for defect in result['defects']:
    print(f"Type: {defect['type']}, Severity: {defect['severity']}")
```

### Detailed Defect Analysis

```python
from quality_control_ai import DefectAnalyzer

analyzer = DefectAnalyzer()

# Analyze defects
analysis = analyzer.analyze_defects(
    defects=result['defects'],
    image_size=(1920, 1080)
)

print(f"Analysis by type: {analysis['type_analysis']}")
print(f"Recommendations: {analysis['recommendations']}")
```

## Detected Defect Types

The system can detect and classify the following types of defects:

- **Scratch**: Thin and long lines
- **Crack**: Wider and deeper lines
- **Dent**: Circular/elliptical deformations
- **Discoloration**: Anomalous color changes
- **Deformation**: Shape changes
- **Missing Part**: Absent components
- **Surface Imperfection**: Surface irregularities
- **Contamination**: Foreign material
- **Size Variation**: Dimensional deviations
- **Other**: Uncategorized defects

## Configuration

### Camera Configuration

```python
camera_config = CameraConfig()
camera_config.update_settings(
    camera_index=0,              # Camera index
    resolution_width=1920,       # Resolution width
    resolution_height=1080,      # Resolution height
    fps=30,                      # Frames per second
    brightness=0.5,              # Brightness (0-1)
    contrast=0.5,                # Contrast (0-1)
    saturation=0.5,              # Saturation (0-1)
    exposure=0.5,                # Exposure (0-1)
    auto_focus=True,             # Autofocus
    white_balance="auto"         # White balance
)
```

### Detection Configuration

```python
detection_config = DetectionConfig()
detection_config.update_settings(
    confidence_threshold=0.5,     # Confidence threshold
    nms_threshold=0.4,            # NMS threshold
    anomaly_threshold=0.7,        # Anomaly threshold
    use_autoencoder=True,         # Use autoencoder
    use_statistical=True,         # Use statistical detection
    object_detection_model="yolov8",  # Detection model
    min_defect_size=10,           # Minimum defect size
    max_defect_size=10000         # Maximum defect size
)
```

## Anomaly Detection Methods

The system uses multiple methods to detect anomalies:

1. **Statistical Detection**: Uses statistical analysis (Z-score) to identify outliers
2. **Autoencoder**: Autoencoder model for reconstruction and anomaly detection
3. **Edge Detection**: Analysis of irregular contours
4. **Color Detection**: Identification of anomalous color changes

## Quality Scoring

The system calculates a quality score (0-100) based on:

- Number and severity of detected anomalies
- Number and severity of classified defects
- Affected area coverage

**Score Interpretation:**
- 90-100: Excellent
- 75-89: Good
- 60-74: Acceptable
- 40-59: Poor (requires review)
- 0-39: Rejected

## Dependencies

- OpenCV (`cv2`)
- NumPy
- PIL/Pillow
- scikit-learn (for statistical analysis)
- ultralytics (optional, for YOLOv8)

## Complete Example

```python
from quality_control_ai import (
    QualityInspector, 
    CameraConfig, 
    DetectionConfig,
    VideoAnalyzer,
    AlertSystem,
    QualityVisualizer,
    ReportGenerator
)

# Configure
camera_config = CameraConfig()
detection_config = DetectionConfig()

# Create inspector
inspector = QualityInspector(camera_config, detection_config)

# Start inspection
if inspector.start_inspection():
    # Inspect
    result = inspector.inspect_frame()
    
    # Show results
    print(f"Quality Score: {result['quality_score']}")
    print(f"Status: {result['summary']['status']}")
    print(f"Recommendation: {result['summary']['recommendation']}")
    
    # Visualize results
    visualizer = QualityVisualizer()
    vis_image = visualizer.visualize_inspection(
        image=result.get('image'),
        objects=result.get('objects'),
        defects=result.get('defects'),
        quality_score=result.get('quality_score')
    )
    cv2.imwrite('inspection_result.jpg', vis_image)
    
    # Generate report
    report_gen = ReportGenerator()
    report_gen.generate_html_report(result, 'report.html')
    
    # Alert system
    alert_system = AlertSystem()
    alerts = alert_system.check_inspection_result(result)
    for alert in alerts:
        print(f"ALERT [{alert.level.value}]: {alert.message}")
    
    # Stop
    inspector.stop_inspection()
    inspector.release()
```

## Video Analysis

```python
from quality_control_ai import VideoAnalyzer, QualityInspector

# Create video analyzer
inspector = QualityInspector()
video_analyzer = VideoAnalyzer(inspector)

# Analyze video file
result = video_analyzer.analyze_video_file(
    video_path="production_line.mp4",
    frame_skip=5,  # Analyze every 5 frames
    max_frames=1000
)

print(f"Average Quality: {result['average_quality_score']}")
print(f"Status: {result['status']}")

# Real-time streaming analysis
for result in video_analyzer.analyze_stream(duration=60):
    if result.get('quality_score', 100) < 60:
        print(f"Alert: Low quality in frame {result['frame_number']}")
```

## Alert System

```python
from quality_control_ai import AlertSystem, AlertLevel

alert_system = AlertSystem()

# Configure thresholds
alert_system.set_threshold("quality_score_low", 40.0)
alert_system.set_threshold("defect_count_critical", 5)

# Custom callback
def on_alert(alert):
    if alert.level == AlertLevel.CRITICAL:
        send_email_notification(alert.message)
    print(f"[{alert.level.value}] {alert.message}")

alert_system.register_callback(on_alert)

# Check result
alerts = alert_system.check_inspection_result(inspection_result)

# Get statistics
stats = alert_system.get_alert_statistics()
print(f"Recent critical alerts: {stats['recent_critical']}")
```

## Advanced Visualization

```python
from quality_control_ai import QualityVisualizer

visualizer = QualityVisualizer()

# Visualize inspection
vis_image = visualizer.visualize_inspection(
    image=image,
    objects=objects,
    anomalies=anomalies,
    defects=defects,
    quality_score=85.5,
    show_labels=True
)

# Create summary image
summary_image = visualizer.create_summary_image(
    image=image,
    inspection_result=result,
    save_path="summary.jpg"
)

# Generate statistics charts
plot_bytes = visualizer.create_statistics_plot(
    inspection_result=result,
    save_path="statistics.png"
)
```

## Report Generation

```python
from quality_control_ai import ReportGenerator

report_gen = ReportGenerator()

# JSON Report
report_gen.generate_json_report(
    inspection_result=result,
    output_path="report.json",
    include_images=False
)

# CSV Report (multiple inspections)
report_gen.generate_csv_report(
    inspection_results=[result1, result2, result3],
    output_path="batch_report.csv"
)

# HTML Report
report_gen.generate_html_report(
    inspection_result=result,
    output_path="report.html",
    include_charts=True
)
```

## Performance Optimization

```python
from quality_control_ai import PerformanceOptimizer, measure_time

optimizer = PerformanceOptimizer(cache_size=200)

# Use cache decorator
@optimizer.cached(ttl=300)  # Cache for 5 minutes
def expensive_detection(image):
    # Expensive operation
    return detect_objects(image)

# Measure execution time
@measure_time
def process_image(image):
    return inspector.inspect_frame(image)

# Get statistics
stats = optimizer.get_performance_stats("expensive_detection")
print(f"Average time: {stats['avg_time']:.4f}s")
```

## Notes

- The system requires a connected camera or images to process
- ML models (YOLOv8, autoencoder) are optional but improve accuracy
- Configuration can be adjusted according to the type of product to be inspected
- The system is extensible and allows adding new defect types

## Author

Blatam Academy

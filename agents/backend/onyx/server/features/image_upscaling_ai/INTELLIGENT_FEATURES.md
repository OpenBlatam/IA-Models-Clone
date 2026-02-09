# Intelligent Features Guide

## Overview

This guide covers intelligent features including adaptive learning, real-time analysis, and smart recommendations.

## Adaptive Learner

### Learning from Experience

The adaptive learner improves recommendations over time by learning from results.

```python
from image_upscaling_ai.models import AdaptiveLearner

learner = AdaptiveLearner(learning_file="./learning.json")

# Record experiences
learner.record_experience(
    image_type="anime",
    scale_factor=4.0,
    method_used="RealESRGAN_x4plus_anime_6B",
    quality_score=0.92,
    processing_time=1.8,
    user_satisfaction=0.95
)

# Get recommendations
method, confidence = learner.recommend_method(
    image_type="anime",
    scale_factor=4.0,
    prioritize_speed=False
)

print(f"Recommended: {method} (confidence: {confidence:.2f})")
```

### Quality and Time Prediction

```python
# Predict quality
predicted_quality, qual_conf = learner.predict_quality(
    image_type="photo",
    scale_factor=4.0,
    method="RealESRGAN_x4plus"
)

# Predict time
predicted_time, time_conf = learner.predict_time(
    image_type="photo",
    scale_factor=4.0,
    method="RealESRGAN_x4plus"
)

print(f"Expected quality: {predicted_quality:.2f} (confidence: {qual_conf:.2f})")
print(f"Expected time: {predicted_time:.2f}s (confidence: {time_conf:.2f})")
```

### Statistics

```python
stats = learner.get_statistics()
print(f"Total records: {stats['total_records']}")
print(f"Methods tested: {stats['methods_tested']}")
print(f"Average quality: {stats['avg_quality']:.2f}")
print(f"Average time: {stats['avg_time']:.2f}s")

# Method performance
for method, perf in stats['method_performance'].items():
    print(f"{method}:")
    print(f"  Quality: {perf['avg_quality']:.2f}")
    print(f"  Time: {perf['avg_time']:.2f}s")
    print(f"  Success rate: {perf['success_rate']:.2%}")
```

## Real-Time Analyzer

### Progress Tracking

Monitor upscaling operations in real-time.

```python
from image_upscaling_ai.models import RealtimeAnalyzer

analyzer = RealtimeAnalyzer(update_interval=0.1)

# Add callbacks
def on_progress(metrics):
    print(f"Stage: {metrics.stage}, Progress: {metrics.progress:.1%}")
    if metrics.time_remaining:
        print(f"Time remaining: {metrics.time_remaining:.1f}s")

def on_quality(quality):
    print(f"Estimated quality: {quality:.2f}")

analyzer.add_progress_callback(on_progress)
analyzer.add_quality_callback(on_quality)

# Start operation
analyzer.start_operation()

# Update stages
analyzer.update_stage("preprocessing")
analyzer.update_progress(0.2, quality_estimate=0.75)

analyzer.update_stage("upscaling")
analyzer.update_progress(0.5, quality_estimate=0.82)

analyzer.update_stage("postprocessing")
analyzer.update_progress(0.9, quality_estimate=0.88)

# Finish
summary = analyzer.finish_operation()
print(f"Total time: {summary['total_time']:.2f}s")
```

### Statistics

```python
stats = analyzer.get_statistics()
print(f"Current stage: {stats['current_stage']}")
print(f"Elapsed time: {stats['elapsed_time']:.2f}s")
print(f"Average quality: {stats['avg_quality']:.2f}")
print(f"Average memory: {stats['avg_memory_mb']:.2f} MB")
```

## Smart Recommender

### Intelligent Recommendations

Get smart recommendations based on image analysis and learning.

```python
from image_upscaling_ai.models import SmartRecommender, AdaptiveLearner, AdvancedImageDetector

learner = AdaptiveLearner()
detector = AdvancedImageDetector()
recommender = SmartRecommender(learner=learner, detector=detector)

# Get recommendation
recommendation = recommender.recommend(
    image,
    target_scale=4.0,
    prioritize_speed=False,
    min_quality=0.7
)

print(f"Recommended method: {recommendation.method}")
print(f"Expected quality: {recommendation.expected_quality:.2f}")
print(f"Expected time: {recommendation.expected_time:.2f}s")
print(f"Confidence: {recommendation.confidence:.2f}")
print(f"Preprocessing: {recommendation.preprocessing_mode}")
print(f"Postprocessing: {recommendation.postprocessing_mode}")

print("\nReasoning:")
for reason in recommendation.reasoning:
    print(f"  - {reason}")
```

### Alternative Recommendations

```python
# Get alternatives
alternatives = recommender.get_alternatives(
    image,
    target_scale=4.0,
    current_method="RealESRGAN_x4plus"
)

for alt in alternatives:
    print(f"{alt.method}: quality={alt.expected_quality:.2f}, time={alt.expected_time:.2f}s")
```

## Complete Intelligent Pipeline

### Full Integration

```python
from image_upscaling_ai.models import (
    SmartRecommender,
    AdaptiveLearner,
    AdvancedImageDetector,
    RealtimeAnalyzer,
    AdaptivePreprocessor,
    AdaptivePostprocessor,
    RealESRGANModelManager,
    QualityValidator
)

# Initialize components
learner = AdaptiveLearner()
detector = AdvancedImageDetector()
recommender = SmartRecommender(learner=learner, detector=detector)
analyzer = RealtimeAnalyzer()
preprocessor = AdaptivePreprocessor()
postprocessor = AdaptivePostprocessor()
manager = RealESRGANModelManager()
validator = QualityValidator()

# Get recommendation
recommendation = recommender.recommend(image, 4.0)

# Setup real-time monitoring
def on_progress(metrics):
    print(f"{metrics.stage}: {metrics.progress:.1%}")

analyzer.add_progress_callback(on_progress)
analyzer.start_operation()

# Preprocess
analyzer.update_stage("preprocessing")
preprocessed = preprocessor.preprocess(
    image,
    mode=recommendation.preprocessing_mode
)
analyzer.update_progress(0.3, quality_estimate=0.75)

# Upscale
analyzer.update_stage("upscaling")
upscaled = await manager.upscale_async(
    preprocessed,
    recommendation.scale_factor,
    model_name=recommendation.method
)
analyzer.update_progress(0.7, quality_estimate=0.85)

# Postprocess
analyzer.update_stage("postprocessing")
final = postprocessor.postprocess(
    upscaled,
    original=image,
    mode=recommendation.postprocessing_mode
)
analyzer.update_progress(1.0, quality_estimate=0.90)

# Validate
report = validator.validate(final, image, recommendation.scale_factor)

# Record experience
learner.record_experience(
    image_type=detector.analyze(image).image_type,
    scale_factor=recommendation.scale_factor,
    method_used=recommendation.method,
    quality_score=report.overall_score,
    processing_time=analyzer.finish_operation()["total_time"],
    user_satisfaction=None  # Could be from user feedback
)
```

## Best Practices

### 1. Learning Integration

```python
# Always record experiences
learner.record_experience(
    image_type,
    scale_factor,
    method_used,
    quality_score,
    processing_time,
    user_satisfaction  # If available
)

# Use recommendations
recommendation = recommender.recommend(image, scale_factor)
# Use recommended method and parameters
```

### 2. Real-Time Monitoring

```python
# Setup callbacks early
analyzer = RealtimeAnalyzer()
analyzer.add_progress_callback(update_ui)
analyzer.add_quality_callback(update_quality_display)

# Update throughout operation
analyzer.start_operation()
# ... update stages and progress ...
summary = analyzer.finish_operation()
```

### 3. Smart Recommendations

```python
# Get recommendation before processing
recommendation = recommender.recommend(
    image,
    target_scale,
    prioritize_speed=need_speed,
    min_quality=min_quality
)

# Check if meets requirements
if recommendation.expected_quality < min_quality:
    # Get alternatives
    alternatives = recommender.get_alternatives(image, target_scale, recommendation.method)
    # Choose best alternative
```

## Use Cases

### Use Case 1: Learning System

```python
# Build learning system over time
learner = AdaptiveLearner()

# Process many images
for image_path in image_paths:
    image = Image.open(image_path)
    
    # Get recommendation
    method, _ = learner.recommend_method("photo", 4.0)
    
    # Process
    result = upscale(image, method, 4.0)
    quality = validate(result)
    
    # Learn
    learner.record_experience("photo", 4.0, method, quality, time_taken)

# System improves over time
```

### Use Case 2: Real-Time UI

```python
# For real-time UI updates
analyzer = RealtimeAnalyzer()

def update_ui(metrics):
    ui.update_progress_bar(metrics.progress)
    ui.update_stage_label(metrics.stage)
    if metrics.time_remaining:
        ui.update_time_label(f"{metrics.time_remaining:.1f}s remaining")
    ui.update_quality_indicator(metrics.quality_estimate)

analyzer.add_progress_callback(update_ui)
```

### Use Case 3: Quality Assurance

```python
# Ensure quality before processing
recommender = SmartRecommender()

recommendation = recommender.recommend(image, 4.0, min_quality=0.8)

if recommendation.expected_quality < 0.8:
    # Warn user or adjust parameters
    print("Warning: Expected quality below threshold")
    print("Consider:")
    for alt in recommender.get_alternatives(image, 4.0, recommendation.method):
        if alt.expected_quality >= 0.8:
            print(f"  - {alt.method} (quality: {alt.expected_quality:.2f})")
```

## Summary

The intelligent features provide:
1. **Adaptive Learning**: Improves over time
2. **Real-Time Analysis**: Monitor operations live
3. **Smart Recommendations**: Optimal parameter selection
4. **Quality Prediction**: Know what to expect
5. **Time Estimation**: Plan operations

Use these features for intelligent, adaptive upscaling systems.



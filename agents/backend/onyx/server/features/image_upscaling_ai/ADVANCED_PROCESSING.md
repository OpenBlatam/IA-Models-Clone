# Advanced Processing Guide

## Overview

This guide covers advanced preprocessing, postprocessing, and quality validation features.

## Adaptive Preprocessing

### Features

- **Noise Detection**: Automatically detects and reduces noise
- **Contrast Enhancement**: Improves contrast when needed
- **Sharpness Adjustment**: Enhances sharpness for better upscaling
- **Color Correction**: Auto white balance and color adjustment
- **Adaptive Filtering**: Adjusts based on image characteristics

### Usage

```python
from image_upscaling_ai.models import AdaptivePreprocessor

preprocessor = AdaptivePreprocessor()

# Auto mode (recommended)
preprocessed = preprocessor.preprocess(image, mode="auto")

# Aggressive mode (for low-quality images)
preprocessed = preprocessor.preprocess(image, mode="aggressive")

# Conservative mode (minimal changes)
preprocessed = preprocessor.preprocess(image, mode="conservative")
```

### Modes

- **auto**: Automatically applies enhancements based on analysis
- **aggressive**: Applies all enhancements (for low-quality images)
- **conservative**: Minimal enhancements (for high-quality images)
- **none**: No preprocessing

## Adaptive Postprocessing

### Features

- **Artifact Detection**: Detects and reduces artifacts
- **Edge Enhancement**: Improves edge sharpness
- **Ringing Reduction**: Reduces ringing artifacts
- **Color Consistency**: Maintains color consistency with original

### Usage

```python
from image_upscaling_ai.models import AdaptivePostprocessor

postprocessor = AdaptivePostprocessor()

# Auto mode
postprocessed = postprocessor.postprocess(
    upscaled_image,
    original=original_image,
    mode="auto"
)

# Aggressive mode (for high artifact levels)
postprocessed = postprocessor.postprocess(
    upscaled_image,
    original=original_image,
    mode="aggressive"
)
```

### Analysis

The postprocessor automatically analyzes:
- Artifact level
- Edge sharpness
- Ringing artifacts
- Color consistency

## Quality Validation

### Comprehensive Validation

```python
from image_upscaling_ai.models import QualityValidator

validator = QualityValidator(
    min_score=0.7,  # Minimum acceptable score
    strict_mode=False  # Stricter validation
)

# Validate upscaled image
report = validator.validate(
    upscaled_image,
    original=original_image,
    scale_factor=4.0
)

print(f"Passed: {report.passed}")
print(f"Score: {report.overall_score:.2f}")
print(f"Issues: {report.issues}")
print(f"Recommendations: {report.recommendations}")
```

### Metrics

The validator calculates:
- **Sharpness**: Edge and detail sharpness
- **Artifacts**: Artifact level
- **Noise**: Noise level
- **Contrast**: Image contrast
- **SSIM**: Structural Similarity Index (if original provided)
- **PSNR**: Peak Signal-to-Noise Ratio (if original provided)
- **Color Accuracy**: Color consistency (if original provided)

### Quality Report

```python
report = validator.validate(upscaled, original, 4.0)

# Check if passed
if report.passed:
    print("Quality is acceptable")
else:
    print("Quality issues detected:")
    for issue in report.issues:
        print(f"  - {issue}")
    
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")

# Access metrics
print(f"Sharpness: {report.metrics['sharpness']:.2f}")
print(f"Artifacts: {report.metrics['artifacts']:.2f}")
print(f"SSIM: {report.metrics.get('ssim', 0):.2f}")
```

## Complete Pipeline

### Full Processing Pipeline

```python
from image_upscaling_ai.models import (
    AdaptivePreprocessor,
    AdaptivePostprocessor,
    QualityValidator,
    RealESRGANModelManager
)

# Initialize components
preprocessor = AdaptivePreprocessor()
postprocessor = AdaptivePostprocessor()
validator = QualityValidator(min_score=0.7)
manager = RealESRGANModelManager()

# Step 1: Preprocess
preprocessed = preprocessor.preprocess(image, mode="auto")

# Step 2: Upscale
upscaled = await manager.upscale_async(preprocessed, 4.0)

# Step 3: Postprocess
final = postprocessor.postprocess(
    upscaled,
    original=image,
    mode="auto"
)

# Step 4: Validate
report = validator.validate(final, image, 4.0)

if not report.passed:
    # Apply fixes based on recommendations
    if "Apply artifact reduction" in report.recommendations:
        final = postprocessor.postprocess(final, image, mode="aggressive")
    
    # Re-validate
    report = validator.validate(final, image, 4.0)
```

## Best Practices

### 1. Preprocessing

```python
# For low-quality images
preprocessed = preprocessor.preprocess(image, mode="aggressive")

# For high-quality images
preprocessed = preprocessor.preprocess(image, mode="conservative")

# For unknown quality
preprocessed = preprocessor.preprocess(image, mode="auto")
```

### 2. Postprocessing

```python
# Always provide original for color consistency
postprocessed = postprocessor.postprocess(
    upscaled,
    original=original,
    mode="auto"
)
```

### 3. Quality Validation

```python
# Validate after upscaling
report = validator.validate(upscaled, original, scale_factor)

# Check and fix issues
if not report.passed:
    # Apply recommended fixes
    for rec in report.recommendations:
        if "artifact reduction" in rec.lower():
            upscaled = postprocessor._reduce_artifacts(upscaled)
        elif "edge enhancement" in rec.lower():
            upscaled = postprocessor._enhance_edges(upscaled)
```

### 4. Quality Thresholds

```python
# Strict validation
validator = QualityValidator(min_score=0.8, strict_mode=True)

# Moderate validation
validator = QualityValidator(min_score=0.6, strict_mode=False)

# Lenient validation
validator = QualityValidator(min_score=0.5, strict_mode=False)
```

## Integration Examples

### Example 1: High-Quality Pipeline

```python
# For best quality results
preprocessor = AdaptivePreprocessor()
postprocessor = AdaptivePostprocessor()
manager = RealESRGANModelManager()

# Preprocess
preprocessed = preprocessor.preprocess(image, mode="auto")

# Upscale with best model
upscaled = await manager.upscale_async(
    preprocessed,
    4.0,
    model_name="RealESRGAN_x4plus"
)

# Postprocess
final = postprocessor.postprocess(
    upscaled,
    original=image,
    mode="aggressive"
)

# Validate
validator = QualityValidator(min_score=0.75)
report = validator.validate(final, image, 4.0)
```

### Example 2: Fast Pipeline

```python
# For speed
preprocessor = AdaptivePreprocessor()
postprocessor = AdaptivePostprocessor()

# Minimal preprocessing
preprocessed = preprocessor.preprocess(image, mode="conservative")

# Fast upscaling
upscaled = await manager.upscale_async(
    preprocessed,
    2.0,
    model_name="RealESRGAN_x2plus"
)

# Minimal postprocessing
final = postprocessor.postprocess(upscaled, mode="conservative")
```

### Example 3: Adaptive Pipeline

```python
# Adapt based on image quality
from image_upscaling_ai.models import AdvancedImageDetector

detector = AdvancedImageDetector()
analysis = detector.analyze(image)

# Adjust preprocessing based on quality
if analysis.quality_score < 0.5:
    preprocessed = preprocessor.preprocess(image, mode="aggressive")
else:
    preprocessed = preprocessor.preprocess(image, mode="auto")

# Upscale
upscaled = await manager.upscale_async(
    preprocessed,
    4.0,
    model_name=analysis.recommended_model
)

# Postprocess
final = postprocessor.postprocess(
    upscaled,
    original=image,
    mode="auto"
)

# Validate
report = validator.validate(final, image, 4.0)
```

## Troubleshooting

### Low Quality Scores

```python
# If quality is low, apply fixes
if report.overall_score < 0.6:
    # Aggressive postprocessing
    final = postprocessor.postprocess(
        upscaled,
        original=image,
        mode="aggressive"
    )
    
    # Re-validate
    report = validator.validate(final, image, 4.0)
```

### High Artifact Levels

```python
# Reduce artifacts
if report.metrics["artifacts"] > 0.4:
    final = postprocessor._reduce_artifacts(upscaled, strength=0.7)
```

### Low Sharpness

```python
# Enhance edges
if report.metrics["sharpness"] < 0.5:
    final = postprocessor._enhance_edges(upscaled, strength=0.6)
```

## Performance Considerations

### Preprocessing

- **Auto mode**: ~50-100ms per image
- **Aggressive mode**: ~100-200ms per image
- **Conservative mode**: ~20-50ms per image

### Postprocessing

- **Auto mode**: ~100-200ms per image
- **Aggressive mode**: ~200-400ms per image
- **Conservative mode**: ~50-100ms per image

### Validation

- **Basic validation**: ~50-100ms per image
- **With original comparison**: ~100-200ms per image

## Summary

The advanced processing pipeline provides:
1. **Adaptive Preprocessing**: Improves input quality
2. **Adaptive Postprocessing**: Enhances output quality
3. **Quality Validation**: Ensures acceptable results
4. **Automatic Recommendations**: Suggests improvements

Use these features for production-quality upscaling results.



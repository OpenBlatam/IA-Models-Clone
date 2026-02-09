# Viral Video Editing Guide

## Overview

The enhanced viral video processing system provides advanced video editing capabilities including screen division, transitions, effects, and sophisticated caption generation. This guide covers the complete viral content creation workflow.

## 🎯 Key Features

### **Advanced Video Editing**
- **Screen Division**: Split screen, grid layouts, picture-in-picture
- **Transitions**: Fade, slide, zoom, flip, glitch effects
- **Video Effects**: Slow motion, mirror, sepia, neon, glitch
- **Caption Styling**: Animated text, custom fonts, positioning

### **Viral Optimization**
- **AI-Powered Captions**: Context-aware caption generation
- **Audience Targeting**: Age and interest-based optimization
- **Trend Integration**: Real-time trending topic incorporation
- **Performance Prediction**: Viral score and engagement metrics

## 🏗️ Architecture

### **Core Components**

```
viral_processor.py
├── EnhancedViralVideoProcessor    # Main processor
├── ViralCaptionGenerator          # Caption generation
├── ViralVideoEditor              # Video editing
└── ViralOptimizer                # Optimization engine
```

### **Model Structure**

```
viral_models.py
├── CaptionSegment                # Individual captions
├── ScreenDivision                # Screen layout
├── Transition                    # Transition effects
├── VideoEffect                   # Video effects
├── ViralVideoVariant            # Complete variant
└── ViralVideoBatchResponse      # Batch results
```

## 🚀 Quick Start

### **Basic Viral Processing**

```python
from onyx.server.features.video.processors.viral_processor import create_high_performance_viral_processor
from onyx.server.features.video.models.video_models import VideoClipRequest

# Create processor
processor = create_high_performance_viral_processor()

# Create request
request = VideoClipRequest(
    youtube_url="https://youtube.com/watch?v=example",
    language="en",
    max_clip_length=60
)

# Generate viral variants
result = processor.process_viral(
    request,
    n_variants=5,
    audience_profile={'age': '18-25', 'interests': ['social_media']}
)

# Access results
for variant in result.variants:
    print(f"Viral Score: {variant.viral_score}")
    print(f"Title: {variant.title}")
    print(f"Captions: {len(variant.captions)}")
```

### **Advanced Configuration**

```python
from onyx.server.features.video.processors.viral_processor import ViralProcessingConfig

# Custom configuration
config = ViralProcessingConfig(
    max_captions_per_video=5,
    enable_screen_division=True,
    enable_transitions=True,
    enable_effects=True,
    enable_animations=True,
    parallel_workers=8
)

processor = EnhancedViralVideoProcessor(config)
```

## 🎨 Screen Division

### **Available Layouts**

```python
from onyx.server.features.video.models.viral_models import ScreenDivisionType

# Horizontal split
horizontal_split = create_split_screen_layout(ScreenDivisionType.SPLIT_HORIZONTAL)

# Vertical split
vertical_split = create_split_screen_layout(ScreenDivisionType.SPLIT_VERTICAL)

# 2x2 grid
grid_2x2 = create_split_screen_layout(ScreenDivisionType.GRID_2X2)

# Picture-in-picture
pip = create_split_screen_layout(ScreenDivisionType.PIP)
```

### **Custom Layout Configuration**

```python
from onyx.server.features.video.models.viral_models import ScreenDivision

# Custom screen division
custom_division = ScreenDivision(
    division_type=ScreenDivisionType.CUSTOM,
    sections=[
        {"position": "main", "width": 0.7, "height": 1.0, "content": "video"},
        {"position": "side", "width": 0.3, "height": 0.5, "content": "captions"},
        {"position": "bottom", "width": 0.3, "height": 0.5, "content": "stats"}
    ],
    border_width=3,
    border_color="#FF0000",
    background_color="#000000"
)
```

## 🎬 Transitions

### **Transition Types**

```python
from onyx.server.features.video.models.viral_models import TransitionType

# Available transitions
transitions = [
    TransitionType.FADE,      # Smooth fade
    TransitionType.SLIDE,     # Slide effect
    TransitionType.ZOOM,      # Zoom effect
    TransitionType.FLIP,      # Flip effect
    TransitionType.GLITCH,    # Glitch effect
    TransitionType.ROTATE,    # Rotation
    TransitionType.WIPE,      # Wipe effect
    TransitionType.DISSOLVE,  # Dissolve
    TransitionType.MORPH,     # Morphing
    TransitionType.PIXELATE   # Pixelation
]
```

### **Custom Transition Configuration**

```python
from onyx.server.features.video.models.viral_models import Transition

# Custom transition
transition = Transition(
    transition_type=TransitionType.GLITCH,
    duration=1.5,
    easing="ease_in_out",
    direction="left_to_right",
    intensity=0.8,
    custom_params={
        "glitch_frequency": 0.1,
        "color_shift": True
    }
)
```

## 🎭 Video Effects

### **Available Effects**

```python
from onyx.server.features.video.models.viral_models import VideoEffect as VideoEffectEnum

# Video effects
effects = [
    VideoEffectEnum.SLOW_MOTION,    # Slow motion
    VideoEffectEnum.FAST_FORWARD,   # Fast forward
    VideoEffectEnum.REVERSE,        # Reverse playback
    VideoEffectEnum.MIRROR,         # Mirror effect
    VideoEffectEnum.INVERT,         # Color inversion
    VideoEffectEnum.SEPIA,          # Sepia filter
    VideoEffectEnum.BLACK_AND_WHITE, # B&W filter
    VideoEffectEnum.VINTAGE,        # Vintage look
    VideoEffectEnum.NEON,           # Neon glow
    VideoEffectEnum.GLITCH,         # Glitch effect
    VideoEffectEnum.PIXELATE,       # Pixelation
    VideoEffectEnum.BLUR,           # Blur effect
    VideoEffectEnum.SHARPEN,        # Sharpen
    VideoEffectEnum.SATURATE,       # Increase saturation
    VideoEffectEnum.DESATURATE      # Decrease saturation
]
```

### **Effect Configuration**

```python
from onyx.server.features.video.models.viral_models import VideoEffect

# Custom video effect
effect = VideoEffect(
    effect_type=VideoEffectEnum.NEON,
    intensity=0.7,
    duration=5.0,
    start_time=2.0,
    end_time=7.0,
    custom_params={
        "neon_color": "#00FFFF",
        "glow_radius": 10
    }
)
```

## 📝 Caption Generation

### **Caption Styling**

```python
from onyx.server.features.video.models.viral_models import CaptionStyle, CaptionSegment

# Styled caption
caption = CaptionSegment(
    text="🔥 This is going viral! 🔥",
    start_time=1.0,
    end_time=4.0,
    font_size=32,
    font_color="#FFFFFF",
    background_color="#000000",
    position="bottom",
    styles=[CaptionStyle.BOLD, CaptionStyle.SHADOW, CaptionStyle.GLOW],
    animation="fade_in",
    opacity=0.9,
    scale=1.2,
    rotation=0.0,
    x_offset=0.0,
    y_offset=-20.0
)
```

### **Caption Configuration**

```python
from onyx.server.features.video.models.viral_models import ViralCaptionConfig

# Custom caption configuration
config = ViralCaptionConfig(
    max_caption_length=100,
    caption_duration=3.0,
    font_family="Impact",
    base_font_size=28,
    caption_position="center",
    use_animations=True,
    use_effects=True,
    viral_keywords=["viral", "trending", "must_see"],
    trending_topics=["tech", "entertainment"],
    language="en",
    tone="casual",
    emoji_usage=True
)
```

## 🎯 Audience Targeting

### **Audience Profiles**

```python
# Different audience profiles
audience_profiles = [
    {
        "age": "13-17",
        "interests": ["tiktok", "dance", "music"],
        "language": "en"
    },
    {
        "age": "18-25", 
        "interests": ["social_media", "entertainment", "gaming"],
        "language": "en"
    },
    {
        "age": "26-35",
        "interests": ["tech", "business", "lifestyle"],
        "language": "en"
    },
    {
        "age": "36-50",
        "interests": ["news", "education", "health"],
        "language": "en"
    }
]

# Process for specific audience
result = processor.process_viral(
    request,
    n_variants=3,
    audience_profile=audience_profiles[1]  # 18-25 age group
)
```

### **Multi-Audience Processing**

```python
# Process for multiple audiences
all_results = []
for audience in audience_profiles:
    result = processor.process_viral(
        request,
        n_variants=2,
        audience_profile=audience
    )
    all_results.append((audience, result))

# Find best performing audience
best_audience = max(all_results, key=lambda x: x[1].average_viral_score)
```

## 📊 Performance Optimization

### **Viral Score Calculation**

The system calculates viral scores based on:

1. **Caption Quality** (30%):
   - Text length optimization
   - Styling and animations
   - Viral keywords usage

2. **Visual Appeal** (40%):
   - Screen division usage
   - Transition effects
   - Video effects

3. **Audience Match** (20%):
   - Age targeting
   - Interest alignment
   - Language preference

4. **Trend Alignment** (10%):
   - Trending topics
   - Current events
   - Seasonal relevance

### **Optimization Strategies**

```python
# High-performance configuration
config = ViralProcessingConfig(
    max_captions_per_video=5,
    enable_screen_division=True,
    enable_transitions=True,
    enable_effects=True,
    enable_animations=True,
    parallel_workers=8,
    batch_size=100
)

# Batch processing for efficiency
results = processor.process_batch_parallel(
    requests,
    n_variants=3,
    audience_profile={'age': '18-35'}
)
```

## 🔧 Advanced Usage

### **Custom Caption Generator**

```python
from onyx.server.features.video.processors.viral_processor import ViralCaptionGenerator

class CustomCaptionGenerator(ViralCaptionGenerator):
    def generate_viral_text(self, config: ViralCaptionConfig) -> str:
        # Custom viral text generation logic
        return f"🎯 {config.trending_topics[0].title()} that's {config.viral_keywords[0]}!"
    
    def generate_caption_segments(self, config: ViralCaptionConfig) -> List[CaptionSegment]:
        # Custom caption segment generation
        segments = super().generate_caption_segments(config)
        
        # Add custom styling
        for segment in segments:
            segment.styles.append(CaptionStyle.NEON)
            segment.animation = "bounce"
        
        return segments
```

### **Custom Video Editor**

```python
from onyx.server.features.video.processors.viral_processor import ViralVideoEditor

class CustomVideoEditor(ViralVideoEditor):
    def apply_custom_effect(self, video_path: str, effect_params: Dict) -> str:
        # Custom effect implementation
        logger.info("Applying custom effect", effect_params=effect_params)
        return f"{video_path}_custom_effect"
    
    def create_custom_transition(self, transition_type: str) -> Transition:
        # Custom transition creation
        return Transition(
            transition_type=TransitionType.CUSTOM,
            duration=2.0,
            custom_params={"custom_type": transition_type}
        )
```

### **Custom Viral Optimizer**

```python
from onyx.server.features.video.processors.viral_processor import ViralOptimizer

class CustomViralOptimizer(ViralOptimizer):
    def calculate_custom_score(self, variant: ViralVideoVariant) -> float:
        # Custom viral score calculation
        base_score = super().calculate_viral_score(
            variant.captions,
            variant.screen_division,
            variant.transitions,
            variant.effects,
            None
        )
        
        # Add custom factors
        custom_bonus = 0.1 if "custom_keyword" in variant.title else 0.0
        return min(base_score + custom_bonus, 1.0)
```

## 📈 Monitoring & Analytics

### **Performance Metrics**

```python
# Get processing statistics
result = processor.process_viral(request, n_variants=5)

print(f"Processing time: {result.processing_time:.2f}s")
print(f"Total variants: {result.total_variants_generated}")
print(f"Successful variants: {result.successful_variants}")
print(f"Average viral score: {result.average_viral_score:.3f}")
print(f"Best viral score: {result.best_viral_score:.3f}")
print(f"Caption quality score: {result.caption_quality_score:.3f}")
print(f"Editing quality score: {result.editing_quality_score:.3f}")
```

### **Variant Analysis**

```python
# Analyze individual variants
for variant in result.variants:
    print(f"Variant ID: {variant.variant_id}")
    print(f"Viral Score: {variant.viral_score:.3f}")
    print(f"Engagement Prediction: {variant.engagement_prediction:.3f}")
    print(f"Estimated Views: {variant.estimated_views}")
    print(f"Estimated Likes: {variant.estimated_likes}")
    print(f"Estimated Shares: {variant.estimated_shares}")
    print(f"Target Audience: {variant.target_audience}")
    print(f"Tags: {variant.tags}")
    print(f"Hashtags: {variant.hashtags}")
```

## 🧪 Testing

### **Unit Testing**

```python
import pytest
from onyx.server.features.video.processors.viral_processor import EnhancedViralVideoProcessor

def test_viral_processing():
    processor = EnhancedViralVideoProcessor()
    request = VideoClipRequest(
        youtube_url="https://youtube.com/watch?v=test",
        language="en",
        max_clip_length=60
    )
    
    result = processor.process_viral(request, n_variants=3)
    
    assert result.success is True
    assert len(result.variants) == 3
    assert all(v.viral_score > 0 for v in result.variants)
```

### **Performance Testing**

```python
def test_viral_performance():
    processor = create_high_performance_viral_processor()
    requests = generate_test_requests(10)
    
    start_time = time.perf_counter()
    results = processor.process_batch_parallel(requests, n_variants=3)
    duration = time.perf_counter() - start_time
    
    assert duration < 60.0  # Should complete within 60 seconds
    assert all(r.success for r in results)
```

## 🚨 Error Handling

### **Robust Error Handling**

```python
# Error handling in viral processing
try:
    result = processor.process_viral(request, n_variants=5)
    
    if not result.success:
        print(f"Processing failed: {result.errors}")
        return
    
    # Process successful results
    for variant in result.variants:
        print(f"Viral score: {variant.viral_score}")
        
except Exception as e:
    logger.error("Viral processing error", error=str(e))
    # Handle error appropriately
```

### **Validation**

```python
# Validate viral variants
for variant in result.variants:
    # Check caption quality
    if not variant.captions:
        logger.warning("Variant has no captions", variant_id=variant.variant_id)
    
    # Check viral score
    if variant.viral_score < 0.3:
        logger.warning("Low viral score", variant_id=variant.variant_id, score=variant.viral_score)
    
    # Check timing
    if variant.total_duration <= 0:
        logger.error("Invalid duration", variant_id=variant.variant_id)
```

## 📚 Best Practices

### **1. Caption Optimization**
- Keep captions under 100 characters for mobile viewing
- Use viral keywords strategically
- Include trending hashtags
- Add emojis for engagement

### **2. Visual Effects**
- Use screen division sparingly (not every variant)
- Apply transitions between major content sections
- Use effects to highlight key moments
- Maintain visual consistency

### **3. Audience Targeting**
- Match content style to audience age group
- Use appropriate language and tone
- Include relevant interests and topics
- Consider cultural preferences

### **4. Performance**
- Use batch processing for large datasets
- Optimize parallel workers based on system resources
- Monitor processing times and adjust configuration
- Cache frequently used effects and transitions

### **5. Quality Control**
- Validate all generated variants
- Check for appropriate content and timing
- Monitor viral scores and engagement predictions
- Test with target audiences

## 🔮 Future Enhancements

### **Planned Features**
- **AI-Powered Video Analysis**: Automatic content analysis
- **Real-Time Trend Integration**: Live trending topic updates
- **Advanced Audio Processing**: Music and sound effect integration
- **Multi-Platform Optimization**: Platform-specific formatting
- **A/B Testing Framework**: Automated variant testing

### **Integration Opportunities**
- **Social Media APIs**: Direct posting to platforms
- **Analytics Platforms**: Performance tracking integration
- **Content Management**: Workflow and approval systems
- **Machine Learning**: Continuous optimization models

## 📖 Examples

See the `examples/` directory for complete working examples:

- `viral_editing_examples.py`: Comprehensive viral editing examples
- `parallel_processing_examples.py`: Parallel processing demonstrations
- `performance_benchmark.py`: Performance testing examples

## 🤝 Contributing

When contributing to the viral editing system:

1. Follow the established patterns and conventions
2. Add comprehensive tests for new features
3. Update documentation and examples
4. Ensure backward compatibility
5. Test with real video content
6. Validate performance impact 
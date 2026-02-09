# LangChain Video Optimization Guide

## Overview

This guide demonstrates how to use LangChain integration for intelligent video content analysis and optimization, specifically designed for short-form video platforms like TikTok, Instagram Reels, and YouTube Shorts.

## Features

### 🧠 Intelligent Content Analysis
- **Content Type Classification**: Automatically identifies educational, entertainment, news, tutorial, etc.
- **Sentiment Analysis**: Determines positive, negative, or neutral sentiment
- **Audience Targeting**: Identifies optimal target demographics
- **Trending Analysis**: Detects trending keywords and topics
- **Viral Potential Assessment**: Predicts viral likelihood

### 🎯 Content Optimization
- **Title Optimization**: Generates click-worthy titles
- **Description Enhancement**: Creates engaging descriptions
- **Tag & Hashtag Optimization**: Suggests trending and relevant tags
- **Caption Generation**: Creates viral captions with timing
- **Timing Optimization**: Optimizes clip length and pacing

### 📱 Short-Form Video Specialization
- **Hook Optimization**: First 3 seconds optimization
- **Retention Strategies**: 8-15 second engagement
- **Call-to-Action Timing**: Optimal CTA placement
- **Platform-Specific Optimization**: TikTok, Instagram, YouTube Shorts
- **Vertical Format Optimization**: Mobile-first design

## Quick Start

### 1. Basic Setup

```python
from onyx.server.features.video.processors.langchain_processor import (
    create_langchain_processor,
    LangChainConfig
)

# Create processor with OpenAI API key
processor = create_langchain_processor(
    api_key="your-openai-api-key",
    model_name="gpt-4",
    enable_agents=True
)
```

### 2. Process Video with LangChain

```python
from onyx.server.features.video.models.video_models import VideoClipRequest

# Create video request
request = VideoClipRequest(
    youtube_url="https://www.youtube.com/watch?v=example",
    language="en",
    max_clip_length=30.0,
    target_platform="tiktok"
)

# Process with LangChain optimization
response = processor.process_video_with_langchain(
    request=request,
    n_variants=5,
    audience_profile={
        "age": "18-25",
        "interests": ["entertainment", "comedy", "viral"],
        "platform": "tiktok"
    }
)

# Access results
print(f"Variants generated: {response.successful_variants}")
print(f"Average viral score: {response.average_viral_score:.3f}")
print(f"Best viral score: {response.best_viral_score:.3f}")
```

## Advanced Configuration

### LangChain Configuration

```python
from onyx.server.features.video.processors.langchain_processor import LangChainConfig

config = LangChainConfig(
    # API Configuration
    openai_api_key="your-api-key",
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=2000,
    
    # Analysis Features
    enable_content_analysis=True,
    enable_engagement_analysis=True,
    enable_viral_analysis=True,
    enable_audience_analysis=True,
    
    # Optimization Features
    enable_title_optimization=True,
    enable_description_optimization=True,
    enable_caption_optimization=True,
    enable_timing_optimization=True,
    
    # Performance
    batch_size=10,
    max_retries=3,
    timeout=30.0,
    cache_results=True,
    
    # Advanced Features
    use_agents=True,
    use_memory=True,
    use_streaming=False,
    enable_debug=False
)

processor = LangChainVideoProcessor(config)
```

### Production-Optimized Setup

```python
from onyx.server.features.video.processors.langchain_processor import (
    create_optimized_langchain_processor
)

# Production-ready processor
processor = create_optimized_langchain_processor(
    api_key="your-api-key",
    batch_size=5,
    max_retries=3
)
```

## Content Analysis

### Understanding Analysis Results

```python
# Access LangChain analysis
if response.variants and response.variants[0].langchain_analysis:
    analysis = response.variants[0].langchain_analysis
    
    print(f"Content Type: {analysis.content_type.value}")
    print(f"Sentiment: {analysis.sentiment}")
    print(f"Engagement Score: {analysis.engagement_score:.3f}")
    print(f"Viral Potential: {analysis.viral_potential:.3f}")
    print(f"Target Audience: {', '.join(analysis.target_audience)}")
    print(f"Trending Keywords: {', '.join(analysis.trending_keywords)}")
    print(f"Optimal Duration: {analysis.optimal_duration:.1f}s")
    print(f"Optimal Format: {analysis.optimal_format}")
```

### Content Types

- **EDUCATIONAL**: Tutorials, how-to videos, learning content
- **ENTERTAINMENT**: Comedy, music, fun content
- **NEWS**: Current events, updates, information
- **TUTORIAL**: Step-by-step guides, instructions
- **REVIEW**: Product reviews, recommendations
- **REACTION**: Reaction videos, commentary
- **COMEDY**: Humorous content, jokes
- **MUSIC**: Music videos, performances
- **GAMING**: Game content, gameplay
- **LIFESTYLE**: Daily life, personal content
- **TECH**: Technology, gadgets, software
- **SPORTS**: Sports content, athletic activities

## Content Optimization

### Title Optimization

```python
# Access optimized titles
if response.variants and response.variants[0].content_optimization:
    optimization = response.variants[0].content_optimization
    
    print(f"Optimal Title: {optimization.optimal_title}")
    print(f"Optimal Tags: {optimization.optimal_tags}")
    print(f"Optimal Hashtags: {optimization.optimal_hashtags}")
    print(f"Engagement Hooks: {optimization.engagement_hooks}")
    print(f"Viral Elements: {optimization.viral_elements}")
```

### Caption Optimization

```python
# Access AI-generated captions
variant = response.variants[0]

for caption in variant.captions:
    print(f"Text: {caption.text}")
    print(f"Timing: {caption.start_time}s - {caption.end_time}s")
    print(f"Engagement Score: {caption.engagement_score:.3f}")
    print(f"Viral Potential: {caption.viral_potential:.3f}")
    print(f"Styles: {[style.value for style in caption.styles]}")
```

## Short-Form Video Optimization

### Understanding Short Video Optimization

```python
# Access short video optimization
if response.variants and response.variants[0].short_video_optimization:
    short_opt = response.variants[0].short_video_optimization
    
    print(f"Optimal Clip Length: {short_opt.optimal_clip_length:.1f}s")
    print(f"Hook Duration: {short_opt.hook_duration:.1f}s")
    print(f"Retention Duration: {short_opt.retention_duration:.1f}s")
    print(f"CTA Duration: {short_opt.call_to_action_duration:.1f}s")
    print(f"Hook Type: {short_opt.hook_type}")
    print(f"Vertical Format: {short_opt.vertical_format}")
    print(f"Engagement Triggers: {short_opt.engagement_triggers}")
    print(f"Viral Hooks: {short_opt.viral_hooks}")
    print(f"Emotional Impact: {short_opt.emotional_impact:.3f}")
```

### Hook Types

- **question**: Ask engaging questions
- **statement**: Make bold statements
- **visual**: Use striking visuals
- **audio**: Use attention-grabbing audio

### Call-to-Action Types

- **subscribe**: Encourage subscriptions
- **like**: Ask for likes
- **share**: Encourage sharing
- **comment**: Ask for comments

## Batch Processing

### Process Multiple Videos

```python
# Create multiple requests
requests = [
    VideoClipRequest(
        youtube_url=f"https://www.youtube.com/watch?v=video-{i}",
        language="en",
        max_clip_length=30.0,
        target_platform="tiktok"
    )
    for i in range(5)
]

audience_profiles = [
    {"age": "18-25", "interests": ["comedy"]},
    {"age": "25-35", "interests": ["education"]},
    {"age": "16-24", "interests": ["gaming"]},
    {"age": "25-40", "interests": ["technology"]},
    {"age": "18-30", "interests": ["lifestyle"]}
]

# Process in batch
responses = []
for request, profile in zip(requests, audience_profiles):
    response = processor.process_video_with_langchain(
        request=request,
        n_variants=3,
        audience_profile=profile
    )
    responses.append(response)

# Analyze batch results
successful_responses = [r for r in responses if r.success]
total_variants = sum(r.successful_variants for r in successful_responses)
avg_viral_score = sum(r.average_viral_score for r in successful_responses) / len(successful_responses)

print(f"Batch completed: {len(successful_responses)}/{len(responses)} successful")
print(f"Total variants: {total_variants}")
print(f"Average viral score: {avg_viral_score:.3f}")
```

## Integration with Viral Processor

### Combined Processing

```python
from onyx.server.features.video.processors.viral_processor import (
    create_optimized_viral_processor
)

# Create viral processor with LangChain
viral_processor = create_optimized_viral_processor(
    api_key="your-openai-api-key",
    batch_size=3,
    max_workers=4
)

# Process with both LangChain and viral optimization
response = viral_processor.process_viral_variants(
    request=request,
    n_variants=8,
    audience_profile=audience_profile,
    use_langchain=True  # Enable LangChain optimization
)

# Access enhanced results
print(f"AI Enhancement Score: {response.ai_enhancement_score:.3f}")
print(f"Optimization Insights: {response.optimization_insights}")
```

## Error Handling

### Graceful Fallback

```python
try:
    response = processor.process_video_with_langchain(
        request=request,
        n_variants=5
    )
    
    if response.success:
        print("LangChain processing successful")
    else:
        print(f"Processing failed: {response.errors}")
        
except Exception as e:
    print(f"Error: {str(e)}")
    # Fall back to standard processing
    response = processor._process_standard(request, 5, None)
```

### Common Issues and Solutions

1. **Missing API Key**
   ```python
   # Will automatically fall back to standard processing
   processor = create_langchain_processor(api_key=None)
   ```

2. **Rate Limiting**
   ```python
   # Use retry configuration
   config = LangChainConfig(
       max_retries=5,
       timeout=60.0
   )
   ```

3. **Invalid Content**
   ```python
   # Check response for errors
   if response.errors:
       print(f"Content issues: {response.errors}")
   ```

## Performance Optimization

### Benchmarking

```python
import time

# Benchmark different configurations
configs = [
    ("Basic", create_langchain_processor(api_key="your-key")),
    ("Optimized", create_optimized_langchain_processor(api_key="your-key")),
    ("Viral + LangChain", create_optimized_viral_processor(api_key="your-key"))
]

for name, processor in configs:
    start_time = time.perf_counter()
    response = processor.process_video_with_langchain(request, 3)
    end_time = time.perf_counter()
    
    print(f"{name}: {end_time - start_time:.2f}s, Score: {response.average_viral_score:.3f}")
```

### Best Practices

1. **Batch Processing**: Process multiple videos together
2. **Caching**: Enable result caching for repeated content
3. **Parallel Processing**: Use multiple workers for large batches
4. **Error Recovery**: Implement fallback mechanisms
5. **Monitoring**: Track processing times and success rates

## Advanced Features

### Custom Prompts

```python
# Access LangChain prompts
from onyx.server.features.video.processors.langchain_processor import LangChainPrompts

print(LangChainPrompts.CONTENT_ANALYSIS_PROMPT)
print(LangChainPrompts.SHORT_VIDEO_OPTIMIZATION_PROMPT)
print(LangChainPrompts.CAPTION_GENERATION_PROMPT)
```

### Agent Usage

```python
# Enable LangChain agents for advanced processing
config = LangChainConfig(
    use_agents=True,
    use_memory=True,
    enable_debug=True
)

processor = LangChainVideoProcessor(config)
```

### Memory and Context

```python
# Use conversation memory for context-aware processing
config = LangChainConfig(
    use_memory=True,
    use_agents=True
)

# Process related videos with context
for i, request in enumerate(requests):
    response = processor.process_video_with_langchain(
        request=request,
        n_variants=3,
        audience_profile=profile
    )
    # Memory maintains context between requests
```

## Monitoring and Analytics

### Performance Metrics

```python
# Access detailed metrics
print(f"LangChain Analysis Time: {response.langchain_analysis_time:.2f}s")
print(f"Content Optimization Time: {response.content_optimization_time:.2f}s")
print(f"AI Enhancement Score: {response.ai_enhancement_score:.3f}")
print(f"Total Processing Time: {response.processing_time:.2f}s")
```

### Quality Metrics

```python
# Access quality scores
variant = response.variants[0]

print(f"Viral Score: {variant.viral_score:.3f}")
print(f"Engagement Prediction: {variant.engagement_prediction:.3f}")
print(f"Estimated Views: {variant.estimated_views:,}")
print(f"Estimated Likes: {variant.estimated_likes:,}")
print(f"Estimated Shares: {variant.estimated_shares:,}")
print(f"Estimated Comments: {variant.estimated_comments:,}")
```

## Troubleshooting

### Common Problems

1. **Slow Processing**
   - Reduce batch size
   - Use faster model (gpt-3.5-turbo)
   - Enable caching

2. **Low Viral Scores**
   - Check content type
   - Verify audience profile
   - Review trending keywords

3. **API Errors**
   - Verify API key
   - Check rate limits
   - Enable retries

4. **Memory Issues**
   - Reduce batch size
   - Disable memory for large batches
   - Use streaming for long content

### Debug Mode

```python
# Enable debug mode for detailed logging
config = LangChainConfig(
    enable_debug=True,
    use_agents=True
)

processor = LangChainVideoProcessor(config)
```

## Examples

See `examples/langchain_examples.py` for comprehensive examples including:

- Basic processing
- Content analysis
- Optimization strategies
- Batch processing
- Error handling
- Performance benchmarking
- Integration examples

## API Reference

### LangChainVideoProcessor

```python
class LangChainVideoProcessor:
    def __init__(self, config: LangChainConfig)
    
    def process_video_with_langchain(
        self,
        request: VideoClipRequest,
        n_variants: int = 5,
        audience_profile: Optional[Dict] = None
    ) -> ViralVideoBatchResponse
```

### LangChainConfig

```python
@dataclass
class LangChainConfig:
    openai_api_key: Optional[str] = None
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    enable_content_analysis: bool = True
    enable_engagement_analysis: bool = True
    enable_viral_analysis: bool = True
    batch_size: int = 10
    max_retries: int = 3
    timeout: float = 30.0
    use_agents: bool = True
    use_memory: bool = True
    enable_debug: bool = False
```

## Conclusion

LangChain integration provides powerful AI-driven optimization for short-form video content. By leveraging intelligent content analysis, automated optimization, and platform-specific strategies, you can significantly improve viral potential and engagement rates.

Key benefits:
- 🧠 Intelligent content analysis
- 🎯 Automated optimization
- 📱 Platform-specific strategies
- ⚡ Batch processing capabilities
- 🔄 Graceful error handling
- 📊 Performance monitoring

For more advanced usage and examples, refer to the comprehensive examples in the `examples/` directory. 
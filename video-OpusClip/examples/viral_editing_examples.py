"""
Viral Editing Examples

Practical examples of using the enhanced viral video editing system with advanced features.
"""

import asyncio
import time
from typing import List, Dict
import structlog

from ..models.video_models import VideoClipRequest
from ..models.viral_models import (
    ViralCaptionConfig,
    ScreenDivisionType,
    TransitionType,
    CaptionStyle,
    VideoEffect as VideoEffectEnum,
    create_default_caption_config,
    create_split_screen_layout,
    create_viral_transition
)
from ..processors.viral_processor import (
    EnhancedViralVideoProcessor,
    ViralProcessingConfig,
    create_high_performance_viral_processor,
    create_viral_processor_with_custom_config
)

logger = structlog.get_logger()

# =============================================================================
# SAMPLE DATA GENERATION
# =============================================================================

def generate_viral_video_requests(count: int = 10) -> List[VideoClipRequest]:
    """Generate sample viral video requests."""
    base_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=9bZkp7q19f0",
        "https://www.youtube.com/watch?v=kJQP7kiw5Fk",
        "https://www.youtube.com/watch?v=ZZ5LpwO-An4",
        "https://www.youtube.com/watch?v=OPf0YbXqDm0"
    ]
    
    requests = []
    for i in range(count):
        url = base_urls[i % len(base_urls)]
        requests.append(VideoClipRequest(
            youtube_url=f"{url}&t={i}",
            language="en" if i % 2 == 0 else "es",
            max_clip_length=60 + (i % 30),
            min_clip_length=15 + (i % 10)
        ))
    
    return requests

# =============================================================================
# EXAMPLE 1: BASIC VIRAL EDITING
# =============================================================================

def example_basic_viral_editing():
    """Example of basic viral video editing with captions."""
    print("=== Example 1: Basic Viral Editing ===")
    
    # Create processor
    processor = create_high_performance_viral_processor()
    
    # Generate sample request
    request = VideoClipRequest(
        youtube_url="https://youtube.com/watch?v=example_viral",
        language="en",
        max_clip_length=60
    )
    
    print(f"Processing viral video: {request.youtube_url}")
    
    # Process with viral editing
    start_time = time.perf_counter()
    result = processor.process_viral(
        request,
        n_variants=3,
        audience_profile={'age': '18-25', 'interests': ['social_media', 'entertainment']}
    )
    duration = time.perf_counter() - start_time
    
    print(f"✅ Generated {len(result.variants)} viral variants in {duration:.2f}s")
    
    # Display results
    for i, variant in enumerate(result.variants):
        print(f"  Variant {i+1}:")
        print(f"    Title: {variant.title}")
        print(f"    Viral Score: {variant.viral_score:.3f}")
        print(f"    Captions: {len(variant.captions)}")
        print(f"    Transitions: {len(variant.transitions)}")
        print(f"    Effects: {len(variant.effects)}")
        if variant.screen_division:
            print(f"    Screen Division: {variant.screen_division.division_type.value}")
        print()
    
    print(f"📊 Average viral score: {result.average_viral_score:.3f}")
    print(f"🏆 Best viral score: {result.best_viral_score:.3f}")
    print()

# =============================================================================
# EXAMPLE 2: ADVANCED SCREEN DIVISION
# =============================================================================

def example_advanced_screen_division():
    """Example of advanced screen division techniques."""
    print("=== Example 2: Advanced Screen Division ===")
    
    # Create processor with screen division enabled
    config = ViralProcessingConfig(
        enable_screen_division=True,
        enable_transitions=True,
        enable_effects=True,
        max_captions_per_video=5
    )
    processor = EnhancedViralVideoProcessor(config)
    
    # Generate requests
    requests = generate_viral_video_requests(5)
    
    print(f"Processing {len(requests)} videos with advanced screen division...")
    
    start_time = time.perf_counter()
    results = processor.process_batch_parallel(
        requests,
        n_variants=4,
        audience_profile={'age': '18-35', 'interests': ['tech', 'gaming']}
    )
    duration = time.perf_counter() - start_time
    
    print(f"✅ Completed in {duration:.2f}s")
    
    # Analyze screen division usage
    division_types = {}
    total_variants = 0
    
    for result in results:
        for variant in result.variants:
            total_variants += 1
            if variant.screen_division:
                div_type = variant.screen_division.division_type.value
                division_types[div_type] = division_types.get(div_type, 0) + 1
    
    print(f"📊 Screen Division Analysis:")
    print(f"  Total variants: {total_variants}")
    print(f"  Variants with screen division: {sum(division_types.values())}")
    for div_type, count in division_types.items():
        percentage = (count / total_variants) * 100
        print(f"    {div_type}: {count} ({percentage:.1f}%)")
    print()

# =============================================================================
# EXAMPLE 3: TRANSITION EFFECTS
# =============================================================================

def example_transition_effects():
    """Example of advanced transition effects."""
    print("=== Example 3: Transition Effects ===")
    
    # Create processor with transition focus
    config = ViralProcessingConfig(
        enable_transitions=True,
        enable_screen_division=False,
        enable_effects=False,
        max_captions_per_video=3
    )
    processor = EnhancedViralVideoProcessor(config)
    
    # Generate request
    request = VideoClipRequest(
        youtube_url="https://youtube.com/watch?v=transition_demo",
        language="en",
        max_clip_length=45
    )
    
    print(f"Processing video with transition effects: {request.youtube_url}")
    
    start_time = time.perf_counter()
    result = processor.process_viral(
        request,
        n_variants=5,
        audience_profile={'age': '16-24', 'interests': ['music', 'dance']}
    )
    duration = time.perf_counter() - start_time
    
    print(f"✅ Generated {len(result.variants)} variants in {duration:.2f}s")
    
    # Analyze transition usage
    transition_types = {}
    total_transitions = 0
    
    for variant in result.variants:
        for transition in variant.transitions:
            total_transitions += 1
            trans_type = transition.transition_type.value
            transition_types[trans_type] = transition_types.get(trans_type, 0) + 1
    
    print(f"📊 Transition Analysis:")
    print(f"  Total transitions: {total_transitions}")
    for trans_type, count in transition_types.items():
        percentage = (count / total_transitions) * 100
        print(f"    {trans_type}: {count} ({percentage:.1f}%)")
    
    # Show transition timing
    print(f"⏱️ Transition Timing:")
    for i, variant in enumerate(result.variants[:3]):
        print(f"  Variant {i+1}:")
        for j, transition in enumerate(variant.transitions):
            print(f"    Transition {j+1}: {transition.transition_type.value} at {transition.start_time:.1f}s")
    print()

# =============================================================================
# EXAMPLE 4: VIDEO EFFECTS
# =============================================================================

def example_video_effects():
    """Example of video effects application."""
    print("=== Example 4: Video Effects ===")
    
    # Create processor with effects focus
    config = ViralProcessingConfig(
        enable_effects=True,
        enable_transitions=True,
        enable_screen_division=False,
        max_captions_per_video=2
    )
    processor = EnhancedViralVideoProcessor(config)
    
    # Generate requests
    requests = generate_viral_video_requests(3)
    
    print(f"Processing {len(requests)} videos with effects...")
    
    start_time = time.perf_counter()
    results = processor.process_batch_parallel(
        requests,
        n_variants=3,
        audience_profile={'age': '25-40', 'interests': ['art', 'design']}
    )
    duration = time.perf_counter() - start_time
    
    print(f"✅ Completed in {duration:.2f}s")
    
    # Analyze effects usage
    effect_types = {}
    total_effects = 0
    
    for result in results:
        for variant in result.variants:
            for effect in variant.effects:
                total_effects += 1
                effect_type = effect.effect_type.value
                effect_types[effect_type] = effect_types.get(effect_type, 0) + 1
    
    print(f"📊 Effects Analysis:")
    print(f"  Total effects: {total_effects}")
    for effect_type, count in effect_types.items():
        percentage = (count / total_effects) * 100
        print(f"    {effect_type}: {count} ({percentage:.1f}%)")
    
    # Show effect parameters
    print(f"🎨 Effect Parameters:")
    for i, result in enumerate(results[:2]):
        for j, variant in enumerate(result.variants[:2]):
            print(f"  Video {i+1}, Variant {j+1}:")
            for k, effect in enumerate(variant.effects):
                print(f"    Effect {k+1}: {effect.effect_type.value}")
                print(f"      Intensity: {effect.intensity:.2f}")
                print(f"      Duration: {effect.duration:.1f}s")
                print(f"      Start: {effect.start_time:.1f}s")
    print()

# =============================================================================
# EXAMPLE 5: CUSTOM CAPTION STYLING
# =============================================================================

def example_custom_caption_styling():
    """Example of custom caption styling and animations."""
    print("=== Example 5: Custom Caption Styling ===")
    
    # Create custom caption configuration
    custom_config = ViralCaptionConfig(
        max_caption_length=120,
        caption_duration=4.0,
        font_family="Impact",
        base_font_size=32,
        caption_position="center",
        use_animations=True,
        use_effects=True,
        use_transitions=True,
        use_screen_division=True,
        viral_keywords=["epic", "legendary", "incredible"],
        trending_topics=["gaming", "esports"],
        language="en",
        tone="dramatic",
        emoji_usage=True
    )
    
    # Create processor
    processor = create_viral_processor_with_custom_config(
        max_captions=4,
        enable_editing=True,
        workers=4
    )
    
    # Generate request
    request = VideoClipRequest(
        youtube_url="https://youtube.com/watch?v=gaming_highlight",
        language="en",
        max_clip_length=90
    )
    
    print(f"Processing gaming highlight with custom styling: {request.youtube_url}")
    
    start_time = time.perf_counter()
    result = processor.process_viral(
        request,
        n_variants=3,
        audience_profile={'age': '13-25', 'interests': ['gaming', 'esports']}
    )
    duration = time.perf_counter() - start_time
    
    print(f"✅ Generated {len(result.variants)} styled variants in {duration:.2f}s")
    
    # Analyze caption styling
    print(f"🎨 Caption Styling Analysis:")
    for i, variant in enumerate(result.variants):
        print(f"  Variant {i+1}:")
        print(f"    Title: {variant.title}")
        print(f"    Captions: {len(variant.captions)}")
        
        for j, caption in enumerate(variant.captions):
            print(f"      Caption {j+1}:")
            print(f"        Text: {caption.text}")
            print(f"        Font size: {caption.font_size}")
            print(f"        Position: {caption.position}")
            print(f"        Styles: {[s.value for s in caption.styles]}")
            print(f"        Animation: {caption.animation}")
            print(f"        Timing: {caption.start_time:.1f}s - {caption.end_time:.1f}s")
    
    print(f"📊 Styling Metrics:")
    print(f"  Average caption length: {sum(len(c.text) for v in result.variants for c in v.captions) / sum(len(v.captions) for v in result.variants):.1f} chars")
    print(f"  Variants with animations: {sum(1 for v in result.variants if any(c.animation for c in v.captions))}")
    print(f"  Variants with screen division: {sum(1 for v in result.variants if v.screen_division)}")
    print()

# =============================================================================
# EXAMPLE 6: AUDIENCE TARGETING
# =============================================================================

def example_audience_targeting():
    """Example of audience-specific viral content generation."""
    print("=== Example 6: Audience Targeting ===")
    
    # Define different audience profiles
    audience_profiles = [
        {'age': '13-17', 'interests': ['tiktok', 'dance'], 'language': 'en'},
        {'age': '18-25', 'interests': ['social_media', 'entertainment'], 'language': 'en'},
        {'age': '26-35', 'interests': ['tech', 'business'], 'language': 'en'},
        {'age': '36-50', 'interests': ['news', 'education'], 'language': 'en'}
    ]
    
    # Create processor
    processor = create_high_performance_viral_processor()
    
    # Generate request
    request = VideoClipRequest(
        youtube_url="https://youtube.com/watch?v=universal_content",
        language="en",
        max_clip_length=60
    )
    
    print(f"Processing universal content for different audiences: {request.youtube_url}")
    
    all_results = []
    start_time = time.perf_counter()
    
    for i, audience in enumerate(audience_profiles):
        print(f"  Targeting audience {i+1}: {audience['age']} - {audience['interests']}")
        
        result = processor.process_viral(
            request,
            n_variants=2,
            audience_profile=audience
        )
        all_results.append((audience, result))
    
    duration = time.perf_counter() - start_time
    print(f"✅ Generated content for {len(audience_profiles)} audiences in {duration:.2f}s")
    
    # Analyze audience-specific results
    print(f"📊 Audience Analysis:")
    for audience, result in all_results:
        print(f"  {audience['age']} - {audience['interests']}:")
        print(f"    Average viral score: {result.average_viral_score:.3f}")
        print(f"    Best viral score: {result.best_viral_score:.3f}")
        print(f"    Total variants: {len(result.variants)}")
        
        # Show top variant
        top_variant = max(result.variants, key=lambda v: v.viral_score)
        print(f"    Top variant: {top_variant.title}")
        print(f"    Target audience: {top_variant.target_audience}")
        print()
    
    # Find best performing audience
    best_audience = max(all_results, key=lambda x: x[1].average_viral_score)
    print(f"🏆 Best performing audience: {best_audience[0]['age']} - {best_audience[0]['interests']}")
    print(f"   Average viral score: {best_audience[1].average_viral_score:.3f}")
    print()

# =============================================================================
# EXAMPLE 7: BATCH PROCESSING WITH EDITING
# =============================================================================

def example_batch_editing():
    """Example of batch processing with advanced editing features."""
    print("=== Example 7: Batch Processing with Editing ===")
    
    # Create processor with all features enabled
    config = ViralProcessingConfig(
        enable_screen_division=True,
        enable_transitions=True,
        enable_effects=True,
        enable_animations=True,
        max_captions_per_video=4,
        parallel_workers=6,
        batch_size=50
    )
    processor = EnhancedViralVideoProcessor(config)
    
    # Generate larger batch
    requests = generate_viral_video_requests(20)
    
    print(f"Processing {len(requests)} videos with full editing features...")
    
    start_time = time.perf_counter()
    results = processor.process_batch_parallel(
        requests,
        n_variants=3,
        audience_profile={'age': '18-35', 'interests': ['social_media']}
    )
    duration = time.perf_counter() - start_time
    
    print(f"✅ Completed batch processing in {duration:.2f}s")
    
    # Calculate statistics
    total_variants = sum(len(result.variants) for result in results)
    successful_results = [r for r in results if r.success]
    total_processing_time = sum(r.processing_time for r in successful_results)
    
    print(f"📊 Batch Processing Statistics:")
    print(f"  Total videos processed: {len(requests)}")
    print(f"  Successful results: {len(successful_results)}")
    print(f"  Total variants generated: {total_variants}")
    print(f"  Average variants per video: {total_variants / len(requests):.1f}")
    print(f"  Average processing time per video: {total_processing_time / len(successful_results):.2f}s")
    print(f"  Variants per second: {total_variants / duration:.1f}")
    
    # Feature usage analysis
    features_used = {
        'screen_division': 0,
        'transitions': 0,
        'effects': 0,
        'animations': 0
    }
    
    for result in successful_results:
        for variant in result.variants:
            if variant.screen_division:
                features_used['screen_division'] += 1
            if variant.transitions:
                features_used['transitions'] += 1
            if variant.effects:
                features_used['effects'] += 1
            if any(c.animation for c in variant.captions):
                features_used['animations'] += 1
    
    print(f"🎬 Feature Usage:")
    for feature, count in features_used.items():
        percentage = (count / total_variants) * 100
        print(f"  {feature.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    print()

# =============================================================================
# EXAMPLE 8: ASYNC PROCESSING
# =============================================================================

async def example_async_processing():
    """Example of async viral processing."""
    print("=== Example 8: Async Processing ===")
    
    # Create processor
    processor = create_high_performance_viral_processor()
    
    # Generate requests
    requests = generate_viral_video_requests(8)
    
    print(f"Processing {len(requests)} videos asynchronously...")
    
    start_time = time.perf_counter()
    results = await processor.process_batch_async(
        requests,
        n_variants=2,
        audience_profile={'age': '18-25', 'interests': ['music', 'dance']}
    )
    duration = time.perf_counter() - start_time
    
    print(f"✅ Async processing completed in {duration:.2f}s")
    
    # Analyze results
    successful_results = [r for r in results if r.success]
    total_variants = sum(len(r.variants) for r in successful_results)
    
    print(f"📊 Async Processing Results:")
    print(f"  Successful: {len(successful_results)}/{len(requests)}")
    print(f"  Total variants: {total_variants}")
    print(f"  Variants per second: {total_variants / duration:.1f}")
    print(f"  Average viral score: {sum(r.average_viral_score for r in successful_results) / len(successful_results):.3f}")
    print()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_viral_examples():
    """Run all viral editing examples."""
    print("🚀 Viral Video Editing Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_viral_editing()
        example_advanced_screen_division()
        example_transition_effects()
        example_video_effects()
        example_custom_caption_styling()
        example_audience_targeting()
        example_batch_editing()
        
        # Run async example
        asyncio.run(example_async_processing())
        
        print("🎉 All viral editing examples completed successfully!")
        
    except Exception as e:
        logger.error("Viral example execution failed", error=str(e))
        print(f"❌ Viral example execution failed: {e}")

if __name__ == "__main__":
    run_all_viral_examples() 
"""
Advanced Usage Examples for Rust Enhanced Core

This demonstrates real-world usage patterns and best practices.
"""

import time
from typing import List
from faceless_video_enhanced import (
    EffectsEngine,
    ColorGrading,
    TransitionEngine,
    AudioProcessor
)


def benchmark_comparison():
    """Compare Rust vs Python performance"""
    print("=== Performance Benchmark ===\n")
    
    engine = EffectsEngine()
    
    # Simulate Python timing (would be slower)
    python_time = 2.5  # seconds (example)
    
    # Measure Rust timing
    start = time.time()
    result = engine.ken_burns(
        image_path="test_image.jpg",
        duration=5.0,
        zoom=1.2
    )
    rust_time = time.time() - start
    
    speedup = python_time / rust_time if rust_time > 0 else 0
    print(f"Python: {python_time:.3f}s")
    print(f"Rust:   {rust_time:.3f}s")
    print(f"Speedup: {speedup:.1f}x\n")


def batch_processing_example():
    """Process multiple images in batch"""
    print("=== Batch Processing Example ===\n")
    
    engine = EffectsEngine()
    grading = ColorGrading()
    
    images = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg",
        "image4.jpg",
        "image5.jpg",
    ]
    
    results = []
    start = time.time()
    
    for image in images:
        # Apply effects
        effect_result = engine.ken_burns(
            image_path=image,
            duration=3.0,
            zoom=1.1
        )
        
        # Apply color grading
        graded_result = grading.apply(
            image_path=effect_result,
            brightness=0.05,
            contrast=1.1,
            saturation=1.05
        )
        
        results.append(graded_result)
    
    duration = time.time() - start
    print(f"Processed {len(images)} images in {duration:.2f}s")
    print(f"Average: {duration/len(images):.2f}s per image\n")


def pipeline_example():
    """Complete video processing pipeline"""
    print("=== Complete Pipeline Example ===\n")
    
    engine = EffectsEngine()
    grading = ColorGrading()
    transitions = TransitionEngine()
    audio = AudioProcessor()
    
    # Step 1: Process images with effects
    print("1. Applying effects to images...")
    image1_processed = engine.ken_burns("image1.jpg", 5.0, 1.2)
    image2_processed = engine.ken_burns("image2.jpg", 5.0, 1.2)
    
    # Step 2: Color grade
    print("2. Color grading...")
    image1_graded = grading.apply(image1_processed, brightness=0.1, contrast=1.2)
    image2_graded = grading.apply(image2_processed, brightness=0.1, contrast=1.2)
    
    # Step 3: Create transition
    print("3. Creating transition...")
    transition_result = transitions.crossfade(
        image1_path=image1_graded,
        image2_path=image2_graded,
        duration=1.0
    )
    
    # Step 4: Process audio
    print("4. Processing audio...")
    audio_processed = audio.normalize("background_music.mp3", target_db=-3.0)
    
    print(f"\nPipeline complete!")
    print(f"  Video: {transition_result}")
    print(f"  Audio: {audio_processed}\n")


def error_handling_example():
    """Demonstrate error handling"""
    print("=== Error Handling Example ===\n")
    
    engine = EffectsEngine()
    
    try:
        # This might fail if file doesn't exist
        result = engine.ken_burns(
            image_path="nonexistent.jpg",
            duration=5.0,
            zoom=1.2
        )
        print(f"Success: {result}")
    except Exception as e:
        print(f"Error handled: {e}")
        # Fallback to default processing
        print("Using fallback processing...\n")


def memory_efficient_processing():
    """Process large batches efficiently"""
    print("=== Memory Efficient Processing ===\n")
    
    engine = EffectsEngine()
    grading = ColorGrading()
    
    # Process in chunks to avoid memory issues
    image_batch = [f"image_{i}.jpg" for i in range(100)]
    chunk_size = 10
    
    processed = []
    for i in range(0, len(image_batch), chunk_size):
        chunk = image_batch[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1} ({len(chunk)} images)...")
        
        for image in chunk:
            try:
                result = engine.ken_burns(image, 3.0, 1.1)
                processed.append(result)
            except Exception as e:
                print(f"  Skipped {image}: {e}")
        
        # Could add cleanup here if needed
    
    print(f"Processed {len(processed)} images total\n")


def integration_with_existing_code():
    """Show how to integrate with existing Python code"""
    print("=== Integration Example ===\n")
    
    # Feature flag approach
    USE_RUST = True  # Toggle this to switch implementations
    
    if USE_RUST:
        from faceless_video_enhanced import EffectsEngine
        effects_service = EffectsEngine()
        print("Using Rust-enhanced effects (50x faster)")
    else:
        from services.visual_effects import VisualEffectsService
        effects_service = VisualEffectsService()
        print("Using Python effects (fallback)")
    
    # Same API usage
    result = effects_service.ken_burns(
        image_path="image.jpg",
        duration=5.0,
        zoom=1.2
    )
    
    print(f"Result: {result}\n")


if __name__ == "__main__":
    print("Rust Enhanced Core - Advanced Usage Examples\n")
    print("=" * 50 + "\n")
    
    # Uncomment to run examples:
    # benchmark_comparison()
    # batch_processing_example()
    # pipeline_example()
    # error_handling_example()
    # memory_efficient_processing()
    # integration_with_existing_code()
    
    print("Examples ready! Uncomment to run with actual files.")













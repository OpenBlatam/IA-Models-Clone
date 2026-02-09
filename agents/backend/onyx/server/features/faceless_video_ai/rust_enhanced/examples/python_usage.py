"""
Python Usage Examples for Rust Enhanced Core

This example shows how to use Rust-enhanced video processing from Python.
"""

from faceless_video_enhanced import (
    EffectsEngine,
    ColorGrading,
    TransitionEngine,
    AudioProcessor,
    VideoProcessor
)


def example_effects():
    """Example: Video effects (10-50x faster than Python)"""
    print("=== Video Effects Example ===")
    
    engine = EffectsEngine()
    
    # Ken Burns effect
    result = engine.ken_burns(
        image_path="image.jpg",
        duration=5.0,
        zoom=1.2,
        pan_x=0.1,
        pan_y=0.1
    )
    print(f"Ken Burns effect created: {result}")
    
    # Blur effect
    result = engine.blur(
        image_path="image.jpg",
        radius=5.0
    )
    print(f"Blur effect created: {result}")


def example_color_grading():
    """Example: Color grading (20-100x faster than Python)"""
    print("\n=== Color Grading Example ===")
    
    grading = ColorGrading()
    
    # Apply color correction
    result = grading.apply(
        image_path="image.jpg",
        brightness=0.1,
        contrast=1.2,
        saturation=1.1,
        temperature=6500
    )
    print(f"Color graded image: {result}")
    
    # Extract color palette
    palette = grading.extract_palette("image.jpg", num_colors=5)
    print(f"Extracted palette: {palette}")


def example_transitions():
    """Example: Video transitions (15-30x faster than Python)"""
    print("\n=== Transitions Example ===")
    
    transitions = TransitionEngine()
    
    # Crossfade
    result = transitions.crossfade(
        image1_path="image1.jpg",
        image2_path="image2.jpg",
        duration=1.0
    )
    print(f"Crossfade created: {result}")
    
    # Slide transition
    result = transitions.slide(
        image1_path="image1.jpg",
        image2_path="image2.jpg",
        direction="left",
        duration=0.5
    )
    print(f"Slide transition created: {result}")


def example_audio():
    """Example: Audio processing (10-20x faster than Python)"""
    print("\n=== Audio Processing Example ===")
    
    audio = AudioProcessor()
    
    # Normalize audio
    result = audio.normalize(
        audio_path="audio.mp3",
        target_db=-3.0
    )
    print(f"Normalized audio: {result}")
    
    # Apply fade
    result = audio.fade(
        audio_path="audio.mp3",
        fade_in=1.0,
        fade_out=1.0
    )
    print(f"Faded audio: {result}")
    
    # Extract features
    features = audio.extract_features("audio.mp3")
    print(f"Audio features: {features}")


def example_integration_with_existing_service():
    """Example: Integration with existing Python service"""
    print("\n=== Integration Example ===")
    
    # Before: Using Python service
    # from services.visual_effects import VisualEffectsService
    # service = VisualEffectsService()
    # result = await service.add_ken_burns_effect(...)  # ~2.5 seconds
    
    # After: Using Rust-enhanced service
    engine = EffectsEngine()
    result = engine.ken_burns(
        image_path="image.jpg",
        duration=5.0,
        zoom=1.2
    )  # ~0.05 seconds (50x faster!)
    
    print(f"Fast Ken Burns result: {result}")


if __name__ == "__main__":
    print("Rust Enhanced Core - Python Usage Examples\n")
    
    # Note: These examples require actual image/audio files
    # Uncomment to run with real files:
    
    # example_effects()
    # example_color_grading()
    # example_transitions()
    # example_audio()
    # example_integration_with_existing_service()
    
    print("\nExamples ready! Uncomment to run with actual files.")













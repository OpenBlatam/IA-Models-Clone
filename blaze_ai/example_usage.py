"""
Example usage of the refactored Blaze AI module.

This script demonstrates how to use the various features of the
Blaze AI module including text generation, image generation, and more.
"""

import asyncio
import json
from pathlib import Path

from . import create_modular_ai, get_logger
from .core.interfaces import CoreConfig


async def example_text_generation(ai):
    """Example of text generation."""
    print("\n=== Text Generation Example ===")
    
    try:
        result = await ai.generate_text(
            prompt="Write a short blog post about artificial intelligence",
            max_length=200,
            temperature=0.7
        )
        
        print(f"Generated text: {result['text']}")
        print(f"Generation time: {result.get('generation_time', 'N/A')}s")
        print(f"Tokens used: {result.get('tokens_used', 'N/A')}")
        
    except Exception as e:
        print(f"Text generation failed: {e}")


async def example_image_generation(ai):
    """Example of image generation."""
    print("\n=== Image Generation Example ===")
    
    try:
        result = await ai.generate_image(
            prompt="A beautiful sunset over mountains, digital art",
            width=512,
            height=512,
            guidance_scale=7.5,
            num_inference_steps=30
        )
        
        print(f"Generated {len(result['images'])} image(s)")
        print(f"Generation time: {result.get('generation_time', 'N/A')}s")
        print(f"Model info: {result.get('model_info', 'N/A')}")
        
        # Save the first image
        if result['images']:
            import base64
            from PIL import Image
            import io
            
            img_data = base64.b64decode(result['images'][0])
            img = Image.open(io.BytesIO(img_data))
            img.save("generated_image.png")
            print("Image saved as 'generated_image.png'")
        
    except Exception as e:
        print(f"Image generation failed: {e}")


async def example_seo_analysis(ai):
    """Example of SEO analysis."""
    print("\n=== SEO Analysis Example ===")
    
    content = """
    Artificial Intelligence (AI) is transforming the way we live and work. 
    From virtual assistants to autonomous vehicles, AI technologies are becoming 
    increasingly prevalent in our daily lives. Machine learning algorithms can 
    now process vast amounts of data to identify patterns and make predictions 
    with remarkable accuracy.
    """
    
    try:
        result = await ai.analyze_seo(
            content=content,
            title="The Future of Artificial Intelligence",
            target_keywords=["artificial intelligence", "AI", "machine learning"]
        )
        
        print(f"SEO Score: {result.get('seo_score', 'N/A')}")
        print(f"Readability Score: {result.get('readability_score', 'N/A')}")
        print(f"Word Count: {result.get('word_count', 'N/A')}")
        print(f"Keywords: {[kw.get('keyword', '') for kw in result.get('keywords', [])]}")
        print(f"Suggestions: {result.get('suggestions', [])}")
        
    except Exception as e:
        print(f"SEO analysis failed: {e}")


async def example_brand_voice(ai):
    """Example of brand voice analysis."""
    print("\n=== Brand Voice Analysis Example ===")
    
    content = """
    Hey there! We're super excited to share some amazing news with you. 
    Our team has been working hard to bring you the best possible experience, 
    and we think you're going to love what we've built.
    """
    
    try:
        result = await ai.apply_brand_voice(
            content=content,
            brand_name="TechCorp",
            action="analyze"
        )
        
        print(f"Brand Voice Score: {result.get('brand_voice_score', 'N/A')}")
        print(f"Voice Characteristics: {result.get('voice_characteristics', 'N/A')}")
        print(f"Suggestions: {result.get('suggestions', [])}")
        
    except Exception as e:
        print(f"Brand voice analysis failed: {e}")


async def example_content_generation(ai):
    """Example of content generation."""
    print("\n=== Content Generation Example ===")
    
    try:
        result = await ai.generate_content(
            content_type="blog",
            topic="The Benefits of Remote Work",
            tone="professional",
            word_count=300
        )
        
        print(f"Generated content: {result['content']}")
        print(f"Word count: {result.get('word_count', 'N/A')}")
        
    except Exception as e:
        print(f"Content generation failed: {e}")


async def example_health_check(ai):
    """Example of health check."""
    print("\n=== Health Check Example ===")
    
    try:
        health = await ai.get_health_status()
        print(f"Overall status: {health.get('overall_status', 'N/A')}")
        print(f"Components: {json.dumps(health.get('components', {}), indent=2)}")
        
    except Exception as e:
        print(f"Health check failed: {e}")


async def main():
    """Main example function."""
    print("Blaze AI Module - Example Usage")
    print("=" * 50)
    
    # Create a simple configuration
    config = CoreConfig(
        system_mode="development",
        log_level="INFO"
    )
    
    # Initialize the AI module
    print("Initializing Blaze AI module...")
    ai = create_modular_ai(config=config)
    
    try:
        # Run examples
        await example_health_check(ai)
        await example_text_generation(ai)
        await example_seo_analysis(ai)
        await example_brand_voice(ai)
        await example_content_generation(ai)
        
        # Note: Image generation requires more setup (models, etc.)
        # Uncomment the line below if you have the required models
        # await example_image_generation(ai)
        
    except Exception as e:
        print(f"Example failed: {e}")
    
    finally:
        # Cleanup
        print("\nShutting down...")
        await ai.shutdown()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())

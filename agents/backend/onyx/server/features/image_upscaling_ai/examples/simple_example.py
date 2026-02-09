"""
Simple Example
==============

Simple example for quick start.
"""

import asyncio
from PIL import Image
from image_upscaling_ai.core.enhanced_service import EnhancedUpscalingService


async def main():
    """Simple upscaling example."""
    # Initialize service
    service = EnhancedUpscalingService()
    
    # Load image
    image = Image.open("input.jpg").convert("RGB")
    
    # Upscale
    result = await service.upscale_image_enhanced(
        image,
        scale_factor=4.0
    )
    
    if result["success"]:
        # Save result
        result["upscaled_image"].save("output.jpg")
        print(f"✅ Upscaled: {result['original_size']} -> {result['upscaled_size']}")
        print(f"Quality: {result['quality_score']:.2f}")
        print(f"Time: {result['processing_time']:.2f}s")
    else:
        print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())



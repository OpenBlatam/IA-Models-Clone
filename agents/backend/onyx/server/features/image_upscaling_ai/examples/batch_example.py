"""
Batch Processing Example
========================

Example of batch processing multiple images.
"""

import asyncio
from pathlib import Path
from PIL import Image
from image_upscaling_ai.models import BatchOptimizer, RealESRGANModelManager


async def main():
    """Batch processing example."""
    # Initialize
    manager = RealESRGANModelManager()
    optimizer = BatchOptimizer(initial_batch_size=2, adaptive=True)
    
    # Get image files
    image_dir = Path("images")
    image_files = list(image_dir.glob("*.jpg"))[:10]  # Process first 10
    
    # Load images
    images = [Image.open(f).convert("RGB") for f in image_files]
    
    # Process function
    async def upscale_image(img):
        return await manager.upscale_async(img, 4.0)
    
    # Process batch
    def progress(current, total):
        print(f"Processing: {current}/{total} ({current/total*100:.1f}%)")
    
    result = await optimizer.process_batch_optimized(
        images,
        upscale_image,
        progress_callback=progress
    )
    
    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    for idx, res in enumerate(result.results):
        if res.get("success"):
            output_path = output_dir / f"upscaled_{idx}.jpg"
            res["result"].save(output_path)
    
    # Print summary
    print(f"\n✅ Batch Complete!")
    print(f"Total: {result.total_items}")
    print(f"Successful: {result.successful}")
    print(f"Failed: {result.failed}")
    print(f"Total time: {result.total_time:.2f}s")
    print(f"Avg time/item: {result.avg_time_per_item:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())



"""
Example Usage - Character Clothing Changer AI
=============================================

Example script showing how to use the clothing changer service.
"""

from pathlib import Path
from character_clothing_changer_ai.core.clothing_changer_service import ClothingChangerService
from character_clothing_changer_ai.config.clothing_changer_config import ClothingChangerConfig

def main():
    """Example usage."""
    # Initialize service
    print("Initializing Clothing Changer Service...")
    service = ClothingChangerService()
    
    # Initialize model (this will download models on first run)
    print("Initializing model (this may take a while on first run)...")
    service.initialize_model()
    
    # Example: Change clothing
    print("\nExample 1: Change clothing in character image")
    print("=" * 50)
    
    # Replace with your image path
    image_path = "path/to/your/character.jpg"
    
    if Path(image_path).exists():
        result = service.change_clothing(
            image=image_path,
            clothing_description="a red elegant dress with golden details",
            character_name="MyCharacter",
            save_tensor=True,
        )
        
        print(f"✓ Clothing changed successfully!")
        print(f"  - Character: {result.get('character_name')}")
        print(f"  - Clothing: {result.get('clothing_description')}")
        print(f"  - Safe tensor saved: {result.get('saved_path')}")
    else:
        print(f"⚠ Image not found: {image_path}")
        print("  Please update the image_path variable with a valid image path")
    
    # Example: List generated tensors
    print("\nExample 2: List generated tensors")
    print("=" * 50)
    
    tensors = service.list_tensors()
    print(f"Found {len(tensors)} generated tensors:")
    for tensor in tensors:
        print(f"  - {tensor['filename']}")
        if tensor.get('metadata'):
            print(f"    Character: {tensor['metadata'].get('character_name', 'unknown')}")
            print(f"    Clothing: {tensor['metadata'].get('clothing_description', 'unknown')}")
    
    # Example: Create ComfyUI workflow
    if tensors:
        print("\nExample 3: Create ComfyUI workflow")
        print("=" * 50)
        
        tensor_path = tensors[0]['path']
        workflow_result = service.create_comfyui_workflow(
            tensor_path=tensor_path,
            prompt="a character wearing elegant clothing, high quality, detailed",
            negative_prompt="blurry, low quality, distorted",
        )
        
        print(f"✓ Workflow created: {workflow_result['workflow_path']}")
        print("  You can now import this workflow into ComfyUI")
    
    # Get model info
    print("\nModel Information")
    print("=" * 50)
    model_info = service.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Clean up
    print("\nCleaning up...")
    service.close()
    print("✓ Done!")

if __name__ == "__main__":
    main()



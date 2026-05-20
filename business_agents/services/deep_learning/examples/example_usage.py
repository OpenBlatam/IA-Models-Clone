"""
Example Usage - Complete Examples
==================================

Complete examples demonstrating how to use the deep learning service.
"""

import numpy as np
from pathlib import Path

# Import service
from ..service import DeepLearningService
from ..config.config_loader import ConfigLoader
from ..data import create_dataloader, split_dataset, SimpleDataset
from ..utils.helpers import set_seed, get_device


def example_basic_training():
    """Example: Basic model training."""
    print("=" * 60)
    print("Example 1: Basic Model Training")
    print("=" * 60)
    
    # Initialize service
    service = DeepLearningService()
    
    # Create model
    model = service.create_model("transformer", model_id="example_model")
    
    # Create synthetic data
    data = np.random.randn(1000, 512).astype(np.float32)
    labels = np.random.randint(0, 2, 1000).astype(np.int64)
    
    # Create dataset and split
    dataset = service.create_dataset(data, labels, dataset_type="simple")
    train_ds, val_ds, test_ds = split_dataset(dataset, 0.7, 0.15, 0.15)
    
    # Create dataloaders
    train_loader = create_dataloader(train_ds, batch_size=32, num_workers=2)
    val_loader = create_dataloader(val_ds, batch_size=32, num_workers=2)
    
    # Train model
    history = service.train_model(model, train_loader, val_loader, model_id="example_model")
    
    # Evaluate
    metrics = service.evaluate_model(model, val_loader)
    print(f"Validation Metrics: {metrics}")
    
    # Save model
    service.save_model(model, "checkpoints/example_model.pt")
    
    print("✅ Training completed!")


def example_huggingface_model():
    """Example: Using HuggingFace models."""
    print("=" * 60)
    print("Example 2: HuggingFace Model")
    print("=" * 60)
    
    try:
        from ..models.transformers_models import HuggingFaceModel
        
        # Create HuggingFace model
        model = HuggingFaceModel(
            model_name="bert-base-uncased",
            task_type="classification",
            num_labels=2
        )
        
        # Make predictions
        texts = ["This is great!", "This is terrible!"]
        results = model.predict(texts)
        print(f"Predictions: {results}")
        
        print("✅ HuggingFace model example completed!")
        
    except ImportError:
        print("⚠️ Transformers library not available")


def example_diffusion_model():
    """Example: Using diffusion models."""
    print("=" * 60)
    print("Example 3: Diffusion Model")
    print("=" * 60)
    
    try:
        from ..models.diffusion_models import DiffusionModel
        
        # Create diffusion model
        model = DiffusionModel(
            model_name="runwayml/stable-diffusion-v1-5",
            model_type="stable-diffusion"
        )
        
        # Generate image
        result = model.generate(
            prompt="A beautiful sunset over mountains",
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        print(f"Generated {len(result['images'])} image(s)")
        print("✅ Diffusion model example completed!")
        
    except ImportError:
        print("⚠️ Diffusers library not available")


def example_with_config():
    """Example: Using YAML configuration."""
    print("=" * 60)
    print("Example 4: Using YAML Configuration")
    print("=" * 60)
    
    # Load configuration
    config_loader = ConfigLoader("config/default_config.yaml")
    
    # Get configurations
    model_config = config_loader.get_model_config()
    training_config = config_loader.get_training_config()
    data_config = config_loader.get_data_config()
    
    print(f"Model: {model_config.name}")
    print(f"Training epochs: {training_config.epochs}")
    print(f"Batch size: {data_config.batch_size}")
    
    # Initialize service with config
    service = DeepLearningService(config_path="config/default_config.yaml")
    
    print("✅ Configuration example completed!")


def example_gradio_demo():
    """Example: Creating Gradio demo."""
    print("=" * 60)
    print("Example 5: Gradio Demo")
    print("=" * 60)
    
    try:
        import gradio as gr
        from ..gradio_apps.model_demo import create_model_demo
        
        # Create model
        service = DeepLearningService()
        model = service.create_model("transformer")
        
        # Create inference function
        def inference_fn(model, text, device):
            # Simple inference example
            return {"prediction": "Example output", "confidence": 0.95}
        
        # Create demo
        demo = create_model_demo(
            model=model,
            inference_fn=inference_fn,
            device=service.device,
            title="Transformer Demo"
        )
        
        # Launch (commented out - uncomment to run)
        # demo.launch(share=True)
        
        print("✅ Gradio demo created!")
        print("Uncomment demo.launch() to run the interface")
        
    except ImportError:
        print("⚠️ Gradio not available")


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    
    # Run examples
    example_basic_training()
    print("\n")
    
    example_huggingface_model()
    print("\n")
    
    example_diffusion_model()
    print("\n")
    
    example_with_config()
    print("\n")
    
    example_gradio_demo()




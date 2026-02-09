"""
Deployment Script for Addiction Recovery AI
"""

import argparse
import torch
from pathlib import Path
from typing import Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from addiction_recovery_ai import (
    create_ultra_fast_engine,
    ModelRegistry,
    export_to_onnx,
    ModelPruner,
    ConfigManager
)


def deploy_model(
    model_type: str = "progress",
    version: str = "1.0.0",
    optimize: bool = True,
    export_onnx: bool = True,
    config_file: Optional[str] = None
):
    """
    Deploy model for production
    
    Args:
        model_type: Type of model (progress, relapse, sentiment)
        version: Model version
        optimize: Whether to optimize model
        export_onnx: Whether to export to ONNX
        config_file: Optional config file
    """
    print("=" * 60)
    print("Deploying Addiction Recovery AI Model")
    print("=" * 60)
    
    # Load configuration
    if config_file:
        config = ConfigManager(config_file)
    else:
        config = ConfigManager()
    
    print(f"\n1. Creating model...")
    engine = create_ultra_fast_engine(
        use_gpu=config.get("model.use_gpu", True)
    )
    
    # Get model based on type
    if model_type == "progress" and hasattr(engine, 'progress_predictor'):
        model = engine.progress_predictor
    elif model_type == "relapse" and hasattr(engine, 'relapse_predictor'):
        model = engine.relapse_predictor
    else:
        print(f"⚠ Model type {model_type} not available")
        return
    
    print(f"✓ Model created")
    
    # Optimize
    if optimize:
        print(f"\n2. Optimizing model...")
        pruned = ModelPruner.prune_model(model, amount=0.2)
        sparsity = ModelPruner.get_sparsity(pruned)
        print(f"✓ Model pruned: {sparsity * 100:.1f}% sparsity")
        model = pruned
    
    # Export ONNX
    if export_onnx:
        print(f"\n3. Exporting to ONNX...")
        input_shape = (1, 10) if model_type == "progress" else (1, 30, 5)
        onnx_path = f"{model_type}_model_v{version}.onnx"
        
        success = export_to_onnx(
            model=model,
            input_shape=input_shape,
            output_path=onnx_path
        )
        
        if success:
            print(f"✓ Model exported to {onnx_path}")
        else:
            print(f"⚠ ONNX export failed")
    
    # Version management
    print(f"\n4. Registering model version...")
    registry = ModelRegistry()
    
    metadata = {
        "model_type": model_type,
        "optimized": optimize,
        "onnx_exported": export_onnx,
        "sparsity": sparsity if optimize else 0.0
    }
    
    model_path = registry.register(
        model=model,
        version=version,
        metadata=metadata
    )
    
    print(f"✓ Model registered: {model_path}")
    print(f"✓ Version: {version}")
    
    print("\n" + "=" * 60)
    print("Deployment complete!")
    print("=" * 60)


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy Addiction Recovery AI Model")
    parser.add_argument("--model-type", type=str, default="progress", 
                       choices=["progress", "relapse", "sentiment"],
                       help="Type of model to deploy")
    parser.add_argument("--version", type=str, default="1.0.0",
                       help="Model version")
    parser.add_argument("--no-optimize", action="store_true",
                       help="Skip optimization")
    parser.add_argument("--no-onnx", action="store_true",
                       help="Skip ONNX export")
    parser.add_argument("--config", type=str, default=None,
                       help="Config file path")
    
    args = parser.parse_args()
    
    deploy_model(
        model_type=args.model_type,
        version=args.version,
        optimize=not args.no_optimize,
        export_onnx=not args.no_onnx,
        config_file=args.config
    )


if __name__ == "__main__":
    main()


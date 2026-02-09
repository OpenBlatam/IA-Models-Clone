"""
Inference Example
Demonstrates optimized inference
"""

import torch
from ml import ViTSkinAnalyzer
from ml.inference import FastInferenceEngine
from utils.advanced_optimization import enable_all_optimizations
from utils.optimization import compile_model, quantize_model

def main():
    """Optimized inference pipeline"""
    
    # 1. Enable optimizations
    enable_all_optimizations()
    
    # 2. Load model
    model = ViTSkinAnalyzer(
        num_conditions=6,
        num_metrics=8,
        use_pretrained=True
    )
    
    # 3. Optimize model
    model = compile_model(model, mode="reduce-overhead")
    model = quantize_model(model, quantization_type="int8_dynamic")
    
    # 4. Create inference engine
    engine = FastInferenceEngine(
        model=model,
        device="cuda",
        use_compile=False,  # Already compiled
        use_quantization=False,  # Already quantized
        batch_size=1
    )
    
    # 5. Inference
    input_tensor = torch.randn(1, 3, 224, 224)
    output = engine.predict(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print("Inference completed!")

if __name__ == "__main__":
    main()














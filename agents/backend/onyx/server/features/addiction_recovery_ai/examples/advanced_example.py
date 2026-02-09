"""
Advanced Example
Complete example with all features
"""

import torch
import torch.nn as nn
from addiction_recovery_ai import (
    # Models
    create_progress_predictor,
    
    # Optimization
    create_ultra_fast_inference,
    create_integrated_pipeline,
    enable_optimizations,
    
    # Validation
    validate_features,
    validate_input,
    
    # Monitoring
    create_system_monitor,
    create_model_monitor,
    
    # Training
    TrainerFactory,
    create_tracker,
    create_checkpoint_manager,
    
    # Export
    export_to_onnx,
    
    # Security
    compute_model_hash,
    sanitize_input,
    
    # Caching
    create_smart_cache,
    
    # Metrics
    calculate_regression_metrics,
    
    # Helpers
    get_device,
    count_parameters,
    initialize_model
)


def main():
    """Advanced example demonstrating all features"""
    
    print("🚀 Advanced Example - All Features")
    print("=" * 50)
    
    # 1. Enable optimizations
    print("\n1. Enabling Optimizations")
    enable_optimizations()
    
    # 2. Get device
    print("\n2. Device Detection")
    device = get_device()
    print(f"Using device: {device}")
    
    # 3. Create model
    print("\n3. Creating Model")
    model = create_progress_predictor()
    
    # Initialize weights
    initialize_model(model, method="xavier")
    
    # Count parameters
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 4. Compute model hash
    print("\n4. Model Security")
    model_hash = compute_model_hash(model)
    print(f"Model hash: {model_hash[:16]}...")
    
    # 5. Validation
    print("\n5. Input Validation")
    features = [30/365, 0.3, 0.4, 0.7]
    is_valid, error = validate_features(features, expected_length=4)
    if is_valid:
        print("✅ Features validated")
        
        # Sanitize
        features_tensor = torch.tensor([features], dtype=torch.float32)
        clean_input = sanitize_input(features_tensor, max_value=1.0, min_value=0.0)
        
        # Validate tensor
        is_valid, error = validate_input(clean_input, expected_shape=(1, 4))
        if is_valid:
            print("✅ Tensor validated")
    else:
        print(f"❌ Validation failed: {error}")
        return
    
    # 6. Ultra-fast inference
    print("\n6. Ultra-Fast Inference")
    engine = create_ultra_fast_inference(model)
    output = engine.predict(clean_input)
    print(f"Prediction: {output.item():.4f}")
    
    # 7. Integrated pipeline
    print("\n7. Integrated Pipeline")
    pipeline = create_integrated_pipeline(
        model,
        enable_validation=True,
        enable_monitoring=True,
        enable_optimization=True
    )
    
    pipeline_output = pipeline.predict(clean_input)
    print(f"Pipeline output: {pipeline_output.item():.4f}")
    
    # Check health
    health = pipeline.get_health_status()
    print(f"Pipeline health: {health['model_health']['status']}")
    
    # 8. Monitoring
    print("\n8. Monitoring")
    system_monitor = create_system_monitor()
    system_health = system_monitor.get_health_status()
    print(f"System status: {system_health['status']}")
    
    model_monitor = create_model_monitor(model)
    model_monitor.record_inference(10.5, success=True)
    model_health = model_monitor.check_model_health()
    print(f"Model health: {model_health['status']}")
    
    # 9. Caching
    print("\n9. Smart Caching")
    cache = create_smart_cache(max_size=100, ttl_seconds=3600)
    cache.put("features_1", features)
    cached_value = cache.get("features_1")
    print(f"Cached value retrieved: {cached_value is not None}")
    
    # 10. Metrics
    print("\n10. Metrics Calculation")
    y_true = [0.7, 0.8, 0.6, 0.9]
    y_pred = [0.72, 0.78, 0.65, 0.88]
    metrics = calculate_regression_metrics(y_true, y_pred)
    print(f"R²: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    # 11. Export (commented out - requires proper setup)
    # print("\n11. Model Export")
    # export_to_onnx(model, input_shape=(1, 4), output_path="model.onnx")
    # print("✅ Model exported to ONNX")
    
    print("\n✅ Advanced Example Complete!")


if __name__ == "__main__":
    main()

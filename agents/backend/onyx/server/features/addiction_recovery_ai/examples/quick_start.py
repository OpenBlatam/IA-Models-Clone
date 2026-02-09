"""
Quick Start Example
Simple example to get started quickly
"""

import torch
from addiction_recovery_ai import (
    create_sentiment_analyzer,
    create_progress_predictor,
    create_ultra_fast_inference,
    create_integrated_pipeline,
    validate_features
)

def main():
    """Quick start example"""
    
    print("🚀 Quick Start - Addiction Recovery AI")
    print("=" * 50)
    
    # 1. Sentiment Analysis
    print("\n1. Sentiment Analysis")
    print("-" * 30)
    analyzer = create_sentiment_analyzer()
    result = analyzer.analyze("I'm feeling great today! Making progress!")
    print(f"Sentiment: {result}")
    
    # 2. Progress Prediction
    print("\n2. Progress Prediction")
    print("-" * 30)
    
    # Validate features
    features = [30/365, 0.3, 0.4, 0.7]  # days_sober, cravings, stress, mood
    is_valid, error = validate_features(features, expected_length=4)
    
    if is_valid:
        predictor = create_progress_predictor()
        engine = create_ultra_fast_inference(predictor)
        
        features_tensor = torch.tensor([features], dtype=torch.float32)
        progress = engine.predict(features_tensor)
        print(f"Progress Score: {progress.item():.4f}")
    else:
        print(f"Validation failed: {error}")
    
    # 3. Integrated Pipeline
    print("\n3. Integrated Pipeline")
    print("-" * 30)
    pipeline = create_integrated_pipeline(
        predictor,
        enable_validation=True,
        enable_monitoring=True,
        enable_optimization=True
    )
    
    output = pipeline.predict(features_tensor)
    print(f"Pipeline Output: {output.item():.4f}")
    
    # Check health
    health = pipeline.get_health_status()
    print(f"Pipeline Health: {health['model_health']['status']}")
    
    print("\n✅ Quick Start Complete!")


if __name__ == "__main__":
    main()










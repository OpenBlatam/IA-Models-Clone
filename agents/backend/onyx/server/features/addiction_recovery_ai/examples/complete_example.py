"""
Complete Example - All Features of Addiction Recovery AI
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from addiction_recovery_ai import (
    # Core
    create_ultra_fast_engine,
    create_enhanced_analyzer,
    # Training
    LoRATrainer,
    apply_lora_to_model,
    HyperparameterOptimizer,
    # Optimization
    export_to_onnx,
    ModelPruner,
    create_ensemble,
    # Production
    ModelRegistry,
    ABTest,
    ModelServer,
    HealthMonitor,
    # Advanced
    ModelInterpreter,
    OnlineLearner,
    AutoML,
    # Security
    APIKeyManager,
    InputSanitizer,
    # Validation
    DataValidator,
    AnomalyDetector
)


def example_complete_workflow():
    """Complete workflow example"""
    print("=" * 60)
    print("Complete Addiction Recovery AI Workflow")
    print("=" * 60)
    
    # 1. Create ultra-fast engine
    print("\n1. Creating Ultra-Fast Engine...")
    engine = create_ultra_fast_engine(use_gpu=torch.cuda.is_available())
    print("✓ Engine created")
    
    # 2. Predict progress
    print("\n2. Predicting Progress...")
    features = {
        "days_sober": 30,
        "cravings_level": 3,
        "stress_level": 4,
        "support_level": 8,
        "mood_score": 7,
        "sleep_quality": 6,
        "exercise_frequency": 3,
        "therapy_sessions": 2,
        "medication_compliance": 1.0,
        "social_activity": 4
    }
    progress = engine.predict_progress(features)
    print(f"✓ Progress: {progress:.2%}")
    
    # 3. Analyze sentiment
    print("\n3. Analyzing Sentiment...")
    sentiment = engine.analyze_sentiment("I'm feeling great today and making progress!")
    print(f"✓ Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")
    
    # 4. Check relapse risk
    print("\n4. Checking Relapse Risk...")
    sequence = [
        {"cravings_level": 3, "stress_level": 4, "mood_score": 7, "triggers_count": 1, "consumed": 0.0}
    ] * 30
    risk = engine.predict_relapse_risk(sequence)
    print(f"✓ Relapse Risk: {risk:.2%}")
    
    # 5. Validate data
    print("\n5. Validating Data...")
    validator = DataValidator()
    is_valid, error = validator.validate_features(features)
    print(f"✓ Data valid: {is_valid}")
    
    # 6. Model versioning
    print("\n6. Versioning Model...")
    registry = ModelRegistry()
    if hasattr(engine, 'progress_predictor'):
        model_path = registry.register(
            engine.progress_predictor,
            version="1.0.0",
            metadata={"accuracy": 0.95, "progress": progress}
        )
        print(f"✓ Model versioned: {model_path}")
    
    # 7. Health monitoring
    print("\n7. Monitoring Health...")
    monitor = HealthMonitor()
    monitor.record_request(success=True, latency=5.2)
    health = monitor.get_health()
    print(f"✓ Health: {health['is_healthy']}, Success Rate: {health['success_rate']:.2%}")
    
    # 8. Security
    print("\n8. Security Features...")
    key_manager = APIKeyManager()
    api_key = key_manager.generate_key(user_id="user123")
    is_valid, key_info = key_manager.validate_key(api_key)
    print(f"✓ API key generated and validated: {is_valid}")
    
    # 9. Interpretability
    print("\n9. Model Interpretability...")
    if hasattr(engine, 'progress_predictor'):
        try:
            background = torch.randn(10, 10)
            interpreter = ModelInterpreter(
                engine.progress_predictor,
                background,
                feature_names=list(features.keys())[:10]
            )
            interpreter.create_shap_explainer()
            test_input = torch.tensor([[features.get(k, 0) for k in list(features.keys())[:10]]])
            explanations = interpreter.explain_shap(test_input)
            print(f"✓ SHAP explanations generated")
        except Exception as e:
            print(f"⚠ Interpretability not available: {e}")
    
    # 10. Benchmark
    print("\n10. Benchmarking...")
    if hasattr(engine, 'benchmark'):
        results = engine.benchmark(num_runs=100)
        print(f"✓ Benchmark complete")
        for model_name, metrics in results.items():
            print(f"  {model_name}: {metrics['avg_time_ms']:.2f}ms")
    
    print("\n" + "=" * 60)
    print("Complete workflow finished successfully!")
    print("=" * 60)


def example_training_workflow():
    """Complete training workflow"""
    print("\n" + "=" * 60)
    print("Training Workflow Example")
    print("=" * 60)
    
    # Create dummy data
    train_data = TensorDataset(
        torch.randn(100, 10),
        torch.rand(100, 1)
    )
    val_data = TensorDataset(
        torch.randn(20, 10),
        torch.rand(20, 1)
    )
    train_loader = DataLoader(train_data, batch_size=32)
    val_loader = DataLoader(val_data, batch_size=16)
    
    # 1. AutoML for architecture
    print("\n1. AutoML Architecture Search...")
    try:
        automl = AutoML(train_loader, val_loader, max_models=3)
        arch_results = automl.search_architecture(input_size=10, output_size=1)
        print(f"✓ Best architecture: {arch_results['best_architecture']}")
    except Exception as e:
        print(f"⚠ AutoML not available: {e}")
    
    # 2. Hyperparameter optimization
    print("\n2. Hyperparameter Optimization...")
    try:
        def model_factory(hidden_size=64, dropout=0.2):
            return nn.Sequential(
                nn.Linear(10, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        
        optimizer = HyperparameterOptimizer(
            model_factory=model_factory,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=5
        )
        best_params = optimizer.optimize()
        print(f"✓ Best parameters: {best_params['best_params']}")
    except Exception as e:
        print(f"⚠ Hyperparameter optimization not available: {e}")
    
    # 3. LoRA fine-tuning
    print("\n3. LoRA Fine-tuning...")
    try:
        model = model_factory()
        model = apply_lora_to_model(model, rank=8)
        trainer = LoRATrainer(model, train_loader, val_loader, rank=8)
        params = trainer.count_parameters()
        print(f"✓ LoRA applied: {params['trainable_percent']:.1f}% trainable")
    except Exception as e:
        print(f"⚠ LoRA not available: {e}")
    
    print("\n" + "=" * 60)
    print("Training workflow complete!")
    print("=" * 60)


def example_production_deployment():
    """Production deployment example"""
    print("\n" + "=" * 60)
    print("Production Deployment Example")
    print("=" * 60)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    
    # 1. Model optimization
    print("\n1. Model Optimization...")
    pruned = ModelPruner.prune_model(model, amount=0.2)
    sparsity = ModelPruner.get_sparsity(pruned)
    print(f"✓ Model pruned: {sparsity * 100:.1f}% sparsity")
    
    # 2. ONNX export
    print("\n2. ONNX Export...")
    try:
        success = export_to_onnx(
            model=pruned,
            input_shape=(1, 10),
            output_path="model.onnx"
        )
        if success:
            print("✓ Model exported to ONNX")
    except Exception as e:
        print(f"⚠ ONNX export not available: {e}")
    
    # 3. Model serving
    print("\n3. Model Serving...")
    try:
        server = ModelServer(model, max_batch_size=32, num_workers=2)
        server.start()
        test_input = torch.randn(1, 10)
        result = server.predict(test_input, timeout=5.0)
        print(f"✓ Model serving: {result.item():.4f}")
        server.stop()
    except Exception as e:
        print(f"⚠ Model serving not available: {e}")
    
    # 4. A/B Testing setup
    print("\n4. A/B Testing Setup...")
    model_a = model
    model_b = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    ab_test = ABTest(model_a, model_b, split_ratio=0.5)
    print("✓ A/B test configured")
    
    print("\n" + "=" * 60)
    print("Production deployment example complete!")
    print("=" * 60)


def main():
    """Run all examples"""
    print("Addiction Recovery AI - Complete Examples")
    print("=" * 60)
    
    try:
        example_complete_workflow()
        example_training_workflow()
        example_production_deployment()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


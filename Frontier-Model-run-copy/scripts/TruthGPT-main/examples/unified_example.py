"""
Unified TruthGPT Example
Demonstrates the new clean, modular architecture
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path

# Import the new unified core
from core import (
    OptimizationEngine, OptimizationConfig, OptimizationLevel,
    ModelManager, ModelConfig, ModelType,
    TrainingManager, TrainingConfig,
    InferenceEngine, InferenceConfig,
    MonitoringSystem
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataset(Dataset):
    """Simple dataset for demonstration"""
    def __init__(self, size=1000, vocab_size=1000, seq_len=50):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        # Generate random data
        self.data = torch.randint(0, vocab_size, (size, seq_len))
        self.targets = torch.randint(0, vocab_size, (size, seq_len))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def main():
    """Main demonstration function"""
    logger.info("🚀 Starting TruthGPT Unified Architecture Demo")
    
    # 1. Setup Configuration
    logger.info("📋 Setting up configurations...")
    
    # Optimization configuration
    optimization_config = OptimizationConfig(
        level=OptimizationLevel.ENHANCED,
        enable_adaptive_precision=True,
        enable_memory_optimization=True,
        enable_kernel_fusion=True
    )
    
    # Model configuration
    model_config = ModelConfig(
        model_type=ModelType.TRANSFORMER,
        model_name="demo_model",
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        vocab_size=1000
    )
    
    # Training configuration
    training_config = TrainingConfig(
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        log_interval=10
    )
    
    # Inference configuration
    inference_config = InferenceConfig(
        batch_size=1,
        max_length=100,
        temperature=0.8,
        top_p=0.9
    )
    
    # 2. Initialize Components
    logger.info("🔧 Initializing components...")
    
    # Optimization engine
    optimizer = OptimizationEngine(optimization_config)
    logger.info(f"✅ Optimization engine initialized (level: {optimization_config.level.value})")
    
    # Model manager
    model_manager = ModelManager(model_config)
    logger.info(f"✅ Model manager initialized (type: {model_config.model_type.value})")
    
    # Training manager
    trainer = TrainingManager(training_config)
    logger.info(f"✅ Training manager initialized ({training_config.epochs} epochs)")
    
    # Inference engine
    inferencer = InferenceEngine(inference_config)
    logger.info(f"✅ Inference engine initialized")
    
    # Monitoring system
    monitor = MonitoringSystem()
    monitor.start_monitoring(interval=2.0)
    logger.info(f"✅ Monitoring system started")
    
    # 3. Load/Create Model
    logger.info("🤖 Loading/creating model...")
    model = model_manager.load_model()
    logger.info(f"✅ Model loaded: {model_manager.get_model_info()}")
    
    # 4. Optimize Model
    logger.info("⚡ Optimizing model...")
    optimized_model = optimizer.optimize_model(model)
    logger.info("✅ Model optimization completed")
    
    # 5. Setup Training
    logger.info("📚 Setting up training...")
    train_dataset = SimpleDataset(size=500, vocab_size=1000, seq_len=32)
    val_dataset = SimpleDataset(size=100, vocab_size=1000, seq_len=32)
    
    trainer.setup_training(optimized_model, train_dataset, val_dataset)
    logger.info("✅ Training setup completed")
    
    # 6. Train Model
    logger.info("🏋️ Starting training...")
    training_results = trainer.train(save_path="demo_model.pth")
    logger.info(f"✅ Training completed: {training_results}")
    
    # 7. Setup Inference
    logger.info("🔮 Setting up inference...")
    inferencer.load_model(optimized_model)
    inferencer.optimize_for_inference()
    logger.info("✅ Inference setup completed")
    
    # 8. Run Inference
    logger.info("🎯 Running inference...")
    test_prompt = [1, 2, 3, 4, 5]  # Simple token sequence
    inference_result = inferencer.generate(test_prompt, max_length=20)
    logger.info(f"✅ Inference completed: {inference_result['tokens_generated']} tokens generated")
    
    # 9. Performance Monitoring
    logger.info("📊 Collecting performance metrics...")
    
    # System metrics
    system_metrics = monitor.metrics_collector.get_system_summary()
    logger.info(f"System metrics: {system_metrics}")
    
    # Model metrics
    model_metrics = monitor.metrics_collector.get_model_summary()
    logger.info(f"Model metrics: {model_metrics}")
    
    # Training metrics
    training_metrics = monitor.metrics_collector.get_training_summary()
    logger.info(f"Training metrics: {training_metrics}")
    
    # Optimization metrics
    optimization_metrics = optimizer.get_performance_metrics()
    logger.info(f"Optimization metrics: {optimization_metrics}")
    
    # Inference metrics
    inference_metrics = inferencer.get_performance_metrics()
    logger.info(f"Inference metrics: {inference_metrics}")
    
    # 10. Export Results
    logger.info("💾 Exporting results...")
    
    # Export comprehensive report
    monitor.export_report("performance_report.json")
    
    # Export metrics
    monitor.metrics_collector.export_metrics("metrics_export.json")
    
    logger.info("✅ Results exported")
    
    # 11. Cleanup
    logger.info("🧹 Cleaning up...")
    monitor.stop_monitoring()
    logger.info("✅ Cleanup completed")
    
    logger.info("🎉 TruthGPT Unified Architecture Demo Completed Successfully!")
    
    return {
        'training_results': training_results,
        'inference_result': inference_result,
        'system_metrics': system_metrics,
        'model_metrics': model_metrics,
        'training_metrics': training_metrics,
        'optimization_metrics': optimization_metrics,
        'inference_metrics': inference_metrics
    }

def demonstrate_optimization_levels():
    """Demonstrate different optimization levels"""
    logger.info("🔬 Demonstrating optimization levels...")
    
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    levels = [
        OptimizationLevel.BASIC,
        OptimizationLevel.ENHANCED,
        OptimizationLevel.ADVANCED,
        OptimizationLevel.ULTRA,
        OptimizationLevel.SUPREME,
        OptimizationLevel.TRANSCENDENT
    ]
    
    for level in levels:
        logger.info(f"Testing {level.value} optimization...")
        config = OptimizationConfig(level=level)
        engine = OptimizationEngine(config)
        optimized_model = engine.optimize_model(model)
        
        metrics = engine.get_performance_metrics()
        logger.info(f"✅ {level.value}: {metrics}")

def demonstrate_model_types():
    """Demonstrate different model types"""
    logger.info("🏗️ Demonstrating model types...")
    
    model_types = [
        ModelType.TRANSFORMER,
        ModelType.CONVOLUTIONAL,
        ModelType.RECURRENT,
        ModelType.HYBRID
    ]
    
    for model_type in model_types:
        logger.info(f"Testing {model_type.value} model...")
        config = ModelConfig(model_type=model_type)
        manager = ModelManager(config)
        model = manager.load_model()
        
        info = manager.get_model_info()
        logger.info(f"✅ {model_type.value}: {info}")

if __name__ == "__main__":
    # Run main demonstration
    results = main()
    
    # Run additional demonstrations
    demonstrate_optimization_levels()
    demonstrate_model_types()
    
    logger.info("🎊 All demonstrations completed!")


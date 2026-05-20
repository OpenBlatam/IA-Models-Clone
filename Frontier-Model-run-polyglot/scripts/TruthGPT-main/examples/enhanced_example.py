"""
Enhanced TruthGPT Example
Demonstrates the improved refactored architecture with all advanced features
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import time
import json
from pathlib import Path

# Import the enhanced unified core
from core import (
    OptimizationEngine, OptimizationConfig, OptimizationLevel,
    ModelManager, ModelConfig, ModelType,
    TrainingManager, TrainingConfig,
    InferenceEngine, InferenceConfig,
    MonitoringSystem
)
from core.benchmarking import BenchmarkRunner, BenchmarkConfig
from core.production import ProductionDeployment, ProductionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedDataset(Dataset):
    """Advanced dataset for demonstration"""
    def __init__(self, size=1000, vocab_size=1000, seq_len=50):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        # Generate more realistic data
        self.data = torch.randint(0, vocab_size, (size, seq_len))
        self.targets = torch.randint(0, vocab_size, (size, seq_len))
        
        # Add some structure to the data
        for i in range(0, size, 10):
            self.data[i] = torch.arange(seq_len) % vocab_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class MockAdvancedTokenizer:
    """Advanced mock tokenizer"""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.pad_token_id = 1
        self.unk_token_id = 2
        
    def encode(self, text, return_tensors="pt", max_length=None):
        # More sophisticated encoding
        tokens = [ord(c) % self.vocab_size for c in text[:max_length or 50]]
        if return_tensors == "pt":
            return torch.tensor([tokens], dtype=torch.long)
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=True):
        # More sophisticated decoding
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in [self.pad_token_id, self.unk_token_id]]
        return ''.join([chr(token_id % 256) for token_id in token_ids if token_id < 256])

def demonstrate_enhanced_optimization():
    """Demonstrate enhanced optimization capabilities"""
    logger.info("🚀 Demonstrating Enhanced Optimization System")
    
    # Test all optimization levels
    levels = [
        OptimizationLevel.BASIC,
        OptimizationLevel.ENHANCED,
        OptimizationLevel.ADVANCED,
        OptimizationLevel.ULTRA,
        OptimizationLevel.SUPREME,
        OptimizationLevel.TRANSCENDENT
    ]
    
    results = {}
    
    for level in levels:
        logger.info(f"⚡ Testing {level.value} optimization...")
        
        # Create configuration
        config = OptimizationConfig(
            level=level,
            enable_adaptive_precision=True,
            enable_memory_optimization=True,
            enable_kernel_fusion=True,
            enable_quantization=(level in [OptimizationLevel.ADVANCED, OptimizationLevel.ULTRA, OptimizationLevel.SUPREME, OptimizationLevel.TRANSCENDENT]),
            enable_sparsity=(level in [OptimizationLevel.ULTRA, OptimizationLevel.SUPREME, OptimizationLevel.TRANSCENDENT]),
            enable_meta_learning=(level in [OptimizationLevel.ULTRA, OptimizationLevel.SUPREME, OptimizationLevel.TRANSCENDENT]),
            enable_neural_architecture_search=(level in [OptimizationLevel.SUPREME, OptimizationLevel.TRANSCENDENT]),
            quantum_simulation=(level == OptimizationLevel.TRANSCENDENT),
            consciousness_simulation=(level == OptimizationLevel.TRANSCENDENT),
            temporal_optimization=(level == OptimizationLevel.TRANSCENDENT)
        )
        
        # Create optimization engine
        engine = OptimizationEngine(config)
        
        # Create test model
        model_config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            vocab_size=1000
        )
        model_manager = ModelManager(model_config)
        model = model_manager.load_model()
        
        # Optimize model
        start_time = time.time()
        optimized_model = engine.optimize_model(model)
        optimization_time = time.time() - start_time
        
        # Get performance metrics
        metrics = engine.get_performance_metrics()
        
        results[level.value] = {
            "optimization_time": optimization_time,
            "metrics": metrics,
            "model_info": model_manager.get_model_info()
        }
        
        logger.info(f"✅ {level.value} optimization completed in {optimization_time:.3f}s")
    
    return results

def demonstrate_advanced_benchmarking():
    """Demonstrate advanced benchmarking capabilities"""
    logger.info("📊 Demonstrating Advanced Benchmarking System")
    
    # Create benchmark configuration
    benchmark_config = BenchmarkConfig(
        num_runs=3,
        warmup_runs=1,
        batch_sizes=[1, 4, 8],
        sequence_lengths=[64, 128, 256],
        measure_memory=True,
        measure_cpu=True,
        measure_gpu=True,
        save_results=True,
        output_dir="benchmark_results"
    )
    
    # Create benchmark runner
    benchmarker = BenchmarkRunner(benchmark_config)
    
    # Create test models with different configurations
    models = {}
    
    # Basic model
    basic_config = ModelConfig(
        model_type=ModelType.TRANSFORMER,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        vocab_size=500
    )
    basic_manager = ModelManager(basic_config)
    models["basic"] = basic_manager.load_model()
    
    # Enhanced model
    enhanced_config = ModelConfig(
        model_type=ModelType.TRANSFORMER,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        vocab_size=1000
    )
    enhanced_manager = ModelManager(enhanced_config)
    models["enhanced"] = enhanced_manager.load_model()
    
    # Optimized model
    optimization_config = OptimizationConfig(level=OptimizationLevel.ENHANCED)
    optimizer = OptimizationEngine(optimization_config)
    optimized_model = optimizer.optimize_model(models["enhanced"])
    models["optimized"] = optimized_model
    
    # Prepare test data
    test_data = {
        "input_ids": torch.randint(0, 1000, (10, 128)),
        "attention_mask": torch.ones(10, 128)
    }
    
    # Run comparative benchmark
    logger.info("🔬 Running comparative benchmark...")
    benchmark_results = benchmarker.run_comparative_benchmark(models, test_data)
    
    # Run optimization benchmark
    logger.info("⚡ Running optimization benchmark...")
    optimization_levels = ["basic", "enhanced", "advanced", "ultra"]
    optimization_results = benchmarker.run_optimization_benchmark(
        models["enhanced"], "enhanced_model", test_data, optimization_levels
    )
    
    return {
        "comparative_results": benchmark_results,
        "optimization_results": optimization_results
    }

def demonstrate_production_deployment():
    """Demonstrate production deployment capabilities"""
    logger.info("🏭 Demonstrating Production Deployment System")
    
    # Create production configuration
    production_config = ProductionConfig(
        service_name="truthgpt_enhanced_service",
        version="2.0.0",
        host="0.0.0.0",
        port=8001,
        log_level="INFO",
        log_file="enhanced_production.log",
        max_workers=4,
        health_check_interval=10,
        metrics_interval=30,
        alert_thresholds={
            "cpu_percent": 75.0,
            "memory_percent": 80.0,
            "gpu_memory_percent": 85.0
        }
    )
    
    # Create production deployment
    deployment = ProductionDeployment(production_config)
    
    try:
        # Deploy the service
        logger.info("🚀 Deploying production service...")
        deployment.deploy()
        
        # Simulate some production operations
        logger.info("📊 Simulating production operations...")
        
        # Create API instance for testing
        api = deployment.api
        
        # Submit some test requests
        request_ids = []
        
        # Health check request
        health_id = api.submit_request("health", {})
        request_ids.append(health_id)
        
        # Generate request
        generate_id = api.submit_request("generate", {
            "prompt": "Hello, world!",
            "max_length": 50,
            "temperature": 0.8
        })
        request_ids.append(generate_id)
        
        # Optimize request
        optimize_id = api.submit_request("optimize", {
            "model_type": "transformer",
            "optimization_level": "enhanced"
        })
        request_ids.append(optimize_id)
        
        # Wait for requests to process
        time.sleep(2)
        
        # Get monitoring summary
        monitor = deployment.api.health_checker
        health_status = monitor.check_health()
        
        logger.info(f"✅ Production deployment successful")
        logger.info(f"📊 Health status: {health_status['healthy']}")
        logger.info(f"📈 Request IDs: {request_ids}")
        
        return {
            "deployment_successful": True,
            "health_status": health_status,
            "request_ids": request_ids
        }
        
    except Exception as e:
        logger.error(f"❌ Production deployment failed: {e}")
        return {
            "deployment_successful": False,
            "error": str(e)
        }
    
    finally:
        # Cleanup
        try:
            deployment.shutdown()
        except:
            pass

def demonstrate_complete_workflow():
    """Demonstrate complete enhanced workflow"""
    logger.info("🎯 Demonstrating Complete Enhanced Workflow")
    
    # 1. Enhanced Model Creation
    logger.info("🤖 Creating enhanced model...")
    model_config = ModelConfig(
        model_type=ModelType.TRANSFORMER,
        model_name="enhanced_demo_model",
        hidden_size=256,
        num_layers=6,
        num_heads=8,
        vocab_size=2000
    )
    model_manager = ModelManager(model_config)
    model = model_manager.load_model()
    
    # 2. Advanced Optimization
    logger.info("⚡ Applying advanced optimization...")
    optimization_config = OptimizationConfig(
        level=OptimizationLevel.ULTRA,
        enable_adaptive_precision=True,
        enable_memory_optimization=True,
        enable_kernel_fusion=True,
        enable_quantization=True,
        enable_sparsity=True,
        enable_meta_learning=True
    )
    optimizer = OptimizationEngine(optimization_config)
    optimized_model = optimizer.optimize_model(model)
    
    # 3. Advanced Training
    logger.info("🏋️ Starting advanced training...")
    training_config = TrainingConfig(
        epochs=3,
        batch_size=8,
        learning_rate=1e-4,
        optimizer="adamw",
        scheduler="cosine",
        mixed_precision=True,
        gradient_clip=1.0,
        early_stopping_patience=2
    )
    
    # Create datasets
    train_dataset = AdvancedDataset(size=500, vocab_size=2000, seq_len=64)
    val_dataset = AdvancedDataset(size=100, vocab_size=2000, seq_len=64)
    
    trainer = TrainingManager(training_config)
    trainer.setup_training(optimized_model, train_dataset, val_dataset)
    training_results = trainer.train()
    
    # 4. Advanced Inference
    logger.info("🔮 Setting up advanced inference...")
    inference_config = InferenceConfig(
        batch_size=1,
        max_length=100,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        use_cache=True,
        cache_size=1000
    )
    
    tokenizer = MockAdvancedTokenizer(vocab_size=2000)
    inferencer = InferenceEngine(inference_config)
    inferencer.load_model(optimized_model, tokenizer)
    inferencer.optimize_for_inference()
    
    # 5. Advanced Monitoring
    logger.info("📊 Setting up advanced monitoring...")
    monitor = MonitoringSystem()
    monitor.start_monitoring(interval=1.0)
    
    # Add alert callback
    def alert_callback(alert_type, data):
        logger.warning(f"🚨 Alert: {alert_type} - {data}")
    
    monitor.add_alert_callback(alert_callback)
    
    # 6. Generate Text
    logger.info("✨ Generating text...")
    generation_results = []
    
    prompts = [
        "The future of AI is",
        "Machine learning can",
        "Neural networks are",
        "Deep learning enables"
    ]
    
    for prompt in prompts:
        result = inferencer.generate(prompt, max_length=30)
        generation_results.append({
            "prompt": prompt,
            "generated": result["generated_text"],
            "tokens_generated": result["tokens_generated"],
            "generation_time": result["generation_time"]
        })
        logger.info(f"📝 '{prompt}' → '{result['generated_text'][:50]}...'")
    
    # 7. Performance Analysis
    logger.info("📈 Analyzing performance...")
    
    # Get comprehensive report
    report = monitor.get_comprehensive_report()
    
    # Get optimization metrics
    optimization_metrics = optimizer.get_performance_metrics()
    
    # Get inference metrics
    inference_metrics = inferencer.get_performance_metrics()
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # 8. Export Results
    logger.info("💾 Exporting results...")
    
    # Export comprehensive report
    with open("enhanced_workflow_report.json", "w") as f:
        json.dump({
            "training_results": training_results,
            "generation_results": generation_results,
            "optimization_metrics": optimization_metrics,
            "inference_metrics": inference_metrics,
            "monitoring_report": report,
            "timestamp": time.time()
        }, f, indent=2)
    
    logger.info("✅ Complete enhanced workflow completed successfully!")
    
    return {
        "training_results": training_results,
        "generation_results": generation_results,
        "optimization_metrics": optimization_metrics,
        "inference_metrics": inference_metrics,
        "monitoring_report": report
    }

def main():
    """Main demonstration function"""
    logger.info("🎉 Starting Enhanced TruthGPT Demonstration")
    logger.info("=" * 60)
    
    try:
        # 1. Enhanced Optimization Demo
        logger.info("\n🚀 PHASE 1: Enhanced Optimization System")
        optimization_results = demonstrate_enhanced_optimization()
        
        # 2. Advanced Benchmarking Demo
        logger.info("\n📊 PHASE 2: Advanced Benchmarking System")
        benchmarking_results = demonstrate_advanced_benchmarking()
        
        # 3. Production Deployment Demo
        logger.info("\n🏭 PHASE 3: Production Deployment System")
        production_results = demonstrate_production_deployment()
        
        # 4. Complete Workflow Demo
        logger.info("\n🎯 PHASE 4: Complete Enhanced Workflow")
        workflow_results = demonstrate_complete_workflow()
        
        # 5. Summary
        logger.info("\n📋 ENHANCED DEMONSTRATION SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"✅ Optimization levels tested: {len(optimization_results)}")
        logger.info(f"✅ Benchmarking completed: {len(benchmarking_results)}")
        logger.info(f"✅ Production deployment: {production_results.get('deployment_successful', False)}")
        logger.info(f"✅ Complete workflow: Successful")
        
        # Performance summary
        if optimization_results:
            avg_optimization_time = sum(r["optimization_time"] for r in optimization_results.values()) / len(optimization_results)
            logger.info(f"⚡ Average optimization time: {avg_optimization_time:.3f}s")
        
        if workflow_results.get("inference_metrics"):
            metrics = workflow_results["inference_metrics"]
            logger.info(f"🔮 Total inferences: {metrics.get('total_inferences', 0)}")
            logger.info(f"📊 Tokens per second: {metrics.get('tokens_per_second', 0):.1f}")
        
        logger.info("\n🎊 Enhanced TruthGPT Demonstration Completed Successfully!")
        
        return {
            "optimization_results": optimization_results,
            "benchmarking_results": benchmarking_results,
            "production_results": production_results,
            "workflow_results": workflow_results
        }
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise

if __name__ == "__main__":
    results = main()
    logger.info("🎉 Enhanced demonstration completed!")


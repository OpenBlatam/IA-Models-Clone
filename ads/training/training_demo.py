"""
Training System Demo for the ads feature.

This demo showcases the complete unified training system that consolidates
all scattered training implementations into a clean, modular architecture.
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

from .base_trainer import TrainingConfig
from .pytorch_trainer import PyTorchTrainer, PyTorchModelConfig, PyTorchDataConfig
from .diffusion_trainer import DiffusionTrainer, DiffusionModelConfig, DiffusionTrainingConfig
from .multi_gpu_trainer import MultiGPUTrainer, GPUConfig, MultiGPUTrainingConfig
from .training_factory import TrainingFactory, TrainerConfig, TrainerType
from .experiment_tracker import ExperimentTracker, ExperimentConfig
from .training_optimizer import TrainingOptimizer, OptimizationConfig, OptimizationLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingSystemDemo:
    """
    Comprehensive demo of the unified training system.
    
    This demo showcases:
    - All trainer types (PyTorch, Diffusion, Multi-GPU)
    - Training factory and configuration management
    - Experiment tracking and optimization
    - Performance monitoring and analysis
    """
    
    def __init__(self):
        """Initialize the training system demo."""
        self.factory = TrainingFactory()
        self.experiment_tracker = ExperimentTracker("./demo_experiments")
        self.optimizer = TrainingOptimizer(OptimizationConfig(level=OptimizationLevel.STANDARD))
        
        logger.info("Training System Demo initialized")
    
    async def run_comprehensive_demo(self):
        """Run the complete training system demo."""
        logger.info("🚀 Starting Comprehensive Training System Demo")
        
        try:
            # 1. Demonstrate PyTorch Trainer
            await self._demo_pytorch_trainer()
            
            # 2. Demonstrate Diffusion Trainer
            await self._demo_diffusion_trainer()
            
            # 3. Demonstrate Multi-GPU Trainer
            await self._demo_multi_gpu_trainer()
            
            # 4. Demonstrate Training Factory
            await self._demo_training_factory()
            
            # 5. Demonstrate Experiment Tracking
            await self._demo_experiment_tracking()
            
            # 6. Demonstrate Training Optimization
            await self._demo_training_optimization()
            
            # 7. Demonstrate System Integration
            await self._demo_system_integration()
            
            logger.info("✅ Comprehensive Training System Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}", exc_info=True)
    
    async def _demo_pytorch_trainer(self):
        """Demonstrate PyTorch trainer functionality."""
        logger.info("\n🔧 Demonstrating PyTorch Trainer")
        
        # Create configuration
        config = TrainingConfig(
            model_name="demo_pytorch_model",
            dataset_name="synthetic_data",
            batch_size=16,
            num_epochs=3,
            learning_rate=0.001,
            mixed_precision=True
        )
        
        model_config = PyTorchModelConfig(
            input_size=10,
            hidden_size=32,
            output_size=1,
            num_layers=2
        )
        
        data_config = PyTorchDataConfig(
            num_samples=500,
            train_split=0.8,
            val_split=0.2
        )
        
        # Create trainer
        trainer = PyTorchTrainer(config, model_config, data_config)
        
        # Setup training
        await trainer.setup_training()
        
        # Get model info
        model_info = trainer.get_model_info()
        logger.info(f"PyTorch Model Info: {model_info}")
        
        # Run training
        logger.info("Starting PyTorch training...")
        result = await trainer.train()
        
        logger.info(f"PyTorch Training Result: {result.to_dict()}")
        
        # Cleanup
        del trainer
    
    async def _demo_diffusion_trainer(self):
        """Demonstrate diffusion trainer functionality."""
        logger.info("\n🎨 Demonstrating Diffusion Trainer")
        
        # Create configuration
        config = TrainingConfig(
            model_name="demo_diffusion_model",
            dataset_name="synthetic_images",
            batch_size=8,
            num_epochs=2,
            learning_rate=0.0001
        )
        
        diffusion_config = DiffusionModelConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            scheduler_type="DDIM",
            num_inference_steps=20
        )
        
        training_config = DiffusionTrainingConfig(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02
        )
        
        # Create trainer
        trainer = DiffusionTrainer(config, diffusion_config, training_config)
        
        # Setup training
        await trainer.setup_training()
        
        # Get model info
        model_info = trainer.get_model_info()
        logger.info(f"Diffusion Model Info: {model_info}")
        
        # Run training
        logger.info("Starting diffusion training...")
        result = await trainer.train()
        
        logger.info(f"Diffusion Training Result: {result.to_dict()}")
        
        # Cleanup
        del trainer
    
    async def _demo_multi_gpu_trainer(self):
        """Demonstrate multi-GPU trainer functionality."""
        logger.info("\n🚀 Demonstrating Multi-GPU Trainer")
        
        # Create configuration
        config = TrainingConfig(
            model_name="demo_multi_gpu_model",
            dataset_name="synthetic_data",
            batch_size=32,
            num_epochs=2,
            learning_rate=0.001
        )
        
        gpu_config = GPUConfig(
            use_multi_gpu=True,
            distributed_training=False,
            batch_size_per_gpu=8
        )
        
        multi_gpu_config = MultiGPUTrainingConfig(
            gpu_config=gpu_config,
            use_data_parallel=True,
            monitor_gpu_memory=True
        )
        
        # Create trainer
        trainer = MultiGPUTrainer(config, gpu_config, multi_gpu_config)
        
        # Setup training
        await trainer.setup_training()
        
        # Get model info
        model_info = trainer.get_model_info()
        logger.info(f"Multi-GPU Model Info: {model_info}")
        
        # Run training
        logger.info("Starting multi-GPU training...")
        result = await trainer.train()
        
        logger.info(f"Multi-GPU Training Result: {result.to_dict()}")
        
        # Cleanup
        if hasattr(trainer, 'cleanup'):
            trainer.cleanup()
        del trainer
    
    async def _demo_training_factory(self):
        """Demonstrate training factory functionality."""
        logger.info("\n🏭 Demonstrating Training Factory")
        
        # List available trainers
        available_trainers = self.factory.list_trainers()
        logger.info(f"Available trainers: {available_trainers}")
        
        # Create different trainer types
        config = TrainingConfig(
            model_name="factory_demo_model",
            dataset_name="synthetic_data",
            batch_size=16,
            num_epochs=1,
            learning_rate=0.001
        )
        
        # Create PyTorch trainer via factory
        pytorch_config = TrainerConfig(
            trainer_type=TrainerType.PYTORCH,
            base_config=config,
            specific_configs={
                "model_config": PyTorchModelConfig(input_size=5, hidden_size=16),
                "data_config": PyTorchDataConfig(num_samples=100)
            }
        )
        
        pytorch_trainer = self.factory.create_trainer(pytorch_config)
        logger.info(f"PyTorch trainer created via factory: {pytorch_trainer.__class__.__name__}")
        
        # Create optimal trainer
        requirements = {"multi_gpu": False, "diffusion_model": False}
        optimal_trainer = self.factory.create_optimal_trainer(config, requirements)
        logger.info(f"Optimal trainer created: {optimal_trainer.__class__.__name__}")
        
        # List instances
        instances = self.factory.list_instances()
        logger.info(f"Factory instances: {instances}")
        
        # Cleanup
        for instance_id in instances:
            self.factory.cleanup_trainer(instance_id)
    
    async def _demo_experiment_tracking(self):
        """Demonstrate experiment tracking functionality."""
        logger.info("\n📊 Demonstrating Experiment Tracking")
        
        # Create experiment
        experiment_config = ExperimentConfig(
            name="demo_experiment",
            description="Demonstration experiment for training system",
            tags=["demo", "training", "ads"],
            hyperparameters={"learning_rate": 0.001, "batch_size": 16},
            dataset_info={"name": "synthetic", "size": 1000},
            model_info={"type": "neural_network", "layers": 3}
        )
        
        experiment_name = self.experiment_tracker.create_experiment(experiment_config)
        logger.info(f"Experiment created: {experiment_name}")
        
        # Start run
        run_id = self.experiment_tracker.start_run(experiment_name)
        logger.info(f"Run started: {run_id}")
        
        # Log some metrics
        from .base_trainer import TrainingMetrics
        
        for epoch in range(3):
            metrics = TrainingMetrics(
                epoch=epoch,
                step=epoch * 10,
                loss=0.5 - epoch * 0.1,
                learning_rate=0.001,
                validation_loss=0.6 - epoch * 0.1
            )
            
            self.experiment_tracker.log_metrics(metrics, run_id)
            logger.info(f"Metrics logged for epoch {epoch}")
        
        # Complete run
        from .base_trainer import TrainingResult, TrainingStatus
        
        result = TrainingResult(
            success=True,
            status=TrainingStatus.COMPLETED,
            training_time=30.0
        )
        
        self.experiment_tracker.complete_run(run_id, result, notes="Demo run completed successfully")
        logger.info("Run completed")
        
        # List experiments and runs
        experiments = self.experiment_tracker.list_experiments()
        runs = self.experiment_tracker.list_runs(experiment_name)
        
        logger.info(f"Experiments: {experiments}")
        logger.info(f"Runs for {experiment_name}: {runs}")
        
        # Get statistics
        stats = self.experiment_tracker.get_statistics()
        logger.info(f"Experiment statistics: {stats}")
    
    async def _demo_training_optimization(self):
        """Demonstrate training optimization functionality."""
        logger.info("\n⚡ Demonstrating Training Optimization")
        
        # Create a simple trainer for optimization
        config = TrainingConfig(
            model_name="optimization_demo_model",
            dataset_name="synthetic_data",
            batch_size=16,
            num_epochs=1,
            learning_rate=0.001
        )
        
        trainer = PyTorchTrainer(config)
        await trainer.setup_training()
        
        # Apply optimizations
        logger.info("Applying training optimizations...")
        result = self.optimizer.optimize_trainer(trainer)
        
        logger.info(f"Optimization result: {result.__dict__}")
        
        # Get optimization summary
        summary = self.optimizer.get_optimization_summary()
        logger.info(f"Optimization summary: {summary}")
        
        # Get recommendations
        recommendations = self.optimizer.get_recommendations(trainer)
        logger.info(f"Optimization recommendations: {recommendations}")
        
        # Cleanup
        del trainer
    
    async def _demo_system_integration(self):
        """Demonstrate complete system integration."""
        logger.info("\n🔗 Demonstrating System Integration")
        
        # Create experiment
        experiment_config = ExperimentConfig(
            name="integration_demo",
            description="Complete system integration demonstration",
            tags=["integration", "demo", "training"],
            hyperparameters={"learning_rate": 0.001, "batch_size": 16},
            dataset_info={"name": "synthetic", "size": 500},
            model_info={"type": "neural_network", "layers": 2}
        )
        
        experiment_name = self.experiment_tracker.create_experiment(experiment_config)
        run_id = self.experiment_tracker.start_run(experiment_name)
        
        # Create optimized trainer via factory
        config = TrainingConfig(
            model_name="integration_model",
            dataset_name="synthetic_data",
            batch_size=16,
            num_epochs=2,
            learning_rate=0.001,
            mixed_precision=True
        )
        
        # Create trainer with optimization
        trainer_config = TrainerConfig(
            trainer_type=TrainerType.PYTORCH,
            base_config=config,
            specific_configs={
                "model_config": PyTorchModelConfig(input_size=8, hidden_size=24),
                "data_config": PyTorchDataConfig(num_samples=200)
            }
        )
        
        trainer = self.factory.create_trainer(trainer_config)
        
        # Apply optimizations
        optimization_result = self.optimizer.optimize_trainer(trainer)
        logger.info(f"Integration optimization result: {optimization_result.success}")
        
        # Setup and run training
        await trainer.setup_training()
        
        # Add callback to log metrics
        def log_metrics_callback(epoch: int, metrics):
            self.experiment_tracker.log_metrics(metrics, run_id)
        
        trainer.add_callback(log_metrics_callback)
        
        # Run training
        logger.info("Running integrated training...")
        result = await trainer.train()
        
        # Complete experiment
        self.experiment_tracker.complete_run(
            run_id, 
            result, 
            notes="Integration demo completed successfully"
        )
        
        # Export experiment
        export_path = self.experiment_tracker.export_experiment(experiment_name)
        logger.info(f"Experiment exported to: {export_path}")
        
        # Cleanup
        self.factory.cleanup_all()
        self.experiment_tracker.cleanup_experiment(experiment_name)
        
        logger.info("System integration demo completed")
    
    def print_system_summary(self):
        """Print a summary of the training system."""
        logger.info("\n" + "="*60)
        logger.info("🎯 TRAINING SYSTEM SUMMARY")
        logger.info("="*60)
        
        logger.info("✅ Base Trainer: Abstract base class with common training interface")
        logger.info("✅ PyTorch Trainer: Standard neural network training with mixed precision")
        logger.info("✅ Diffusion Trainer: Advanced diffusion model training")
        logger.info("✅ Multi-GPU Trainer: Distributed and parallel training")
        logger.info("✅ Training Factory: Unified trainer creation and management")
        logger.info("✅ Experiment Tracker: Comprehensive experiment monitoring")
        logger.info("✅ Training Optimizer: Performance optimization and monitoring")
        
        logger.info("\n🔧 Key Features:")
        logger.info("   • Clean Architecture with clear separation of concerns")
        logger.info("   • Unified interface across all trainer types")
        logger.info("   • Comprehensive experiment tracking and optimization")
        logger.info("   • Factory pattern for trainer management")
        logger.info("   • Performance monitoring and optimization")
        logger.info("   • Extensible design for future trainer types")
        
        logger.info("\n📁 Consolidated from scattered implementations:")
        logger.info("   • pytorch_example.py → PyTorchTrainer")
        logger.info("   • diffusion_service.py → DiffusionTrainer")
        logger.info("   • multi_gpu_training.py → MultiGPUTrainer")
        logger.info("   • experiment_tracker.py → ExperimentTracker")
        logger.info("   • Various optimization files → TrainingOptimizer")
        
        logger.info("="*60)

async def main():
    """Main demo function."""
    demo = TrainingSystemDemo()
    
    try:
        await demo.run_comprehensive_demo()
        demo.print_system_summary()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
    
    finally:
        # Cleanup
        demo.factory.cleanup_all()

if __name__ == "__main__":
    asyncio.run(main())

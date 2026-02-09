from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from core.multi_gpu_training import (
from core.optimized_training_optimizer import (
from core.training_logger import create_training_logger
from core.error_handling import ErrorHandler
from typing import Any, List, Dict, Optional
import logging
"""
Multi-GPU Training Demonstration

Comprehensive demonstration of the multi-GPU training system with
DataParallel and DistributedDataParallel support.
"""


    MultiGPUTrainer, MultiGPUConfig, create_multi_gpu_trainer,
    optimize_model_for_multi_gpu, setup_distributed_environment,
    launch_distributed_training, get_free_port
)
    OptimizedTrainingOptimizer, create_optimized_training_optimizer,
    train_model_with_optimization
)


class EmailSequenceDataset(Dataset):
    """Email sequence dataset for multi-GPU training demonstration"""
    
    def __init__(self, num_samples: int = 1000, sequence_length: int = 50, vocab_size: int = 1000):
        
    """__init__ function."""
self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        
        # Generate synthetic email sequence data
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            # Generate random sequence
            sequence = np.random.randint(0, vocab_size, sequence_length)
            self.data.append(sequence)
            
            # Generate label (engagement prediction: 0 or 1)
            label = np.random.randint(0, 2)
            self.labels.append(label)
    
    def __len__(self) -> Any:
        return self.num_samples
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class EmailSequenceModel(nn.Module):
    """Email sequence model for multi-GPU training demonstration"""
    
    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 128, hidden_dim: int = 256, num_classes: int = 2):
        
    """__init__ function."""
super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x) -> Any:
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]
        output = self.dropout(last_output)
        logits = self.classifier(output)
        
        return logits


class MultiGPUTrainingDemo:
    """Multi-GPU training demonstration"""
    
    def __init__(self, demo_name: str = "multi_gpu_training_demo"):
        
    """__init__ function."""
self.demo_name = demo_name
        self.logger = create_training_logger(
            experiment_name=demo_name,
            log_dir="logs/multi_gpu_demo",
            log_level="INFO",
            enable_visualization=True
        )
        self.error_handler = ErrorHandler(debug_mode=True)
        
        # Demo results
        self.results = {}
    
    async def demo_data_parallel_training(self) -> Dict[str, Any]:
        """Demonstrate DataParallel training"""
        
        self.logger.log_info("=== DataParallel Training Demo ===")
        
        try:
            # Create dataset and data loaders
            train_dataset = EmailSequenceDataset(num_samples=2000, sequence_length=50)
            val_dataset = EmailSequenceDataset(num_samples=500, sequence_length=50)
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
            
            # Create model
            model = EmailSequenceModel()
            
            # Configure DataParallel
            multi_gpu_config = MultiGPUConfig(
                training_mode="data_parallel",
                enable_data_parallel=True,
                enable_distributed=False,
                enable_gpu_monitoring=True,
                device_ids=None  # Use all available GPUs
            )
            
            # Create optimized training optimizer
            optimizer = create_optimized_training_optimizer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                experiment_name="data_parallel_demo",
                debug_mode=True,
                enable_pytorch_debugging=True,
                multi_gpu_config=multi_gpu_config,
                max_epochs=3,
                learning_rate=0.001,
                early_stopping_patience=5
            )
            
            # Get training information
            training_info = optimizer.multi_gpu_trainer.get_training_info()
            self.logger.log_info(f"DataParallel training info: {json.dumps(training_info, indent=2)}")
            
            # Train model
            start_time = time.time()
            results = await optimizer.train()
            training_time = time.time() - start_time
            
            # Get performance summary
            performance_summary = optimizer.multi_gpu_trainer.get_performance_summary()
            
            demo_results = {
                "training_mode": "data_parallel",
                "training_time": training_time,
                "results": results,
                "performance_summary": performance_summary,
                "training_info": training_info
            }
            
            self.logger.log_info(f"DataParallel training completed in {training_time:.2f} seconds")
            self.logger.log_info(f"Performance summary: {json.dumps(performance_summary, indent=2)}")
            
            return demo_results
            
        except Exception as e:
            self.logger.log_error(e, "DataParallel demo", "demo_data_parallel_training")
            return {"error": str(e)}
    
    async def demo_distributed_training(self) -> Dict[str, Any]:
        """Demonstrate DistributedDataParallel training"""
        
        self.logger.log_info("=== DistributedDataParallel Training Demo ===")
        
        try:
            # Create dataset and data loaders
            train_dataset = EmailSequenceDataset(num_samples=2000, sequence_length=50)
            val_dataset = EmailSequenceDataset(num_samples=500, sequence_length=50)
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
            
            # Create model
            model = EmailSequenceModel()
            
            # Configure DistributedDataParallel
            multi_gpu_config = MultiGPUConfig(
                training_mode="distributed",
                enable_data_parallel=False,
                enable_distributed=True,
                enable_gpu_monitoring=True,
                backend="nccl",
                sync_bn=True
            )
            
            # Create optimized training optimizer
            optimizer = create_optimized_training_optimizer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                experiment_name="distributed_demo",
                debug_mode=True,
                enable_pytorch_debugging=True,
                multi_gpu_config=multi_gpu_config,
                max_epochs=3,
                learning_rate=0.001,
                early_stopping_patience=5
            )
            
            # Get training information
            training_info = optimizer.multi_gpu_trainer.get_training_info()
            self.logger.log_info(f"Distributed training info: {json.dumps(training_info, indent=2)}")
            
            # Train model
            start_time = time.time()
            results = await optimizer.train()
            training_time = time.time() - start_time
            
            # Get performance summary
            performance_summary = optimizer.multi_gpu_trainer.get_performance_summary()
            
            demo_results = {
                "training_mode": "distributed",
                "training_time": training_time,
                "results": results,
                "performance_summary": performance_summary,
                "training_info": training_info
            }
            
            self.logger.log_info(f"Distributed training completed in {training_time:.2f} seconds")
            self.logger.log_info(f"Performance summary: {json.dumps(performance_summary, indent=2)}")
            
            return demo_results
            
        except Exception as e:
            self.logger.log_error(e, "Distributed demo", "demo_distributed_training")
            return {"error": str(e)}
    
    async def demo_auto_detection(self) -> Dict[str, Any]:
        """Demonstrate automatic training mode detection"""
        
        self.logger.log_info("=== Auto Detection Demo ===")
        
        try:
            # Create dataset and data loaders
            train_dataset = EmailSequenceDataset(num_samples=2000, sequence_length=50)
            val_dataset = EmailSequenceDataset(num_samples=500, sequence_length=50)
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
            
            # Create model
            model = EmailSequenceModel()
            
            # Configure auto detection
            multi_gpu_config = MultiGPUConfig(
                training_mode="auto",
                enable_data_parallel=True,
                enable_distributed=True,
                enable_gpu_monitoring=True
            )
            
            # Create optimized training optimizer
            optimizer = create_optimized_training_optimizer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                experiment_name="auto_detection_demo",
                debug_mode=True,
                enable_pytorch_debugging=True,
                multi_gpu_config=multi_gpu_config,
                max_epochs=3,
                learning_rate=0.001,
                early_stopping_patience=5
            )
            
            # Get training information
            training_info = optimizer.multi_gpu_trainer.get_training_info()
            self.logger.log_info(f"Auto detection training info: {json.dumps(training_info, indent=2)}")
            
            # Train model
            start_time = time.time()
            results = await optimizer.train()
            training_time = time.time() - start_time
            
            # Get performance summary
            performance_summary = optimizer.multi_gpu_trainer.get_performance_summary()
            
            demo_results = {
                "training_mode": "auto_detection",
                "detected_mode": training_info.get("training_mode", "unknown"),
                "training_time": training_time,
                "results": results,
                "performance_summary": performance_summary,
                "training_info": training_info
            }
            
            self.logger.log_info(f"Auto detection training completed in {training_time:.2f} seconds")
            self.logger.log_info(f"Detected mode: {demo_results['detected_mode']}")
            self.logger.log_info(f"Performance summary: {json.dumps(performance_summary, indent=2)}")
            
            return demo_results
            
        except Exception as e:
            self.logger.log_error(e, "Auto detection demo", "demo_auto_detection")
            return {"error": str(e)}
    
    def demo_gpu_monitoring(self) -> Dict[str, Any]:
        """Demonstrate GPU monitoring capabilities"""
        
        self.logger.log_info("=== GPU Monitoring Demo ===")
        
        try:
            # Create multi-GPU trainer for monitoring
            multi_gpu_config = MultiGPUConfig(
                training_mode="auto",
                enable_gpu_monitoring=True
            )
            
            trainer = create_multi_gpu_trainer(
                multi_gpu_config=multi_gpu_config,
                logger=self.logger
            )
            
            # Get GPU information
            gpu_info = trainer.gpu_monitor.get_gpu_info()
            self.logger.log_info(f"GPU Information: {json.dumps(gpu_info, indent=2)}")
            
            # Simulate GPU monitoring
            self.logger.log_info("Starting GPU monitoring simulation...")
            
            # Create a simple model and run some operations
            model = EmailSequenceModel()
            if torch.cuda.is_available():
                model = model.cuda()
            
            # Record GPU metrics for a few iterations
            for i in range(10):
                # Simulate some GPU operations
                if torch.cuda.is_available():
                    dummy_input = torch.randn(32, 50).cuda()
                    with torch.no_grad():
                        _ = model(dummy_input)
                
                # Record metrics
                trainer.record_gpu_metrics()
                time.sleep(0.1)
            
            # Get monitoring summary
            monitoring_summary = trainer.gpu_monitor.get_gpu_summary()
            
            demo_results = {
                "gpu_info": gpu_info,
                "monitoring_summary": monitoring_summary
            }
            
            self.logger.log_info(f"GPU monitoring summary: {json.dumps(monitoring_summary, indent=2)}")
            
            return demo_results
            
        except Exception as e:
            self.logger.log_error(e, "GPU monitoring demo", "demo_gpu_monitoring")
            return {"error": str(e)}
    
    def demo_performance_comparison(self) -> Dict[str, Any]:
        """Demonstrate performance comparison between different training modes"""
        
        self.logger.log_info("=== Performance Comparison Demo ===")
        
        try:
            # Create dataset
            train_dataset = EmailSequenceDataset(num_samples=1000, sequence_length=50)
            val_dataset = EmailSequenceDataset(num_samples=200, sequence_length=50)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
            
            comparison_results = {}
            
            # Test single GPU
            self.logger.log_info("Testing single GPU performance...")
            model_single = EmailSequenceModel()
            
            single_gpu_config = MultiGPUConfig(
                training_mode="single_gpu",
                enable_data_parallel=False,
                enable_distributed=False,
                enable_gpu_monitoring=True
            )
            
            optimizer_single = create_optimized_training_optimizer(
                model=model_single,
                train_loader=train_loader,
                val_loader=val_loader,
                experiment_name="single_gpu_comparison",
                multi_gpu_config=single_gpu_config,
                max_epochs=2,
                learning_rate=0.001
            )
            
            # Benchmark single GPU
            single_gpu_benchmark = optimizer_single.benchmark_performance(num_iterations=50)
            single_gpu_info = optimizer_single.multi_gpu_trainer.get_training_info()
            
            comparison_results["single_gpu"] = {
                "benchmark": single_gpu_benchmark,
                "training_info": single_gpu_info
            }
            
            # Test DataParallel if multiple GPUs available
            if torch.cuda.device_count() > 1:
                self.logger.log_info("Testing DataParallel performance...")
                model_dp = EmailSequenceModel()
                
                dp_config = MultiGPUConfig(
                    training_mode="data_parallel",
                    enable_data_parallel=True,
                    enable_distributed=False,
                    enable_gpu_monitoring=True
                )
                
                optimizer_dp = create_optimized_training_optimizer(
                    model=model_dp,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    experiment_name="data_parallel_comparison",
                    multi_gpu_config=dp_config,
                    max_epochs=2,
                    learning_rate=0.001
                )
                
                # Benchmark DataParallel
                dp_benchmark = optimizer_dp.benchmark_performance(num_iterations=50)
                dp_info = optimizer_dp.multi_gpu_trainer.get_training_info()
                
                comparison_results["data_parallel"] = {
                    "benchmark": dp_benchmark,
                    "training_info": dp_info
                }
            
            self.logger.log_info(f"Performance comparison results: {json.dumps(comparison_results, indent=2)}")
            
            return comparison_results
            
        except Exception as e:
            self.logger.log_error(e, "Performance comparison demo", "demo_performance_comparison")
            return {"error": str(e)}
    
    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run comprehensive multi-GPU training demonstration"""
        
        self.logger.log_info("Starting comprehensive multi-GPU training demonstration...")
        
        try:
            # Run all demos
            self.results = {
                "gpu_monitoring": self.demo_gpu_monitoring(),
                "performance_comparison": self.demo_performance_comparison(),
                "auto_detection": await self.demo_auto_detection()
            }
            
            # Only run DataParallel and Distributed demos if multiple GPUs are available
            if torch.cuda.device_count() > 1:
                self.results["data_parallel"] = await self.demo_data_parallel_training()
                
                # Note: Distributed training demo requires proper setup
                # self.results["distributed"] = await self.demo_distributed_training()
            else:
                self.logger.log_info("Single GPU detected, skipping multi-GPU specific demos")
            
            # Create comprehensive summary
            summary = {
                "demo_name": self.demo_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_available": torch.cuda.is_available(),
                "results": self.results
            }
            
            # Save results
            results_path = Path("logs/multi_gpu_demo") / f"{self.demo_name}_results.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(summary, f, indent=2)
            
            self.logger.log_info(f"Comprehensive demo completed. Results saved to {results_path}")
            
            return summary
            
        except Exception as e:
            self.logger.log_error(e, "Comprehensive demo", "run_comprehensive_demo")
            return {"error": str(e)}
    
    def print_summary(self) -> Any:
        """Print demonstration summary"""
        
        print("\n" + "="*60)
        print("MULTI-GPU TRAINING DEMONSTRATION SUMMARY")
        print("="*60)
        
        if "error" in self.results:
            print(f"❌ Demo failed: {self.results['error']}")
            return
        
        # GPU Information
        gpu_info = self.results.get("gpu_monitoring", {}).get("gpu_info", {})
        if gpu_info:
            print(f"\n🖥️  GPU Information:")
            print(f"   CUDA Available: {gpu_info.get('cuda_available', False)}")
            print(f"   Device Count: {gpu_info.get('device_count', 0)}")
            
            for device_id, device_info in gpu_info.get("devices", {}).items():
                if "name" in device_info:
                    print(f"   GPU {device_id}: {device_info['name']}")
        
        # Training Modes Tested
        print(f"\n🚀 Training Modes Tested:")
        for demo_name, demo_result in self.results.items():
            if "error" not in demo_result and "training_mode" in demo_result:
                training_mode = demo_result["training_mode"]
                training_time = demo_result.get("training_time", 0)
                print(f"   {demo_name.replace('_', ' ').title()}: {training_mode} ({training_time:.2f}s)")
        
        # Performance Comparison
        perf_comparison = self.results.get("performance_comparison", {})
        if "error" not in perf_comparison:
            print(f"\n⚡ Performance Comparison:")
            for mode, data in perf_comparison.items():
                if "benchmark" in data:
                    benchmark = data["benchmark"]
                    avg_time = benchmark.get("average_forward_time", 0)
                    throughput = benchmark.get("throughput", 0)
                    print(f"   {mode.replace('_', ' ').title()}: {avg_time:.4f}s/step, {throughput:.1f} samples/s")
        
        # Auto Detection Results
        auto_detection = self.results.get("auto_detection", {})
        if "error" not in auto_detection:
            detected_mode = auto_detection.get("detected_mode", "unknown")
            print(f"\n🤖 Auto Detection: {detected_mode}")
        
        print("\n" + "="*60)
        print("Demo completed successfully! Check logs for detailed information.")
        print("="*60)


async def main():
    """Main demonstration function"""
    
    print("Multi-GPU Training System Demonstration")
    print("="*50)
    
    # Create and run demo
    demo = MultiGPUTrainingDemo("comprehensive_multi_gpu_demo")
    
    try:
        results = await demo.run_comprehensive_demo()
        demo.print_summary()
        
        return results
        
    except Exception as e:
        print(f"Demo failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main()) 
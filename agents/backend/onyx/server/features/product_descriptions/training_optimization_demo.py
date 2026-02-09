from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from training_optimization import (
from typing import Any, List, Dict, Optional
import logging
"""
Training Optimization Demo

This demo showcases the training optimization system with:
- Early stopping with different strategies
- Learning rate scheduling with various algorithms
- Gradient optimization and clipping
- Training monitoring and visualization
- Performance comparison between different configurations
- Real-world cybersecurity training scenarios
"""



    EarlyStoppingConfig, LRSchedulerConfig, TrainingOptimizationConfig,
    EarlyStoppingMode, LRSchedulerType, OptimizedTrainer, create_optimized_trainer,
    load_checkpoint
)


class TrainingOptimizationDemo:
    """Comprehensive demo for training optimization."""
    
    def __init__(self) -> Any:
        self.demo_dir = Path("./demo_output")
        self.demo_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.demo_dir / "checkpoints").mkdir(exist_ok=True)
        (self.demo_dir / "plots").mkdir(exist_ok=True)
        (self.demo_dir / "logs").mkdir(exist_ok=True)
        
        self.results = {}
        
    async def run_comprehensive_demo(self) -> Any:
        """Run the complete demo showcasing all features."""
        print("🚀 Starting Training Optimization Demo")
        print("=" * 80)
        
        # Generate synthetic dataset
        await self._generate_synthetic_dataset()
        
        # Demo 1: Basic Early Stopping
        await self._demo_basic_early_stopping()
        
        # Demo 2: Learning Rate Scheduling
        await self._demo_learning_rate_scheduling()
        
        # Demo 3: Gradient Optimization
        await self._demo_gradient_optimization()
        
        # Demo 4: Performance Comparison
        await self._demo_performance_comparison()
        
        # Demo 5: Advanced Configurations
        await self._demo_advanced_configurations()
        
        # Demo 6: Checkpoint Management
        await self._demo_checkpoint_management()
        
        # Demo 7: Training Analysis
        await self._demo_training_analysis()
        
        # Demo 8: Real-world Scenarios
        await self._demo_real_world_scenarios()
        
        # Save results
        self._save_demo_results()
        
        print("\n✅ Demo completed successfully!")
        print(f"Results saved to: {self.demo_dir / 'demo_results.json'}")
    
    async def _generate_synthetic_dataset(self) -> Any:
        """Generate synthetic dataset for demonstration."""
        print("\n📊 Generating Synthetic Dataset...")
        
        # Generate classification data
        X, y = make_classification(
            n_samples=10000,
            n_features=50,
            n_informative=30,
            n_redundant=10,
            n_classes=3,
            n_clusters_per_class=2,
            random_state=42
        )
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.LongTensor(y_val)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)
        
        # Create datasets
        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val, y_val)
        self.test_dataset = TensorDataset(X_test, y_test)
        
        # Create dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        
        print(f"✅ Generated dataset:")
        print(f"   - Train: {len(self.train_dataset)} samples")
        print(f"   - Validation: {len(self.val_dataset)} samples")
        print(f"   - Test: {len(self.test_dataset)} samples")
    
    def _create_simple_model(self) -> nn.Module:
        """Create a simple neural network model."""
        return nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    async def _demo_basic_early_stopping(self) -> Any:
        """Demo basic early stopping functionality."""
        print("\n🛑 Demo 1: Basic Early Stopping")
        print("-" * 50)
        
        # Create configurations
        configs = [
            ("No Early Stopping", None),
            ("Early Stopping (patience=5)", EarlyStoppingConfig(patience=5)),
            ("Early Stopping (patience=10)", EarlyStoppingConfig(patience=10)),
            ("Early Stopping (patience=15)", EarlyStoppingConfig(patience=15))
        ]
        
        results = {}
        
        for name, early_stopping_config in configs:
            print(f"Testing {name}...")
            
            # Create trainer configuration
            if early_stopping_config:
                trainer_config = TrainingOptimizationConfig(
                    early_stopping=early_stopping_config,
                    save_checkpoints=True,
                    checkpoint_dir=str(self.demo_dir / "checkpoints" / name.replace(" ", "_"))
                )
            else:
                trainer_config = TrainingOptimizationConfig(
                    save_checkpoints=True,
                    checkpoint_dir=str(self.demo_dir / "checkpoints" / name.replace(" ", "_"))
                )
            
            # Create trainer
            trainer = OptimizedTrainer(trainer_config)
            
            # Create model and optimizer
            model = self._create_simple_model()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Train model
            start_time = time.time()
            summary = await trainer.train(
                model, self.train_loader, self.val_loader,
                optimizer, criterion, num_epochs=50, device=torch.device('cpu')
            )
            training_time = time.time() - start_time
            
            # Store results
            results[name] = {
                "training_time": training_time,
                "total_epochs": summary["total_epochs"],
                "best_val_loss": summary["best_val_loss"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "final_val_accuracy": summary["final_val_accuracy"]
            }
            
            print(f"   ✅ {name}: {summary['total_epochs']} epochs, "
                  f"Best Val Acc: {summary['best_val_accuracy']:.4f}, "
                  f"Time: {training_time:.2f}s")
        
        self.results["basic_early_stopping"] = results
        
        # Find best configuration
        best_config = max(results.items(), key=lambda x: x[1]["best_val_accuracy"])
        print(f"\n🏆 Best configuration: {best_config[0]} "
              f"(Accuracy: {best_config[1]['best_val_accuracy']:.4f})")
    
    async def _demo_learning_rate_scheduling(self) -> Any:
        """Demo different learning rate scheduling strategies."""
        print("\n📈 Demo 2: Learning Rate Scheduling")
        print("-" * 50)
        
        # Create different scheduler configurations
        scheduler_configs = [
            ("Step LR", LRSchedulerConfig(
                scheduler_type=LRSchedulerType.STEP,
                step_size=10,
                gamma=0.5
            )),
            ("Multi-Step LR", LRSchedulerConfig(
                scheduler_type=LRSchedulerType.MULTI_STEP,
                milestones=[10, 20, 30],
                gamma=0.5
            )),
            ("Exponential LR", LRSchedulerConfig(
                scheduler_type=LRSchedulerType.EXPONENTIAL,
                gamma=0.95
            )),
            ("Cosine Annealing", LRSchedulerConfig(
                scheduler_type=LRSchedulerType.COSINE_ANNEALING,
                T_max=50,
                eta_min=1e-6
            )),
            ("Reduce on Plateau", LRSchedulerConfig(
                scheduler_type=LRSchedulerType.REDUCE_ON_PLATEAU,
                patience=5,
                factor=0.5,
                min_lr=1e-6
            ))
        ]
        
        results = {}
        
        for name, lr_config in scheduler_configs:
            print(f"Testing {name}...")
            
            # Create trainer configuration
            trainer_config = TrainingOptimizationConfig(
                lr_scheduler=lr_config,
                early_stopping=EarlyStoppingConfig(patience=15),
                save_checkpoints=True,
                checkpoint_dir=str(self.demo_dir / "checkpoints" / name.replace(" ", "_"))
            )
            
            # Create trainer
            trainer = OptimizedTrainer(trainer_config)
            
            # Create model and optimizer
            model = self._create_simple_model()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Train model
            start_time = time.time()
            summary = await trainer.train(
                model, self.train_loader, self.val_loader,
                optimizer, criterion, num_epochs=50, device=torch.device('cpu')
            )
            training_time = time.time() - start_time
            
            # Store results
            results[name] = {
                "training_time": training_time,
                "total_epochs": summary["total_epochs"],
                "best_val_loss": summary["best_val_loss"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "final_val_accuracy": summary["final_val_accuracy"]
            }
            
            print(f"   ✅ {name}: {summary['total_epochs']} epochs, "
                  f"Best Val Acc: {summary['best_val_accuracy']:.4f}, "
                  f"Time: {training_time:.2f}s")
        
        self.results["learning_rate_scheduling"] = results
        
        # Find best scheduler
        best_scheduler = max(results.items(), key=lambda x: x[1]["best_val_accuracy"])
        print(f"\n🏆 Best scheduler: {best_scheduler[0]} "
              f"(Accuracy: {best_scheduler[1]['best_val_accuracy']:.4f})")
    
    async def _demo_gradient_optimization(self) -> Any:
        """Demo gradient optimization techniques."""
        print("\n⚡ Demo 3: Gradient Optimization")
        print("-" * 50)
        
        # Create different gradient optimization configurations
        gradient_configs = [
            ("No Gradient Clipping", TrainingOptimizationConfig(
                gradient_clip_norm=None,
                gradient_clip_value=None
            )),
            ("Gradient Clipping (norm=1.0)", TrainingOptimizationConfig(
                gradient_clip_norm=1.0,
                gradient_clip_value=None
            )),
            ("Gradient Clipping (norm=0.5)", TrainingOptimizationConfig(
                gradient_clip_norm=0.5,
                gradient_clip_value=None
            )),
            ("Gradient Clipping (value=0.1)", TrainingOptimizationConfig(
                gradient_clip_norm=None,
                gradient_clip_value=0.1
            )),
            ("Gradient Accumulation", TrainingOptimizationConfig(
                gradient_clip_norm=1.0,
                gradient_accumulation_steps=2
            ))
        ]
        
        results = {}
        
        for name, config in gradient_configs:
            print(f"Testing {name}...")
            
            # Add early stopping to config
            config.early_stopping = EarlyStoppingConfig(patience=15)
            config.save_checkpoints = True
            config.checkpoint_dir = str(self.demo_dir / "checkpoints" / name.replace(" ", "_"))
            
            # Create trainer
            trainer = OptimizedTrainer(config)
            
            # Create model and optimizer
            model = self._create_simple_model()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Train model
            start_time = time.time()
            summary = await trainer.train(
                model, self.train_loader, self.val_loader,
                optimizer, criterion, num_epochs=50, device=torch.device('cpu')
            )
            training_time = time.time() - start_time
            
            # Store results
            results[name] = {
                "training_time": training_time,
                "total_epochs": summary["total_epochs"],
                "best_val_loss": summary["best_val_loss"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "gradient_stats": summary.get("gradient_stats", {})
            }
            
            print(f"   ✅ {name}: {summary['total_epochs']} epochs, "
                  f"Best Val Acc: {summary['best_val_accuracy']:.4f}, "
                  f"Time: {training_time:.2f}s")
        
        self.results["gradient_optimization"] = results
        
        # Find best gradient optimization
        best_gradient = max(results.items(), key=lambda x: x[1]["best_val_accuracy"])
        print(f"\n🏆 Best gradient optimization: {best_gradient[0]} "
              f"(Accuracy: {best_gradient[1]['best_val_accuracy']:.4f})")
    
    async def _demo_performance_comparison(self) -> Any:
        """Demo performance comparison between different configurations."""
        print("\n📊 Demo 4: Performance Comparison")
        print("-" * 50)
        
        # Create optimized configurations
        configurations = [
            ("Baseline", TrainingOptimizationConfig()),
            ("Optimized", TrainingOptimizationConfig(
                early_stopping=EarlyStoppingConfig(patience=10),
                lr_scheduler=LRSchedulerConfig(
                    scheduler_type=LRSchedulerType.REDUCE_ON_PLATEAU,
                    patience=5,
                    factor=0.5
                ),
                gradient_clip_norm=1.0,
                save_checkpoints=True
            )),
            ("Highly Optimized", TrainingOptimizationConfig(
                early_stopping=EarlyStoppingConfig(patience=15, min_delta=1e-4),
                lr_scheduler=LRSchedulerConfig(
                    scheduler_type=LRSchedulerType.COSINE_ANNEALING,
                    T_max=50,
                    eta_min=1e-6
                ),
                gradient_clip_norm=1.0,
                gradient_accumulation_steps=2,
                save_checkpoints=True,
                save_best_only=True
            ))
        ]
        
        results = {}
        
        for name, config in configurations:
            print(f"Testing {name} configuration...")
            
            # Set checkpoint directory
            config.checkpoint_dir = str(self.demo_dir / "checkpoints" / name.replace(" ", "_"))
            
            # Create trainer
            trainer = OptimizedTrainer(config)
            
            # Create model and optimizer
            model = self._create_simple_model()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Train model
            start_time = time.time()
            summary = await trainer.train(
                model, self.train_loader, self.val_loader,
                optimizer, criterion, num_epochs=50, device=torch.device('cpu')
            )
            training_time = time.time() - start_time
            
            # Store results
            results[name] = {
                "training_time": training_time,
                "total_epochs": summary["total_epochs"],
                "best_val_loss": summary["best_val_loss"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "final_val_accuracy": summary["final_val_accuracy"],
                "avg_epoch_time": summary["avg_epoch_time"]
            }
            
            print(f"   ✅ {name}: {summary['total_epochs']} epochs, "
                  f"Best Val Acc: {summary['best_val_accuracy']:.4f}, "
                  f"Time: {training_time:.2f}s, "
                  f"Avg Epoch: {summary['avg_epoch_time']:.2f}s")
        
        self.results["performance_comparison"] = results
        
        # Find best overall configuration
        best_config = max(results.items(), key=lambda x: x[1]["best_val_accuracy"])
        print(f"\n🏆 Best overall configuration: {best_config[0]} "
              f"(Accuracy: {best_config[1]['best_val_accuracy']:.4f})")
    
    async def _demo_advanced_configurations(self) -> Any:
        """Demo advanced training configurations."""
        print("\n🔬 Demo 5: Advanced Configurations")
        print("-" * 50)
        
        # Create advanced configurations
        advanced_configs = [
            ("One Cycle Policy", TrainingOptimizationConfig(
                lr_scheduler=LRSchedulerConfig(
                    scheduler_type=LRSchedulerType.ONE_CYCLE,
                    max_lr=1e-2,
                    epochs=50,
                    steps_per_epoch=len(self.train_loader),
                    pct_start=0.3
                ),
                early_stopping=EarlyStoppingConfig(patience=20),
                gradient_clip_norm=1.0
            )),
            ("Cyclic LR", TrainingOptimizationConfig(
                lr_scheduler=LRSchedulerConfig(
                    scheduler_type=LRSchedulerType.CYCLIC,
                    base_lr=1e-6,
                    max_lr=1e-3,
                    step_size_up=len(self.train_loader) * 5,
                    step_size_down=len(self.train_loader) * 5
                ),
                early_stopping=EarlyStoppingConfig(patience=15)
            )),
            ("Cosine Annealing with Warm Restarts", TrainingOptimizationConfig(
                lr_scheduler=LRSchedulerConfig(
                    scheduler_type=LRSchedulerType.COSINE_ANNEALING_WARM_RESTARTS,
                    T_0=10,
                    T_mult=2,
                    eta_min=1e-6
                ),
                early_stopping=EarlyStoppingConfig(patience=25)
            ))
        ]
        
        results = {}
        
        for name, config in advanced_configs:
            print(f"Testing {name}...")
            
            # Set checkpoint directory
            config.checkpoint_dir = str(self.demo_dir / "checkpoints" / name.replace(" ", "_"))
            config.save_checkpoints = True
            
            # Create trainer
            trainer = OptimizedTrainer(config)
            
            # Create model and optimizer
            model = self._create_simple_model()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Train model
            start_time = time.time()
            summary = await trainer.train(
                model, self.train_loader, self.val_loader,
                optimizer, criterion, num_epochs=50, device=torch.device('cpu')
            )
            training_time = time.time() - start_time
            
            # Store results
            results[name] = {
                "training_time": training_time,
                "total_epochs": summary["total_epochs"],
                "best_val_loss": summary["best_val_loss"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "final_val_accuracy": summary["final_val_accuracy"]
            }
            
            print(f"   ✅ {name}: {summary['total_epochs']} epochs, "
                  f"Best Val Acc: {summary['best_val_accuracy']:.4f}, "
                  f"Time: {training_time:.2f}s")
        
        self.results["advanced_configurations"] = results
        
        # Find best advanced configuration
        best_advanced = max(results.items(), key=lambda x: x[1]["best_val_accuracy"])
        print(f"\n🏆 Best advanced configuration: {best_advanced[0]} "
              f"(Accuracy: {best_advanced[1]['best_val_accuracy']:.4f})")
    
    async def _demo_checkpoint_management(self) -> Any:
        """Demo checkpoint management and loading."""
        print("\n💾 Demo 6: Checkpoint Management")
        print("-" * 50)
        
        # Create trainer with checkpointing
        config = TrainingOptimizationConfig(
            early_stopping=EarlyStoppingConfig(patience=10),
            save_checkpoints=True,
            save_best_only=True,
            save_last=True,
            checkpoint_dir=str(self.demo_dir / "checkpoints" / "checkpoint_demo")
        )
        
        trainer = OptimizedTrainer(config)
        
        # Create model and optimizer
        model = self._create_simple_model()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        print("Training model with checkpointing...")
        summary = await trainer.train(
            model, self.train_loader, self.val_loader,
            optimizer, criterion, num_epochs=30, device=torch.device('cpu')
        )
        
        # Check checkpoint files
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        
        print(f"✅ Generated {len(checkpoint_files)} checkpoint files:")
        for checkpoint_file in checkpoint_files:
            print(f"   - {checkpoint_file.name}")
        
        # Load and test checkpoint
        if checkpoint_files:
            best_checkpoint = checkpoint_dir / "best_model_epoch_0.pt"
            if best_checkpoint.exists():
                print(f"\nLoading checkpoint: {best_checkpoint}")
                
                # Create new model and optimizer
                new_model = self._create_simple_model()
                new_optimizer = optim.Adam(new_model.parameters(), lr=1e-3)
                
                # Load checkpoint
                checkpoint_data = load_checkpoint(new_model, new_optimizer, str(best_checkpoint))
                
                print(f"✅ Loaded checkpoint from epoch {checkpoint_data['epoch']}")
                print(f"   - Validation loss: {checkpoint_data['metrics']['val_loss']:.4f}")
                print(f"   - Validation accuracy: {checkpoint_data['metrics']['val_accuracy']:.4f}")
        
        self.results["checkpoint_management"] = {
            "total_checkpoints": len(checkpoint_files),
            "checkpoint_files": [str(f) for f in checkpoint_files],
            "training_summary": summary
        }
    
    async def _demo_training_analysis(self) -> Any:
        """Demo training analysis and visualization."""
        print("\n📈 Demo 7: Training Analysis")
        print("-" * 50)
        
        # Create trainer with monitoring
        config = TrainingOptimizationConfig(
            early_stopping=EarlyStoppingConfig(patience=15),
            lr_scheduler=LRSchedulerConfig(
                scheduler_type=LRSchedulerType.REDUCE_ON_PLATEAU,
                patience=5,
                factor=0.5
            ),
            gradient_clip_norm=1.0,
            save_checkpoints=True,
            checkpoint_dir=str(self.demo_dir / "checkpoints" / "analysis_demo"),
            tensorboard_logging=True
        )
        
        trainer = OptimizedTrainer(config)
        
        # Create model and optimizer
        model = self._create_simple_model()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        print("Training model with comprehensive monitoring...")
        summary = await trainer.train(
            model, self.train_loader, self.val_loader,
            optimizer, criterion, num_epochs=40, device=torch.device('cpu')
        )
        
        # Generate analysis plots
        print("Generating training analysis plots...")
        plots_dir = self.demo_dir / "plots"
        
        # Training curves
        trainer.plot_training_analysis(str(plots_dir / "training_analysis.png"))
        
        # Early stopping analysis
        early_stopping_history = summary.get("early_stopping_history", [])
        if early_stopping_history:
            print(f"✅ Early stopping triggered at epoch {len(early_stopping_history)}")
        
        # Gradient statistics
        gradient_stats = summary.get("gradient_stats", {})
        if gradient_stats:
            print("✅ Gradient statistics:")
            for key, value in gradient_stats.items():
                print(f"   - {key}: {value:.4f}")
        
        self.results["training_analysis"] = {
            "training_summary": summary,
            "early_stopping_history": early_stopping_history,
            "gradient_stats": gradient_stats,
            "plots_generated": [
                str(plots_dir / "training_analysis.png")
            ]
        }
    
    async def _demo_real_world_scenarios(self) -> Any:
        """Demo real-world cybersecurity training scenarios."""
        print("\n🛡️ Demo 8: Real-world Scenarios")
        print("-" * 50)
        
        # Scenario 1: Threat Detection with Limited Data
        print("Scenario 1: Threat Detection with Limited Data")
        scenario1_config = TrainingOptimizationConfig(
            early_stopping=EarlyStoppingConfig(
                patience=20,
                min_delta=1e-4,
                min_epochs=10
            ),
            lr_scheduler=LRSchedulerConfig(
                scheduler_type=LRSchedulerType.COSINE_ANNEALING,
                T_max=100,
                eta_min=1e-6
            ),
            gradient_clip_norm=1.0,
            save_checkpoints=True,
            checkpoint_dir=str(self.demo_dir / "checkpoints" / "scenario1")
        )
        
        trainer1 = OptimizedTrainer(scenario1_config)
        model1 = self._create_simple_model()
        optimizer1 = optim.Adam(model1.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion1 = nn.CrossEntropyLoss()
        
        summary1 = await trainer1.train(
            model1, self.train_loader, self.val_loader,
            optimizer1, criterion1, num_epochs=100, device=torch.device('cpu')
        )
        
        # Scenario 2: Anomaly Detection with Imbalanced Data
        print("Scenario 2: Anomaly Detection with Imbalanced Data")
        scenario2_config = TrainingOptimizationConfig(
            early_stopping=EarlyStoppingConfig(
                patience=15,
                mode=EarlyStoppingMode.MAX,
                monitor="val_f1"
            ),
            lr_scheduler=LRSchedulerConfig(
                scheduler_type=LRSchedulerType.REDUCE_ON_PLATEAU,
                patience=7,
                factor=0.3,
                mode="max"
            ),
            gradient_clip_norm=0.5,
            save_checkpoints=True,
            checkpoint_dir=str(self.demo_dir / "checkpoints" / "scenario2")
        )
        
        trainer2 = OptimizedTrainer(scenario2_config)
        model2 = self._create_simple_model()
        optimizer2 = optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion2 = nn.CrossEntropyLoss()
        
        summary2 = await trainer2.train(
            model2, self.train_loader, self.val_loader,
            optimizer2, criterion2, num_epochs=80, device=torch.device('cpu')
        )
        
        # Scenario 3: Real-time Training with Fast Convergence
        print("Scenario 3: Real-time Training with Fast Convergence")
        scenario3_config = TrainingOptimizationConfig(
            early_stopping=EarlyStoppingConfig(
                patience=8,
                min_delta=1e-3,
                min_epochs=5
            ),
            lr_scheduler=LRSchedulerConfig(
                scheduler_type=LRSchedulerType.ONE_CYCLE,
                max_lr=1e-2,
                epochs=50,
                steps_per_epoch=len(self.train_loader)
            ),
            gradient_clip_norm=1.0,
            gradient_accumulation_steps=2,
            save_checkpoints=True,
            checkpoint_dir=str(self.demo_dir / "checkpoints" / "scenario3")
        )
        
        trainer3 = OptimizedTrainer(scenario3_config)
        model3 = self._create_simple_model()
        optimizer3 = optim.Adam(model3.parameters(), lr=1e-3)
        criterion3 = nn.CrossEntropyLoss()
        
        summary3 = await trainer3.train(
            model3, self.train_loader, self.val_loader,
            optimizer3, criterion3, num_epochs=50, device=torch.device('cpu')
        )
        
        # Compare scenarios
        scenarios = {
            "Threat Detection (Limited Data)": summary1,
            "Anomaly Detection (Imbalanced)": summary2,
            "Real-time Training (Fast Convergence)": summary3
        }
        
        print("\n📊 Scenario Comparison:")
        for name, summary in scenarios.items():
            print(f"   {name}:")
            print(f"     - Epochs: {summary['total_epochs']}")
            print(f"     - Best Val Accuracy: {summary['best_val_accuracy']:.4f}")
            print(f"     - Training Time: {summary['total_training_time']:.2f}s")
            print(f"     - Avg Epoch Time: {summary['avg_epoch_time']:.2f}s")
        
        self.results["real_world_scenarios"] = scenarios
        
        # Find best scenario
        best_scenario = max(scenarios.items(), key=lambda x: x[1]["best_val_accuracy"])
        print(f"\n🏆 Best scenario: {best_scenario[0]} "
              f"(Accuracy: {best_scenario[1]['best_val_accuracy']:.4f})")
    
    def _save_demo_results(self) -> Any:
        """Save demo results to file."""
        results_file = self.demo_dir / "demo_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Convert results to JSON-serializable format
        serializable_results = json.loads(
            json.dumps(self.results, default=convert_numpy, indent=2)
        )
        
        with open(results_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n📁 Demo results saved to: {results_file}")
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self) -> Any:
        """Generate a summary report of the demo."""
        report_file = self.demo_dir / "demo_summary.md"
        
        with open(report_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("# Training Optimization Demo Summary\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Demo Overview\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("This demo showcases comprehensive training optimization capabilities for cybersecurity applications.\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Key Features Demonstrated\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("1. **Early Stopping** - Multiple strategies for preventing overfitting\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("2. **Learning Rate Scheduling** - Various LR scheduling algorithms\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("3. **Gradient Optimization** - Gradient clipping and accumulation\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("4. **Performance Comparison** - Comparing different configurations\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("5. **Advanced Configurations** - One-cycle, cyclic, and warm restart policies\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("6. **Checkpoint Management** - Saving and loading model checkpoints\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("7. **Training Analysis** - Comprehensive monitoring and visualization\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("8. **Real-world Scenarios** - Practical cybersecurity training scenarios\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Results Summary\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if "basic_early_stopping" in self.results:
                f.write("### Early Stopping Results\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                for name, results in self.results["basic_early_stopping"].items():
                    f.write(f"- **{name}**: {results['total_epochs']} epochs, "
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                           f"Best Val Acc: {results['best_val_accuracy']:.4f}, "
                           f"Time: {results['training_time']:.2f}s\n")
                f.write("\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if "learning_rate_scheduling" in self.results:
                f.write("### Learning Rate Scheduling Results\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                for name, results in self.results["learning_rate_scheduling"].items():
                    f.write(f"- **{name}**: {results['total_epochs']} epochs, "
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                           f"Best Val Acc: {results['best_val_accuracy']:.4f}, "
                           f"Time: {results['training_time']:.2f}s\n")
                f.write("\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if "performance_comparison" in self.results:
                f.write("### Performance Comparison Results\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                for name, results in self.results["performance_comparison"].items():
                    f.write(f"- **{name}**: {results['total_epochs']} epochs, "
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                           f"Best Val Acc: {results['best_val_accuracy']:.4f}, "
                           f"Time: {results['training_time']:.2f}s\n")
                f.write("\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if "real_world_scenarios" in self.results:
                f.write("### Real-world Scenarios Results\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                for name, results in self.results["real_world_scenarios"].items():
                    f.write(f"- **{name}**: {results['total_epochs']} epochs, "
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                           f"Best Val Acc: {results['best_val_accuracy']:.4f}, "
                           f"Time: {results['total_training_time']:.2f}s\n")
                f.write("\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Best Practices\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("1. **Use early stopping** to prevent overfitting and save training time\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("2. **Choose appropriate LR schedulers** based on your data and model\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("3. **Apply gradient clipping** for stable training\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("4. **Monitor training metrics** to understand model behavior\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("5. **Save checkpoints** for model recovery and deployment\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("6. **Use appropriate configurations** for different scenarios\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("7. **Analyze training curves** to optimize hyperparameters\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("8. **Consider real-world constraints** in training optimization\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Files Generated\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("- `demo_results.json` - Complete demo results\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("- `checkpoints/` - Model checkpoints\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("- `plots/` - Training analysis plots\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("- `logs/` - Training logs\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("## Next Steps\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("1. Integrate with your existing training pipeline\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("2. Customize configurations for your specific use case\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("3. Monitor performance in production\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("4. Implement additional optimizations as needed\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("5. Add more cybersecurity-specific scenarios\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        print(f"📋 Summary report generated: {report_file}")


async def main():
    """Main demo function."""
    demo = TrainingOptimizationDemo()
    await demo.run_comprehensive_demo()


match __name__:
    case "__main__":
    asyncio.run(main()) 
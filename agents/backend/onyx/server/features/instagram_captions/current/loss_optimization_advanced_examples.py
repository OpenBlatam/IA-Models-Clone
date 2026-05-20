"""
Advanced Loss Functions and Optimization Examples

This module provides advanced examples demonstrating loss functions and optimization
algorithms integrated with custom model architectures. It includes:

1. Advanced loss function combinations and custom losses
2. Optimization algorithm comparisons and analysis
3. Integration with custom model architectures
4. Loss landscape analysis and gradient flow visualization
5. Real-world training scenarios with different loss-optimizer combinations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

# Import loss optimization system
try:
    from loss_optimization_system import (
        LossFunctions, Optimizers, LearningRateSchedulers,
        LossOptimizationAnalyzer, CustomLossOptimizationSchemes
    )
    LOSS_OPT_AVAILABLE = True
except ImportError:
    print("Warning: loss_optimization_system not found. Some examples may not work.")
    LOSS_OPT_AVAILABLE = False

# Import custom model architectures
try:
    from custom_model_architectures import (
        CustomTransformerModel, CustomCNNModel, CustomRNNModel, 
        CNNTransformerHybrid, create_model_from_config
    )
    CUSTOM_MODELS_AVAILABLE = True
except ImportError:
    print("Warning: custom_model_architectures not found. Some examples may not work.")
    CUSTOM_MODELS_AVAILABLE = False


class AdvancedLossOptimizationExamples:
    """Advanced examples demonstrating sophisticated loss and optimization techniques."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize components
        self.setup_components()
    
    def load_config(self) -> Dict:
        """Load loss optimization configuration."""
        try:
            with open('loss_optimization_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            print("Loss optimization configuration loaded successfully")
            return config
        except FileNotFoundError:
            print("Configuration file not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default configuration if file not found."""
        return {
            'loss_functions': {
                'global': {'default_loss': 'cross_entropy'},
                'classification': {'cross_entropy': {'enabled': True}}
            },
            'optimization_algorithms': {
                'global': {'default_optimizer': 'adam'},
                'adaptive': {'adam': {'enabled': True}}
            },
            'learning_rate_schedulers': {
                'global': {'default_scheduler': 'step'},
                'step_based': {'step_lr': {'enabled': True}}
            }
        }
    
    def setup_components(self):
        """Setup all system components."""
        print("\n=== Setting up Loss Optimization Components ===")
        
        # Create simple models for demonstration
        self.simple_model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 50)
        ).to(self.device)
        
        # Create custom models if available
        if CUSTOM_MODELS_AVAILABLE:
            self.transformer_model = CustomTransformerModel(
                vocab_size=1000, d_model=128, nhead=8, num_layers=4
            ).to(self.device)
            
            self.cnn_model = CustomCNNModel(
                input_channels=3, num_classes=10, base_channels=32
            ).to(self.device)
            
            self.rnn_model = CustomRNNModel(
                input_size=100, hidden_size=128, num_layers=3, num_classes=5
            ).to(self.device)
            
            self.hybrid_model = CNNTransformerHybrid(
                input_channels=3, num_classes=10, d_model=128, nhead=8
            ).to(self.device)
        
        print("Components setup completed!")
    
    def demonstrate_loss_function_comparison(self):
        """Demonstrate comparison of different loss functions."""
        print("\n=== Loss Function Comparison ===")
        
        if not LOSS_OPT_AVAILABLE:
            print("Loss optimization system not available, skipping...")
            return
        
        # Create sample data
        predictions = torch.randn(100, 10, device=self.device)
        targets = torch.randint(0, 10, (100,), device=self.device)
        
        # Test different loss functions
        loss_functions = {
            'Cross Entropy': lambda p, t: LossFunctions.cross_entropy_loss(p, t),
            'Focal Loss': lambda p, t: LossFunctions.focal_loss(p, t),
            'Huber Loss': lambda p, t: LossFunctions.huber_loss(p, t),
            'Smooth L1': lambda p, t: LossFunctions.smooth_l1_loss(p, t)
        }
        
        loss_results = {}
        
        for name, loss_fn in loss_functions.items():
            try:
                loss_value = loss_fn(predictions, targets).item()
                loss_results[name] = loss_value
                print(f"  {name}: {loss_value:.4f}")
            except Exception as e:
                print(f"  {name}: Error - {e}")
        
        # Find best loss function (lowest value)
        if loss_results:
            best_loss = min(loss_results.items(), key=lambda x: x[1])
            print(f"\nBest loss function: {best_loss[0]} ({best_loss[1]:.4f})")
        
        print("Loss function comparison completed!")
    
    def demonstrate_optimizer_comparison(self):
        """Demonstrate comparison of different optimization algorithms."""
        print("\n=== Optimizer Comparison ===")
        
        if not LOSS_OPT_AVAILABLE:
            print("Loss optimization system not available, skipping...")
            return
        
        # Create identical models for comparison
        models = {}
        optimizers = {}
        
        optimizer_configs = {
            'SGD': {'type': 'sgd', 'lr': 0.01, 'momentum': 0.9},
            'Adam': {'type': 'adam', 'lr': 0.001},
            'AdamW': {'type': 'adamw', 'lr': 0.001},
            'RMSprop': {'type': 'rmsprop', 'lr': 0.01},
            'Adagrad': {'type': 'adagrad', 'lr': 0.01}
        }
        
        for name, config in optimizer_configs.items():
            # Create model
            model = nn.Sequential(
                nn.Linear(50, 100),
                nn.ReLU(),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            ).to(self.device)
            
            # Create optimizer
            optimizer = Optimizers.custom_optimizer(model, config['type'], **config)
            
            models[name] = model
            optimizers[name] = optimizer
        
        # Create training data
        x_train = torch.randn(1000, 50, device=self.device)
        y_train = torch.randint(0, 10, (1000,), device=self.device)
        
        # Training comparison
        training_results = {}
        loss_fn = LossFunctions.cross_entropy_loss
        
        for name, model in models.items():
            print(f"\nTraining with {name} optimizer:")
            
            optimizer = optimizers[name]
            losses = []
            
            # Training loop
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = model(x_train)
                loss = loss_fn(outputs, y_train)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
                if epoch % 3 == 0:
                    print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
            
            training_results[name] = {
                'final_loss': losses[-1],
                'loss_history': losses,
                'convergence_rate': (losses[0] - losses[-1]) / losses[0]
            }
        
        # Compare results
        print("\nOptimizer comparison results:")
        for name, results in training_results.items():
            print(f"  {name}:")
            print(f"    Final loss: {results['final_loss']:.6f}")
            print(f"    Convergence rate: {results['convergence_rate']:.2%}")
        
        # Find best optimizer
        best_optimizer = min(training_results.keys(), 
                           key=lambda x: training_results[x]['final_loss'])
        print(f"\nBest optimizer: {best_optimizer}")
        print("Optimizer comparison completed!")
    
    def demonstrate_learning_rate_scheduler_comparison(self):
        """Demonstrate comparison of different learning rate schedulers."""
        print("\n=== Learning Rate Scheduler Comparison ===")
        
        if not LOSS_OPT_AVAILABLE:
            print("Loss optimization system not available, skipping...")
            return
        
        # Create model and optimizer
        model = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        ).to(self.device)
        
        optimizer = Optimizers.adam_optimizer(model, lr=0.01)
        
        # Create different schedulers
        schedulers = {
            'Step LR': LearningRateSchedulers.step_scheduler(optimizer, step_size=5, gamma=0.5),
            'Exponential LR': LearningRateSchedulers.exponential_scheduler(optimizer, gamma=0.95),
            'Cosine Annealing': LearningRateSchedulers.cosine_scheduler(optimizer, T_max=20),
            'Reduce LR on Plateau': LearningRateSchedulers.reduce_lr_on_plateau_scheduler(
                optimizer, patience=3, factor=0.5
            )
        }
        
        # Create training data
        x_train = torch.randn(1000, 50, device=self.device)
        y_train = torch.randint(0, 10, (1000,), device=self.device)
        
        # Training with different schedulers
        scheduler_results = {}
        loss_fn = LossFunctions.cross_entropy_loss
        
        for name, scheduler in schedulers.items():
            print(f"\nTraining with {name} scheduler:")
            
            # Reset model and optimizer
            model = nn.Sequential(
                nn.Linear(50, 100),
                nn.ReLU(),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            ).to(self.device)
            
            optimizer = Optimizers.adam_optimizer(model, lr=0.01)
            
            # Recreate scheduler with new optimizer
            if name == 'Step LR':
                scheduler = LearningRateSchedulers.step_scheduler(optimizer, step_size=5, gamma=0.5)
            elif name == 'Exponential LR':
                scheduler = LearningRateSchedulers.exponential_scheduler(optimizer, gamma=0.95)
            elif name == 'Cosine Annealing':
                scheduler = LearningRateSchedulers.cosine_scheduler(optimizer, T_max=20)
            elif name == 'Reduce LR on Plateau':
                scheduler = LearningRateSchedulers.reduce_lr_on_plateau_scheduler(
                    optimizer, patience=3, factor=0.5
                )
            
            losses = []
            learning_rates = []
            
            # Training loop
            for epoch in range(20):
                optimizer.zero_grad()
                outputs = model(x_train)
                loss = loss_fn(outputs, y_train)
                loss.backward()
                optimizer.step()
                
                # Step scheduler (except ReduceLROnPlateau)
                if name != 'Reduce LR on Plateau':
                    scheduler.step()
                else:
                    scheduler.step(loss)
                
                losses.append(loss.item())
                learning_rates.append(optimizer.param_groups[0]['lr'])
                
                if epoch % 5 == 0:
                    print(f"  Epoch {epoch}: Loss = {loss.item():.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
            
            scheduler_results[name] = {
                'losses': losses,
                'learning_rates': learning_rates,
                'final_loss': losses[-1],
                'final_lr': learning_rates[-1]
            }
        
        # Compare results
        print("\nScheduler comparison results:")
        for name, results in scheduler_results.items():
            print(f"  {name}:")
            print(f"    Final loss: {results['final_loss']:.6f}")
            print(f"    Final LR: {results['final_lr']:.6f}")
            print(f"    LR reduction: {(0.01 - results['final_lr']) / 0.01:.2%}")
        
        print("Learning rate scheduler comparison completed!")
    
    def demonstrate_loss_landscape_analysis(self):
        """Demonstrate loss landscape analysis."""
        print("\n=== Loss Landscape Analysis ===")
        
        if not LOSS_OPT_AVAILABLE:
            print("Loss optimization system not available, skipping...")
            return
        
        # Create model and data
        model = nn.Sequential(
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        ).to(self.device)
        
        data = torch.randn(100, 20, device=self.device)
        targets = torch.randint(0, 10, (100,), device=self.device)
        loss_fn = LossFunctions.cross_entropy_loss
        
        # Analyze loss landscape
        print("Analyzing loss landscape...")
        landscape_analysis = LossOptimizationAnalyzer.analyze_loss_landscape(
            model, data, targets, loss_fn, num_points=50
        )
        
        print(f"Current loss: {landscape_analysis['current_loss']:.4f}")
        print(f"Loss standard deviation: {landscape_analysis['loss_std']:.4f}")
        print(f"Loss range: [{landscape_analysis['loss_range'][0]:.4f}, {landscape_analysis['loss_range'][1]:.4f}]")
        print(f"Loss variance: {landscape_analysis['loss_variance']:.4f}")
        
        # Analyze gradient flow
        print("\nAnalyzing gradient flow...")
        gradient_analysis = LossOptimizationAnalyzer.analyze_gradient_flow(
            model, loss_fn, data, targets
        )
        
        print(f"Total gradient norm: {gradient_analysis['total_gradient_norm']:.4f}")
        print(f"Loss value: {gradient_analysis['loss_value']:.4f}")
        
        # Print gradient statistics for each layer
        for name, stats in gradient_analysis['gradient_stats'].items():
            print(f"  {name}: norm={stats['norm']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        print("Loss landscape analysis completed!")
    
    def demonstrate_task_specific_schemes(self):
        """Demonstrate task-specific loss and optimization schemes."""
        print("\n=== Task-Specific Schemes ===")
        
        if not LOSS_OPT_AVAILABLE:
            print("Loss optimization system not available, skipping...")
            return
        
        # Test classification scheme
        print("\n1. Testing Classification Scheme:")
        classification_model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        ).to(self.device)
        
        loss_fn, optimizer, scheduler = CustomLossOptimizationSchemes.classification_scheme(
            classification_model, num_classes=10
        )
        
        print(f"  Loss function: {loss_fn.__name__ if hasattr(loss_fn, '__name__') else 'Custom'}")
        print(f"  Optimizer: {type(optimizer).__name__}")
        print(f"  Scheduler: {type(scheduler).__name__}")
        
        # Test regression scheme
        print("\n2. Testing Regression Scheme:")
        regression_model = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        ).to(self.device)
        
        loss_fn, optimizer, scheduler = CustomLossOptimizationSchemes.regression_scheme(
            regression_model, loss_type='mse'
        )
        
        print(f"  Loss function: {loss_fn.__name__ if hasattr(loss_fn, '__name__') else 'Custom'}")
        print(f"  Optimizer: {type(optimizer).__name__}")
        print(f"  Scheduler: {type(scheduler).__name__}")
        
        # Test segmentation scheme
        print("\n3. Testing Segmentation Scheme:")
        segmentation_model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 5)
        ).to(self.device)
        
        loss_fn, optimizer, scheduler = CustomLossOptimizationSchemes.segmentation_scheme(
            segmentation_model, num_classes=5
        )
        
        print(f"  Loss function: {loss_fn.__name__ if hasattr(loss_fn, '__name__') else 'Custom'}")
        print(f"  Optimizer: {type(optimizer).__name__}")
        print(f"  Scheduler: {type(scheduler).__name__}")
        
        # Test metric learning scheme
        print("\n4. Testing Metric Learning Scheme:")
        metric_model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 64)  # Embedding dimension
        ).to(self.device)
        
        loss_fn, optimizer, scheduler = CustomLossOptimizationSchemes.metric_learning_scheme(
            metric_model, margin=1.0
        )
        
        print(f"  Loss function: {loss_fn.__name__ if hasattr(loss_fn, '__name__') else 'Custom'}")
        print(f"  Optimizer: {type(optimizer).__name__}")
        print(f"  Scheduler: {type(scheduler).__name__}")
        
        print("Task-specific schemes demonstration completed!")
    
    def demonstrate_custom_model_integration(self):
        """Demonstrate integration with custom model architectures."""
        print("\n=== Custom Model Integration ===")
        
        if not all([LOSS_OPT_AVAILABLE, CUSTOM_MODELS_AVAILABLE]):
            print("Required components not available, skipping...")
            return
        
        # Test with transformer model
        print("\n1. Testing Transformer Model:")
        transformer_loss_fn, transformer_optimizer, transformer_scheduler = (
            CustomLossOptimizationSchemes.classification_scheme(
                self.transformer_model, num_classes=1000
            )
        )
        
        print(f"  Transformer - Loss: {transformer_loss_fn.__name__ if hasattr(transformer_loss_fn, '__name__') else 'Custom'}")
        print(f"  Transformer - Optimizer: {type(transformer_optimizer).__name__}")
        print(f"  Transformer - Scheduler: {type(transformer_scheduler).__name__}")
        
        # Test with CNN model
        print("\n2. Testing CNN Model:")
        cnn_loss_fn, cnn_optimizer, cnn_scheduler = (
            CustomLossOptimizationSchemes.classification_scheme(
                self.cnn_model, num_classes=10
            )
        )
        
        print(f"  CNN - Loss: {cnn_loss_fn.__name__ if hasattr(cnn_loss_fn, '__name__') else 'Custom'}")
        print(f"  CNN - Optimizer: {type(cnn_optimizer).__name__}")
        print(f"  CNN - Scheduler: {type(cnn_scheduler).__name__}")
        
        # Test with RNN model
        print("\n3. Testing RNN Model:")
        rnn_loss_fn, rnn_optimizer, rnn_scheduler = (
            CustomLossOptimizationSchemes.classification_scheme(
                self.rnn_model, num_classes=5
            )
        )
        
        print(f"  RNN - Loss: {rnn_loss_fn.__name__ if hasattr(rnn_loss_fn, '__name__') else 'Custom'}")
        print(f"  RNN - Optimizer: {type(rnn_optimizer).__name__}")
        print(f"  RNN - Scheduler: {type(rnn_scheduler).__name__}")
        
        # Test with hybrid model
        print("\n4. Testing Hybrid Model:")
        hybrid_loss_fn, hybrid_optimizer, hybrid_scheduler = (
            CustomLossOptimizationSchemes.classification_scheme(
                self.hybrid_model, num_classes=10
            )
        )
        
        print(f"  Hybrid - Loss: {hybrid_loss_fn.__name__ if hasattr(hybrid_loss_fn, '__name__') else 'Custom'}")
        print(f"  Hybrid - Optimizer: {type(hybrid_optimizer).__name__}")
        print(f"  Hybrid - Scheduler: {type(hybrid_scheduler).__name__}")
        
        print("Custom model integration demonstration completed!")
    
    def demonstrate_optimization_convergence_analysis(self):
        """Demonstrate optimization convergence analysis."""
        print("\n=== Optimization Convergence Analysis ===")
        
        if not LOSS_OPT_AVAILABLE:
            print("Loss optimization system not available, skipping...")
            return
        
        # Create model and training data
        model = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        ).to(self.device)
        
        x_train = torch.randn(1000, 50, device=self.device)
        y_train = torch.randint(0, 10, (1000,), device=self.device)
        
        # Setup training
        loss_fn = LossFunctions.cross_entropy_loss
        optimizer = Optimizers.adam_optimizer(model, lr=0.001)
        
        # Training loop with convergence monitoring
        losses = []
        convergence_checks = []
        
        print("Training with convergence monitoring:")
        
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = loss_fn(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Check convergence every 5 epochs
            if epoch % 5 == 0:
                convergence_result = LossOptimizationAnalyzer.check_optimization_convergence(
                    losses, patience=5, tolerance=1e-6
                )
                convergence_checks.append(convergence_result)
                
                print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
                print(f"    Converged: {convergence_result['converged']}")
                print(f"    Loss variance: {convergence_result['loss_variance']:.6f}")
                
                if convergence_result['converged']:
                    print(f"    Optimization converged at epoch {epoch}!")
                    break
        
        # Final convergence analysis
        final_convergence = LossOptimizationAnalyzer.check_optimization_convergence(
            losses, patience=10, tolerance=1e-6
        )
        
        print(f"\nFinal convergence analysis:")
        print(f"  Converged: {final_convergence['converged']}")
        print(f"  Loss variance: {final_convergence['loss_variance']:.6f}")
        print(f"  Loss change: {final_convergence['loss_change']:.6f}")
        print(f"  Final loss: {losses[-1]:.6f}")
        
        print("Optimization convergence analysis completed!")
    
    def run_all_advanced_examples(self):
        """Run all advanced loss and optimization examples."""
        print("Advanced Loss Functions and Optimization Examples")
        print("=" * 70)
        
        examples = [
            self.demonstrate_loss_function_comparison,
            self.demonstrate_optimizer_comparison,
            self.demonstrate_learning_rate_scheduler_comparison,
            self.demonstrate_loss_landscape_analysis,
            self.demonstrate_task_specific_schemes,
            self.demonstrate_custom_model_integration,
            self.demonstrate_optimization_convergence_analysis
        ]
        
        for i, example in enumerate(examples, 1):
            try:
                print(f"\n[{i}/{len(examples)}] Running: {example.__name__}")
                example()
            except Exception as e:
                print(f"Error in {example.__name__}: {e}")
                print("Continuing with next example...")
        
        print("\n" + "=" * 70)
        print("All advanced loss and optimization examples completed!")
        print("The loss functions and optimization system is now fully demonstrated.")


def main():
    """Main function to run the advanced loss and optimization examples."""
    try:
        # Create and run the advanced examples
        examples = AdvancedLossOptimizationExamples()
        examples.run_all_advanced_examples()
        
    except Exception as e:
        print(f"Error running advanced loss and optimization examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



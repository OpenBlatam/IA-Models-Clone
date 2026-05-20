"""
Advanced Weight Initialization and Normalization Examples

This module provides advanced examples demonstrating weight initialization and
normalization techniques integrated with custom model architectures. It includes:

1. Advanced initialization strategies for different architectures
2. Weight normalization techniques and their effects
3. Integration with custom model architectures
4. Performance analysis and benchmarking
5. Real-world initialization scenarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import yaml
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

# Import weight initialization system
try:
    from weight_initialization_system import (
        WeightInitializer, WeightNormalizer, InitializationAnalyzer,
        CustomInitializationSchemes
    )
    WEIGHT_INIT_AVAILABLE = True
except ImportError:
    print("Warning: weight_initialization_system not found. Some examples may not work.")
    WEIGHT_INIT_AVAILABLE = False

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


class AdvancedWeightInitExamples:
    """Advanced examples demonstrating sophisticated weight initialization techniques."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize components
        self.setup_components()
    
    def load_config(self) -> Dict:
        """Load weight initialization configuration."""
        try:
            with open('weight_init_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            print("Weight initialization configuration loaded successfully")
            return config
        except FileNotFoundError:
            print("Configuration file not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default configuration if file not found."""
        return {
            'weight_initialization': {
                'global': {'default_method': 'xavier_uniform'},
                'methods': {'xavier_uniform': {'enabled': True}}
            },
            'initialization_analysis': {
                'analysis': {'enabled': True, 'check_quality': True}
            }
        }
    
    def setup_components(self):
        """Setup all system components."""
        print("\n=== Setting up Weight Initialization Components ===")
        
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
    
    def demonstrate_initialization_methods_comparison(self):
        """Demonstrate comparison of different initialization methods."""
        print("\n=== Initialization Methods Comparison ===")
        
        if not WEIGHT_INIT_AVAILABLE:
            print("Weight initialization system not available, skipping...")
            return
        
        # Create identical models for comparison
        models = {}
        init_methods = ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal']
        
        for method in init_methods:
            model = nn.Sequential(
                nn.Linear(50, 100),
                nn.ReLU(),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            ).to(self.device)
            
            # Initialize with specific method
            WeightInitializer.initialize_model(model, method)
            models[method] = model
        
        # Analyze and compare
        analyzer = InitializationAnalyzer()
        comparison_results = {}
        
        for method, model in models.items():
            print(f"\nAnalyzing {method} initialization:")
            
            # Analyze weights
            stats = analyzer.analyze_weights(model)
            
            # Check quality
            quality = analyzer.check_initialization_quality(model)
            
            comparison_results[method] = {
                'stats': stats,
                'quality': quality
            }
            
            # Print summary
            total_params = sum(stats[name]['numel'] for name in stats)
            avg_std = np.mean([stats[name]['std'] for name in stats])
            quality_score = quality['overall_score']
            
            print(f"  Total parameters: {total_params:,}")
            print(f"  Average std: {avg_std:.4f}")
            print(f"  Quality score: {quality_score:.2f}")
        
        # Find best method
        best_method = max(comparison_results.keys(), 
                         key=lambda x: comparison_results[x]['quality']['overall_score'])
        print(f"\nBest initialization method: {best_method}")
        print("Initialization methods comparison completed!")
    
    def demonstrate_architecture_specific_initialization(self):
        """Demonstrate architecture-specific initialization schemes."""
        print("\n=== Architecture-Specific Initialization ===")
        
        if not all([WEIGHT_INIT_AVAILABLE, CUSTOM_MODELS_AVAILABLE]):
            print("Required components not available, skipping...")
            return
        
        # Test transformer initialization
        print("\n1. Testing Transformer Initialization:")
        transformer_copy = CustomTransformerModel(
            vocab_size=1000, d_model=128, nhead=8, num_layers=4
        ).to(self.device)
        
        CustomInitializationSchemes.transformer_initialization(transformer_copy, d_model=128)
        
        # Analyze transformer weights
        analyzer = InitializationAnalyzer()
        transformer_stats = analyzer.analyze_weights(transformer_copy)
        transformer_quality = analyzer.check_initialization_quality(transformer_copy)
        
        print(f"  Transformer quality score: {transformer_quality['overall_score']:.2f}")
        
        # Test CNN initialization
        print("\n2. Testing CNN Initialization:")
        cnn_copy = CustomCNNModel(
            input_channels=3, num_classes=10, base_channels=32
        ).to(self.device)
        
        CustomInitializationSchemes.cnn_initialization(cnn_copy)
        
        # Analyze CNN weights
        cnn_stats = analyzer.analyze_weights(cnn_copy)
        cnn_quality = analyzer.check_initialization_quality(cnn_copy)
        
        print(f"  CNN quality score: {cnn_quality['overall_score']:.2f}")
        
        # Test RNN initialization
        print("\n3. Testing RNN Initialization:")
        rnn_copy = CustomRNNModel(
            input_size=100, hidden_size=128, num_layers=3, num_classes=5
        ).to(self.device)
        
        CustomInitializationSchemes.rnn_initialization(rnn_copy, num_layers=3)
        
        # Analyze RNN weights
        rnn_stats = analyzer.analyze_weights(rnn_copy)
        rnn_quality = analyzer.check_initialization_quality(rnn_copy)
        
        print(f"  RNN quality score: {rnn_quality['overall_score']:.2f}")
        
        print("Architecture-specific initialization demonstration completed!")
    
    def demonstrate_weight_normalization_effects(self):
        """Demonstrate the effects of weight normalization."""
        print("\n=== Weight Normalization Effects ===")
        
        if not WEIGHT_INIT_AVAILABLE:
            print("Weight initialization system not available, skipping...")
            return
        
        # Create a model
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 150),
            nn.ReLU(),
            nn.Linear(150, 100)
        ).to(self.device)
        
        # Initialize weights
        WeightInitializer.initialize_model(model, 'xavier_uniform')
        
        # Analyze before normalization
        analyzer = InitializationAnalyzer()
        before_stats = analyzer.analyze_weights(model)
        before_quality = analyzer.check_initialization_quality(model)
        
        print(f"Before normalization - Quality score: {before_quality['overall_score']:.2f}")
        
        # Apply weight normalization
        WeightNormalizer.apply_normalization(model, 'weight_norm')
        
        # Analyze after normalization
        after_stats = analyzer.analyze_weights(model)
        after_quality = analyzer.check_initialization_quality(model)
        
        print(f"After normalization - Quality score: {after_quality['overall_score']:.2f}")
        
        # Compare statistics
        print("\nWeight statistics comparison:")
        for name in before_stats.keys():
            if name in after_stats:
                before_std = before_stats[name]['std']
                after_std = after_stats[name]['std']
                print(f"  {name}: std {before_std:.4f} -> {after_std:.4f}")
        
        print("Weight normalization effects demonstration completed!")
    
    def demonstrate_initialization_performance_benchmark(self):
        """Demonstrate performance benchmarking of different initialization methods."""
        print("\n=== Initialization Performance Benchmark ===")
        
        if not WEIGHT_INIT_AVAILABLE:
            print("Weight initialization system not available, skipping...")
            return
        
        # Create a large model for benchmarking
        large_model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1500),
            nn.ReLU(),
            nn.Linear(1500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100)
        ).to(self.device)
        
        # Benchmark different initialization methods
        init_methods = ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal']
        benchmark_results = {}
        
        for method in init_methods:
            print(f"\nBenchmarking {method}...")
            
            # Measure initialization time
            start_time = time.time()
            WeightInitializer.initialize_model(large_model, method)
            init_time = time.time() - start_time
            
            # Measure analysis time
            analyzer = InitializationAnalyzer()
            start_time = time.time()
            quality = analyzer.check_initialization_quality(large_model)
            analysis_time = time.time() - start_time
            
            benchmark_results[method] = {
                'init_time': init_time,
                'analysis_time': analysis_time,
                'quality_score': quality['overall_score']
            }
            
            print(f"  Initialization time: {init_time:.4f}s")
            print(f"  Analysis time: {analysis_time:.4f}s")
            print(f"  Quality score: {quality['overall_score']:.2f}")
        
        # Find best performing method
        best_method = max(benchmark_results.keys(), 
                         key=lambda x: benchmark_results[x]['quality_score'] / 
                                     (benchmark_results[x]['init_time'] + benchmark_results[x]['analysis_time']))
        
        print(f"\nBest performing method: {best_method}")
        print("Performance benchmark completed!")
    
    def demonstrate_initialization_debugging(self):
        """Demonstrate debugging techniques for weight initialization."""
        print("\n=== Initialization Debugging ===")
        
        if not WEIGHT_INIT_AVAILABLE:
            print("Weight initialization system not available, skipping...")
            return
        
        # Create a problematic model
        problematic_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ).to(self.device)
        
        # Manually set problematic weights
        with torch.no_grad():
            # Set some weights to extreme values
            problematic_model[0].weight.data.fill_(100.0)
            problematic_model[2].weight.data.fill_(0.0)
            problematic_model[4].weight.data.fill_(-100.0)
        
        # Analyze problematic model
        analyzer = InitializationAnalyzer()
        problematic_stats = analyzer.analyze_weights(problematic_model)
        problematic_quality = analyzer.check_initialization_quality(problematic_model)
        
        print("Problematic model analysis:")
        print(f"  Quality score: {problematic_quality['overall_score']:.2f}")
        
        # Check for specific issues
        print("\nDetected issues:")
        for name, param_stats in problematic_stats.items():
            if param_stats['std'] > 10.0:
                print(f"  {name}: Very high standard deviation ({param_stats['std']:.4f})")
            if param_stats['norm'] < 1e-6:
                print(f"  {name}: Very small weight norm ({param_stats['norm']:.4f})")
            if param_stats['max'] > 100.0 or param_stats['min'] < -100.0:
                print(f"  {name}: Extreme weight values detected")
        
        # Fix the model with proper initialization
        print("\nFixing model with proper initialization...")
        WeightInitializer.initialize_model(problematic_model, 'xavier_uniform')
        
        # Re-analyze fixed model
        fixed_stats = analyzer.analyze_weights(problematic_model)
        fixed_quality = analyzer.check_initialization_quality(problematic_model)
        
        print(f"Fixed model quality score: {fixed_quality['overall_score']:.2f}")
        print("Initialization debugging demonstration completed!")
    
    def demonstrate_hybrid_model_initialization(self):
        """Demonstrate initialization of hybrid models."""
        print("\n=== Hybrid Model Initialization ===")
        
        if not all([WEIGHT_INIT_AVAILABLE, CUSTOM_MODELS_AVAILABLE]):
            print("Required components not available, skipping...")
            return
        
        # Create hybrid model
        hybrid_model = CNNTransformerHybrid(
            input_channels=3, num_classes=10, d_model=128, nhead=8
        ).to(self.device)
        
        # Initialize different components with appropriate methods
        print("Initializing hybrid model components...")
        
        # Initialize CNN components
        CustomInitializationSchemes.cnn_initialization(hybrid_model)
        
        # Initialize transformer components
        CustomInitializationSchemes.transformer_initialization(hybrid_model, d_model=128)
        
        # Initialize remaining components with default method
        WeightInitializer.initialize_model(hybrid_model, 'xavier_uniform')
        
        # Analyze hybrid model
        analyzer = InitializationAnalyzer()
        hybrid_stats = analyzer.analyze_weights(hybrid_model)
        hybrid_quality = analyzer.check_initialization_quality(hybrid_model)
        
        print(f"Hybrid model quality score: {hybrid_quality['overall_score']:.2f}")
        
        # Analyze component-specific statistics
        print("\nComponent-specific analysis:")
        cnn_params = {k: v for k, v in hybrid_stats.items() if 'conv' in k.lower()}
        transformer_params = {k: v for k, v in hybrid_stats.items() if 'attention' in k.lower() or 'attn' in k.lower()}
        
        if cnn_params:
            cnn_avg_std = np.mean([v['std'] for v in cnn_params.values()])
            print(f"  CNN parameters average std: {cnn_avg_std:.4f}")
        
        if transformer_params:
            transformer_avg_std = np.mean([v['std'] for v in transformer_params.values()])
            print(f"  Transformer parameters average std: {transformer_avg_std:.4f}")
        
        print("Hybrid model initialization demonstration completed!")
    
    def demonstrate_initialization_with_training(self):
        """Demonstrate initialization effects on training."""
        print("\n=== Initialization Effects on Training ===")
        
        if not WEIGHT_INIT_AVAILABLE:
            print("Weight initialization system not available, skipping...")
            return
        
        # Create training data
        x_train = torch.randn(1000, 50, device=self.device)
        y_train = torch.randn(1000, 10, device=self.device)
        
        # Test different initialization methods
        init_methods = ['xavier_uniform', 'kaiming_normal', 'orthogonal']
        training_results = {}
        
        for method in init_methods:
            print(f"\nTesting {method} initialization:")
            
            # Create and initialize model
            model = nn.Sequential(
                nn.Linear(50, 100),
                nn.ReLU(),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            ).to(self.device)
            
            WeightInitializer.initialize_model(model, method)
            
            # Setup training
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Training loop
            losses = []
            for epoch in range(5):
                optimizer.zero_grad()
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
                if epoch % 2 == 0:
                    print(f"    Epoch {epoch}: Loss = {loss.item():.6f}")
            
            training_results[method] = {
                'final_loss': losses[-1],
                'loss_history': losses,
                'convergence_rate': (losses[0] - losses[-1]) / losses[0]
            }
        
        # Compare results
        print("\nTraining comparison:")
        for method, results in training_results.items():
            print(f"  {method}:")
            print(f"    Final loss: {results['final_loss']:.6f}")
            print(f"    Convergence rate: {results['convergence_rate']:.2%}")
        
        # Find best method for training
        best_training_method = min(training_results.keys(), 
                                 key=lambda x: training_results[x]['final_loss'])
        print(f"\nBest method for training: {best_training_method}")
        print("Initialization training effects demonstration completed!")
    
    def run_all_advanced_examples(self):
        """Run all advanced weight initialization examples."""
        print("Advanced Weight Initialization Examples")
        print("=" * 60)
        
        examples = [
            self.demonstrate_initialization_methods_comparison,
            self.demonstrate_architecture_specific_initialization,
            self.demonstrate_weight_normalization_effects,
            self.demonstrate_initialization_performance_benchmark,
            self.demonstrate_initialization_debugging,
            self.demonstrate_hybrid_model_initialization,
            self.demonstrate_initialization_with_training
        ]
        
        for i, example in enumerate(examples, 1):
            try:
                print(f"\n[{i}/{len(examples)}] Running: {example.__name__}")
                example()
            except Exception as e:
                print(f"Error in {example.__name__}: {e}")
                print("Continuing with next example...")
        
        print("\n" + "=" * 60)
        print("All advanced weight initialization examples completed!")
        print("The weight initialization system is now fully demonstrated.")


def main():
    """Main function to run the advanced weight initialization examples."""
    try:
        # Create and run the advanced examples
        examples = AdvancedWeightInitExamples()
        examples.run_all_advanced_examples()
        
    except Exception as e:
        print(f"Error running advanced weight initialization examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



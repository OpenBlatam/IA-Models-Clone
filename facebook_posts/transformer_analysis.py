from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time
from advanced_transformers import (
    TransformerConfig, ModelType, AdvancedTransformerModel, 
    LargeLanguageModel, TransformerFactory
)
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Transformer Analysis Tools
Comprehensive analysis tools for transformers and LLMs.
"""

    TransformerConfig, ModelType, AdvancedTransformerModel, 
    LargeLanguageModel, TransformerFactory
)


@dataclass
class AnalysisConfig:
    """Configuration for transformer analysis."""
    # Analysis types
    attention_analysis: bool = True
    gradient_analysis: bool = True
    parameter_analysis: bool = True
    performance_analysis: bool = True
    
    # Visualization settings
    save_plots: bool = True
    plot_format: str = "png"
    dpi: int = 300
    
    # Analysis settings
    num_samples: int = 100
    max_sequence_length: int = 512
    batch_size: int = 4


class TransformerAnalyzer:
    """Comprehensive transformer analyzer."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.analysis_results = {}
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> Any:
        """Setup logging for analysis."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('transformer_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_model(self, model: nn.Module, config: TransformerConfig) -> Dict[str, Any]:
        """Comprehensive model analysis."""
        self.logger.info("Starting comprehensive transformer analysis")
        
        results = {}
        
        # Parameter analysis
        if self.config.parameter_analysis:
            results['parameter_analysis'] = self._analyze_parameters(model)
        
        # Attention analysis
        if self.config.attention_analysis:
            results['attention_analysis'] = self._analyze_attention_patterns(model, config)
        
        # Gradient analysis
        if self.config.gradient_analysis:
            results['gradient_analysis'] = self._analyze_gradients(model, config)
        
        # Performance analysis
        if self.config.performance_analysis:
            results['performance_analysis'] = self._analyze_performance(model, config)
        
        self.analysis_results = results
        return results
    
    def _analyze_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model parameters."""
        self.logger.info("Analyzing model parameters")
        
        total_params = 0
        trainable_params = 0
        layer_params = {}
        layer_types = {}
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
            
            # Analyze by layer type
            layer_name = name.split('.')[0]
            if layer_name not in layer_params:
                layer_params[layer_name] = 0
                layer_types[layer_name] = type(param).__name__
            layer_params[layer_name] += param_count
        
        # Parameter distribution analysis
        param_norms = []
        param_means = []
        param_stds = []
        
        for param in model.parameters():
            if param.requires_grad:
                param_norms.append(param.norm().item())
                param_means.append(param.mean().item())
                param_stds.append(param.std().item())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layer_parameters': layer_params,
            'layer_types': layer_types,
            'parameter_norms': {
                'mean': np.mean(param_norms),
                'std': np.std(param_norms),
                'min': np.min(param_norms),
                'max': np.max(param_norms)
            },
            'parameter_means': {
                'mean': np.mean(param_means),
                'std': np.std(param_means)
            },
            'parameter_stds': {
                'mean': np.mean(param_stds),
                'std': np.std(param_stds)
            }
        }
    
    def _analyze_attention_patterns(self, model: nn.Module, config: TransformerConfig) -> Dict[str, Any]:
        """Analyze attention patterns."""
        self.logger.info("Analyzing attention patterns")
        
        # Create dummy input
        batch_size = self.config.batch_size
        seq_length = min(self.config.max_sequence_length, config.max_position_embeddings)
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        model.eval()
        attention_patterns = []
        
        with torch.no_grad():
            # Forward pass with attention output
            outputs = model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                output_attentions=True
            )
            
            # Extract attention patterns
            if 'attentions' in outputs and outputs['attentions'] is not None:
                for layer_idx, attention in enumerate(outputs['attentions']):
                    if attention is not None:
                        # Average across heads and batch
                        avg_attention = attention.mean(dim=1).mean(dim=0)
                        attention_patterns.append({
                            'layer': layer_idx,
                            'attention_matrix': avg_attention.cpu().numpy(),
                            'attention_stats': {
                                'mean': avg_attention.mean().item(),
                                'std': avg_attention.std().item(),
                                'max': avg_attention.max().item(),
                                'min': avg_attention.min().item()
                            }
                        })
        
        return {
            'attention_patterns': attention_patterns,
            'num_layers': len(attention_patterns),
            'sequence_length': seq_length
        }
    
    def _analyze_gradients(self, model: nn.Module, config: TransformerConfig) -> Dict[str, Any]:
        """Analyze gradient flow and statistics."""
        self.logger.info("Analyzing gradient flow")
        
        # Create dummy input and loss
        batch_size = self.config.batch_size
        seq_length = min(self.config.max_sequence_length, config.max_position_embeddings)
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        
        model.train()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device)
        )
        
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Analyze gradients
        gradient_stats = {}
        layer_gradients = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                
                layer_name = name.split('.')[0]
                if layer_name not in layer_gradients:
                    layer_gradients[layer_name] = []
                
                layer_gradients[layer_name].append({
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std
                })
                
                gradient_stats[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std
                }
        
        # Compute layer-wise statistics
        layer_stats = {}
        for layer_name, gradients in layer_gradients.items():
            norms = [g['norm'] for g in gradients]
            means = [g['mean'] for g in gradients]
            stds = [g['std'] for g in gradients]
            
            layer_stats[layer_name] = {
                'norm_mean': np.mean(norms),
                'norm_std': np.std(norms),
                'mean_mean': np.mean(means),
                'mean_std': np.std(means),
                'std_mean': np.mean(stds),
                'std_std': np.std(stds)
            }
        
        return {
            'gradient_stats': gradient_stats,
            'layer_gradients': layer_gradients,
            'layer_stats': layer_stats,
            'total_grad_norm': sum(g['norm'] for g in gradient_stats.values())
        }
    
    def _analyze_performance(self, model: nn.Module, config: TransformerConfig) -> Dict[str, Any]:
        """Analyze model performance."""
        self.logger.info("Analyzing model performance")
        
        batch_size = self.config.batch_size
        seq_length = min(self.config.max_sequence_length, config.max_position_embeddings)
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        model.eval()
        
        # Measure inference time
        times = []
        memory_usage = []
        
        for _ in range(self.config.num_samples):
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device)
                )
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Measure memory usage
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
        
        # Compute statistics
        inference_time = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'throughput': batch_size / np.mean(times)  # samples per second
        }
        
        memory_stats = {}
        if memory_usage:
            memory_stats = {
                'mean': np.mean(memory_usage),
                'std': np.std(memory_usage),
                'max': np.max(memory_usage)
            }
        
        return {
            'inference_time': inference_time,
            'memory_usage': memory_stats,
            'batch_size': batch_size,
            'sequence_length': seq_length
        }
    
    def visualize_attention_patterns(self, attention_analysis: Dict[str, Any], save_path: str = "attention_patterns"):
        """Visualize attention patterns."""
        self.logger.info("Visualizing attention patterns")
        
        attention_patterns = attention_analysis['attention_patterns']
        num_layers = len(attention_patterns)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, pattern in enumerate(attention_patterns[:6]):  # Show first 6 layers
            attention_matrix = pattern['attention_matrix']
            
            # Create heatmap
            im = axes[i].imshow(attention_matrix, cmap='viridis', aspect='auto')
            axes[i].set_title(f'Layer {pattern["layer"]}')
            axes[i].set_xlabel('Key Position')
            axes[i].set_ylabel('Query Position')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(f"{save_path}.{self.config.plot_format}", 
                       dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def visualize_gradient_flow(self, gradient_analysis: Dict[str, Any], save_path: str = "gradient_flow"):
        """Visualize gradient flow."""
        self.logger.info("Visualizing gradient flow")
        
        layer_stats = gradient_analysis['layer_stats']
        layers = list(layer_stats.keys())
        norms = [layer_stats[layer]['norm_mean'] for layer in layers]
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(layers)), norms)
        plt.xlabel('Layer')
        plt.ylabel('Average Gradient Norm')
        plt.title('Gradient Flow Across Layers')
        plt.xticks(range(len(layers)), layers, rotation=45)
        
        # Add value labels on bars
        for bar, norm in zip(bars, norms):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{norm:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(f"{save_path}.{self.config.plot_format}", 
                       dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def visualize_parameter_distribution(self, parameter_analysis: Dict[str, Any], save_path: str = "parameter_distribution"):
        """Visualize parameter distribution."""
        self.logger.info("Visualizing parameter distribution")
        
        layer_params = parameter_analysis['layer_parameters']
        layers = list(layer_params.keys())
        param_counts = list(layer_params.values())
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(param_counts, labels=layers, autopct='%1.1f%%', startangle=90)
        plt.title('Parameter Distribution Across Layers')
        
        if self.config.save_plots:
            plt.savefig(f"{save_path}.{self.config.plot_format}", 
                       dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def generate_analysis_report(self, results: Dict[str, Any], save_path: str = "transformer_analysis_report"):
        """Generate comprehensive analysis report."""
        self.logger.info("Generating analysis report")
        
        report = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'config': self.config.__dict__,
            'results': results
        }
        
        # Save JSON report
        with open(f"{save_path}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate text report
        with open(f"{save_path}.txt", 'w') as f:
            f.write("Transformer Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Timestamp: {report['analysis_timestamp']}\n")
            f.write(f"Device: {report['device']}\n\n")
            if 'parameter_analysis' in results:
                f.write("Parameter Analysis:\n")
                f.write("-" * 20 + "\n")
                param_analysis = results['parameter_analysis']
                f.write(f"Total Parameters: {param_analysis['total_parameters']:,}\n")
                f.write(f"Trainable Parameters: {param_analysis['trainable_parameters']:,}\n")
                f.write(f"Parameter Norm - Mean: {param_analysis['parameter_norms']['mean']:.6f}\n")
                f.write(f"Parameter Norm - Std: {param_analysis['parameter_norms']['std']:.6f}\n\n")
            if 'performance_analysis' in results:
                f.write("Performance Analysis:\n")
                f.write("-" * 20 + "\n")
                perf_analysis = results['performance_analysis']
                f.write(f"Inference Time - Mean: {perf_analysis['inference_time']['mean']:.6f}s\n")
                f.write(f"Inference Time - Std: {perf_analysis['inference_time']['std']:.6f}s\n")
                f.write(f"Throughput: {perf_analysis['inference_time']['throughput']:.2f} samples/s\n")
                if 'memory_usage' in perf_analysis and perf_analysis['memory_usage']:
                    f.write(f"Memory Usage - Mean: {perf_analysis['memory_usage']['mean']:.2f} MB\n")
                f.write("\n")
            if 'attention_analysis' in results:
                f.write("Attention Analysis:\n")
                f.write("-" * 20 + "\n")
                attn_analysis = results['attention_analysis']
                f.write(f"Number of Layers: {attn_analysis['num_layers']}\n")
                f.write(f"Sequence Length: {attn_analysis['sequence_length']}\n")
                f.write(f"Attention Patterns Analyzed: {len(attn_analysis['attention_patterns'])}\n\n")
        
        self.logger.info(f"Analysis report saved to {save_path}.json and {save_path}.txt")


def demonstrate_transformer_analysis():
    """Demonstrate transformer analysis capabilities."""
    print("Transformer Analysis Demonstration")
    print("=" * 45)
    
    # Create analysis configuration
    analysis_config = AnalysisConfig(
        attention_analysis=True,
        gradient_analysis=True,
        parameter_analysis=True,
        performance_analysis=True,
        save_plots=True
    )
    
    # Create analyzer
    analyzer = TransformerAnalyzer(analysis_config)
    
    # Test different model configurations
    model_configs = [
        TransformerConfig(
            model_type=ModelType.ENCODER_ONLY,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            use_flash_attention=True
        ),
        TransformerConfig(
            model_type=ModelType.CAUSAL_LM,
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            use_flash_attention=False
        )
    ]
    
    results = {}
    
    for i, config in enumerate(model_configs):
        print(f"\nAnalyzing {config.model_type.value} model:")
        
        try:
            # Create model
            model = TransformerFactory.create_model(config)
            
            # Analyze model
            analysis_results = analyzer.analyze_model(model, config)
            
            # Visualize results
            if 'attention_analysis' in analysis_results:
                analyzer.visualize_attention_patterns(
                    analysis_results['attention_analysis'], 
                    f"attention_patterns_model_{i}"
                )
            
            if 'gradient_analysis' in analysis_results:
                analyzer.visualize_gradient_flow(
                    analysis_results['gradient_analysis'],
                    f"gradient_flow_model_{i}"
                )
            
            if 'parameter_analysis' in analysis_results:
                analyzer.visualize_parameter_distribution(
                    analysis_results['parameter_analysis'],
                    f"parameter_distribution_model_{i}"
                )
            
            # Generate report
            analyzer.generate_analysis_report(
                analysis_results,
                f"transformer_analysis_report_model_{i}"
            )
            
            results[f"model_{i}"] = {
                'config': config,
                'results': analysis_results,
                'success': True
            }
            
            # Print summary
            param_analysis = analysis_results['parameter_analysis']
            perf_analysis = analysis_results['performance_analysis']
            
            print(f"  Total Parameters: {param_analysis['total_parameters']:,}")
            print(f"  Inference Time: {perf_analysis['inference_time']['mean']:.6f}s")
            print(f"  Throughput: {perf_analysis['inference_time']['throughput']:.2f} samples/s")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[f"model_{i}"] = {
                'config': config,
                'error': str(e),
                'success': False
            }
    
    return results


if __name__ == "__main__":
    # Demonstrate transformer analysis
    results = demonstrate_transformer_analysis()
    print("\nTransformer analysis demonstration completed!") 
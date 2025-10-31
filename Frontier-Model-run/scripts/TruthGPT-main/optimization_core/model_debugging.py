"""
Advanced Neural Network Debugging System for TruthGPT Optimization Core
Model debugging, error analysis, and performance diagnostics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
import pickle
from abc import ABC, abstractmethod
import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DebuggingLevel(Enum):
    """Debugging levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class ErrorType(Enum):
    """Error types"""
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    DATA_LEAKAGE = "data_leakage"
    CONVERGENCE_ISSUES = "convergence_issues"
    MEMORY_LEAKS = "memory_leaks"
    NUMERICAL_INSTABILITY = "numerical_instability"

class DiagnosticType(Enum):
    """Diagnostic types"""
    GRADIENT_ANALYSIS = "gradient_analysis"
    ACTIVATION_ANALYSIS = "activation_analysis"
    WEIGHT_ANALYSIS = "weight_analysis"
    LOSS_ANALYSIS = "loss_analysis"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    MEMORY_ANALYSIS = "memory_analysis"
    CONVERGENCE_ANALYSIS = "convergence_analysis"

@dataclass
class DebuggingConfig:
    """Configuration for model debugging"""
    # Debugging settings
    debugging_level: DebuggingLevel = DebuggingLevel.INTERMEDIATE
    enable_gradient_monitoring: bool = True
    enable_activation_monitoring: bool = True
    enable_weight_monitoring: bool = True
    enable_loss_monitoring: bool = True
    
    # Monitoring settings
    gradient_threshold: float = 1.0
    activation_threshold: float = 10.0
    weight_threshold: float = 5.0
    loss_threshold: float = 100.0
    
    # Analysis settings
    analysis_frequency: int = 10
    history_length: int = 1000
    enable_real_time_monitoring: bool = True
    
    # Advanced features
    enable_automatic_fixes: bool = False
    enable_error_prediction: bool = True
    enable_performance_optimization: bool = True
    enable_memory_optimization: bool = True
    
    def __post_init__(self):
        """Validate debugging configuration"""
        if self.gradient_threshold <= 0:
            raise ValueError("Gradient threshold must be positive")
        if self.activation_threshold <= 0:
            raise ValueError("Activation threshold must be positive")
        if self.weight_threshold <= 0:
            raise ValueError("Weight threshold must be positive")
        if self.loss_threshold <= 0:
            raise ValueError("Loss threshold must be positive")
        if self.analysis_frequency < 1:
            raise ValueError("Analysis frequency must be at least 1")
        if self.history_length < 1:
            raise ValueError("History length must be at least 1")

class GradientMonitor:
    """Gradient monitoring system"""
    
    def __init__(self, config: DebuggingConfig):
        self.config = config
        self.gradient_history = []
        self.gradient_stats = {}
        logger.info("✅ Gradient Monitor initialized")
    
    def monitor_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """Monitor gradients during training"""
        gradient_info = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                grad_max = param.grad.max().item()
                grad_min = param.grad.min().item()
                
                gradient_info[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'max': grad_max,
                    'min': grad_min,
                    'explosion': grad_norm > self.config.gradient_threshold,
                    'vanishing': grad_norm < 1e-8
                }
                
                # Store history
                self.gradient_history.append({
                    'name': name,
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'max': grad_max,
                    'min': grad_min,
                    'timestamp': time.time()
                })
        
        return gradient_info
    
    def analyze_gradient_issues(self, gradient_info: Dict[str, Any]) -> List[ErrorType]:
        """Analyze gradient issues"""
        issues = []
        
        for name, info in gradient_info.items():
            if info['explosion']:
                issues.append(ErrorType.GRADIENT_EXPLOSION)
            if info['vanishing']:
                issues.append(ErrorType.GRADIENT_VANISHING)
        
        return list(set(issues))  # Remove duplicates
    
    def get_gradient_statistics(self) -> Dict[str, float]:
        """Get gradient statistics"""
        if not self.gradient_history:
            return {}
        
        norms = [h['norm'] for h in self.gradient_history]
        
        return {
            'mean_norm': np.mean(norms),
            'std_norm': np.std(norms),
            'max_norm': np.max(norms),
            'min_norm': np.min(norms),
            'explosion_count': sum(1 for n in norms if n > self.config.gradient_threshold),
            'vanishing_count': sum(1 for n in norms if n < 1e-8)
        }

class ActivationMonitor:
    """Activation monitoring system"""
    
    def __init__(self, config: DebuggingConfig):
        self.config = config
        self.activation_history = []
        self.activation_stats = {}
        logger.info("✅ Activation Monitor initialized")
    
    def monitor_activations(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Monitor activations during forward pass"""
        activation_info = {}
        
        def activation_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activation_norm = output.norm().item()
                activation_mean = output.mean().item()
                activation_std = output.std().item()
                activation_max = output.max().item()
                activation_min = output.min().item()
                
                activation_info[module.__class__.__name__] = {
                    'norm': activation_norm,
                    'mean': activation_mean,
                    'std': activation_std,
                    'max': activation_max,
                    'min': activation_min,
                    'saturation': activation_max > self.config.activation_threshold,
                    'dead_neurons': (output == 0).float().mean().item()
                }
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(activation_hook)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Store history
        self.activation_history.append({
            'activations': activation_info,
            'timestamp': time.time()
        })
        
        return activation_info
    
    def analyze_activation_issues(self, activation_info: Dict[str, Any]) -> List[ErrorType]:
        """Analyze activation issues"""
        issues = []
        
        for name, info in activation_info.items():
            if info['saturation']:
                issues.append(ErrorType.NUMERICAL_INSTABILITY)
            if info['dead_neurons'] > 0.5:  # More than 50% dead neurons
                issues.append(ErrorType.GRADIENT_VANISHING)
        
        return list(set(issues))
    
    def get_activation_statistics(self) -> Dict[str, float]:
        """Get activation statistics"""
        if not self.activation_history:
            return {}
        
        all_norms = []
        all_means = []
        all_stds = []
        
        for history in self.activation_history:
            for name, info in history['activations'].items():
                all_norms.append(info['norm'])
                all_means.append(info['mean'])
                all_stds.append(info['std'])
        
        return {
            'mean_norm': np.mean(all_norms),
            'std_norm': np.std(all_norms),
            'max_norm': np.max(all_norms),
            'min_norm': np.min(all_norms),
            'mean_mean': np.mean(all_means),
            'mean_std': np.mean(all_stds)
        }

class WeightMonitor:
    """Weight monitoring system"""
    
    def __init__(self, config: DebuggingConfig):
        self.config = config
        self.weight_history = []
        self.weight_stats = {}
        logger.info("✅ Weight Monitor initialized")
    
    def monitor_weights(self, model: nn.Module) -> Dict[str, Any]:
        """Monitor weights during training"""
        weight_info = {}
        
        for name, param in model.named_parameters():
            weight_norm = param.norm().item()
            weight_mean = param.mean().item()
            weight_std = param.std().item()
            weight_max = param.max().item()
            weight_min = param.min().item()
            
            weight_info[name] = {
                'norm': weight_norm,
                'mean': weight_mean,
                'std': weight_std,
                'max': weight_max,
                'min': weight_min,
                'explosion': weight_norm > self.config.weight_threshold,
                'vanishing': weight_norm < 1e-8
            }
            
            # Store history
            self.weight_history.append({
                'name': name,
                'norm': weight_norm,
                'mean': weight_mean,
                'std': weight_std,
                'max': weight_max,
                'min': weight_min,
                'timestamp': time.time()
            })
        
        return weight_info
    
    def analyze_weight_issues(self, weight_info: Dict[str, Any]) -> List[ErrorType]:
        """Analyze weight issues"""
        issues = []
        
        for name, info in weight_info.items():
            if info['explosion']:
                issues.append(ErrorType.NUMERICAL_INSTABILITY)
            if info['vanishing']:
                issues.append(ErrorType.GRADIENT_VANISHING)
        
        return list(set(issues))
    
    def get_weight_statistics(self) -> Dict[str, float]:
        """Get weight statistics"""
        if not self.weight_history:
            return {}
        
        norms = [h['norm'] for h in self.weight_history]
        
        return {
            'mean_norm': np.mean(norms),
            'std_norm': np.std(norms),
            'max_norm': np.max(norms),
            'min_norm': np.min(norms),
            'explosion_count': sum(1 for n in norms if n > self.config.weight_threshold),
            'vanishing_count': sum(1 for n in norms if n < 1e-8)
        }

class LossMonitor:
    """Loss monitoring system"""
    
    def __init__(self, config: DebuggingConfig):
        self.config = config
        self.loss_history = []
        self.loss_stats = {}
        logger.info("✅ Loss Monitor initialized")
    
    def monitor_loss(self, loss: float, epoch: int = None) -> Dict[str, Any]:
        """Monitor loss during training"""
        loss_info = {
            'loss': loss,
            'epoch': epoch,
            'timestamp': time.time(),
            'anomaly': loss > self.config.loss_threshold
        }
        
        # Store history
        self.loss_history.append(loss_info)
        
        return loss_info
    
    def analyze_loss_issues(self, loss_history: List[Dict[str, Any]]) -> List[ErrorType]:
        """Analyze loss issues"""
        issues = []
        
        if len(loss_history) < 2:
            return issues
        
        losses = [h['loss'] for h in loss_history]
        
        # Check for overfitting
        if len(losses) > 10:
            recent_losses = losses[-10:]
            if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                issues.append(ErrorType.OVERFITTING)
        
        # Check for underfitting
        if len(losses) > 5:
            recent_losses = losses[-5:]
            if all(recent_losses[i] <= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                issues.append(ErrorType.UNDERFITTING)
        
        # Check for convergence issues
        if len(losses) > 20:
            recent_losses = losses[-20:]
            loss_std = np.std(recent_losses)
            if loss_std < 1e-6:
                issues.append(ErrorType.CONVERGENCE_ISSUES)
        
        return list(set(issues))
    
    def get_loss_statistics(self) -> Dict[str, float]:
        """Get loss statistics"""
        if not self.loss_history:
            return {}
        
        losses = [h['loss'] for h in self.loss_history]
        
        return {
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'max_loss': np.max(losses),
            'min_loss': np.min(losses),
            'current_loss': losses[-1] if losses else 0,
            'loss_trend': np.polyfit(range(len(losses)), losses, 1)[0] if len(losses) > 1 else 0
        }

class PerformanceAnalyzer:
    """Performance analyzer"""
    
    def __init__(self, config: DebuggingConfig):
        self.config = config
        self.performance_history = []
        self.performance_stats = {}
        logger.info("✅ Performance Analyzer initialized")
    
    def analyze_performance(self, model: nn.Module, test_data: torch.Tensor,
                          test_labels: torch.Tensor) -> Dict[str, Any]:
        """Analyze model performance"""
        model.eval()
        
        with torch.no_grad():
            predictions = model(test_data)
            predicted_labels = torch.argmax(predictions, dim=1)
            
            # Calculate metrics
            accuracy = (predicted_labels == test_labels).float().mean().item()
            loss = F.cross_entropy(predictions, test_labels).item()
            
            # Calculate confidence
            confidence = F.softmax(predictions, dim=1).max(dim=1)[0].mean().item()
            
            # Calculate prediction diversity
            prediction_diversity = len(torch.unique(predicted_labels)) / len(test_labels)
        
        performance_info = {
            'accuracy': accuracy,
            'loss': loss,
            'confidence': confidence,
            'prediction_diversity': prediction_diversity,
            'timestamp': time.time()
        }
        
        # Store history
        self.performance_history.append(performance_info)
        
        return performance_info
    
    def analyze_performance_issues(self, performance_info: Dict[str, Any]) -> List[ErrorType]:
        """Analyze performance issues"""
        issues = []
        
        if performance_info['accuracy'] < 0.5:
            issues.append(ErrorType.UNDERFITTING)
        
        if performance_info['confidence'] > 0.99:
            issues.append(ErrorType.OVERFITTING)
        
        if performance_info['prediction_diversity'] < 0.1:
            issues.append(ErrorType.CONVERGENCE_ISSUES)
        
        return list(set(issues))
    
    def get_performance_statistics(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.performance_history:
            return {}
        
        accuracies = [h['accuracy'] for h in self.performance_history]
        losses = [h['loss'] for h in self.performance_history]
        confidences = [h['confidence'] for h in self.performance_history]
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'max_accuracy': np.max(accuracies),
            'min_accuracy': np.min(accuracies),
            'mean_loss': np.mean(losses),
            'mean_confidence': np.mean(confidences),
            'accuracy_trend': np.polyfit(range(len(accuracies)), accuracies, 1)[0] if len(accuracies) > 1 else 0
        }

class ModelDebugger:
    """Main model debugging system"""
    
    def __init__(self, config: DebuggingConfig):
        self.config = config
        
        # Components
        self.gradient_monitor = GradientMonitor(config)
        self.activation_monitor = ActivationMonitor(config)
        self.weight_monitor = WeightMonitor(config)
        self.loss_monitor = LossMonitor(config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        
        # Debugging state
        self.debug_history = []
        self.error_history = []
        self.fix_history = []
        
        logger.info("✅ Model Debugger initialized")
    
    def debug_model(self, model: nn.Module, input_tensor: torch.Tensor = None,
                   test_data: torch.Tensor = None, test_labels: torch.Tensor = None,
                   loss: float = None, epoch: int = None) -> Dict[str, Any]:
        """Debug model comprehensively"""
        logger.info("🔍 Starting model debugging")
        
        debug_results = {}
        
        # Gradient monitoring
        if self.config.enable_gradient_monitoring:
            gradient_info = self.gradient_monitor.monitor_gradients(model)
            gradient_issues = self.gradient_monitor.analyze_gradient_issues(gradient_info)
            debug_results['gradients'] = {
                'info': gradient_info,
                'issues': gradient_issues,
                'statistics': self.gradient_monitor.get_gradient_statistics()
            }
        
        # Activation monitoring
        if self.config.enable_activation_monitoring and input_tensor is not None:
            activation_info = self.activation_monitor.monitor_activations(model, input_tensor)
            activation_issues = self.activation_monitor.analyze_activation_issues(activation_info)
            debug_results['activations'] = {
                'info': activation_info,
                'issues': activation_issues,
                'statistics': self.activation_monitor.get_activation_statistics()
            }
        
        # Weight monitoring
        if self.config.enable_weight_monitoring:
            weight_info = self.weight_monitor.monitor_weights(model)
            weight_issues = self.weight_monitor.analyze_weight_issues(weight_info)
            debug_results['weights'] = {
                'info': weight_info,
                'issues': weight_issues,
                'statistics': self.weight_monitor.get_weight_statistics()
            }
        
        # Loss monitoring
        if self.config.enable_loss_monitoring and loss is not None:
            loss_info = self.loss_monitor.monitor_loss(loss, epoch)
            loss_issues = self.loss_monitor.analyze_loss_issues(self.loss_monitor.loss_history)
            debug_results['loss'] = {
                'info': loss_info,
                'issues': loss_issues,
                'statistics': self.loss_monitor.get_loss_statistics()
            }
        
        # Performance analysis
        if test_data is not None and test_labels is not None:
            performance_info = self.performance_analyzer.analyze_performance(
                model, test_data, test_labels
            )
            performance_issues = self.performance_analyzer.analyze_performance_issues(performance_info)
            debug_results['performance'] = {
                'info': performance_info,
                'issues': performance_issues,
                'statistics': self.performance_analyzer.get_performance_statistics()
            }
        
        # Store debug history
        self.debug_history.append({
            'results': debug_results,
            'timestamp': time.time(),
            'epoch': epoch
        })
        
        return debug_results
    
    def identify_issues(self, debug_results: Dict[str, Any]) -> List[ErrorType]:
        """Identify all issues from debug results"""
        all_issues = []
        
        for component, data in debug_results.items():
            if 'issues' in data:
                all_issues.extend(data['issues'])
        
        return list(set(all_issues))
    
    def suggest_fixes(self, issues: List[ErrorType]) -> Dict[ErrorType, List[str]]:
        """Suggest fixes for identified issues"""
        fixes = {}
        
        for issue in issues:
            if issue == ErrorType.GRADIENT_EXPLOSION:
                fixes[issue] = [
                    "Reduce learning rate",
                    "Apply gradient clipping",
                    "Use batch normalization",
                    "Check for exploding activations"
                ]
            elif issue == ErrorType.GRADIENT_VANISHING:
                fixes[issue] = [
                    "Increase learning rate",
                    "Use residual connections",
                    "Use different activation functions",
                    "Check for dead neurons"
                ]
            elif issue == ErrorType.OVERFITTING:
                fixes[issue] = [
                    "Add regularization (L1/L2)",
                    "Use dropout",
                    "Increase training data",
                    "Reduce model complexity"
                ]
            elif issue == ErrorType.UNDERFITTING:
                fixes[issue] = [
                    "Increase model complexity",
                    "Reduce regularization",
                    "Increase training time",
                    "Check data quality"
                ]
            elif issue == ErrorType.CONVERGENCE_ISSUES:
                fixes[issue] = [
                    "Adjust learning rate schedule",
                    "Use different optimizer",
                    "Check data preprocessing",
                    "Verify loss function"
                ]
            elif issue == ErrorType.NUMERICAL_INSTABILITY:
                fixes[issue] = [
                    "Use mixed precision training",
                    "Check for NaN values",
                    "Use stable activation functions",
                    "Normalize inputs"
                ]
        
        return fixes
    
    def apply_automatic_fixes(self, model: nn.Module, issues: List[ErrorType]) -> nn.Module:
        """Apply automatic fixes to model"""
        if not self.config.enable_automatic_fixes:
            return model
        
        fixed_model = model
        
        for issue in issues:
            if issue == ErrorType.GRADIENT_EXPLOSION:
                # Apply gradient clipping
                for param in fixed_model.parameters():
                    if param.grad is not None:
                        torch.nn.utils.clip_grad_norm_(fixed_model.parameters(), max_norm=1.0)
            
            elif issue == ErrorType.GRADIENT_VANISHING:
                # Add residual connections (simplified)
                # This would require more complex model modification
        
        return fixed_model
    
    def generate_debug_report(self, debug_results: Dict[str, Any]) -> str:
        """Generate comprehensive debug report"""
        report = []
        report.append("=" * 50)
        report.append("MODEL DEBUGGING REPORT")
        report.append("=" * 50)
        
        # Overall summary
        all_issues = self.identify_issues(debug_results)
        report.append(f"\nOverall Issues Found: {len(all_issues)}")
        for issue in all_issues:
            report.append(f"  - {issue.value}")
        
        # Component-wise analysis
        for component, data in debug_results.items():
            report.append(f"\n{component.upper()} ANALYSIS:")
            report.append("-" * 30)
            
            if 'issues' in data and data['issues']:
                report.append(f"Issues: {', '.join([issue.value for issue in data['issues']])}")
            else:
                report.append("No issues detected")
            
            if 'statistics' in data:
                stats = data['statistics']
                for stat_name, stat_value in stats.items():
                    report.append(f"{stat_name}: {stat_value:.6f}")
        
        # Suggested fixes
        if all_issues:
            report.append(f"\nSUGGESTED FIXES:")
            report.append("-" * 20)
            fixes = self.suggest_fixes(all_issues)
            for issue, fix_list in fixes.items():
                report.append(f"\n{issue.value}:")
                for fix in fix_list:
                    report.append(f"  - {fix}")
        
        return "\n".join(report)
    
    def visualize_debugging(self, save_path: str = None):
        """Visualize debugging results"""
        if not self.debug_history:
            logger.warning("No debug history to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract data for visualization
        epochs = [h.get('epoch', i) for i, h in enumerate(self.debug_history)]
        
        # Plot 1: Gradient norms over time
        if 'gradients' in self.debug_history[0]['results']:
            gradient_norms = []
            for h in self.debug_history:
                if 'gradients' in h['results'] and 'statistics' in h['results']['gradients']:
                    gradient_norms.append(h['results']['gradients']['statistics'].get('mean_norm', 0))
                else:
                    gradient_norms.append(0)
            
            axes[0, 0].plot(epochs, gradient_norms, 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Mean Gradient Norm')
            axes[0, 0].set_title('Gradient Norms Over Time')
            axes[0, 0].grid(True)
        
        # Plot 2: Loss over time
        if 'loss' in self.debug_history[0]['results']:
            losses = []
            for h in self.debug_history:
                if 'loss' in h['results'] and 'statistics' in h['results']['loss']:
                    losses.append(h['results']['loss']['statistics'].get('current_loss', 0))
                else:
                    losses.append(0)
            
            axes[0, 1].plot(epochs, losses, 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Loss Over Time')
            axes[0, 1].grid(True)
        
        # Plot 3: Accuracy over time
        if 'performance' in self.debug_history[0]['results']:
            accuracies = []
            for h in self.debug_history:
                if 'performance' in h['results'] and 'statistics' in h['results']['performance']:
                    accuracies.append(h['results']['performance']['statistics'].get('mean_accuracy', 0))
                else:
                    accuracies.append(0)
            
            axes[0, 2].plot(epochs, accuracies, 'g-', linewidth=2)
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].set_title('Accuracy Over Time')
            axes[0, 2].grid(True)
        
        # Plot 4: Issue frequency
        issue_counts = defaultdict(int)
        for h in self.debug_history:
            issues = self.identify_issues(h['results'])
            for issue in issues:
                issue_counts[issue.value] += 1
        
        if issue_counts:
            axes[1, 0].bar(issue_counts.keys(), issue_counts.values(), color='orange')
            axes[1, 0].set_xlabel('Issue Type')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Issue Frequency')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Weight norms over time
        if 'weights' in self.debug_history[0]['results']:
            weight_norms = []
            for h in self.debug_history:
                if 'weights' in h['results'] and 'statistics' in h['results']['weights']:
                    weight_norms.append(h['results']['weights']['statistics'].get('mean_norm', 0))
                else:
                    weight_norms.append(0)
            
            axes[1, 1].plot(epochs, weight_norms, 'purple', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Mean Weight Norm')
            axes[1, 1].set_title('Weight Norms Over Time')
            axes[1, 1].grid(True)
        
        # Plot 6: Activation norms over time
        if 'activations' in self.debug_history[0]['results']:
            activation_norms = []
            for h in self.debug_history:
                if 'activations' in h['results'] and 'statistics' in h['results']['activations']:
                    activation_norms.append(h['results']['activations']['statistics'].get('mean_norm', 0))
                else:
                    activation_norms.append(0)
            
            axes[1, 2].plot(epochs, activation_norms, 'brown', linewidth=2)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Mean Activation Norm')
            axes[1, 2].set_title('Activation Norms Over Time')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

# Factory functions
def create_debugging_config(**kwargs) -> DebuggingConfig:
    """Create debugging configuration"""
    return DebuggingConfig(**kwargs)

def create_gradient_monitor(config: DebuggingConfig) -> GradientMonitor:
    """Create gradient monitor"""
    return GradientMonitor(config)

def create_activation_monitor(config: DebuggingConfig) -> ActivationMonitor:
    """Create activation monitor"""
    return ActivationMonitor(config)

def create_weight_monitor(config: DebuggingConfig) -> WeightMonitor:
    """Create weight monitor"""
    return WeightMonitor(config)

def create_loss_monitor(config: DebuggingConfig) -> LossMonitor:
    """Create loss monitor"""
    return LossMonitor(config)

def create_performance_analyzer(config: DebuggingConfig) -> PerformanceAnalyzer:
    """Create performance analyzer"""
    return PerformanceAnalyzer(config)

def create_model_debugger(config: DebuggingConfig) -> ModelDebugger:
    """Create model debugger"""
    return ModelDebugger(config)

# Example usage
def example_model_debugging():
    """Example of model debugging"""
    # Create configuration
    config = create_debugging_config(
        debugging_level=DebuggingLevel.INTERMEDIATE,
        enable_gradient_monitoring=True,
        enable_activation_monitoring=True,
        enable_weight_monitoring=True,
        enable_loss_monitoring=True,
        gradient_threshold=1.0,
        activation_threshold=10.0,
        weight_threshold=5.0,
        loss_threshold=100.0,
        analysis_frequency=10,
        history_length=1000,
        enable_real_time_monitoring=True,
        enable_automatic_fixes=False,
        enable_error_prediction=True,
        enable_performance_optimization=True,
        enable_memory_optimization=True
    )
    
    # Create model debugger
    model_debugger = create_model_debugger(config)
    
    # Create dummy model
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Create dummy data
    np.random.seed(42)
    input_tensor = torch.randn(1, 784)
    test_data = torch.randn(100, 784)
    test_labels = torch.randint(0, 10, (100,))
    
    # Simulate training with debugging
    for epoch in range(5):
        # Forward pass
        output = model(input_tensor)
        loss = F.cross_entropy(output, torch.randint(0, 10, (1,)))
        
        # Backward pass
        loss.backward()
        
        # Debug model
        debug_results = model_debugger.debug_model(
            model, input_tensor, test_data, test_labels, loss.item(), epoch
        )
        
        # Clear gradients
        model.zero_grad()
    
    # Generate debug report
    debug_report = model_debugger.generate_debug_report(debug_results)
    
    print(f"✅ Model Debugging Example Complete!")
    print(f"🔍 Model Debugging Statistics:")
    print(f"   Debugging Level: {config.debugging_level.value}")
    print(f"   Gradient Monitoring: {'Enabled' if config.enable_gradient_monitoring else 'Disabled'}")
    print(f"   Activation Monitoring: {'Enabled' if config.enable_activation_monitoring else 'Disabled'}")
    print(f"   Weight Monitoring: {'Enabled' if config.enable_weight_monitoring else 'Disabled'}")
    print(f"   Loss Monitoring: {'Enabled' if config.enable_loss_monitoring else 'Disabled'}")
    print(f"   Gradient Threshold: {config.gradient_threshold}")
    print(f"   Activation Threshold: {config.activation_threshold}")
    print(f"   Weight Threshold: {config.weight_threshold}")
    print(f"   Loss Threshold: {config.loss_threshold}")
    print(f"   Analysis Frequency: {config.analysis_frequency}")
    print(f"   History Length: {config.history_length}")
    print(f"   Real-time Monitoring: {'Enabled' if config.enable_real_time_monitoring else 'Disabled'}")
    print(f"   Automatic Fixes: {'Enabled' if config.enable_automatic_fixes else 'Disabled'}")
    print(f"   Error Prediction: {'Enabled' if config.enable_error_prediction else 'Disabled'}")
    print(f"   Performance Optimization: {'Enabled' if config.enable_performance_optimization else 'Disabled'}")
    print(f"   Memory Optimization: {'Enabled' if config.enable_memory_optimization else 'Disabled'}")
    
    print(f"\n📊 Debugging Results:")
    print(f"   Debug History Length: {len(model_debugger.debug_history)}")
    print(f"   Error History Length: {len(model_debugger.error_history)}")
    print(f"   Fix History Length: {len(model_debugger.fix_history)}")
    
    # Identify issues
    all_issues = model_debugger.identify_issues(debug_results)
    print(f"   Issues Found: {len(all_issues)}")
    for issue in all_issues:
        print(f"     - {issue.value}")
    
    print(f"\n🔧 Suggested Fixes:")
    fixes = model_debugger.suggest_fixes(all_issues)
    for issue, fix_list in fixes.items():
        print(f"   {issue.value}:")
        for fix in fix_list:
            print(f"     - {fix}")
    
    print(f"\n📋 Debug Report:")
    print(debug_report)
    
    return model_debugger

# Export utilities
__all__ = [
    'DebuggingLevel',
    'ErrorType',
    'DiagnosticType',
    'DebuggingConfig',
    'GradientMonitor',
    'ActivationMonitor',
    'WeightMonitor',
    'LossMonitor',
    'PerformanceAnalyzer',
    'ModelDebugger',
    'create_debugging_config',
    'create_gradient_monitor',
    'create_activation_monitor',
    'create_weight_monitor',
    'create_loss_monitor',
    'create_performance_analyzer',
    'create_model_debugger',
    'example_model_debugging'
]

if __name__ == "__main__":
    example_model_debugging()
    print("✅ Model debugging example completed successfully!")

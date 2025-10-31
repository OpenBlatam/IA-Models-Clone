from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass
import time
from typing import Any, List, Dict, Optional
import asyncio
"""
Gradient Analysis Tools for HeyGen AI.

Advanced gradient analysis and monitoring using PyTorch autograd
for deep learning model optimization following PEP 8 style guidelines.
"""


logger = logging.getLogger(__name__)


@dataclass
class GradientAnalysisConfig:
    """Configuration for gradient analysis."""

    # Analysis parameters
    compute_hessian: bool = False
    compute_gradient_norms: bool = True
    compute_gradient_angles: bool = True
    compute_gradient_correlation: bool = True
    compute_gradient_flow: bool = True
    
    # Monitoring parameters
    log_every_n_steps: int = 100
    save_gradient_plots: bool = True
    plot_dir: str = "gradient_plots"
    
    # Analysis thresholds
    gradient_norm_threshold: float = 1.0
    gradient_angle_threshold: float = 0.1
    correlation_threshold: float = 0.8


class GradientAnalyzer:
    """Advanced gradient analyzer using PyTorch autograd."""

    def __init__(self, config: GradientAnalysisConfig):
        """Initialize gradient analyzer.

        Args:
            config: Gradient analysis configuration.
        """
        self.config = config
        self.gradient_history = []
        self.gradient_norms_history = []
        self.gradient_angles_history = []
        self.gradient_correlation_history = []
        self.gradient_flow_history = []

    def analyze_gradients(
        self,
        model: nn.Module,
        loss_function: Callable,
        input_data: torch.Tensor,
        target_data: torch.Tensor
    ) -> Dict[str, Any]:
        """Comprehensive gradient analysis.

        Args:
            model: PyTorch model.
            loss_function: Loss function.
            input_data: Input data.
            target_data: Target data.

        Returns:
            Dict[str, Any]: Gradient analysis results.
        """
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        output = model(input_data)
        loss = loss_function(output, target_data)
        
        # Backward pass
        loss.backward()
        
        analysis_results = {
            "loss": loss.item(),
            "gradients": {},
            "gradient_norms": {},
            "gradient_angles": {},
            "gradient_correlation": {},
            "gradient_flow": {}
        }
        
        # Collect gradients
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        analysis_results["gradients"] = gradients
        
        # Compute gradient norms
        if self.config.compute_gradient_norms:
            gradient_norms = self._compute_gradient_norms(gradients)
            analysis_results["gradient_norms"] = gradient_norms
            self.gradient_norms_history.append(gradient_norms)
        
        # Compute gradient angles
        if self.config.compute_gradient_angles:
            gradient_angles = self._compute_gradient_angles(gradients)
            analysis_results["gradient_angles"] = gradient_angles
            self.gradient_angles_history.append(gradient_angles)
        
        # Compute gradient correlation
        if self.config.compute_gradient_correlation:
            gradient_correlation = self._compute_gradient_correlation(gradients)
            analysis_results["gradient_correlation"] = gradient_correlation
            self.gradient_correlation_history.append(gradient_correlation)
        
        # Compute gradient flow
        if self.config.compute_gradient_flow:
            gradient_flow = self._compute_gradient_flow(gradients)
            analysis_results["gradient_flow"] = gradient_flow
            self.gradient_flow_history.append(gradient_flow)
        
        # Store in history
        self.gradient_history.append(analysis_results)
        
        return analysis_results

    def _compute_gradient_norms(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute gradient norms for each parameter.

        Args:
            gradients: Dictionary of gradients.

        Returns:
            Dict[str, float]: Gradient norms for each parameter.
        """
        gradient_norms = {}
        
        for name, grad_tensor in gradients.items():
            gradient_norms[name] = grad_tensor.norm().item()
        
        return gradient_norms

    def _compute_gradient_angles(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute gradient angles between consecutive layers.

        Args:
            gradients: Dictionary of gradients.

        Returns:
            Dict[str, float]: Gradient angles between layers.
        """
        gradient_angles = {}
        gradient_names = list(gradients.keys())
        
        for i in range(len(gradient_names) - 1):
            current_grad = gradients[gradient_names[i]].flatten()
            next_grad = gradients[gradient_names[i + 1]].flatten()
            
            # Compute cosine similarity
            cos_sim = torch.dot(current_grad, next_grad) / (
                current_grad.norm() * next_grad.norm()
            )
            
            # Convert to angle
            angle = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)).item()
            gradient_angles[f"{gradient_names[i]}_to_{gradient_names[i+1]}"] = angle
        
        return gradient_angles

    def _compute_gradient_correlation(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute gradient correlation between parameters.

        Args:
            gradients: Dictionary of gradients.

        Returns:
            Dict[str, float]: Gradient correlations.
        """
        gradient_correlation = {}
        gradient_names = list(gradients.keys())
        
        for i in range(len(gradient_names)):
            for j in range(i + 1, len(gradient_names)):
                grad1 = gradients[gradient_names[i]].flatten()
                grad2 = gradients[gradient_names[j]].flatten()
                
                # Compute correlation
                correlation = torch.corrcoef(
                    torch.stack([grad1, grad2])
                )[0, 1].item()
                
                gradient_correlation[f"{gradient_names[i]}_{gradient_names[j]}"] = correlation
        
        return gradient_correlation

    def _compute_gradient_flow(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute gradient flow metrics.

        Args:
            gradients: Dictionary of gradients.

        Returns:
            Dict[str, float]: Gradient flow metrics.
        """
        gradient_flow = {}
        
        # Compute total gradient norm
        total_grad_norm = 0.0
        for grad_tensor in gradients.values():
            total_grad_norm += grad_tensor.norm().item() ** 2
        total_grad_norm = np.sqrt(total_grad_norm)
        
        gradient_flow["total_gradient_norm"] = total_grad_norm
        
        # Compute gradient variance
        all_gradients = torch.cat([g.flatten() for g in gradients.values()])
        gradient_flow["gradient_variance"] = all_gradients.var().item()
        
        # Compute gradient mean
        gradient_flow["gradient_mean"] = all_gradients.mean().item()
        
        # Compute gradient sparsity
        gradient_flow["gradient_sparsity"] = (
            (all_gradients == 0).float().mean().item()
        )
        
        return gradient_flow

    def detect_gradient_issues(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Detect potential gradient issues.

        Args:
            analysis_results: Gradient analysis results.

        Returns:
            List[str]: List of detected issues.
        """
        issues = []
        
        # Check for exploding gradients
        gradient_norms = analysis_results.get("gradient_norms", {})
        for name, norm in gradient_norms.items():
            if norm > self.config.gradient_norm_threshold:
                issues.append(f"Exploding gradients in {name}: {norm:.4f}")
        
        # Check for vanishing gradients
        for name, norm in gradient_norms.items():
            if norm < 1e-6:
                issues.append(f"Vanishing gradients in {name}: {norm:.4f}")
        
        # Check gradient angles
        gradient_angles = analysis_results.get("gradient_angles", {})
        for name, angle in gradient_angles.items():
            if angle < self.config.gradient_angle_threshold:
                issues.append(f"Small gradient angle in {name}: {angle:.4f}")
        
        # Check gradient correlation
        gradient_correlation = analysis_results.get("gradient_correlation", {})
        for name, correlation in gradient_correlation.items():
            if abs(correlation) > self.config.correlation_threshold:
                issues.append(f"High gradient correlation in {name}: {correlation:.4f}")
        
        return issues

    def plot_gradient_analysis(self, save_path: Optional[str] = None):
        """Plot gradient analysis results.

        Args:
            save_path: Optional path to save plots.
        """
        if not self.gradient_history:
            logger.warning("No gradient history to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Gradient Analysis Results")
        
        # Plot gradient norms over time
        if self.gradient_norms_history:
            ax1 = axes[0, 0]
            steps = range(len(self.gradient_norms_history))
            for param_name in self.gradient_norms_history[0].keys():
                norms = [step[param_name] for step in self.gradient_norms_history]
                ax1.plot(steps, norms, label=param_name)
            ax1.set_title("Gradient Norms Over Time")
            ax1.set_xlabel("Training Step")
            ax1.set_ylabel("Gradient Norm")
            ax1.legend()
            ax1.grid(True)
        
        # Plot gradient flow metrics
        if self.gradient_flow_history:
            ax2 = axes[0, 1]
            steps = range(len(self.gradient_flow_history))
            flow_metrics = ["total_gradient_norm", "gradient_variance", "gradient_sparsity"]
            for metric in flow_metrics:
                values = [step[metric] for step in self.gradient_flow_history]
                ax2.plot(steps, values, label=metric)
            ax2.set_title("Gradient Flow Metrics")
            ax2.set_xlabel("Training Step")
            ax2.set_ylabel("Metric Value")
            ax2.legend()
            ax2.grid(True)
        
        # Plot gradient correlation heatmap
        if self.gradient_correlation_history:
            ax3 = axes[1, 0]
            latest_correlation = self.gradient_correlation_history[-1]
            if latest_correlation:
                param_names = list(latest_correlation.keys())
                correlation_matrix = np.zeros((len(param_names), len(param_names)))
                
                for i, name1 in enumerate(param_names):
                    for j, name2 in enumerate(param_names):
                        key = f"{name1}_{name2}" if name1 < name2 else f"{name2}_{name1}"
                        if key in latest_correlation:
                            correlation_matrix[i, j] = latest_correlation[key]
                
                im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax3.set_title("Gradient Correlation Matrix")
                ax3.set_xticks(range(len(param_names)))
                ax3.set_yticks(range(len(param_names)))
                ax3.set_xticklabels(param_names, rotation=45)
                ax3.set_yticklabels(param_names)
                plt.colorbar(im, ax=ax3)
        
        # Plot gradient angles
        if self.gradient_angles_history:
            ax4 = axes[1, 1]
            steps = range(len(self.gradient_angles_history))
            for angle_name in self.gradient_angles_history[0].keys():
                angles = [step[angle_name] for step in self.gradient_angles_history]
                ax4.plot(steps, angles, label=angle_name)
            ax4.set_title("Gradient Angles Over Time")
            ax4.set_xlabel("Training Step")
            ax4.set_ylabel("Angle (radians)")
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

    def get_gradient_statistics(self) -> Dict[str, Any]:
        """Get comprehensive gradient statistics.

        Returns:
            Dict[str, Any]: Gradient statistics.
        """
        if not self.gradient_history:
            return {}
        
        stats = {
            "total_steps": len(self.gradient_history),
            "average_loss": np.mean([step["loss"] for step in self.gradient_history]),
            "loss_std": np.std([step["loss"] for step in self.gradient_history]),
            "gradient_norms": {},
            "gradient_flow": {}
        }
        
        # Compute average gradient norms
        if self.gradient_norms_history:
            param_names = self.gradient_norms_history[0].keys()
            for param_name in param_names:
                norms = [step[param_name] for step in self.gradient_norms_history]
                stats["gradient_norms"][param_name] = {
                    "mean": np.mean(norms),
                    "std": np.std(norms),
                    "min": np.min(norms),
                    "max": np.max(norms)
                }
        
        # Compute average gradient flow metrics
        if self.gradient_flow_history:
            flow_metrics = self.gradient_flow_history[0].keys()
            for metric in flow_metrics:
                values = [step[metric] for step in self.gradient_flow_history]
                stats["gradient_flow"][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
        
        return stats


class GradientMonitoring:
    """Real-time gradient monitoring during training."""

    def __init__(self, config: GradientAnalysisConfig):
        """Initialize gradient monitoring.

        Args:
            config: Gradient analysis configuration.
        """
        self.config = config
        self.analyzer = GradientAnalyzer(config)
        self.monitoring_stats = []
        self.issue_history = []

    def monitor_training_step(
        self,
        model: nn.Module,
        loss_function: Callable,
        input_data: torch.Tensor,
        target_data: torch.Tensor,
        step: int
    ) -> Dict[str, Any]:
        """Monitor gradients during training step.

        Args:
            model: PyTorch model.
            loss_function: Loss function.
            input_data: Input data.
            target_data: Target data.
            step: Current training step.

        Returns:
            Dict[str, Any]: Monitoring results.
        """
        # Perform gradient analysis
        analysis_results = self.analyzer.analyze_gradients(
            model, loss_function, input_data, target_data
        )
        
        # Detect issues
        issues = self.analyzer.detect_gradient_issues(analysis_results)
        
        # Store monitoring results
        monitoring_result = {
            "step": step,
            "analysis": analysis_results,
            "issues": issues,
            "timestamp": time.time()
        }
        
        self.monitoring_stats.append(monitoring_result)
        
        # Log issues
        if issues:
            self.issue_history.extend(issues)
            logger.warning(f"Step {step}: {len(issues)} gradient issues detected")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        # Log progress
        if step % self.config.log_every_n_steps == 0:
            logger.info(
                f"Step {step}: Loss = {analysis_results['loss']:.4f}, "
                f"Issues = {len(issues)}"
            )
        
        return monitoring_result

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary.

        Returns:
            Dict[str, Any]: Monitoring summary.
        """
        if not self.monitoring_stats:
            return {}
        
        total_steps = len(self.monitoring_stats)
        total_issues = len(self.issue_history)
        
        # Get gradient statistics
        gradient_stats = self.analyzer.get_gradient_statistics()
        
        summary = {
            "total_steps": total_steps,
            "total_issues": total_issues,
            "issue_rate": total_issues / total_steps if total_steps > 0 else 0,
            "gradient_statistics": gradient_stats,
            "recent_issues": self.issue_history[-10:] if self.issue_history else []
        }
        
        return summary

    def save_monitoring_report(self, file_path: str):
        """Save monitoring report.

        Args:
            file_path: Path to save report.
        """
        summary = self.get_monitoring_summary()
        
        with open(file_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("Gradient Monitoring Report\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("=" * 50 + "\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write(f"Total Training Steps: {summary['total_steps']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"Total Issues Detected: {summary['total_issues']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"Issue Rate: {summary['issue_rate']:.4f}\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("Recent Issues:\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            for issue in summary['recent_issues']:
                f.write(f"  - {issue}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("\nGradient Statistics:\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            for category, stats in summary['gradient_statistics'].items():
                f.write(f"\n{category}:\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if isinstance(stats, dict):
                    for metric, values in stats.items():
                        if isinstance(values, dict):
                            f.write(f"  {metric}:\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                            for sub_metric, value in values.items():
                                f.write(f"    {sub_metric}: {value:.4f}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        else:
                            f.write(f"  {metric}: {values:.4f}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")


def create_gradient_analyzer(config: GradientAnalysisConfig) -> GradientAnalyzer:
    """Factory function to create gradient analyzer.

    Args:
        config: Gradient analysis configuration.

    Returns:
        GradientAnalyzer: Created gradient analyzer.
    """
    return GradientAnalyzer(config)


def create_gradient_monitoring(config: GradientAnalysisConfig) -> GradientMonitoring:
    """Factory function to create gradient monitoring.

    Args:
        config: Gradient analysis configuration.

    Returns:
        GradientMonitoring: Created gradient monitoring.
    """
    return GradientMonitoring(config) 
"""
Advanced Gradient Analysis and Debugging

Provides comprehensive gradient analysis following PyTorch best practices:
- Gradient statistics
- Gradient flow visualization
- Vanishing/exploding gradient detection
- Gradient clipping recommendations
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from collections import defaultdict

logger = logging.getLogger(__name__)


class GradientAnalyzer:
    """
    Advanced gradient analyzer for debugging training issues.
    
    Features:
    - Gradient statistics per layer
    - Vanishing/exploding gradient detection
    - Gradient flow analysis
    - Automatic recommendations
    """
    
    def __init__(
        self,
        model: nn.Module,
        threshold_vanishing: float = 1e-6,
        threshold_exploding: float = 1e3,
        track_history: bool = True
    ):
        """
        Initialize gradient analyzer.
        
        Args:
            model: Model to analyze
            threshold_vanishing: Threshold for vanishing gradients
            threshold_exploding: Threshold for exploding gradients
            track_history: Track gradient history
        """
        self.model = model
        self.threshold_vanishing = threshold_vanishing
        self.threshold_exploding = threshold_exploding
        self.track_history = track_history
        self.history: Dict[str, List[float]] = defaultdict(list)
    
    def analyze_gradients(
        self,
        step: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze gradients across all model parameters.
        
        Args:
            step: Current training step (for history tracking)
            
        Returns:
            Dictionary with gradient statistics
        """
        stats = {
            "layer_stats": {},
            "global_stats": {},
            "issues": [],
            "recommendations": []
        }
        
        all_norms = []
        all_grads = []
        
        # Analyze each parameter
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_norm = grad.norm().item()
                
                # Layer statistics
                layer_stats = {
                    "norm": grad_norm,
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "min": grad.min().item(),
                    "max": grad.max().item(),
                    "numel": grad.numel(),
                    "has_nan": torch.isnan(grad).any().item(),
                    "has_inf": torch.isinf(grad).any().item()
                }
                
                stats["layer_stats"][name] = layer_stats
                all_norms.append(grad_norm)
                all_grads.append(grad.flatten())
                
                # Track history
                if self.track_history and step is not None:
                    self.history[name].append(grad_norm)
                
                # Detect issues
                if layer_stats["has_nan"]:
                    stats["issues"].append({
                        "type": "nan",
                        "layer": name,
                        "severity": "critical"
                    })
                
                if layer_stats["has_inf"]:
                    stats["issues"].append({
                        "type": "inf",
                        "layer": name,
                        "severity": "critical"
                    })
                
                if grad_norm < self.threshold_vanishing:
                    stats["issues"].append({
                        "type": "vanishing",
                        "layer": name,
                        "norm": grad_norm,
                        "severity": "high"
                    })
                
                if grad_norm > self.threshold_exploding:
                    stats["issues"].append({
                        "type": "exploding",
                        "layer": name,
                        "norm": grad_norm,
                        "severity": "high"
                    })
        
        # Global statistics
        if all_norms:
            all_grads_tensor = torch.cat(all_grads)
            stats["global_stats"] = {
                "total_norm": torch.norm(torch.stack([torch.tensor(n) for n in all_norms])).item(),
                "mean_norm": sum(all_norms) / len(all_norms),
                "max_norm": max(all_norms),
                "min_norm": min(all_norms),
                "mean_grad": all_grads_tensor.mean().item(),
                "std_grad": all_grads_tensor.std().item()
            }
        
        # Generate recommendations
        stats["recommendations"] = self._generate_recommendations(stats)
        
        return stats
    
    def _generate_recommendations(
        self,
        stats: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on gradient analysis."""
        recommendations = []
        
        if not stats["global_stats"]:
            recommendations.append("No gradients found. Check if backward() was called.")
            return recommendations
        
        # Check for vanishing gradients
        vanishing_count = sum(
            1 for issue in stats["issues"]
            if issue["type"] == "vanishing"
        )
        if vanishing_count > 0:
            recommendations.append(
                f"{vanishing_count} layers with vanishing gradients. "
                "Consider: gradient clipping, learning rate adjustment, "
                "or skip connections."
            )
        
        # Check for exploding gradients
        exploding_count = sum(
            1 for issue in stats["issues"]
            if issue["type"] == "exploding"
        )
        if exploding_count > 0:
            recommendations.append(
                f"{exploding_count} layers with exploding gradients. "
                "Consider: gradient clipping (max_norm=1.0), "
                "lower learning rate, or gradient accumulation."
            )
        
        # Check global norm
        total_norm = stats["global_stats"].get("total_norm", 0)
        if total_norm > 10.0:
            recommendations.append(
                f"High global gradient norm ({total_norm:.2f}). "
                "Consider gradient clipping."
            )
        elif total_norm < 0.01:
            recommendations.append(
                f"Low global gradient norm ({total_norm:.2f}). "
                "Consider: learning rate increase, "
                "warmup schedule, or check data."
            )
        
        # Check for NaN/Inf
        nan_count = sum(
            1 for issue in stats["issues"]
            if issue["type"] in ["nan", "inf"]
        )
        if nan_count > 0:
            recommendations.append(
                f"{nan_count} layers with NaN/Inf gradients. "
                "Critical issue! Check: loss function, "
                "input data, or numerical stability."
            )
        
        return recommendations
    
    def get_gradient_flow(
        self,
        step: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get gradient flow through the model.
        
        Returns gradient norms for each layer, useful for
        visualizing gradient flow.
        """
        flow = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                flow[name] = param.grad.norm().item()
        return flow
    
    def check_gradient_health(self) -> Tuple[bool, List[str]]:
        """
        Quick health check of gradients.
        
        Returns:
            Tuple of (is_healthy, list_of_warnings)
        """
        stats = self.analyze_gradients()
        warnings = []
        
        # Check for critical issues
        critical_issues = [
            issue for issue in stats["issues"]
            if issue["severity"] == "critical"
        ]
        
        if critical_issues:
            warnings.append(
                f"Critical issues found: {len(critical_issues)} layers "
                "with NaN/Inf gradients"
            )
        
        # Check for high severity issues
        high_severity = [
            issue for issue in stats["issues"]
            if issue["severity"] == "high"
        ]
        
        if high_severity:
            warnings.append(
                f"High severity issues: {len(high_severity)} layers "
                "with vanishing/exploding gradients"
            )
        
        is_healthy = len(critical_issues) == 0 and len(high_severity) == 0
        
        return is_healthy, warnings
    
    def get_history_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of gradient history."""
        summary = {}
        for name, history in self.history.items():
            if history:
                summary[name] = {
                    "mean": sum(history) / len(history),
                    "min": min(history),
                    "max": max(history),
                    "std": (
                        sum((x - sum(history) / len(history))**2 for x in history) / len(history)
                    ) ** 0.5 if len(history) > 1 else 0.0
                }
        return summary


class GradientMonitor:
    """
    Real-time gradient monitoring during training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        log_interval: int = 100,
        alert_threshold: float = 1e3
    ):
        """
        Initialize gradient monitor.
        
        Args:
            model: Model to monitor
            log_interval: Steps between logging
            alert_threshold: Threshold for alerts
        """
        self.model = model
        self.log_interval = log_interval
        self.alert_threshold = alert_threshold
        self.analyzer = GradientAnalyzer(model)
        self.step_count = 0
    
    def monitor_step(self, step: int):
        """
        Monitor gradients at current step.
        
        Args:
            step: Current training step
        """
        self.step_count = step
        
        if step % self.log_interval == 0:
            stats = self.analyzer.analyze_gradients(step=step)
            
            # Log global stats
            if stats["global_stats"]:
                logger.info(
                    f"Step {step} - Gradient stats: "
                    f"total_norm={stats['global_stats']['total_norm']:.4f}, "
                    f"mean_norm={stats['global_stats']['mean_norm']:.4f}"
                )
            
            # Alert on issues
            if stats["issues"]:
                for issue in stats["issues"]:
                    if issue["severity"] == "critical":
                        logger.error(
                            f"Step {step} - Critical gradient issue: "
                            f"{issue['type']} in {issue['layer']}"
                        )
                    elif issue["severity"] == "high":
                        logger.warning(
                            f"Step {step} - High severity gradient issue: "
                            f"{issue['type']} in {issue['layer']}"
                        )
            
            # Log recommendations
            if stats["recommendations"]:
                for rec in stats["recommendations"]:
                    logger.info(f"Step {step} - Recommendation: {rec}")


__all__ = [
    "GradientAnalyzer",
    "GradientMonitor",
]




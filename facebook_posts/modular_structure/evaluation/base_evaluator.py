"""
📊 Base Evaluator Class

Abstract base class for all evaluation operations.
Provides common interface for model evaluation and metrics calculation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
import json
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    
    This class provides a common interface for:
    - Model evaluation and testing
    - Metrics calculation and analysis
    - Results visualization and reporting
    - Performance comparison and benchmarking
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], name: str = "base_evaluator"):
        """
        Initialize the base evaluator.
        
        Args:
            model: Model to evaluate
            config: Evaluation configuration dictionary
            name: Evaluator name for identification
        """
        self.model = model
        self.config = config
        self.name = name
        self.device = next(model.parameters()).device
        
        # Evaluation state
        self.evaluation_results = {}
        self.evaluation_history = []
        self.metrics_calculator = None
        
        # Initialize evaluation components
        self._setup_evaluation_components()
        self._log_evaluation_info()
    
    @abstractmethod
    def _setup_evaluation_components(self) -> None:
        """
        Setup evaluation components like metrics calculator.
        Must be implemented by subclasses.
        """
        pass
    
    def _log_evaluation_info(self) -> None:
        """Log evaluation setup information."""
        logger.info(f"Evaluator {self.name} initialized:")
        logger.info(f"  Model: {self.model.__class__.__name__}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Config: {list(self.config.keys())}")
    
    def get_evaluation_info(self) -> Dict[str, Any]:
        """Get comprehensive evaluation information."""
        return {
            "name": self.name,
            "model": self.model.__class__.__name__,
            "device": str(self.device),
            "config": self.config,
            "evaluation_results": self.evaluation_results,
            "evaluation_history": len(self.evaluation_history)
        }
    
    def get_evaluation_results(self) -> Dict[str, Any]:
        """Get current evaluation results."""
        return self.evaluation_results.copy()
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history."""
        return self.evaluation_history.copy()
    
    def evaluate_model(self, test_loader: DataLoader, 
                      save_predictions: bool = False) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            save_predictions: Whether to save predictions
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting model evaluation: {self.name}")
        start_time = time.time()
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        # Evaluation loop
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Evaluating")
            for inputs, targets in progress_bar:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Store predictions and targets
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())
                all_inputs.append(inputs.cpu())
                
                # Update progress bar
                progress_bar.set_postfix({'batch': len(all_predictions)})
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_inputs = torch.cat(all_inputs, dim=0)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        # Store results
        evaluation_result = {
            "timestamp": time.time(),
            "metrics": metrics,
            "predictions_shape": list(all_predictions.shape),
            "targets_shape": list(all_targets.shape),
            "inputs_shape": list(all_inputs.shape),
            "evaluation_time": time.time() - start_time
        }
        
        # Save predictions if requested
        if save_predictions:
            predictions_path = self.config.get("predictions_save_path", "predictions.pt")
            torch.save({
                "predictions": all_predictions,
                "targets": all_targets,
                "inputs": all_inputs,
                "metrics": metrics
            }, predictions_path)
            evaluation_result["predictions_saved"] = predictions_path
        
        # Update evaluation state
        self.evaluation_results = evaluation_result
        self.evaluation_history.append(evaluation_result)
        
        logger.info(f"Evaluation completed in {evaluation_result['evaluation_time']:.2f}s")
        logger.info(f"Metrics: {metrics}")
        
        return evaluation_result
    
    @abstractmethod
    def _calculate_metrics(self, predictions: torch.Tensor, 
                          targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        Must be implemented by subclasses.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary containing calculated metrics
        """
        pass
    
    def evaluate_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluate a single batch.
        
        Args:
            inputs: Input batch
            targets: Target batch
            
        Returns:
            Dictionary containing batch evaluation results
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate batch metrics
            batch_metrics = self._calculate_metrics(outputs, targets)
            
            # Calculate loss if criterion is available
            if hasattr(self, 'criterion') and self.criterion:
                loss = self.criterion(outputs, targets).item()
                batch_metrics['loss'] = loss
        
        return {
            "predictions": outputs.cpu(),
            "targets": targets.cpu(),
            "inputs": inputs.cpu(),
            "metrics": batch_metrics
        }
    
    def compare_models(self, other_evaluator: 'BaseEvaluator', 
                      test_loader: DataLoader) -> Dict[str, Any]:
        """
        Compare this model with another model.
        
        Args:
            other_evaluator: Another evaluator to compare with
            test_loader: Test data loader
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Comparing models: {self.name} vs {other_evaluator.name}")
        
        # Evaluate both models
        self_results = self.evaluate_model(test_loader)
        other_results = other_evaluator.evaluate_model(test_loader)
        
        # Create comparison
        comparison = {
            "model_1": {
                "name": self.name,
                "results": self_results
            },
            "model_2": {
                "name": other_evaluator.name,
                "results": other_results
            },
            "comparison": {}
        }
        
        # Compare metrics
        for metric_name in self_results["metrics"].keys():
            if metric_name in other_results["metrics"]:
                val1 = self_results["metrics"][metric_name]
                val2 = other_results["metrics"][metric_name]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff = val1 - val2
                    improvement = (diff / val2 * 100) if val2 != 0 else float('inf')
                    
                    comparison["comparison"][metric_name] = {
                        "model_1_value": val1,
                        "model_2_value": val2,
                        "difference": diff,
                        "improvement_percent": improvement,
                        "winner": self.name if diff > 0 else other_evaluator.name
                    }
        
        # Compare evaluation times
        time_diff = self_results["evaluation_time"] - other_results["evaluation_time"]
        comparison["comparison"]["evaluation_time"] = {
            "model_1_time": self_results["evaluation_time"],
            "model_2_time": other_results["evaluation_time"],
            "time_difference": time_diff,
            "faster_model": self.name if time_diff < 0 else other_evaluator.name
        }
        
        return comparison
    
    def save_evaluation_results(self, path: Union[str, Path]) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            path: Path to save the evaluation results
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "evaluation_results": self.evaluation_results,
            "evaluation_history": self.evaluation_history,
            "evaluation_info": self.get_evaluation_info()
        }
        
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to: {path}")
    
    def load_evaluation_results(self, path: Union[str, Path]) -> None:
        """
        Load evaluation results from JSON file.
        
        Args:
            path: Path to load the evaluation results from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation results file not found: {path}")
        
        with open(path, 'r') as f:
            data_dict = json.load(f)
        
        # Update evaluation state
        if "evaluation_results" in data_dict:
            self.evaluation_results = data_dict["evaluation_results"]
        if "evaluation_history" in data_dict:
            self.evaluation_history = data_dict["evaluation_history"]
        
        logger.info(f"Evaluation results loaded from: {path}")
    
    def get_evaluation_summary(self) -> str:
        """
        Generate a summary of the evaluation.
        
        Returns:
            String summary of the evaluation
        """
        if not self.evaluation_results:
            return f"Evaluation Summary: {self.name}\n{'=' * 50}\nNo evaluation results available."
        
        metrics = self.evaluation_results.get("metrics", {})
        evaluation_time = self.evaluation_results.get("evaluation_time", 0)
        
        summary = f"""
Evaluation Summary: {self.name}
{'=' * 50}
Evaluation Time: {evaluation_time:.2f}s
Predictions Shape: {self.evaluation_results.get('predictions_shape', 'N/A')}
Targets Shape: {self.evaluation_results.get('targets_shape', 'N/A')}

Metrics:
"""
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                summary += f"  - {metric_name}: {metric_value:.4f}\n"
            else:
                summary += f"  - {metric_name}: {metric_value}\n"
        
        summary += f"\nTotal Evaluations: {len(self.evaluation_history)}"
        
        return summary.strip()
    
    def reset_evaluation_state(self) -> None:
        """Reset evaluation state and clear results."""
        self.evaluation_results = {}
        self.evaluation_history = []
        logger.info(f"Evaluation state reset for: {self.name}")
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary.
        
        Returns:
            Dictionary containing performance summary
        """
        if not self.evaluation_results:
            return {"error": "No evaluation results available"}
        
        metrics = self.evaluation_results.get("metrics", {})
        
        # Calculate performance statistics
        performance_summary = {
            "model_name": self.name,
            "evaluation_timestamp": self.evaluation_results.get("timestamp"),
            "evaluation_time": self.evaluation_results.get("evaluation_time"),
            "metrics": metrics,
            "metrics_count": len(metrics),
            "predictions_info": {
                "shape": self.evaluation_results.get("predictions_shape"),
                "dtype": str(type(metrics.get(list(metrics.keys())[0]))) if metrics else "N/A"
            }
        }
        
        # Add metric statistics if available
        numeric_metrics = [v for v in metrics.values() if isinstance(v, (int, float))]
        if numeric_metrics:
            performance_summary["metric_statistics"] = {
                "min": min(numeric_metrics),
                "max": max(numeric_metrics),
                "mean": np.mean(numeric_metrics),
                "std": np.std(numeric_metrics)
            }
        
        return performance_summary







"""
A/B Testing Framework for Model Comparison
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ABTest:
    """A/B test for comparing models"""
    
    def __init__(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        split_ratio: float = 0.5,
        random_seed: Optional[int] = None
    ):
        """
        Initialize A/B test
        
        Args:
            model_a: Model A (control)
            model_b: Model B (treatment)
            split_ratio: Ratio for model A (rest goes to B)
            random_seed: Random seed for splitting
        """
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.results_a = []
        self.results_b = []
        self.metadata = {
            "started_at": datetime.now().isoformat(),
            "split_ratio": split_ratio
        }
        
        logger.info(f"ABTest initialized: split_ratio={split_ratio}")
    
    def assign_model(self, user_id: str) -> str:
        """
        Assign model to user
        
        Args:
            user_id: User identifier
        
        Returns:
            Model identifier ("A" or "B")
        """
        # Deterministic assignment based on user_id hash
        hash_value = hash(user_id) % 100
        threshold = self.split_ratio * 100
        
        return "A" if hash_value < threshold else "B"
    
    def predict(
        self,
        user_id: str,
        inputs: torch.Tensor,
        model_id: Optional[str] = None
    ) -> Tuple[torch.Tensor, str]:
        """
        Get prediction from assigned model
        
        Args:
            user_id: User identifier
            inputs: Input tensor
            model_id: Optional model ID (if None, assigns automatically)
        
        Returns:
            Tuple of (prediction, model_id)
        """
        if model_id is None:
            model_id = self.assign_model(user_id)
        
        if model_id == "A":
            model = self.model_a
        else:
            model = self.model_b
        
        model.eval()
        with torch.no_grad():
            prediction = model(inputs)
        
        return prediction, model_id
    
    def log_result(
        self,
        user_id: str,
        model_id: str,
        prediction: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log test result
        
        Args:
            user_id: User identifier
            model_id: Model identifier
            prediction: Prediction
            ground_truth: Optional ground truth
            metadata: Optional metadata
        """
        result = {
            "user_id": user_id,
            "model_id": model_id,
            "prediction": prediction.item() if prediction.numel() == 1 else prediction.tolist(),
            "ground_truth": ground_truth.item() if ground_truth is not None and ground_truth.numel() == 1 else None,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        if model_id == "A":
            self.results_a.append(result)
        else:
            self.results_b.append(result)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get A/B test statistics
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "model_a": {
                "count": len(self.results_a),
                "results": self.results_a
            },
            "model_b": {
                "count": len(self.results_b),
                "results": self.results_b
            },
            "metadata": self.metadata
        }
        
        # Calculate metrics if ground truth available
        if self.results_a and self.results_a[0].get("ground_truth") is not None:
            # Calculate accuracy, MSE, etc.
            errors_a = [
                abs(r["prediction"] - r["ground_truth"])
                for r in self.results_a
                if r.get("ground_truth") is not None
            ]
            errors_b = [
                abs(r["prediction"] - r["ground_truth"])
                for r in self.results_b
                if r.get("ground_truth") is not None
            ]
            
            if errors_a:
                stats["model_a"]["mae"] = np.mean(errors_a)
                stats["model_a"]["mse"] = np.mean([e**2 for e in errors_a])
            
            if errors_b:
                stats["model_b"]["mae"] = np.mean(errors_b)
                stats["model_b"]["mse"] = np.mean([e**2 for e in errors_b])
        
        return stats
    
    def is_significant(
        self,
        metric: str = "mae",
        alpha: float = 0.05
    ) -> Tuple[bool, float]:
        """
        Check if difference is statistically significant
        
        Args:
            metric: Metric to compare
            alpha: Significance level
        
        Returns:
            Tuple of (is_significant, p_value)
        """
        from scipy import stats
        
        stats_dict = self.get_statistics()
        
        values_a = [r.get(metric) for r in stats_dict["model_a"]["results"]]
        values_b = [r.get(metric) for r in stats_dict["model_b"]["results"]]
        
        values_a = [v for v in values_a if v is not None]
        values_b = [v for v in values_b if v is not None]
        
        if len(values_a) < 2 or len(values_b) < 2:
            return False, 1.0
        
        # T-test
        t_stat, p_value = stats.ttest_ind(values_a, values_b)
        
        is_significant = p_value < alpha
        
        return is_significant, p_value


class MultiVariateTest:
    """Multi-variate test for multiple models"""
    
    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize multi-variate test
        
        Args:
            models: Dictionary of model_id -> model
            weights: Optional weights for each model
        """
        self.models = models
        self.weights = weights or {k: 1.0/len(models) for k in models.keys()}
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        self.results = defaultdict(list)
        
        logger.info(f"MultiVariateTest initialized with {len(models)} models")
    
    def assign_model(self, user_id: str) -> str:
        """Assign model to user based on weights"""
        hash_value = hash(user_id) % 100
        
        cumulative = 0
        for model_id, weight in self.weights.items():
            cumulative += weight * 100
            if hash_value < cumulative:
                return model_id
        
        return list(self.models.keys())[-1]
    
    def predict(self, user_id: str, inputs: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Get prediction from assigned model"""
        model_id = self.assign_model(user_id)
        model = self.models[model_id]
        
        model.eval()
        with torch.no_grad():
            prediction = model(inputs)
        
        return prediction, model_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get test statistics"""
        stats = {}
        for model_id in self.models.keys():
            model_results = self.results[model_id]
            stats[model_id] = {
                "count": len(model_results),
                "results": model_results
            }
        
        return stats


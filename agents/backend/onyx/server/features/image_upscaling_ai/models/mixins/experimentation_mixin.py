"""
Experimentation Mixin

Contains A/B testing and experimentation functionality.
"""

import logging
import random
import time
from typing import Union, Dict, Any, List, Optional, Tuple
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class ExperimentationMixin:
    """
    Mixin providing A/B testing and experimentation functionality.
    
    This mixin contains:
    - A/B testing
    - Method comparison
    - Experiment tracking
    - Statistical analysis
    - Winner selection
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize experimentation mixin."""
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_experiments'):
            self._experiments = {}
    
    def run_ab_test(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method_a: str,
        method_b: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run A/B test comparing two methods.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            method_a: First method to test
            method_b: Second method to test
            metrics: Optional list of metrics to compare
            
        Returns:
            Dictionary with A/B test results
        """
        if metrics is None:
            metrics = ["quality", "speed", "artifacts"]
        
        # Run method A
        start_a = time.time()
        result_a = self.upscale(image, scale_factor, method_a, return_metrics=True)
        time_a = time.time() - start_a
        
        # Run method B
        start_b = time.time()
        result_b = self.upscale(image, scale_factor, method_b, return_metrics=True)
        time_b = time.time() - start_b
        
        # Compare results
        comparison = {}
        
        if isinstance(result_a, tuple):
            image_a, metrics_a = result_a
        else:
            image_a = result_a
            metrics_a = {}
        
        if isinstance(result_b, tuple):
            image_b, metrics_b = result_b
        else:
            image_b = result_b
            metrics_b = {}
        
        # Quality comparison
        if "quality" in metrics:
            quality_a = metrics_a.get("quality_score", 0.5)
            quality_b = metrics_b.get("quality_score", 0.5)
            comparison["quality"] = {
                "method_a": quality_a,
                "method_b": quality_b,
                "winner": "a" if quality_a > quality_b else "b",
                "difference": abs(quality_a - quality_b),
            }
        
        # Speed comparison
        if "speed" in metrics:
            comparison["speed"] = {
                "method_a": time_a,
                "method_b": time_b,
                "winner": "a" if time_a < time_b else "b",
                "difference": abs(time_a - time_b),
            }
        
        # Overall winner
        winners = [comp.get("winner") for comp in comparison.values()]
        overall_winner = max(set(winners), key=winners.count) if winners else "tie"
        
        return {
            "method_a": method_a,
            "method_b": method_b,
            "result_a": image_a,
            "result_b": image_b,
            "metrics_a": metrics_a,
            "metrics_b": metrics_b,
            "time_a": time_a,
            "time_b": time_b,
            "comparison": comparison,
            "overall_winner": overall_winner,
        }
    
    def run_method_comparison(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        methods: List[str],
        criteria: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Compare multiple methods and rank them.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            methods: List of methods to compare
            criteria: Comparison criteria ('quality', 'speed', 'balanced')
            
        Returns:
            Dictionary with comparison results
        """
        results = []
        
        for method in methods:
            start = time.time()
            result = self.upscale(image, scale_factor, method, return_metrics=True)
            duration = time.time() - start
            
            if isinstance(result, tuple):
                upscaled_image, metrics = result
            else:
                upscaled_image = result
                metrics = {}
            
            quality_score = metrics.get("quality_score", 0.5)
            
            results.append({
                "method": method,
                "quality": quality_score,
                "speed": duration,
                "score": self._calculate_score(quality_score, duration, criteria),
                "result": upscaled_image,
                "metrics": metrics,
            })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "criteria": criteria,
            "methods": methods,
            "results": results,
            "winner": results[0]["method"] if results else None,
            "ranking": [r["method"] for r in results],
        }
    
    def _calculate_score(
        self,
        quality: float,
        speed: float,
        criteria: str
    ) -> float:
        """Calculate overall score based on criteria."""
        if criteria == "quality":
            return quality
        elif criteria == "speed":
            return 1.0 / (speed + 0.001)  # Inverse of speed
        else:  # balanced
            # Normalize and combine
            normalized_quality = quality
            normalized_speed = 1.0 / (speed + 0.001)
            return (normalized_quality * 0.6) + (normalized_speed * 0.4)
    
    def create_experiment(
        self,
        experiment_name: str,
        hypothesis: str,
        methods: List[str],
        test_images: List[Union[str, Path]]
    ) -> bool:
        """
        Create a new experiment.
        
        Args:
            experiment_name: Name of experiment
            hypothesis: Hypothesis to test
            methods: Methods to test
            test_images: Images to test on
            
        Returns:
            True if successful
        """
        if not hasattr(self, '_experiments'):
            self._experiments = {}
        
        self._experiments[experiment_name] = {
            "hypothesis": hypothesis,
            "methods": methods,
            "test_images": test_images,
            "results": [],
            "status": "created",
        }
        
        logger.info(f"Experiment '{experiment_name}' created")
        return True
    
    def run_experiment(
        self,
        experiment_name: str,
        scale_factor: float = 2.0,
        criteria: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Run an experiment.
        
        Args:
            experiment_name: Name of experiment
            scale_factor: Scale factor
            criteria: Comparison criteria
            
        Returns:
            Dictionary with experiment results
        """
        if not hasattr(self, '_experiments') or experiment_name not in self._experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        experiment = self._experiments[experiment_name]
        experiment["status"] = "running"
        
        all_results = []
        
        for image_path in experiment["test_images"]:
            comparison = self.run_method_comparison(
                image_path, scale_factor, experiment["methods"], criteria
            )
            all_results.append(comparison)
        
        # Aggregate results
        method_scores = {}
        for result in all_results:
            for method_result in result["results"]:
                method = method_result["method"]
                if method not in method_scores:
                    method_scores[method] = []
                method_scores[method].append(method_result["score"])
        
        # Calculate averages
        method_averages = {
            method: sum(scores) / len(scores)
            for method, scores in method_scores.items()
        }
        
        winner = max(method_averages.items(), key=lambda x: x[1])[0] if method_averages else None
        
        experiment["results"] = all_results
        experiment["method_averages"] = method_averages
        experiment["winner"] = winner
        experiment["status"] = "completed"
        
        return {
            "experiment": experiment_name,
            "hypothesis": experiment["hypothesis"],
            "winner": winner,
            "method_averages": method_averages,
            "results": all_results,
        }
    
    def list_experiments(self) -> List[str]:
        """List available experiments."""
        if not hasattr(self, '_experiments'):
            return []
        return list(self._experiments.keys())
    
    def get_experiment_info(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an experiment."""
        if not hasattr(self, '_experiments') or experiment_name not in self._experiments:
            return None
        return self._experiments[experiment_name]



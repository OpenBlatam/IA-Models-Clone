"""
A/B Testing Framework
Test different models and configurations
"""

from typing import Dict, Any, Optional, List
import logging
import time
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_name: str
    variant_a: Dict[str, Any]
    variant_b: Dict[str, Any]
    traffic_split: float = 0.5  # 0.5 = 50/50 split
    metric: str = "accuracy"
    min_samples: int = 100


class ABTestRunner:
    """
    A/B testing framework for models
    """
    
    def __init__(self):
        self.tests: Dict[str, ABTestConfig] = {}
        self.results: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"A": [], "B": []})
        self.active_tests: Dict[str, bool] = {}
    
    def register_test(self, config: ABTestConfig):
        """Register an A/B test"""
        self.tests[config.test_name] = config
        self.active_tests[config.test_name] = True
        logger.info(f"Registered A/B test: {config.test_name}")
    
    def run_test(
        self,
        test_name: str,
        input_data: Any,
        variant: str = "auto"  # "A", "B", or "auto"
    ) -> tuple:
        """Run A/B test and return result"""
        if test_name not in self.tests:
            raise ValueError(f"Test {test_name} not registered")
        
        config = self.tests[test_name]
        
        # Select variant
        if variant == "auto":
            import random
            variant = "A" if random.random() < config.traffic_split else "B"
        
        # Get model/config for variant
        variant_config = config.variant_a if variant == "A" else config.variant_b
        
        # Run inference
        start_time = time.time()
        result = self._run_inference(variant_config, input_data)
        latency = time.time() - start_time
        
        # Store result
        self.results[test_name][variant].append({
            "result": result,
            "latency": latency,
            "timestamp": time.time()
        })
        
        return result, variant
    
    def _run_inference(self, config: Dict[str, Any], input_data: Any) -> Any:
        """Run inference with given config"""
        # This would use the actual model from config
        # For now, return placeholder
        return {"prediction": "placeholder"}
    
    def get_test_results(self, test_name: str) -> Dict[str, Any]:
        """Get A/B test results"""
        if test_name not in self.tests:
            raise ValueError(f"Test {test_name} not registered")
        
        config = self.tests[test_name]
        results_a = self.results[test_name]["A"]
        results_b = self.results[test_name]["B"]
        
        # Calculate metrics
        def calculate_metric(results, metric_name):
            if metric_name == "accuracy":
                # Calculate accuracy from results
                return sum(1 for r in results if r.get("result", {}).get("correct", False)) / len(results) if results else 0
            elif metric_name == "latency":
                return np.mean([r["latency"] for r in results]) if results else 0
            else:
                return 0
        
        metric_a = calculate_metric(results_a, config.metric)
        metric_b = calculate_metric(results_b, config.metric)
        
        # Statistical significance (simplified)
        improvement = ((metric_b - metric_a) / metric_a * 100) if metric_a > 0 else 0
        
        return {
            "test_name": test_name,
            "variant_a": {
                "metric": metric_a,
                "samples": len(results_a)
            },
            "variant_b": {
                "metric": metric_b,
                "samples": len(results_b)
            },
            "improvement": improvement,
            "statistically_significant": len(results_a) >= config.min_samples and len(results_b) >= config.min_samples
        }
    
    def stop_test(self, test_name: str):
        """Stop an A/B test"""
        if test_name in self.active_tests:
            self.active_tests[test_name] = False
            logger.info(f"Stopped A/B test: {test_name}")


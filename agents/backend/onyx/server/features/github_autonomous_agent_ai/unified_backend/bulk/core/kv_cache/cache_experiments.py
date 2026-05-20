"""
Cache experimentation utilities.

Provides A/B testing and experimentation capabilities.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    """Experiment types."""
    STRATEGY = "strategy"
    COMPRESSION = "compression"
    QUANTIZATION = "quantization"
    SIZE = "size"


class CacheExperiment:
    """
    Cache experiment manager.
    
    Provides A/B testing and experimentation capabilities.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize cache experiment.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def create_experiment(
        self,
        experiment_name: str,
        experiment_type: ExperimentType,
        variants: List[Dict[str, Any]],
        duration: float = 3600.0
    ) -> Dict[str, Any]:
        """
        Create a new experiment.
        
        Args:
            experiment_name: Name of experiment
            experiment_type: Type of experiment
            variants: List of variant configurations
            duration: Experiment duration in seconds
            
        Returns:
            Experiment configuration
        """
        experiment = {
            "name": experiment_name,
            "type": experiment_type.value,
            "variants": variants,
            "duration": duration,
            "start_time": time.time(),
            "end_time": time.time() + duration,
            "active": True
        }
        
        self.experiments[experiment_name] = experiment
        logger.info(f"Created experiment: {experiment_name}")
        
        return experiment
    
    def run_experiment(
        self,
        experiment_name: str,
        test_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run experiment.
        
        Args:
            experiment_name: Name of experiment
            test_fn: Optional test function
            
        Returns:
            Experiment results
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        experiment = self.experiments[experiment_name]
        
        results = {
            "experiment_name": experiment_name,
            "start_time": time.time(),
            "variants": {}
        }
        
        # Run each variant
        for i, variant in enumerate(experiment["variants"]):
            variant_name = variant.get("name", f"variant_{i}")
            
            # Create cache with variant config
            from kv_cache import KVCacheConfig, BaseKVCache
            variant_config = KVCacheConfig(**variant.get("config", {}))
            variant_cache = BaseKVCache(variant_config)
            
            # Run test if provided
            if test_fn:
                variant_result = test_fn(variant_cache)
            else:
                # Default test: run benchmark
                from kv_cache import CacheBenchmark
                benchmark = CacheBenchmark(variant_cache)
                variant_result = benchmark.run_full_benchmark(num_operations=1000)
            
            results["variants"][variant_name] = {
                "config": variant.get("config", {}),
                "result": variant_result
            }
        
        results["end_time"] = time.time()
        results["duration"] = results["end_time"] - results["start_time"]
        
        self.results[experiment_name] = results
        
        return results
    
    def compare_experiments(
        self,
        experiment_name: str
    ) -> Dict[str, Any]:
        """
        Compare experiment variants.
        
        Args:
            experiment_name: Name of experiment
            
        Returns:
            Comparison results
        """
        if experiment_name not in self.results:
            raise ValueError(f"Experiment results {experiment_name} not found")
        
        results = self.results[experiment_name]
        
        comparison = {
            "experiment_name": experiment_name,
            "variants": {}
        }
        
        # Extract key metrics for comparison
        for variant_name, variant_data in results["variants"].items():
            result = variant_data.get("result", {})
            
            # Extract metrics
            if "forward" in result:
                forward = result["forward"]
                comparison["variants"][variant_name] = {
                    "hit_rate": forward.get("hit_rate", 0.0),
                    "throughput": forward.get("throughput_ops_per_sec", 0.0),
                    "avg_latency_ms": forward.get("avg_time_ms", 0.0)
                }
        
        # Find best variant
        if comparison["variants"]:
            best_variant = max(
                comparison["variants"].items(),
                key=lambda x: x[1].get("hit_rate", 0.0)
            )
            comparison["best_variant"] = best_variant[0]
        
        return comparison
    
    def list_experiments(self) -> List[str]:
        """
        List all experiments.
        
        Returns:
            List of experiment names
        """
        return list(self.experiments.keys())


class CacheABTesting:
    """
    A/B testing for cache configurations.
    
    Provides A/B testing capabilities.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize A/B testing.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.test_groups: Dict[str, Any] = {}
    
    def create_test_group(
        self,
        group_name: str,
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
        traffic_split: float = 0.5
    ) -> Dict[str, Any]:
        """
        Create A/B test group.
        
        Args:
            group_name: Name of test group
            config_a: Configuration A
            config_b: Configuration B
            traffic_split: Traffic split (0.0 to 1.0 for A)
            
        Returns:
            Test group configuration
        """
        from kv_cache import KVCacheConfig, BaseKVCache
        
        cache_a = BaseKVCache(KVCacheConfig(**config_a))
        cache_b = BaseKVCache(KVCacheConfig(**config_b))
        
        test_group = {
            "name": group_name,
            "cache_a": cache_a,
            "cache_b": cache_b,
            "config_a": config_a,
            "config_b": config_b,
            "traffic_split": traffic_split,
            "stats_a": {"hits": 0, "misses": 0},
            "stats_b": {"hits": 0, "misses": 0}
        }
        
        self.test_groups[group_name] = test_group
        
        return test_group
    
    def get_cache_for_request(
        self,
        group_name: str,
        request_id: Optional[str] = None
    ) -> Any:
        """
        Get cache instance for request (A or B).
        
        Args:
            group_name: Test group name
            request_id: Optional request ID for consistent routing
            
        Returns:
            Cache instance (A or B)
        """
        if group_name not in self.test_groups:
            return self.cache
        
        test_group = self.test_groups[group_name]
        
        # Route based on traffic split
        import random
        if request_id:
            # Consistent routing based on request ID
            hash_value = hash(request_id) % 100
            use_a = hash_value < (test_group["traffic_split"] * 100)
        else:
            use_a = random.random() < test_group["traffic_split"]
        
        cache = test_group["cache_a"] if use_a else test_group["cache_b"]
        
        # Track stats
        if use_a:
            test_group["stats_a"]["hits"] += 1
        else:
            test_group["stats_b"]["hits"] += 1
        
        return cache
    
    def get_test_results(self, group_name: str) -> Dict[str, Any]:
        """
        Get A/B test results.
        
        Args:
            group_name: Test group name
            
        Returns:
            Test results
        """
        if group_name not in self.test_groups:
            return {}
        
        test_group = self.test_groups[group_name]
        
        stats_a = test_group["stats_a"]
        stats_b = test_group["stats_b"]
        
        total_a = stats_a["hits"] + stats_a["misses"]
        total_b = stats_b["hits"] + stats_b["misses"]
        
        hit_rate_a = stats_a["hits"] / total_a if total_a > 0 else 0.0
        hit_rate_b = stats_b["hits"] / total_b if total_b > 0 else 0.0
        
        return {
            "group_name": group_name,
            "variant_a": {
                "config": test_group["config_a"],
                "stats": stats_a,
                "hit_rate": hit_rate_a,
                "total_requests": total_a
            },
            "variant_b": {
                "config": test_group["config_b"],
                "stats": stats_b,
                "hit_rate": hit_rate_b,
                "total_requests": total_b
            },
            "winner": "A" if hit_rate_a > hit_rate_b else "B" if hit_rate_b > hit_rate_a else "TIE"
        }


"""
Comparator Module

Compare and analyze different configurations.
"""

from typing import Dict, Any, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class ConfigComparator:
    """
    Compare and analyze generator configurations.
    """
    
    @staticmethod
    def compare(
        config1: Dict[str, Any],
        config2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two configurations.
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Comparison results
        """
        all_keys = set(config1.keys()) | set(config2.keys())
        
        comparison = {
            "identical": True,
            "differences": {},
            "only_in_config1": {},
            "only_in_config2": {},
            "similarity_score": 0.0
        }
        
        differences = 0
        total_keys = len(all_keys)
        
        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)
            
            if key not in config1:
                comparison["only_in_config2"][key] = val2
                differences += 1
                comparison["identical"] = False
            elif key not in config2:
                comparison["only_in_config1"][key] = val1
                differences += 1
                comparison["identical"] = False
            elif val1 != val2:
                comparison["differences"][key] = {
                    "config1": val1,
                    "config2": val2
                }
                differences += 1
                comparison["identical"] = False
        
        if total_keys > 0:
            comparison["similarity_score"] = (total_keys - differences) / total_keys
        
        return comparison
    
    @staticmethod
    def find_differences(
        configs: List[Dict[str, Any]],
        keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Find differences across multiple configurations.
        
        Args:
            configs: List of configurations
            keys: Specific keys to compare (None for all)
            
        Returns:
            Differences analysis
        """
        if not configs:
            return {"error": "No configurations provided"}
        
        all_keys = set()
        for config in configs:
            all_keys.update(config.keys())
        
        if keys:
            all_keys = all_keys & set(keys)
        
        differences = {
            "varying_keys": [],
            "constant_keys": [],
            "key_values": {}
        }
        
        for key in all_keys:
            values = [config.get(key) for config in configs]
            unique_values = set(values)
            
            if len(unique_values) > 1:
                differences["varying_keys"].append(key)
                differences["key_values"][key] = {
                    "values": list(unique_values),
                    "count": len(unique_values)
                }
            else:
                differences["constant_keys"].append(key)
                differences["key_values"][key] = {
                    "value": values[0],
                    "constant": True
                }
        
        return differences
    
    @staticmethod
    def merge_configs(
        configs: List[Dict[str, Any]],
        strategy: str = "override"
    ) -> Dict[str, Any]:
        """
        Merge multiple configurations.
        
        Args:
            configs: List of configurations to merge
            strategy: Merge strategy (override, first, last, most_common)
            
        Returns:
            Merged configuration
        """
        if not configs:
            return {}
        
        if strategy == "first":
            return configs[0].copy()
        elif strategy == "last":
            return configs[-1].copy()
        elif strategy == "override":
            merged = {}
            for config in configs:
                merged.update(config)
            return merged
        elif strategy == "most_common":
            # Find most common value for each key
            merged = {}
            all_keys = set()
            for config in configs:
                all_keys.update(config.keys())
            
            for key in all_keys:
                values = [config.get(key) for config in configs if key in config]
                if values:
                    # Get most common value
                    from collections import Counter
                    counter = Counter(values)
                    merged[key] = counter.most_common(1)[0][0]
            
            return merged
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
    
    @staticmethod
    def get_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary of a configuration.
        
        Args:
            config: Configuration to summarize
            
        Returns:
            Summary dictionary
        """
        summary = {
            "total_keys": len(config),
            "keys": list(config.keys()),
            "has_framework": "framework" in config,
            "has_model_type": "model_type" in config,
            "has_training_params": any(
                k in config for k in ["batch_size", "learning_rate", "num_epochs"]
            ),
            "has_optimization": any(
                k in config for k in ["mixed_precision", "gradient_clipping"]
            )
        }
        
        if "framework" in config:
            summary["framework"] = config["framework"]
        if "model_type" in config:
            summary["model_type"] = config["model_type"]
        if "batch_size" in config:
            summary["batch_size"] = config["batch_size"]
        if "learning_rate" in config:
            summary["learning_rate"] = config["learning_rate"]
        
        return summary


def compare_configs(
    config1: Dict[str, Any],
    config2: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare two configurations."""
    return ConfigComparator.compare(config1, config2)


def merge_configs(
    configs: List[Dict[str, Any]],
    strategy: str = "override"
) -> Dict[str, Any]:
    """Merge multiple configurations."""
    return ConfigComparator.merge_configs(configs, strategy)
















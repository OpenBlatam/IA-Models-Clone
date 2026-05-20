"""
Advanced cache validation.

Provides comprehensive validation and integrity checks.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Optional
import torch

from kv_cache.types import TensorPair

logger = logging.getLogger(__name__)


class AdvancedCacheValidator:
    """
    Advanced cache validator.
    
    Provides comprehensive validation and integrity checks.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize advanced validator.
        
        Args:
            cache: Cache instance to validate
        """
        self.cache = cache
        self.validation_history: List[Dict[str, Any]] = []
    
    def validate_integrity(self) -> Dict[str, Any]:
        """
        Validate cache integrity.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            "timestamp": time.time(),
            "passed": True,
            "issues": [],
            "warnings": []
        }
        
        stats = self.cache.get_stats()
        storage = self.cache.storage
        
        # Check storage consistency
        positions = storage.get_positions()
        num_entries = storage.size()
        
        if len(positions) != num_entries:
            validation["passed"] = False
            validation["issues"].append({
                "type": "size_mismatch",
                "message": f"Position count ({len(positions)}) != size ({num_entries})"
            })
        
        # Check for duplicate positions
        if len(positions) != len(set(positions)):
            validation["passed"] = False
            validation["issues"].append({
                "type": "duplicate_positions",
                "message": "Duplicate positions found"
            })
        
        # Check memory consistency
        calculated_memory = storage.get_total_memory_mb()
        reported_memory = stats.get("storage_memory_mb", 0.0)
        
        if abs(calculated_memory - reported_memory) > 0.1:  # Allow 0.1 MB difference
            validation["warnings"].append({
                "type": "memory_mismatch",
                "message": f"Calculated ({calculated_memory:.2f}) != reported ({reported_memory:.2f})"
            })
        
        # Check tensor validity
        invalid_tensors = []
        for pos in positions[:100]:  # Sample first 100
            entry = storage.get(pos)
            if entry:
                key, value = entry
                if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                    invalid_tensors.append(pos)
                elif torch.isnan(key).any() or torch.isnan(value).any():
                    invalid_tensors.append(pos)
                elif torch.isinf(key).any() or torch.isinf(value).any():
                    invalid_tensors.append(pos)
        
        if invalid_tensors:
            validation["warnings"].append({
                "type": "invalid_tensors",
                "message": f"Found {len(invalid_tensors)} invalid tensors",
                "positions": invalid_tensors[:10]  # First 10
            })
        
        # Check stats consistency
        total_ops = stats.get("hits", 0) + stats.get("misses", 0)
        if total_ops > 0:
            calculated_hit_rate = stats.get("hits", 0) / total_ops
            reported_hit_rate = stats.get("hit_rate", 0.0)
            
            if abs(calculated_hit_rate - reported_hit_rate) > 0.01:
                validation["warnings"].append({
                    "type": "hit_rate_mismatch",
                    "message": f"Calculated ({calculated_hit_rate:.4f}) != reported ({reported_hit_rate:.4f})"
                })
        
        self.validation_history.append(validation)
        return validation
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate cache configuration.
        
        Returns:
            Dictionary with configuration validation results
        """
        validation = {
            "timestamp": time.time(),
            "passed": True,
            "issues": [],
            "warnings": []
        }
        
        config = self.cache.config
        
        # Check max_tokens
        if config.max_tokens <= 0:
            validation["passed"] = False
            validation["issues"].append({
                "type": "invalid_max_tokens",
                "message": f"max_tokens must be > 0, got {config.max_tokens}"
            })
        
        # Check num_heads
        if config.num_heads <= 0:
            validation["passed"] = False
            validation["issues"].append({
                "type": "invalid_num_heads",
                "message": f"num_heads must be > 0, got {config.num_heads}"
            })
        
        # Check head_dim
        if config.head_dim <= 0:
            validation["passed"] = False
            validation["issues"].append({
                "type": "invalid_head_dim",
                "message": f"head_dim must be > 0, got {config.head_dim}"
            })
        
        # Check compression ratio
        if config.use_compression:
            if config.compression_ratio <= 0 or config.compression_ratio > 1:
                validation["warnings"].append({
                    "type": "invalid_compression_ratio",
                    "message": f"compression_ratio should be in (0, 1], got {config.compression_ratio}"
                })
        
        # Check quantization bits
        if config.use_quantization:
            if config.quantization_bits not in [4, 8]:
                validation["warnings"].append({
                    "type": "unusual_quantization_bits",
                    "message": f"quantization_bits should be 4 or 8, got {config.quantization_bits}"
                })
        
        return validation
    
    def validate_operations(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        Validate cache operations.
        
        Args:
            num_samples: Number of operations to test
            
        Returns:
            Dictionary with operation validation results
        """
        validation = {
            "timestamp": time.time(),
            "passed": True,
            "issues": [],
            "operations_tested": num_samples
        }
        
        import random
        
        # Test put operations
        for i in range(num_samples):
            try:
                key = torch.randn(32, 128, device=self.cache.device)
                value = torch.randn(32, 128, device=self.cache.device)
                self.cache.put(i, key, value)
            except Exception as e:
                validation["passed"] = False
                validation["issues"].append({
                    "type": "put_failed",
                    "position": i,
                    "error": str(e)
                })
        
        # Test get operations
        for i in range(min(num_samples, self.cache.storage.size())):
            try:
                result = self.cache.get(i)
                if result is not None:
                    key, value = result
                    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                        validation["passed"] = False
                        validation["issues"].append({
                            "type": "get_invalid_type",
                            "position": i
                        })
            except Exception as e:
                validation["passed"] = False
                validation["issues"].append({
                    "type": "get_failed",
                    "position": i,
                    "error": str(e)
                })
        
        return validation
    
    def run_full_validation(self) -> Dict[str, Any]:
        """
        Run full validation suite.
        
        Returns:
            Dictionary with all validation results
        """
        logger.info("Running full validation suite...")
        
        results = {
            "timestamp": time.time(),
            "integrity": self.validate_integrity(),
            "configuration": self.validate_configuration(),
            "operations": self.validate_operations()
        }
        
        results["overall_passed"] = all(
            r["passed"] for r in [
                results["integrity"],
                results["configuration"],
                results["operations"]
            ]
        )
        
        return results


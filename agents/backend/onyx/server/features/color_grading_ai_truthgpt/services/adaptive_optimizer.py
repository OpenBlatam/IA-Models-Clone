"""
Adaptive Optimizer for Color Grading AI
========================================

ML-based adaptive optimization that learns from usage patterns.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class OptimizationPattern:
    """Optimization pattern learned from usage."""
    pattern_id: str
    input_characteristics: Dict[str, Any]
    optimal_params: Dict[str, Any]
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None


class AdaptiveOptimizer:
    """
    Adaptive optimizer that learns from usage.
    
    Features:
    - Pattern learning
    - Automatic optimization
    - Usage-based adaptation
    - Performance tracking
    - Pattern matching
    """
    
    def __init__(self):
        """Initialize adaptive optimizer."""
        self._patterns: Dict[str, OptimizationPattern] = {}
        self._usage_history: List[Dict[str, Any]] = []
        self._max_history = 10000
        self._learning_rate = 0.1
    
    def learn_from_usage(
        self,
        input_characteristics: Dict[str, Any],
        used_params: Dict[str, Any],
        success: bool,
        quality_score: Optional[float] = None
    ):
        """
        Learn from usage pattern.
        
        Args:
            input_characteristics: Input media characteristics
            used_params: Parameters that were used
            success: Whether operation was successful
            quality_score: Optional quality score
        """
        # Find or create pattern
        pattern_id = self._find_or_create_pattern(input_characteristics)
        pattern = self._patterns[pattern_id]
        
        # Update pattern
        pattern.usage_count += 1
        pattern.last_used = datetime.now()
        
        if success:
            # Update success rate
            pattern.success_rate = (
                (pattern.success_rate * (pattern.usage_count - 1) + 1.0) / pattern.usage_count
            )
            
            # Update optimal params (weighted average)
            if quality_score:
                weight = quality_score / 10.0  # Normalize to 0-1
            else:
                weight = 1.0
            
            # Merge params with learning rate
            for key, value in used_params.items():
                if key in pattern.optimal_params:
                    current = pattern.optimal_params[key]
                    if isinstance(current, (int, float)) and isinstance(value, (int, float)):
                        pattern.optimal_params[key] = (
                            current * (1 - self._learning_rate * weight) +
                            value * (self._learning_rate * weight)
                        )
                else:
                    pattern.optimal_params[key] = value
        else:
            # Decrease success rate
            pattern.success_rate = (
                (pattern.success_rate * (pattern.usage_count - 1)) / pattern.usage_count
            )
        
        # Record in history
        self._record_usage(input_characteristics, used_params, success, quality_score)
    
    def optimize_params(
        self,
        input_characteristics: Dict[str, Any],
        base_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize parameters based on learned patterns.
        
        Args:
            input_characteristics: Input media characteristics
            base_params: Optional base parameters
            
        Returns:
            Optimized parameters
        """
        # Find matching pattern
        pattern = self._find_best_pattern(input_characteristics)
        
        if pattern:
            # Use learned optimal params
            optimized = pattern.optimal_params.copy()
            
            # Merge with base params if provided
            if base_params:
                for key, value in base_params.items():
                    if key not in optimized:
                        optimized[key] = value
                    else:
                        # Weighted merge
                        optimized[key] = (
                            optimized[key] * 0.7 + value * 0.3
                        )
            
            logger.info(f"Optimized params using pattern {pattern.pattern_id}")
            return optimized
        
        # No pattern found, return base or defaults
        return base_params or self._get_default_params(input_characteristics)
    
    def _find_or_create_pattern(self, characteristics: Dict[str, Any]) -> str:
        """Find or create optimization pattern."""
        # Generate pattern ID from characteristics
        import hashlib
        import json
        char_str = json.dumps(characteristics, sort_keys=True)
        pattern_id = hashlib.md5(char_str.encode()).hexdigest()[:8]
        
        if pattern_id not in self._patterns:
            self._patterns[pattern_id] = OptimizationPattern(
                pattern_id=pattern_id,
                input_characteristics=characteristics,
                optimal_params={}
            )
            logger.info(f"Created new optimization pattern: {pattern_id}")
        
        return pattern_id
    
    def _find_best_pattern(self, characteristics: Dict[str, Any]) -> Optional[OptimizationPattern]:
        """Find best matching pattern."""
        if not self._patterns:
            return None
        
        # Simple matching (in production, use more sophisticated matching)
        best_pattern = None
        best_score = 0.0
        
        for pattern in self._patterns.values():
            score = self._calculate_similarity(characteristics, pattern.input_characteristics)
            # Weight by success rate
            weighted_score = score * pattern.success_rate
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_pattern = pattern
        
        # Only return if similarity is high enough
        if best_score > 0.5:
            return best_pattern
        
        return None
    
    def _calculate_similarity(
        self,
        char1: Dict[str, Any],
        char2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between characteristics."""
        # Simple similarity calculation
        common_keys = set(char1.keys()) & set(char2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1 = char1[key]
            val2 = char2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalize difference
                max_val = max(abs(val1), abs(val2), 1.0)
                diff = abs(val1 - val2) / max_val
                similarity = 1.0 - min(diff, 1.0)
            elif val1 == val2:
                similarity = 1.0
            else:
                similarity = 0.0
            
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _get_default_params(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Get default parameters based on characteristics."""
        # Simple defaults
        return {
            "brightness": 0.0,
            "contrast": 1.0,
            "saturation": 1.0,
        }
    
    def _record_usage(
        self,
        characteristics: Dict[str, Any],
        params: Dict[str, Any],
        success: bool,
        quality_score: Optional[float]
    ):
        """Record usage in history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "characteristics": characteristics,
            "params": params,
            "success": success,
            "quality_score": quality_score,
        }
        
        self._usage_history.append(entry)
        if len(self._usage_history) > self._max_history:
            self._usage_history = self._usage_history[-self._max_history:]
    
    def get_patterns(self) -> List[Dict[str, Any]]:
        """Get all optimization patterns."""
        return [
            {
                "pattern_id": p.pattern_id,
                "usage_count": p.usage_count,
                "success_rate": p.success_rate,
                "last_used": p.last_used.isoformat() if p.last_used else None,
            }
            for p in self._patterns.values()
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "patterns_count": len(self._patterns),
            "history_count": len(self._usage_history),
            "avg_success_rate": (
                sum(p.success_rate for p in self._patterns.values()) / len(self._patterns)
                if self._patterns else 0.0
            ),
        }





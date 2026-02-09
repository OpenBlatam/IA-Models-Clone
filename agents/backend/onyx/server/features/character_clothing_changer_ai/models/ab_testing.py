"""
A/B Testing for Flux2 Clothing Changer
=======================================

A/B testing and experimentation framework.
"""

import time
import random
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class Variant(Enum):
    """Test variant."""
    CONTROL = "control"
    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"
    VARIANT_C = "variant_c"


@dataclass
class ABTest:
    """A/B test configuration."""
    test_id: str
    name: str
    variants: Dict[Variant, Dict[str, Any]]
    traffic_split: Dict[Variant, float]
    start_date: float
    end_date: Optional[float] = None
    enabled: bool = True
    target_metric: str = "conversion_rate"
    min_sample_size: int = 100


@dataclass
class TestResult:
    """A/B test result."""
    test_id: str
    variant: Variant
    metric_value: float
    sample_size: int
    confidence: float
    is_significant: bool


class ABTesting:
    """A/B testing system."""
    
    def __init__(self):
        """Initialize A/B testing."""
        self.tests: Dict[str, ABTest] = {}
        self.assignments: Dict[str, Dict[str, Variant]] = defaultdict(dict)
        self.results: Dict[str, Dict[Variant, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.statistics: Dict[str, Dict[str, Any]] = {}
    
    def create_test(
        self,
        test_id: str,
        name: str,
        variants: Dict[Variant, Dict[str, Any]],
        traffic_split: Dict[Variant, float],
        target_metric: str = "conversion_rate",
        min_sample_size: int = 100,
    ) -> ABTest:
        """
        Create A/B test.
        
        Args:
            test_id: Test identifier
            name: Test name
            variants: Test variants
            traffic_split: Traffic split per variant
            target_metric: Target metric
            min_sample_size: Minimum sample size
            
        Returns:
            Created test
        """
        # Validate traffic split
        total = sum(traffic_split.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Traffic split must sum to 1.0, got {total}")
        
        test = ABTest(
            test_id=test_id,
            name=name,
            variants=variants,
            traffic_split=traffic_split,
            start_date=time.time(),
            target_metric=target_metric,
            min_sample_size=min_sample_size,
        )
        
        self.tests[test_id] = test
        logger.info(f"Created A/B test: {test_id}")
        return test
    
    def assign_variant(
        self,
        test_id: str,
        user_id: str,
    ) -> Variant:
        """
        Assign variant to user.
        
        Args:
            test_id: Test identifier
            user_id: User identifier
            
        Returns:
            Assigned variant
        """
        if test_id not in self.tests:
            return Variant.CONTROL
        
        test = self.tests[test_id]
        
        if not test.enabled:
            return Variant.CONTROL
        
        # Check if already assigned
        if user_id in self.assignments[test_id]:
            return self.assignments[test_id][user_id]
        
        # Consistent assignment based on user_id
        variant = self._consistent_assignment(test_id, user_id, test.traffic_split)
        
        self.assignments[test_id][user_id] = variant
        return variant
    
    def record_conversion(
        self,
        test_id: str,
        user_id: str,
        metric_value: float,
    ) -> None:
        """
        Record conversion metric.
        
        Args:
            test_id: Test identifier
            user_id: User identifier
            metric_value: Metric value
        """
        if test_id not in self.assignments or user_id not in self.assignments[test_id]:
            return
        
        variant = self.assignments[test_id][user_id]
        self.results[test_id][variant].append(metric_value)
    
    def get_results(
        self,
        test_id: str,
    ) -> Dict[Variant, TestResult]:
        """
        Get test results.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Test results per variant
        """
        if test_id not in self.tests:
            return {}
        
        test = self.tests[test_id]
        results = {}
        
        for variant in test.variants.keys():
            values = self.results[test_id][variant]
            
            if not values:
                continue
            
            metric_value = sum(values) / len(values)
            sample_size = len(values)
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(test_id, variant)
            is_significant = confidence > 0.95 and sample_size >= test.min_sample_size
            
            results[variant] = TestResult(
                test_id=test_id,
                variant=variant,
                metric_value=metric_value,
                sample_size=sample_size,
                confidence=confidence,
                is_significant=is_significant,
            )
        
        return results
    
    def _consistent_assignment(
        self,
        test_id: str,
        user_id: str,
        traffic_split: Dict[Variant, float],
    ) -> Variant:
        """Consistent variant assignment."""
        # Use hash for consistent assignment
        hash_value = int(hashlib.md5(f"{test_id}:{user_id}".encode()).hexdigest(), 16)
        random.seed(hash_value)
        
        rand = random.random()
        cumulative = 0.0
        
        for variant, split in sorted(traffic_split.items(), key=lambda x: x[0].value):
            cumulative += split
            if rand <= cumulative:
                return variant
        
        return Variant.CONTROL
    
    def _calculate_confidence(
        self,
        test_id: str,
        variant: Variant,
    ) -> float:
        """Calculate statistical confidence."""
        # Simplified confidence calculation
        values = self.results[test_id][variant]
        control_values = self.results[test_id].get(Variant.CONTROL, [])
        
        if not values or not control_values:
            return 0.0
        
        variant_mean = sum(values) / len(values)
        control_mean = sum(control_values) / len(control_values)
        
        if control_mean == 0:
            return 0.0
        
        # Simple confidence based on difference
        improvement = (variant_mean - control_mean) / control_mean
        confidence = min(1.0, abs(improvement) * 2)
        
        return confidence
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get A/B testing statistics."""
        return {
            "total_tests": len(self.tests),
            "active_tests": len([t for t in self.tests.values() if t.enabled]),
            "total_assignments": sum(len(assignments) for assignments in self.assignments.values()),
        }



"""
A/B Testing Service
Manages A/B tests for video generation
"""

from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime
import logging
import random

logger = logging.getLogger(__name__)


class ABTest:
    """Represents an A/B test"""
    
    def __init__(
        self,
        test_id: str,
        name: str,
        variants: List[Dict[str, Any]],
        metrics: List[str],
        created_at: datetime
    ):
        self.test_id = test_id
        self.name = name
        self.variants = variants
        self.metrics = metrics
        self.created_at = created_at
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "variants": self.variants,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "results": self.results,
        }


class ABTestingService:
    """Manages A/B tests"""
    
    def __init__(self):
        # In-memory storage (use database in production)
        self.tests: Dict[str, ABTest] = {}
        self.assignments: Dict[UUID, str] = {}  # video_id -> variant
    
    def create_test(
        self,
        name: str,
        variants: List[Dict[str, Any]],
        metrics: List[str] = None
    ) -> ABTest:
        """
        Create new A/B test
        
        Args:
            name: Test name
            variants: List of variant configurations
            metrics: Metrics to track
            
        Returns:
            Created test
        """
        test_id = f"test_{len(self.tests) + 1}"
        
        test = ABTest(
            test_id=test_id,
            name=name,
            variants=variants,
            metrics=metrics or ["completion_rate", "quality_score", "engagement"],
            created_at=datetime.utcnow()
        )
        
        self.tests[test_id] = test
        logger.info(f"Created A/B test: {test_id}")
        
        return test
    
    def assign_variant(self, test_id: str, video_id: UUID) -> Dict[str, Any]:
        """
        Assign variant to video (random assignment)
        
        Args:
            test_id: Test ID
            video_id: Video ID
            
        Returns:
            Assigned variant configuration
        """
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")
        
        # Random assignment
        variant = random.choice(test.variants)
        self.assignments[video_id] = variant.get("name", "variant_a")
        
        logger.debug(f"Assigned variant {variant.get('name')} to video {video_id}")
        return variant
    
    def record_result(
        self,
        test_id: str,
        video_id: UUID,
        metrics: Dict[str, Any]
    ):
        """Record test results"""
        test = self.tests.get(test_id)
        if not test:
            return
        
        variant_name = self.assignments.get(video_id, "unknown")
        
        if variant_name not in test.results:
            test.results[variant_name] = {
                "count": 0,
                "metrics": {metric: [] for metric in test.metrics},
            }
        
        test.results[variant_name]["count"] += 1
        
        for metric, value in metrics.items():
            if metric in test.results[variant_name]["metrics"]:
                test.results[variant_name]["metrics"][metric].append(value)
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get test results with statistics"""
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")
        
        results = {}
        
        for variant_name, variant_data in test.results.items():
            metrics_stats = {}
            
            for metric, values in variant_data["metrics"].items():
                if values:
                    metrics_stats[metric] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                    }
            
            results[variant_name] = {
                "count": variant_data["count"],
                "metrics": metrics_stats,
            }
        
        return {
            "test": test.to_dict(),
            "results": results,
        }
    
    def get_winner(self, test_id: str, metric: str = "completion_rate") -> Optional[str]:
        """Get winning variant based on metric"""
        test = self.tests.get(test_id)
        if not test:
            return None
        
        best_variant = None
        best_score = -1
        
        for variant_name, variant_data in test.results.items():
            metric_values = variant_data["metrics"].get(metric, [])
            if metric_values:
                avg_score = sum(metric_values) / len(metric_values)
                if avg_score > best_score:
                    best_score = avg_score
                    best_variant = variant_name
        
        return best_variant


_ab_testing_service: Optional[ABTestingService] = None


def get_ab_testing_service() -> ABTestingService:
    """Get A/B testing service instance (singleton)"""
    global _ab_testing_service
    if _ab_testing_service is None:
        _ab_testing_service = ABTestingService()
    return _ab_testing_service


"""
Analytics and Performance Monitoring System
==========================================

This module provides comprehensive analytics, performance monitoring,
and reporting capabilities for the test generation system.
"""

import time
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
import statistics
from collections import defaultdict, deque

from .base_architecture import TestCase, GenerationMetrics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for test generation"""
    timestamp: datetime = field(default_factory=datetime.now)
    generation_time: float = 0.0
    test_count: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    success_rate: float = 0.0


@dataclass
class QualityMetrics:
    """Quality metrics for generated tests"""
    timestamp: datetime = field(default_factory=datetime.now)
    average_test_length: float = 0.0
    assertion_density: float = 0.0
    coverage_estimate: float = 0.0
    complexity_score: float = 0.0
    naming_quality: float = 0.0
    documentation_quality: float = 0.0
    edge_case_coverage: float = 0.0


@dataclass
class UsageMetrics:
    """Usage metrics for the system"""
    timestamp: datetime = field(default_factory=datetime.now)
    total_requests: int = 0
    unique_functions: int = 0
    generator_usage: Dict[str, int] = field(default_factory=dict)
    pattern_usage: Dict[str, int] = field(default_factory=dict)
    preset_usage: Dict[str, int] = field(default_factory=dict)
    error_types: Dict[str, int] = field(default_factory=dict)


class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.performance_history: deque = deque(maxlen=max_history)
        self.quality_history: deque = deque(maxlen=max_history)
        self.usage_history: deque = deque(maxlen=max_history)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Real-time metrics
        self.current_metrics = {
            "active_generations": 0,
            "total_generations": 0,
            "total_errors": 0,
            "average_generation_time": 0.0,
            "cache_hit_rate": 0.0
        }
    
    def record_generation(
        self,
        generation_time: float,
        test_count: int,
        success: bool,
        generator_type: str,
        pattern_used: str,
        preset_used: str,
        error_type: Optional[str] = None
    ):
        """Record a test generation event"""
        
        # Record performance metrics
        perf_metrics = PerformanceMetrics(
            generation_time=generation_time,
            test_count=test_count,
            success_rate=1.0 if success else 0.0,
            error_count=1 if not success else 0
        )
        self.performance_history.append(perf_metrics)
        
        # Record usage metrics
        usage_metrics = UsageMetrics(
            total_requests=1,
            unique_functions=1,
            generator_usage={generator_type: 1},
            pattern_usage={pattern_used: 1},
            preset_usage={preset_used: 1}
        )
        if error_type:
            usage_metrics.error_types[error_type] = 1
        
        self.usage_history.append(usage_metrics)
        
        # Update real-time metrics
        self._update_real_time_metrics(generation_time, success)
    
    def record_quality_metrics(self, test_cases: List[TestCase]):
        """Record quality metrics for generated tests"""
        
        if not test_cases:
            return
        
        # Calculate quality metrics
        avg_length = statistics.mean(len(tc.test_code) for tc in test_cases)
        assertion_count = sum(tc.test_code.lower().count('assert') for tc in test_cases)
        assertion_density = assertion_count / len(test_cases) if test_cases else 0
        
        # Estimate coverage based on test types
        coverage_estimate = self._estimate_coverage(test_cases)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(test_cases)
        
        # Calculate naming quality
        naming_quality = self._calculate_naming_quality(test_cases)
        
        # Calculate documentation quality
        doc_quality = self._calculate_documentation_quality(test_cases)
        
        # Calculate edge case coverage
        edge_case_coverage = self._calculate_edge_case_coverage(test_cases)
        
        quality_metrics = QualityMetrics(
            average_test_length=avg_length,
            assertion_density=assertion_density,
            coverage_estimate=coverage_estimate,
            complexity_score=complexity_score,
            naming_quality=naming_quality,
            documentation_quality=doc_quality,
            edge_case_coverage=edge_case_coverage
        )
        
        self.quality_history.append(quality_metrics)
    
    def _update_real_time_metrics(self, generation_time: float, success: bool):
        """Update real-time metrics"""
        self.current_metrics["total_generations"] += 1
        if not success:
            self.current_metrics["total_errors"] += 1
        
        # Update average generation time
        total_time = sum(pm.generation_time for pm in self.performance_history)
        self.current_metrics["average_generation_time"] = total_time / len(self.performance_history)
        
        # Update cache hit rate
        total_hits = sum(pm.cache_hits for pm in self.performance_history)
        total_requests = sum(pm.cache_hits + pm.cache_misses for pm in self.performance_history)
        self.current_metrics["cache_hit_rate"] = total_hits / total_requests if total_requests > 0 else 0
    
    def _estimate_coverage(self, test_cases: List[TestCase]) -> float:
        """Estimate test coverage based on test types"""
        if not test_cases:
            return 0.0
        
        coverage_score = 0.0
        total_tests = len(test_cases)
        
        # Base coverage from test count
        coverage_score += min(total_tests * 0.1, 0.5)
        
        # Coverage from test categories
        category_coverage = {
            "functional": 0.3,
            "edge_case": 0.2,
            "performance": 0.1,
            "security": 0.1,
            "integration": 0.1
        }
        
        for test_case in test_cases:
            category = test_case.category.value if test_case.category else "functional"
            coverage_score += category_coverage.get(category, 0.1)
        
        return min(coverage_score, 1.0)
    
    def _calculate_complexity_score(self, test_cases: List[TestCase]) -> float:
        """Calculate complexity score for test cases"""
        if not test_cases:
            return 0.0
        
        complexity_scores = []
        for test_case in test_cases:
            score = 0.0
            
            # Code complexity
            code_lines = len(test_case.test_code.split('\n'))
            score += min(code_lines * 0.1, 0.5)
            
            # Setup/teardown complexity
            if test_case.setup_code:
                score += 0.2
            if test_case.teardown_code:
                score += 0.2
            
            # Test type complexity
            if test_case.test_type and test_case.test_type.value in ["integration", "performance"]:
                score += 0.3
            
            complexity_scores.append(score)
        
        return statistics.mean(complexity_scores)
    
    def _calculate_naming_quality(self, test_cases: List[TestCase]) -> float:
        """Calculate naming quality score"""
        if not test_cases:
            return 0.0
        
        quality_scores = []
        for test_case in test_cases:
            score = 0.0
            
            # Name length and descriptiveness
            if len(test_case.name) > 15:
                score += 0.3
            elif len(test_case.name) > 10:
                score += 0.2
            
            # Name format
            if test_case.name.startswith("test_"):
                score += 0.2
            
            # Descriptive words
            descriptive_words = ["test", "should", "when", "given", "then"]
            if any(word in test_case.name.lower() for word in descriptive_words):
                score += 0.3
            
            quality_scores.append(score)
        
        return statistics.mean(quality_scores)
    
    def _calculate_documentation_quality(self, test_cases: List[TestCase]) -> float:
        """Calculate documentation quality score"""
        if not test_cases:
            return 0.0
        
        quality_scores = []
        for test_case in test_cases:
            score = 0.0
            
            # Description presence and length
            if test_case.description:
                if len(test_case.description) > 50:
                    score += 0.5
                elif len(test_case.description) > 20:
                    score += 0.3
                else:
                    score += 0.1
            
            # Code comments
            if "#" in test_case.test_code:
                score += 0.2
            
            quality_scores.append(score)
        
        return statistics.mean(quality_scores)
    
    def _calculate_edge_case_coverage(self, test_cases: List[TestCase]) -> float:
        """Calculate edge case coverage"""
        if not test_cases:
            return 0.0
        
        edge_case_count = 0
        for test_case in test_cases:
            if (test_case.category and test_case.category.value == "edge_case") or \
               "edge" in test_case.name.lower() or \
               "boundary" in test_case.name.lower():
                edge_case_count += 1
        
        return edge_case_count / len(test_cases)
    
    def get_performance_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get performance summary for a time window"""
        
        if time_window:
            cutoff_time = datetime.now() - time_window
            recent_metrics = [m for m in self.performance_history if m.timestamp >= cutoff_time]
        else:
            recent_metrics = list(self.performance_history)
        
        if not recent_metrics:
            return {"error": "No metrics available for the specified time window"}
        
        # Calculate statistics
        generation_times = [m.generation_time for m in recent_metrics]
        test_counts = [m.test_count for m in recent_metrics]
        success_rates = [m.success_rate for m in recent_metrics]
        
        return {
            "time_window": str(time_window) if time_window else "all_time",
            "total_generations": len(recent_metrics),
            "average_generation_time": statistics.mean(generation_times),
            "median_generation_time": statistics.median(generation_times),
            "min_generation_time": min(generation_times),
            "max_generation_time": max(generation_times),
            "average_test_count": statistics.mean(test_counts),
            "total_tests_generated": sum(test_counts),
            "average_success_rate": statistics.mean(success_rates),
            "total_errors": sum(1 for m in recent_metrics if m.error_count > 0)
        }
    
    def get_quality_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get quality summary for a time window"""
        
        if time_window:
            cutoff_time = datetime.now() - time_window
            recent_metrics = [m for m in self.quality_history if m.timestamp >= cutoff_time]
        else:
            recent_metrics = list(self.quality_history)
        
        if not recent_metrics:
            return {"error": "No quality metrics available for the specified time window"}
        
        # Calculate statistics
        avg_lengths = [m.average_test_length for m in recent_metrics]
        assertion_densities = [m.assertion_density for m in recent_metrics]
        coverage_estimates = [m.coverage_estimate for m in recent_metrics]
        complexity_scores = [m.complexity_score for m in recent_metrics]
        naming_qualities = [m.naming_quality for m in recent_metrics]
        doc_qualities = [m.documentation_quality for m in recent_metrics]
        edge_case_coverages = [m.edge_case_coverage for m in recent_metrics]
        
        return {
            "time_window": str(time_window) if time_window else "all_time",
            "total_quality_measurements": len(recent_metrics),
            "average_test_length": statistics.mean(avg_lengths),
            "average_assertion_density": statistics.mean(assertion_densities),
            "average_coverage_estimate": statistics.mean(coverage_estimates),
            "average_complexity_score": statistics.mean(complexity_scores),
            "average_naming_quality": statistics.mean(naming_qualities),
            "average_documentation_quality": statistics.mean(doc_qualities),
            "average_edge_case_coverage": statistics.mean(edge_case_coverages)
        }
    
    def get_usage_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get usage summary for a time window"""
        
        if time_window:
            cutoff_time = datetime.now() - time_window
            recent_metrics = [m for m in self.usage_history if m.timestamp >= cutoff_time]
        else:
            recent_metrics = list(self.usage_history)
        
        if not recent_metrics:
            return {"error": "No usage metrics available for the specified time window"}
        
        # Aggregate usage data
        total_requests = sum(m.total_requests for m in recent_metrics)
        total_functions = sum(m.unique_functions for m in recent_metrics)
        
        # Aggregate generator usage
        generator_usage = defaultdict(int)
        for m in recent_metrics:
            for gen_type, count in m.generator_usage.items():
                generator_usage[gen_type] += count
        
        # Aggregate pattern usage
        pattern_usage = defaultdict(int)
        for m in recent_metrics:
            for pattern_type, count in m.pattern_usage.items():
                pattern_usage[pattern_type] += count
        
        # Aggregate preset usage
        preset_usage = defaultdict(int)
        for m in recent_metrics:
            for preset_type, count in m.preset_usage.items():
                preset_usage[preset_type] += count
        
        # Aggregate error types
        error_types = defaultdict(int)
        for m in recent_metrics:
            for error_type, count in m.error_types.items():
                error_types[error_type] += count
        
        return {
            "time_window": str(time_window) if time_window else "all_time",
            "total_requests": total_requests,
            "total_functions": total_functions,
            "generator_usage": dict(generator_usage),
            "pattern_usage": dict(pattern_usage),
            "preset_usage": dict(preset_usage),
            "error_types": dict(error_types)
        }
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        return self.current_metrics.copy()
    
    def export_metrics(self, output_path: str, format: str = "json"):
        """Export metrics to file"""
        try:
            metrics_data = {
                "performance_history": [asdict(m) for m in self.performance_history],
                "quality_history": [asdict(m) for m in self.quality_history],
                "usage_history": [asdict(m) for m in self.usage_history],
                "current_metrics": self.current_metrics,
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                if format.lower() == "json":
                    json.dump(metrics_data, f, indent=2, default=str)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Metrics exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False
    
    def clear_history(self):
        """Clear all metrics history"""
        self.performance_history.clear()
        self.quality_history.clear()
        self.usage_history.clear()
        self.logger.info("Metrics history cleared")


class AnalyticsDashboard:
    """Analytics dashboard for visualizing metrics"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data"""
        
        try:
            # Get summaries for different time windows
            performance_1h = self.monitor.get_performance_summary(timedelta(hours=1))
            performance_24h = self.monitor.get_performance_summary(timedelta(hours=24))
            performance_all = self.monitor.get_performance_summary()
            
            quality_1h = self.monitor.get_quality_summary(timedelta(hours=1))
            quality_24h = self.monitor.get_quality_summary(timedelta(hours=24))
            quality_all = self.monitor.get_quality_summary()
            
            usage_1h = self.monitor.get_usage_summary(timedelta(hours=1))
            usage_24h = self.monitor.get_usage_summary(timedelta(hours=24))
            usage_all = self.monitor.get_usage_summary()
            
            real_time = self.monitor.get_real_time_metrics()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "performance": {
                    "last_hour": performance_1h,
                    "last_24_hours": performance_24h,
                    "all_time": performance_all
                },
                "quality": {
                    "last_hour": quality_1h,
                    "last_24_hours": quality_24h,
                    "all_time": quality_all
                },
                "usage": {
                    "last_hour": usage_1h,
                    "last_24_hours": usage_24h,
                    "all_time": usage_all
                },
                "real_time": real_time,
                "recommendations": self._generate_recommendations(performance_all, quality_all, usage_all)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate dashboard data: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(
        self,
        performance: Dict[str, Any],
        quality: Dict[str, Any],
        usage: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on metrics"""
        
        recommendations = []
        
        # Performance recommendations
        if performance.get("average_generation_time", 0) > 5.0:
            recommendations.append("Consider optimizing test generation performance - average time is high")
        
        if performance.get("average_success_rate", 1.0) < 0.9:
            recommendations.append("Success rate is below 90% - investigate error patterns")
        
        # Quality recommendations
        if quality.get("average_coverage_estimate", 0) < 0.8:
            recommendations.append("Test coverage is below 80% - consider adding more comprehensive tests")
        
        if quality.get("average_naming_quality", 0) < 0.7:
            recommendations.append("Test naming quality could be improved - use more descriptive names")
        
        if quality.get("average_documentation_quality", 0) < 0.6:
            recommendations.append("Test documentation quality is low - add more detailed descriptions")
        
        # Usage recommendations
        if usage.get("total_requests", 0) > 1000:
            recommendations.append("High usage detected - consider implementing caching for better performance")
        
        # General recommendations
        recommendations.append("Regularly review and update test generation patterns")
        recommendations.append("Monitor system performance and scale resources as needed")
        
        return recommendations


# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Global analytics dashboard instance
analytics_dashboard = AnalyticsDashboard(performance_monitor)










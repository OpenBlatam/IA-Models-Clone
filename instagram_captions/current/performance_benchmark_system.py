#!/usr/bin/env python3
"""
Performance Benchmarking System
Comprehensive performance analysis and optimization recommendations
"""

import time
import gc
import psutil
import threading
import multiprocessing
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import logging

# Import from advanced error handling system
from advanced_error_handling_debugging_system import (
    DebugLevel,
    AdvancedDebugger,
    PerformanceProfiler,
    MemoryTracker,
    CPUTracker
)


class BenchmarkType(Enum):
    """Types of benchmarks"""
    FUNCTION = "function"
    MEMORY = "memory"
    CPU = "cpu"
    I_O = "i_o"
    NETWORK = "network"
    CONCURRENT = "concurrent"
    STRESS = "stress"
    COMPARISON = "comparison"


class BenchmarkResult(Enum):
    """Benchmark result categories"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class BenchmarkMetrics:
    """Benchmark performance metrics"""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    latency: float
    error_rate: float
    success_rate: float
    iterations: int
    concurrent_users: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    iterations: int = 100
    warmup_iterations: int = 10
    concurrent_users: int = 1
    timeout: float = 30.0
    memory_limit: float = 1024.0  # MB
    cpu_limit: float = 80.0  # Percentage
    error_threshold: float = 0.05  # 5%
    performance_threshold: float = 1.0  # seconds


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report"""
    benchmark_name: str
    benchmark_type: BenchmarkType
    config: BenchmarkConfig
    metrics: List[BenchmarkMetrics]
    summary: Dict[str, Any]
    recommendations: List[str]
    performance_score: float
    result_category: BenchmarkResult
    timestamp: float = field(default_factory=time.time)


class PerformanceBenchmarker:
    """Advanced performance benchmarking system"""
    
    def __init__(self, debug_level: DebugLevel = DebugLevel.DETAILED):
        """Initialize the performance benchmarker"""
        self.debug_level = debug_level
        self.debugger = AdvancedDebugger(debug_level)
        self.profiler = PerformanceProfiler()
        self.memory_tracker = MemoryTracker()
        self.cpu_tracker = CPUTracker()
        self.benchmark_history: List[BenchmarkReport] = []
        self.baseline_metrics: Dict[str, BenchmarkMetrics] = {}
        
        # Performance thresholds
        self.thresholds = {
            "excellent": {"time": 0.1, "memory": 50, "cpu": 20},
            "good": {"time": 0.5, "memory": 100, "cpu": 40},
            "average": {"time": 1.0, "memory": 200, "cpu": 60},
            "poor": {"time": 2.0, "memory": 500, "cpu": 80},
            "critical": {"time": float('inf'), "memory": float('inf'), "cpu": float('inf')}
        }
    
    @property
    def logger(self):
        """Get logger instance"""
        return logging.getLogger(__name__)
    
    def benchmark_function(self, 
                          func: Callable, 
                          args: tuple = (), 
                          kwargs: dict = None,
                          config: BenchmarkConfig = None) -> BenchmarkReport:
        """Benchmark a single function"""
        kwargs = kwargs or {}
        config = config or BenchmarkConfig()
        
        self.logger.info(f"Starting function benchmark: {func.__name__}")
        
        # Warmup phase
        self._warmup_function(func, args, kwargs, config.warmup_iterations)
        
        # Benchmark phase
        metrics = []
        errors = 0
        
        for i in range(config.iterations):
            try:
                metric = self._execute_single_benchmark(func, args, kwargs)
                metrics.append(metric)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i + 1}/{config.iterations} iterations")
                    
            except Exception as e:
                errors += 1
                self.logger.error(f"Error in iteration {i}: {e}")
        
        # Calculate summary
        summary = self._calculate_summary(metrics, errors, config)
        performance_score = self._calculate_performance_score(summary)
        result_category = self._categorize_performance(performance_score)
        recommendations = self._generate_recommendations(summary, result_category)
        
        # Create report
        report = BenchmarkReport(
            benchmark_name=func.__name__,
            benchmark_type=BenchmarkType.FUNCTION,
            config=config,
            metrics=metrics,
            summary=summary,
            recommendations=recommendations,
            performance_score=performance_score,
            result_category=result_category
        )
        
        self.benchmark_history.append(report)
        self.logger.info(f"Function benchmark completed: {func.__name__} - Score: {performance_score:.2f}")
        
        return report
    
    def benchmark_memory(self, 
                        func: Callable, 
                        args: tuple = (), 
                        kwargs: dict = None,
                        config: BenchmarkConfig = None) -> BenchmarkReport:
        """Benchmark memory usage of a function"""
        kwargs = kwargs or {}
        config = config or BenchmarkConfig()
        
        self.logger.info(f"Starting memory benchmark: {func.__name__}")
        
        # Force garbage collection before benchmark
        gc.collect()
        
        metrics = []
        initial_memory = self.memory_tracker.get_memory_usage()
        
        for i in range(config.iterations):
            try:
                # Record memory before
                memory_before = self.memory_tracker.get_memory_usage()
                
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record memory after
                memory_after = self.memory_tracker.get_memory_usage()
                
                # Calculate memory delta
                memory_delta = memory_after["total"] - memory_before["total"]
                
                metric = BenchmarkMetrics(
                    execution_time=execution_time,
                    memory_usage=memory_delta,
                    cpu_usage=self.cpu_tracker.get_cpu_usage(),
                    throughput=1.0 / execution_time if execution_time > 0 else 0,
                    latency=execution_time * 1000,  # Convert to milliseconds
                    error_rate=0.0,
                    success_rate=1.0,
                    iterations=1,
                    concurrent_users=1
                )
                
                metrics.append(metric)
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error in memory benchmark iteration {i}: {e}")
        
        # Calculate summary
        summary = self._calculate_summary(metrics, 0, config)
        performance_score = self._calculate_memory_score(summary)
        result_category = self._categorize_performance(performance_score)
        recommendations = self._generate_memory_recommendations(summary, result_category)
        
        report = BenchmarkReport(
            benchmark_name=f"{func.__name__}_memory",
            benchmark_type=BenchmarkType.MEMORY,
            config=config,
            metrics=metrics,
            summary=summary,
            recommendations=recommendations,
            performance_score=performance_score,
            result_category=result_category
        )
        
        self.benchmark_history.append(report)
        return report
    
    def benchmark_concurrent(self, 
                           func: Callable, 
                           args: tuple = (), 
                           kwargs: dict = None,
                           config: BenchmarkConfig = None) -> BenchmarkReport:
        """Benchmark function with concurrent execution"""
        kwargs = kwargs or {}
        config = config or BenchmarkConfig()
        
        self.logger.info(f"Starting concurrent benchmark: {func.__name__} with {config.concurrent_users} users")
        
        metrics = []
        
        def worker():
            """Worker function for concurrent execution"""
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                return execution_time, True, None
            except Exception as e:
                return 0, False, str(e)
        
        # Execute concurrent benchmarks
        with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            futures = [executor.submit(worker) for _ in range(config.iterations)]
            
            for future in futures:
                try:
                    execution_time, success, error = future.result(timeout=config.timeout)
                    
                    metric = BenchmarkMetrics(
                        execution_time=execution_time,
                        memory_usage=self.memory_tracker.get_memory_usage()["total"],
                        cpu_usage=self.cpu_tracker.get_cpu_usage(),
                        throughput=1.0 / execution_time if execution_time > 0 else 0,
                        latency=execution_time * 1000,
                        error_rate=0.0 if success else 1.0,
                        success_rate=1.0 if success else 0.0,
                        iterations=1,
                        concurrent_users=config.concurrent_users
                    )
                    
                    metrics.append(metric)
                    
                except Exception as e:
                    self.logger.error(f"Error in concurrent benchmark: {e}")
        
        # Calculate summary
        summary = self._calculate_summary(metrics, 0, config)
        performance_score = self._calculate_concurrent_score(summary)
        result_category = self._categorize_performance(performance_score)
        recommendations = self._generate_concurrent_recommendations(summary, result_category)
        
        report = BenchmarkReport(
            benchmark_name=f"{func.__name__}_concurrent",
            benchmark_type=BenchmarkType.CONCURRENT,
            config=config,
            metrics=metrics,
            summary=summary,
            recommendations=recommendations,
            performance_score=performance_score,
            result_category=result_category
        )
        
        self.benchmark_history.append(report)
        return report
    
    def benchmark_stress(self, 
                        func: Callable, 
                        args: tuple = (), 
                        kwargs: dict = None,
                        config: BenchmarkConfig = None) -> BenchmarkReport:
        """Stress test a function under high load"""
        kwargs = kwargs or {}
        config = config or BenchmarkConfig()
        
        # Increase iterations for stress test
        stress_config = BenchmarkConfig(
            iterations=config.iterations * 10,
            warmup_iterations=config.warmup_iterations,
            concurrent_users=min(config.concurrent_users * 2, multiprocessing.cpu_count()),
            timeout=config.timeout * 2,
            memory_limit=config.memory_limit,
            cpu_limit=config.cpu_limit,
            error_threshold=config.error_threshold,
            performance_threshold=config.performance_threshold
        )
        
        self.logger.info(f"Starting stress test: {func.__name__}")
        
        return self.benchmark_concurrent(func, args, kwargs, stress_config)
    
    def compare_functions(self, 
                         functions: List[Tuple[Callable, str]], 
                         args: tuple = (), 
                         kwargs: dict = None,
                         config: BenchmarkConfig = None) -> Dict[str, BenchmarkReport]:
        """Compare multiple functions"""
        kwargs = kwargs or {}
        config = config or BenchmarkConfig()
        
        self.logger.info(f"Starting function comparison with {len(functions)} functions")
        
        results = {}
        
        for func, name in functions:
            try:
                report = self.benchmark_function(func, args, kwargs, config)
                results[name] = report
            except Exception as e:
                self.logger.error(f"Error benchmarking function {name}: {e}")
        
        # Generate comparison analysis
        self._generate_comparison_analysis(results)
        
        return results
    
    def set_baseline(self, name: str, metrics: BenchmarkMetrics):
        """Set baseline metrics for comparison"""
        self.baseline_metrics[name] = metrics
        self.logger.info(f"Baseline set for {name}")
    
    def compare_to_baseline(self, name: str, current_metrics: BenchmarkMetrics) -> Dict[str, float]:
        """Compare current metrics to baseline"""
        if name not in self.baseline_metrics:
            return {"error": "Baseline not found"}
        
        baseline = self.baseline_metrics[name]
        
        comparison = {
            "execution_time_change": ((current_metrics.execution_time - baseline.execution_time) / baseline.execution_time) * 100,
            "memory_usage_change": ((current_metrics.memory_usage - baseline.memory_usage) / baseline.memory_usage) * 100,
            "cpu_usage_change": ((current_metrics.cpu_usage - baseline.cpu_usage) / baseline.cpu_usage) * 100,
            "throughput_change": ((current_metrics.throughput - baseline.throughput) / baseline.throughput) * 100,
            "latency_change": ((current_metrics.latency - baseline.latency) / baseline.latency) * 100
        }
        
        return comparison
    
    def export_report(self, report: BenchmarkReport, filepath: str = None) -> str:
        """Export benchmark report to JSON"""
        if filepath is None:
            timestamp = int(time.time())
            filepath = f"benchmark_report_{report.benchmark_name}_{timestamp}.json"
        
        # Convert report to serializable format
        report_dict = {
            "benchmark_name": report.benchmark_name,
            "benchmark_type": report.benchmark_type.value,
            "config": {
                "iterations": report.config.iterations,
                "warmup_iterations": report.config.warmup_iterations,
                "concurrent_users": report.config.concurrent_users,
                "timeout": report.config.timeout,
                "memory_limit": report.config.memory_limit,
                "cpu_limit": report.config.cpu_limit,
                "error_threshold": report.config.error_threshold,
                "performance_threshold": report.config.performance_threshold
            },
            "summary": report.summary,
            "recommendations": report.recommendations,
            "performance_score": report.performance_score,
            "result_category": report.result_category.value,
            "timestamp": report.timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        self.logger.info(f"Benchmark report exported to {filepath}")
        return filepath
    
    def get_benchmark_history(self) -> List[BenchmarkReport]:
        """Get all benchmark reports"""
        return self.benchmark_history
    
    def clear_history(self):
        """Clear benchmark history"""
        self.benchmark_history.clear()
        self.logger.info("Benchmark history cleared")
    
    def _warmup_function(self, func: Callable, args: tuple, kwargs: dict, iterations: int):
        """Warmup function to stabilize performance"""
        self.logger.info(f"Warming up function {func.__name__} with {iterations} iterations")
        
        for i in range(iterations):
            try:
                func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Warning: Error during warmup iteration {i}: {e}")
    
    def _execute_single_benchmark(self, func: Callable, args: tuple, kwargs: dict) -> BenchmarkMetrics:
        """Execute a single benchmark iteration"""
        # Record initial state
        memory_before = self.memory_tracker.get_memory_usage()
        cpu_before = self.cpu_tracker.get_cpu_usage()
        
        # Execute function
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Record final state
        memory_after = self.memory_tracker.get_memory_usage()
        cpu_after = self.cpu_tracker.get_cpu_usage()
        
        # Calculate metrics
        memory_delta = memory_after["total"] - memory_before["total"]
        cpu_delta = cpu_after - cpu_before
        
        return BenchmarkMetrics(
            execution_time=execution_time,
            memory_usage=memory_delta,
            cpu_usage=cpu_delta,
            throughput=1.0 / execution_time if execution_time > 0 else 0,
            latency=execution_time * 1000,  # Convert to milliseconds
            error_rate=0.0,
            success_rate=1.0,
            iterations=1,
            concurrent_users=1
        )
    
    def _calculate_summary(self, metrics: List[BenchmarkMetrics], errors: int, config: BenchmarkConfig) -> Dict[str, Any]:
        """Calculate summary statistics from metrics"""
        if not metrics:
            return {}
        
        total_iterations = len(metrics) + errors
        success_rate = len(metrics) / total_iterations if total_iterations > 0 else 0
        
        execution_times = [m.execution_time for m in metrics]
        memory_usages = [m.memory_usage for m in metrics]
        cpu_usages = [m.cpu_usage for m in metrics]
        throughputs = [m.throughput for m in metrics]
        latencies = [m.latency for m in metrics]
        
        summary = {
            "total_iterations": total_iterations,
            "successful_iterations": len(metrics),
            "failed_iterations": errors,
            "success_rate": success_rate,
            "error_rate": 1.0 - success_rate,
            
            # Execution time statistics
            "avg_execution_time": statistics.mean(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "std_execution_time": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            "p50_execution_time": statistics.median(execution_times),
            "p95_execution_time": np.percentile(execution_times, 95),
            "p99_execution_time": np.percentile(execution_times, 99),
            
            # Memory statistics
            "avg_memory_usage": statistics.mean(memory_usages),
            "min_memory_usage": min(memory_usages),
            "max_memory_usage": max(memory_usages),
            "std_memory_usage": statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0,
            
            # CPU statistics
            "avg_cpu_usage": statistics.mean(cpu_usages),
            "min_cpu_usage": min(cpu_usages),
            "max_cpu_usage": max(cpu_usages),
            "std_cpu_usage": statistics.stdev(cpu_usages) if len(cpu_usages) > 1 else 0,
            
            # Throughput statistics
            "avg_throughput": statistics.mean(throughputs),
            "min_throughput": min(throughputs),
            "max_throughput": max(throughputs),
            "std_throughput": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
            
            # Latency statistics
            "avg_latency": statistics.mean(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            
            # Performance indicators
            "is_within_memory_limit": summary.get("avg_memory_usage", 0) <= config.memory_limit,
            "is_within_cpu_limit": summary.get("avg_cpu_usage", 0) <= config.cpu_limit,
            "is_within_error_threshold": summary.get("error_rate", 0) <= config.error_threshold,
            "is_within_performance_threshold": summary.get("avg_execution_time", 0) <= config.performance_threshold
        }
        
        return summary
    
    def _calculate_performance_score(self, summary: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)"""
        if not summary:
            return 0.0
        
        # Weighted scoring based on different metrics
        weights = {
            "execution_time": 0.3,
            "memory_usage": 0.25,
            "cpu_usage": 0.2,
            "success_rate": 0.15,
            "throughput": 0.1
        }
        
        scores = {}
        
        # Execution time score (lower is better)
        avg_time = summary.get("avg_execution_time", 0)
        if avg_time <= self.thresholds["excellent"]["time"]:
            scores["execution_time"] = 100
        elif avg_time <= self.thresholds["good"]["time"]:
            scores["execution_time"] = 80
        elif avg_time <= self.thresholds["average"]["time"]:
            scores["execution_time"] = 60
        elif avg_time <= self.thresholds["poor"]["time"]:
            scores["execution_time"] = 40
        else:
            scores["execution_time"] = 20
        
        # Memory usage score (lower is better)
        avg_memory = summary.get("avg_memory_usage", 0)
        if avg_memory <= self.thresholds["excellent"]["memory"]:
            scores["memory_usage"] = 100
        elif avg_memory <= self.thresholds["good"]["memory"]:
            scores["memory_usage"] = 80
        elif avg_memory <= self.thresholds["average"]["memory"]:
            scores["memory_usage"] = 60
        elif avg_memory <= self.thresholds["poor"]["memory"]:
            scores["memory_usage"] = 40
        else:
            scores["memory_usage"] = 20
        
        # CPU usage score (lower is better)
        avg_cpu = summary.get("avg_cpu_usage", 0)
        if avg_cpu <= self.thresholds["excellent"]["cpu"]:
            scores["cpu_usage"] = 100
        elif avg_cpu <= self.thresholds["good"]["cpu"]:
            scores["cpu_usage"] = 80
        elif avg_cpu <= self.thresholds["average"]["cpu"]:
            scores["cpu_usage"] = 60
        elif avg_cpu <= self.thresholds["poor"]["cpu"]:
            scores["cpu_usage"] = 40
        else:
            scores["cpu_usage"] = 20
        
        # Success rate score (higher is better)
        success_rate = summary.get("success_rate", 0)
        scores["success_rate"] = success_rate * 100
        
        # Throughput score (higher is better)
        avg_throughput = summary.get("avg_throughput", 0)
        if avg_throughput >= 1000:
            scores["throughput"] = 100
        elif avg_throughput >= 100:
            scores["throughput"] = 80
        elif avg_throughput >= 10:
            scores["throughput"] = 60
        elif avg_throughput >= 1:
            scores["throughput"] = 40
        else:
            scores["throughput"] = 20
        
        # Calculate weighted average
        total_score = sum(scores[metric] * weights[metric] for metric in weights)
        
        return total_score
    
    def _calculate_memory_score(self, summary: Dict[str, Any]) -> float:
        """Calculate memory-specific performance score"""
        if not summary:
            return 0.0
        
        avg_memory = summary.get("avg_memory_usage", 0)
        memory_variance = summary.get("std_memory_usage", 0)
        
        # Base score from average memory usage
        if avg_memory <= 10:
            base_score = 100
        elif avg_memory <= 50:
            base_score = 80
        elif avg_memory <= 100:
            base_score = 60
        elif avg_memory <= 200:
            base_score = 40
        else:
            base_score = 20
        
        # Penalty for high variance (memory leaks)
        variance_penalty = min(memory_variance / 10, 20)
        
        return max(base_score - variance_penalty, 0)
    
    def _calculate_concurrent_score(self, summary: Dict[str, Any]) -> float:
        """Calculate concurrent performance score"""
        if not summary:
            return 0.0
        
        success_rate = summary.get("success_rate", 0)
        avg_latency = summary.get("avg_latency", 0)
        throughput = summary.get("avg_throughput", 0)
        
        # Success rate is critical for concurrent performance
        success_score = success_rate * 100
        
        # Latency score (lower is better)
        if avg_latency <= 10:
            latency_score = 100
        elif avg_latency <= 50:
            latency_score = 80
        elif avg_latency <= 100:
            latency_score = 60
        elif avg_latency <= 500:
            latency_score = 40
        else:
            latency_score = 20
        
        # Throughput score (higher is better)
        if throughput >= 100:
            throughput_score = 100
        elif throughput >= 50:
            throughput_score = 80
        elif throughput >= 10:
            throughput_score = 60
        elif throughput >= 1:
            throughput_score = 40
        else:
            throughput_score = 20
        
        # Weighted average with emphasis on success rate
        total_score = (success_score * 0.5) + (latency_score * 0.3) + (throughput_score * 0.2)
        
        return total_score
    
    def _categorize_performance(self, score: float) -> BenchmarkResult:
        """Categorize performance based on score"""
        if score >= 90:
            return BenchmarkResult.EXCELLENT
        elif score >= 75:
            return BenchmarkResult.GOOD
        elif score >= 60:
            return BenchmarkResult.AVERAGE
        elif score >= 40:
            return BenchmarkResult.POOR
        else:
            return BenchmarkResult.CRITICAL
    
    def _generate_recommendations(self, summary: Dict[str, Any], result_category: BenchmarkResult) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Execution time recommendations
        avg_time = summary.get("avg_execution_time", 0)
        if avg_time > self.thresholds["good"]["time"]:
            recommendations.append("Consider optimizing algorithm complexity or using caching")
            recommendations.append("Profile the function to identify bottlenecks")
        
        # Memory recommendations
        avg_memory = summary.get("avg_memory_usage", 0)
        if avg_memory > self.thresholds["good"]["memory"]:
            recommendations.append("Review memory allocation patterns")
            recommendations.append("Consider using generators for large datasets")
            recommendations.append("Implement proper cleanup and garbage collection")
        
        # CPU recommendations
        avg_cpu = summary.get("avg_cpu_usage", 0)
        if avg_cpu > self.thresholds["good"]["cpu"]:
            recommendations.append("Consider parallelization or multiprocessing")
            recommendations.append("Optimize CPU-intensive operations")
        
        # Success rate recommendations
        success_rate = summary.get("success_rate", 0)
        if success_rate < 0.95:
            recommendations.append("Improve error handling and input validation")
            recommendations.append("Add retry mechanisms for transient failures")
        
        # General recommendations based on result category
        if result_category == BenchmarkResult.CRITICAL:
            recommendations.append("Immediate optimization required - consider refactoring")
            recommendations.append("Review system resources and scaling requirements")
        elif result_category == BenchmarkResult.POOR:
            recommendations.append("Significant optimization needed")
            recommendations.append("Consider using more efficient data structures")
        elif result_category == BenchmarkResult.AVERAGE:
            recommendations.append("Moderate optimization opportunities available")
        elif result_category == BenchmarkResult.GOOD:
            recommendations.append("Minor optimizations possible for better performance")
        elif result_category == BenchmarkResult.EXCELLENT:
            recommendations.append("Performance is excellent - maintain current implementation")
        
        return recommendations
    
    def _generate_memory_recommendations(self, summary: Dict[str, Any], result_category: BenchmarkResult) -> List[str]:
        """Generate memory-specific recommendations"""
        recommendations = []
        
        avg_memory = summary.get("avg_memory_usage", 0)
        memory_variance = summary.get("std_memory_usage", 0)
        
        if avg_memory > 100:
            recommendations.append("High memory usage detected - consider memory optimization")
            recommendations.append("Review data structures and object lifecycle")
        
        if memory_variance > 50:
            recommendations.append("High memory variance suggests potential memory leaks")
            recommendations.append("Implement proper resource cleanup")
            recommendations.append("Use memory profiling tools to identify leaks")
        
        if result_category in [BenchmarkResult.CRITICAL, BenchmarkResult.POOR]:
            recommendations.append("Critical memory optimization required")
            recommendations.append("Consider using memory-efficient algorithms")
        
        return recommendations
    
    def _generate_concurrent_recommendations(self, summary: Dict[str, Any], result_category: BenchmarkResult) -> List[str]:
        """Generate concurrent-specific recommendations"""
        recommendations = []
        
        success_rate = summary.get("success_rate", 0)
        avg_latency = summary.get("avg_latency", 0)
        
        if success_rate < 0.95:
            recommendations.append("Low success rate in concurrent execution")
            recommendations.append("Review thread safety and synchronization")
            recommendations.append("Implement proper error handling for concurrent operations")
        
        if avg_latency > 100:
            recommendations.append("High latency in concurrent execution")
            recommendations.append("Consider connection pooling or resource sharing")
            recommendations.append("Review locking mechanisms and contention")
        
        if result_category in [BenchmarkResult.CRITICAL, BenchmarkResult.POOR]:
            recommendations.append("Critical concurrent performance issues")
            recommendations.append("Consider redesigning for better concurrency")
        
        return recommendations
    
    def _generate_comparison_analysis(self, results: Dict[str, BenchmarkReport]):
        """Generate analysis for function comparison"""
        if len(results) < 2:
            return
        
        self.logger.info("Generating comparison analysis...")
        
        # Find best and worst performers
        scores = [(name, report.performance_score) for name, report in results.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        best_performer = scores[0]
        worst_performer = scores[-1]
        
        self.logger.info(f"Best performer: {best_performer[0]} (Score: {best_performer[1]:.2f})")
        self.logger.info(f"Worst performer: {worst_performer[0]} (Score: {worst_performer[1]:.2f})")
        
        # Performance differences
        if len(scores) >= 2:
            performance_gap = best_performer[1] - worst_performer[1]
            self.logger.info(f"Performance gap: {performance_gap:.2f} points")
            
            if performance_gap > 30:
                self.logger.warning("Large performance gap detected - significant optimization opportunities")
            elif performance_gap > 15:
                self.logger.info("Moderate performance gap - some optimization opportunities")
            else:
                self.logger.info("Small performance gap - implementations are comparable")


# Example usage and demonstration functions
def example_fast_function():
    """Example fast function for benchmarking"""
    return sum(range(1000))


def example_slow_function():
    """Example slow function for benchmarking"""
    result = 0
    for i in range(10000):
        result += i ** 2
    return result


def example_memory_intensive_function():
    """Example memory-intensive function for benchmarking"""
    large_list = [i for i in range(100000)]
    result = sum(large_list)
    return result


def example_cpu_intensive_function():
    """Example CPU-intensive function for benchmarking"""
    result = 0
    for i in range(100000):
        result += (i ** 0.5) * (i ** 0.3)
    return result


def example_network_simulation():
    """Simulate network operation"""
    time.sleep(0.1)  # Simulate network delay
    return "network_response"


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create benchmarker
    benchmarker = PerformanceBenchmarker(DebugLevel.DETAILED)
    
    # Example benchmarks
    print("🔍 Performance Benchmarking System Demo")
    print("=" * 50)
    
    # Function benchmark
    print("\n1. Function Benchmark:")
    report = benchmarker.benchmark_function(example_fast_function)
    print(f"   Function: {report.benchmark_name}")
    print(f"   Score: {report.performance_score:.2f}")
    print(f"   Category: {report.result_category.value}")
    print(f"   Avg Time: {report.summary['avg_execution_time']:.4f}s")
    
    # Memory benchmark
    print("\n2. Memory Benchmark:")
    memory_report = benchmarker.benchmark_memory(example_memory_intensive_function)
    print(f"   Function: {memory_report.benchmark_name}")
    print(f"   Score: {memory_report.performance_score:.2f}")
    print(f"   Category: {memory_report.result_category.value}")
    print(f"   Avg Memory: {memory_report.summary['avg_memory_usage']:.2f}MB")
    
    # Concurrent benchmark
    print("\n3. Concurrent Benchmark:")
    concurrent_config = BenchmarkConfig(iterations=50, concurrent_users=4)
    concurrent_report = benchmarker.benchmark_concurrent(
        example_network_simulation, 
        config=concurrent_config
    )
    print(f"   Function: {concurrent_report.benchmark_name}")
    print(f"   Score: {concurrent_report.performance_score:.2f}")
    print(f"   Category: {concurrent_report.result_category.value}")
    print(f"   Success Rate: {concurrent_report.summary['success_rate']:.2%}")
    
    # Function comparison
    print("\n4. Function Comparison:")
    functions = [
        (example_fast_function, "Fast Function"),
        (example_slow_function, "Slow Function"),
        (example_cpu_intensive_function, "CPU Intensive")
    ]
    comparison_results = benchmarker.compare_functions(functions)
    
    for name, report in comparison_results.items():
        print(f"   {name}: Score {report.performance_score:.2f} ({report.result_category.value})")
    
    print("\n✅ Benchmarking demo completed!")




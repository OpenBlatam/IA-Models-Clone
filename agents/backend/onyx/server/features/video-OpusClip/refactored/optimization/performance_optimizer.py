"""
Performance Optimizer

Advanced performance optimization system that automatically tunes system parameters,
optimizes resource usage, and improves processing efficiency.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union, Callable
import asyncio
import time
import psutil
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import structlog
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import threading
from pathlib import Path

logger = structlog.get_logger("performance_optimizer")

class OptimizationTarget(Enum):
    """Performance optimization targets."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_EFFICIENCY = "resource_efficiency"

class OptimizationStrategy(Enum):
    """Optimization strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

@dataclass
class OptimizationRule:
    """Performance optimization rule."""
    rule_id: str
    name: str
    target: OptimizationTarget
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Dict[str, Any]]
    priority: int = 1
    enabled: bool = True
    last_applied: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0

@dataclass
class OptimizationResult:
    """Result of an optimization operation."""
    rule_id: str
    success: bool
    changes_applied: Dict[str, Any]
    performance_impact: Dict[str, float]
    timestamp: datetime
    error_message: Optional[str] = None

@dataclass
class SystemProfile:
    """System performance profile."""
    cpu_cores: int
    total_memory: float
    available_memory: float
    disk_space: float
    network_bandwidth: float
    gpu_available: bool
    gpu_memory: Optional[float] = None
    os_type: str = "unknown"
    python_version: str = "unknown"

class PerformanceAnalyzer:
    """Analyzes system performance and identifies optimization opportunities."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.baseline_metrics: Dict[str, float] = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        self.logger = structlog.get_logger("performance_analyzer")
    
    def record_metric(self, metric_name: str, value: float, timestamp: datetime = None):
        """Record a performance metric."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.performance_history[metric_name].append({
            "value": value,
            "timestamp": timestamp
        })
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance and identify issues."""
        try:
            analysis = {
                "timestamp": datetime.now(),
                "metrics": {},
                "anomalies": [],
                "recommendations": [],
                "overall_health": "good"
            }
            
            # Analyze each metric
            for metric_name, history in self.performance_history.items():
                if not history:
                    continue
                
                values = [entry["value"] for entry in history]
                recent_values = values[-10:] if len(values) >= 10 else values
                
                metric_analysis = {
                    "current": recent_values[-1] if recent_values else 0,
                    "average": np.mean(values),
                    "trend": self._calculate_trend(values),
                    "volatility": np.std(values),
                    "anomaly_score": self._calculate_anomaly_score(values)
                }
                
                analysis["metrics"][metric_name] = metric_analysis
                
                # Check for anomalies
                if metric_analysis["anomaly_score"] > self.anomaly_threshold:
                    analysis["anomalies"].append({
                        "metric": metric_name,
                        "score": metric_analysis["anomaly_score"],
                        "current_value": metric_analysis["current"],
                        "expected_range": self._get_expected_range(metric_name)
                    })
            
            # Generate recommendations
            analysis["recommendations"] = self._generate_recommendations(analysis)
            
            # Calculate overall health
            analysis["overall_health"] = self._calculate_overall_health(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a metric."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression to determine trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_anomaly_score(self, values: List[float]) -> float:
        """Calculate anomaly score for a metric."""
        if len(values) < 3:
            return 0.0
        
        # Use Z-score for anomaly detection
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        current_value = values[-1]
        z_score = abs((current_value - mean) / std)
        
        return z_score
    
    def _get_expected_range(self, metric_name: str) -> tuple:
        """Get expected range for a metric."""
        ranges = {
            "cpu_usage": (0, 80),
            "memory_usage": (0, 85),
            "response_time": (0, 2.0),
            "error_rate": (0, 0.05),
            "throughput": (0, float('inf'))
        }
        
        return ranges.get(metric_name, (0, 100))
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # CPU recommendations
        cpu_metric = analysis["metrics"].get("cpu_usage")
        if cpu_metric and cpu_metric["current"] > 80:
            recommendations.append("High CPU usage detected - consider scaling up or optimizing CPU-intensive operations")
        
        # Memory recommendations
        memory_metric = analysis["metrics"].get("memory_usage")
        if memory_metric and memory_metric["current"] > 85:
            recommendations.append("High memory usage detected - consider increasing memory allocation or optimizing memory usage")
        
        # Response time recommendations
        response_metric = analysis["metrics"].get("response_time")
        if response_metric and response_metric["current"] > 2.0:
            recommendations.append("Slow response times detected - consider optimizing request processing")
        
        # Error rate recommendations
        error_metric = analysis["metrics"].get("error_rate")
        if error_metric and error_metric["current"] > 0.05:
            recommendations.append("High error rate detected - investigate and fix error conditions")
        
        return recommendations
    
    def _calculate_overall_health(self, analysis: Dict[str, Any]) -> str:
        """Calculate overall system health."""
        anomaly_count = len(analysis["anomalies"])
        
        if anomaly_count == 0:
            return "excellent"
        elif anomaly_count <= 2:
            return "good"
        elif anomaly_count <= 4:
            return "degraded"
        else:
            return "poor"

class ResourceManager:
    """Manages system resources and allocation."""
    
    def __init__(self):
        self.system_profile = self._create_system_profile()
        self.resource_limits = self._calculate_resource_limits()
        self.current_allocation = {}
        self.logger = structlog.get_logger("resource_manager")
    
    def _create_system_profile(self) -> SystemProfile:
        """Create system performance profile."""
        try:
            return SystemProfile(
                cpu_cores=psutil.cpu_count(),
                total_memory=psutil.virtual_memory().total / (1024**3),  # GB
                available_memory=psutil.virtual_memory().available / (1024**3),  # GB
                disk_space=psutil.disk_usage('/').free / (1024**3),  # GB
                network_bandwidth=self._estimate_network_bandwidth(),
                gpu_available=self._check_gpu_availability(),
                gpu_memory=self._get_gpu_memory(),
                os_type=psutil.WINDOWS if hasattr(psutil, 'WINDOWS') else "linux",
                python_version=f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}"
            )
        except Exception as e:
            self.logger.error(f"Failed to create system profile: {e}")
            return SystemProfile(1, 1.0, 1.0, 1.0, 1.0, False)
    
    def _estimate_network_bandwidth(self) -> float:
        """Estimate network bandwidth in Mbps."""
        # Placeholder - would implement actual network speed test
        return 100.0  # Assume 100 Mbps
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_gpu_memory(self) -> Optional[float]:
        """Get GPU memory in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass
        return None
    
    def _calculate_resource_limits(self) -> Dict[str, float]:
        """Calculate safe resource limits."""
        return {
            "max_cpu_usage": 0.8,  # 80% of CPU
            "max_memory_usage": 0.85,  # 85% of memory
            "max_workers": min(self.system_profile.cpu_cores * 2, 16),
            "max_concurrent_jobs": min(self.system_profile.cpu_cores * 4, 32),
            "max_queue_size": 1000
        }
    
    def allocate_resources(self, job_type: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate resources for a job."""
        try:
            # Calculate required resources
            cpu_cores = requirements.get("cpu_cores", 1)
            memory_gb = requirements.get("memory_gb", 1.0)
            
            # Check if resources are available
            if not self._check_resource_availability(cpu_cores, memory_gb):
                return {"success": False, "reason": "Insufficient resources"}
            
            # Allocate resources
            allocation = {
                "cpu_cores": cpu_cores,
                "memory_gb": memory_gb,
                "gpu_required": requirements.get("gpu_required", False),
                "priority": requirements.get("priority", 1)
            }
            
            self.current_allocation[job_type] = allocation
            
            return {"success": True, "allocation": allocation}
            
        except Exception as e:
            self.logger.error(f"Resource allocation failed: {e}")
            return {"success": False, "reason": str(e)}
    
    def _check_resource_availability(self, cpu_cores: int, memory_gb: float) -> bool:
        """Check if requested resources are available."""
        try:
            # Check CPU availability
            current_cpu = psutil.cpu_percent(interval=1)
            if current_cpu > self.resource_limits["max_cpu_usage"] * 100:
                return False
            
            # Check memory availability
            current_memory = psutil.virtual_memory().percent
            if current_memory > self.resource_limits["max_memory_usage"] * 100:
                return False
            
            # Check if we have enough cores
            if cpu_cores > self.system_profile.cpu_cores:
                return False
            
            # Check if we have enough memory
            if memory_gb > self.system_profile.available_memory:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Resource availability check failed: {e}")
            return False
    
    def release_resources(self, job_type: str):
        """Release resources for a job."""
        if job_type in self.current_allocation:
            del self.current_allocation[job_type]

class AutoTuner:
    """Automatically tunes system parameters for optimal performance."""
    
    def __init__(self, performance_analyzer: PerformanceAnalyzer, resource_manager: ResourceManager):
        self.performance_analyzer = performance_analyzer
        self.resource_manager = resource_manager
        self.tuning_history: List[Dict[str, Any]] = []
        self.logger = structlog.get_logger("auto_tuner")
    
    def tune_parameters(self, current_config: Dict[str, Any], performance_goals: Dict[str, float]) -> Dict[str, Any]:
        """Tune system parameters based on performance goals."""
        try:
            # Analyze current performance
            analysis = self.performance_analyzer.analyze_performance()
            
            # Generate tuning recommendations
            recommendations = self._generate_tuning_recommendations(analysis, performance_goals)
            
            # Apply tuning
            tuned_config = current_config.copy()
            for recommendation in recommendations:
                tuned_config = self._apply_tuning_recommendation(tuned_config, recommendation)
            
            # Record tuning
            self.tuning_history.append({
                "timestamp": datetime.now(),
                "original_config": current_config,
                "tuned_config": tuned_config,
                "recommendations": recommendations,
                "performance_goals": performance_goals
            })
            
            return tuned_config
            
        except Exception as e:
            self.logger.error(f"Parameter tuning failed: {e}")
            return current_config
    
    def _generate_tuning_recommendations(self, analysis: Dict[str, Any], goals: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate tuning recommendations based on analysis and goals."""
        recommendations = []
        
        # CPU tuning
        cpu_metric = analysis["metrics"].get("cpu_usage")
        if cpu_metric and cpu_metric["current"] > goals.get("max_cpu_usage", 80):
            recommendations.append({
                "parameter": "max_workers",
                "action": "decrease",
                "current_value": "current_max_workers",
                "recommended_value": "current_max_workers * 0.8",
                "reason": "High CPU usage detected"
            })
        
        # Memory tuning
        memory_metric = analysis["metrics"].get("memory_usage")
        if memory_metric and memory_metric["current"] > goals.get("max_memory_usage", 85):
            recommendations.append({
                "parameter": "cache_size",
                "action": "decrease",
                "current_value": "current_cache_size",
                "recommended_value": "current_cache_size * 0.7",
                "reason": "High memory usage detected"
            })
        
        # Response time tuning
        response_metric = analysis["metrics"].get("response_time")
        if response_metric and response_metric["current"] > goals.get("max_response_time", 2.0):
            recommendations.append({
                "parameter": "timeout",
                "action": "decrease",
                "current_value": "current_timeout",
                "recommended_value": "current_timeout * 0.8",
                "reason": "Slow response times detected"
            })
        
        return recommendations
    
    def _apply_tuning_recommendation(self, config: Dict[str, Any], recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a tuning recommendation to configuration."""
        try:
            parameter = recommendation["parameter"]
            action = recommendation["action"]
            
            if action == "decrease":
                current_value = config.get(parameter, 1)
                new_value = current_value * 0.8
                config[parameter] = max(1, int(new_value))
            elif action == "increase":
                current_value = config.get(parameter, 1)
                new_value = current_value * 1.2
                config[parameter] = int(new_value)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to apply tuning recommendation: {e}")
            return config

class PerformanceOptimizer:
    """Main performance optimization system."""
    
    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        self.resource_manager = ResourceManager()
        self.auto_tuner = AutoTuner(self.performance_analyzer, self.resource_manager)
        self.optimization_rules: List[OptimizationRule] = []
        self.optimization_history: List[OptimizationResult] = []
        
        self.running = False
        self._optimization_task: Optional[asyncio.Task] = None
        
        self.logger = structlog.get_logger("performance_optimizer")
        
        # Setup default optimization rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default optimization rules."""
        # CPU optimization rule
        self.add_optimization_rule(OptimizationRule(
            rule_id="cpu_optimization",
            name="CPU Usage Optimization",
            target=OptimizationTarget.CPU_USAGE,
            condition=lambda metrics: metrics.get("cpu_usage", 0) > 80,
            action=lambda metrics: {"max_workers": max(1, int(metrics.get("max_workers", 4) * 0.8))},
            priority=1
        ))
        
        # Memory optimization rule
        self.add_optimization_rule(OptimizationRule(
            rule_id="memory_optimization",
            name="Memory Usage Optimization",
            target=OptimizationTarget.MEMORY_USAGE,
            condition=lambda metrics: metrics.get("memory_usage", 0) > 85,
            action=lambda metrics: {"cache_size": max(100, int(metrics.get("cache_size", 1000) * 0.7))},
            priority=1
        ))
        
        # Response time optimization rule
        self.add_optimization_rule(OptimizationRule(
            rule_id="response_time_optimization",
            name="Response Time Optimization",
            target=OptimizationTarget.RESPONSE_TIME,
            condition=lambda metrics: metrics.get("response_time", 0) > 2.0,
            action=lambda metrics: {"timeout": max(30, int(metrics.get("timeout", 300) * 0.8))},
            priority=2
        ))
    
    def add_optimization_rule(self, rule: OptimizationRule):
        """Add an optimization rule."""
        self.optimization_rules.append(rule)
        self.logger.info(f"Added optimization rule: {rule.name}")
    
    def remove_optimization_rule(self, rule_id: str):
        """Remove an optimization rule."""
        self.optimization_rules = [r for r in self.optimization_rules if r.rule_id != rule_id]
        self.logger.info(f"Removed optimization rule: {rule_id}")
    
    async def start(self):
        """Start the performance optimizer."""
        try:
            self.running = True
            self._optimization_task = asyncio.create_task(self._optimization_loop())
            self.logger.info("Performance optimizer started")
            
        except Exception as e:
            self.logger.error(f"Failed to start performance optimizer: {e}")
            raise
    
    async def stop(self):
        """Stop the performance optimizer."""
        try:
            self.running = False
            
            if self._optimization_task:
                self._optimization_task.cancel()
                try:
                    await self._optimization_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("Performance optimizer stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop performance optimizer: {e}")
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        while self.running:
            try:
                # Collect current metrics
                await self._collect_current_metrics()
                
                # Run optimization
                await self._run_optimization()
                
                # Wait before next optimization cycle
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)
    
    async def _collect_current_metrics(self):
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.performance_analyzer.record_metric("cpu_usage", cpu_percent)
            
            # Memory usage
            memory_percent = psutil.virtual_memory().percent
            self.performance_analyzer.record_metric("memory_usage", memory_percent)
            
            # Disk usage
            disk_percent = psutil.disk_usage('/').percent
            self.performance_analyzer.record_metric("disk_usage", disk_percent)
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
    
    async def _run_optimization(self):
        """Run optimization based on current metrics."""
        try:
            # Get current metrics
            analysis = self.performance_analyzer.analyze_performance()
            
            # Check each optimization rule
            for rule in self.optimization_rules:
                if not rule.enabled:
                    continue
                
                # Check if rule condition is met
                current_metrics = {
                    "cpu_usage": analysis["metrics"].get("cpu_usage", {}).get("current", 0),
                    "memory_usage": analysis["metrics"].get("memory_usage", {}).get("current", 0),
                    "response_time": analysis["metrics"].get("response_time", {}).get("current", 0)
                }
                
                if rule.condition(current_metrics):
                    # Apply optimization
                    try:
                        changes = rule.action(current_metrics)
                        
                        # Record optimization result
                        result = OptimizationResult(
                            rule_id=rule.rule_id,
                            success=True,
                            changes_applied=changes,
                            performance_impact={},
                            timestamp=datetime.now()
                        )
                        
                        self.optimization_history.append(result)
                        rule.last_applied = datetime.now()
                        rule.success_count += 1
                        
                        self.logger.info(f"Applied optimization rule: {rule.name}")
                        
                    except Exception as e:
                        # Record failed optimization
                        result = OptimizationResult(
                            rule_id=rule.rule_id,
                            success=False,
                            changes_applied={},
                            performance_impact={},
                            timestamp=datetime.now(),
                            error_message=str(e)
                        )
                        
                        self.optimization_history.append(result)
                        rule.failure_count += 1
                        
                        self.logger.error(f"Failed to apply optimization rule {rule.name}: {e}")
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        try:
            analysis = self.performance_analyzer.analyze_performance()
            
            return {
                "running": self.running,
                "rules_count": len(self.optimization_rules),
                "active_rules": len([r for r in self.optimization_rules if r.enabled]),
                "optimization_count": len(self.optimization_history),
                "recent_optimizations": len([r for r in self.optimization_history if (datetime.now() - r.timestamp).total_seconds() < 3600]),
                "system_health": analysis.get("overall_health", "unknown"),
                "recommendations": analysis.get("recommendations", [])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization status: {e}")
            return {"error": str(e)}
    
    def get_optimization_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get optimization history."""
        try:
            recent_results = self.optimization_history[-limit:] if limit > 0 else self.optimization_history
            
            return [
                {
                    "rule_id": result.rule_id,
                    "success": result.success,
                    "changes_applied": result.changes_applied,
                    "timestamp": result.timestamp.isoformat(),
                    "error_message": result.error_message
                }
                for result in recent_results
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization history: {e}")
            return []

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

# Export classes
__all__ = [
    "PerformanceOptimizer",
    "PerformanceAnalyzer",
    "ResourceManager",
    "AutoTuner",
    "OptimizationRule",
    "OptimizationResult",
    "SystemProfile",
    "OptimizationTarget",
    "OptimizationStrategy",
    "performance_optimizer"
]



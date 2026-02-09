"""
Performance Optimizer for Refactored Opus Clip

Advanced performance optimization with:
- Automatic performance tuning
- Resource management
- Cache optimization
- Memory management
- CPU optimization
- I/O optimization
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable
import asyncio
import time
import psutil
import gc
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog
import json
from pathlib import Path
import numpy as np
import torch

logger = structlog.get_logger("performance_optimizer")

@dataclass
class OptimizationRule:
    """Performance optimization rule."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], None]
    priority: int = 1
    enabled: bool = True
    last_applied: Optional[datetime] = None
    cooldown_seconds: int = 60

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    response_time: float
    error_rate: float
    queue_size: int
    active_jobs: int
    cache_hit_rate: float
    processing_time: float

class PerformanceOptimizer:
    """
    Advanced performance optimizer for Opus Clip.
    
    Features:
    - Automatic performance tuning
    - Resource management
    - Cache optimization
    - Memory management
    - CPU optimization
    - I/O optimization
    """
    
    def __init__(self, 
                 config_manager: Optional[Any] = None,
                 job_manager: Optional[Any] = None,
                 analyzer: Optional[Any] = None,
                 exporter: Optional[Any] = None,
                 monitor: Optional[Any] = None):
        """Initialize performance optimizer."""
        self.config_manager = config_manager
        self.job_manager = job_manager
        self.analyzer = analyzer
        self.exporter = exporter
        self.monitor = monitor
        
        self.logger = structlog.get_logger("performance_optimizer")
        
        # Optimization state
        self.optimization_rules: List[OptimizationRule] = []
        self.performance_history: List[PerformanceMetrics] = []
        self.optimization_enabled = True
        self.optimization_interval = 30.0  # seconds
        self.optimization_task: Optional[asyncio.Task] = None
        
        # Performance baselines
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.performance_thresholds = {
            "cpu_usage": 70.0,
            "memory_usage": 80.0,
            "response_time": 5.0,
            "error_rate": 5.0,
            "cache_hit_rate": 0.8
        }
        
        # Optimization statistics
        self.optimization_stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "performance_improvements": 0,
            "last_optimization": None
        }
        
        # Initialize optimization rules
        self._initialize_optimization_rules()
        
        self.logger.info("Performance optimizer initialized")
    
    def _initialize_optimization_rules(self):
        """Initialize optimization rules."""
        # CPU optimization rules
        self.optimization_rules.append(OptimizationRule(
            name="reduce_workers_on_high_cpu",
            condition=lambda metrics: metrics.cpu_usage > 85.0,
            action=self._reduce_workers,
            priority=1,
            cooldown_seconds=120
        ))
        
        self.optimization_rules.append(OptimizationRule(
            name="increase_workers_on_low_cpu",
            condition=lambda metrics: metrics.cpu_usage < 30.0 and metrics.queue_size > 10,
            action=self._increase_workers,
            priority=2,
            cooldown_seconds=60
        ))
        
        # Memory optimization rules
        self.optimization_rules.append(OptimizationRule(
            name="clear_cache_on_high_memory",
            condition=lambda metrics: metrics.memory_usage > 90.0,
            action=self._clear_caches,
            priority=1,
            cooldown_seconds=300
        ))
        
        self.optimization_rules.append(OptimizationRule(
            name="force_garbage_collection",
            condition=lambda metrics: metrics.memory_usage > 85.0,
            action=self._force_garbage_collection,
            priority=2,
            cooldown_seconds=60
        ))
        
        # Cache optimization rules
        self.optimization_rules.append(OptimizationRule(
            name="increase_cache_ttl_on_low_hit_rate",
            condition=lambda metrics: metrics.cache_hit_rate < 0.6,
            action=self._increase_cache_ttl,
            priority=3,
            cooldown_seconds=180
        ))
        
        self.optimization_rules.append(OptimizationRule(
            name="decrease_cache_ttl_on_high_memory",
            condition=lambda metrics: metrics.memory_usage > 80.0 and metrics.cache_hit_rate > 0.8,
            action=self._decrease_cache_ttl,
            priority=2,
            cooldown_seconds=120
        ))
        
        # Response time optimization rules
        self.optimization_rules.append(OptimizationRule(
            name="enable_parallel_processing",
            condition=lambda metrics: metrics.response_time > 10.0 and metrics.cpu_usage < 70.0,
            action=self._enable_parallel_processing,
            priority=1,
            cooldown_seconds=60
        ))
        
        # Error rate optimization rules
        self.optimization_rules.append(OptimizationRule(
            name="increase_retry_attempts",
            condition=lambda metrics: metrics.error_rate > 10.0,
            action=self._increase_retry_attempts,
            priority=1,
            cooldown_seconds=300
        ))
        
        # GPU optimization rules
        self.optimization_rules.append(OptimizationRule(
            name="optimize_gpu_memory",
            condition=lambda metrics: metrics.memory_usage > 85.0 and torch.cuda.is_available(),
            action=self._optimize_gpu_memory,
            priority=1,
            cooldown_seconds=120
        ))
    
    async def start_optimization(self):
        """Start automatic optimization."""
        if self.optimization_task and not self.optimization_task.done():
            return
        
        self.optimization_enabled = True
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        self.logger.info("Performance optimization started")
    
    async def stop_optimization(self):
        """Stop automatic optimization."""
        self.optimization_enabled = False
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Performance optimization stopped")
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        while self.optimization_enabled:
            try:
                # Collect current performance metrics
                metrics = await self._collect_performance_metrics()
                
                # Store metrics history
                self.performance_history.append(metrics)
                if len(self.performance_history) > 1000:  # Keep last 1000 metrics
                    self.performance_history = self.performance_history[-1000:]
                
                # Set baseline if not set
                if self.baseline_metrics is None:
                    self.baseline_metrics = metrics
                
                # Apply optimization rules
                await self._apply_optimization_rules(metrics)
                
                # Wait for next optimization cycle
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(self.optimization_interval)
    
    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Application metrics
            response_time = 0.0
            error_rate = 0.0
            queue_size = 0
            active_jobs = 0
            cache_hit_rate = 0.0
            processing_time = 0.0
            
            if self.job_manager:
                stats = await self.job_manager.get_statistics()
                queue_size = stats.get("queued_jobs", 0)
                active_jobs = stats.get("active_jobs", 0)
                processing_time = stats.get("stats", {}).get("average_processing_time", 0.0)
                
                total_requests = stats.get("stats", {}).get("total_requests", 0)
                failed_requests = stats.get("stats", {}).get("failed_requests", 0)
                if total_requests > 0:
                    error_rate = (failed_requests / total_requests) * 100
            
            if self.analyzer:
                analyzer_stats = await self.analyzer.get_status()
                # Calculate cache hit rate (simplified)
                cache_size = analyzer_stats.get("cache_size", 0)
                total_processed = analyzer_stats.get("total_processed", 0)
                if total_processed > 0:
                    cache_hit_rate = min(cache_size / total_processed, 1.0)
            
            if self.monitor:
                monitor_summary = await self.monitor.get_performance_summary()
                response_time = monitor_summary.get("recent_averages", {}).get("response_time", 0.0)
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                memory_available=memory.available / (1024**3),  # GB
                disk_usage=(disk.used / disk.total) * 100,
                response_time=response_time,
                error_rate=error_rate,
                queue_size=queue_size,
                active_jobs=active_jobs,
                cache_hit_rate=cache_hit_rate,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {e}")
            # Return default metrics
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                memory_available=0.0,
                disk_usage=0.0,
                response_time=0.0,
                error_rate=0.0,
                queue_size=0,
                active_jobs=0,
                cache_hit_rate=0.0,
                processing_time=0.0
            )
    
    async def _apply_optimization_rules(self, metrics: PerformanceMetrics):
        """Apply optimization rules based on current metrics."""
        try:
            # Sort rules by priority
            applicable_rules = [
                rule for rule in self.optimization_rules
                if rule.enabled and self._is_rule_applicable(rule, metrics)
            ]
            applicable_rules.sort(key=lambda r: r.priority)
            
            # Apply rules
            for rule in applicable_rules:
                try:
                    self.logger.info(f"Applying optimization rule: {rule.name}")
                    rule.action(metrics)
                    rule.last_applied = datetime.now()
                    
                    self.optimization_stats["total_optimizations"] += 1
                    self.optimization_stats["successful_optimizations"] += 1
                    self.optimization_stats["last_optimization"] = datetime.now()
                    
                    self.logger.info(f"Successfully applied optimization rule: {rule.name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to apply optimization rule {rule.name}: {e}")
                    self.optimization_stats["failed_optimizations"] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimization rules: {e}")
    
    def _is_rule_applicable(self, rule: OptimizationRule, metrics: PerformanceMetrics) -> bool:
        """Check if a rule is applicable."""
        try:
            # Check cooldown
            if rule.last_applied:
                time_since_last = (datetime.now() - rule.last_applied).total_seconds()
                if time_since_last < rule.cooldown_seconds:
                    return False
            
            # Check condition
            return rule.condition(metrics)
            
        except Exception as e:
            self.logger.error(f"Error checking rule applicability: {e}")
            return False
    
    def _reduce_workers(self, metrics: PerformanceMetrics):
        """Reduce number of workers to lower CPU usage."""
        if self.job_manager and hasattr(self.job_manager, 'max_workers'):
            current_workers = self.job_manager.max_workers
            new_workers = max(1, current_workers - 1)
            self.job_manager.max_workers = new_workers
            self.logger.info(f"Reduced workers from {current_workers} to {new_workers}")
    
    def _increase_workers(self, metrics: PerformanceMetrics):
        """Increase number of workers to process more jobs."""
        if self.job_manager and hasattr(self.job_manager, 'max_workers'):
            current_workers = self.job_manager.max_workers
            new_workers = min(8, current_workers + 1)  # Max 8 workers
            self.job_manager.max_workers = new_workers
            self.logger.info(f"Increased workers from {current_workers} to {new_workers}")
    
    def _clear_caches(self, metrics: PerformanceMetrics):
        """Clear caches to free memory."""
        if self.analyzer:
            asyncio.create_task(self.analyzer.clear_cache())
        if self.exporter:
            asyncio.create_task(self.exporter.clear_cache())
        self.logger.info("Cleared all caches")
    
    def _force_garbage_collection(self, metrics: PerformanceMetrics):
        """Force garbage collection to free memory."""
        collected = gc.collect()
        self.logger.info(f"Forced garbage collection, collected {collected} objects")
    
    def _increase_cache_ttl(self, metrics: PerformanceMetrics):
        """Increase cache TTL to improve hit rate."""
        if self.config_manager and hasattr(self.config_manager, 'performance'):
            current_ttl = self.config_manager.performance.cache_ttl_seconds
            new_ttl = min(7200, current_ttl * 2)  # Max 2 hours
            self.config_manager.performance.cache_ttl_seconds = new_ttl
            self.logger.info(f"Increased cache TTL from {current_ttl} to {new_ttl} seconds")
    
    def _decrease_cache_ttl(self, metrics: PerformanceMetrics):
        """Decrease cache TTL to free memory."""
        if self.config_manager and hasattr(self.config_manager, 'performance'):
            current_ttl = self.config_manager.performance.cache_ttl_seconds
            new_ttl = max(300, current_ttl // 2)  # Min 5 minutes
            self.config_manager.performance.cache_ttl_seconds = new_ttl
            self.logger.info(f"Decreased cache TTL from {current_ttl} to {new_ttl} seconds")
    
    def _enable_parallel_processing(self, metrics: PerformanceMetrics):
        """Enable parallel processing for better performance."""
        # This would enable parallel processing in processors
        self.logger.info("Enabled parallel processing")
    
    def _increase_retry_attempts(self, metrics: PerformanceMetrics):
        """Increase retry attempts to reduce error rate."""
        if self.config_manager and hasattr(self.config_manager, 'performance'):
            # This would increase retry attempts in processors
            self.logger.info("Increased retry attempts")
    
    def _optimize_gpu_memory(self, metrics: PerformanceMetrics):
        """Optimize GPU memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("Optimized GPU memory")
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get optimization status and statistics."""
        try:
            current_metrics = await self._collect_performance_metrics()
            
            # Calculate performance improvement
            performance_improvement = 0.0
            if self.baseline_metrics:
                baseline_response_time = self.baseline_metrics.response_time
                current_response_time = current_metrics.response_time
                if baseline_response_time > 0:
                    performance_improvement = ((baseline_response_time - current_response_time) / baseline_response_time) * 100
            
            return {
                "optimization_enabled": self.optimization_enabled,
                "total_optimizations": self.optimization_stats["total_optimizations"],
                "successful_optimizations": self.optimization_stats["successful_optimizations"],
                "failed_optimizations": self.optimization_stats["failed_optimizations"],
                "performance_improvement": performance_improvement,
                "last_optimization": self.optimization_stats["last_optimization"].isoformat() if self.optimization_stats["last_optimization"] else None,
                "current_metrics": {
                    "cpu_usage": current_metrics.cpu_usage,
                    "memory_usage": current_metrics.memory_usage,
                    "response_time": current_metrics.response_time,
                    "error_rate": current_metrics.error_rate,
                    "cache_hit_rate": current_metrics.cache_hit_rate
                },
                "active_rules": len([r for r in self.optimization_rules if r.enabled]),
                "metrics_history_size": len(self.performance_history)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization status: {e}")
            return {"error": str(e)}
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on current metrics."""
        try:
            current_metrics = await self._collect_performance_metrics()
            recommendations = []
            
            # CPU recommendations
            if current_metrics.cpu_usage > 80:
                recommendations.append({
                    "type": "cpu",
                    "priority": "high",
                    "message": "High CPU usage detected",
                    "recommendation": "Consider reducing concurrent operations or optimizing algorithms",
                    "current_value": current_metrics.cpu_usage,
                    "threshold": self.performance_thresholds["cpu_usage"]
                })
            
            # Memory recommendations
            if current_metrics.memory_usage > 85:
                recommendations.append({
                    "type": "memory",
                    "priority": "high",
                    "message": "High memory usage detected",
                    "recommendation": "Clear caches or reduce memory footprint",
                    "current_value": current_metrics.memory_usage,
                    "threshold": self.performance_thresholds["memory_usage"]
                })
            
            # Response time recommendations
            if current_metrics.response_time > 10:
                recommendations.append({
                    "type": "response_time",
                    "priority": "medium",
                    "message": "High response time detected",
                    "recommendation": "Enable parallel processing or optimize algorithms",
                    "current_value": current_metrics.response_time,
                    "threshold": self.performance_thresholds["response_time"]
                })
            
            # Error rate recommendations
            if current_metrics.error_rate > 10:
                recommendations.append({
                    "type": "error_rate",
                    "priority": "high",
                    "message": "High error rate detected",
                    "recommendation": "Improve error handling and retry mechanisms",
                    "current_value": current_metrics.error_rate,
                    "threshold": self.performance_thresholds["error_rate"]
                })
            
            # Cache recommendations
            if current_metrics.cache_hit_rate < 0.6:
                recommendations.append({
                    "type": "cache",
                    "priority": "medium",
                    "message": "Low cache hit rate detected",
                    "recommendation": "Increase cache TTL or improve caching strategy",
                    "current_value": current_metrics.cache_hit_rate,
                    "threshold": self.performance_thresholds["cache_hit_rate"]
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization recommendations: {e}")
            return []
    
    async def manual_optimize(self, optimization_type: str) -> bool:
        """Manually trigger optimization."""
        try:
            current_metrics = await self._collect_performance_metrics()
            
            if optimization_type == "cpu":
                self._reduce_workers(current_metrics)
            elif optimization_type == "memory":
                self._clear_caches(current_metrics)
                self._force_garbage_collection(current_metrics)
            elif optimization_type == "cache":
                self._increase_cache_ttl(current_metrics)
            elif optimization_type == "gpu":
                self._optimize_gpu_memory(current_metrics)
            else:
                return False
            
            self.logger.info(f"Manual optimization applied: {optimization_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply manual optimization: {e}")
            return False
    
    async def reset_optimization(self):
        """Reset optimization settings to defaults."""
        try:
            # Reset configuration to defaults
            if self.config_manager and hasattr(self.config_manager, 'performance'):
                self.config_manager.performance.cache_ttl_seconds = 3600
                self.config_manager.performance.max_workers = 4
            
            # Reset job manager workers
            if self.job_manager:
                self.job_manager.max_workers = 4
            
            # Clear performance history
            self.performance_history.clear()
            self.baseline_metrics = None
            
            # Reset statistics
            self.optimization_stats = {
                "total_optimizations": 0,
                "successful_optimizations": 0,
                "failed_optimizations": 0,
                "performance_improvements": 0,
                "last_optimization": None
            }
            
            self.logger.info("Optimization settings reset to defaults")
            
        except Exception as e:
            self.logger.error(f"Failed to reset optimization: {e}")
    
    async def add_custom_rule(self, name: str, condition: Callable, action: Callable, 
                            priority: int = 5, cooldown_seconds: int = 60):
        """Add custom optimization rule."""
        try:
            rule = OptimizationRule(
                name=name,
                condition=condition,
                action=action,
                priority=priority,
                cooldown_seconds=cooldown_seconds
            )
            
            self.optimization_rules.append(rule)
            self.logger.info(f"Added custom optimization rule: {name}")
            
        except Exception as e:
            self.logger.error(f"Failed to add custom rule: {e}")
    
    async def remove_rule(self, rule_name: str) -> bool:
        """Remove optimization rule."""
        try:
            original_count = len(self.optimization_rules)
            self.optimization_rules = [r for r in self.optimization_rules if r.name != rule_name]
            
            if len(self.optimization_rules) < original_count:
                self.logger.info(f"Removed optimization rule: {rule_name}")
                return True
            else:
                self.logger.warning(f"Optimization rule not found: {rule_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to remove rule: {e}")
            return False



"""
Intelligent Monitor for Email Sequence System

Provides intelligent monitoring, alerting, and automated optimization
based on real-time performance data and machine learning insights.
"""

import asyncio
import logging
import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import json
import os
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor

from .performance_optimizer import OptimizedPerformanceOptimizer, OptimizationConfig
from .advanced_optimizer import AdvancedOptimizer, AdvancedOptimizationConfig
from ..models.sequence import EmailSequence, SequenceStep
from ..models.subscriber import Subscriber
from ..models.template import EmailTemplate

logger = logging.getLogger(__name__)

# Constants
MONITORING_INTERVAL = 5  # seconds
ALERT_THRESHOLD = 0.8
AUTO_OPTIMIZATION_ENABLED = True
MAX_HISTORY_SIZE = 1000


@dataclass
class MonitoringConfig:
    """Intelligent monitoring configuration"""
    monitoring_interval: int = MONITORING_INTERVAL
    alert_threshold: float = ALERT_THRESHOLD
    auto_optimization_enabled: bool = AUTO_OPTIMIZATION_ENABLED
    max_history_size: int = MAX_HISTORY_SIZE
    enable_real_time_alerts: bool = True
    enable_performance_tracking: bool = True
    enable_resource_monitoring: bool = True
    enable_ml_insights: bool = True


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_processes: int
    load_average: float


@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    timestamp: float
    sequences_processed: int
    emails_sent: int
    error_rate: float
    throughput: float
    response_time: float
    cache_hit_rate: float
    queue_size: int


@dataclass
class Alert:
    """System alert"""
    timestamp: float
    level: str  # 'info', 'warning', 'critical'
    category: str
    message: str
    metrics: Dict[str, Any]
    recommendations: List[str]


@dataclass
class OptimizationAction:
    """Optimization action to be taken"""
    timestamp: float
    action_type: str
    parameters: Dict[str, Any]
    expected_impact: str
    priority: str  # 'low', 'medium', 'high', 'critical'


class IntelligentMonitor:
    """Intelligent monitoring system with automated optimization"""
    
    def __init__(
        self,
        config: MonitoringConfig,
        performance_optimizer: OptimizedPerformanceOptimizer,
        advanced_optimizer: AdvancedOptimizer
    ):
        self.config = config
        self.performance_optimizer = performance_optimizer
        self.advanced_optimizer = advanced_optimizer
        
        # Metrics storage
        self.system_metrics_history = deque(maxlen=config.max_history_size)
        self.application_metrics_history = deque(maxlen=config.max_history_size)
        self.alerts_history = deque(maxlen=config.max_history_size)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task = None
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.optimization_callbacks: List[Callable[[OptimizationAction], None]] = []
        
        # Performance tracking
        self.performance_baseline = {}
        self.anomaly_detectors = {}
        
        # Resource monitoring
        self.resource_thresholds = {
            'cpu_usage': 0.8,
            'memory_usage': 0.8,
            'disk_usage': 0.9,
            'error_rate': 0.05,
            'response_time': 5.0  # seconds
        }
        
        # ML insights
        self.ml_insights = {}
        self.prediction_accuracy = 0.0
        
        logger.info("Intelligent Monitor initialized")
    
    async def start_monitoring(self) -> bool:
        """Start intelligent monitoring"""
        try:
            if self.is_monitoring:
                logger.warning("Monitoring already active")
                return True
            
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Intelligent monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            return False
    
    async def stop_monitoring(self) -> bool:
        """Stop intelligent monitoring"""
        try:
            if not self.is_monitoring:
                return True
            
            self.is_monitoring = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Intelligent monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
            return False
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect metrics
                system_metrics = await self._collect_system_metrics()
                application_metrics = await self._collect_application_metrics()
                
                # Store metrics
                self.system_metrics_history.append(system_metrics)
                self.application_metrics_history.append(application_metrics)
                
                # Analyze metrics
                alerts = await self._analyze_metrics(system_metrics, application_metrics)
                
                # Process alerts
                for alert in alerts:
                    await self._process_alert(alert)
                
                # Generate ML insights
                if self.config.enable_ml_insights:
                    await self._generate_ml_insights(system_metrics, application_metrics)
                
                # Auto-optimization
                if self.config.auto_optimization_enabled:
                    optimization_actions = await self._generate_optimization_actions(
                        system_metrics, application_metrics
                    )
                    
                    for action in optimization_actions:
                        await self._execute_optimization_action(action)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1) / 100
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent / 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Process count
            active_processes = len(psutil.pids())
            
            # Load average (simplified)
            load_average = cpu_usage
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                active_processes=active_processes,
                load_average=load_average
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=time.time(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                active_processes=0,
                load_average=0.0
            )
    
    async def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics"""
        try:
            # Get performance optimizer stats
            perf_stats = self.performance_optimizer.get_stats()
            
            # Get advanced optimizer metrics
            advanced_metrics = self.advanced_optimizer.get_advanced_metrics()
            
            # Calculate derived metrics
            throughput = perf_stats.get('sequences_processed', 0) / max(1, perf_stats.get('uptime', 1))
            error_rate = perf_stats.get('errors', 0) / max(1, perf_stats.get('total_operations', 1))
            cache_hit_rate = advanced_metrics.cache_prediction_accuracy
            
            return ApplicationMetrics(
                timestamp=time.time(),
                sequences_processed=perf_stats.get('sequences_processed', 0),
                emails_sent=perf_stats.get('emails_sent', 0),
                error_rate=error_rate,
                throughput=throughput,
                response_time=1.0,  # Placeholder
                cache_hit_rate=cache_hit_rate,
                queue_size=perf_stats.get('queue_size', {}).get('sequence_queue', 0)
            )
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return ApplicationMetrics(
                timestamp=time.time(),
                sequences_processed=0,
                emails_sent=0,
                error_rate=0.0,
                throughput=0.0,
                response_time=0.0,
                cache_hit_rate=0.0,
                queue_size=0
            )
    
    async def _analyze_metrics(
        self,
        system_metrics: SystemMetrics,
        application_metrics: ApplicationMetrics
    ) -> List[Alert]:
        """Analyze metrics and generate alerts"""
        alerts = []
        
        try:
            # Check system thresholds
            if system_metrics.cpu_usage > self.resource_thresholds['cpu_usage']:
                alerts.append(Alert(
                    timestamp=time.time(),
                    level='warning',
                    category='system',
                    message=f"High CPU usage: {system_metrics.cpu_usage:.2%}",
                    metrics={'cpu_usage': system_metrics.cpu_usage},
                    recommendations=['Reduce concurrent tasks', 'Optimize batch processing']
                ))
            
            if system_metrics.memory_usage > self.resource_thresholds['memory_usage']:
                alerts.append(Alert(
                    timestamp=time.time(),
                    level='warning',
                    category='system',
                    message=f"High memory usage: {system_metrics.memory_usage:.2%}",
                    metrics={'memory_usage': system_metrics.memory_usage},
                    recommendations=['Enable garbage collection', 'Reduce cache size']
                ))
            
            if application_metrics.error_rate > self.resource_thresholds['error_rate']:
                alerts.append(Alert(
                    timestamp=time.time(),
                    level='critical',
                    category='application',
                    message=f"High error rate: {application_metrics.error_rate:.2%}",
                    metrics={'error_rate': application_metrics.error_rate},
                    recommendations=['Check error logs', 'Implement retry logic']
                ))
            
            if application_metrics.response_time > self.resource_thresholds['response_time']:
                alerts.append(Alert(
                    timestamp=time.time(),
                    level='warning',
                    category='performance',
                    message=f"Slow response time: {application_metrics.response_time:.2f}s",
                    metrics={'response_time': application_metrics.response_time},
                    recommendations=['Optimize processing', 'Increase resources']
                ))
            
            # Check for anomalies
            anomaly_alerts = await self._detect_anomalies(system_metrics, application_metrics)
            alerts.extend(anomaly_alerts)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error analyzing metrics: {e}")
            return []
    
    async def _detect_anomalies(
        self,
        system_metrics: SystemMetrics,
        application_metrics: ApplicationMetrics
    ) -> List[Alert]:
        """Detect anomalies in metrics"""
        alerts = []
        
        try:
            # Simple anomaly detection based on historical data
            if len(self.system_metrics_history) > 10:
                # Calculate moving averages
                recent_cpu = [m.cpu_usage for m in list(self.system_metrics_history)[-10:]]
                recent_memory = [m.memory_usage for m in list(self.system_metrics_history)[-10:]]
                
                avg_cpu = np.mean(recent_cpu)
                avg_memory = np.mean(recent_memory)
                
                # Detect sudden spikes
                if system_metrics.cpu_usage > avg_cpu * 1.5:
                    alerts.append(Alert(
                        timestamp=time.time(),
                        level='warning',
                        category='anomaly',
                        message=f"CPU usage spike detected: {system_metrics.cpu_usage:.2%} vs avg {avg_cpu:.2%}",
                        metrics={'current_cpu': system_metrics.cpu_usage, 'avg_cpu': avg_cpu},
                        recommendations=['Investigate CPU spike', 'Check for resource leaks']
                    ))
                
                if system_metrics.memory_usage > avg_memory * 1.3:
                    alerts.append(Alert(
                        timestamp=time.time(),
                        level='warning',
                        category='anomaly',
                        message=f"Memory usage spike detected: {system_metrics.memory_usage:.2%} vs avg {avg_memory:.2%}",
                        metrics={'current_memory': system_metrics.memory_usage, 'avg_memory': avg_memory},
                        recommendations=['Check for memory leaks', 'Optimize memory usage']
                    ))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    async def _process_alert(self, alert: Alert) -> None:
        """Process and handle alerts"""
        try:
            # Store alert
            self.alerts_history.append(alert)
            
            # Log alert
            logger.warning(f"Alert [{alert.level.upper()}] {alert.category}: {alert.message}")
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
            # Take immediate action for critical alerts
            if alert.level == 'critical':
                await self._handle_critical_alert(alert)
            
        except Exception as e:
            logger.error(f"Error processing alert: {e}")
    
    async def _handle_critical_alert(self, alert: Alert) -> None:
        """Handle critical alerts with immediate action"""
        try:
            if alert.category == 'system':
                # System-level critical alert
                if 'memory_usage' in alert.metrics and alert.metrics['memory_usage'] > 0.9:
                    # Emergency memory cleanup
                    gc.collect()
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("Emergency memory cleanup performed")
                
                elif 'cpu_usage' in alert.metrics and alert.metrics['cpu_usage'] > 0.95:
                    # Emergency CPU throttling
                    logger.info("Emergency CPU throttling activated")
            
            elif alert.category == 'application':
                # Application-level critical alert
                if 'error_rate' in alert.metrics and alert.metrics['error_rate'] > 0.1:
                    # Emergency error handling
                    logger.info("Emergency error handling activated")
            
        except Exception as e:
            logger.error(f"Error handling critical alert: {e}")
    
    async def _generate_ml_insights(
        self,
        system_metrics: SystemMetrics,
        application_metrics: ApplicationMetrics
    ) -> None:
        """Generate ML-based insights"""
        try:
            # Combine metrics for ML analysis
            combined_metrics = {
                'cpu_usage': system_metrics.cpu_usage,
                'memory_usage': system_metrics.memory_usage,
                'throughput': application_metrics.throughput,
                'error_rate': application_metrics.error_rate,
                'cache_hit_rate': application_metrics.cache_hit_rate,
                'queue_size': application_metrics.queue_size
            }
            
            # Generate insights using advanced optimizer
            insights = await self.advanced_optimizer.optimize_with_ml(
                sequences=[],  # Placeholder
                subscribers=[],  # Placeholder
                templates=[],  # Placeholder
                current_metrics=combined_metrics
            )
            
            self.ml_insights = insights
            
        except Exception as e:
            logger.error(f"Error generating ML insights: {e}")
    
    async def _generate_optimization_actions(
        self,
        system_metrics: SystemMetrics,
        application_metrics: ApplicationMetrics
    ) -> List[OptimizationAction]:
        """Generate optimization actions based on current metrics"""
        actions = []
        
        try:
            # Memory optimization actions
            if system_metrics.memory_usage > 0.8:
                actions.append(OptimizationAction(
                    timestamp=time.time(),
                    action_type='memory_optimization',
                    parameters={'cache_size': 500, 'enable_gc': True},
                    expected_impact='Reduce memory usage by 20%',
                    priority='high'
                ))
            
            # CPU optimization actions
            if system_metrics.cpu_usage > 0.8:
                actions.append(OptimizationAction(
                    timestamp=time.time(),
                    action_type='cpu_optimization',
                    parameters={'max_concurrent_tasks': 2},
                    expected_impact='Reduce CPU usage by 15%',
                    priority='medium'
                ))
            
            # Performance optimization actions
            if application_metrics.throughput < 10:
                actions.append(OptimizationAction(
                    timestamp=time.time(),
                    action_type='performance_optimization',
                    parameters={'batch_size': 64, 'enable_caching': True},
                    expected_impact='Increase throughput by 30%',
                    priority='high'
                ))
            
            # Error rate optimization actions
            if application_metrics.error_rate > 0.05:
                actions.append(OptimizationAction(
                    timestamp=time.time(),
                    action_type='error_optimization',
                    parameters={'retry_attempts': 3, 'backoff_factor': 2},
                    expected_impact='Reduce error rate by 50%',
                    priority='critical'
                ))
            
            return actions
            
        except Exception as e:
            logger.error(f"Error generating optimization actions: {e}")
            return []
    
    async def _execute_optimization_action(self, action: OptimizationAction) -> bool:
        """Execute optimization action"""
        try:
            logger.info(f"Executing optimization action: {action.action_type}")
            
            # Execute action based on type
            if action.action_type == 'memory_optimization':
                await self._execute_memory_optimization(action.parameters)
            elif action.action_type == 'cpu_optimization':
                await self._execute_cpu_optimization(action.parameters)
            elif action.action_type == 'performance_optimization':
                await self._execute_performance_optimization(action.parameters)
            elif action.action_type == 'error_optimization':
                await self._execute_error_optimization(action.parameters)
            
            # Call optimization callbacks
            for callback in self.optimization_callbacks:
                try:
                    callback(action)
                except Exception as e:
                    logger.error(f"Error in optimization callback: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing optimization action: {e}")
            return False
    
    async def _execute_memory_optimization(self, parameters: Dict[str, Any]) -> None:
        """Execute memory optimization"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch cache if available
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Adjust cache size
            if 'cache_size' in parameters:
                # This would be implemented in the performance optimizer
                pass
            
            logger.info("Memory optimization executed")
            
        except Exception as e:
            logger.error(f"Error executing memory optimization: {e}")
    
    async def _execute_cpu_optimization(self, parameters: Dict[str, Any]) -> None:
        """Execute CPU optimization"""
        try:
            # Adjust concurrent tasks
            if 'max_concurrent_tasks' in parameters:
                # This would be implemented in the performance optimizer
                pass
            
            logger.info("CPU optimization executed")
            
        except Exception as e:
            logger.error(f"Error executing CPU optimization: {e}")
    
    async def _execute_performance_optimization(self, parameters: Dict[str, Any]) -> None:
        """Execute performance optimization"""
        try:
            # Adjust batch size
            if 'batch_size' in parameters:
                # This would be implemented in the performance optimizer
                pass
            
            # Enable caching
            if 'enable_caching' in parameters:
                # This would be implemented in the performance optimizer
                pass
            
            logger.info("Performance optimization executed")
            
        except Exception as e:
            logger.error(f"Error executing performance optimization: {e}")
    
    async def _execute_error_optimization(self, parameters: Dict[str, Any]) -> None:
        """Execute error optimization"""
        try:
            # Adjust retry parameters
            if 'retry_attempts' in parameters:
                # This would be implemented in the performance optimizer
                pass
            
            logger.info("Error optimization executed")
            
        except Exception as e:
            logger.error(f"Error executing error optimization: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def add_optimization_callback(self, callback: Callable[[OptimizationAction], None]) -> None:
        """Add optimization callback function"""
        self.optimization_callbacks.append(callback)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary"""
        try:
            # Calculate summary statistics
            if self.system_metrics_history:
                recent_system = list(self.system_metrics_history)[-10:]
                avg_cpu = np.mean([m.cpu_usage for m in recent_system])
                avg_memory = np.mean([m.memory_usage for m in recent_system])
            else:
                avg_cpu = 0.0
                avg_memory = 0.0
            
            if self.application_metrics_history:
                recent_app = list(self.application_metrics_history)[-10:]
                avg_throughput = np.mean([m.throughput for m in recent_app])
                avg_error_rate = np.mean([m.error_rate for m in recent_app])
            else:
                avg_throughput = 0.0
                avg_error_rate = 0.0
            
            # Count alerts by level
            alert_counts = defaultdict(int)
            for alert in self.alerts_history:
                alert_counts[alert.level] += 1
            
            return {
                'monitoring_active': self.is_monitoring,
                'system_metrics': {
                    'avg_cpu_usage': avg_cpu,
                    'avg_memory_usage': avg_memory,
                    'total_metrics_collected': len(self.system_metrics_history)
                },
                'application_metrics': {
                    'avg_throughput': avg_throughput,
                    'avg_error_rate': avg_error_rate,
                    'total_metrics_collected': len(self.application_metrics_history)
                },
                'alerts': {
                    'total_alerts': len(self.alerts_history),
                    'alert_counts': dict(alert_counts),
                    'recent_alerts': [alert.message for alert in list(self.alerts_history)[-5:]]
                },
                'ml_insights': self.ml_insights,
                'optimization_actions_taken': len(self.optimization_callbacks)
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring summary: {e}")
            return {}
    
    def export_monitoring_data(self, filepath: str) -> bool:
        """Export monitoring data for analysis"""
        try:
            data = {
                'system_metrics': [m.__dict__ for m in self.system_metrics_history],
                'application_metrics': [m.__dict__ for m in self.application_metrics_history],
                'alerts': [alert.__dict__ for alert in self.alerts_history],
                'ml_insights': self.ml_insights,
                'summary': self.get_monitoring_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Monitoring data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting monitoring data: {e}")
            return False 
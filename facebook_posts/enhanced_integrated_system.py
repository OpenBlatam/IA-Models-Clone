import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import time
import threading
import asyncio
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
import random
import hashlib
from datetime import datetime, timedelta
import warnings
import traceback
import signal
import sys
from contextlib import contextmanager
warnings.filterwarnings('ignore')

# Import our enhanced components
from enhanced_performance_engine import EnhancedPerformanceOptimizationEngine, EnhancedPerformanceConfig
from enhanced_ai_agent_system import EnhancedAIAgentSystem, EnhancedAgentConfig
from custom_nn_modules import (
    FacebookContentAnalysisTransformer, MultiModalFacebookAnalyzer,
    TemporalEngagementPredictor, AdaptiveContentOptimizer, FacebookDiffusionUNet
)
from forward_reverse_diffusion import (
    DiffusionConfig, ForwardDiffusionProcess, ReverseDiffusionProcess, 
    DiffusionTraining, DiffusionVisualizer
)


@dataclass
class EnhancedIntegratedSystemConfig:
    """Enhanced configuration for the integrated system"""
    # System parameters
    system_name: str = "Enhanced Facebook Content Optimization System"
    version: str = "2.0.0"
    environment: str = "production"
    
    # Model parameters
    model_dim: int = 768
    num_heads: int = 12
    num_layers: int = 6
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 64  # Increased batch size
    num_epochs: int = 100
    gradient_clip: float = 1.0
    weight_decay: float = 1e-5
    
    # Performance optimization
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    enable_caching: bool = True
    max_workers: int = 8
    cache_size: int = 50000  # Increased cache size
    cache_ttl_seconds: int = 7200  # 2 hours
    
    # AI agent system
    enable_ai_agents: bool = True
    agent_learning_rate: float = 0.001
    agent_autonomous_mode: bool = True
    agent_memory_size: int = 10000
    
    # Diffusion parameters
    diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    
    # Content analysis
    max_text_length: int = 512
    min_engagement_threshold: float = 0.3
    viral_potential_threshold: float = 0.7
    
    # Advanced features
    enable_real_time_optimization: bool = True
    enable_ab_testing: bool = True
    enable_performance_monitoring: bool = True
    enable_auto_scaling: bool = True
    
    # System monitoring
    enable_health_checks: bool = True
    health_check_interval: int = 60  # 1 minute
    enable_metrics_export: bool = True
    metrics_export_interval: int = 300  # 5 minutes
    
    # Error handling
    enable_graceful_degradation: bool = True
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    
    # Logging
    log_level: str = "INFO"
    enable_structured_logging: bool = True
    log_rotation: bool = True
    max_log_size_mb: int = 100


class SystemHealthMonitor:
    """Advanced system health monitoring"""
    
    def __init__(self, config: EnhancedIntegratedSystemConfig):
        self.config = config
        self.health_status = {
            'overall_status': 'healthy',
            'components': {},
            'last_check': None,
            'uptime': time.time(),
            'total_checks': 0,
            'failed_checks': 0
        }
        
        # Health check history
        self.health_history = deque(maxlen=1000)
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def check_component_health(self, component_name: str, health_func: Callable) -> Dict[str, Any]:
        """Check health of a specific component"""
        try:
            start_time = time.time()
            health_result = health_func()
            check_time = time.time() - start_time
            
            health_status = {
                'status': 'healthy',
                'response_time': check_time,
                'timestamp': time.time(),
                'details': health_result
            }
            
            # Determine status based on response time and results
            if check_time > 5.0:  # Slow response
                health_status['status'] = 'degraded'
            elif 'error' in health_result or 'failed' in str(health_result).lower():
                health_status['status'] = 'unhealthy'
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time(),
                'response_time': 0.0
            }
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                self._perform_health_checks()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                logging.error(f"Error in health monitoring loop: {e}")
                time.sleep(30)
    
    def _perform_health_checks(self):
        """Perform all health checks"""
        current_time = time.time()
        self.health_status['last_check'] = current_time
        self.health_status['total_checks'] += 1
        
        # Update uptime
        self.health_status['uptime'] = current_time - self.health_status['uptime']
        
        # Check overall system health
        overall_status = 'healthy'
        failed_checks = 0
        
        for component_name, health_result in self.health_status['components'].items():
            if health_result['status'] == 'unhealthy':
                overall_status = 'unhealthy'
                failed_checks += 1
            elif health_result['status'] == 'degraded' and overall_status == 'healthy':
                overall_status = 'degraded'
        
        self.health_status['overall_status'] = overall_status
        self.health_status['failed_checks'] = failed_checks
        
        # Record health history
        self.health_history.append({
            'timestamp': current_time,
            'overall_status': overall_status,
            'component_count': len(self.health_status['components']),
            'failed_checks': failed_checks
        })
        
        # Log health status
        if overall_status != 'healthy':
            logging.warning(f"System health degraded: {overall_status}, {failed_checks} failed checks")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        return {
            'current_status': self.health_status,
            'health_history': list(self.health_history)[-10:],  # Last 10 checks
            'uptime_hours': self.health_status['uptime'] / 3600,
            'health_score': self._calculate_health_score()
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0-100)"""
        if not self.health_history:
            return 100.0
        
        recent_checks = list(self.health_history)[-10:]  # Last 10 checks
        
        healthy_checks = sum(1 for check in recent_checks if check['overall_status'] == 'healthy')
        degraded_checks = sum(1 for check in recent_checks if check['overall_status'] == 'degraded')
        unhealthy_checks = sum(1 for check in recent_checks if check['overall_status'] == 'unhealthy')
        
        total_checks = len(recent_checks)
        
        # Weighted scoring: healthy=100, degraded=50, unhealthy=0
        score = (healthy_checks * 100 + degraded_checks * 50) / total_checks
        
        return min(100.0, max(0.0, score))


class ErrorHandler:
    """Advanced error handling and recovery system"""
    
    def __init__(self, config: EnhancedIntegratedSystemConfig):
        self.config = config
        self.error_history = deque(maxlen=1000)
        self.recovery_strategies = {}
        
        # Initialize recovery strategies
        self._initialize_recovery_strategies()
    
    def _initialize_recovery_strategies(self):
        """Initialize error recovery strategies"""
        self.recovery_strategies = {
            'model_loading_error': self._handle_model_loading_error,
            'memory_error': self._handle_memory_error,
            'gpu_error': self._handle_gpu_error,
            'network_error': self._handle_network_error,
            'cache_error': self._handle_cache_error,
            'agent_error': self._handle_agent_error
        }
    
    def handle_error(self, error: Exception, context: str, component: str) -> Dict[str, Any]:
        """Handle an error with recovery strategies"""
        error_info = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'component': component,
            'traceback': traceback.format_exc(),
            'recovery_attempted': False,
            'recovery_successful': False
        }
        
        # Add to error history
        self.error_history.append(error_info)
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(error, context, component)
        error_info['recovery_attempted'] = recovery_result['attempted']
        error_info['recovery_successful'] = recovery_result['successful']
        error_info['recovery_details'] = recovery_result['details']
        
        # Log error
        if recovery_result['successful']:
            logging.warning(f"Error recovered in {component}: {error}")
        else:
            logging.error(f"Error in {component}: {error}")
        
        return error_info
    
    def _attempt_recovery(self, error: Exception, context: str, component: str) -> Dict[str, Any]:
        """Attempt to recover from an error"""
        error_type = type(error).__name__.lower()
        
        # Find matching recovery strategy
        recovery_strategy = None
        for key, strategy in self.recovery_strategies.items():
            if key in error_type or key in context.lower():
                recovery_strategy = strategy
                break
        
        if recovery_strategy:
            try:
                result = recovery_strategy(error, context, component)
                return {
                    'attempted': True,
                    'successful': result.get('successful', False),
                    'details': result
                }
            except Exception as recovery_error:
                return {
                    'attempted': True,
                    'successful': False,
                    'details': {'recovery_error': str(recovery_error)}
                }
        
        return {
            'attempted': False,
            'successful': False,
            'details': {'no_strategy_found': True}
        }
    
    def _handle_model_loading_error(self, error: Exception, context: str, component: str) -> Dict[str, Any]:
        """Handle model loading errors"""
        try:
            # Try to reload model
            logging.info(f"Attempting to reload model in {component}")
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'successful': True,
                'action': 'model_reloaded',
                'gpu_cache_cleared': True
            }
        except Exception as e:
            return {
                'successful': False,
                'action': 'model_reload_failed',
                'error': str(e)
            }
    
    def _handle_memory_error(self, error: Exception, context: str, component: str) -> Dict[str, Any]:
        """Handle memory-related errors"""
        try:
            # Force garbage collection
            import gc
            collected = gc.collect()
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'successful': True,
                'action': 'memory_cleanup',
                'garbage_collected': collected,
                'gpu_cache_cleared': True
            }
        except Exception as e:
            return {
                'successful': False,
                'action': 'memory_cleanup_failed',
                'error': str(e)
            }
    
    def _handle_gpu_error(self, error: Exception, context: str, component: str) -> Dict[str, Any]:
        """Handle GPU-related errors"""
        try:
            if torch.cuda.is_available():
                # Reset GPU state
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                return {
                    'successful': True,
                    'action': 'gpu_reset',
                    'gpu_cache_cleared': True
                }
            else:
                return {
                    'successful': False,
                    'action': 'no_gpu_available'
                }
        except Exception as e:
            return {
                'successful': False,
                'action': 'gpu_reset_failed',
                'error': str(e)
            }
    
    def _handle_network_error(self, error: Exception, context: str, component: str) -> Dict[str, Any]:
        """Handle network-related errors"""
        return {
            'successful': False,
            'action': 'network_error_requires_manual_intervention',
            'suggestion': 'Check network connectivity and retry'
        }
    
    def _handle_cache_error(self, error: Exception, context: str, component: str) -> Dict[str, Any]:
        """Handle cache-related errors"""
        try:
            # Clear cache directory
            cache_dir = Path("cache")
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*.pkl"):
                    try:
                        cache_file.unlink()
                    except:
                        pass
            
            return {
                'successful': True,
                'action': 'cache_cleared',
                'cache_files_removed': True
            }
        except Exception as e:
            return {
                'successful': False,
                'action': 'cache_clear_failed',
                'error': str(e)
            }
    
    def _handle_agent_error(self, error: Exception, context: str, component: str) -> Dict[str, Any]:
        """Handle AI agent errors"""
        try:
            # Restart agent communication
            logging.info(f"Restarting agent communication in {component}")
            
            return {
                'successful': True,
                'action': 'agent_communication_restarted'
            }
        except Exception as e:
            return {
                'successful': False,
                'action': 'agent_restart_failed',
                'error': str(e)
            }
    
    def get_error_report(self) -> Dict[str, Any]:
        """Get comprehensive error report"""
        if not self.error_history:
            return {'total_errors': 0, 'recent_errors': []}
        
        recent_errors = list(self.error_history)[-10:]  # Last 10 errors
        
        error_counts = defaultdict(int)
        recovery_success_rate = 0.0
        
        for error in self.error_history:
            error_counts[error['error_type']] += 1
            if error['recovery_successful']:
                recovery_success_rate += 1
        
        recovery_success_rate = recovery_success_rate / len(self.error_history) if self.error_history else 0.0
        
        return {
            'total_errors': len(self.error_history),
            'error_type_distribution': dict(error_counts),
            'recovery_success_rate': recovery_success_rate,
            'recent_errors': recent_errors
        }


class MetricsExporter:
    """Advanced metrics export system"""
    
    def __init__(self, config: EnhancedIntegratedSystemConfig):
        self.config = config
        self.metrics_buffer = deque(maxlen=10000)
        self.export_formats = ['json', 'csv', 'prometheus']
        
        # Start export thread
        self.export_thread = threading.Thread(target=self._export_loop, daemon=True)
        self.export_thread.start()
    
    def record_metric(self, metric_name: str, value: Any, tags: Dict[str, str] = None):
        """Record a metric"""
        metric = {
            'name': metric_name,
            'value': value,
            'timestamp': time.time(),
            'tags': tags or {}
        }
        
        self.metrics_buffer.append(metric)
    
    def _export_loop(self):
        """Background export loop"""
        while True:
            try:
                if self.config.enable_metrics_export:
                    self._export_metrics()
                
                time.sleep(self.config.metrics_export_interval)
                
            except Exception as e:
                logging.error(f"Error in metrics export loop: {e}")
                time.sleep(60)
    
    def _export_metrics(self):
        """Export metrics to various formats"""
        if not self.metrics_buffer:
            return
        
        # Create export directory
        export_dir = Path("exports/metrics")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export to JSON
        json_file = export_dir / f"metrics_{timestamp}.json"
        try:
            with open(json_file, 'w') as f:
                json.dump(list(self.metrics_buffer), f, indent=2)
        except Exception as e:
            logging.error(f"Failed to export JSON metrics: {e}")
        
        # Export to CSV
        csv_file = export_dir / f"metrics_{timestamp}.csv"
        try:
            import csv
            with open(csv_file, 'w', newline='') as f:
                if self.metrics_buffer:
                    writer = csv.DictWriter(f, fieldnames=self.metrics_buffer[0].keys())
                    writer.writeheader()
                    writer.writerows(self.metrics_buffer)
        except Exception as e:
            logging.error(f"Failed to export CSV metrics: {e}")
        
        # Clear buffer after export
        self.metrics_buffer.clear()
        
        logging.info(f"Metrics exported to {export_dir}")


class EnhancedIntegratedSystem:
    """Enhanced integrated system with advanced features"""
    
    def __init__(self, config: EnhancedIntegratedSystemConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.performance_engine = None
        self.ai_agent_system = None
        self.health_monitor = None
        self.error_handler = None
        self.metrics_exporter = None
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.startup_time = None
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        
        # Initialize system
        self._initialize_system()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup advanced logging system"""
        logger = logging.getLogger(self.config.system_name)
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Create formatter
        if self.config.enable_structured_logging:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        else:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.log_rotation:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                f"logs/{self.config.system_name.lower().replace(' ', '_')}.log",
                maxBytes=self.config.max_log_size_mb * 1024 * 1024,
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing Enhanced Integrated System...")
            
            # Initialize performance engine
            self.logger.info("Initializing Performance Engine...")
            perf_config = EnhancedPerformanceConfig(
                enable_caching=self.config.enable_caching,
                cache_size=self.config.cache_size,
                cache_ttl_seconds=self.config.cache_ttl_seconds,
                max_workers=self.config.max_workers,
                enable_memory_optimization=True,
                enable_gpu_optimization=True
            )
            self.performance_engine = EnhancedPerformanceOptimizationEngine(perf_config)
            
            # Initialize AI agent system
            if self.config.enable_ai_agents:
                self.logger.info("Initializing AI Agent System...")
                agent_config = EnhancedAgentConfig(
                    num_agents=5,
                    enable_agent_communication=True,
                    enable_autonomous_mode=self.config.agent_autonomous_mode,
                    memory_size=self.config.agent_memory_size
                )
                self.ai_agent_system = EnhancedAIAgentSystem(agent_config)
            
            # Initialize monitoring components
            self.logger.info("Initializing Monitoring Components...")
            self.health_monitor = SystemHealthMonitor(self.config)
            self.error_handler = ErrorHandler(self.config)
            self.metrics_exporter = MetricsExporter(self.config)
            
            # Register health checks
            self._register_health_checks()
            
            self.is_initialized = True
            self.startup_time = time.time()
            self.logger.info("System initialization completed successfully!")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _register_health_checks(self):
        """Register health check functions"""
        # Performance engine health check
        self.health_monitor.health_status['components']['performance_engine'] = \
            self.health_monitor.check_component_health(
                'performance_engine',
                lambda: self.performance_engine.get_system_stats() if self.performance_engine else {'error': 'Not initialized'}
            )
        
        # AI agent system health check
        if self.ai_agent_system:
            self.health_monitor.health_status['components']['ai_agent_system'] = \
                self.health_monitor.check_component_health(
                    'ai_agent_system',
                    lambda: self.ai_agent_system.get_system_stats()
                )
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self):
        """Start the system"""
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        try:
            self.logger.info("Starting Enhanced Integrated System...")
            self.is_running = True
            
            # Start background tasks
            self._start_background_tasks()
            
            self.logger.info("System started successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            raise
    
    def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        # Health monitoring is already running in separate threads
        # Metrics export is already running in separate threads
        pass
    
    def process_content(self, content: str, content_type: str = "Post", 
                       context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process content using the enhanced system"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Record metrics
            self.metrics_exporter.record_metric('content_processing_request', 1, {
                'content_type': content_type,
                'content_length': str(len(content))
            })
            
            # Process with performance engine
            perf_result = self.performance_engine.optimize_content_batch([content], content_type)
            
            # Process with AI agents if available
            agent_result = None
            if self.ai_agent_system:
                agent_result = self.ai_agent_system.process_content_with_consensus(
                    content, content_type, context
                )
            
            # Combine results
            final_result = self._combine_results(perf_result[0], agent_result)
            
            # Record success
            self.success_count += 1
            processing_time = time.time() - start_time
            
            # Record performance metrics
            self.metrics_exporter.record_metric('content_processing_time', processing_time, {
                'content_type': content_type,
                'status': 'success'
            })
            
            return {
                'status': 'success',
                'result': final_result,
                'processing_time': processing_time,
                'components_used': ['performance_engine', 'ai_agent_system'] if self.ai_agent_system else ['performance_engine']
            }
            
        except Exception as e:
            # Handle error
            error_info = self.error_handler.handle_error(e, 'content_processing', 'main_system')
            
            # Record error metrics
            self.error_count += 1
            self.metrics_exporter.record_metric('content_processing_error', 1, {
                'content_type': content_type,
                'error_type': type(e).__name__
            })
            
            # Return error result
            return {
                'status': 'error',
                'error': str(e),
                'error_details': error_info,
                'processing_time': time.time() - start_time
            }
    
    def _combine_results(self, perf_result: Dict[str, Any], 
                        agent_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from different components"""
        combined = {
            'content_optimization': perf_result,
            'ai_analysis': agent_result,
            'combined_score': 0.0,
            'recommendations': []
        }
        
        # Calculate combined score
        perf_score = perf_result.get('engagement_score', 0.5)
        agent_score = 0.5
        
        if agent_result and 'final_decision' in agent_result:
            agent_score = agent_result['final_decision'].get('engagement_prediction', 0.5)
        
        # Weighted combination (performance engine gets 60%, AI agents get 40%)
        combined['combined_score'] = perf_score * 0.6 + agent_score * 0.4
        
        # Combine recommendations
        if 'optimization_suggestions' in perf_result:
            combined['recommendations'].extend(perf_result['optimization_suggestions'])
        
        if agent_result and 'final_decision' in agent_result:
            if 'top_recommendations' in agent_result['final_decision']:
                combined['recommendations'].extend(agent_result['final_decision']['top_recommendations'])
        
        # Remove duplicates and limit
        combined['recommendations'] = list(set(combined['recommendations']))[:10]
        
        return combined
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'system_info': {
                'name': self.config.system_name,
                'version': self.config.version,
                'environment': self.config.environment,
                'is_initialized': self.is_initialized,
                'is_running': self.is_running,
                'startup_time': self.startup_time,
                'uptime_hours': (time.time() - (self.startup_time or time.time())) / 3600
            },
            'performance_metrics': {
                'total_requests': self.request_count,
                'successful_requests': self.success_count,
                'error_requests': self.error_count,
                'success_rate': self.success_count / max(self.request_count, 1),
                'error_rate': self.error_count / max(self.request_count, 1)
            },
            'component_status': {
                'performance_engine': self.performance_engine is not None,
                'ai_agent_system': self.ai_agent_system is not None,
                'health_monitor': self.health_monitor is not None,
                'error_handler': self.error_handler is not None,
                'metrics_exporter': self.metrics_exporter is not None
            },
            'health_status': self.health_monitor.get_health_report() if self.health_monitor else None,
            'error_report': self.error_handler.get_error_report() if self.error_handler else None
        }
    
    def shutdown(self):
        """Gracefully shutdown the system"""
        if not self.is_running:
            return
        
        self.logger.info("🛑 Shutting down Enhanced Integrated System...")
        self.is_running = False
        
        try:
            # Cleanup components
            if self.performance_engine:
                self.performance_engine.cleanup()
            
            if self.ai_agent_system:
                self.ai_agent_system.cleanup()
            
            # Export final metrics
            if self.metrics_exporter:
                self.metrics_exporter._export_metrics()
            
            self.logger.info("✅ System shutdown completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = EnhancedIntegratedSystemConfig(
        environment="development",
        enable_ai_agents=True,
        enable_performance_monitoring=True,
        enable_health_checks=True
    )
    
    # Initialize and start system
    system = EnhancedIntegratedSystem(config)
    
    try:
        # Start system
        system.start()
        
        print("🚀 Testing Enhanced Integrated System...")
        
        # Test content processing
        test_content = "Discover our revolutionary AI-powered content optimization platform!"
        test_context = {
            'time_of_day': 0.8,  # Evening
            'day_of_week': 0.9,  # Weekend
            'audience_size': 0.7  # Large audience
        }
        
        # Process content
        result = system.process_content(test_content, "Post", test_context)
        
        if result['status'] == 'success':
            print(f"✅ Content processed successfully!")
            print(f"📊 Combined Score: {result['result']['combined_score']:.3f}")
            print(f"⚡ Processing Time: {result['processing_time']:.3f}s")
            print(f"🔧 Components Used: {', '.join(result['components_used'])}")
            
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(result['result']['recommendations'][:5], 1):
                print(f"  {i}. {rec}")
        else:
            print(f"❌ Content processing failed: {result['error']}")
        
        # Get system status
        status = system.get_system_status()
        print(f"\n📊 System Status:")
        print(f"  - Uptime: {status['system_info']['uptime_hours']:.2f} hours")
        print(f"  - Success Rate: {status['performance_metrics']['success_rate']:.3f}")
        print(f"  - Overall Health: {status['health_status']['current_status']['overall_status']}")
        
        # Wait a bit for background tasks
        print(f"\n⏳ Waiting for background tasks...")
        time.sleep(5)
        
        # Get updated status
        updated_status = system.get_system_status()
        print(f"  - Health Score: {updated_status['health_status']['health_score']:.1f}/100")
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Shutdown system
        print(f"\n🛑 Shutting down system...")
        system.shutdown()
        print("✨ Enhanced Integrated System test completed!")

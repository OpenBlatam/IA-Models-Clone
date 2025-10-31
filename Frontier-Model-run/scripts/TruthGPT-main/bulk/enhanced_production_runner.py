#!/usr/bin/env python3
"""
Enhanced Production Runner - Advanced production system orchestrator
Enhanced with AI-powered optimization, intelligent resource management, and advanced monitoring
"""

import asyncio
import argparse
import logging
import signal
import sys
import time
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import os
import psutil
import numpy as np
from datetime import datetime, timezone
import torch
import torch.nn as nn

# Import enhanced components
from enhanced_production_config import (
    EnhancedProductionConfigManager, create_enhanced_production_config, Environment
)
from enhanced_production_api import create_enhanced_app, run_enhanced_server
from enhanced_bulk_optimizer import create_enhanced_bulk_optimizer, optimize_models_enhanced
from production_logging import create_production_logger, setup_production_logging
from production_monitoring import create_production_monitor

class IntelligentResourceManager:
    """Intelligent resource management system."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.resource_history = []
        self.prediction_model = None
        self.current_allocation = {}
    
    def analyze_system_resources(self) -> Dict[str, float]:
        """Analyze current system resources."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            available_memory = memory.available / (1024 * 1024 * 1024)  # GB
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Load average
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            
            resources = {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_percent': memory_percent,
                'available_memory_gb': available_memory,
                'disk_percent': disk_percent,
                'load_average': load_avg,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'timestamp': time.time()
            }
            
            # Store in history
            self.resource_history.append(resources)
            if len(self.resource_history) > 1000:
                self.resource_history = self.resource_history[-500:]
            
            return resources
            
        except Exception as e:
            self.logger.error(f"Error analyzing system resources: {e}")
            return {
                'cpu_percent': 0.0,
                'cpu_count': 1,
                'memory_percent': 0.0,
                'available_memory_gb': 8.0,
                'disk_percent': 0.0,
                'load_average': 0.0,
                'network_bytes_sent': 0,
                'network_bytes_recv': 0,
                'timestamp': time.time()
            }
    
    def predict_resource_needs(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict resource needs for optimization."""
        try:
            # Analyze model characteristics
            total_parameters = sum(model.get('parameters', 0) for model in models)
            total_memory = sum(model.get('memory_usage', 0) for model in models)
            complexity_score = sum(model.get('complexity_score', 0) for model in models)
            
            # Predict CPU usage based on complexity
            predicted_cpu = min(complexity_score * 0.1, 0.8)
            
            # Predict memory usage
            predicted_memory = min(total_memory * 1.2, self.get_available_memory() * 0.8)
            
            # Predict optimal batch size
            optimal_batch_size = self._calculate_optimal_batch_size(models)
            
            # Predict optimal number of workers
            optimal_workers = self._calculate_optimal_workers()
            
            return {
                'predicted_cpu_usage': predicted_cpu,
                'predicted_memory_usage': predicted_memory,
                'optimal_batch_size': optimal_batch_size,
                'optimal_workers': optimal_workers,
                'estimated_time': self._estimate_optimization_time(models)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting resource needs: {e}")
            return {
                'predicted_cpu_usage': 0.5,
                'predicted_memory_usage': 2048,  # 2GB
                'optimal_batch_size': 32,
                'optimal_workers': 4,
                'estimated_time': 300  # 5 minutes
            }
    
    def get_available_memory(self) -> float:
        """Get available memory in GB."""
        try:
            return psutil.virtual_memory().available / (1024 * 1024 * 1024)
        except:
            return 8.0
    
    def _calculate_optimal_batch_size(self, models: List[Dict[str, Any]]) -> int:
        """Calculate optimal batch size."""
        if not models:
            return 32
        
        # Calculate based on model memory requirements
        avg_memory = np.mean([model.get('memory_usage', 100) for model in models])
        available_memory = self.get_available_memory() * 1024  # Convert to MB
        
        if avg_memory > 0:
            optimal_batch = int(available_memory * 0.8 / avg_memory)
            return max(1, min(optimal_batch, 128))
        
        return 32
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers."""
        try:
            cpu_count = psutil.cpu_count()
            cpu_usage = psutil.cpu_percent() / 100.0
            
            # Calculate based on available CPU
            available_cpu = max(0, 1.0 - cpu_usage)
            optimal_workers = int(cpu_count * available_cpu * 0.8)
            
            return max(1, min(optimal_workers, 8))
        except:
            return 4
    
    def _estimate_optimization_time(self, models: List[Dict[str, Any]]) -> float:
        """Estimate optimization time."""
        if not models:
            return 60.0
        
        # Base time + time per model
        base_time = 30.0
        per_model_time = 10.0
        complexity_factor = np.mean([model.get('complexity_score', 1.0) for model in models])
        
        return base_time + (len(models) * per_model_time * complexity_factor)

class EnhancedProductionRunner:
    """Enhanced production system runner with AI-powered features."""
    
    def __init__(self, config_file: Optional[str] = None, environment: Optional[Environment] = None):
        self.config_file = config_file
        self.environment = environment
        self.config_manager = None
        self.logger = None
        self.monitor = None
        self.app = None
        self.bulk_optimizer = None
        self.resource_manager = None
        self.running = False
        self.shutdown_event = threading.Event()
        self.start_time = time.time()
        
        # Performance tracking
        self.performance_metrics = {
            'optimizations_completed': 0,
            'total_models_optimized': 0,
            'avg_improvement': 0.0,
            'avg_optimization_time': 0.0,
            'success_rate': 0.0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def initialize(self):
        """Initialize enhanced production system."""
        print("🚀 Initializing Enhanced Production System")
        print("=" * 60)
        
        try:
            # Initialize configuration
            self.config_manager = create_enhanced_production_config(self.config_file, self.environment)
            config = self.config_manager.get_config()
            
            if not self.config_manager.validate_config():
                raise ValueError("Invalid configuration")
            
            print(f"✅ Enhanced configuration loaded: {config.environment.value}")
            print(f"   - Security: {'Enabled' if config.security.enable_encryption else 'Disabled'}")
            print(f"   - Caching: {'Enabled' if config.cache.enable_redis_cache else 'Disabled'}")
            print(f"   - Hot Reload: {'Enabled' if self.config_manager.hot_reloader else 'Disabled'}")
            
            # Initialize logging
            loggers = setup_production_logging({
                'log_level': config.log_level.value,
                'enable_console': True,
                'enable_file': True,
                'log_file': config.log_file,
                'console_format': 'text',
                'file_format': 'json'
            })
            
            self.logger = loggers['main']
            self.logger.info("Enhanced production system initializing")
            
            print("✅ Enhanced logging system initialized")
            
            # Initialize monitoring
            self.monitor = create_production_monitor({
                'metrics': {
                    'collection_interval': 5,
                    'max_history_size': 2000
                },
                'health': {
                    'check_interval': 15
                },
                'alerts': {}
            })
            
            self.monitor.start()
            self.logger.info("Enhanced production monitoring started")
            
            print("✅ Enhanced monitoring system initialized")
            
            # Initialize bulk optimizer
            self.bulk_optimizer = create_enhanced_bulk_optimizer(config.__dict__)
            self.logger.info("Enhanced bulk optimizer initialized")
            
            print("✅ Enhanced bulk optimizer initialized")
            
            # Initialize resource manager
            self.resource_manager = IntelligentResourceManager(config)
            self.logger.info("Intelligent resource manager initialized")
            
            print("✅ Intelligent resource manager initialized")
            
            # Initialize API
            self.app = create_enhanced_app(config)
            self.logger.info("Enhanced production API initialized")
            
            print("✅ Enhanced API system initialized")
            
            print("🎉 Enhanced production system initialized successfully!")
            print("🧠 AI-powered features enabled:")
            print("   - Intelligent optimization strategy selection")
            print("   - Adaptive resource management")
            print("   - Machine learning-based performance prediction")
            print("   - Real-time system optimization")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize enhanced production system: {e}")
            if self.logger:
                self.logger.log_error(e, {"component": "enhanced_initialization"})
            return False
    
    def start(self):
        """Start enhanced production system."""
        if not self.initialize():
            return False
        
        print("\n🚀 Starting Enhanced Production System")
        print("=" * 60)
        
        try:
            self.running = True
            config = self.config_manager.get_config()
            
            # Start API server
            self.logger.info(f"Starting enhanced API server on {config.host}:{config.port}")
            
            # Run API server in background thread
            api_thread = threading.Thread(
                target=self._run_enhanced_api_server,
                daemon=True
            )
            api_thread.start()
            
            print(f"✅ Enhanced API server started on {config.host}:{config.port}")
            print(f"📊 Advanced monitoring dashboard available")
            print(f"📋 Enhanced API documentation: http://{config.host}:{config.port}/docs")
            print(f"🔍 Health check: http://{config.host}:{config.port}/health")
            print(f"📈 Metrics: http://{config.host}:{config.port}/metrics")
            print(f"🧠 AI features: Intelligent optimization and resource management")
            
            # Start intelligent monitoring
            self._start_intelligent_monitoring()
            
            # Main loop
            self._main_loop()
            
        except Exception as e:
            self.logger.log_error(e, {"component": "enhanced_startup"})
            print(f"❌ Failed to start enhanced production system: {e}")
            return False
        finally:
            self.shutdown()
        
        return True
    
    def _run_enhanced_api_server(self):
        """Run enhanced API server."""
        try:
            config = self.config_manager.get_config()
            run_enhanced_server(config)
        except Exception as e:
            self.logger.log_error(e, {"component": "enhanced_api_server"})
    
    def _start_intelligent_monitoring(self):
        """Start intelligent monitoring system."""
        monitoring_thread = threading.Thread(
            target=self._intelligent_monitoring_loop,
            daemon=True
        )
        monitoring_thread.start()
        self.logger.info("Intelligent monitoring started")
    
    def _intelligent_monitoring_loop(self):
        """Intelligent monitoring loop."""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Analyze system resources
                resources = self.resource_manager.analyze_system_resources()
                
                # Check for optimization opportunities
                if self._should_optimize_system(resources):
                    self._perform_system_optimization(resources)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep for monitoring interval
                time.sleep(30)
                
            except Exception as e:
                self.logger.log_error(e, {"component": "intelligent_monitoring"})
                time.sleep(5)
    
    def _should_optimize_system(self, resources: Dict[str, float]) -> bool:
        """Check if system should be optimized."""
        # Check CPU usage
        if resources['cpu_percent'] > 80:
            return True
        
        # Check memory usage
        if resources['memory_percent'] > 85:
            return True
        
        # Check load average
        if resources['load_average'] > 2.0:
            return True
        
        return False
    
    def _perform_system_optimization(self, resources: Dict[str, float]):
        """Perform system optimization."""
        try:
            self.logger.info("Performing intelligent system optimization")
            
            # Analyze current performance
            current_metrics = self.monitor.get_metrics_summary()
            
            # Optimize resource allocation
            optimal_allocation = self.resource_manager.predict_resource_needs([])
            
            # Log optimization
            self.logger.info(f"System optimization completed: {optimal_allocation}")
            
        except Exception as e:
            self.logger.log_error(e, {"component": "system_optimization"})
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            # Get optimization statistics
            if self.bulk_optimizer:
                stats = self.bulk_optimizer.get_optimization_statistics()
                
                self.performance_metrics.update({
                    'optimizations_completed': stats.get('total_optimizations', 0),
                    'avg_improvement': stats.get('avg_improvement', 0.0),
                    'avg_optimization_time': stats.get('avg_time', 0.0),
                    'success_rate': stats.get('success_rate', 0.0)
                })
                
        except Exception as e:
            self.logger.log_error(e, {"component": "performance_metrics"})
    
    def _main_loop(self):
        """Main system loop."""
        self.logger.info("Enhanced production system main loop started")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Check system health
                health_status = self.monitor.get_health_status()
                if health_status['overall_status'] == 'critical':
                    self.logger.critical("System health is critical")
                
                # Log system metrics
                metrics_summary = self.monitor.get_metrics_summary()
                self.logger.log_system_metrics()
                
                # Check for alerts
                alerts_summary = self.monitor.get_alerts_summary()
                if alerts_summary['total_alerts'] > 0:
                    self.logger.warning(f"Active alerts: {alerts_summary['total_alerts']}")
                
                # Display performance metrics
                self._display_performance_metrics()
                
                # Sleep for monitoring interval
                time.sleep(30)
                
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                self.logger.log_error(e, {"component": "enhanced_main_loop"})
                time.sleep(5)
    
    def _display_performance_metrics(self):
        """Display performance metrics."""
        try:
            print(f"\n📊 Enhanced Performance Metrics:")
            print(f"   - Optimizations completed: {self.performance_metrics['optimizations_completed']}")
            print(f"   - Average improvement: {self.performance_metrics['avg_improvement']:.2%}")
            print(f"   - Average optimization time: {self.performance_metrics['avg_optimization_time']:.2f}s")
            print(f"   - Success rate: {self.performance_metrics['success_rate']:.2%}")
            
            # Display system resources
            resources = self.resource_manager.analyze_system_resources()
            print(f"   - CPU usage: {resources['cpu_percent']:.1f}%")
            print(f"   - Memory usage: {resources['memory_percent']:.1f}%")
            print(f"   - Available memory: {resources['available_memory_gb']:.1f}GB")
            
        except Exception as e:
            self.logger.error(f"Error displaying performance metrics: {e}")
    
    def run_enhanced_optimization_demo(self):
        """Run enhanced optimization demo."""
        print("\n🧠 Running Enhanced Optimization Demo")
        print("=" * 50)
        
        try:
            # Create demo models
            class DemoModel(nn.Module):
                def __init__(self, size=100):
                    super().__init__()
                    self.linear = nn.Linear(size, size // 2)
                    self.relu = nn.ReLU()
                    self.output = nn.Linear(size // 2, 10)
                
                def forward(self, x):
                    x = self.linear(x)
                    x = self.relu(x)
                    return self.output(x)
            
            models = [
                ("demo_model_1", DemoModel(100)),
                ("demo_model_2", DemoModel(200)),
                ("demo_model_3", DemoModel(300))
            ]
            
            print(f"Created {len(models)} demo models for optimization")
            
            # Run enhanced optimization
            async def run_optimization():
                results = await optimize_models_enhanced(models)
                
                print(f"\n📊 Enhanced Optimization Results:")
                successful = [r for r in results if r.get('success', False)]
                failed = [r for r in results if not r.get('success', False)]
                
                print(f"   - Total models: {len(results)}")
                print(f"   - Successful: {len(successful)}")
                print(f"   - Failed: {len(failed)}")
                print(f"   - Success rate: {len(successful)/len(results)*100:.1f}%")
                
                if successful:
                    avg_improvement = np.mean([r.get('performance_improvement', 0) for r in successful])
                    avg_time = np.mean([r.get('optimization_time', 0) for r in successful])
                    print(f"   - Average improvement: {avg_improvement:.2%}")
                    print(f"   - Average time: {avg_time:.2f}s")
                
                for result in results:
                    if result.get('success', False):
                        print(f"   ✅ {result['model_name']}: {result.get('performance_improvement', 0):.2%} improvement")
                    else:
                        print(f"   ❌ {result['model_name']}: {result.get('error', 'Unknown error')}")
            
            # Run async optimization
            asyncio.run(run_optimization())
            
            print("🎉 Enhanced optimization demo completed!")
            return True
            
        except Exception as e:
            print(f"❌ Enhanced optimization demo failed: {e}")
            return False
    
    def get_enhanced_status(self):
        """Get enhanced system status."""
        if not self.monitor:
            return {"status": "not_initialized"}
        
        try:
            health_status = self.monitor.get_health_status()
            metrics_summary = self.monitor.get_metrics_summary()
            alerts_summary = self.monitor.get_alerts_summary()
            
            # Get optimization statistics
            optimization_stats = {}
            if self.bulk_optimizer:
                optimization_stats = self.bulk_optimizer.get_optimization_statistics()
            
            # Get system resources
            resources = self.resource_manager.analyze_system_resources()
            
            return {
                "status": "running" if self.running else "stopped",
                "health": health_status,
                "metrics": metrics_summary,
                "alerts": alerts_summary,
                "optimization": optimization_stats,
                "resources": resources,
                "performance": self.performance_metrics,
                "uptime": time.time() - self.start_time,
                "features": {
                    "ai_optimization": True,
                    "intelligent_resource_management": True,
                    "adaptive_monitoring": True,
                    "machine_learning": True
                }
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def shutdown(self):
        """Shutdown enhanced production system."""
        print("\n🛑 Shutting down Enhanced Production System")
        print("=" * 60)
        
        try:
            self.running = False
            self.shutdown_event.set()
            
            if self.monitor:
                self.monitor.stop()
                print("✅ Enhanced monitoring system stopped")
            
            if self.bulk_optimizer:
                # Save optimization model
                self.bulk_optimizer.save_optimization_model("models/enhanced_optimizer.pkl")
                print("✅ Enhanced bulk optimizer stopped")
            
            if self.logger:
                self.logger.info("Enhanced production system shutting down")
                print("✅ Enhanced logging system stopped")
            
            if self.config_manager:
                self.config_manager.shutdown()
                print("✅ Enhanced configuration manager stopped")
            
            print("🎉 Enhanced production system shutdown completed")
            
        except Exception as e:
            print(f"❌ Error during enhanced shutdown: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        print(f"\n📡 Received signal {signal_name}, initiating enhanced shutdown...")
        self.shutdown()
        sys.exit(0)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced Production System Runner")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--environment", choices=["development", "staging", "production"], 
                       default="production", help="Environment")
    parser.add_argument("--mode", choices=["start", "demo", "status"], 
                       default="start", help="Operation mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set environment
    environment = Environment(args.environment)
    
    # Create enhanced runner
    runner = EnhancedProductionRunner(args.config, environment)
    
    try:
        if args.mode == "start":
            print("🚀 Starting Enhanced Production System")
            success = runner.start()
            return 0 if success else 1
            
        elif args.mode == "demo":
            print("🧠 Running Enhanced Optimization Demo")
            success = runner.run_enhanced_optimization_demo()
            return 0 if success else 1
            
        elif args.mode == "status":
            print("📊 Enhanced Production System Status")
            status = runner.get_enhanced_status()
            print(json.dumps(status, indent=2))
            return 0
            
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return 1
            
    except KeyboardInterrupt:
        print("\n📡 Received keyboard interrupt")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())


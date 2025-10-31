#!/usr/bin/env python3
"""
Production Runner - Main production system runner
Orchestrates all production components and provides unified interface
"""

import asyncio
import argparse
import logging
import signal
import sys
import time
import threading
from typing import Dict, Any, Optional
from pathlib import Path
import json
import os

# Import production components
from production_config import (
    ProductionConfigManager, create_production_config, Environment
)
from production_logging import (
    create_production_logger, setup_production_logging
)
from production_monitoring import (
    create_production_monitor, ProductionMonitor
)
from production_api import (
    create_app, run_production_server
)
from production_deployment import (
    create_production_deployment, ProductionDeployment
)
from production_tests import run_production_tests

class ProductionRunner:
    """Production system runner."""
    
    def __init__(self, config_file: Optional[str] = None, environment: Optional[Environment] = None):
        self.config_file = config_file
        self.environment = environment
        self.config_manager = None
        self.logger = None
        self.monitor = None
        self.app = None
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def initialize(self):
        """Initialize production system."""
        print("🚀 Initializing Production System")
        print("=" * 50)
        
        try:
            # Initialize configuration
            self.config_manager = create_production_config(self.config_file, self.environment)
            config = self.config_manager.get_config()
            
            if not self.config_manager.validate_config():
                raise ValueError("Invalid configuration")
            
            print(f"✅ Configuration loaded: {config.environment.value}")
            
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
            self.logger.info("Production system initializing")
            
            print("✅ Logging system initialized")
            
            # Initialize monitoring
            self.monitor = create_production_monitor({
                'metrics': {
                    'collection_interval': 10,
                    'max_history_size': 1000
                },
                'health': {
                    'check_interval': 30
                },
                'alerts': {}
            })
            
            self.monitor.start()
            self.logger.info("Production monitoring started")
            
            print("✅ Monitoring system initialized")
            
            # Initialize API
            self.app = create_app()
            self.logger.info("Production API initialized")
            
            print("✅ API system initialized")
            
            print("🎉 Production system initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize production system: {e}")
            if self.logger:
                self.logger.log_error(e, {"component": "initialization"})
            return False
    
    def start(self):
        """Start production system."""
        if not self.initialize():
            return False
        
        print("\n🚀 Starting Production System")
        print("=" * 50)
        
        try:
            self.running = True
            config = self.config_manager.get_config()
            
            # Start API server
            self.logger.info(f"Starting API server on {config.host}:{config.port}")
            
            # Run API server in background thread
            api_thread = threading.Thread(
                target=self._run_api_server,
                daemon=True
            )
            api_thread.start()
            
            print(f"✅ API server started on {config.host}:{config.port}")
            print(f"📊 Monitoring dashboard available")
            print(f"📋 API documentation: http://{config.host}:{config.port}/docs")
            print(f"🔍 Health check: http://{config.host}:{config.port}/health")
            print(f"📈 Metrics: http://{config.host}:{config.port}/metrics")
            
            # Main loop
            self._main_loop()
            
        except Exception as e:
            self.logger.log_error(e, {"component": "startup"})
            print(f"❌ Failed to start production system: {e}")
            return False
        finally:
            self.shutdown()
        
        return True
    
    def _run_api_server(self):
        """Run API server."""
        try:
            config = self.config_manager.get_config()
            run_production_server(
                host=config.host,
                port=config.port,
                workers=config.workers,
                reload=False
            )
        except Exception as e:
            self.logger.log_error(e, {"component": "api_server"})
    
    def _main_loop(self):
        """Main system loop."""
        self.logger.info("Production system main loop started")
        
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
                
                # Sleep for monitoring interval
                time.sleep(30)
                
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                self.logger.log_error(e, {"component": "main_loop"})
                time.sleep(5)
    
    def shutdown(self):
        """Shutdown production system."""
        print("\n🛑 Shutting down Production System")
        print("=" * 50)
        
        try:
            self.running = False
            self.shutdown_event.set()
            
            if self.monitor:
                self.monitor.stop()
                print("✅ Monitoring system stopped")
            
            if self.logger:
                self.logger.info("Production system shutting down")
                print("✅ Logging system stopped")
            
            print("🎉 Production system shutdown completed")
            
        except Exception as e:
            print(f"❌ Error during shutdown: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        print(f"\n📡 Received signal {signal_name}, initiating shutdown...")
        self.shutdown()
        sys.exit(0)
    
    def run_tests(self):
        """Run production tests."""
        print("🧪 Running Production Tests")
        print("=" * 50)
        
        return run_production_tests()
    
    def create_deployment(self, output_dir: str = "deployment"):
        """Create deployment files."""
        print("🚀 Creating Production Deployment")
        print("=" * 50)
        
        try:
            config = self.config_manager.get_config()
            
            deployment = create_production_deployment({
                "app_name": "bulk-optimization",
                "app_version": "1.0.0",
                "environment": config.environment.value,
                "replicas": config.workers,
                "cpu_limit": "2",
                "memory_limit": "4Gi"
            })
            
            # Create Docker deployment
            docker_dir = os.path.join(output_dir, "docker")
            deployment.create_docker_deployment(docker_dir)
            print(f"✅ Docker deployment files created in {docker_dir}")
            
            # Create Kubernetes deployment
            k8s_dir = os.path.join(output_dir, "kubernetes")
            deployment.create_kubernetes_deployment(k8s_dir)
            print(f"✅ Kubernetes deployment files created in {k8s_dir}")
            
            print("🎉 Deployment files created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create deployment: {e}")
            return False
    
    def get_status(self):
        """Get system status."""
        if not self.monitor:
            return {"status": "not_initialized"}
        
        try:
            health_status = self.monitor.get_health_status()
            metrics_summary = self.monitor.get_metrics_summary()
            alerts_summary = self.monitor.get_alerts_summary()
            
            return {
                "status": "running" if self.running else "stopped",
                "health": health_status,
                "metrics": metrics_summary,
                "alerts": alerts_summary,
                "uptime": time.time() - getattr(self, 'start_time', time.time())
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Production System Runner")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--environment", choices=["development", "staging", "production"], 
                       default="production", help="Environment")
    parser.add_argument("--mode", choices=["start", "test", "deploy", "status"], 
                       default="start", help="Operation mode")
    parser.add_argument("--output-dir", default="deployment", 
                       help="Output directory for deployment files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set environment
    environment = Environment(args.environment)
    
    # Create runner
    runner = ProductionRunner(args.config, environment)
    
    try:
        if args.mode == "start":
            print("🚀 Starting Production System")
            success = runner.start()
            return 0 if success else 1
            
        elif args.mode == "test":
            print("🧪 Running Production Tests")
            success = runner.run_tests()
            return 0 if success else 1
            
        elif args.mode == "deploy":
            print("🚀 Creating Production Deployment")
            success = runner.create_deployment(args.output_dir)
            return 0 if success else 1
            
        elif args.mode == "status":
            print("📊 Production System Status")
            status = runner.get_status()
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


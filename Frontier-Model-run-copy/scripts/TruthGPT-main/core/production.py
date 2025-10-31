"""
Production System
Production-ready features for deployment, logging, and monitoring
"""

import torch
import torch.nn as nn
import logging
import json
import time
import os
import sys
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import threading
import queue
import signal
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ProductionConfig:
    """Configuration for production deployment"""
    # Service settings
    service_name: str = "truthgpt_service"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "production.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Performance settings
    max_workers: int = 4
    request_timeout: int = 30
    max_memory_usage: float = 0.9  # 90% of available memory
    
    # Monitoring settings
    health_check_interval: int = 30
    metrics_interval: int = 60
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_percent": 80.0,
        "memory_percent": 85.0,
        "gpu_memory_percent": 90.0
    })
    
    # Deployment settings
    auto_restart: bool = True
    graceful_shutdown: bool = True
    shutdown_timeout: int = 30

class ProductionLogger:
    """Production-ready logging system"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup production logger"""
        logger = logging.getLogger(self.config.service_name)
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(self.config.log_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.config.log_file,
            maxBytes=self.config.max_log_size,
            backupCount=self.config.backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(self.config.log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def log_request(self, request_id: str, endpoint: str, duration: float, status: str):
        """Log API request"""
        self.logger.info(f"Request {request_id}: {endpoint} - {duration:.3f}s - {status}")
    
    def log_error(self, request_id: str, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        context_str = f" - Context: {context}" if context else ""
        self.logger.error(f"Request {request_id}: {str(error)}{context_str}", exc_info=True)
    
    def log_performance(self, metric: str, value: float, unit: str = ""):
        """Log performance metric"""
        self.logger.info(f"Performance: {metric} = {value:.3f} {unit}")
    
    def log_system_event(self, event: str, details: Dict[str, Any] = None):
        """Log system event"""
        details_str = f" - Details: {details}" if details else ""
        self.logger.info(f"System Event: {event}{details_str}")

class HealthChecker:
    """Health check system for production monitoring"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.healthy = True
        self.last_check = time.time()
        self.check_results = {}
        
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        self.last_check = time.time()
        results = {
            "timestamp": self.last_check,
            "healthy": True,
            "checks": {}
        }
        
        # System health checks
        results["checks"]["system"] = self._check_system_health()
        results["checks"]["memory"] = self._check_memory_health()
        results["checks"]["gpu"] = self._check_gpu_health()
        results["checks"]["service"] = self._check_service_health()
        
        # Overall health status
        all_healthy = all(check["healthy"] for check in results["checks"].values())
        results["healthy"] = all_healthy
        self.healthy = all_healthy
        
        return results
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check system-level health"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            healthy = (
                cpu_percent < self.config.alert_thresholds["cpu_percent"] and
                memory.percent < self.config.alert_thresholds["memory_percent"]
            )
            
            return {
                "healthy": healthy,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / 1024 / 1024 / 1024
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory health"""
        try:
            memory = psutil.virtual_memory()
            healthy = memory.percent < self.config.alert_thresholds["memory_percent"]
            
            return {
                "healthy": healthy,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / 1024 / 1024 / 1024,
                "memory_available_gb": memory.available / 1024 / 1024 / 1024
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    def _check_gpu_health(self) -> Dict[str, Any]:
        """Check GPU health"""
        try:
            if not torch.cuda.is_available():
                return {
                    "healthy": True,
                    "gpu_available": False,
                    "message": "No GPU available"
                }
            
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024  # GB
            gpu_memory_percent = (gpu_memory_allocated / gpu_memory_total) * 100
            
            healthy = gpu_memory_percent < self.config.alert_thresholds["gpu_memory_percent"]
            
            return {
                "healthy": healthy,
                "gpu_available": True,
                "gpu_memory_percent": gpu_memory_percent,
                "gpu_memory_allocated_gb": gpu_memory_allocated,
                "gpu_memory_total_gb": gpu_memory_total
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    def _check_service_health(self) -> Dict[str, Any]:
        """Check service-level health"""
        try:
            # Check if service is responsive
            current_time = time.time()
            time_since_last_check = current_time - self.last_check
            
            healthy = time_since_last_check < 60  # Service should respond within 60 seconds
            
            return {
                "healthy": healthy,
                "time_since_last_check": time_since_last_check,
                "service_uptime": current_time - self.last_check
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

class ProductionAPI:
    """Production API server"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = ProductionLogger(config)
        self.health_checker = HealthChecker(config)
        self.running = False
        self.request_queue = queue.Queue()
        self.workers = []
        
    def start(self):
        """Start the production API server"""
        self.logger.log_system_event("Starting production API server", {
            "host": self.config.host,
            "port": self.config.port,
            "max_workers": self.config.max_workers
        })
        
        self.running = True
        
        # Start worker threads
        for i in range(self.config.max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        # Start health check thread
        health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        health_thread.start()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.log_system_event("Production API server started successfully")
    
    def stop(self):
        """Stop the production API server"""
        self.logger.log_system_event("Stopping production API server")
        
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=self.config.shutdown_timeout)
        
        self.logger.log_system_event("Production API server stopped")
    
    def _worker_loop(self):
        """Worker thread loop"""
        while self.running:
            try:
                # Get request from queue
                request = self.request_queue.get(timeout=1)
                self._process_request(request)
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.log_error("worker", e)
    
    def _process_request(self, request: Dict[str, Any]):
        """Process a single request"""
        request_id = request.get("id", "unknown")
        endpoint = request.get("endpoint", "unknown")
        
        start_time = time.time()
        
        try:
            # Process the request
            result = self._handle_request(request)
            
            duration = time.time() - start_time
            self.logger.log_request(request_id, endpoint, duration, "success")
            
            # Send response
            if "callback" in request:
                request["callback"](result)
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_error(request_id, e, {"endpoint": endpoint})
            self.logger.log_request(request_id, endpoint, duration, "error")
    
    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a specific request"""
        endpoint = request.get("endpoint")
        
        if endpoint == "health":
            return self.health_checker.check_health()
        elif endpoint == "generate":
            return self._handle_generate_request(request)
        elif endpoint == "optimize":
            return self._handle_optimize_request(request)
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")
    
    def _handle_generate_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text generation request"""
        # This would integrate with the InferenceEngine
        return {
            "status": "success",
            "message": "Generation request processed",
            "request_id": request.get("id")
        }
    
    def _handle_optimize_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model optimization request"""
        # This would integrate with the OptimizationEngine
        return {
            "status": "success",
            "message": "Optimization request processed",
            "request_id": request.get("id")
        }
    
    def _health_check_loop(self):
        """Health check loop"""
        while self.running:
            try:
                health_status = self.health_checker.check_health()
                
                if not health_status["healthy"]:
                    self.logger.log_system_event("Health check failed", health_status)
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.log_error("health_checker", e)
                time.sleep(self.config.health_check_interval)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.log_system_event(f"Received signal {signum}, initiating graceful shutdown")
        self.stop()
        sys.exit(0)
    
    def submit_request(self, endpoint: str, data: Dict[str, Any], callback: Optional[Callable] = None) -> str:
        """Submit a request to the API"""
        request_id = f"req_{int(time.time() * 1000)}"
        
        request = {
            "id": request_id,
            "endpoint": endpoint,
            "data": data,
            "callback": callback,
            "timestamp": time.time()
        }
        
        self.request_queue.put(request)
        return request_id

class ProductionDeployment:
    """Production deployment manager"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = ProductionLogger(config)
        self.api = ProductionAPI(config)
        
    def deploy(self):
        """Deploy the production service"""
        self.logger.log_system_event("Starting production deployment")
        
        try:
            # Validate configuration
            self._validate_config()
            
            # Setup environment
            self._setup_environment()
            
            # Start services
            self._start_services()
            
            self.logger.log_system_event("Production deployment completed successfully")
            
        except Exception as e:
            self.logger.log_error("deployment", e)
            raise
    
    def _validate_config(self):
        """Validate production configuration"""
        required_fields = ["service_name", "host", "port", "log_level"]
        
        for field in required_fields:
            if not hasattr(self.config, field) or getattr(self.config, field) is None:
                raise ValueError(f"Required configuration field missing: {field}")
    
    def _setup_environment(self):
        """Setup production environment"""
        # Create log directory
        log_dir = Path(self.config.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ["TRUTHGPT_SERVICE_NAME"] = self.config.service_name
        os.environ["TRUTHGPT_VERSION"] = self.config.version
        
        self.logger.log_system_event("Environment setup completed")
    
    def _start_services(self):
        """Start production services"""
        # Start API server
        self.api.start()
        
        self.logger.log_system_event("All services started successfully")
    
    def shutdown(self):
        """Shutdown production services"""
        self.logger.log_system_event("Initiating production shutdown")
        
        # Stop API server
        self.api.stop()
        
        self.logger.log_system_event("Production shutdown completed")

class ProductionMonitor:
    """Production monitoring system"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = ProductionLogger(config)
        self.metrics = {}
        self.alerts = []
        
    def start_monitoring(self):
        """Start production monitoring"""
        self.logger.log_system_event("Starting production monitoring")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Collect metrics
                self._collect_metrics()
                
                # Check alerts
                self._check_alerts()
                
                time.sleep(self.config.metrics_interval)
                
            except Exception as e:
                self.logger.log_error("monitor", e)
                time.sleep(self.config.metrics_interval)
    
    def _collect_metrics(self):
        """Collect system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            metrics = {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / 1024 / 1024 / 1024,
                "memory_available_gb": memory.available / 1024 / 1024 / 1024
            }
            
            # GPU metrics if available
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
                metrics.update({
                    "gpu_memory_allocated_gb": gpu_memory_allocated,
                    "gpu_memory_total_gb": gpu_memory_total,
                    "gpu_memory_percent": (gpu_memory_allocated / gpu_memory_total) * 100
                })
            
            self.metrics[time.time()] = metrics
            
            # Log performance metrics
            self.logger.log_performance("cpu_percent", cpu_percent, "%")
            self.logger.log_performance("memory_percent", memory.percent, "%")
            
        except Exception as e:
            self.logger.log_error("metrics_collection", e)
    
    def _check_alerts(self):
        """Check for alert conditions"""
        if not self.metrics:
            return
        
        latest_metrics = max(self.metrics.values(), key=lambda x: x["timestamp"])
        
        # Check CPU alert
        if latest_metrics["cpu_percent"] > self.config.alert_thresholds["cpu_percent"]:
            self._trigger_alert("high_cpu", latest_metrics["cpu_percent"])
        
        # Check memory alert
        if latest_metrics["memory_percent"] > self.config.alert_thresholds["memory_percent"]:
            self._trigger_alert("high_memory", latest_metrics["memory_percent"])
        
        # Check GPU alert
        if "gpu_memory_percent" in latest_metrics:
            if latest_metrics["gpu_memory_percent"] > self.config.alert_thresholds["gpu_memory_percent"]:
                self._trigger_alert("high_gpu_memory", latest_metrics["gpu_memory_percent"])
    
    def _trigger_alert(self, alert_type: str, value: float):
        """Trigger an alert"""
        alert = {
            "type": alert_type,
            "value": value,
            "timestamp": time.time(),
            "threshold": self.config.alert_thresholds.get(alert_type.replace("high_", "").replace("_", "_percent"), 0)
        }
        
        self.alerts.append(alert)
        self.logger.log_system_event(f"Alert triggered: {alert_type}", alert)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.metrics:
            return {}
        
        recent_metrics = list(self.metrics.values())[-10:]  # Last 10 measurements
        
        summary = {
            "total_measurements": len(self.metrics),
            "recent_measurements": len(recent_metrics),
            "avg_cpu_percent": sum(m["cpu_percent"] for m in recent_metrics) / len(recent_metrics),
            "avg_memory_percent": sum(m["memory_percent"] for m in recent_metrics) / len(recent_metrics),
            "max_cpu_percent": max(m["cpu_percent"] for m in recent_metrics),
            "max_memory_percent": max(m["memory_percent"] for m in recent_metrics),
            "alerts_count": len(self.alerts),
            "latest_alert": self.alerts[-1] if self.alerts else None
        }
        
        return summary


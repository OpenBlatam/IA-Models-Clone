#!/usr/bin/env python3
"""
Document Workflow Chain v2.0 - Optimized Startup Script
======================================================

High-performance startup script with:
- Advanced configuration management
- Health checks and monitoring
- Graceful shutdown handling
- Performance optimization
- Error handling and recovery
- Development and production modes
"""

import asyncio
import sys
import os
import signal
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
import json
import psutil
import uvicorn
from contextlib import asynccontextmanager

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from config_v2 import (
        get_settings, get_ai_config, get_database_config, get_cache_config,
        get_security_config, get_performance_config, get_monitoring_config,
        validate_config, ConfigManager
    )
    from workflow_chain_v2 import WorkflowChainManager
    from api_v2 import app
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)


class StartupManager:
    """Manages application startup and shutdown"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.settings = get_settings()
        self.workflow_manager: Optional[WorkflowChainManager] = None
        self.server_process: Optional[Any] = None
        self.startup_time = time.time()
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.settings.log_level.value)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Setup file handler if configured
        handlers = [console_handler]
        if self.settings.log_file:
            file_handler = logging.FileHandler(self.settings.log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            force=True
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    async def check_dependencies(self) -> bool:
        """Check system dependencies"""
        self.logger.info("🔍 Checking dependencies...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.logger.error("❌ Python 3.8+ required")
            return False
        
        # Check required packages
        required_packages = [
            'fastapi', 'uvicorn', 'pydantic', 'sqlalchemy', 'redis'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
            return False
        
        # Check system resources
        memory = psutil.virtual_memory()
        if memory.available < 100 * 1024 * 1024:  # 100MB
            self.logger.warning("⚠️ Low memory available")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.free < 1024 * 1024 * 1024:  # 1GB
            self.logger.warning("⚠️ Low disk space available")
        
        self.logger.info("✅ Dependencies check passed")
        return True
    
    async def validate_configuration(self) -> bool:
        """Validate configuration"""
        self.logger.info("🔧 Validating configuration...")
        
        if not validate_config():
            self.logger.error("❌ Configuration validation failed")
            return False
        
        # Check AI configuration
        ai_config = get_ai_config()
        if ai_config.client_type.value != "local" and not ai_config.api_key:
            self.logger.warning("⚠️ AI API key not configured")
        
        # Check database configuration
        db_config = get_database_config()
        if db_config.type.value == "sqlite":
            db_path = Path(db_config.url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check cache configuration
        cache_config = get_cache_config()
        if cache_config.type.value == "redis":
            try:
                import redis
                r = redis.from_url(cache_config.redis_url)
                r.ping()
                self.logger.info("✅ Redis connection successful")
            except Exception as e:
                self.logger.warning(f"⚠️ Redis connection failed: {e}")
        
        self.logger.info("✅ Configuration validation passed")
        return True
    
    async def initialize_workflow_manager(self) -> bool:
        """Initialize workflow manager"""
        self.logger.info("🚀 Initializing workflow manager...")
        
        try:
            self.workflow_manager = WorkflowChainManager()
            
            # Add global plugins
            self.workflow_manager.add_global_plugin("performance_monitor", PerformanceMonitor())
            self.workflow_manager.add_global_plugin("error_handler", ErrorHandler())
            
            # Subscribe to events
            self.workflow_manager.subscribe_to_events("chain_created", self._on_chain_created)
            self.workflow_manager.subscribe_to_events("global_node_added", self._on_node_added)
            
            self.logger.info("✅ Workflow manager initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize workflow manager: {e}")
            return False
    
    async def _on_chain_created(self, data: Dict[str, Any]):
        """Handle chain created event"""
        chain_id = data["chain_id"]
        self.logger.info(f"📝 Chain created: {chain_id}")
    
    async def _on_node_added(self, data: Dict[str, Any]):
        """Handle node added event"""
        chain_id = data["chain_id"]
        node_id = data["node_id"]
        self.logger.debug(f"📄 Node added: {node_id} to chain {chain_id}")
    
    async def run_health_checks(self) -> bool:
        """Run health checks"""
        self.logger.info("🏥 Running health checks...")
        
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > 90:
                self.logger.warning(f"⚠️ High memory usage: {memory_percent}%")
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.logger.warning(f"⚠️ High CPU usage: {cpu_percent}%")
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 90:
                self.logger.warning(f"⚠️ High disk usage: {disk_percent}%")
            
            self.logger.info("✅ Health checks passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Health checks failed: {e}")
            return False
    
    async def start_server(self) -> bool:
        """Start the server"""
        self.logger.info("🌐 Starting server...")
        
        try:
            # Get performance configuration
            perf_config = get_performance_config()
            
            # Configure uvicorn
            config = uvicorn.Config(
                app=app,
                host=self.settings.host,
                port=self.settings.port,
                workers=self.settings.workers,
                log_level=self.settings.log_level.value.lower(),
                access_log=True,
                reload=self.settings.reload and self.settings.environment == "development",
                loop="asyncio",
                http="httptools" if self.settings.environment == "production" else "auto",
                ws="websockets",
                lifespan="on",
                timeout_keep_alive=perf_config.keep_alive_timeout,
                limit_concurrency=perf_config.max_concurrent_requests,
                limit_max_requests=1000,
                timeout_graceful_shutdown=30
            )
            
            # Create server
            server = uvicorn.Server(config)
            
            # Start server in background
            self.server_process = asyncio.create_task(server.serve())
            
            self.logger.info(f"✅ Server started on {self.settings.host}:{self.settings.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start server: {e}")
            return False
    
    async def run_monitoring(self):
        """Run monitoring loop"""
        monitoring_config = get_monitoring_config()
        
        if not monitoring_config.enabled:
            return
        
        self.logger.info("📊 Starting monitoring...")
        
        while not self.shutdown_requested:
            try:
                # Get system statistics
                stats = await self.workflow_manager.get_global_statistics()
                
                # Log statistics
                self.logger.info(f"📈 Stats: {stats['total_chains']} chains, "
                               f"{stats['total_nodes']} nodes, "
                               f"{stats['total_words']} words")
                
                # Sleep for monitoring interval
                await asyncio.sleep(monitoring_config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def graceful_shutdown(self):
        """Graceful shutdown"""
        self.logger.info("🛑 Initiating graceful shutdown...")
        
        try:
            # Cancel server task
            if self.server_process and not self.server_process.done():
                self.server_process.cancel()
                try:
                    await self.server_process
                except asyncio.CancelledError:
                    pass
            
            # Close workflow manager
            if self.workflow_manager:
                # Save any pending data
                self.logger.info("💾 Saving workflow data...")
            
            # Calculate uptime
            uptime = time.time() - self.startup_time
            self.logger.info(f"⏱️ Total uptime: {uptime:.2f} seconds")
            
            self.logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Shutdown error: {e}")
    
    async def run(self):
        """Main run method"""
        self.logger.info(f"🚀 Starting {self.settings.app_name} v{self.settings.app_version}")
        self.logger.info(f"🌍 Environment: {self.settings.environment}")
        self.logger.info(f"🐍 Python: {sys.version}")
        
        try:
            # Check dependencies
            if not await self.check_dependencies():
                return False
            
            # Validate configuration
            if not await self.validate_configuration():
                return False
            
            # Initialize workflow manager
            if not await self.initialize_workflow_manager():
                return False
            
            # Run health checks
            if not await self.run_health_checks():
                return False
            
            # Start server
            if not await self.start_server():
                return False
            
            # Start monitoring
            monitoring_task = asyncio.create_task(self.run_monitoring())
            
            # Wait for shutdown signal
            while not self.shutdown_requested:
                await asyncio.sleep(1)
            
            # Cancel monitoring
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
            
            # Graceful shutdown
            await self.graceful_shutdown()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Startup failed: {e}")
            return False


class PerformanceMonitor:
    """Performance monitoring plugin"""
    
    def __init__(self):
        self.metrics = {
            "requests": 0,
            "errors": 0,
            "avg_response_time": 0.0,
            "memory_usage": 0.0,
            "cpu_usage": 0.0
        }
    
    async def track_request(self, duration: float, success: bool):
        """Track request metrics"""
        self.metrics["requests"] += 1
        if not success:
            self.metrics["errors"] += 1
        
        # Update average response time
        total_requests = self.metrics["requests"]
        current_avg = self.metrics["avg_response_time"]
        self.metrics["avg_response_time"] = (
            (current_avg * (total_requests - 1) + duration) / total_requests
        )
    
    async def update_system_metrics(self):
        """Update system metrics"""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        self.metrics["memory_usage"] = memory.percent
        self.metrics["cpu_usage"] = cpu


class ErrorHandler:
    """Error handling plugin"""
    
    def __init__(self):
        self.error_count = 0
        self.error_types = {}
    
    async def handle_error(self, error: Exception, context: str = ""):
        """Handle error"""
        self.error_count += 1
        error_type = type(error).__name__
        
        if error_type not in self.error_types:
            self.error_types[error_type] = 0
        self.error_types[error_type] += 1
        
        logging.error(f"Error in {context}: {error}")


def main():
    """Main entry point"""
    try:
        # Create startup manager
        startup_manager = StartupManager()
        
        # Run the application
        success = asyncio.run(startup_manager.run())
        
        if success:
            print("✅ Application started successfully")
            sys.exit(0)
        else:
            print("❌ Application failed to start")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()





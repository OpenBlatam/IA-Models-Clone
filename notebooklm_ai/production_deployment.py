from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
import time
import json
import os
import sys
import signal
import traceback
from typing import Dict, Any, List, Optional
import threading
from pathlib import Path
import subprocess
import psutil
import structlog
import uvicorn
from fastapi import FastAPI
import redis.asyncio as redis
from production_config import get_config, ProductionConfig
from main_production_advanced import ProductionApp
from production_worker import WorkerPool
from production_monitoring import ProductionMonitoring
            import httpx
from typing import Any, List, Dict, Optional
"""
Production Deployment Script
============================

Main deployment script that orchestrates all production components:
- Main API server
- Worker system
- Monitoring system
- Health checks
- Graceful shutdown
"""


# Production imports

# Local imports

class ProductionDeployment:
    """Production deployment orchestrator"""
    
    def __init__(self) -> Any:
        self.config = get_config()
        self.logger = structlog.get_logger()
        
        # Component instances
        self.api_server = None
        self.worker_pool = None
        self.monitoring = None
        self.redis_client = None
        
        # Status tracking
        self.is_running = False
        self.startup_time = None
        self.components_status = {}
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Production deployment initialized")
    
    def _signal_handler(self, signum, frame) -> Any:
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.shutdown())
    
    async def startup(self) -> Any:
        """Startup all production components"""
        self.logger.info("🚀 Starting Production Deployment")
        self.startup_time = time.time()
        
        try:
            # Step 1: Validate configuration
            await self._validate_configuration()
            
            # Step 2: Initialize Redis connection
            await self._initialize_redis()
            
            # Step 3: Start monitoring system
            await self._start_monitoring()
            
            # Step 4: Start worker pool
            await self._start_workers()
            
            # Step 5: Start API server
            await self._start_api_server()
            
            # Step 6: Run health checks
            await self._run_health_checks()
            
            # Step 7: Mark as running
            self.is_running = True
            
            self.logger.info("✅ Production deployment started successfully")
            self._print_startup_summary()
            
        except Exception as e:
            self.logger.error(f"Failed to start production deployment: {e}")
            self.logger.error(traceback.format_exc())
            await self.shutdown()
            raise
    
    async def _validate_configuration(self) -> bool:
        """Validate production configuration"""
        self.logger.info("Validating configuration...")
        
        errors = self.config.validate()
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)
        
        # Create necessary directories
        self.config.create_directories()
        
        self.logger.info("✅ Configuration validated")
    
    async def _initialize_redis(self) -> Any:
        """Initialize Redis connection"""
        self.logger.info("Initializing Redis connection...")
        
        try:
            self.redis_client = redis.from_url(self.config.get_redis_url())
            await self.redis_client.ping()
            self.logger.info("✅ Redis connection established")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def _start_monitoring(self) -> Any:
        """Start monitoring system"""
        self.logger.info("Starting monitoring system...")
        
        try:
            self.monitoring = ProductionMonitoring(self.config)
            
            # Start monitoring in background
            monitoring_task = asyncio.create_task(self.monitoring.start())
            
            # Wait a bit for monitoring to start
            await asyncio.sleep(2)
            
            self.components_status['monitoring'] = 'running'
            self.logger.info("✅ Monitoring system started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            raise
    
    async def _start_workers(self) -> Any:
        """Start worker pool"""
        self.logger.info("Starting worker pool...")
        
        try:
            self.worker_pool = WorkerPool(self.config, self.redis_client)
            
            # Start workers in background
            worker_task = asyncio.create_task(self.worker_pool.start())
            
            # Wait a bit for workers to start
            await asyncio.sleep(3)
            
            self.components_status['workers'] = 'running'
            self.logger.info("✅ Worker pool started")
            
        except Exception as e:
            self.logger.error(f"Failed to start workers: {e}")
            raise
    
    async async def _start_api_server(self) -> Any:
        """Start API server"""
        self.logger.info("Starting API server...")
        
        try:
            # Create production app
            self.api_server = ProductionApp()
            await self.api_server.startup()
            
            # Configure uvicorn
            config = uvicorn.Config(
                app=self.api_server.app,
                host=self.config.api.host,
                port=self.config.api.port,
                workers=self.config.api.workers,
                log_level="info",
                access_log=True,
                loop="asyncio",
                http="httptools",
                ws="websockets",
                lifespan="on",
            )
            
            # Create server
            self.server = uvicorn.Server(config)
            
            # Start server in background
            server_task = asyncio.create_task(self.server.serve())
            
            # Wait a bit for server to start
            await asyncio.sleep(5)
            
            self.components_status['api_server'] = 'running'
            self.logger.info("✅ API server started")
            
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
            raise
    
    async def _run_health_checks(self) -> Any:
        """Run initial health checks"""
        self.logger.info("Running health checks...")
        
        try:
            # Wait for all components to be ready
            await asyncio.sleep(10)
            
            # Run health checks
            health_results = await self._check_all_components()
            
            # Check overall health
            all_healthy = all(status == 'healthy' for status in health_results.values())
            
            if all_healthy:
                self.logger.info("✅ All components healthy")
            else:
                self.logger.warning("⚠️ Some components have issues:")
                for component, status in health_results.items():
                    if status != 'healthy':
                        self.logger.warning(f"  - {component}: {status}")
            
            self.components_status['health_checks'] = health_results
            
        except Exception as e:
            self.logger.error(f"Health checks failed: {e}")
            # Don't fail startup for health check issues
    
    async def _check_all_components(self) -> Dict[str, str]:
        """Check health of all components"""
        results = {}
        
        # Check API server
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{self.config.api.host}:{self.config.api.port}/health", timeout=5)
                results['api_server'] = 'healthy' if response.status_code == 200 else 'unhealthy'
        except:
            results['api_server'] = 'unhealthy'
        
        # Check monitoring
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://0.0.0.0:8002/health", timeout=5)
                results['monitoring'] = 'healthy' if response.status_code == 200 else 'unhealthy'
        except:
            results['monitoring'] = 'unhealthy'
        
        # Check Redis
        try:
            await self.redis_client.ping()
            results['redis'] = 'healthy'
        except:
            results['redis'] = 'unhealthy'
        
        # Check workers
        try:
            # This would check worker pool status
            results['workers'] = 'healthy'
        except:
            results['workers'] = 'unhealthy'
        
        return results
    
    def _print_startup_summary(self) -> Any:
        """Print startup summary"""
        uptime = time.time() - self.startup_time
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    PRODUCTION DEPLOYMENT                     ║
║                        STARTUP SUMMARY                       ║
╠══════════════════════════════════════════════════════════════╣
║ Environment: {self.config.environment.value:>40} ║
║ Startup Time: {uptime:.2f}s{'':>35} ║
║ API Server: {self.config.api.host}:{self.config.api.port}{'':>30} ║
║ Monitoring: 0.0.0.0:8002{'':>35} ║
║ Workers: {self.config.performance.max_workers}{'':>40} ║
║ Redis: {self.config.redis.host}:{self.config.redis.port}{'':>35} ║
╠══════════════════════════════════════════════════════════════╣
║ Component Status:                                            ║
"""
        
        for component, status in self.components_status.items():
            status_icon = "✅" if status == 'running' else "❌"
            summary += f"║ {status_icon} {component}: {status}{'':>40} ║\n"
        
        summary += """╠══════════════════════════════════════════════════════════════╣
║ Health Checks:                                              ║
"""
        
        if 'health_checks' in self.components_status:
            for component, status in self.components_status['health_checks'].items():
                status_icon = "✅" if status == 'healthy' else "❌"
                summary += f"║ {status_icon} {component}: {status}{'':>40} ║\n"
        
        summary += """╠══════════════════════════════════════════════════════════════╣
║ Endpoints:                                                  ║
║ • API Documentation: http://localhost:8001/docs             ║
║ • Health Check: http://localhost:8001/health                ║
║ • Metrics: http://localhost:8001/metrics                    ║
║ • Monitoring Dashboard: http://localhost:8002               ║
║ • Prometheus Metrics: http://localhost:8002/metrics         ║
╚══════════════════════════════════════════════════════════════╝
"""
        
        print(summary)
        self.logger.info("Production deployment startup summary printed")
    
    async def shutdown(self) -> Any:
        """Gracefully shutdown all components"""
        if not self.is_running:
            return
        
        self.logger.info("🛑 Shutting down production deployment")
        self.is_running = False
        
        shutdown_tasks = []
        
        # Shutdown API server
        if self.api_server:
            self.logger.info("Shutting down API server...")
            shutdown_tasks.append(self.api_server.shutdown())
        
        # Shutdown worker pool
        if self.worker_pool:
            self.logger.info("Shutting down worker pool...")
            shutdown_tasks.append(self.worker_pool.stop())
        
        # Shutdown monitoring
        if self.monitoring:
            self.logger.info("Shutting down monitoring...")
            shutdown_tasks.append(self.monitoring.stop())
        
        # Wait for all shutdowns to complete
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        self.logger.info("✅ Production deployment shutdown completed")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get deployment status"""
        uptime = time.time() - self.startup_time if self.startup_time else 0
        
        return {
            "is_running": self.is_running,
            "uptime": uptime,
            "components": self.components_status,
            "configuration": {
                "environment": self.config.environment.value,
                "api_host": self.config.api.host,
                "api_port": self.config.api.port,
                "workers": self.config.performance.max_workers,
                "redis_host": self.config.redis.host,
                "redis_port": self.config.redis.port
            }
        }

class DeploymentManager:
    """Deployment manager for different environments"""
    
    def __init__(self) -> Any:
        self.logger = structlog.get_logger()
        self.deployment = None
    
    async def deploy(self, environment: str = "production"):
        """Deploy the system"""
        self.logger.info(f"Starting deployment for environment: {environment}")
        
        try:
            # Set environment
            os.environ["ENVIRONMENT"] = environment
            
            # Create deployment
            self.deployment = ProductionDeployment()
            
            # Start deployment
            await self.deployment.startup()
            
            # Keep running
            while self.deployment.is_running:
                await asyncio.sleep(1)
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            raise
    
    async def stop(self) -> Any:
        """Stop deployment"""
        if self.deployment:
            await self.deployment.shutdown()

def setup_production_logging():
    """Setup production logging"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/deployment.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

async def main():
    """Main deployment function"""
    # Setup logging
    setup_production_logging()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    logger = structlog.get_logger()
    logger.info("🚀 Starting Production Deployment Manager")
    
    try:
        # Get environment from command line or default
        environment = sys.argv[1] if len(sys.argv) > 1 else "production"
        
        # Create deployment manager
        manager = DeploymentManager()
        
        # Deploy
        await manager.deploy(environment)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Cleanup
        if 'manager' in locals():
            await manager.stop()
        logger.info("✅ Deployment manager shutdown completed")

def run_docker_compose():
    """Run docker-compose deployment"""
    try:
        # Check if docker-compose file exists
        compose_file = "deployment/docker-compose.advanced.yml"
        if not os.path.exists(compose_file):
            print(f"Docker compose file not found: {compose_file}")
            return False
        
        # Run docker-compose
        print("🚀 Starting Docker Compose deployment...")
        result = subprocess.run([
            "docker-compose", "-f", compose_file, "up", "-d"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Docker Compose deployment started successfully")
            print("\nServices:")
            print("• API Server: http://localhost:8001")
            print("• Monitoring Dashboard: http://localhost:8002")
            print("• Grafana: http://localhost:3000")
            print("• Prometheus: http://localhost:9090")
            return True
        else:
            print(f"❌ Docker Compose deployment failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Docker Compose error: {e}")
        return False

def stop_docker_compose():
    """Stop docker-compose deployment"""
    try:
        compose_file = "deployment/docker-compose.advanced.yml"
        if not os.path.exists(compose_file):
            print(f"Docker compose file not found: {compose_file}")
            return False
        
        print("🛑 Stopping Docker Compose deployment...")
        result = subprocess.run([
            "docker-compose", "-f", compose_file, "down"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Docker Compose deployment stopped")
            return True
        else:
            print(f"❌ Failed to stop Docker Compose: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Docker Compose stop error: {e}")
        return False

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "docker":
            # Docker deployment
            if len(sys.argv) > 2 and sys.argv[2] == "stop":
                stop_docker_compose()
            else:
                run_docker_compose()
        elif command in ["dev", "development"]:
            # Development deployment
            os.environ["ENVIRONMENT"] = "development"
            asyncio.run(main())
        elif command in ["staging"]:
            # Staging deployment
            os.environ["ENVIRONMENT"] = "staging"
            asyncio.run(main())
        elif command in ["prod", "production"]:
            # Production deployment
            os.environ["ENVIRONMENT"] = "production"
            asyncio.run(main())
        else:
            print("Usage:")
            print("  python production_deployment.py [dev|staging|prod|docker]")
            print("  python production_deployment.py docker stop")
    else:
        # Default to production
        os.environ["ENVIRONMENT"] = "production"
        asyncio.run(main()) 
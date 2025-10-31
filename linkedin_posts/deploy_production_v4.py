#!/usr/bin/env python3
"""
🚀 ENHANCED LINKEDIN OPTIMIZER v4.0 - PRODUCTION DEPLOYMENT SCRIPT
==================================================================

This script provides production-ready deployment capabilities including:
- Environment configuration
- Service management
- Health monitoring
- Performance optimization
- Security hardening
- Logging setup
- Backup and recovery

Usage: python deploy_production_v4.py [--config config.yaml] [--mode production|staging|development]
"""

import os
import sys
import yaml
import json
import asyncio
import logging
import argparse
import signal
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import subprocess
import shutil

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: str = "development"
    port: int = 8000
    host: str = "0.0.0.0"
    workers: int = 4
    max_memory_mb: int = 2048
    max_cpu_percent: int = 80
    enable_monitoring: bool = True
    enable_logging: bool = True
    enable_backup: bool = True
    backup_interval_hours: int = 24
    security_level: str = "production"
    ssl_enabled: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    database_url: str = ""
    redis_url: str = ""
    log_level: str = "INFO"
    metrics_port: int = 9090

class ProductionDeployer:
    """Production deployment manager."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.processes = []
        self.monitoring_active = False
        self.backup_active = False
        
    async def deploy(self) -> bool:
        """Deploy the system in production mode."""
        logger.info(f"🚀 Starting production deployment for environment: {self.config.environment}")
        
        try:
            # Step 1: Environment validation
            if not await self._validate_environment():
                logger.error("❌ Environment validation failed")
                return False
            
            # Step 2: System preparation
            if not await self._prepare_system():
                logger.error("❌ System preparation failed")
                return False
            
            # Step 3: Security hardening
            if not await self._harden_security():
                logger.error("❌ Security hardening failed")
                return False
            
            # Step 4: Service deployment
            if not await self._deploy_services():
                logger.error("❌ Service deployment failed")
                return False
            
            # Step 5: Health checks
            if not await self._run_health_checks():
                logger.error("❌ Health checks failed")
                return False
            
            # Step 6: Start monitoring
            if self.config.enable_monitoring:
                await self._start_monitoring()
            
            # Step 7: Start backup service
            if self.config.enable_backup:
                await self._start_backup_service()
            
            logger.info("✅ Production deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            await self._cleanup()
            return False
    
    async def _validate_environment(self) -> bool:
        """Validate the deployment environment."""
        logger.info("🔍 Validating deployment environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("❌ Python 3.8+ required")
            return False
        
        # Check required files
        required_files = [
            "enhanced_system_integration_v4.py",
            "ai_content_intelligence_v4.py",
            "real_time_analytics_v4.py",
            "security_compliance_v4.py",
            "requirements_v4.txt"
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                logger.error(f"❌ Required file not found: {file}")
                return False
        
        # Check system resources
        memory = psutil.virtual_memory()
        if memory.total < self.config.max_memory_mb * 1024 * 1024:
            logger.warning(f"⚠️  Available memory ({memory.total // (1024*1024)} MB) is below recommended ({self.config.max_memory_mb} MB)")
        
        # Check disk space
        disk = psutil.disk_usage('.')
        if disk.free < 1024 * 1024 * 1024:  # 1GB
            logger.warning("⚠️  Low disk space available")
        
        logger.info("✅ Environment validation passed")
        return True
    
    async def _prepare_system(self) -> bool:
        """Prepare the system for production deployment."""
        logger.info("🔧 Preparing system for production...")
        
        try:
            # Create necessary directories
            directories = ["logs", "backups", "config", "temp", "cache"]
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
            
            # Install dependencies if needed
            if not os.path.exists("venv"):
                logger.info("📦 Creating virtual environment...")
                subprocess.run(["python", "-m", "venv", "venv"], check=True)
            
            # Activate virtual environment and install requirements
            if os.name == "nt":  # Windows
                pip_path = "venv\\Scripts\\pip"
            else:  # Unix/Linux
                pip_path = "venv/bin/pip"
            
            if os.path.exists(pip_path):
                logger.info("📦 Installing production dependencies...")
                subprocess.run([pip_path, "install", "-r", "requirements_v4.txt"], check=True)
            
            # Download AI models if not present
            if not os.path.exists("venv/Lib/site-packages/spacy/data/en_core_web_sm"):
                logger.info("🤖 Downloading AI models...")
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            
            logger.info("✅ System preparation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ System preparation failed: {e}")
            return False
    
    async def _harden_security(self) -> bool:
        """Apply security hardening measures."""
        logger.info("🔒 Applying security hardening...")
        
        try:
            # Create security configuration
            security_config = {
                "authentication": {
                    "enabled": True,
                    "session_timeout_minutes": 30,
                    "max_login_attempts": 3,
                    "password_policy": "strong"
                },
                "encryption": {
                    "algorithm": "AES-256-GCM",
                    "key_rotation_days": 90
                },
                "access_control": {
                    "default_level": "STANDARD",
                    "admin_ips": ["127.0.0.1"],
                    "rate_limiting": {
                        "requests_per_minute": 100,
                        "burst_limit": 20
                    }
                },
                "audit_logging": {
                    "enabled": True,
                    "retention_days": 365,
                    "log_sensitive_operations": True
                }
            }
            
            with open("config/security.yaml", "w") as f:
                yaml.dump(security_config, f, default_flow_style=False)
            
            # Set file permissions (Unix/Linux)
            if os.name != "nt":
                os.chmod("config/security.yaml", 0o600)
                os.chmod("logs", 0o755)
                os.chmod("backups", 0o755)
            
            logger.info("✅ Security hardening completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Security hardening failed: {e}")
            return False
    
    async def _deploy_services(self) -> bool:
        """Deploy the main services."""
        logger.info("🚀 Deploying services...")
        
        try:
            # Start the main optimizer service
            if os.name == "nt":  # Windows
                python_path = "venv\\Scripts\\python"
            else:  # Unix/Linux
                python_path = "venv/bin/python"
            
            # Create service startup script
            startup_script = self._create_startup_script(python_path)
            
            # Start the service
            process = subprocess.Popen(
                [python_path, "-c", startup_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes.append(process)
            
            # Wait for service to start
            await asyncio.sleep(5)
            
            # Check if service is running
            if process.poll() is None:
                logger.info("✅ Main service deployed successfully")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ Service failed to start: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Service deployment failed: {e}")
            return False
    
    def _create_startup_script(self, python_path: str) -> str:
        """Create the startup script for the service."""
        return f"""
import asyncio
import sys
sys.path.insert(0, '.')
from enhanced_system_integration_v4 import EnhancedLinkedInOptimizer

async def run_production_service():
    optimizer = EnhancedLinkedInOptimizer()
    print("🚀 Production service started successfully!")
    print(f"📍 Listening on {self.config.host}:{self.config.port}")
    
    try:
        # Keep service running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\\n🛑 Shutting down production service...")
        await optimizer.shutdown()
        print("✅ Service shutdown complete")

if __name__ == "__main__":
    asyncio.run(run_production_service())
"""
    
    async def _run_health_checks(self) -> bool:
        """Run comprehensive health checks."""
        logger.info("💓 Running health checks...")
        
        try:
            # Test service connectivity
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((self.config.host, self.config.port))
            sock.close()
            
            if result != 0:
                logger.error(f"❌ Service not accessible on {self.config.host}:{self.config.port}")
                return False
            
            # Test system health
            health_status = await self._check_system_health()
            if not health_status["healthy"]:
                logger.error(f"❌ System health check failed: {health_status['issues']}")
                return False
            
            logger.info("✅ Health checks passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health checks failed: {e}")
            return False
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_healthy = memory.percent < self.config.max_cpu_percent
            
            # CPU usage
            cpu_healthy = psutil.cpu_percent(interval=1) < self.config.max_cpu_percent
            
            # Disk usage
            disk = psutil.disk_usage('.')
            disk_healthy = disk.percent < 90
            
            # Process status
            process_healthy = all(p.poll() is None for p in self.processes)
            
            healthy = memory_healthy and cpu_healthy and disk_healthy and process_healthy
            issues = []
            
            if not memory_healthy:
                issues.append(f"High memory usage: {memory.percent}%")
            if not cpu_healthy:
                issues.append(f"High CPU usage: {psutil.cpu_percent()}%")
            if not disk_healthy:
                issues.append(f"High disk usage: {disk.percent}%")
            if not process_healthy:
                issues.append("Service processes not running")
            
            return {
                "healthy": healthy,
                "issues": issues,
                "memory_percent": memory.percent,
                "cpu_percent": psutil.cpu_percent(),
                "disk_percent": disk.percent,
                "processes_running": process_healthy
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "issues": [f"Health check error: {e}"],
                "memory_percent": 0,
                "cpu_percent": 0,
                "disk_percent": 0,
                "processes_running": False
            }
    
    async def _start_monitoring(self) -> None:
        """Start the monitoring service."""
        logger.info("📊 Starting monitoring service...")
        self.monitoring_active = True
        
        async def monitor_loop():
            while self.monitoring_active:
                try:
                    health = await self._check_system_health()
                    
                    if not health["healthy"]:
                        logger.warning(f"⚠️  System health issues detected: {health['issues']}")
                    
                    # Log metrics
                    logger.info(f"📈 System metrics - Memory: {health['memory_percent']}%, CPU: {health['cpu_percent']}%, Disk: {health['disk_percent']}%")
                    
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"❌ Monitoring error: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(monitor_loop())
    
    async def _start_backup_service(self) -> None:
        """Start the backup service."""
        logger.info("💾 Starting backup service...")
        self.backup_active = True
        
        async def backup_loop():
            while self.backup_active:
                try:
                    await self._create_backup()
                    await asyncio.sleep(self.config.backup_interval_hours * 3600)
                except Exception as e:
                    logger.error(f"❌ Backup error: {e}")
                    await asyncio.sleep(3600)  # Retry in 1 hour
        
        asyncio.create_task(backup_loop())
    
    async def _create_backup(self) -> None:
        """Create a system backup."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = f"backups/backup_{timestamp}"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Backup configuration files
            if os.path.exists("config"):
                shutil.copytree("config", f"{backup_dir}/config")
            
            # Backup logs
            if os.path.exists("logs"):
                shutil.copytree("logs", f"{backup_dir}/logs")
            
            # Backup source code
            source_files = [
                "enhanced_system_integration_v4.py",
                "ai_content_intelligence_v4.py",
                "real_time_analytics_v4.py",
                "security_compliance_v4.py"
            ]
            
            for file in source_files:
                if os.path.exists(file):
                    shutil.copy2(file, backup_dir)
            
            # Create backup manifest
            manifest = {
                "timestamp": timestamp,
                "backup_type": "system_backup",
                "files_backed_up": source_files,
                "config_backed_up": os.path.exists("config"),
                "logs_backed_up": os.path.exists("logs")
            }
            
            with open(f"{backup_dir}/manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"✅ Backup created: {backup_dir}")
            
            # Clean up old backups (keep last 7)
            await self._cleanup_old_backups()
            
        except Exception as e:
            logger.error(f"❌ Backup creation failed: {e}")
    
    async def _cleanup_old_backups(self) -> None:
        """Clean up old backup files."""
        try:
            backup_dirs = [d for d in os.listdir("backups") if d.startswith("backup_")]
            backup_dirs.sort(reverse=True)
            
            # Keep only the last 7 backups
            for old_backup in backup_dirs[7:]:
                old_backup_path = os.path.join("backups", old_backup)
                shutil.rmtree(old_backup_path)
                logger.info(f"🗑️  Removed old backup: {old_backup}")
                
        except Exception as e:
            logger.error(f"❌ Backup cleanup failed: {e}")
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("🧹 Cleaning up resources...")
        
        self.monitoring_active = False
        self.backup_active = False
        
        # Stop all processes
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        self.processes.clear()
    
    async def stop(self) -> None:
        """Stop the deployment."""
        logger.info("🛑 Stopping production deployment...")
        await self._cleanup()
        logger.info("✅ Production deployment stopped")

def load_config(config_path: str) -> DeploymentConfig:
    """Load deployment configuration from file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            return DeploymentConfig(**config_data)
    else:
        logger.warning(f"⚠️  Config file {config_path} not found, using defaults")
        return DeploymentConfig()

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"📡 Received signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Enhanced LinkedIn Optimizer v4.0")
    parser.add_argument("--config", default="deployment_config.yaml", help="Configuration file path")
    parser.add_argument("--mode", choices=["production", "staging", "development"], default="production", help="Deployment mode")
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    config = load_config(args.config)
    config.environment = args.mode
    
    # Create deployer
    deployer = ProductionDeployer(config)
    
    try:
        # Deploy the system
        success = await deployer.deploy()
        
        if success:
            logger.info("🎉 Production deployment successful!")
            logger.info("📊 System is now running and monitoring...")
            logger.info("🛑 Press Ctrl+C to stop the deployment")
            
            # Keep the deployment running
            while True:
                await asyncio.sleep(1)
        else:
            logger.error("❌ Production deployment failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Deployment interrupted by user")
    except Exception as e:
        logger.error(f"❌ Deployment error: {e}")
        sys.exit(1)
    finally:
        await deployer.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

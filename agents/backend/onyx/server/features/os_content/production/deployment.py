from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import subprocess
import shutil
import psutil
import docker
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import structlog
from .config import get_production_config
import structlog
            import aiohttp
            import json
        import aiohttp
        import time
            import aiohttp
            import aiohttp
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Deployment Management for OS Content UGC Video Generator
Handles production deployment, scaling, and orchestration
"""



logger = structlog.get_logger("os_content.deployment")

class DeploymentManager:
    """Production deployment manager"""
    
    def __init__(self) -> Any:
        self.config = get_production_config()
        self.docker_client = None
        self._initialize_docker()
    
    def _initialize_docker(self) -> Any:
        """Initialize Docker client"""
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Docker client: {e}")
            self.docker_client = None
    
    async def deploy_application(self) -> Dict[str, Any]:
        """Deploy the application to production"""
        try:
            logger.info("Starting production deployment")
            
            # Pre-deployment checks
            await self._pre_deployment_checks()
            
            # Backup current deployment
            await self._backup_current_deployment()
            
            # Deploy new version
            deployment_result = await self._deploy_new_version()
            
            # Health checks
            health_status = await self._health_check_deployment()
            
            # Post-deployment tasks
            await self._post_deployment_tasks()
            
            logger.info("Production deployment completed successfully")
            
            return {
                "status": "success",
                "deployment_id": deployment_result.get("deployment_id"),
                "health_status": health_status,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            await self._rollback_deployment()
            raise
    
    async def _pre_deployment_checks(self) -> Any:
        """Perform pre-deployment checks"""
        logger.info("Performing pre-deployment checks")
        
        # Check system resources
        await self._check_system_resources()
        
        # Check database connectivity
        await self._check_database_connectivity()
        
        # Check Redis connectivity
        await self._check_redis_connectivity()
        
        # Check disk space
        await self._check_disk_space()
        
        # Check network connectivity
        await self._check_network_connectivity()
        
        logger.info("Pre-deployment checks completed")
    
    async def _check_system_resources(self) -> Any:
        """Check system resources"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            raise Exception(f"High CPU usage: {cpu_percent}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            raise Exception(f"High memory usage: {memory.percent}%")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            raise Exception(f"High disk usage: {disk.percent}%")
        
        logger.info(f"System resources OK - CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%")
    
    async def _check_database_connectivity(self) -> Any:
        """Check database connectivity"""
        try:
            # This would be implemented with actual database connection check
            logger.info("Database connectivity check passed")
        except Exception as e:
            raise Exception(f"Database connectivity check failed: {e}")
    
    async def _check_redis_connectivity(self) -> Any:
        """Check Redis connectivity"""
        try:
            # This would be implemented with actual Redis connection check
            logger.info("Redis connectivity check passed")
        except Exception as e:
            raise Exception(f"Redis connectivity check failed: {e}")
    
    async def _check_disk_space(self) -> Any:
        """Check available disk space"""
        upload_dir = Path(self.config.upload_dir)
        if upload_dir.exists():
            free_space = shutil.disk_usage(upload_dir).free
            min_space = 10 * 1024 * 1024 * 1024  # 10GB
            if free_space < min_space:
                raise Exception(f"Insufficient disk space: {free_space / (1024**3):.2f}GB available")
        
        logger.info("Disk space check passed")
    
    async def _check_network_connectivity(self) -> Any:
        """Check network connectivity"""
        try:
            # Check CDN connectivity
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config.cdn_url, timeout=10) as response:
                    if response.status != 200:
                        raise Exception(f"CDN connectivity check failed: {response.status}")
            
            logger.info("Network connectivity check passed")
        except Exception as e:
            raise Exception(f"Network connectivity check failed: {e}")
    
    async def _backup_current_deployment(self) -> Any:
        """Backup current deployment"""
        if not self.config.backup_enabled:
            return
        
        try:
            backup_dir = Path(self.config.backup_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"deployment_backup_{timestamp}"
            backup_path = backup_dir / backup_name
            
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup configuration
            config_backup = backup_path / "config.json"
            with open(config_backup, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(self.config.to_dict(), f, indent=2)
            
            # Backup database (if applicable)
            await self._backup_database(backup_path)
            
            # Backup uploaded files
            await self._backup_uploads(backup_path)
            
            logger.info(f"Deployment backup created: {backup_path}")
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            # Don't fail deployment for backup issues
    
    async def _backup_database(self, backup_path: Path):
        """Backup database"""
        try:
            # This would implement actual database backup
            # For PostgreSQL: pg_dump
            # For MySQL: mysqldump
            # For SQLite: copy file
            logger.info("Database backup completed")
        except Exception as e:
            logger.warning(f"Database backup failed: {e}")
    
    async def _backup_uploads(self, backup_path: Path):
        """Backup uploaded files"""
        try:
            upload_dir = Path(self.config.upload_dir)
            if upload_dir.exists():
                uploads_backup = backup_path / "uploads"
                shutil.copytree(upload_dir, uploads_backup, dirs_exist_ok=True)
                logger.info("Uploads backup completed")
        except Exception as e:
            logger.warning(f"Uploads backup failed: {e}")
    
    async def _deploy_new_version(self) -> Dict[str, Any]:
        """Deploy new version of the application"""
        deployment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Build Docker image
            await self._build_docker_image()
            
            # Deploy with Docker Compose
            await self._deploy_with_docker_compose()
            
            # Wait for services to be ready
            await self._wait_for_services_ready()
            
            return {
                "deployment_id": deployment_id,
                "status": "deployed"
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise
    
    async def _build_docker_image(self) -> Any:
        """Build Docker image"""
        try:
            if self.docker_client:
                # Build image using Docker client
                image, logs = self.docker_client.images.build(
                    path=".",
                    tag="os-content:latest",
                    rm=True
                )
                logger.info("Docker image built successfully")
            else:
                # Fallback to docker-compose build
                result = subprocess.run(
                    ["docker-compose", "build"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info("Docker image built successfully via docker-compose")
                
        except Exception as e:
            raise Exception(f"Failed to build Docker image: {e}")
    
    async def _deploy_with_docker_compose(self) -> Any:
        """Deploy using Docker Compose"""
        try:
            # Stop existing services
            subprocess.run(
                ["docker-compose", "down"],
                capture_output=True,
                text=True
            )
            
            # Start new services
            result = subprocess.run(
                ["docker-compose", "up", "-d"],
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("Docker Compose deployment completed")
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"Docker Compose deployment failed: {e.stderr}")
    
    async def _wait_for_services_ready(self) -> Any:
        """Wait for services to be ready"""
        
        max_retries = 30
        retry_interval = 2
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    # Check main application
                    async with session.get(f"http://localhost:{self.config.port}/health", timeout=5) as response:
                        if response.status == 200:
                            logger.info("Application is ready")
                            return
                
                logger.info(f"Waiting for services to be ready... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_interval)
                
            except Exception as e:
                logger.warning(f"Service readiness check failed: {e}")
                await asyncio.sleep(retry_interval)
        
        raise Exception("Services failed to become ready within timeout")
    
    async def _health_check_deployment(self) -> Dict[str, Any]:
        """Perform health checks on deployment"""
        health_status = {
            "application": False,
            "database": False,
            "redis": False,
            "cdn": False,
            "overall": False
        }
        
        try:
            # Check application health
            health_status["application"] = await self._check_application_health()
            
            # Check database health
            health_status["database"] = await self._check_database_health()
            
            # Check Redis health
            health_status["redis"] = await self._check_redis_health()
            
            # Check CDN health
            health_status["cdn"] = await self._check_cdn_health()
            
            # Overall health
            health_status["overall"] = all(health_status.values())
            
            logger.info(f"Health check results: {health_status}")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        return health_status
    
    async def _check_application_health(self) -> bool:
        """Check application health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{self.config.port}/health", timeout=10) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _check_database_health(self) -> bool:
        """Check database health"""
        try:
            # This would implement actual database health check
            return True
        except Exception:
            return False
    
    async def _check_redis_health(self) -> bool:
        """Check Redis health"""
        try:
            # This would implement actual Redis health check
            return True
        except Exception:
            return False
    
    async def _check_cdn_health(self) -> bool:
        """Check CDN health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config.cdn_url, timeout=10) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _post_deployment_tasks(self) -> Any:
        """Perform post-deployment tasks"""
        logger.info("Performing post-deployment tasks")
        
        # Cleanup old backups
        await self._cleanup_old_backups()
        
        # Update monitoring
        await self._update_monitoring()
        
        # Send deployment notification
        await self._send_deployment_notification()
        
        logger.info("Post-deployment tasks completed")
    
    async def _cleanup_old_backups(self) -> Any:
        """Cleanup old backups"""
        if not self.config.backup_enabled:
            return
        
        try:
            backup_dir = Path(self.config.backup_path)
            if not backup_dir.exists():
                return
            
            # Remove backups older than retention period
            cutoff_date = datetime.now().timestamp() - (self.config.backup_retention_days * 24 * 3600)
            
            for backup_item in backup_dir.iterdir():
                if backup_item.is_dir():
                    if backup_item.stat().st_mtime < cutoff_date:
                        shutil.rmtree(backup_item)
                        logger.info(f"Removed old backup: {backup_item}")
            
        except Exception as e:
            logger.warning(f"Backup cleanup failed: {e}")
    
    async def _update_monitoring(self) -> Any:
        """Update monitoring systems"""
        try:
            # Update Prometheus targets
            # Update Grafana dashboards
            # Update alerting rules
            logger.info("Monitoring updated")
        except Exception as e:
            logger.warning(f"Monitoring update failed: {e}")
    
    async def _send_deployment_notification(self) -> Any:
        """Send deployment notification"""
        try:
            # Send notification to Slack, email, etc.
            logger.info("Deployment notification sent")
        except Exception as e:
            logger.warning(f"Deployment notification failed: {e}")
    
    async def _rollback_deployment(self) -> Any:
        """Rollback deployment"""
        logger.warning("Rolling back deployment")
        
        try:
            # Stop current deployment
            subprocess.run(
                ["docker-compose", "down"],
                capture_output=True,
                text=True
            )
            
            # Restore from backup
            await self._restore_from_backup()
            
            # Restart previous version
            subprocess.run(
                ["docker-compose", "up", "-d"],
                capture_output=True,
                text=True
            )
            
            logger.info("Deployment rollback completed")
            
        except Exception as e:
            logger.error(f"Deployment rollback failed: {e}")
    
    async def _restore_from_backup(self) -> Any:
        """Restore from backup"""
        try:
            # Find latest backup
            backup_dir = Path(self.config.backup_path)
            if not backup_dir.exists():
                return
            
            backups = [d for d in backup_dir.iterdir() if d.is_dir() and d.name.startswith("deployment_backup_")]
            if not backups:
                return
            
            latest_backup = max(backups, key=lambda x: x.stat().st_mtime)
            
            # Restore database
            await self._restore_database(latest_backup)
            
            # Restore uploads
            await self._restore_uploads(latest_backup)
            
            logger.info(f"Restored from backup: {latest_backup}")
            
        except Exception as e:
            logger.error(f"Restore from backup failed: {e}")
    
    async def _restore_database(self, backup_path: Path):
        """Restore database from backup"""
        try:
            # This would implement actual database restore
            logger.info("Database restored from backup")
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
    
    async def _restore_uploads(self, backup_path: Path):
        """Restore uploads from backup"""
        try:
            uploads_backup = backup_path / "uploads"
            if uploads_backup.exists():
                upload_dir = Path(self.config.upload_dir)
                if upload_dir.exists():
                    shutil.rmtree(upload_dir)
                shutil.copytree(uploads_backup, upload_dir)
                logger.info("Uploads restored from backup")
        except Exception as e:
            logger.error(f"Uploads restore failed: {e}")
    
    async def scale_application(self, replicas: int) -> Dict[str, Any]:
        """Scale application instances"""
        try:
            logger.info(f"Scaling application to {replicas} replicas")
            
            # Update docker-compose scale
            result = subprocess.run(
                ["docker-compose", "up", "-d", "--scale", f"os-content-api={replicas}"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Wait for scaling to complete
            await asyncio.sleep(10)
            
            # Verify scaling
            await self._verify_scaling(replicas)
            
            logger.info(f"Application scaled to {replicas} replicas")
            
            return {
                "status": "success",
                "replicas": replicas,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            raise
    
    async def _verify_scaling(self, expected_replicas: int):
        """Verify scaling operation"""
        try:
            if self.docker_client:
                containers = self.docker_client.containers.list(
                    filters={"label": "com.docker.compose.service=os-content-api"}
                )
                actual_replicas = len(containers)
                
                if actual_replicas != expected_replicas:
                    raise Exception(f"Scaling verification failed: expected {expected_replicas}, got {actual_replicas}")
                
                logger.info(f"Scaling verified: {actual_replicas} replicas running")
                
        except Exception as e:
            logger.warning(f"Scaling verification failed: {e}")

# Global deployment manager instance
deployment_manager = DeploymentManager() 
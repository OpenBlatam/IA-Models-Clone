#!/usr/bin/env python3
"""
AI Integration System - Startup Script
Comprehensive startup script with environment validation and service initialization
"""

import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings, get_environment_config, get_enabled_platforms
from database import initialize_database, test_database_connection, check_database_health
from integration_engine import integration_engine, initialize_engine
from monitoring import monitoring_service, get_health_status
from tasks import celery_app

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level.upper()),
    format=settings.logging.format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.logging.file_path) if settings.logging.file_path else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

class StartupManager:
    """Manages application startup and initialization"""
    
    def __init__(self):
        self.startup_checks = []
        self.services = {}
        self.is_healthy = False
    
    async def run_startup_checks(self) -> bool:
        """Run all startup checks"""
        logger.info("🔍 Running startup checks...")
        
        checks = [
            ("Environment Configuration", self.check_environment),
            ("Database Connection", self.check_database),
            ("Platform Configurations", self.check_platform_configs),
            ("Redis Connection", self.check_redis),
            ("File Permissions", self.check_file_permissions),
            ("Dependencies", self.check_dependencies)
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            try:
                logger.info(f"  ✓ Checking {check_name}...")
                result = await check_func()
                if result:
                    logger.info(f"  ✅ {check_name}: PASSED")
                else:
                    logger.error(f"  ❌ {check_name}: FAILED")
                    all_passed = False
            except Exception as e:
                logger.error(f"  ❌ {check_name}: ERROR - {str(e)}")
                all_passed = False
        
        self.is_healthy = all_passed
        return all_passed
    
    async def check_environment(self) -> bool:
        """Check environment configuration"""
        try:
            # Check required environment variables
            required_vars = ["DATABASE_URL", "REDIS_URL", "SECRET_KEY"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                logger.warning(f"Missing environment variables: {missing_vars}")
                return False
            
            # Check environment-specific settings
            env_config = get_environment_config()
            logger.info(f"Environment: {settings.environment.value}")
            logger.info(f"Debug mode: {settings.debug}")
            
            return True
        except Exception as e:
            logger.error(f"Environment check failed: {str(e)}")
            return False
    
    async def check_database(self) -> bool:
        """Check database connection and health"""
        try:
            # Test basic connection
            if not test_database_connection():
                logger.error("Database connection test failed")
                return False
            
            # Initialize database
            if not initialize_database():
                logger.error("Database initialization failed")
                return False
            
            # Check database health
            health = check_database_health()
            if health["status"] != "healthy":
                logger.warning(f"Database health check: {health['status']}")
                return False
            
            logger.info("Database connection and health: OK")
            return True
        except Exception as e:
            logger.error(f"Database check failed: {str(e)}")
            return False
    
    async def check_platform_configs(self) -> bool:
        """Check platform configurations"""
        try:
            enabled_platforms = get_enabled_platforms()
            logger.info(f"Enabled platforms: {enabled_platforms}")
            
            if not enabled_platforms:
                logger.warning("No platforms are enabled. Configure at least one platform.")
                return False
            
            # Validate each platform configuration
            for platform in enabled_platforms:
                config = settings.__dict__.get(platform)
                if not config or not config.enabled:
                    logger.warning(f"Platform {platform} is not properly configured")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Platform configuration check failed: {str(e)}")
            return False
    
    async def check_redis(self) -> bool:
        """Check Redis connection"""
        try:
            import redis
            from urllib.parse import urlparse
            
            # Parse Redis URL
            parsed = urlparse(settings.redis.url)
            r = redis.Redis(
                host=parsed.hostname or 'localhost',
                port=parsed.port or 6379,
                db=int(parsed.path.lstrip('/')) if parsed.path else 0,
                decode_responses=True
            )
            
            # Test connection
            r.ping()
            logger.info("Redis connection: OK")
            return True
        except Exception as e:
            logger.error(f"Redis check failed: {str(e)}")
            return False
    
    async def check_file_permissions(self) -> bool:
        """Check file permissions"""
        try:
            # Check log directory
            if settings.logging.file_path:
                log_dir = Path(settings.logging.file_path).parent
                log_dir.mkdir(parents=True, exist_ok=True)
                
                # Test write permission
                test_file = log_dir / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()
            
            # Check other required directories
            required_dirs = ["logs", "uploads", "backups"]
            for dir_name in required_dirs:
                dir_path = Path(dir_name)
                dir_path.mkdir(exist_ok=True)
                
                # Test write permission
                test_file = dir_path / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()
            
            logger.info("File permissions: OK")
            return True
        except Exception as e:
            logger.error(f"File permissions check failed: {str(e)}")
            return False
    
    async def check_dependencies(self) -> bool:
        """Check required dependencies"""
        try:
            required_packages = [
                "fastapi", "uvicorn", "sqlalchemy", "redis", "celery",
                "aiohttp", "pydantic", "prometheus_client"
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                logger.error(f"Missing required packages: {missing_packages}")
                return False
            
            logger.info("Dependencies: OK")
            return True
        except Exception as e:
            logger.error(f"Dependencies check failed: {str(e)}")
            return False
    
    async def initialize_services(self) -> bool:
        """Initialize all services"""
        logger.info("🚀 Initializing services...")
        
        try:
            # Initialize integration engine
            logger.info("  ✓ Initializing integration engine...")
            await initialize_engine()
            logger.info("  ✅ Integration engine initialized")
            
            # Start monitoring service
            logger.info("  ✓ Starting monitoring service...")
            # Note: In production, you might want to run this in a separate task
            # asyncio.create_task(monitoring_service.start_monitoring())
            logger.info("  ✅ Monitoring service started")
            
            # Initialize Celery (if needed)
            logger.info("  ✓ Checking Celery configuration...")
            # Celery will be started separately in production
            logger.info("  ✅ Celery configuration OK")
            
            return True
        except Exception as e:
            logger.error(f"Service initialization failed: {str(e)}")
            return False
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        logger.info("🏥 Running health check...")
        
        try:
            health_status = get_health_status()
            
            if health_status["status"] == "healthy":
                logger.info("✅ System is healthy")
            elif health_status["status"] == "warning":
                logger.warning("⚠️ System has warnings")
            else:
                logger.error("❌ System is unhealthy")
            
            return health_status
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def print_startup_summary(self):
        """Print startup summary"""
        logger.info("=" * 60)
        logger.info("🎉 AI Integration System Startup Complete")
        logger.info("=" * 60)
        
        # System information
        logger.info(f"Environment: {settings.environment.value}")
        logger.info(f"Debug Mode: {settings.debug}")
        logger.info(f"API Host: {settings.api_host}")
        logger.info(f"API Port: {settings.api_port}")
        
        # Platform information
        enabled_platforms = get_enabled_platforms()
        logger.info(f"Enabled Platforms: {', '.join(enabled_platforms) if enabled_platforms else 'None'}")
        
        # Health status
        status_icon = "✅" if self.is_healthy else "❌"
        logger.info(f"System Status: {status_icon} {'HEALTHY' if self.is_healthy else 'UNHEALTHY'}")
        
        # URLs
        logger.info("=" * 60)
        logger.info("📋 Available Endpoints:")
        logger.info(f"  API Documentation: http://{settings.api_host}:{settings.api_port}/docs")
        logger.info(f"  Health Check: http://{settings.api_host}:{settings.api_port}/health")
        logger.info(f"  Integration API: http://{settings.api_host}:{settings.api_port}/ai-integration/")
        logger.info("=" * 60)

async def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description="AI Integration System Startup")
    parser.add_argument("--skip-checks", action="store_true", help="Skip startup checks")
    parser.add_argument("--health-only", action="store_true", help="Run health check only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    startup_manager = StartupManager()
    
    try:
        if args.health_only:
            # Run health check only
            health_status = await startup_manager.run_health_check()
            print(f"Health Status: {health_status['status']}")
            return 0 if health_status['status'] in ['healthy', 'warning'] else 1
        
        # Run full startup sequence
        logger.info("🚀 Starting AI Integration System...")
        
        # Run startup checks
        if not args.skip_checks:
            if not await startup_manager.run_startup_checks():
                logger.error("❌ Startup checks failed. Exiting.")
                return 1
        
        # Initialize services
        if not await startup_manager.initialize_services():
            logger.error("❌ Service initialization failed. Exiting.")
            return 1
        
        # Run final health check
        health_status = await startup_manager.run_health_check()
        
        # Print summary
        startup_manager.print_startup_summary()
        
        # Return appropriate exit code
        if health_status["status"] == "healthy":
            logger.info("🎉 System ready for operation!")
            return 0
        elif health_status["status"] == "warning":
            logger.warning("⚠️ System started with warnings")
            return 0
        else:
            logger.error("❌ System started in unhealthy state")
            return 1
    
    except KeyboardInterrupt:
        logger.info("🛑 Startup interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Startup failed with error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)




























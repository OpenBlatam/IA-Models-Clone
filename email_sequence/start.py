#!/usr/bin/env python3
"""
Email Sequence AI System - Startup Script

This script provides various ways to start the email sequence system
with different configurations and options.
"""

import asyncio
import sys
import os
import argparse
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import get_settings
from core.dependencies import init_database, init_redis, init_services
from core.monitoring import init_monitoring
from main import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def setup_services():
    """Setup all required services"""
    logger.info("Setting up services...")
    
    try:
        # Initialize database
        await init_database()
        logger.info("✅ Database initialized")
        
        # Initialize Redis
        await init_redis()
        logger.info("✅ Redis initialized")
        
        # Initialize services
        await init_services()
        logger.info("✅ Services initialized")
        
        # Initialize monitoring
        await init_monitoring()
        logger.info("✅ Monitoring initialized")
        
        logger.info("🎉 All services setup successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error setting up services: {e}")
        raise


async def health_check():
    """Perform health check on all services"""
    logger.info("Performing health check...")
    
    from core.dependencies import check_database_health, check_redis_health, check_services_health
    
    try:
        # Check database
        db_healthy = await check_database_health()
        logger.info(f"Database health: {'✅ Healthy' if db_healthy else '❌ Unhealthy'}")
        
        # Check Redis
        redis_healthy = await check_redis_health()
        logger.info(f"Redis health: {'✅ Healthy' if redis_healthy else '❌ Unhealthy'}")
        
        # Check services
        services_healthy = await check_services_health()
        logger.info(f"Services health: {'✅ Healthy' if services_healthy else '❌ Unhealthy'}")
        
        if db_healthy and redis_healthy and services_healthy:
            logger.info("🎉 All services are healthy!")
            return True
        else:
            logger.error("❌ Some services are unhealthy!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False


def start_development_server():
    """Start development server with auto-reload"""
    import uvicorn
    
    settings = get_settings()
    
    logger.info("🚀 Starting development server...")
    logger.info(f"📊 API Documentation: http://{settings.api_host}:{settings.api_port}/docs")
    logger.info(f"📈 Metrics: http://{settings.api_host}:{settings.metrics_port}")
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
        access_log=True,
        use_colors=True
    )


def start_production_server():
    """Start production server with Gunicorn"""
    import subprocess
    
    settings = get_settings()
    
    logger.info("🚀 Starting production server...")
    logger.info(f"📊 API Documentation: http://{settings.api_host}:{settings.api_port}/docs")
    logger.info(f"📈 Metrics: http://{settings.api_host}:{settings.metrics_port}")
    
    # Gunicorn command
    cmd = [
        "gunicorn",
        "main:app",
        "-w", "4",
        "-k", "uvicorn.workers.UvicornWorker",
        "--bind", f"{settings.api_host}:{settings.api_port}",
        "--access-logfile", "-",
        "--error-logfile", "-",
        "--log-level", settings.log_level.lower()
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to start production server: {e}")
        sys.exit(1)


async def run_migrations():
    """Run database migrations"""
    logger.info("Running database migrations...")
    
    try:
        import subprocess
        result = subprocess.run(["alembic", "upgrade", "head"], check=True, capture_output=True, text=True)
        logger.info("✅ Migrations completed successfully")
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Migration failed: {e}")
        logger.error(e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        logger.warning("⚠️ Alembic not found. Skipping migrations.")


def create_sample_data():
    """Create sample data for testing"""
    logger.info("Creating sample data...")
    
    try:
        # This would create sample sequences, templates, etc.
        # For now, just log that it would be implemented
        logger.info("✅ Sample data creation would be implemented here")
    except Exception as e:
        logger.error(f"❌ Error creating sample data: {e}")


def show_configuration():
    """Show current configuration"""
    settings = get_settings()
    
    print("\n" + "="*60)
    print("📋 EMAIL SEQUENCE AI CONFIGURATION")
    print("="*60)
    print(f"App Name: {settings.app_name}")
    print(f"Version: {settings.app_version}")
    print(f"Environment: {settings.environment}")
    print(f"Debug: {settings.debug}")
    print(f"API Host: {settings.api_host}")
    print(f"API Port: {settings.api_port}")
    print(f"Database URL: {settings.database_url}")
    print(f"Redis URL: {settings.redis_url}")
    print(f"OpenAI Model: {settings.openai_model}")
    print(f"SMTP Host: {settings.smtp_host}")
    print(f"SMTP Port: {settings.smtp_port}")
    print(f"From Email: {settings.from_email}")
    print(f"Max Concurrent Sequences: {settings.max_concurrent_sequences}")
    print(f"Max Concurrent Emails: {settings.max_concurrent_emails}")
    print(f"Cache TTL: {settings.cache_ttl_seconds}s")
    print(f"Rate Limit: {settings.rate_limit_requests_per_minute}/min")
    print(f"Metrics Enabled: {settings.enable_metrics}")
    print(f"Metrics Port: {settings.metrics_port}")
    print("="*60 + "\n")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Email Sequence AI System")
    parser.add_argument(
        "command",
        choices=[
            "dev", "prod", "setup", "health", "migrate", 
            "sample-data", "config", "test"
        ],
        help="Command to run"
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override API host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override API port"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes (production only)"
    )
    
    args = parser.parse_args()
    
    # Override settings if provided
    if args.host or args.port:
        settings = get_settings()
        if args.host:
            settings.api_host = args.host
        if args.port:
            settings.api_port = args.port
    
    if args.command == "config":
        show_configuration()
        return
    
    if args.command == "setup":
        await setup_services()
        return
    
    if args.command == "health":
        healthy = await health_check()
        sys.exit(0 if healthy else 1)
    
    if args.command == "migrate":
        await run_migrations()
        return
    
    if args.command == "sample-data":
        create_sample_data()
        return
    
    if args.command == "test":
        logger.info("Running tests...")
        import subprocess
        try:
            subprocess.run(["pytest", "tests/", "-v"], check=True)
            logger.info("✅ All tests passed!")
        except subprocess.CalledProcessError:
            logger.error("❌ Some tests failed!")
            sys.exit(1)
        return
    
    # Setup services before starting server
    await setup_services()
    
    if args.command == "dev":
        start_development_server()
    elif args.command == "prod":
        start_production_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)































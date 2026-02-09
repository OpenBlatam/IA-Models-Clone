"""
Deployment Optimizations

Optimizations for:
- Docker optimization
- Container orchestration
- CI/CD pipelines
- Environment configuration
- Health checks
- Graceful shutdown
"""

import logging
import os
from typing import Optional, Dict, Any, List
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class DockerOptimizer:
    """Docker optimization utilities."""
    
    @staticmethod
    def create_optimized_dockerfile(
        base_image: str = "python:3.11-slim",
        workdir: str = "/app",
        requirements_file: str = "requirements.txt"
    ) -> str:
        """
        Generate optimized Dockerfile.
        
        Args:
            base_image: Base Docker image
            workdir: Working directory
            requirements_file: Requirements file path
            
        Returns:
            Dockerfile content
        """
        return f"""# Multi-stage build for smaller image
FROM {base_image} as builder

WORKDIR {workdir}

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY {requirements_file} .
RUN pip install --no-cache-dir --user -r {requirements_file}

# Final stage
FROM {base_image}

WORKDIR {workdir}

# Copy only installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Make sure scripts are executable
RUN chmod +x *.py

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8020

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8020/health')"

# Run application
CMD ["python", "main.py"]
"""
    
    @staticmethod
    def create_docker_compose(
        service_name: str = "suno-clone-ai",
        image: str = "suno-clone-ai:latest",
        port: int = 8020,
        replicas: int = 2
    ) -> str:
        """
        Generate optimized docker-compose.yml.
        
        Args:
            service_name: Service name
            image: Docker image
            port: Port to expose
            replicas: Number of replicas
            
        Returns:
            docker-compose.yml content
        """
        return f"""version: '3.8'

services:
  {service_name}:
    image: {image}
    ports:
      - "{port}:8020"
    environment:
      - PYTHONUNBUFFERED=1
      - USE_GPU=true
    deploy:
      replicas: {replicas}
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8020/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - suno-network

networks:
  suno-network:
    driver: bridge
"""


class GracefulShutdown:
    """Graceful shutdown handling."""
    
    def __init__(self):
        """Initialize graceful shutdown."""
        self.shutdown_event = None
        self.cleanup_handlers: List[callable] = []
    
    def register_cleanup(self, handler: callable) -> None:
        """
        Register cleanup handler.
        
        Args:
            handler: Cleanup function
        """
        self.cleanup_handlers.append(handler)
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def shutdown(self) -> None:
        """Execute shutdown."""
        logger.info("Shutting down gracefully...")
        
        # Run cleanup handlers
        for handler in self.cleanup_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.run(handler())
                else:
                    handler()
            except Exception as e:
                logger.error(f"Cleanup handler failed: {e}")
        
        logger.info("Shutdown complete")


class EnvironmentConfig:
    """Environment configuration optimization."""
    
    @staticmethod
    def load_config(
        env_file: str = ".env",
        required_vars: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Load environment configuration.
        
        Args:
            env_file: Environment file path
            required_vars: Required environment variables
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        # Load from .env file
        env_path = Path(env_file)
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
        
        # Override with environment variables
        for key in config.keys():
            env_value = os.environ.get(key)
            if env_value:
                config[key] = env_value
        
        # Check required variables
        if required_vars:
            missing = [var for var in required_vars if var not in config]
            if missing:
                raise ValueError(f"Missing required environment variables: {missing}")
        
        return config
    
    @staticmethod
    def get_optimized_settings() -> Dict[str, Any]:
        """
        Get optimized settings based on environment.
        
        Returns:
            Optimized settings dictionary
        """
        env = os.environ.get("ENVIRONMENT", "development")
        
        if env == "production":
            return {
                "debug": False,
                "workers": 4,
                "log_level": "INFO",
                "enable_cache": True,
                "enable_compression": True,
                "pool_size": 20,
            }
        elif env == "staging":
            return {
                "debug": False,
                "workers": 2,
                "log_level": "DEBUG",
                "enable_cache": True,
                "enable_compression": True,
                "pool_size": 10,
            }
        else:  # development
            return {
                "debug": True,
                "workers": 1,
                "log_level": "DEBUG",
                "enable_cache": False,
                "enable_compression": False,
                "pool_size": 5,
            }


import asyncio  # Add import









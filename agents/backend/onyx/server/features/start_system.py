#!/usr/bin/env python3
"""
Blatam Academy System Startup Script
====================================

Complete startup script for the integrated Blatam Academy system.
"""

import os
import sys
import subprocess
import time
import requests
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemStarter:
    """System startup manager."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.services = {
            "integration-system": {
                "port": 8000,
                "health_endpoint": "/health",
                "description": "Integration System (Main Gateway)"
            },
            "content-redundancy": {
                "port": 8001,
                "health_endpoint": "/health",
                "description": "Content Redundancy Detector"
            },
            "bul": {
                "port": 8002,
                "health_endpoint": "/health",
                "description": "BUL (Business Unlimited)"
            },
            "gamma-app": {
                "port": 8003,
                "health_endpoint": "/health",
                "description": "Gamma App"
            },
            "business-agents": {
                "port": 8004,
                "health_endpoint": "/health",
                "description": "Business Agents"
            },
            "export-ia": {
                "port": 8005,
                "health_endpoint": "/health",
                "description": "Export IA"
            }
        }
        
        self.docker_services = [
            "postgres",
            "redis",
            "integration-system",
            "content-redundancy",
            "bul",
            "gamma-app",
            "business-agents",
            "export-ia",
            "nginx",
            "prometheus",
            "grafana"
        ]
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed."""
        
        logger.info("Checking dependencies...")
        
        # Check Docker
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Docker is not installed or not running")
                return False
            logger.info(f"Docker found: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("Docker is not installed")
            return False
        
        # Check Docker Compose
        try:
            result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Docker Compose is not installed")
                return False
            logger.info(f"Docker Compose found: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("Docker Compose is not installed")
            return False
        
        # Check Python
        logger.info(f"Python version: {sys.version}")
        
        # Check required Python packages
        required_packages = [
            "fastapi",
            "uvicorn",
            "pydantic",
            "requests",
            "httpx"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"Python package {package} is available")
            except ImportError:
                logger.warning(f"Python package {package} is not installed")
        
        return True
    
    def create_environment_file(self) -> None:
        """Create environment file with default values."""
        
        env_file = self.base_path / ".env"
        
        if not env_file.exists():
            logger.info("Creating .env file with default values...")
            
            env_content = """# Blatam Academy System Environment Variables

# Application Settings
APP_NAME=Blatam Academy Integration System
APP_VERSION=1.0.0
DEBUG=false
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Security
SECRET_KEY=your-secure-secret-key-change-this-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
ALGORITHM=HS256

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/blatam_academy
DATABASE_ECHO=false

# Redis
REDIS_URL=redis://localhost:6379
REDIS_ENABLED=true

# System Endpoints
CONTENT_REDUNDANCY_ENDPOINT=http://localhost:8001
BUL_ENDPOINT=http://localhost:8002
GAMMA_APP_ENDPOINT=http://localhost:8003
BUSINESS_AGENTS_ENDPOINT=http://localhost:8004
EXPORT_IA_ENDPOINT=http://localhost:8005

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600

# Health Check
HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=10

# File Upload
MAX_FILE_SIZE=52428800
ALLOWED_FILE_TYPES=pdf,docx,txt,md,html,json,xml

# CORS
CORS_ORIGINS=*
CORS_METHODS=*
CORS_HEADERS=*

# Monitoring
METRICS_ENABLED=true
METRICS_INTERVAL=60

# Cache
CACHE_ENABLED=true
CACHE_TTL=300

# Integration
INTEGRATION_RETRY_ATTEMPTS=3
INTEGRATION_RETRY_DELAY=1

# API Keys (Optional - Add your keys here)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OPENROUTER_API_KEY=
"""
            
            with open(env_file, 'w') as f:
                f.write(env_content)
            
            logger.info("Environment file created. Please update API keys if needed.")
        else:
            logger.info("Environment file already exists")
    
    def start_docker_services(self) -> bool:
        """Start Docker services."""
        
        logger.info("Starting Docker services...")
        
        try:
            # Build and start services
            cmd = ["docker-compose", "up", "-d", "--build"]
            result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to start Docker services: {result.stderr}")
                return False
            
            logger.info("Docker services started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Docker services: {str(e)}")
            return False
    
    def wait_for_service(self, service_name: str, max_wait: int = 60) -> bool:
        """Wait for a service to be ready."""
        
        service_config = self.services.get(service_name)
        if not service_config:
            logger.warning(f"Unknown service: {service_name}")
            return False
        
        port = service_config["port"]
        health_endpoint = service_config["health_endpoint"]
        url = f"http://localhost:{port}{health_endpoint}"
        
        logger.info(f"Waiting for {service_name} to be ready...")
        
        for i in range(max_wait):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"{service_name} is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
            if i % 10 == 0 and i > 0:
                logger.info(f"Still waiting for {service_name}... ({i}/{max_wait})")
        
        logger.error(f"{service_name} failed to start within {max_wait} seconds")
        return False
    
    def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service."""
        
        service_config = self.services.get(service_name)
        if not service_config:
            return {"error": f"Unknown service: {service_name}"}
        
        port = service_config["port"]
        health_endpoint = service_config["health_endpoint"]
        url = f"http://localhost:{port}{health_endpoint}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response": response.json(),
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {
                    "status": "unhealthy",
                    "status_code": response.status_code,
                    "error": response.text
                }
        except requests.exceptions.RequestException as e:
            return {
                "status": "offline",
                "error": str(e)
            }
    
    def check_all_services(self) -> Dict[str, Any]:
        """Check health of all services."""
        
        logger.info("Checking health of all services...")
        
        results = {}
        healthy_count = 0
        
        for service_name in self.services:
            health = self.check_service_health(service_name)
            results[service_name] = health
            
            if health.get("status") == "healthy":
                healthy_count += 1
        
        results["summary"] = {
            "total_services": len(self.services),
            "healthy_services": healthy_count,
            "unhealthy_services": len(self.services) - healthy_count
        }
        
        return results
    
    def start_system(self) -> bool:
        """Start the complete system."""
        
        logger.info("Starting Blatam Academy System...")
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("Dependency check failed")
            return False
        
        # Create environment file
        self.create_environment_file()
        
        # Start Docker services
        if not self.start_docker_services():
            logger.error("Failed to start Docker services")
            return False
        
        # Wait for services to be ready
        logger.info("Waiting for services to be ready...")
        time.sleep(10)  # Initial wait for containers to start
        
        all_ready = True
        for service_name in self.services:
            if not self.wait_for_service(service_name, max_wait=120):
                all_ready = False
        
        if not all_ready:
            logger.error("Some services failed to start")
            return False
        
        # Final health check
        health_results = self.check_all_services()
        
        logger.info("System startup completed!")
        logger.info(f"Services status: {health_results['summary']}")
        
        # Print service URLs
        logger.info("\nService URLs:")
        logger.info("=" * 50)
        logger.info("Integration System (Main): http://localhost:8000")
        logger.info("Content Redundancy:       http://localhost:8001")
        logger.info("BUL:                      http://localhost:8002")
        logger.info("Gamma App:                http://localhost:8003")
        logger.info("Business Agents:          http://localhost:8004")
        logger.info("Export IA:                http://localhost:8005")
        logger.info("Nginx (Load Balancer):    http://localhost:80")
        logger.info("Prometheus (Metrics):     http://localhost:9090")
        logger.info("Grafana (Dashboard):      http://localhost:3000")
        logger.info("Kibana (Logs):            http://localhost:5601")
        logger.info("=" * 50)
        
        return True
    
    def stop_system(self) -> bool:
        """Stop the complete system."""
        
        logger.info("Stopping Blatam Academy System...")
        
        try:
            cmd = ["docker-compose", "down"]
            result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to stop Docker services: {result.stderr}")
                return False
            
            logger.info("System stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping system: {str(e)}")
            return False
    
    def restart_system(self) -> bool:
        """Restart the complete system."""
        
        logger.info("Restarting Blatam Academy System...")
        
        if not self.stop_system():
            return False
        
        time.sleep(5)  # Wait for services to stop
        
        return self.start_system()
    
    def show_status(self) -> None:
        """Show system status."""
        
        logger.info("Blatam Academy System Status")
        logger.info("=" * 50)
        
        health_results = self.check_all_services()
        
        for service_name, health in health_results.items():
            if service_name == "summary":
                continue
            
            service_config = self.services[service_name]
            status = health.get("status", "unknown")
            
            if status == "healthy":
                logger.info(f"✅ {service_config['description']:<30} - {status.upper()}")
            elif status == "unhealthy":
                logger.info(f"❌ {service_config['description']:<30} - {status.upper()}")
            else:
                logger.info(f"⚠️  {service_config['description']:<30} - {status.upper()}")
        
        summary = health_results.get("summary", {})
        logger.info("=" * 50)
        logger.info(f"Total Services: {summary.get('total_services', 0)}")
        logger.info(f"Healthy: {summary.get('healthy_services', 0)}")
        logger.info(f"Unhealthy: {summary.get('unhealthy_services', 0)}")

def main():
    """Main function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Blatam Academy System Manager")
    parser.add_argument("command", choices=["start", "stop", "restart", "status"], 
                       help="Command to execute")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    starter = SystemStarter()
    
    if args.command == "start":
        success = starter.start_system()
        sys.exit(0 if success else 1)
    elif args.command == "stop":
        success = starter.stop_system()
        sys.exit(0 if success else 1)
    elif args.command == "restart":
        success = starter.restart_system()
        sys.exit(0 if success else 1)
    elif args.command == "status":
        starter.show_status()
        sys.exit(0)

if __name__ == "__main__":
    main()

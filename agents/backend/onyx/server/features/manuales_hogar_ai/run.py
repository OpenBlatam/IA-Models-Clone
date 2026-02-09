#!/usr/bin/env python3
"""
Simple Python script to run Manuales Hogar AI
Usage: python run.py [dev|prod|staging] [--no-build] [--skip-health] [--migrate]
"""

import os
import sys
import subprocess
import time
import requests
import argparse
from pathlib import Path

def check_docker():
    """Check if Docker is running."""
    try:
        subprocess.run(["docker", "info"], 
                      capture_output=True, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not running. Please start Docker and try again.")
        return False

def check_env_file():
    """Check and create .env file if needed."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("⚠️  .env file not found. Creating from .env.example...")
            env_file.write_text(env_example.read_text())
            print("✅ Created .env file. Please edit it with your configuration.")
            print("   Required: OPENROUTER_API_KEY")
            input("Press Enter to continue or Ctrl+C to edit .env first...")
        else:
            print("❌ .env.example not found. Please create .env manually.")
            return False
    return True

def load_env():
    """Load environment variables from .env file."""
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

def start_services(environment="dev", no_build=False):
    """Start Docker services."""
    # Check for docker-compose or docker compose
    try:
        subprocess.run(["docker-compose", "--version"], 
                      capture_output=True, check=True)
        compose_cmd = "docker-compose"
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(["docker", "compose", "version"], 
                          capture_output=True, check=True)
            compose_cmd = "docker compose"
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ docker-compose is not installed")
            sys.exit(1)
    
    if environment == "prod":
        print("📦 Starting production environment...")
        cmd = [compose_cmd, "-f", "docker-compose.prod.yml", "up", "-d"]
        if not no_build:
            cmd.append("--build")
    else:
        print("🔧 Starting development environment...")
        cmd = [compose_cmd, "up", "-d"]
        if not no_build:
            cmd.append("--build")
    
    subprocess.run(cmd, check=True)
    return compose_cmd

def wait_for_health(max_retries=30):
    """Wait for service to be healthy."""
    print("⏳ Waiting for services to be ready...")
    time.sleep(5)
    
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/api/v1/health", timeout=2)
            if response.status_code == 200:
                print("✅ Service is healthy!")
                return True
        except requests.RequestException:
            pass
        
        print(f"   Waiting... ({i+1}/{max_retries})")
        time.sleep(2)
    
    return False

def check_port(port=8000):
    """Check if port is available."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Start Manuales Hogar AI')
    parser.add_argument('environment', nargs='?', default='dev', 
                       choices=['dev', 'prod', 'staging'],
                       help='Environment to run (default: dev)')
    parser.add_argument('--no-build', action='store_true',
                       help='Skip building Docker images')
    parser.add_argument('--skip-health', action='store_true',
                       help='Skip health check')
    parser.add_argument('--migrate', action='store_true',
                       help='Run database migrations')
    
    args = parser.parse_args()
    
    print("🚀 Starting Manuales Hogar AI...")
    print(f"Environment: {args.environment}")
    
    # Check prerequisites
    if not check_docker():
        sys.exit(1)
    
    if not check_env_file():
        sys.exit(1)
    
    # Check port availability
    if not check_port(8000):
        print("⚠️  Port 8000 is already in use.")
        print("   The service may not start correctly.")
        try:
            input("Press Enter to continue anyway or Ctrl+C to stop...")
        except KeyboardInterrupt:
            sys.exit(1)
    
    # Load environment
    load_env()
    
    # Check required variables
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  OPENROUTER_API_KEY not set in .env")
        print("   The service will start but API calls will fail.")
        try:
            input("Press Enter to continue anyway or Ctrl+C to set it first...")
        except KeyboardInterrupt:
            sys.exit(1)
    
    # Start services
    try:
        compose_cmd = start_services(args.environment, args.no_build)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start services: {e}")
        sys.exit(1)
    
    # Run migrations if requested
    if args.migrate:
        print("🔄 Running database migrations...")
        time.sleep(3)
        try:
            subprocess.run([compose_cmd, "exec", "-T", "app", "alembic", "upgrade", "head"], 
                          check=True, timeout=60)
            print("✅ Migrations completed")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("⚠️  Migration failed, continuing...")
    
    # Wait for health
    is_healthy = True
    if not args.skip_health:
        is_healthy = wait_for_health()
    
    if not is_healthy:
        print("⚠️  Service may not be ready yet. Check logs with: docker-compose logs")
    else:
        print("")
        print("🎉 Manuales Hogar AI is running!")
        print("")
        print("📍 API URL: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("❤️  Health: http://localhost:8000/api/v1/health")
        print("")
        print("📋 Useful commands:")
        print(f"   View logs:    {compose_cmd} logs -f")
        print(f"   Stop:         {compose_cmd} down")
        print(f"   Restart:      {compose_cmd} restart")
        print(f"   Shell:        {compose_cmd} exec app bash")
        print(f"   Status:       {compose_cmd} ps")
        print("")

if __name__ == "__main__":
    main()


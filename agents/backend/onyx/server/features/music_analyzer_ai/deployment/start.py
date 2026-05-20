#!/usr/bin/env python3
"""
One-command startup script for Music Analyzer AI
Cross-platform Python script
Usage: python start.py [dev|prod]
"""

import os
import sys
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

# Colors for terminal
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_colored(message, color=Colors.NC):
    """Print colored message"""
    print(f"{color}{message}{Colors.NC}")

def check_docker():
    """Check if Docker is running"""
    try:
        subprocess.run(['docker', 'info'], 
                     stdout=subprocess.DEVNULL, 
                     stderr=subprocess.DEVNULL, 
                     check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_docker_compose():
    """Check if docker-compose is available"""
    # Try new docker compose first
    try:
        subprocess.run(['docker', 'compose', 'version'], 
                     stdout=subprocess.DEVNULL, 
                     stderr=subprocess.DEVNULL, 
                     check=True)
        return 'docker compose'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try old docker-compose
    try:
        subprocess.run(['docker-compose', '--version'], 
                     stdout=subprocess.DEVNULL, 
                     stderr=subprocess.DEVNULL, 
                     check=True)
        return 'docker-compose'
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def create_env_file(env_path, environment):
    """Create .env file template"""
    import secrets
    
    env_content = f"""ENVIRONMENT={environment}
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
LOG_LEVEL=INFO
CACHE_ENABLED=true
POSTGRES_PASSWORD={secrets.token_urlsafe(32)}
REDIS_PASSWORD={secrets.token_urlsafe(32)}
GRAFANA_PASSWORD=admin
DATABASE_URL=postgresql://music_analyzer:changeme@postgres:5432/music_analyzer_db
"""
    with open(env_path, 'w') as f:
        f.write(env_content)

def check_health(url, max_retries=30, retry_delay=2):
    """Check if service is healthy"""
    for i in range(max_retries):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except (urllib.error.URLError, OSError):
            if i < max_retries - 1:
                print('.', end='', flush=True)
                time.sleep(retry_delay)
    return False

def main():
    # Get environment from command line
    environment = sys.argv[1] if len(sys.argv) > 1 else 'dev'
    
    # Normalize environment name
    if environment in ['development', 'dev']:
        environment = 'dev'
        compose_file = 'docker-compose.dev.yml'
    elif environment in ['production', 'prod']:
        environment = 'prod'
        compose_file = 'docker-compose.prod.yml'
    else:
        compose_file = 'docker-compose.yml'
    
    print_colored('🚀 Starting Music Analyzer AI...', Colors.GREEN)
    print_colored(f'📦 Mode: {environment}', Colors.YELLOW)
    
    # Check Docker
    if not check_docker():
        print_colored('❌ Docker is not running. Please start Docker first.', Colors.RED)
        sys.exit(1)
    
    # Check docker-compose
    docker_compose = check_docker_compose()
    if not docker_compose:
        print_colored('❌ docker-compose is not installed.', Colors.RED)
        sys.exit(1)
    
    # Navigate to deployment directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check for .env file
    env_path = script_dir.parent / '.env'
    if not env_path.exists() and environment != 'dev':
        print_colored('⚠️  .env file not found. Creating template...', Colors.YELLOW)
        create_env_file(env_path, environment)
        print_colored('📝 Please update .env with your actual credentials!', Colors.YELLOW)
    
    # Build images
    print_colored('🔨 Building Docker images...', Colors.GREEN)
    build_cmd = docker_compose.split() + ['-f', compose_file, 'build', '--quiet']
    subprocess.run(build_cmd, check=True)
    
    # Start services
    print_colored('🚀 Starting services...', Colors.GREEN)
    up_cmd = docker_compose.split() + ['-f', compose_file, 'up', '-d']
    subprocess.run(up_cmd, check=True)
    
    # Wait for services
    print_colored('⏳ Waiting for services to be ready...', Colors.GREEN)
    time.sleep(5)
    
    # Check health
    health_url = 'http://localhost:8010/health'
    healthy = check_health(health_url)
    print()  # New line after dots
    
    if healthy:
        print_colored('✅ All services are running!', Colors.GREEN)
        print()
        print_colored('📊 Service URLs:', Colors.GREEN)
        print('  🌐 API:          http://localhost:8010')
        print('  ❤️  Health:       http://localhost:8010/health')
        print('  📖 Docs:          http://localhost:8010/docs')
        
        if environment in ['dev', 'prod']:
            print('  📈 Grafana:       http://localhost:3000')
            print('  📊 Prometheus:    http://localhost:9090')
        
        print()
        print_colored('📝 Useful commands:', Colors.GREEN)
        print(f'  View logs:    {" ".join(docker_compose.split())} -f {compose_file} logs -f')
        print(f'  Stop:        {" ".join(docker_compose.split())} -f {compose_file} down')
        print(f'  Restart:     {" ".join(docker_compose.split())} -f {compose_file} restart')
        print()
    else:
        print_colored('⚠️  Services may still be starting. Check logs with:', Colors.YELLOW)
        print(f'  {" ".join(docker_compose.split())} -f {compose_file} logs')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print_colored('\n\n⚠️  Interrupted by user', Colors.YELLOW)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print_colored(f'\n❌ Error: {e}', Colors.RED)
        sys.exit(1)
    except Exception as e:
        print_colored(f'\n❌ Unexpected error: {e}', Colors.RED)
        sys.exit(1)





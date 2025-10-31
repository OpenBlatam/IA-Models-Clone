#!/usr/bin/env python3
"""
Gamma App - Deployment Script
Script to deploy Gamma App to production
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_docker():
    """Check if Docker is installed and running"""
    print("🐳 Checking Docker...")
    
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker found: {result.stdout.strip()}")
        else:
            print("❌ Docker not found")
            return False
    except FileNotFoundError:
        print("❌ Docker not installed")
        return False
    
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is running")
        else:
            print("❌ Docker is not running")
            return False
    except Exception as e:
        print(f"❌ Error checking Docker: {e}")
        return False
    
    return True

def check_docker_compose():
    """Check if Docker Compose is available"""
    print("🐳 Checking Docker Compose...")
    
    try:
        result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker Compose found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker Compose not found")
            return False
    except FileNotFoundError:
        print("❌ Docker Compose not installed")
        return False

def build_images():
    """Build Docker images"""
    print("🔨 Building Docker images...")
    
    try:
        result = subprocess.run([
            "docker-compose", "build", "--no-cache"
        ], cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print("✅ Docker images built successfully")
            return True
        else:
            print("❌ Failed to build Docker images")
            return False
    except Exception as e:
        print(f"❌ Error building images: {e}")
        return False

def run_tests():
    """Run tests before deployment"""
    print("🧪 Running tests...")
    
    try:
        result = subprocess.run([
            "python", "scripts/test.py"
        ], cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("❌ Tests failed")
            return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def create_env_file():
    """Create production .env file"""
    print("📝 Creating production environment file...")
    
    env_example = Path(__file__).parent.parent / "env.example"
    env_prod = Path(__file__).parent.parent / ".env.production"
    
    if env_example.exists():
        shutil.copy(env_example, env_prod)
        print("✅ Production environment file created")
        print("⚠️  Please edit .env.production with your production values")
        return True
    else:
        print("❌ env.example not found")
        return False

def deploy_services():
    """Deploy services with Docker Compose"""
    print("🚀 Deploying services...")
    
    try:
        # Stop existing services
        subprocess.run(["docker-compose", "down"], cwd=Path(__file__).parent.parent)
        
        # Start services
        result = subprocess.run([
            "docker-compose", "up", "-d"
        ], cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print("✅ Services deployed successfully")
            return True
        else:
            print("❌ Failed to deploy services")
            return False
    except Exception as e:
        print(f"❌ Error deploying services: {e}")
        return False

def check_services():
    """Check if services are running"""
    print("🔍 Checking service status...")
    
    try:
        result = subprocess.run([
            "docker-compose", "ps"
        ], cwd=Path(__file__).parent.parent, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("📊 Service Status:")
            print(result.stdout)
            return True
        else:
            print("❌ Failed to check service status")
            return False
    except Exception as e:
        print(f"❌ Error checking services: {e}")
        return False

def show_deployment_info():
    """Show deployment information"""
    print("\n" + "=" * 50)
    print("🎉 Deployment Complete!")
    print("=" * 50)
    print("📚 API Documentation: http://localhost:8030/docs")
    print("🔧 Health Check: http://localhost:8030/health")
    print("📊 Metrics: http://localhost:8031")
    print("=" * 50)
    print("📋 Useful Commands:")
    print("  docker-compose logs -f gamma_app    # View logs")
    print("  docker-compose ps                   # Check status")
    print("  docker-compose down                 # Stop services")
    print("  docker-compose restart gamma_app    # Restart app")
    print("=" * 50)

def main():
    """Main deployment function"""
    print("🚀 Gamma App Deployment Script")
    print("=" * 50)
    
    steps = [
        ("Docker Check", check_docker),
        ("Docker Compose Check", check_docker_compose),
        ("Create Environment File", create_env_file),
        ("Run Tests", run_tests),
        ("Build Images", build_images),
        ("Deploy Services", deploy_services),
        ("Check Services", check_services)
    ]
    
    for step_name, step_func in steps:
        print(f"\n🔍 {step_name}...")
        if not step_func():
            print(f"❌ {step_name} failed. Deployment aborted.")
            sys.exit(1)
        print(f"✅ {step_name} completed")
    
    show_deployment_info()

if __name__ == "__main__":
    main()




























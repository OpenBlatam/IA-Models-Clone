#!/usr/bin/env python3
"""
Deployment Script for Copywriting Service
========================================

Script to deploy the service to various environments.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command: str, description: str, check: bool = True) -> bool:
    """Run a shell command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def build_docker_image(tag: str = "copywriting-service:latest") -> bool:
    """Build Docker image"""
    return run_command(f"docker build -t {tag} .", f"Building Docker image {tag}")


def run_docker_compose(environment: str = "production") -> bool:
    """Run docker-compose for specified environment"""
    env_file = f".env.{environment}"
    if not Path(env_file).exists():
        print(f"❌ Environment file {env_file} not found")
        return False
    
    return run_command(
        f"docker-compose --env-file {env_file} up -d",
        f"Starting services with {environment} environment"
    )


def stop_docker_compose() -> bool:
    """Stop docker-compose services"""
    return run_command("docker-compose down", "Stopping services")


def run_tests() -> bool:
    """Run test suite"""
    return run_command("python -m pytest tests/ -v", "Running test suite")


def run_linting() -> bool:
    """Run code quality checks"""
    commands = [
        ("python -m black --check .", "Checking code formatting"),
        ("python -m isort --check-only .", "Checking import sorting"),
        ("python -m flake8 .", "Checking code style"),
        ("python -m mypy .", "Checking type hints")
    ]
    
    all_passed = True
    for command, description in commands:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed


def create_production_env() -> bool:
    """Create production environment file"""
    env_file = Path(".env.production")
    if env_file.exists():
        print("✅ Production environment file already exists")
        return True
    
    print("🔄 Creating production environment file...")
    try:
        with open("env.example", 'r') as f:
            content = f.read()
        
        # Update for production
        content = content.replace("ENVIRONMENT=development", "ENVIRONMENT=production")
        content = content.replace("API_RELOAD=true", "API_RELOAD=false")
        content = content.replace("API_DEBUG=false", "API_DEBUG=false")
        content = content.replace("LOG_LEVEL=INFO", "LOG_LEVEL=WARNING")
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Production environment file created")
        print("⚠️  Please update .env.production with your production values")
        return True
    except Exception as e:
        print(f"❌ Failed to create production environment file: {e}")
        return False


def deploy_local() -> bool:
    """Deploy locally"""
    print("🚀 Deploying locally...")
    
    steps = [
        ("Running tests", run_tests),
        ("Running linting", run_linting),
        ("Building Docker image", lambda: build_docker_image()),
        ("Starting services", lambda: run_docker_compose("local"))
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"❌ Deployment failed at: {step_name}")
            return False
    
    print("✅ Local deployment completed successfully!")
    print("🌐 Service available at: http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    return True


def deploy_production() -> bool:
    """Deploy to production"""
    print("🚀 Deploying to production...")
    
    # Create production environment file if it doesn't exist
    if not Path(".env.production").exists():
        if not create_production_env():
            return False
    
    steps = [
        ("Running tests", run_tests),
        ("Running linting", run_linting),
        ("Building Docker image", lambda: build_docker_image("copywriting-service:prod")),
        ("Starting production services", lambda: run_docker_compose("production"))
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"❌ Production deployment failed at: {step_name}")
            return False
    
    print("✅ Production deployment completed successfully!")
    return True


def deploy_staging() -> bool:
    """Deploy to staging"""
    print("🚀 Deploying to staging...")
    
    steps = [
        ("Running tests", run_tests),
        ("Building Docker image", lambda: build_docker_image("copywriting-service:staging")),
        ("Starting staging services", lambda: run_docker_compose("staging"))
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"❌ Staging deployment failed at: {step_name}")
            return False
    
    print("✅ Staging deployment completed successfully!")
    return True


def rollback() -> bool:
    """Rollback deployment"""
    print("🔄 Rolling back deployment...")
    
    # Stop current services
    if not stop_docker_compose():
        print("❌ Failed to stop services")
        return False
    
    # You could add logic here to restore from backup
    # For now, just stop the services
    print("✅ Rollback completed (services stopped)")
    return True


def status() -> bool:
    """Check deployment status"""
    print("📊 Checking deployment status...")
    
    commands = [
        ("docker-compose ps", "Service status"),
        ("docker-compose logs --tail=10", "Recent logs"),
        ("curl -f http://localhost:8000/api/v2/copywriting/health", "Health check")
    ]
    
    for command, description in commands:
        print(f"\n🔍 {description}:")
        run_command(command, "", check=False)
    
    return True


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy Copywriting Service")
    parser.add_argument(
        "environment",
        choices=["local", "staging", "production"],
        help="Deployment environment"
    )
    parser.add_argument(
        "--action",
        choices=["deploy", "rollback", "status"],
        default="deploy",
        help="Deployment action"
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Copywriting Service Deployment")
    print(f"Environment: {args.environment}")
    print(f"Action: {args.action}")
    print("=" * 50)
    
    if args.action == "deploy":
        if args.environment == "local":
            success = deploy_local()
        elif args.environment == "staging":
            success = deploy_staging()
        elif args.environment == "production":
            success = deploy_production()
    elif args.action == "rollback":
        success = rollback()
    elif args.action == "status":
        success = status()
    
    if success:
        print("\n✅ Operation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Operation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()































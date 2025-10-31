#!/usr/bin/env python3
"""
Ultimate Facebook Posts System Installer
========================================

Installs and configures the Ultimate Facebook Posts System v4.0
following FastAPI best practices and modern Python patterns.
"""

import asyncio
import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UltimateSystemInstaller:
    """Ultimate system installer"""
    
    def __init__(self, base_path: str = "."):
        """Initialize installer"""
        self.base_path = Path(base_path)
        self.install_log = []
        self.config = {}
        
        logger.info("Ultimate System Installer initialized")
    
    async def install(self) -> bool:
        """Run complete installation process"""
        try:
            logger.info("Starting Ultimate Facebook Posts System installation...")
            
            # Step 1: Check prerequisites
            if not await self.check_prerequisites():
                logger.error("Prerequisites check failed")
                return False
            
            # Step 2: Install dependencies
            if not await self.install_dependencies():
                logger.error("Dependencies installation failed")
                return False
            
            # Step 3: Setup configuration
            if not await self.setup_configuration():
                logger.error("Configuration setup failed")
                return False
            
            # Step 4: Initialize database
            if not await self.initialize_database():
                logger.error("Database initialization failed")
                return False
            
            # Step 5: Setup directories
            if not await self.setup_directories():
                logger.error("Directory setup failed")
                return False
            
            # Step 6: Run tests
            if not await self.run_tests():
                logger.warning("Tests failed - continuing with installation")
            
            # Step 7: Generate installation report
            await self.generate_installation_report()
            
            logger.info("Ultimate Facebook Posts System installed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return False
    
    async def check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        logger.info("Checking prerequisites...")
        
        prerequisites = {
            "python_version": self.check_python_version(),
            "pip_available": self.check_pip_available(),
            "git_available": self.check_git_available(),
            "disk_space": self.check_disk_space()
        }
        
        all_passed = all(prerequisites.values())
        
        if not all_passed:
            logger.error("Prerequisites check failed:")
            for check, passed in prerequisites.items():
                if not passed:
                    logger.error(f"  - {check}: FAILED")
        else:
            logger.info("All prerequisites passed")
        
        return all_passed
    
    def check_python_version(self) -> bool:
        """Check Python version"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_pip_available(self) -> bool:
        """Check if pip is available"""
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         check=True, capture_output=True)
            logger.info("pip is available")
            return True
        except subprocess.CalledProcessError:
            logger.error("pip is not available")
            return False
    
    def check_git_available(self) -> bool:
        """Check if git is available"""
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            logger.info("git is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("git is not available - some features may not work")
            return True  # Not critical
    
    def check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            import shutil
            free_space = shutil.disk_usage(self.base_path).free
            required_space = 1024 * 1024 * 1024  # 1GB
            
            if free_space < required_space:
                logger.error(f"Insufficient disk space: {free_space / (1024**3):.1f}GB available, 1GB required")
                return False
            
            logger.info(f"Disk space: {free_space / (1024**3):.1f}GB available")
            return True
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True
    
    async def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        logger.info("Installing dependencies...")
        
        try:
            # Install from requirements file
            requirements_file = self.base_path / "requirements_improved.txt"
            if requirements_file.exists():
                logger.info("Installing from requirements_improved.txt")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], check=True, capture_output=True, text=True)
                
                logger.info("Dependencies installed successfully")
                return True
            else:
                # Install basic dependencies
                basic_deps = [
                    "fastapi>=0.104.0",
                    "uvicorn[standard]>=0.24.0",
                    "pydantic>=2.5.0",
                    "pydantic-settings>=2.1.0",
                    "aiohttp>=3.9.0",
                    "redis>=5.0.0",
                    "structlog>=23.0.0"
                ]
                
                for dep in basic_deps:
                    logger.info(f"Installing {dep}")
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", dep
                    ], check=True, capture_output=True)
                
                logger.info("Basic dependencies installed successfully")
                return True
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    async def setup_configuration(self) -> bool:
        """Setup system configuration"""
        logger.info("Setting up configuration...")
        
        try:
            # Create .env file if it doesn't exist
            env_file = self.base_path / ".env"
            env_example = self.base_path / "env.example"
            
            if not env_file.exists() and env_example.exists():
                logger.info("Creating .env file from template")
                with open(env_example, 'r') as f:
                    content = f.read()
                
                with open(env_file, 'w') as f:
                    f.write(content)
                
                logger.info("Please update .env file with your configuration")
            
            # Create config directory
            config_dir = self.base_path / "config"
            config_dir.mkdir(exist_ok=True)
            
            # Create logs directory
            logs_dir = self.base_path / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Create data directory
            data_dir = self.base_path / "data"
            data_dir.mkdir(exist_ok=True)
            
            logger.info("Configuration setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration setup failed: {e}")
            return False
    
    async def initialize_database(self) -> bool:
        """Initialize database"""
        logger.info("Initializing database...")
        
        try:
            # For SQLite, just ensure the directory exists
            db_dir = self.base_path / "data"
            db_dir.mkdir(exist_ok=True)
            
            # Create database file
            db_file = db_dir / "facebook_posts.db"
            if not db_file.exists():
                db_file.touch()
                logger.info("Database file created")
            
            logger.info("Database initialized")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    async def setup_directories(self) -> bool:
        """Setup required directories"""
        logger.info("Setting up directories...")
        
        try:
            directories = [
                "logs",
                "data",
                "cache",
                "config",
                "temp",
                "backups"
            ]
            
            for dir_name in directories:
                dir_path = self.base_path / dir_name
                dir_path.mkdir(exist_ok=True)
                logger.info(f"Created directory: {dir_name}")
            
            logger.info("Directories setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Directory setup failed: {e}")
            return False
    
    async def run_tests(self) -> bool:
        """Run system tests"""
        logger.info("Running tests...")
        
        try:
            # Basic import test
            test_script = """
import sys
sys.path.append('.')
try:
    from launch_ultimate_system import create_app
    from api.routes import router
    from api.dependencies import get_facebook_engine
    print("All imports successful")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)
"""
            
            result = subprocess.run([
                sys.executable, "-c", test_script
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Basic tests passed")
                return True
            else:
                logger.error(f"Tests failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
    
    async def generate_installation_report(self) -> None:
        """Generate installation report"""
        logger.info("Generating installation report...")
        
        try:
            report = {
                "installation_timestamp": asyncio.get_event_loop().time(),
                "version": "4.0.0",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "install_log": self.install_log,
                "directories_created": [
                    "logs", "data", "cache", "config", "temp", "backups"
                ],
                "files_created": [
                    ".env", "config/", "logs/", "data/facebook_posts.db"
                ],
                "next_steps": [
                    "Update .env file with your configuration",
                    "Run: python launch_ultimate_system.py --mode dev",
                    "Visit http://localhost:8000/docs for API documentation",
                    "Configure your AI API key in .env file"
                ]
            }
            
            report_file = self.base_path / "INSTALLATION_REPORT.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Installation report saved to: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate installation report: {e}")


async def main():
    """Main installation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultimate Facebook Posts System Installer")
    parser.add_argument("--path", default=".", help="Installation path")
    parser.add_argument("--skip-tests", action="store_true", help="Skip tests")
    parser.add_argument("--force", action="store_true", help="Force installation")
    
    args = parser.parse_args()
    
    installer = UltimateSystemInstaller(args.path)
    
    if args.skip_tests:
        installer.run_tests = lambda: True
    
    success = await installer.install()
    
    if success:
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Update .env file with your configuration")
        print("2. Run: python launch_ultimate_system.py --mode dev")
        print("3. Visit http://localhost:8000/docs for API documentation")
        sys.exit(0)
    else:
        print("\n❌ Installation failed!")
        print("Check the logs above for details")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


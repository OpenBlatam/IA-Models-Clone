"""
Setup script for the improved Facebook Posts API system
Automated installation and configuration
"""

import os
import sys
import subprocess
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class FacebookPostsSystemSetup:
    """Setup manager for the Facebook Posts API system"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.env_file = self.project_root / ".env"
        self.requirements_file = self.project_root / "requirements_improved.txt"
        
    def print_header(self, title: str):
        """Print formatted header"""
        print("\n" + "=" * 60)
        print(f"🚀 {title}")
        print("=" * 60)
    
    def print_step(self, step: str, description: str = ""):
        """Print formatted step"""
        print(f"\n📋 {step}")
        if description:
            print(f"   {description}")
    
    def print_success(self, message: str):
        """Print success message"""
        print(f"✅ {message}")
    
    def print_error(self, message: str):
        """Print error message"""
        print(f"❌ {message}")
    
    def print_warning(self, message: str):
        """Print warning message"""
        print(f"⚠️  {message}")
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        self.print_step("Checking Python Version")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.print_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        
        self.print_success(f"Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    
    def create_virtual_environment(self) -> bool:
        """Create virtual environment"""
        self.print_step("Creating Virtual Environment")
        
        try:
            if self.venv_path.exists():
                self.print_warning("Virtual environment already exists")
                return True
            
            subprocess.run([
                sys.executable, "-m", "venv", str(self.venv_path)
            ], check=True, capture_output=True)
            
            self.print_success("Virtual environment created successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to create virtual environment: {e}")
            return False
    
    def get_pip_command(self) -> str:
        """Get pip command for the virtual environment"""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "pip.exe")
        else:  # Unix/Linux/macOS
            return str(self.venv_path / "bin" / "pip")
    
    def get_python_command(self) -> str:
        """Get python command for the virtual environment"""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "python.exe")
        else:  # Unix/Linux/macOS
            return str(self.venv_path / "bin" / "python")
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        self.print_step("Installing Dependencies")
        
        try:
            pip_cmd = self.get_pip_command()
            
            # Upgrade pip first
            subprocess.run([
                pip_cmd, "install", "--upgrade", "pip"
            ], check=True, capture_output=True)
            
            # Install requirements
            if self.requirements_file.exists():
                subprocess.run([
                    pip_cmd, "install", "-r", str(self.requirements_file)
                ], check=True, capture_output=True)
            else:
                # Install basic requirements if file doesn't exist
                basic_requirements = [
                    "fastapi==0.104.1",
                    "uvicorn[standard]==0.24.0",
                    "pydantic==2.5.0",
                    "pydantic-settings==2.1.0",
                    "httpx==0.25.2",
                    "structlog==23.2.0",
                    "python-dotenv==1.0.0"
                ]
                
                for req in basic_requirements:
                    subprocess.run([
                        pip_cmd, "install", req
                    ], check=True, capture_output=True)
            
            self.print_success("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to install dependencies: {e}")
            return False
    
    def create_env_file(self) -> bool:
        """Create environment configuration file"""
        self.print_step("Creating Environment Configuration")
        
        try:
            if self.env_file.exists():
                self.print_warning("Environment file already exists")
                return True
            
            env_content = """# Facebook Posts API - Environment Configuration
# Copy this file and customize for your environment

# API Configuration
API_TITLE=Ultimate Facebook Posts API
API_VERSION=4.0.0
API_DESCRIPTION=AI-powered Facebook post generation system
DEBUG=true

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1
MAX_REQUEST_SIZE=10485760

# Database Configuration
DATABASE_URL=sqlite:///./facebook_posts.db
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis Configuration
REDIS_URL=redis://localhost:6379
CACHE_DEFAULT_TTL=3600
CACHE_MAX_CONNECTIONS=10

# AI Configuration
AI_API_KEY=your_openai_api_key_here
AI_MODEL=gpt-3.5-turbo
AI_MAX_TOKENS=2000
AI_TEMPERATURE=0.7
AI_TIMEOUT=30

# Analytics Configuration
ANALYTICS_API_KEY=your_analytics_api_key_here
ANALYTICS_ENDPOINT=https://api.analytics.example.com
ANALYTICS_TIMEOUT=10

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_FILE=logs/facebook_posts.log

# Performance Configuration
MAX_CONCURRENT_REQUESTS=1000
REQUEST_TIMEOUT=30
BACKGROUND_TASK_TIMEOUT=300

# Security Configuration
CORS_ORIGINS=*
CORS_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_HEADERS=*
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# Engine Configuration
ENABLE_CACHING=true
ENABLE_METRICS=true
ENABLE_HEALTH_CHECKS=true
"""
            
            self.env_file.write_text(env_content)
            self.print_success("Environment file created successfully")
            self.print_warning("Please edit .env file with your actual configuration values")
            return True
            
        except Exception as e:
            self.print_error(f"Failed to create environment file: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories"""
        self.print_step("Creating Directory Structure")
        
        directories = [
            "logs",
            "data",
            "cache",
            "uploads",
            "tests",
            "docs"
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(exist_ok=True)
                
                # Create .gitkeep files for empty directories
                gitkeep_file = dir_path / ".gitkeep"
                if not gitkeep_file.exists():
                    gitkeep_file.write_text("")
            
            self.print_success("Directory structure created successfully")
            return True
            
        except Exception as e:
            self.print_error(f"Failed to create directories: {e}")
            return False
    
    def create_gitignore(self) -> bool:
        """Create .gitignore file"""
        self.print_step("Creating .gitignore file")
        
        try:
            gitignore_path = self.project_root / ".gitignore"
            
            if gitignore_path.exists():
                self.print_warning(".gitignore already exists")
                return True
            
            gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
env.bak/
venv.bak/

# Environment Variables
.env
.env.local
.env.production
.env.staging

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
logs/
*.log

# Database
*.db
*.sqlite
*.sqlite3

# Cache
cache/
.cache/
.pytest_cache/

# Coverage
htmlcov/
.coverage
.coverage.*
coverage.xml

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# Temporary files
*.tmp
*.temp
temp/
tmp/

# Uploads
uploads/
media/

# API Keys and Secrets
secrets/
keys/
*.pem
*.key
"""
            
            gitignore_path.write_text(gitignore_content)
            self.print_success(".gitignore file created successfully")
            return True
            
        except Exception as e:
            self.print_error(f"Failed to create .gitignore: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run system tests"""
        self.print_step("Running System Tests")
        
        try:
            python_cmd = self.get_python_command()
            test_file = self.project_root / "tests" / "test_improved_api.py"
            
            if not test_file.exists():
                self.print_warning("Test file not found, skipping tests")
                return True
            
            result = subprocess.run([
                python_cmd, "-m", "pytest", str(test_file), "-v"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.print_success("All tests passed successfully")
                return True
            else:
                self.print_error("Some tests failed")
                print("Test output:")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except Exception as e:
            self.print_error(f"Failed to run tests: {e}")
            return False
    
    def create_startup_script(self) -> bool:
        """Create startup script"""
        self.print_step("Creating Startup Script")
        
        try:
            if os.name == 'nt':  # Windows
                script_path = self.project_root / "start_server.bat"
                script_content = """@echo off
echo Starting Facebook Posts API Server...
call venv\\Scripts\\activate
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
pause
"""
            else:  # Unix/Linux/macOS
                script_path = self.project_root / "start_server.sh"
                script_content = """#!/bin/bash
echo "Starting Facebook Posts API Server..."
source venv/bin/activate
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
                # Make script executable
                os.chmod(script_path, 0o755)
            
            script_path.write_text(script_content)
            self.print_success("Startup script created successfully")
            return True
            
        except Exception as e:
            self.print_error(f"Failed to create startup script: {e}")
            return False
    
    def print_usage_instructions(self):
        """Print usage instructions"""
        self.print_header("Setup Complete - Usage Instructions")
        
        print("""
🎉 Facebook Posts API system has been set up successfully!

📋 Next Steps:

1. Configure Environment:
   - Edit the .env file with your actual configuration values
   - Set your API keys and database URLs
   - Adjust settings for your environment

2. Start the Server:
   - Windows: Double-click start_server.bat
   - Linux/macOS: Run ./start_server.sh
   - Or manually: source venv/bin/activate && uvicorn app:app --reload

3. Test the API:
   - Open http://localhost:8000/docs for interactive API documentation
   - Run the demo: python demo_improved_api.py
   - Check health: curl http://localhost:8000/health

4. Development:
   - Activate virtual environment: source venv/bin/activate (Linux/macOS) or venv\\Scripts\\activate (Windows)
   - Install additional packages: pip install package_name
   - Run tests: pytest tests/ -v

📚 Documentation:
   - API Docs: http://localhost:8000/docs
   - README: README_IMPROVEMENTS.md
   - Tests: tests/test_improved_api.py

🔧 Configuration:
   - Environment: .env file
   - Logs: logs/ directory
   - Cache: cache/ directory

🚀 Features:
   - FastAPI with async support
   - Comprehensive error handling
   - Rate limiting and security
   - Background tasks
   - Health checks and metrics
   - Comprehensive testing
   - Production-ready configuration

Happy coding! 🎉
""")
    
    def setup_system(self) -> bool:
        """Run complete system setup"""
        self.print_header("Facebook Posts API - System Setup")
        
        steps = [
            ("Python Version Check", self.check_python_version),
            ("Virtual Environment", self.create_virtual_environment),
            ("Dependencies Installation", self.install_dependencies),
            ("Directory Structure", self.create_directories),
            ("Environment Configuration", self.create_env_file),
            ("Git Configuration", self.create_gitignore),
            ("Startup Script", self.create_startup_script),
            ("System Tests", self.run_tests)
        ]
        
        success_count = 0
        total_steps = len(steps)
        
        for step_name, step_function in steps:
            try:
                if step_function():
                    success_count += 1
                else:
                    self.print_error(f"Step '{step_name}' failed")
            except Exception as e:
                self.print_error(f"Step '{step_name}' failed with exception: {e}")
        
        # Print summary
        self.print_header("Setup Summary")
        print(f"✅ Successful steps: {success_count}/{total_steps}")
        
        if success_count == total_steps:
            self.print_success("System setup completed successfully!")
            self.print_usage_instructions()
            return True
        else:
            self.print_error("System setup completed with errors")
            print(f"Please review the failed steps and run the setup again")
            return False


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Facebook Posts API system")
    parser.add_argument(
        "--project-root",
        type=str,
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests during setup"
    )
    
    args = parser.parse_args()
    
    setup = FacebookPostsSystemSetup(args.project_root)
    
    if args.skip_tests:
        # Override the test function to always return True
        setup.run_tests = lambda: True
    
    success = setup.setup_system()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Setup completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()































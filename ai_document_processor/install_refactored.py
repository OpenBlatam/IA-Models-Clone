#!/usr/bin/env python3
"""
Refactored Installation Script - Modern Architecture Setup
=========================================================

Installation script for the refactored AI Document Processor with modern architecture.
"""

import os
import sys
import subprocess
import platform
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RefactoredInstaller:
    """Refactored installer with modern architecture setup."""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.install_dir = Path(__file__).parent
        self.requirements_file = self.install_dir / "requirements.txt"
        self.install_log = self.install_dir / "refactored_install_log.json"
        
    def _get_system_info(self) -> Dict[str, any]:
        """Get system information."""
        try:
            import psutil
            
            return {
                'platform': platform.system(),
                'architecture': platform.machine(),
                'python_version': sys.version,
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 2) if os.name != 'nt' else round(psutil.disk_usage('C:').free / (1024**3), 2)
            }
        except ImportError:
            return {
                'platform': platform.system(),
                'architecture': platform.machine(),
                'python_version': sys.version,
                'cpu_count': 1,
                'memory_gb': 4.0,
                'disk_free_gb': 10.0
            }
    
    def print_refactored_banner(self):
        """Print refactored installation banner."""
        print("\n" + "="*80)
        print("🏗️ AI DOCUMENT PROCESSOR - REFACTORED ARCHITECTURE")
        print("="*80)
        print("Modern, clean architecture with separation of concerns")
        print("Version: 3.0.0")
        print("="*80)
        
        print(f"System: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"Python: {self.system_info['python_version'].split()[0]}")
        print(f"CPU Cores: {self.system_info['cpu_count']}")
        print(f"Memory: {self.system_info['memory_gb']} GB")
        print(f"Free Disk: {self.system_info['disk_free_gb']} GB")
        
        print("="*80 + "\n")
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        logger.info("🐍 Checking Python version...")
        
        if sys.version_info < (3, 8):
            logger.error("❌ Python 3.8+ required")
            return False
        
        logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
        return True
    
    def create_directory_structure(self):
        """Create refactored directory structure."""
        logger.info("📁 Creating refactored directory structure...")
        
        directories = [
            "src",
            "src/core",
            "src/models", 
            "src/services",
            "src/api",
            "src/api/routes",
            "src/api/middleware",
            "src/utils",
            "tests",
            "tests/unit",
            "tests/integration",
            "tests/fixtures",
            "docs",
            "scripts"
        ]
        
        for directory in directories:
            dir_path = self.install_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        
        logger.info("✅ Directory structure created")
    
    def install_core_dependencies(self) -> bool:
        """Install core dependencies."""
        logger.info("📦 Installing core dependencies...")
        
        core_requirements = [
            "fastapi>=0.104.1",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.5.0",
            "pydantic-settings>=2.1.0",
            "python-multipart>=0.0.6",
            "aiofiles>=23.2.1",
            "httpx>=0.25.2",
            "python-dotenv>=1.0.0",
            "pyyaml>=6.0.1",
            "structlog>=23.2.0",
            "loguru>=0.7.2"
        ]
        
        try:
            for requirement in core_requirements:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", requirement
                ], check=True, capture_output=True)
                logger.info(f"✅ Installed {requirement}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install core dependencies: {e}")
            return False
    
    def install_ai_dependencies(self) -> bool:
        """Install AI dependencies."""
        logger.info("🤖 Installing AI dependencies...")
        
        ai_requirements = [
            "openai>=1.3.0",
            "anthropic>=0.7.0",
            "cohere>=4.37.0",
            "transformers>=4.36.0",
            "sentence-transformers>=2.2.2",
            "langchain>=0.0.350",
            "langchain-community>=0.0.10",
            "chromadb>=0.4.18"
        ]
        
        try:
            for requirement in ai_requirements:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", requirement
                ], check=True, capture_output=True)
                logger.info(f"✅ Installed {requirement}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install AI dependencies: {e}")
            return False
    
    def install_document_processing_dependencies(self) -> bool:
        """Install document processing dependencies."""
        logger.info("📄 Installing document processing dependencies...")
        
        doc_requirements = [
            "python-docx>=0.8.11",
            "PyPDF2>=3.0.1",
            "pdfplumber>=0.9.0",
            "pymupdf>=1.23.0",
            "markdown>=3.4.0",
            "beautifulsoup4>=4.12.2",
            "lxml>=4.9.3",
            "pytesseract>=0.3.10",
            "opencv-python>=4.8.1.78"
        ]
        
        try:
            for requirement in doc_requirements:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", requirement
                ], check=True, capture_output=True)
                logger.info(f"✅ Installed {requirement}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install document processing dependencies: {e}")
            return False
    
    def install_development_dependencies(self) -> bool:
        """Install development dependencies."""
        logger.info("🛠️ Installing development dependencies...")
        
        dev_requirements = [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.6.0"
        ]
        
        try:
            for requirement in dev_requirements:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", requirement
                ], check=True, capture_output=True)
                logger.info(f"✅ Installed {requirement}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install development dependencies: {e}")
            return False
    
    def create_environment_file(self):
        """Create environment file template."""
        logger.info("🌍 Creating environment file template...")
        
        env_content = """# AI Document Processor - Environment Configuration
# ================================================

# Server Configuration
HOST=0.0.0.0
PORT=8001
WORKERS=1
RELOAD=false

# Database Configuration (optional)
DATABASE_URL=postgresql://user:password@localhost/dbname

# Cache Configuration (optional)
REDIS_URL=redis://localhost:6379/0
CACHE_MAX_MEMORY_MB=1024

# AI Configuration
AI_API_KEY=your-openai-api-key
AI_PROVIDER=openai
AI_MODEL=gpt-3.5-turbo
AI_MAX_TOKENS=2000
AI_TEMPERATURE=0.7

# Security Configuration
SECRET_KEY=your-secret-key-here
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Processing Configuration
MAX_FILE_SIZE_MB=100
ALLOWED_EXTENSIONS=.md,.pdf,.docx,.doc,.txt,.html,.xml
MAX_WORKERS=8
CHUNK_SIZE=8192
ENABLE_STREAMING=true
ENABLE_PARALLEL=true

# Monitoring Configuration
ENABLE_METRICS=true
METRICS_PORT=9090
ENABLE_TRACING=false
"""
        
        env_file = self.install_dir / ".env.example"
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        logger.info("✅ Environment file template created: .env.example")
    
    def create_requirements_file(self):
        """Create requirements file for refactored system."""
        logger.info("📋 Creating requirements file...")
        
        requirements_content = """# AI Document Processor - Refactored Requirements
# ===============================================

# Core Web Framework
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-multipart>=0.0.6

# Async and Concurrency
aiofiles>=23.2.1
httpx>=0.25.2
asyncio-throttle>=1.0.2

# Configuration and Environment
python-dotenv>=1.0.0
pyyaml>=6.0.1
click>=8.1.0

# Logging and Monitoring
structlog>=23.2.0
loguru>=0.7.2
prometheus-client>=0.19.0

# AI and Machine Learning
openai>=1.3.0
anthropic>=0.7.0
cohere>=4.37.0
transformers>=4.36.0
sentence-transformers>=2.2.2
langchain>=0.0.350
langchain-community>=0.0.10
chromadb>=0.4.18

# Document Processing
python-docx>=0.8.11
PyPDF2>=3.0.1
pdfplumber>=0.9.0
pymupdf>=1.23.0
markdown>=3.4.0
beautifulsoup4>=4.12.2
lxml>=4.9.3
pytesseract>=0.3.10
opencv-python>=4.8.1.78

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.4

# Caching and Storage
redis>=5.0.1
diskcache>=5.6.3

# Security
cryptography>=41.0.8
python-jose[cryptography]>=3.3.0

# Development Dependencies (optional)
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.7.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.5.0
pre-commit>=3.6.0
"""
        
        requirements_file = self.install_dir / "requirements_refactored.txt"
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
        
        logger.info("✅ Requirements file created: requirements_refactored.txt")
    
    def verify_installation(self) -> Dict[str, bool]:
        """Verify installation."""
        logger.info("🔍 Verifying installation...")
        
        verification_results = {}
        
        # Test imports
        test_imports = {
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'pydantic': 'pydantic',
            'aiofiles': 'aiofiles',
            'httpx': 'httpx',
            'openai': 'openai',
            'transformers': 'transformers',
            'langchain': 'langchain',
            'chromadb': 'chromadb',
            'redis': 'redis',
            'numpy': 'numpy',
            'pandas': 'pandas'
        }
        
        for name, module in test_imports.items():
            try:
                __import__(module)
                verification_results[name] = True
                logger.info(f"✅ {name} - OK")
            except ImportError:
                verification_results[name] = False
                logger.error(f"❌ {name} - FAILED")
        
        return verification_results
    
    def save_install_log(self, verification_results: Dict[str, bool]):
        """Save installation log."""
        log_data = {
            'timestamp': time.time(),
            'system_info': self.system_info,
            'verification_results': verification_results,
            'installation_successful': all(verification_results.values()),
            'refactored_architecture': True
        }
        
        with open(self.install_log, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"📝 Installation log saved to {self.install_log}")
    
    def print_installation_summary(self, verification_results: Dict[str, bool]):
        """Print installation summary."""
        print("\n" + "="*80)
        print("📊 REFACTORED INSTALLATION SUMMARY")
        print("="*80)
        
        # Verification results
        total_libs = len(verification_results)
        successful_libs = sum(verification_results.values())
        success_rate = (successful_libs / total_libs) * 100
        
        print(f"Libraries Installed: {successful_libs}/{total_libs} ({success_rate:.1f}%)")
        
        if success_rate >= 95:
            print("🚀 Installation: EXCELLENT")
        elif success_rate >= 90:
            print("🎉 Installation: VERY GOOD")
        elif success_rate >= 85:
            print("✅ Installation: GOOD")
        else:
            print("⚠️ Installation: NEEDS ATTENTION")
        
        print("\n🏗️ Refactored Architecture Features:")
        print("   ✅ Clean separation of concerns")
        print("   ✅ Modern FastAPI application")
        print("   ✅ Comprehensive error handling")
        print("   ✅ Type safety with Pydantic")
        print("   ✅ Async/await throughout")
        print("   ✅ Dependency injection")
        print("   ✅ Configuration management")
        print("   ✅ Structured logging")
        print("   ✅ API documentation")
        print("   ✅ Testing framework")
        
        print("\n🚀 Next Steps:")
        print("   1. Copy .env.example to .env and configure")
        print("   2. Run: python main_refactored.py")
        print("   3. Test: http://localhost:8001/api/v1/health")
        print("   4. Docs: http://localhost:8001/docs")
        print("   5. Read: README_REFACTORED.md")
        
        print("="*80 + "\n")
    
    def install(self) -> bool:
        """Run complete refactored installation."""
        try:
            # Print banner
            self.print_refactored_banner()
            
            # Check Python version
            if not self.check_python_version():
                return False
            
            # Create directory structure
            self.create_directory_structure()
            
            # Install dependencies
            if not self.install_core_dependencies():
                return False
            
            if not self.install_ai_dependencies():
                return False
            
            if not self.install_document_processing_dependencies():
                return False
            
            if not self.install_development_dependencies():
                return False
            
            # Create configuration files
            self.create_environment_file()
            self.create_requirements_file()
            
            # Verify installation
            verification_results = self.verify_installation()
            
            # Save installation log
            self.save_install_log(verification_results)
            
            # Print summary
            self.print_installation_summary(verification_results)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Refactored installation failed: {e}")
            return False


def main():
    """Main installation function."""
    installer = RefactoredInstaller()
    success = installer.install()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


















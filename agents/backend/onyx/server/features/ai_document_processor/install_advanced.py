#!/usr/bin/env python3
"""
Advanced Installation Script - Next Generation AI Document Processor
================================================================

Intelligent installation script for the next generation of AI document processing.
"""

import os
import sys
import subprocess
import platform
import shutil
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import aiofiles
import psutil
import GPUtil
import cpuinfo
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import click
import typer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_install.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

console = Console()

# System information
SYSTEM_INFO = {
    'platform': platform.system(),
    'architecture': platform.architecture()[0],
    'python_version': sys.version,
    'cpu_count': psutil.cpu_count(),
    'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
    'disk_space_gb': round(shutil.disk_usage('/').free / (1024**3), 2) if platform.system() != 'Windows' else round(shutil.disk_usage('C:').free / (1024**3), 2)
}

# GPU information
try:
    gpus = GPUtil.getGPUs()
    SYSTEM_INFO['gpu_count'] = len(gpus)
    SYSTEM_INFO['gpu_info'] = [{'name': gpu.name, 'memory': gpu.memoryTotal} for gpu in gpus]
except:
    SYSTEM_INFO['gpu_count'] = 0
    SYSTEM_INFO['gpu_info'] = []

# CPU information
try:
    cpu_info = cpuinfo.get_cpu_info()
    SYSTEM_INFO['cpu_name'] = cpu_info.get('brand_raw', 'Unknown')
    SYSTEM_INFO['cpu_cores'] = cpu_info.get('count', psutil.cpu_count())
except:
    SYSTEM_INFO['cpu_name'] = 'Unknown'
    SYSTEM_INFO['cpu_cores'] = psutil.cpu_count()


class AdvancedInstaller:
    """Advanced installer for the next generation AI document processor."""
    
    def __init__(self):
        self.console = Console()
        self.requirements_file = "requirements_advanced.txt"
        self.install_log = []
        self.installed_packages = []
        self.failed_packages = []
        self.warnings = []
        self.start_time = time.time()
        
    def display_system_info(self):
        """Display system information."""
        table = Table(title="System Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Platform", SYSTEM_INFO['platform'])
        table.add_row("Architecture", SYSTEM_INFO['architecture'])
        table.add_row("Python Version", SYSTEM_INFO['python_version'].split()[0])
        table.add_row("CPU", SYSTEM_INFO['cpu_name'])
        table.add_row("CPU Cores", str(SYSTEM_INFO['cpu_cores']))
        table.add_row("Memory", f"{SYSTEM_INFO['memory_gb']} GB")
        table.add_row("Disk Space", f"{SYSTEM_INFO['disk_space_gb']} GB")
        table.add_row("GPU Count", str(SYSTEM_INFO['gpu_count']))
        
        if SYSTEM_INFO['gpu_info']:
            for i, gpu in enumerate(SYSTEM_INFO['gpu_info']):
                table.add_row(f"GPU {i+1}", f"{gpu['name']} ({gpu['memory']} MB)")
        
        self.console.print(table)
        
        # Recommendations
        recommendations = []
        if SYSTEM_INFO['memory_gb'] < 8:
            recommendations.append("⚠️  Consider upgrading to at least 8GB RAM for optimal performance")
        if SYSTEM_INFO['disk_space_gb'] < 10:
            recommendations.append("⚠️  Low disk space. Consider freeing up space")
        if SYSTEM_INFO['gpu_count'] == 0:
            recommendations.append("💡 No GPU detected. CPU-only mode will be used")
        else:
            recommendations.append("🚀 GPU detected! GPU acceleration will be enabled")
        
        if recommendations:
            self.console.print("\n[bold yellow]Recommendations:[/bold yellow]")
            for rec in recommendations:
                self.console.print(f"  {rec}")
    
    def check_prerequisites(self) -> bool:
        """Check system prerequisites."""
        self.console.print("\n[bold blue]Checking Prerequisites...[/bold blue]")
        
        issues = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            issues.append(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        
        # Check pip
        try:
            import pip
            pip_version = pip.__version__
            self.console.print(f"✅ pip {pip_version} found")
        except ImportError:
            issues.append("pip not found")
        
        # Check virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.console.print("✅ Virtual environment detected")
        else:
            self.console.print("⚠️  No virtual environment detected. Consider using one.")
        
        # Check disk space
        if SYSTEM_INFO['disk_space_gb'] < 5:
            issues.append(f"Insufficient disk space: {SYSTEM_INFO['disk_space_gb']} GB (need at least 5 GB)")
        
        # Check memory
        if SYSTEM_INFO['memory_gb'] < 4:
            issues.append(f"Insufficient memory: {SYSTEM_INFO['memory_gb']} GB (need at least 4 GB)")
        
        if issues:
            self.console.print("\n[bold red]Prerequisites Issues:[/bold red]")
            for issue in issues:
                self.console.print(f"  ❌ {issue}")
            return False
        
        self.console.print("\n[bold green]✅ All prerequisites met![/bold green]")
        return True
    
    def upgrade_pip(self):
        """Upgrade pip to latest version."""
        self.console.print("\n[bold blue]Upgrading pip...[/bold blue]")
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], check=True, capture_output=True, text=True)
            self.console.print("✅ pip upgraded successfully")
        except subprocess.CalledProcessError as e:
            self.console.print(f"⚠️  pip upgrade failed: {e}")
            self.warnings.append(f"pip upgrade failed: {e}")
    
    def install_build_tools(self):
        """Install build tools for compilation."""
        self.console.print("\n[bold blue]Installing Build Tools...[/bold blue]")
        
        build_tools = [
            "setuptools",
            "wheel",
            "cython",
            "numpy",
            "scipy"
        ]
        
        for tool in build_tools:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "--upgrade", tool
                ], check=True, capture_output=True, text=True)
                self.console.print(f"✅ {tool} installed")
                self.installed_packages.append(tool)
            except subprocess.CalledProcessError as e:
                self.console.print(f"⚠️  {tool} installation failed: {e}")
                self.failed_packages.append(tool)
    
    def install_core_packages(self):
        """Install core packages first."""
        self.console.print("\n[bold blue]Installing Core Packages...[/bold blue]")
        
        core_packages = [
            "fastapi>=0.104.1",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.5.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "requests>=2.31.0",
            "aiofiles>=23.2.1",
            "python-multipart>=0.0.6",
            "python-dotenv>=1.0.0",
            "rich>=13.5.0",
            "click>=8.1.0",
            "typer>=0.9.0"
        ]
        
        for package in core_packages:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], check=True, capture_output=True, text=True)
                self.console.print(f"✅ {package.split('>=')[0]} installed")
                self.installed_packages.append(package.split('>=')[0])
            except subprocess.CalledProcessError as e:
                self.console.print(f"❌ {package.split('>=')[0]} installation failed: {e}")
                self.failed_packages.append(package.split('>=')[0])
    
    def install_ai_packages(self):
        """Install AI and ML packages."""
        self.console.print("\n[bold blue]Installing AI & ML Packages...[/bold blue]")
        
        ai_packages = [
            "openai>=1.3.0",
            "anthropic>=0.7.0",
            "transformers>=4.36.0",
            "sentence-transformers>=2.2.2",
            "torch>=2.1.1",
            "torchvision>=0.16.1",
            "scikit-learn>=1.3.0",
            "chromadb>=0.4.18",
            "langchain>=0.0.350",
            "huggingface-hub>=0.19.0"
        ]
        
        for package in ai_packages:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], check=True, capture_output=True, text=True)
                self.console.print(f"✅ {package.split('>=')[0]} installed")
                self.installed_packages.append(package.split('>=')[0])
            except subprocess.CalledProcessError as e:
                self.console.print(f"❌ {package.split('>=')[0]} installation failed: {e}")
                self.failed_packages.append(package.split('>=')[0])
    
    def install_document_packages(self):
        """Install document processing packages."""
        self.console.print("\n[bold blue]Installing Document Processing Packages...[/bold blue]")
        
        doc_packages = [
            "PyPDF2>=3.0.1",
            "pdfplumber>=0.9.0",
            "pymupdf>=1.23.0",
            "python-docx>=0.8.11",
            "python-pptx>=0.6.21",
            "openpyxl>=3.1.2",
            "markdown>=3.4.0",
            "beautifulsoup4>=4.12.2",
            "lxml>=4.9.3",
            "pytesseract>=0.3.10",
            "easyocr>=1.7.0",
            "opencv-python>=4.8.1.78",
            "Pillow>=10.1.0"
        ]
        
        for package in doc_packages:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], check=True, capture_output=True, text=True)
                self.console.print(f"✅ {package.split('>=')[0]} installed")
                self.installed_packages.append(package.split('>=')[0])
            except subprocess.CalledProcessError as e:
                self.console.print(f"❌ {package.split('>=')[0]} installation failed: {e}")
                self.failed_packages.append(package.split('>=')[0])
    
    def install_performance_packages(self):
        """Install performance and caching packages."""
        self.console.print("\n[bold blue]Installing Performance & Caching Packages...[/bold blue]")
        
        perf_packages = [
            "redis>=5.0.1",
            "orjson>=3.9.10",
            "msgpack>=1.0.7",
            "lz4>=4.3.2",
            "uvloop>=0.19.0",
            "aioredis>=2.0.1",
            "diskcache>=5.6.3"
        ]
        
        for package in perf_packages:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], check=True, capture_output=True, text=True)
                self.console.print(f"✅ {package.split('>=')[0]} installed")
                self.installed_packages.append(package.split('>=')[0])
            except subprocess.CalledProcessError as e:
                self.console.print(f"❌ {package.split('>=')[0]} installation failed: {e}")
                self.failed_packages.append(package.split('>=')[0])
    
    def install_monitoring_packages(self):
        """Install monitoring and observability packages."""
        self.console.print("\n[bold blue]Installing Monitoring & Observability Packages...[/bold blue]")
        
        monitoring_packages = [
            "prometheus-client>=0.19.0",
            "structlog>=23.2.0",
            "sentry-sdk>=1.38.0",
            "psutil>=5.9.6",
            "elasticsearch>=8.11.0",
            "loguru>=0.7.2"
        ]
        
        for package in monitoring_packages:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], check=True, capture_output=True, text=True)
                self.console.print(f"✅ {package.split('>=')[0]} installed")
                self.installed_packages.append(package.split('>=')[0])
            except subprocess.CalledProcessError as e:
                self.console.print(f"❌ {package.split('>=')[0]} installation failed: {e}")
                self.failed_packages.append(package.split('>=')[0])
    
    def install_security_packages(self):
        """Install security packages."""
        self.console.print("\n[bold blue]Installing Security Packages...[/bold blue]")
        
        security_packages = [
            "cryptography>=41.0.8",
            "pyjwt>=2.8.0",
            "passlib[bcrypt]>=1.7.4",
            "python-jose[cryptography]>=3.3.0"
        ]
        
        for package in security_packages:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], check=True, capture_output=True, text=True)
                self.console.print(f"✅ {package.split('>=')[0]} installed")
                self.installed_packages.append(package.split('>=')[0])
            except subprocess.CalledProcessError as e:
                self.console.print(f"❌ {package.split('>=')[0]} installation failed: {e}")
                self.failed_packages.append(package.split('>=')[0])
    
    def install_optional_packages(self):
        """Install optional advanced packages."""
        self.console.print("\n[bold blue]Installing Optional Advanced Packages...[/bold blue]")
        
        optional_packages = [
            "whisper>=1.1.10",
            "speechrecognition>=3.10.0",
            "librosa>=0.10.1",
            "plotly>=5.17.0",
            "dash>=2.14.2",
            "streamlit>=1.28.2",
            "gradio>=4.7.1",
            "celery>=5.3.4",
            "dask[complete]>=2023.12.0",
            "polars>=0.20.0"
        ]
        
        for package in optional_packages:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], check=True, capture_output=True, text=True)
                self.console.print(f"✅ {package.split('>=')[0]} installed")
                self.installed_packages.append(package.split('>=')[0])
            except subprocess.CalledProcessError as e:
                self.console.print(f"⚠️  {package.split('>=')[0]} installation failed (optional): {e}")
                self.warnings.append(f"Optional package {package.split('>=')[0]} failed: {e}")
    
    def create_config_file(self):
        """Create configuration file."""
        self.console.print("\n[bold blue]Creating Configuration File...[/bold blue]")
        
        config_content = """# Advanced AI Document Processor Configuration
# ================================================

# Core settings
APP_NAME = "Advanced AI Document Processor"
VERSION = "3.0.0"
DEBUG = False
ENVIRONMENT = "production"

# Performance settings
MAX_WORKERS = 16
MAX_MEMORY_GB = 32
CACHE_SIZE_MB = 4096
COMPRESSION_LEVEL = 6
MAX_FILE_SIZE_MB = 100

# AI settings
OPENAI_API_KEY = ""  # Add your OpenAI API key
ANTHROPIC_API_KEY = ""  # Add your Anthropic API key
COHERE_API_KEY = ""  # Add your Cohere API key
HUGGINGFACE_TOKEN = ""  # Add your Hugging Face token

# Model configurations
DEFAULT_LLM_MODEL = "gpt-4-turbo"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_VISION_MODEL = "gpt-4-vision-preview"
DEFAULT_AUDIO_MODEL = "whisper-1"

# Advanced AI features
ENABLE_MULTIMODAL_AI = True
ENABLE_VISION_PROCESSING = True
ENABLE_AUDIO_PROCESSING = True
ENABLE_CODE_ANALYSIS = True
ENABLE_SENTIMENT_ANALYSIS = True
ENABLE_ENTITY_EXTRACTION = True
ENABLE_TOPIC_MODELING = True
ENABLE_CLUSTERING = True
ENABLE_ANOMALY_DETECTION = True

# Document processing
SUPPORTED_FORMATS = [
    'pdf', 'docx', 'pptx', 'txt', 'md', 'html', 'xml', 'json', 'csv', 'xlsx',
    'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'svg',
    'mp3', 'wav', 'flac', 'ogg', 'm4a',
    'zip', 'tar', 'gz', 'rar'
]

# Security
ENABLE_ADVANCED_SECURITY = True
JWT_SECRET = "your-advanced-secret-key-change-this"
JWT_ALGORITHM = "HS512"
JWT_EXPIRE_HOURS = 24
ENABLE_AUDIT_LOGGING = True
ENABLE_RATE_LIMITING = True

# Monitoring
ENABLE_ADVANCED_MONITORING = True
METRICS_PORT = 9090
ENABLE_ELASTICSEARCH = True
ELASTICSEARCH_URL = "http://localhost:9200"
LOG_LEVEL = "INFO"

# Database
DATABASE_URL = "postgresql://user:pass@localhost/advanced_docs"
REDIS_URL = "redis://localhost:6379"
ENABLE_VECTOR_DATABASE = True
VECTOR_DATABASE_URL = "http://localhost:8000"

# Advanced features
ENABLE_WORKFLOW_AUTOMATION = True
ENABLE_DOCUMENT_COMPARISON = True
ENABLE_VERSION_CONTROL = True
ENABLE_COLLABORATIVE_EDITING = True
ENABLE_REAL_TIME_SYNC = True
ENABLE_ADVANCED_ANALYTICS = True
ENABLE_PREDICTIVE_ANALYTICS = True
ENABLE_ML_PIPELINE = True
"""
        
        try:
            with open("advanced_config.py", "w") as f:
                f.write(config_content)
            self.console.print("✅ Configuration file created: advanced_config.py")
        except Exception as e:
            self.console.print(f"❌ Failed to create config file: {e}")
    
    def create_startup_script(self):
        """Create startup script."""
        self.console.print("\n[bold blue]Creating Startup Script...[/bold blue]")
        
        startup_content = """#!/usr/bin/env python3
\"\"\"
Advanced AI Document Processor Startup Script
==========================================
\"\"\"

import uvicorn
from advanced_features import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        workers=1,
        log_level="info",
        access_log=True,
        reload=False
    )
"""
        
        try:
            with open("start_advanced.py", "w") as f:
                f.write(startup_content)
            self.console.print("✅ Startup script created: start_advanced.py")
        except Exception as e:
            self.console.print(f"❌ Failed to create startup script: {e}")
    
    def run_tests(self):
        """Run basic tests."""
        self.console.print("\n[bold blue]Running Basic Tests...[/bold blue]")
        
        test_imports = [
            "fastapi",
            "uvicorn",
            "pydantic",
            "numpy",
            "pandas",
            "openai",
            "transformers",
            "torch",
            "redis",
            "elasticsearch"
        ]
        
        passed_tests = 0
        total_tests = len(test_imports)
        
        for module in test_imports:
            try:
                __import__(module)
                self.console.print(f"✅ {module} import successful")
                passed_tests += 1
            except ImportError as e:
                self.console.print(f"❌ {module} import failed: {e}")
        
        self.console.print(f"\n[bold blue]Test Results: {passed_tests}/{total_tests} passed[/bold blue]")
        
        if passed_tests == total_tests:
            self.console.print("[bold green]✅ All tests passed![/bold green]")
        else:
            self.console.print(f"[bold yellow]⚠️  {total_tests - passed_tests} tests failed[/bold yellow]")
    
    def generate_report(self):
        """Generate installation report."""
        self.console.print("\n[bold blue]Generating Installation Report...[/bold blue]")
        
        end_time = time.time()
        installation_time = end_time - self.start_time
        
        report = {
            "installation_summary": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
                "duration_seconds": round(installation_time, 2),
                "system_info": SYSTEM_INFO
            },
            "packages": {
                "installed": self.installed_packages,
                "failed": self.failed_packages,
                "warnings": self.warnings
            },
            "statistics": {
                "total_installed": len(self.installed_packages),
                "total_failed": len(self.failed_packages),
                "total_warnings": len(self.warnings),
                "success_rate": round(len(self.installed_packages) / (len(self.installed_packages) + len(self.failed_packages)) * 100, 2) if (len(self.installed_packages) + len(self.failed_packages)) > 0 else 0
            }
        }
        
        try:
            with open("installation_report.json", "w") as f:
                json.dump(report, f, indent=2)
            self.console.print("✅ Installation report saved: installation_report.json")
        except Exception as e:
            self.console.print(f"❌ Failed to save report: {e}")
        
        # Display summary
        table = Table(title="Installation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Duration", f"{installation_time:.2f} seconds")
        table.add_row("Packages Installed", str(len(self.installed_packages)))
        table.add_row("Packages Failed", str(len(self.failed_packages)))
        table.add_row("Warnings", str(len(self.warnings)))
        table.add_row("Success Rate", f"{report['statistics']['success_rate']}%")
        
        self.console.print(table)
        
        if self.failed_packages:
            self.console.print("\n[bold red]Failed Packages:[/bold red]")
            for package in self.failed_packages:
                self.console.print(f"  ❌ {package}")
        
        if self.warnings:
            self.console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in self.warnings:
                self.console.print(f"  ⚠️  {warning}")
    
    def install_all(self):
        """Install all packages and components."""
        self.console.print(Panel.fit(
            "[bold blue]Advanced AI Document Processor - Next Generation Installation[/bold blue]\n"
            "Installing the most advanced AI document processing system...",
            border_style="blue"
        ))
        
        # Display system info
        self.display_system_info()
        
        # Check prerequisites
        if not self.check_prerequisites():
            self.console.print("\n[bold red]Prerequisites check failed. Please fix the issues and try again.[/bold red]")
            return False
        
        # Upgrade pip
        self.upgrade_pip()
        
        # Install build tools
        self.install_build_tools()
        
        # Install core packages
        self.install_core_packages()
        
        # Install AI packages
        self.install_ai_packages()
        
        # Install document packages
        self.install_document_packages()
        
        # Install performance packages
        self.install_performance_packages()
        
        # Install monitoring packages
        self.install_monitoring_packages()
        
        # Install security packages
        self.install_security_packages()
        
        # Install optional packages
        self.install_optional_packages()
        
        # Create configuration file
        self.create_config_file()
        
        # Create startup script
        self.create_startup_script()
        
        # Run tests
        self.run_tests()
        
        # Generate report
        self.generate_report()
        
        self.console.print(Panel.fit(
            "[bold green]✅ Advanced AI Document Processor Installation Complete![/bold green]\n\n"
            "Next steps:\n"
            "1. Configure your API keys in advanced_config.py\n"
            "2. Start the server: python start_advanced.py\n"
            "3. Access the API at: http://localhost:8001\n"
            "4. View documentation at: http://localhost:8001/docs",
            border_style="green"
        ))
        
        return True


def main():
    """Main installation function."""
    installer = AdvancedInstaller()
    
    try:
        success = installer.install_all()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        installer.console.print("\n[bold red]Installation interrupted by user[/bold red]")
        sys.exit(1)
    except Exception as e:
        installer.console.print(f"\n[bold red]Installation failed: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
🚀 Extended Libraries Installation Script for Enhanced Facebook Content Optimization System
========================================================================================

This script installs all the extended libraries needed for the enhanced system:
- Core AI/ML libraries (PyTorch, Transformers, etc.)
- Advanced analytics (Plotly, Bokeh, Altair, etc.)
- Monitoring tools (Prometheus, psutil, etc.)
- Business intelligence (Streamlit, Dash, etc.)
- And many more specialized libraries

Usage:
    python install_extended_libraries.py [--category CATEGORY] [--skip-existing] [--verbose]
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('library_installation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class LibraryInstaller:
    """Comprehensive library installer with progress tracking and error handling"""
    
    def __init__(self, skip_existing: bool = False, verbose: bool = False):
        self.skip_existing = skip_existing
        self.verbose = verbose
        self.installation_log = []
        self.failed_installations = []
        self.successful_installations = []
        
        # Library categories with their packages
        self.library_categories = {
            "core_ai_ml": {
                "name": "Core AI/ML Libraries",
                "packages": [
                    "torch>=2.0.0",
                    "torchvision>=0.15.0",
                    "torchaudio>=2.0.0",
                    "transformers>=4.30.0",
                    "diffusers>=0.18.0",
                    "accelerate>=0.20.0",
                    "peft>=0.4.0",
                    "bitsandbytes>=0.41.0",
                    "sentence-transformers>=2.2.0",
                    "datasets>=2.12.0",
                    "tokenizers>=0.13.0"
                ]
            },
            "computer_vision": {
                "name": "Computer Vision & Image Processing",
                "packages": [
                    "opencv-python>=4.8.0",
                    "Pillow>=10.0.0",
                    "imageio>=2.31.0",
                    "scikit-image>=0.21.0",
                    "albumentations>=1.3.0",
                    "imgaug>=0.4.0",
                    "kornia>=0.6.0"
                ]
            },
            "nlp": {
                "name": "Natural Language Processing",
                "packages": [
                    "spacy>=3.6.0",
                    "nltk>=3.8.0",
                    "textblob>=0.17.0",
                    "gensim>=4.3.0",
                    "wordcloud>=1.9.0",
                    "textstat>=0.7.0",
                    "language-tool-python>=2.7.0"
                ]
            },
            "advanced_ml": {
                "name": "Advanced ML & Statistics",
                "packages": [
                    "scikit-learn>=1.3.0",
                    "scipy>=1.11.0",
                    "statsmodels>=0.14.0",
                    "xgboost>=1.7.0",
                    "lightgbm>=4.0.0",
                    "catboost>=1.2.0",
                    "optuna>=3.2.0",
                    "hyperopt>=0.2.7",
                    "ray[tune]>=2.6.0"
                ]
            },
            "llm_integration": {
                "name": "Large Language Models",
                "packages": [
                    "openai>=1.0.0",
                    "anthropic>=0.7.0",
                    "cohere>=4.0.0",
                    "huggingface-hub>=0.16.0",
                    "langchain>=0.0.300",
                    "langchain-community>=0.0.10",
                    "llama-index>=0.8.0",
                    "chromadb>=0.4.0",
                    "faiss-cpu>=1.7.4"
                ]
            },
            "cv_ai": {
                "name": "Computer Vision AI",
                "packages": [
                    "ultralytics>=8.0.0",
                    "supervision>=0.16.0",
                    "roboflow>=1.2.0",
                    "autodistill>=0.1.0"
                ]
            },
            "audio_processing": {
                "name": "Audio Processing",
                "packages": [
                    "librosa>=0.10.0",
                    "pydub>=0.25.0",
                    "soundfile>=0.12.0",
                    "webrtcvad>=2.0.10",
                    "speechrecognition>=3.10.0",
                    "pyaudio>=0.2.11"
                ]
            },
            "data_processing": {
                "name": "Data Processing & Analytics",
                "packages": [
                    "pandas>=2.0.0",
                    "numpy>=1.24.0",
                    "polars>=0.19.0",
                    "vaex>=4.15.0",
                    "dask>=2023.8.0",
                    "modin>=0.20.0"
                ]
            },
            "visualization": {
                "name": "Data Visualization",
                "packages": [
                    "matplotlib>=3.7.0",
                    "seaborn>=0.12.0",
                    "plotly>=5.15.0",
                    "bokeh>=3.2.0",
                    "altair>=5.1.0",
                    "holoviews>=1.17.0",
                    "datashader>=0.16.0",
                    "pygal>=2.4.0"
                ]
            },
            "business_intelligence": {
                "name": "Business Intelligence",
                "packages": [
                    "streamlit>=1.25.0",
                    "dash>=2.11.0",
                    "gradio>=3.40.0",
                    "panel>=1.2.0",
                    "voila>=0.4.0",
                    "jupyter>=1.0.0",
                    "jupyterlab>=4.0.0"
                ]
            },
            "databases": {
                "name": "Database & Storage",
                "packages": [
                    "sqlalchemy>=2.0.0",
                    "alembic>=1.11.0",
                    "psycopg2-binary>=2.9.0",
                    "pymysql>=1.1.0",
                    "asyncpg>=0.28.0",
                    "pymongo>=4.4.0",
                    "redis>=4.6.0",
                    "elasticsearch>=8.8.0"
                ]
            },
            "web_frameworks": {
                "name": "API & Web Development",
                "packages": [
                    "fastapi>=0.100.0",
                    "uvicorn>=0.23.0",
                    "flask>=2.3.0",
                    "django>=4.2.0",
                    "starlette>=0.27.0",
                    "sanic>=23.6.0",
                    "requests>=2.31.0",
                    "httpx>=0.24.0",
                    "aiohttp>=3.8.0",
                    "websockets>=11.0.0"
                ]
            },
            "monitoring": {
                "name": "Monitoring & Observability",
                "packages": [
                    "prometheus-client>=0.17.0",
                    "grafana-api>=1.0.3",
                    "jaeger-client>=4.8.0",
                    "opentelemetry-api>=1.20.0",
                    "opentelemetry-sdk>=1.20.0",
                    "sentry-sdk>=1.28.0"
                ]
            },
            "logging": {
                "name": "Logging & Debugging",
                "packages": [
                    "structlog>=23.1.0",
                    "loguru>=0.7.0",
                    "python-json-logger>=2.0.0",
                    "colorama>=0.4.6",
                    "rich>=13.4.0"
                ]
            },
            "performance": {
                "name": "Performance Monitoring",
                "packages": [
                    "psutil>=5.9.0",
                    "memory-profiler>=0.61.0",
                    "line-profiler>=4.1.0",
                    "py-spy>=0.3.14",
                    "scalene>=1.5.0"
                ]
            },
            "cloud_devops": {
                "name": "Cloud & DevOps",
                "packages": [
                    "kubernetes>=26.1.0",
                    "docker>=6.1.0",
                    "helm>=0.1.0",
                    "terraform>=0.1.0",
                    "ansible>=8.0.0",
                    "pulumi>=0.1.0"
                ]
            },
            "social_media": {
                "name": "Social Media & Marketing",
                "packages": [
                    "facebook-sdk>=3.1.0",
                    "tweepy>=4.14.0",
                    "instagram-private-api>=1.6.0",
                    "linkedin-api>=2.0.0",
                    "youtube-api-python>=0.8.0",
                    "google-analytics-data>=0.18.0",
                    "facebook-business>=17.0.0",
                    "google-ads>=21.0.0"
                ]
            },
            "seo_content": {
                "name": "SEO & Content",
                "packages": [
                    "beautifulsoup4>=4.12.0",
                    "selenium>=4.10.0",
                    "scrapy>=2.8.0",
                    "newspaper3k>=0.2.8",
                    "readability-lxml>=0.8.1"
                ]
            },
            "advanced_ai": {
                "name": "Advanced AI Capabilities",
                "packages": [
                    "clip-by-openai>=1.0",
                    "open-clip-torch>=2.20.0",
                    "flamingo-pytorch>=0.1.0",
                    "minigpt4>=0.1.0",
                    "whisper>=1.1.0",
                    "whisperx>=3.0.0",
                    "faster-whisper>=0.9.0",
                    "pyannote.audio>=2.1.1"
                ]
            },
            "enterprise": {
                "name": "Enterprise Features",
                "packages": [
                    "prefect>=2.10.0",
                    "airflow>=2.7.0",
                    "luigi>=3.3.0",
                    "celery>=5.3.0",
                    "kafka-python>=2.0.0",
                    "pika>=1.3.0",
                    "aio-pika>=9.0.0"
                ]
            },
            "testing": {
                "name": "Testing & Quality",
                "packages": [
                    "pytest>=7.4.0",
                    "pytest-asyncio>=0.21.0",
                    "pytest-cov>=4.1.0",
                    "pytest-mock>=3.11.0",
                    "unittest2>=1.1.0"
                ]
            },
            "code_quality": {
                "name": "Code Quality",
                "packages": [
                    "black>=23.7.0",
                    "isort>=5.12.0",
                    "flake8>=6.0.0",
                    "mypy>=1.5.0",
                    "bandit>=1.7.0",
                    "safety>=2.3.0"
                ]
            },
            "utilities": {
                "name": "Utilities & Helpers",
                "packages": [
                    "python-dateutil>=2.8.0",
                    "pytz>=2023.3",
                    "arrow>=1.2.0",
                    "pendulum>=2.1.0",
                    "pathlib2>=2.3.7",
                    "watchdog>=3.0.0",
                    "python-magic>=0.4.27",
                    "aiofiles>=23.1.0"
                ]
            },
            "specialized_tools": {
                "name": "Specialized Tools",
                "packages": [
                    "PyPDF2>=3.0.0",
                    "pdfplumber>=0.9.0",
                    "reportlab>=4.0.0",
                    "weasyprint>=60.0",
                    "openpyxl>=3.1.0",
                    "xlsxwriter>=3.1.0",
                    "xlrd>=2.0.0",
                    "python-docx>=0.8.11",
                    "email-validator>=2.0.0",
                    "jinja2>=3.1.0"
                ]
            },
            "ml_tools": {
                "name": "Machine Learning Tools",
                "packages": [
                    "feature-engine>=1.6.0",
                    "category-encoders>=2.6.0",
                    "imbalanced-learn>=0.11.0",
                    "mlxtend>=0.22.0",
                    "shap>=0.42.0",
                    "lime>=0.2.0",
                    "eli5>=0.13.0",
                    "interpret>=0.4.0",
                    "auto-sklearn>=0.15.0",
                    "autogluon>=0.8.0",
                    "pycaret>=3.1.0",
                    "flaml>=1.2.0"
                ]
            },
            "advanced_analytics": {
                "name": "Advanced Analytics",
                "packages": [
                    "prophet>=1.1.4",
                    "arch>=6.2.0",
                    "pyflux>=0.4.0",
                    "geopandas>=0.13.0",
                    "folium>=0.14.0",
                    "geopy>=2.3.0",
                    "shapely>=2.0.0",
                    "yfinance>=0.2.18",
                    "pandas-ta>=0.3.14b0",
                    "ta-lib>=0.4.0",
                    "finrl>=0.3.0"
                ]
            },
            "performance_optimization": {
                "name": "Performance & Optimization",
                "packages": [
                    "diskcache>=5.6.0",
                    "cachetools>=5.3.0",
                    "memcached>=1.59",
                    "joblib>=1.3.0",
                    "cupy>=12.0.0",
                    "numba>=0.57.0",
                    "cython>=3.0.0",
                    "mypyc>=1.5.0"
                ]
            },
            "security": {
                "name": "Security & Compliance",
                "packages": [
                    "cryptography>=41.0.0",
                    "bcrypt>=4.0.0",
                    "passlib>=1.7.4",
                    "python-jose>=3.3.0",
                    "marshmallow>=3.20.0",
                    "cerberus>=1.3.0"
                ]
            },
            "development": {
                "name": "Development Tools",
                "packages": [
                    "ipython>=8.14.0",
                    "notebook>=7.0.0",
                    "sphinx>=7.1.0",
                    "mkdocs>=1.5.0",
                    "pdoc>=14.0.0"
                ]
            },
            "deployment": {
                "name": "Deployment & Containerization",
                "packages": [
                    "docker-compose>=1.29.0",
                    "aws-lambda-powertools>=2.20.0",
                    "google-cloud-functions>=1.8.0",
                    "azure-functions>=1.15.0"
                ]
            },
            "monitoring_alerting": {
                "name": "Monitoring & Alerting",
                "packages": [
                    "GPUtil>=1.4.0",
                    "nvidia-ml-py>=11.0.0",
                    "apm-client>=0.1.0",
                    "newrelic>=8.8.0",
                    "datadog>=0.44.0"
                ]
            },
            "integration": {
                "name": "Integration & Connectors",
                "packages": [
                    "slack-sdk>=3.21.0",
                    "discord.py>=2.3.0",
                    "telegram-bot>=0.13.0",
                    "twilio>=8.5.0",
                    "stripe>=6.0.0",
                    "paypal>=1.0.0",
                    "braintree>=4.0.0"
                ]
            },
            "experimental": {
                "name": "Experimental & Cutting-Edge",
                "packages": [
                    "qiskit>=0.44.0",
                    "cirq>=1.2.0",
                    "pennylane>=0.30.0",
                    "fedml>=0.7.0",
                    "syft>=0.5.0",
                    "tensorflow-lite>=2.13.0",
                    "onnxruntime>=1.15.0",
                    "openvino>=2023.0.0"
                ]
            },
            "utilities_libs": {
                "name": "Utility Libraries",
                "packages": [
                    "pyyaml>=6.0.0",
                    "toml>=0.10.0",
                    "json5>=0.9.0",
                    "urllib3>=2.0.0",
                    "certifi>=2023.7.0",
                    "charset-normalizer>=3.2.0"
                ]
            },
            "development_deps": {
                "name": "Development Dependencies",
                "packages": [
                    "pre-commit>=3.3.0",
                    "tox>=4.6.0",
                    "coverage>=7.2.0",
                    "pytest-benchmark>=4.0.0",
                    "sphinx-rtd-theme>=1.3.0",
                    "myst-parser>=2.0.0",
                    "sphinx-autodoc-typehints>=1.24.0"
                ]
            }
        }
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 9):
            logger.error(f"Python 3.9+ required. Current version: {version.major}.{version.minor}")
            return False
        logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_pip_version(self) -> bool:
        """Check pip version and upgrade if needed"""
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                  capture_output=True, text=True, check=True)
            logger.info(f"Pip version: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error checking pip version: {e}")
            return False
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version"""
        try:
            logger.info("Upgrading pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=not self.verbose)
            logger.info("Pip upgraded successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error upgrading pip: {e}")
            return False
    
    def check_package_installed(self, package_name: str) -> bool:
        """Check if a package is already installed"""
        try:
            # Extract package name without version constraints
            base_name = package_name.split('>=')[0].split('==')[0].split('<=')[0]
            subprocess.run([sys.executable, "-c", f"import {base_name}"], 
                         check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, ImportError):
            return False
    
    def install_package(self, package: str) -> Dict[str, Any]:
        """Install a single package"""
        start_time = time.time()
        result = {
            "package": package,
            "success": False,
            "error": None,
            "duration": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check if already installed
            if self.skip_existing and self.check_package_installed(package):
                result["success"] = True
                result["message"] = "Already installed"
                logger.info(f"✓ {package} (already installed)")
                return result
            
            # Install package
            logger.info(f"Installing {package}...")
            cmd = [sys.executable, "-m", "pip", "install", package]
            
            if self.verbose:
                subprocess.run(cmd, check=True)
            else:
                subprocess.run(cmd, check=True, capture_output=True)
            
            result["success"] = True
            result["duration"] = time.time() - start_time
            result["message"] = "Installed successfully"
            
            logger.info(f"✓ {package} (installed in {result['duration']:.2f}s)")
            
        except subprocess.CalledProcessError as e:
            result["success"] = False
            result["error"] = str(e)
            result["duration"] = time.time() - start_time
            logger.error(f"✗ {package} (failed: {e})")
        
        return result
    
    def install_category(self, category: str) -> Dict[str, Any]:
        """Install all packages in a category"""
        if category not in self.library_categories:
            return {"error": f"Unknown category: {category}"}
        
        category_info = self.library_categories[category]
        packages = category_info["packages"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Installing {category_info['name']} ({len(packages)} packages)")
        logger.info(f"{'='*60}")
        
        results = {
            "category": category,
            "name": category_info["name"],
            "total_packages": len(packages),
            "successful": 0,
            "failed": 0,
            "packages": []
        }
        
        for i, package in enumerate(packages, 1):
            logger.info(f"\n[{i}/{len(packages)}] Installing {package}")
            result = self.install_package(package)
            results["packages"].append(result)
            
            if result["success"]:
                results["successful"] += 1
                self.successful_installations.append(package)
            else:
                results["failed"] += 1
                self.failed_installations.append(package)
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.5)
        
        logger.info(f"\n{category_info['name']} installation completed:")
        logger.info(f"  ✓ Successful: {results['successful']}")
        logger.info(f"  ✗ Failed: {results['failed']}")
        
        return results
    
    def install_all_categories(self) -> Dict[str, Any]:
        """Install all library categories"""
        logger.info("🚀 Starting comprehensive library installation...")
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Skip existing: {self.skip_existing}")
        logger.info(f"Verbose mode: {self.verbose}")
        
        # Check prerequisites
        if not self.check_python_version():
            return {"error": "Python version check failed"}
        
        if not self.check_pip_version():
            return {"error": "Pip version check failed"}
        
        # Upgrade pip
        if not self.upgrade_pip():
            logger.warning("Failed to upgrade pip, continuing anyway...")
        
        # Install all categories
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "total_categories": len(self.library_categories),
            "categories": {},
            "summary": {
                "total_packages": 0,
                "successful_packages": 0,
                "failed_packages": 0
            }
        }
        
        for category in self.library_categories.keys():
            try:
                result = self.install_category(category)
                all_results["categories"][category] = result
                
                if "error" not in result:
                    all_results["summary"]["total_packages"] += result["total_packages"]
                    all_results["summary"]["successful_packages"] += result["successful"]
                    all_results["summary"]["failed_packages"] += result["failed"]
                
            except Exception as e:
                logger.error(f"Error installing category {category}: {e}")
                all_results["categories"][category] = {"error": str(e)}
        
        # Print summary
        self._print_installation_summary(all_results)
        
        # Save installation log
        self._save_installation_log(all_results)
        
        return all_results
    
    def _print_installation_summary(self, results: Dict[str, Any]):
        """Print installation summary"""
        logger.info(f"\n{'='*80}")
        logger.info("📊 INSTALLATION SUMMARY")
        logger.info(f"{'='*80}")
        
        summary = results["summary"]
        logger.info(f"Total packages: {summary['total_packages']}")
        logger.info(f"Successful: {summary['successful_packages']}")
        logger.info(f"Failed: {summary['failed_packages']}")
        logger.info(f"Success rate: {(summary['successful_packages']/summary['total_packages']*100):.1f}%" if summary['total_packages'] > 0 else "N/A")
        
        if self.failed_installations:
            logger.info(f"\n❌ Failed installations ({len(self.failed_installations)}):")
            for package in self.failed_installations:
                logger.info(f"  - {package}")
        
        logger.info(f"\n✅ Successful installations ({len(self.successful_installations)}):")
        for package in self.successful_installations[:10]:  # Show first 10
            logger.info(f"  - {package}")
        
        if len(self.successful_installations) > 10:
            logger.info(f"  ... and {len(self.successful_installations) - 10} more")
    
    def _save_installation_log(self, results: Dict[str, Any]):
        """Save installation log to file"""
        log_file = f"library_installation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(log_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Installation log saved to: {log_file}")
        except Exception as e:
            logger.error(f"Error saving installation log: {e}")
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories"""
        return list(self.library_categories.keys())
    
    def get_category_info(self, category: str) -> Dict[str, Any]:
        """Get information about a specific category"""
        if category not in self.library_categories:
            return {"error": f"Unknown category: {category}"}
        
        return {
            "category": category,
            **self.library_categories[category],
            "package_count": len(self.library_categories[category]["packages"])
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Install extended libraries for Enhanced Facebook Content Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install all libraries
  python install_extended_libraries.py
  
  # Install specific category
  python install_extended_libraries.py --category core_ai_ml
  
  # Skip already installed packages
  python install_extended_libraries.py --skip-existing
  
  # Verbose output
  python install_extended_libraries.py --verbose
  
  # List available categories
  python install_extended_libraries.py --list-categories
        """
    )
    
    parser.add_argument(
        "--category", "-c",
        help="Install specific category only"
    )
    
    parser.add_argument(
        "--skip-existing", "-s",
        action="store_true",
        help="Skip packages that are already installed"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--list-categories", "-l",
        action="store_true",
        help="List available categories and exit"
    )
    
    args = parser.parse_args()
    
    # Create installer
    installer = LibraryInstaller(
        skip_existing=args.skip_existing,
        verbose=args.verbose
    )
    
    # List categories if requested
    if args.list_categories:
        print("📚 Available Library Categories:")
        print("=" * 50)
        for category, info in installer.library_categories.items():
            print(f"{category:25} - {info['name']} ({len(info['packages'])} packages)")
        return
    
    # Install libraries
    if args.category:
        if args.category not in installer.library_categories:
            logger.error(f"Unknown category: {args.category}")
            logger.info("Use --list-categories to see available categories")
            return
        
        result = installer.install_category(args.category)
        if "error" in result:
            logger.error(f"Installation failed: {result['error']}")
    else:
        result = installer.install_all_categories()
        if "error" in result:
            logger.error(f"Installation failed: {result['error']}")
    
    logger.info("\n🎉 Installation process completed!")


if __name__ == "__main__":
    main()

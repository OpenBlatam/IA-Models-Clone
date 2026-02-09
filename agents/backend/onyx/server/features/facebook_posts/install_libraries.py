#!/usr/bin/env python3
"""
🚀 Extended Libraries Installation Script
========================================

Simple script to install additional libraries for the enhanced system.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a single package"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """Install extended libraries"""
    print("🚀 Installing Extended Libraries for Enhanced Facebook Content Optimization System")
    print("=" * 80)
    
    # Core AI/ML libraries
    core_packages = [
        "plotly>=5.15.0",
        "bokeh>=3.2.0",
        "altair>=5.1.0",
        "psutil>=5.9.0",
        "prometheus-client>=0.17.0",
        "structlog>=23.1.0",
        "rich>=13.4.0",
        "polars>=0.19.0",
        "dask>=2023.8.0",
        "statsmodels>=0.14.0",
        "prophet>=1.1.4",
        "yfinance>=0.2.18",
        "pandas-ta>=0.3.14b0",
        "geopandas>=0.13.0",
        "folium>=0.14.0",
        "shap>=0.42.0",
        "optuna>=3.2.0",
        "streamlit>=1.25.0",
        "dash>=2.11.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "sqlalchemy>=2.0.0",
        "redis>=4.6.0",
        "pymongo>=4.4.0",
        "celery>=5.3.0",
        "kafka-python>=2.0.0",
        "docker>=6.1.0",
        "kubernetes>=26.1.0",
        "pytest>=7.4.0",
        "black>=23.7.0",
        "isort>=5.12.0",
        "mypy>=1.5.0",
        "pre-commit>=3.3.0",
        "sphinx>=7.1.0",
        "mkdocs>=1.5.0"
    ]
    
    successful = 0
    failed = 0
    
    for package in core_packages:
        if install_package(package):
            successful += 1
        else:
            failed += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  📦 Total: {len(core_packages)}")
    
    if failed == 0:
        print("\n🎉 All libraries installed successfully!")
    else:
        print(f"\n⚠️  {failed} packages failed to install. Check the errors above.")

if __name__ == "__main__":
    main()

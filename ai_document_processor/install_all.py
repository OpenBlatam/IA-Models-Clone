#!/usr/bin/env python3
"""
Complete Installation Script - All Optimizations
===============================================

Complete installation script that installs all optimized libraries and configurations.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """Print installation banner"""
    print("\n" + "="*80)
    print("🚀 AI DOCUMENT PROCESSOR - COMPLETE INSTALLATION")
    print("="*80)
    print("Installing optimized libraries and configurations for maximum performance")
    print("="*80 + "\n")

def run_command(cmd, description):
    """Run a command with error handling"""
    logger.info(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e.stderr}")
        return False

def install_requirements():
    """Install requirements from optimized file"""
    requirements_file = Path(__file__).parent / "requirements_optimized.txt"
    if requirements_file.exists():
        return run_command(
            f"{sys.executable} -m pip install -r {requirements_file}",
            "Installing optimized requirements"
        )
    else:
        # Fallback to regular requirements
        return run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing requirements"
        )

def setup_environment():
    """Setup environment variables and optimizations"""
    logger.info("🔧 Setting up environment optimizations...")
    
    # Set environment variables for performance
    env_vars = {
        'OMP_NUM_THREADS': str(os.cpu_count() or 1),
        'MKL_NUM_THREADS': str(os.cpu_count() or 1),
        'NUMEXPR_NUM_THREADS': str(os.cpu_count() or 1),
        'OPENBLAS_NUM_THREADS': str(os.cpu_count() or 1),
        'VECLIB_MAXIMUM_THREADS': str(os.cpu_count() or 1),
        'NUMBA_NUM_THREADS': str(os.cpu_count() or 1),
        'PYTHONUNBUFFERED': '1',
        'PYTHONDONTWRITEBYTECODE': '1'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.debug(f"Set {key}={value}")
    
    logger.info("✅ Environment optimizations applied")
    return True

def test_installation():
    """Test that key libraries are working"""
    logger.info("🧪 Testing installation...")
    
    test_imports = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'numpy',
        'pandas',
        'torch',
        'transformers',
        'redis',
        'psutil'
    ]
    
    failed_imports = []
    for module in test_imports:
        try:
            __import__(module)
            logger.info(f"✅ {module} - OK")
        except ImportError:
            logger.error(f"❌ {module} - FAILED")
            failed_imports.append(module)
    
    if failed_imports:
        logger.warning(f"⚠️ {len(failed_imports)} libraries failed to import: {failed_imports}")
        return False
    else:
        logger.info("✅ All key libraries imported successfully")
        return True

def run_benchmarks():
    """Run performance benchmarks"""
    logger.info("⚡ Running performance benchmarks...")
    
    # Run library benchmarks
    benchmark_script = Path(__file__).parent / "benchmark_libraries.py"
    if benchmark_script.exists():
        return run_command(
            f"{sys.executable} {benchmark_script}",
            "Running library benchmarks"
        )
    else:
        logger.warning("⚠️ Benchmark script not found, skipping benchmarks")
        return True

def create_startup_scripts():
    """Create convenient startup scripts"""
    logger.info("📝 Creating startup scripts...")
    
    install_dir = Path(__file__).parent
    
    # Create Windows batch file
    batch_content = """@echo off
echo Starting AI Document Processor with optimized libraries...
python start_fast.py
pause
"""
    with open(install_dir / "start_optimized.bat", 'w') as f:
        f.write(batch_content)
    
    # Create Unix shell script
    shell_content = """#!/bin/bash
echo "Starting AI Document Processor with optimized libraries..."
python3 start_fast.py
"""
    with open(install_dir / "start_optimized.sh", 'w') as f:
        f.write(shell_content)
    
    # Make shell script executable
    if os.name != 'nt':
        os.chmod(install_dir / "start_optimized.sh", 0o755)
    
    logger.info("✅ Startup scripts created")
    return True

def print_completion_summary():
    """Print installation completion summary"""
    print("\n" + "="*80)
    print("🎉 INSTALLATION COMPLETE!")
    print("="*80)
    
    print("\n🚀 Quick Start:")
    if os.name == 'nt':
        print("   start_optimized.bat")
    else:
        print("   ./start_optimized.sh")
    print("   # OR")
    print("   python start_fast.py")
    
    print("\n📊 Performance Testing:")
    print("   python benchmark_libraries.py")
    print("   python benchmark_speed.py")
    
    print("\n🔧 Configuration:")
    print("   Edit .env file for custom settings")
    print("   python library_config.py - for library optimizations")
    
    print("\n📚 Documentation:")
    print("   SPEED_IMPROVEMENTS.md - Performance guide")
    print("   requirements_optimized.txt - Enhanced libraries")
    print("   library_config.py - Library optimization config")
    
    print("\n🌐 API Endpoints:")
    print("   • Health: http://localhost:8001/health")
    print("   • Metrics: http://localhost:8001/metrics")
    print("   • Process: http://localhost:8001/process")
    print("   • Docs: http://localhost:8001/docs")
    
    print("\n💡 Performance Tips:")
    print("   • Use Redis for better cache performance")
    print("   • Process files in batches for better throughput")
    print("   • Monitor /metrics endpoint for performance data")
    print("   • Run benchmarks to verify optimizations")
    
    print("="*80 + "\n")

def main():
    """Main installation function"""
    start_time = time.time()
    
    try:
        print_banner()
        
        # Step 1: Setup environment
        if not setup_environment():
            logger.error("❌ Environment setup failed")
            return False
        
        # Step 2: Install requirements
        if not install_requirements():
            logger.error("❌ Requirements installation failed")
            return False
        
        # Step 3: Test installation
        if not test_installation():
            logger.warning("⚠️ Some libraries failed to import, but continuing...")
        
        # Step 4: Run benchmarks (optional)
        run_benchmarks()
        
        # Step 5: Create startup scripts
        if not create_startup_scripts():
            logger.warning("⚠️ Failed to create startup scripts")
        
        # Step 6: Print completion summary
        print_completion_summary()
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Installation completed in {total_time:.1f} seconds")
        
        return True
        
    except KeyboardInterrupt:
        logger.info("🛑 Installation interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Installation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


















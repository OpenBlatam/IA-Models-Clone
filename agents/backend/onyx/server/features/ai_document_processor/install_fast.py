#!/usr/bin/env python3
"""
Fast Installation Script - Quick Setup for Speed Optimizations
=============================================================

Automated installation and configuration script for the fast AI document processor.
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastInstaller:
    """Fast installation and configuration manager"""
    
    def __init__(self):
        self.system_info = self.get_system_info()
        self.install_dir = Path(__file__).parent
        self.config_file = self.install_dir / ".env"
    
    def get_system_info(self):
        """Get system information for optimization"""
        import psutil
        
        return {
            'platform': platform.system(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'python_version': sys.version,
            'architecture': platform.machine()
        }
    
    def print_banner(self):
        """Print installation banner"""
        print("\n" + "="*80)
        print("🚀 FAST AI DOCUMENT PROCESSOR - INSTALLATION")
        print("="*80)
        print(f"System: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"CPU Cores: {self.system_info['cpu_count']}")
        print(f"Memory: {self.system_info['memory_gb']} GB")
        print(f"Python: {self.system_info['python_version'].split()[0]}")
        print("="*80 + "\n")
    
    def check_requirements(self):
        """Check system requirements"""
        logger.info("🔍 Checking system requirements...")
        
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append("Python 3.8+ required")
        
        # Check memory
        if self.system_info['memory_gb'] < 2:
            issues.append("At least 2GB RAM recommended")
        
        # Check CPU
        if self.system_info['cpu_count'] < 2:
            issues.append("At least 2 CPU cores recommended")
        
        if issues:
            logger.warning("⚠️ System requirements issues:")
            for issue in issues:
                logger.warning(f"   - {issue}")
            return False
        
        logger.info("✅ System requirements met")
        return True
    
    def install_dependencies(self):
        """Install required dependencies"""
        logger.info("📦 Installing dependencies...")
        
        try:
            # Install from requirements.txt
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True, cwd=self.install_dir)
            
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def create_optimized_config(self):
        """Create optimized configuration based on system"""
        logger.info("⚙️ Creating optimized configuration...")
        
        # Determine optimal settings based on system
        if self.system_info['memory_gb'] >= 8 and self.system_info['cpu_count'] >= 8:
            preset = 'ultra_fast'
        elif self.system_info['memory_gb'] >= 4 and self.system_info['cpu_count'] >= 4:
            preset = 'balanced'
        else:
            preset = 'memory_efficient'
        
        # Calculate optimal settings
        max_workers = min(32, self.system_info['cpu_count'] * 2 + 4)
        cache_memory = min(2048, int(self.system_info['memory_gb'] * 0.25 * 1024))
        
        config = {
            # Performance settings
            'MAX_WORKERS': str(max_workers),
            'CACHE_MAX_MEMORY_MB': str(cache_memory),
            'ENABLE_STREAMING': 'true',
            'ENABLE_PARALLEL_AI': 'true' if self.system_info['cpu_count'] >= 4 else 'false',
            'ENABLE_UVLOOP': 'true' if self.system_info['platform'] != 'Windows' else 'false',
            'ENABLE_COMPRESSION': 'true',
            
            # Cache settings
            'CACHE_DEFAULT_TTL': '3600',
            'ENABLE_CACHE': 'true',
            
            # AI settings optimized for speed
            'OPENAI_MODEL': 'gpt-3.5-turbo',
            'OPENAI_TEMPERATURE': '0.3',
            'OPENAI_TIMEOUT': '30',
            
            # File processing
            'MAX_FILE_SIZE': '104857600',  # 100MB
            'MAX_BATCH_SIZE': '10',
            'CHUNK_SIZE': '8192',
            
            # Memory optimization
            'MEMORY_OPTIMIZATION': 'true',
            'GC_THRESHOLD': '100',
            'MAX_MEMORY_USAGE_PERCENT': '80.0',
            
            # Monitoring
            'ENABLE_METRICS': 'true',
            'METRICS_RETENTION_HOURS': '24',
            'HEALTH_CHECK_INTERVAL': '30',
            
            # Logging
            'LOG_LEVEL': 'INFO',
            'ENABLE_STRUCTURED_LOGGING': 'true',
            
            # Security
            'ENABLE_RATE_LIMITING': 'true',
            'RATE_LIMIT_REQUESTS': '100',
            'RATE_LIMIT_WINDOW': '60',
            
            # Advanced optimizations
            'ENABLE_GZIP': 'true',
            'GZIP_MINIMUM_SIZE': '1000',
            'BATCH_PROCESSING_TIMEOUT': '300',
            'AI_PROCESSING_TIMEOUT': '60',
            'AI_RETRY_ATTEMPTS': '3',
            'AI_RETRY_DELAY': '1.0',
            'FILE_BUFFER_SIZE': '65536',
            'ENABLE_FILE_STREAMING': 'true',
            'MAX_CONCURRENT_FILES': '50'
        }
        
        # Write configuration file
        with open(self.config_file, 'w') as f:
            f.write("# Fast AI Document Processor Configuration\n")
            f.write("# Auto-generated optimized settings\n\n")
            for key, value in config.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"✅ Configuration created: {preset} preset")
        logger.info(f"   Max Workers: {max_workers}")
        logger.info(f"   Cache Memory: {cache_memory} MB")
        logger.info(f"   Parallel AI: {config['ENABLE_PARALLEL_AI']}")
        logger.info(f"   UVLoop: {config['ENABLE_UVLOOP']}")
        
        return True
    
    def create_startup_scripts(self):
        """Create convenient startup scripts"""
        logger.info("📝 Creating startup scripts...")
        
        # Create Windows batch file
        if self.system_info['platform'] == 'Windows':
            batch_content = f"""@echo off
echo Starting Fast AI Document Processor...
python start_fast.py
pause
"""
            with open(self.install_dir / "start_fast.bat", 'w') as f:
                f.write(batch_content)
        
        # Create Unix shell script
        shell_content = f"""#!/bin/bash
echo "Starting Fast AI Document Processor..."
python3 start_fast.py
"""
        with open(self.install_dir / "start_fast.sh", 'w') as f:
            f.write(shell_content)
        
        # Make shell script executable
        if self.system_info['platform'] != 'Windows':
            os.chmod(self.install_dir / "start_fast.sh", 0o755)
        
        logger.info("✅ Startup scripts created")
        return True
    
    def run_benchmark(self):
        """Run initial benchmark"""
        logger.info("🏃 Running initial benchmark...")
        
        try:
            result = subprocess.run([
                sys.executable, "benchmark_speed.py"
            ], cwd=self.install_dir, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Benchmark completed successfully")
                return True
            else:
                logger.warning(f"⚠️ Benchmark completed with warnings: {result.stderr}")
                return True
                
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Benchmark timed out (this is normal for first run)")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Benchmark failed: {e}")
            return True
    
    def print_usage_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*80)
        print("🎉 INSTALLATION COMPLETE!")
        print("="*80)
        
        print("\n🚀 Quick Start:")
        if self.system_info['platform'] == 'Windows':
            print("   start_fast.bat")
        else:
            print("   ./start_fast.sh")
        print("   # OR")
        print("   python start_fast.py")
        
        print("\n📊 API Endpoints:")
        print("   • Health Check: http://localhost:8001/health")
        print("   • Metrics: http://localhost:8001/metrics")
        print("   • Process Document: http://localhost:8001/process")
        print("   • API Docs: http://localhost:8001/docs")
        
        print("\n🔧 Configuration:")
        print(f"   Config file: {self.config_file}")
        print("   Edit .env file to customize settings")
        
        print("\n📈 Performance Tips:")
        print("   • Use Redis for better cache performance")
        print("   • Process files in batches for better throughput")
        print("   • Monitor /metrics endpoint for performance data")
        print("   • Run benchmark_speed.py to test improvements")
        
        print("\n📚 Documentation:")
        print("   • SPEED_IMPROVEMENTS.md - Detailed performance guide")
        print("   • README.md - General documentation")
        print("   • API docs at http://localhost:8001/docs")
        
        print("\n🆘 Support:")
        print("   • Check /health endpoint for system status")
        print("   • View logs in fast_processor.log")
        print("   • Run benchmark for performance analysis")
        
        print("="*80 + "\n")
    
    def install(self):
        """Run complete installation"""
        try:
            self.print_banner()
            
            # Check requirements
            if not self.check_requirements():
                logger.warning("⚠️ Continuing despite requirements issues...")
            
            # Install dependencies
            if not self.install_dependencies():
                logger.error("❌ Installation failed at dependencies step")
                return False
            
            # Create configuration
            if not self.create_optimized_config():
                logger.error("❌ Installation failed at configuration step")
                return False
            
            # Create startup scripts
            if not self.create_startup_scripts():
                logger.error("❌ Installation failed at startup scripts step")
                return False
            
            # Run benchmark (optional)
            self.run_benchmark()
            
            # Print instructions
            self.print_usage_instructions()
            
            logger.info("🎉 Installation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Installation failed: {e}")
            return False

def main():
    """Main installation function"""
    installer = FastInstaller()
    success = installer.install()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()




















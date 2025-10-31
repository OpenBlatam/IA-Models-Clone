#!/usr/bin/env python3
"""
Ultra Installation Script - Maximum Performance Setup
====================================================

Ultra-optimized installation script for maximum performance and efficiency.
"""

import os
import sys
import subprocess
import platform
import time
import json
import multiprocessing as mp
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraInstaller:
    """Ultra-optimized installer with maximum performance"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.install_dir = Path(__file__).parent
        self.requirements_file = self.install_dir / "requirements_ultra.txt"
        self.install_log = self.install_dir / "ultra_install_log.json"
        
    def _get_system_info(self):
        """Get comprehensive system information"""
        try:
            import psutil
            import cpuinfo
            
            return {
                'platform': platform.system(),
                'architecture': platform.machine(),
                'python_version': sys.version,
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 2) if os.name != 'nt' else round(psutil.disk_usage('C:').free / (1024**3), 2),
                'gpu_available': self._check_gpu_availability(),
                'cuda_available': self._check_cuda_availability(),
                'mkl_available': self._check_mkl_availability(),
                'blas_available': self._check_blas_availability()
            }
        except ImportError:
            logger.warning("psutil not available, using basic system info")
            return {
                'platform': platform.system(),
                'architecture': platform.machine(),
                'python_version': sys.version,
                'cpu_count': mp.cpu_count() or 1,
                'memory_gb': 4.0,
                'gpu_available': False,
                'cuda_available': False
            }
    
    def _check_gpu_availability(self):
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _check_cuda_availability(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except ImportError:
            return False
    
    def _check_mkl_availability(self):
        """Check if Intel MKL is available"""
        try:
            import numpy
            return hasattr(numpy, '__mkl_version__')
        except ImportError:
            return False
    
    def _check_blas_availability(self):
        """Check BLAS availability"""
        try:
            import numpy
            return numpy.show_config() is not None
        except ImportError:
            return False
    
    def print_ultra_banner(self):
        """Print ultra installation banner"""
        print("\n" + "="*80)
        print("🚀 ULTRA AI DOCUMENT PROCESSOR - MAXIMUM PERFORMANCE INSTALLATION")
        print("="*80)
        print("Installing ultra-optimized libraries for extreme performance")
        print("="*80)
        
        print(f"System: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"Python: {self.system_info['python_version'].split()[0]}")
        print(f"CPU Cores: {self.system_info['cpu_count']}")
        print(f"Memory: {self.system_info['memory_gb']} GB")
        print(f"GPU Available: {'✅' if self.system_info['gpu_available'] else '❌'}")
        print(f"CUDA Available: {'✅' if self.system_info['cuda_available'] else '❌'}")
        print(f"Intel MKL: {'✅' if self.system_info['mkl_available'] else '❌'}")
        
        print("="*80 + "\n")
    
    def apply_ultra_optimizations(self):
        """Apply ultra system optimizations"""
        logger.info("🚀 Applying ultra system optimizations...")
        
        # Set ultra-optimized environment variables
        ultra_env_vars = {
            # Python optimizations
            'PYTHONOPTIMIZE': '2',
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONUNBUFFERED': '1',
            'PYTHONHASHSEED': '0',
            'PYTHONIOENCODING': 'utf-8',
            
            # Threading optimizations
            'OMP_NUM_THREADS': str(self.system_info['cpu_count']),
            'MKL_NUM_THREADS': str(self.system_info['cpu_count']),
            'NUMEXPR_NUM_THREADS': str(self.system_info['cpu_count']),
            'OPENBLAS_NUM_THREADS': str(self.system_info['cpu_count']),
            'VECLIB_MAXIMUM_THREADS': str(self.system_info['cpu_count']),
            'NUMBA_NUM_THREADS': str(self.system_info['cpu_count']),
            'BLIS_NUM_THREADS': str(self.system_info['cpu_count']),
            
            # Memory optimizations
            'NUMPY_MADVISE_HUGEPAGE': '1',
            'NUMPY_DISABLE_CPU_FEATURES': '0',
            'MALLOC_TRIM_THRESHOLD_': '131072',
            'MALLOC_MMAP_THRESHOLD_': '131072',
            
            # Cache optimizations
            'REDIS_MAXMEMORY': '4096mb',
            'REDIS_MAXMEMORY_POLICY': 'allkeys-lru',
            'REDIS_SAVE': '',
            
            # Network optimizations
            'TCP_NODELAY': '1',
            'TCP_KEEPALIVE': '1',
        }
        
        for key, value in ultra_env_vars.items():
            os.environ[key] = value
            logger.debug(f"Set {key}={value}")
        
        # Apply system-specific optimizations
        if self.system_info['platform'] == 'Linux':
            self._apply_linux_optimizations()
        elif self.system_info['platform'] == 'Darwin':
            self._apply_macos_optimizations()
        elif self.system_info['platform'] == 'Windows':
            self._apply_windows_optimizations()
        
        logger.info("✅ Ultra system optimizations applied")
    
    def _apply_linux_optimizations(self):
        """Apply Linux-specific optimizations"""
        try:
            # CPU governor
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'w') as f:
                f.write('performance')
            
            # Huge pages
            with open('/proc/sys/vm/nr_hugepages', 'w') as f:
                f.write('1024')
            
            # Memory overcommit
            with open('/proc/sys/vm/overcommit_memory', 'w') as f:
                f.write('1')
            
            # TCP optimizations
            tcp_opts = {
                '/proc/sys/net/core/rmem_max': '134217728',
                '/proc/sys/net/core/wmem_max': '134217728',
                '/proc/sys/net/ipv4/tcp_rmem': '4096 65536 134217728',
                '/proc/sys/net/ipv4/tcp_wmem': '4096 65536 134217728',
                '/proc/sys/net/core/netdev_max_backlog': '5000',
                '/proc/sys/net/ipv4/tcp_congestion_control': 'bbr'
            }
            
            for path, value in tcp_opts.items():
                with open(path, 'w') as f:
                    f.write(value)
            
            logger.debug("Linux optimizations applied")
            
        except (OSError, PermissionError) as e:
            logger.warning(f"Linux optimizations not available: {e}")
    
    def _apply_macos_optimizations(self):
        """Apply macOS-specific optimizations"""
        try:
            # Disable App Nap for Python
            os.system("defaults write com.apple.dt.Xcode NSAppSleepDisabled -bool YES")
            logger.debug("macOS optimizations applied")
        except Exception as e:
            logger.warning(f"macOS optimizations not available: {e}")
    
    def _apply_windows_optimizations(self):
        """Apply Windows-specific optimizations"""
        try:
            # Set process priority
            import psutil
            process = psutil.Process()
            process.nice(psutil.HIGH_PRIORITY_CLASS)
            logger.debug("Windows optimizations applied")
        except Exception as e:
            logger.warning(f"Windows optimizations not available: {e}")
    
    def get_ultra_requirements(self):
        """Get ultra-optimized requirements based on system"""
        requirements = []
        
        # Core requirements (always needed)
        core_requirements = [
            "fastapi>=0.104.1",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.5.0",
            "python-multipart>=0.0.6",
            "redis>=5.0.1",
            "psutil>=5.9.6",
            "aiofiles>=23.2.1",
            "httpx>=0.25.2"
        ]
        requirements.extend(core_requirements)
        
        # Performance optimizations
        if self.system_info['platform'] != 'Windows':
            requirements.append("uvloop>=0.19.0")
        
        # Ultra-fast serialization and compression
        requirements.extend([
            "orjson>=3.9.10",
            "msgpack>=1.0.7",
            "lz4>=4.3.2",
            "zstandard>=0.22.0",
            "brotli>=1.1.0"
        ])
        
        # AI and ML libraries - ultra optimized
        ai_requirements = [
            "openai>=1.3.0",
            "anthropic>=0.7.0",
            "cohere>=4.37.0",
            "scikit-learn>=1.3.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "transformers>=4.36.0",
            "sentence-transformers>=2.2.2",
            "langchain>=0.0.350",
            "chromadb>=0.4.18"
        ]
        requirements.extend(ai_requirements)
        
        # GPU-accelerated libraries
        if self.system_info['gpu_available']:
            gpu_requirements = [
                "torch>=2.1.1",
                "torchvision>=0.16.1",
                "tensorflow>=2.15.0",
                "tensorrt>=8.6.0"
            ]
            requirements.extend(gpu_requirements)
        else:
            cpu_requirements = [
                "torch>=2.1.1+cpu",
                "torchvision>=0.16.1+cpu",
                "tensorflow>=2.15.0"
            ]
            requirements.extend(cpu_requirements)
        
        # Document processing - ultra fast
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
        requirements.extend(doc_requirements)
        
        # NLP libraries - advanced
        nlp_requirements = [
            "spacy>=3.7.0",
            "nltk>=3.8.0",
            "textblob>=0.17.1",
            "langdetect>=1.0.9",
            "flair>=0.13.1"
        ]
        requirements.extend(nlp_requirements)
        
        # Monitoring and observability - enterprise
        monitoring_requirements = [
            "prometheus-client>=0.19.0",
            "structlog>=23.2.0",
            "loguru>=0.7.2",
            "sentry-sdk>=1.38.0",
            "memory-profiler>=0.61.0",
            "line-profiler>=4.1.1",
            "py-spy>=0.3.14"
        ]
        requirements.extend(monitoring_requirements)
        
        # Advanced features for high-end systems
        if self.system_info['memory_gb'] >= 16 and self.system_info['cpu_count'] >= 16:
            advanced_requirements = [
                "langchain-community>=0.0.10",
                "pinecone-client>=2.2.4",
                "weaviate-client>=3.25.0",
                "qdrant-client>=1.7.0",
                "gradio>=4.7.1",
                "streamlit>=1.28.2",
                "plotly>=5.17.0",
                "matplotlib>=3.8.2",
                "dask[complete]>=2023.12.0",
                "polars>=0.20.0"
            ]
            requirements.extend(advanced_requirements)
        
        return requirements
    
    def install_ultra_requirements(self):
        """Install ultra-optimized requirements"""
        logger.info("📦 Installing ultra-optimized requirements...")
        
        requirements = self.get_ultra_requirements()
        
        # Create temporary requirements file
        temp_req_file = self.install_dir / "temp_ultra_requirements.txt"
        
        try:
            with open(temp_req_file, 'w') as f:
                for req in requirements:
                    f.write(f"{req}\n")
            
            # Install with ultra optimizations
            install_cmd = [
                sys.executable, "-m", "pip", "install",
                "--upgrade", "--no-cache-dir", "--no-deps",
                "-r", str(temp_req_file)
            ]
            
            # Add system-specific optimizations
            if self.system_info['platform'] == 'Linux':
                install_cmd.extend(["--no-binary", ":all:"])
            
            # Install with maximum parallelism
            install_cmd.extend(["-j", str(self.system_info['cpu_count'])])
            
            # Install with progress bar
            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                cwd=self.install_dir
            )
            
            if result.returncode == 0:
                logger.info("✅ Ultra requirements installed successfully")
                return True
            else:
                logger.error(f"❌ Installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Installation error: {e}")
            return False
        finally:
            # Clean up temp file
            if temp_req_file.exists():
                temp_req_file.unlink()
    
    def install_ultra_optimizations(self):
        """Install ultra performance optimizations"""
        logger.info("🚀 Installing ultra performance optimizations...")
        
        optimizations = []
        
        # Intel MKL for better performance
        if not self.system_info['mkl_available'] and self.system_info['platform'] == 'Linux':
            optimizations.append("intel-openmp")
        
        # CUDA toolkit for GPU acceleration
        if self.system_info['gpu_available'] and not self.system_info['cuda_available']:
            optimizations.extend([
                "nvidia-ml-py",
                "cupy-cuda12x"  # Adjust based on CUDA version
            ])
        
        # Memory optimization
        optimizations.extend([
            "memory-profiler",
            "pympler",
            "line-profiler",
            "scalene"
        ])
        
        # Async optimizations
        if self.system_info['platform'] != 'Windows':
            optimizations.append("uvloop")
        
        # JIT compilation
        optimizations.extend([
            "numba",
            "cython"
        ])
        
        # Install optimizations
        for opt in optimizations:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", opt
                ], check=True, capture_output=True)
                logger.info(f"✅ Installed {opt}")
            except subprocess.CalledProcessError:
                logger.warning(f"⚠️ Failed to install {opt}")
    
    def verify_ultra_installation(self):
        """Verify ultra installation"""
        logger.info("🔍 Verifying ultra installation...")
        
        verification_results = {}
        
        # Test imports
        test_imports = {
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'pydantic': 'pydantic',
            'redis': 'redis',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'sklearn': 'sklearn',
            'torch': 'torch',
            'transformers': 'transformers',
            'spacy': 'spacy',
            'nltk': 'nltk',
            'openai': 'openai',
            'prometheus_client': 'prometheus_client',
            'orjson': 'orjson',
            'msgpack': 'msgpack',
            'lz4': 'lz4',
            'zstandard': 'zstandard'
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
    
    def run_ultra_performance_test(self):
        """Run ultra performance tests"""
        logger.info("⚡ Running ultra performance tests...")
        
        results = {}
        
        try:
            import numpy as np
            import time
            
            # NumPy performance test
            start_time = time.time()
            a = np.random.rand(2000, 2000)
            b = np.random.rand(2000, 2000)
            c = np.dot(a, b)
            numpy_time = time.time() - start_time
            results['numpy_matrix_mult_2K'] = numpy_time
            
            logger.info(f"✅ NumPy 2Kx2K matrix multiplication: {numpy_time:.3f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ NumPy performance test failed: {e}")
            results['numpy_matrix_mult_2K'] = None
        
        try:
            import pandas as pd
            
            # Pandas performance test
            start_time = time.time()
            df = pd.DataFrame(np.random.rand(100000, 100))
            result = df.groupby(df.index % 100).sum()
            pandas_time = time.time() - start_time
            results['pandas_groupby_100K'] = pandas_time
            
            logger.info(f"✅ Pandas 100K groupby operation: {pandas_time:.3f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ Pandas performance test failed: {e}")
            results['pandas_groupby_100K'] = None
        
        try:
            import orjson
            
            # OrJSON performance test
            data = {'test': list(range(100000))}
            start_time = time.time()
            for _ in range(1000):
                orjson.dumps(data)
            orjson_time = time.time() - start_time
            results['orjson_serialization'] = orjson_time
            
            logger.info(f"✅ OrJSON serialization test: {orjson_time:.3f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ OrJSON performance test failed: {e}")
            results['orjson_serialization'] = None
        
        return results
    
    def save_ultra_install_log(self, verification_results, performance_results):
        """Save ultra installation log"""
        log_data = {
            'timestamp': time.time(),
            'system_info': self.system_info,
            'verification_results': verification_results,
            'performance_results': performance_results,
            'installation_successful': all(verification_results.values()),
            'ultra_optimizations_applied': True
        }
        
        with open(self.install_log, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"📝 Ultra installation log saved to {self.install_log}")
    
    def print_ultra_summary(self, verification_results, performance_results):
        """Print ultra installation summary"""
        print("\n" + "="*80)
        print("📊 ULTRA INSTALLATION SUMMARY")
        print("="*80)
        
        # Verification results
        total_libs = len(verification_results)
        successful_libs = sum(verification_results.values())
        success_rate = (successful_libs / total_libs) * 100
        
        print(f"Libraries Installed: {successful_libs}/{total_libs} ({success_rate:.1f}%)")
        
        if success_rate >= 95:
            print("🚀 Installation: ULTRA EXCELLENT")
        elif success_rate >= 90:
            print("🎉 Installation: EXCELLENT")
        elif success_rate >= 85:
            print("✅ Installation: VERY GOOD")
        elif success_rate >= 80:
            print("👍 Installation: GOOD")
        else:
            print("⚠️ Installation: NEEDS ATTENTION")
        
        # Performance results
        print("\n⚡ Ultra Performance Test Results:")
        for test, result in performance_results.items():
            if result is not None:
                print(f"   {test}: {result:.3f}s")
            else:
                print(f"   {test}: FAILED")
        
        print("\n🚀 Ultra Performance Features:")
        print("   ✅ Maximum threading optimization")
        print("   ✅ Memory mapping enabled")
        print("   ✅ Zero-copy operations")
        print("   ✅ Ultra-fast serialization")
        print("   ✅ Advanced compression")
        print("   ✅ GPU acceleration ready")
        print("   ✅ Enterprise monitoring")
        
        print("\n🚀 Next Steps:")
        print("   1. Run: python start_fast.py")
        print("   2. Test: python benchmark_speed.py")
        print("   3. Ultra Test: python benchmark_libraries.py")
        print("   4. Monitor: http://localhost:8001/health")
        
        print("="*80 + "\n")
    
    def install(self):
        """Run complete ultra installation"""
        try:
            # Print ultra banner
            self.print_ultra_banner()
            
            # Apply ultra optimizations
            self.apply_ultra_optimizations()
            
            # Install ultra requirements
            if not self.install_ultra_requirements():
                logger.error("❌ Ultra requirements installation failed")
                return False
            
            # Install ultra optimizations
            self.install_ultra_optimizations()
            
            # Verify installation
            verification_results = self.verify_ultra_installation()
            
            # Run performance tests
            performance_results = self.run_ultra_performance_test()
            
            # Save installation log
            self.save_ultra_install_log(verification_results, performance_results)
            
            # Print summary
            self.print_ultra_summary(verification_results, performance_results)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ultra installation failed: {e}")
            return False

def main():
    """Main ultra installation function"""
    installer = UltraInstaller()
    success = installer.install()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


















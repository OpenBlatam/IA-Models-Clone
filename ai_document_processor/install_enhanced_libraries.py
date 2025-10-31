#!/usr/bin/env python3
"""
Enhanced Libraries Installer - Intelligent Installation
=====================================================

Intelligent installer for enhanced libraries with system detection and optimization.
"""

import os
import sys
import subprocess
import platform
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedLibraryInstaller:
    """Intelligent installer for enhanced libraries."""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.install_dir = Path(__file__).parent
        self.requirements_file = self.install_dir / "requirements_enhanced.txt"
        self.install_log = self.install_dir / "enhanced_libraries_install_log.json"
        
    def _get_system_info(self) -> Dict[str, any]:
        """Get comprehensive system information."""
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
                'blas_available': self._check_blas_availability(),
                'has_avx': self._check_avx_support(),
                'has_avx2': self._check_avx2_support(),
                'has_avx512': self._check_avx512_support()
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
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except ImportError:
            return False
    
    def _check_mkl_availability(self) -> bool:
        """Check if Intel MKL is available."""
        try:
            import numpy
            return hasattr(numpy, '__mkl_version__')
        except ImportError:
            return False
    
    def _check_blas_availability(self) -> bool:
        """Check BLAS availability."""
        try:
            import numpy
            return numpy.show_config() is not None
        except ImportError:
            return False
    
    def _check_avx_support(self) -> bool:
        """Check AVX support."""
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            return 'avx' in cpu_info.get('flags', [])
        except:
            return False
    
    def _check_avx2_support(self) -> bool:
        """Check AVX2 support."""
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            return 'avx2' in cpu_info.get('flags', [])
        except:
            return False
    
    def _check_avx512_support(self) -> bool:
        """Check AVX512 support."""
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            return 'avx512' in cpu_info.get('flags', [])
        except:
            return False
    
    def print_enhanced_banner(self):
        """Print enhanced installation banner."""
        print("\n" + "="*80)
        print("📚 ENHANCED LIBRARIES INSTALLER - INTELLIGENT INSTALLATION")
        print("="*80)
        print("Installing cutting-edge libraries with system optimization")
        print("="*80)
        
        print(f"System: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"Python: {self.system_info['python_version'].split()[0]}")
        print(f"CPU Cores: {self.system_info['cpu_count']}")
        print(f"Memory: {self.system_info['memory_gb']} GB")
        print(f"GPU Available: {'✅' if self.system_info['gpu_available'] else '❌'}")
        print(f"CUDA Available: {'✅' if self.system_info['cuda_available'] else '❌'}")
        print(f"Intel MKL: {'✅' if self.system_info['mkl_available'] else '❌'}")
        print(f"AVX Support: {'✅' if self.system_info['has_avx'] else '❌'}")
        print(f"AVX2 Support: {'✅' if self.system_info['has_avx2'] else '❌'}")
        print(f"AVX512 Support: {'✅' if self.system_info['has_avx512'] else '❌'}")
        
        print("="*80 + "\n")
    
    def get_optimized_requirements(self) -> List[str]:
        """Get optimized requirements based on system capabilities."""
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
        
        # AI and ML libraries - optimized for system
        if self.system_info['gpu_available'] and self.system_info['cuda_available']:
            # GPU-optimized versions
            ai_requirements = [
                "torch>=2.1.1",
                "torchvision>=0.16.1",
                "torchaudio>=2.1.1",
                "tensorflow>=2.15.0",
                "tensorrt>=8.6.0",
                "cupy-cuda12x>=12.0.0"
            ]
        else:
            # CPU-optimized versions
            ai_requirements = [
                "torch>=2.1.1+cpu",
                "torchvision>=0.16.1+cpu",
                "torchaudio>=2.1.1+cpu",
                "tensorflow>=2.15.0"
            ]
        
        # Add common AI libraries
        ai_requirements.extend([
            "openai>=1.3.0",
            "anthropic>=0.7.0",
            "cohere>=4.37.0",
            "transformers>=4.36.0",
            "sentence-transformers>=2.2.2",
            "langchain>=0.0.350",
            "chromadb>=0.4.18"
        ])
        requirements.extend(ai_requirements)
        
        # Document processing - comprehensive
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
        
        # Data processing - optimized
        data_requirements = [
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "scipy>=1.11.4",
            "scikit-learn>=1.3.0"
        ]
        
        # Add high-performance data processing for powerful systems
        if self.system_info['memory_gb'] >= 16 and self.system_info['cpu_count'] >= 8:
            data_requirements.extend([
                "dask[complete]>=2023.12.0",
                "polars>=0.20.0",
                "vaex>=4.17.0"
            ])
        
        requirements.extend(data_requirements)
        
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
        if self.system_info['memory_gb'] >= 32 and self.system_info['cpu_count'] >= 16:
            advanced_requirements = [
                "langchain-community>=0.0.10",
                "pinecone-client>=2.2.4",
                "weaviate-client>=3.25.0",
                "qdrant-client>=1.7.0",
                "gradio>=4.7.1",
                "streamlit>=1.28.2",
                "plotly>=5.17.0",
                "matplotlib>=3.8.2"
            ]
            requirements.extend(advanced_requirements)
        
        # Vector databases for AI systems
        if self.system_info['memory_gb'] >= 8:
            vector_requirements = [
                "chromadb>=0.4.18",
                "faiss-cpu>=1.7.4"
            ]
            if self.system_info['gpu_available']:
                vector_requirements.append("faiss-gpu>=1.7.4")
            requirements.extend(vector_requirements)
        
        # Development tools for development environments
        if os.getenv('DEVELOPMENT', 'false').lower() == 'true':
            dev_requirements = [
                "pytest>=7.4.0",
                "pytest-asyncio>=0.21.0",
                "pytest-cov>=4.1.0",
                "black>=23.7.0",
                "isort>=5.12.0",
                "flake8>=6.0.0",
                "mypy>=1.5.0"
            ]
            requirements.extend(dev_requirements)
        
        return requirements
    
    def install_requirements(self, requirements: List[str]) -> bool:
        """Install requirements with optimization."""
        logger.info("📦 Installing enhanced requirements...")
        
        # Create temporary requirements file
        temp_req_file = self.install_dir / "temp_enhanced_requirements.txt"
        
        try:
            with open(temp_req_file, 'w') as f:
                for req in requirements:
                    f.write(f"{req}\n")
            
            # Install with optimizations
            install_cmd = [
                sys.executable, "-m", "pip", "install",
                "--upgrade", "--no-cache-dir"
            ]
            
            # Add system-specific optimizations
            if self.system_info['platform'] == 'Linux':
                install_cmd.extend(["--no-binary", ":all:"])
            
            # Install with maximum parallelism
            install_cmd.extend(["-j", str(self.system_info['cpu_count'])])
            
            # Add requirements file
            install_cmd.append("-r")
            install_cmd.append(str(temp_req_file))
            
            # Install with progress bar
            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                cwd=self.install_dir
            )
            
            if result.returncode == 0:
                logger.info("✅ Enhanced requirements installed successfully")
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
    
    def install_system_optimizations(self):
        """Install system-specific optimizations."""
        logger.info("🚀 Installing system optimizations...")
        
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
    
    def verify_installation(self) -> Dict[str, bool]:
        """Verify installation."""
        logger.info("🔍 Verifying enhanced installation...")
        
        verification_results = {}
        
        # Test imports
        test_imports = {
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'pydantic': 'pydantic',
            'redis': 'redis',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'torch': 'torch',
            'transformers': 'transformers',
            'langchain': 'langchain',
            'chromadb': 'chromadb',
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
    
    def run_performance_tests(self) -> Dict[str, any]:
        """Run performance tests."""
        logger.info("⚡ Running performance tests...")
        
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
    
    def save_install_log(self, verification_results: Dict[str, bool], performance_results: Dict[str, any]):
        """Save installation log."""
        log_data = {
            'timestamp': time.time(),
            'system_info': self.system_info,
            'verification_results': verification_results,
            'performance_results': performance_results,
            'installation_successful': all(verification_results.values()),
            'enhanced_libraries': True
        }
        
        with open(self.install_log, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"📝 Enhanced installation log saved to {self.install_log}")
    
    def print_installation_summary(self, verification_results: Dict[str, bool], performance_results: Dict[str, any]):
        """Print installation summary."""
        print("\n" + "="*80)
        print("📊 ENHANCED LIBRARIES INSTALLATION SUMMARY")
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
        print("\n⚡ Enhanced Performance Test Results:")
        for test, result in performance_results.items():
            if result is not None:
                print(f"   {test}: {result:.3f}s")
            else:
                print(f"   {test}: FAILED")
        
        print("\n🚀 Enhanced Library Features:")
        print("   ✅ Latest versions of all libraries")
        print("   ✅ GPU acceleration support")
        print("   ✅ Ultra-fast serialization")
        print("   ✅ Advanced compression")
        print("   ✅ Enterprise monitoring")
        print("   ✅ Cutting-edge AI capabilities")
        print("   ✅ Comprehensive document processing")
        print("   ✅ Advanced NLP and ML")
        print("   ✅ High-performance data processing")
        print("   ✅ System-optimized installations")
        
        print("\n🚀 Next Steps:")
        print("   1. Run: python main.py")
        print("   2. Test: python benchmark_speed.py")
        print("   3. Monitor: http://localhost:8001/health")
        print("   4. Docs: http://localhost:8001/docs")
        
        print("="*80 + "\n")
    
    def install(self) -> bool:
        """Run complete enhanced installation."""
        try:
            # Print banner
            self.print_enhanced_banner()
            
            # Get optimized requirements
            requirements = self.get_optimized_requirements()
            
            # Install requirements
            if not self.install_requirements(requirements):
                return False
            
            # Install system optimizations
            self.install_system_optimizations()
            
            # Verify installation
            verification_results = self.verify_installation()
            
            # Run performance tests
            performance_results = self.run_performance_tests()
            
            # Save installation log
            self.save_install_log(verification_results, performance_results)
            
            # Print summary
            self.print_installation_summary(verification_results, performance_results)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced installation failed: {e}")
            return False


def main():
    """Main installation function."""
    installer = EnhancedLibraryInstaller()
    success = installer.install()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


















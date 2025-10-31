#!/usr/bin/env python3
"""
Intelligent Library Installer - Optimized for Performance
========================================================

Automatically installs the best libraries for the system with performance optimizations.
"""

import os
import sys
import subprocess
import platform
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntelligentLibraryInstaller:
    """Intelligent library installer with system optimization"""
    
    def __init__(self):
        self.system_info = self.get_system_info()
        self.install_dir = Path(__file__).parent
        self.requirements_file = self.install_dir / "requirements_optimized.txt"
        self.install_log = self.install_dir / "install_log.json"
        
    def get_system_info(self) -> Dict:
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
                'gpu_available': self.check_gpu_availability(),
                'cuda_available': self.check_cuda_availability(),
                'mkl_available': self.check_mkl_availability(),
                'blas_available': self.check_blas_availability()
            }
        except ImportError:
            logger.warning("psutil not available, using basic system info")
            return {
                'platform': platform.system(),
                'architecture': platform.machine(),
                'python_version': sys.version,
                'cpu_count': os.cpu_count() or 1,
                'memory_gb': 4.0,  # Default assumption
                'gpu_available': False,
                'cuda_available': False
            }
    
    def check_gpu_availability(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def check_cuda_availability(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except ImportError:
            return False
    
    def check_mkl_availability(self) -> bool:
        """Check if Intel MKL is available"""
        try:
            import numpy
            return hasattr(numpy, '__mkl_version__')
        except ImportError:
            return False
    
    def check_blas_availability(self) -> bool:
        """Check BLAS availability"""
        try:
            import numpy
            return numpy.show_config() is not None
        except ImportError:
            return False
    
    def print_system_analysis(self):
        """Print system analysis and recommendations"""
        print("\n" + "="*80)
        print("🔍 SYSTEM ANALYSIS & LIBRARY RECOMMENDATIONS")
        print("="*80)
        
        print(f"Platform: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"Python: {self.system_info['python_version'].split()[0]}")
        print(f"CPU Cores: {self.system_info['cpu_count']}")
        print(f"Memory: {self.system_info['memory_gb']} GB")
        print(f"GPU Available: {'✅' if self.system_info['gpu_available'] else '❌'}")
        print(f"CUDA Available: {'✅' if self.system_info['cuda_available'] else '❌'}")
        print(f"Intel MKL: {'✅' if self.system_info['mkl_available'] else '❌'}")
        
        # Performance recommendations
        print("\n💡 PERFORMANCE RECOMMENDATIONS:")
        
        if self.system_info['memory_gb'] >= 16:
            print("   ✅ High memory system - can use large models and datasets")
        elif self.system_info['memory_gb'] >= 8:
            print("   ⚡ Good memory - balanced configuration recommended")
        else:
            print("   ⚠️ Limited memory - lightweight configuration recommended")
        
        if self.system_info['cpu_count'] >= 16:
            print("   ✅ High CPU count - excellent for parallel processing")
        elif self.system_info['cpu_count'] >= 8:
            print("   ⚡ Good CPU count - good for parallel processing")
        else:
            print("   ⚠️ Limited CPU - consider single-threaded optimizations")
        
        if self.system_info['gpu_available']:
            print("   🚀 GPU available - can use CUDA-accelerated libraries")
        else:
            print("   💻 CPU-only system - will use CPU-optimized libraries")
        
        print("="*80 + "\n")
    
    def get_optimized_requirements(self) -> List[str]:
        """Get optimized requirements based on system capabilities"""
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
        
        # AI and ML libraries
        ai_requirements = [
            "openai>=1.3.0",
            "scikit-learn>=1.3.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "transformers>=4.36.0",
            "sentence-transformers>=2.2.2"
        ]
        requirements.extend(ai_requirements)
        
        # GPU-accelerated libraries
        if self.system_info['gpu_available']:
            gpu_requirements = [
                "torch>=2.1.1",
                "torchvision>=0.16.1",
                "tensorflow>=2.15.0"
            ]
            requirements.extend(gpu_requirements)
        else:
            cpu_requirements = [
                "torch>=2.1.1+cpu",
                "torchvision>=0.16.1+cpu",
                "tensorflow>=2.15.0"
            ]
            requirements.extend(cpu_requirements)
        
        # Document processing
        doc_requirements = [
            "python-docx>=0.8.11",
            "PyPDF2>=3.0.1",
            "pdfplumber>=0.9.0",
            "pymupdf>=1.23.0",
            "markdown>=3.4.0",
            "beautifulsoup4>=4.12.2",
            "lxml>=4.9.3"
        ]
        requirements.extend(doc_requirements)
        
        # NLP libraries
        nlp_requirements = [
            "spacy>=3.7.0",
            "nltk>=3.8.0",
            "textblob>=0.17.1",
            "langdetect>=1.0.9"
        ]
        requirements.extend(nlp_requirements)
        
        # Monitoring and observability
        monitoring_requirements = [
            "prometheus-client>=0.19.0",
            "structlog>=23.2.0",
            "loguru>=0.7.2"
        ]
        requirements.extend(monitoring_requirements)
        
        # Advanced features for high-end systems
        if self.system_info['memory_gb'] >= 8 and self.system_info['cpu_count'] >= 8:
            advanced_requirements = [
                "langchain>=0.0.350",
                "chromadb>=0.4.18",
                "gradio>=4.7.1",
                "streamlit>=1.28.2",
                "plotly>=5.17.0",
                "matplotlib>=3.8.2"
            ]
            requirements.extend(advanced_requirements)
        
        return requirements
    
    def install_with_optimization(self, requirements: List[str]) -> bool:
        """Install requirements with system-specific optimizations"""
        logger.info("📦 Installing optimized libraries...")
        
        # Create temporary requirements file
        temp_req_file = self.install_dir / "temp_requirements.txt"
        
        try:
            with open(temp_req_file, 'w') as f:
                for req in requirements:
                    f.write(f"{req}\n")
            
            # Install with optimizations
            install_cmd = [
                sys.executable, "-m", "pip", "install",
                "--upgrade", "--no-cache-dir",
                "-r", str(temp_req_file)
            ]
            
            # Add system-specific optimizations
            if self.system_info['platform'] == 'Linux':
                install_cmd.extend(["--no-binary", ":all:"])
            
            # Install with progress bar
            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                cwd=self.install_dir
            )
            
            if result.returncode == 0:
                logger.info("✅ Libraries installed successfully")
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
    
    def install_optional_optimizations(self):
        """Install optional performance optimizations"""
        logger.info("🚀 Installing optional performance optimizations...")
        
        optimizations = []
        
        # Intel MKL for better performance
        if not self.system_info['mkl_available'] and self.system_info['platform'] == 'Linux':
            optimizations.append("intel-openmp")
        
        # CUDA toolkit for GPU acceleration
        if self.system_info['gpu_available'] and not self.system_info['cuda_available']:
            optimizations.append("nvidia-ml-py")
        
        # Memory optimization
        optimizations.extend([
            "memory-profiler",
            "pympler",
            "line-profiler"
        ])
        
        # Async optimizations
        if self.system_info['platform'] != 'Windows':
            optimizations.append("uvloop")
        
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
        """Verify that key libraries are properly installed"""
        logger.info("🔍 Verifying installation...")
        
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
            'prometheus_client': 'prometheus_client'
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
    
    def run_performance_test(self) -> Dict[str, float]:
        """Run basic performance tests"""
        logger.info("⚡ Running performance tests...")
        
        results = {}
        
        try:
            import numpy as np
            import time
            
            # NumPy performance test
            start_time = time.time()
            a = np.random.rand(1000, 1000)
            b = np.random.rand(1000, 1000)
            c = np.dot(a, b)
            numpy_time = time.time() - start_time
            results['numpy_matrix_mult'] = numpy_time
            
            logger.info(f"✅ NumPy matrix multiplication: {numpy_time:.3f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ NumPy performance test failed: {e}")
            results['numpy_matrix_mult'] = None
        
        try:
            import pandas as pd
            
            # Pandas performance test
            start_time = time.time()
            df = pd.DataFrame(np.random.rand(10000, 100))
            result = df.groupby(df.index % 10).sum()
            pandas_time = time.time() - start_time
            results['pandas_groupby'] = pandas_time
            
            logger.info(f"✅ Pandas groupby operation: {pandas_time:.3f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ Pandas performance test failed: {e}")
            results['pandas_groupby'] = None
        
        return results
    
    def save_installation_log(self, verification_results: Dict[str, bool], performance_results: Dict[str, float]):
        """Save installation log for future reference"""
        log_data = {
            'timestamp': time.time(),
            'system_info': self.system_info,
            'verification_results': verification_results,
            'performance_results': performance_results,
            'installation_successful': all(verification_results.values())
        }
        
        with open(self.install_log, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"📝 Installation log saved to {self.install_log}")
    
    def print_installation_summary(self, verification_results: Dict[str, bool], performance_results: Dict[str, float]):
        """Print installation summary"""
        print("\n" + "="*80)
        print("📊 INSTALLATION SUMMARY")
        print("="*80)
        
        # Verification results
        total_libs = len(verification_results)
        successful_libs = sum(verification_results.values())
        success_rate = (successful_libs / total_libs) * 100
        
        print(f"Libraries Installed: {successful_libs}/{total_libs} ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("🎉 Installation: EXCELLENT")
        elif success_rate >= 80:
            print("✅ Installation: GOOD")
        elif success_rate >= 70:
            print("⚠️ Installation: FAIR")
        else:
            print("❌ Installation: NEEDS ATTENTION")
        
        # Performance results
        print("\n⚡ Performance Test Results:")
        for test, result in performance_results.items():
            if result is not None:
                print(f"   {test}: {result:.3f}s")
            else:
                print(f"   {test}: FAILED")
        
        print("\n🚀 Next Steps:")
        print("   1. Run: python start_fast.py")
        print("   2. Test: python benchmark_speed.py")
        print("   3. Monitor: http://localhost:8001/health")
        
        print("="*80 + "\n")
    
    def install(self):
        """Run complete intelligent installation"""
        try:
            # Print system analysis
            self.print_system_analysis()
            
            # Get optimized requirements
            requirements = self.get_optimized_requirements()
            logger.info(f"📋 Installing {len(requirements)} optimized libraries...")
            
            # Install libraries
            if not self.install_with_optimization(requirements):
                logger.error("❌ Core installation failed")
                return False
            
            # Install optional optimizations
            self.install_optional_optimizations()
            
            # Verify installation
            verification_results = self.verify_installation()
            
            # Run performance tests
            performance_results = self.run_performance_test()
            
            # Save installation log
            self.save_installation_log(verification_results, performance_results)
            
            # Print summary
            self.print_installation_summary(verification_results, performance_results)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Installation failed: {e}")
            return False

def main():
    """Main installation function"""
    installer = IntelligentLibraryInstaller()
    success = installer.install()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


















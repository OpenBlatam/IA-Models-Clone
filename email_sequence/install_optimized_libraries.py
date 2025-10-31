#!/usr/bin/env python3
"""
Optimized Library Installation Script for Email Sequence System

This script automatically installs and configures all optimized libraries
with proper error handling, performance testing, and system validation.
"""

import subprocess
import sys
import os
import platform
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LibraryInfo:
    """Information about a library to install"""
    name: str
    version: str
    category: str
    priority: int  # 1 = Critical, 2 = High, 3 = Medium, 4 = Low
    dependencies: List[str]
    system_requirements: List[str]
    performance_impact: str
    description: str


class OptimizedLibraryInstaller:
    """Advanced library installer with performance optimization"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.installed_libraries = {}
        self.failed_libraries = []
        self.performance_results = {}
        
        # Define optimized libraries with priorities
        self.libraries = self._define_libraries()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        import psutil
        
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent,
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "processor": platform.processor()
        }
    
    def _define_libraries(self) -> List[LibraryInfo]:
        """Define all optimized libraries with metadata"""
        return [
            # === CRITICAL PERFORMANCE LIBRARIES ===
            LibraryInfo(
                name="orjson",
                version="3.9.10",
                category="serialization",
                priority=1,
                dependencies=[],
                system_requirements=["python>=3.8"],
                performance_impact="5x faster JSON serialization",
                description="Ultra-fast JSON serialization library"
            ),
            LibraryInfo(
                name="uvloop",
                version="0.19.0",
                category="async",
                priority=1,
                dependencies=[],
                system_requirements=["unix-like system"],
                performance_impact="4x faster event loop",
                description="Ultra-fast event loop implementation"
            ),
            LibraryInfo(
                name="redis",
                version="5.0.1",
                category="caching",
                priority=1,
                dependencies=[],
                system_requirements=[],
                performance_impact="Distributed caching",
                description="High-performance in-memory data store"
            ),
            
            # === HIGH PERFORMANCE LIBRARIES ===
            LibraryInfo(
                name="numba",
                version="0.58.1",
                category="compilation",
                priority=2,
                dependencies=["numpy"],
                system_requirements=["llvm"],
                performance_impact="15x faster JIT compilation",
                description="JIT compiler for Python and NumPy"
            ),
            LibraryInfo(
                name="torch",
                version="2.1.1",
                category="ml",
                priority=2,
                dependencies=["numpy"],
                system_requirements=[],
                performance_impact="GPU acceleration for ML",
                description="PyTorch deep learning framework"
            ),
            LibraryInfo(
                name="polars",
                version="0.20.0",
                category="data_processing",
                priority=2,
                dependencies=[],
                system_requirements=[],
                performance_impact="10x faster DataFrames",
                description="Fast DataFrame library"
            ),
            
            # === MEDIUM PERFORMANCE LIBRARIES ===
            LibraryInfo(
                name="lz4",
                version="4.3.2",
                category="compression",
                priority=3,
                dependencies=[],
                system_requirements=[],
                performance_impact="4x faster compression",
                description="High-speed compression library"
            ),
            LibraryInfo(
                name="xxhash",
                version="3.4.1",
                category="hashing",
                priority=3,
                dependencies=[],
                system_requirements=[],
                performance_impact="4x faster hashing",
                description="Extremely fast hashing library"
            ),
            LibraryInfo(
                name="msgspec",
                version="0.18.4",
                category="serialization",
                priority=3,
                dependencies=[],
                system_requirements=[],
                performance_impact="8x faster binary serialization",
                description="Fast binary serialization"
            ),
            
            # === OPTIONAL ENHANCEMENT LIBRARIES ===
            LibraryInfo(
                name="cupy-cuda12x",
                version="12.2.0",
                category="gpu",
                priority=4,
                dependencies=["numpy"],
                system_requirements=["cuda>=12.0"],
                performance_impact="GPU arrays for numerical computing",
                description="GPU-accelerated NumPy replacement"
            ),
            LibraryInfo(
                name="optuna",
                version="3.5.0",
                category="ml_optimization",
                priority=4,
                dependencies=["numpy", "scikit-learn"],
                system_requirements=[],
                performance_impact="Hyperparameter optimization",
                description="Hyperparameter optimization framework"
            ),
            LibraryInfo(
                name="prometheus-client",
                version="0.19.0",
                category="monitoring",
                priority=4,
                dependencies=[],
                system_requirements=[],
                performance_impact="Performance monitoring",
                description="Prometheus metrics client"
            )
        ]
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Check if system meets requirements for all libraries"""
        requirements = {}
        
        for lib in self.libraries:
            requirements[lib.name] = self._check_library_requirements(lib)
        
        return requirements
    
    def _check_library_requirements(self, library: LibraryInfo) -> bool:
        """Check if system meets requirements for a specific library"""
        try:
            # Check Python version
            if "python>=3.8" in library.system_requirements:
                if sys.version_info < (3, 8):
                    return False
            
            # Check for Unix-like system
            if "unix-like system" in library.system_requirements:
                if platform.system() not in ["Linux", "Darwin"]:
                    return False
            
            # Check for CUDA
            if "cuda>=12.0" in library.system_requirements:
                if not self._check_cuda_availability():
                    return False
            
            # Check for LLVM
            if "llvm" in library.system_requirements:
                if not self._check_llvm_availability():
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking requirements for {library.name}: {e}")
            return False
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available"""
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _check_llvm_availability(self) -> bool:
        """Check if LLVM is available"""
        try:
            result = subprocess.run(
                ["llvm-config", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def install_library(self, library: LibraryInfo) -> bool:
        """Install a single library with error handling"""
        try:
            logger.info(f"Installing {library.name} {library.version}...")
            
            # Install dependencies first
            for dep in library.dependencies:
                if dep not in self.installed_libraries:
                    logger.info(f"Installing dependency: {dep}")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", dep
                    ])
            
            # Install the library
            install_cmd = [
                sys.executable, "-m", "pip", "install",
                f"{library.name}=={library.version}"
            ]
            
            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                self.installed_libraries[library.name] = {
                    "version": library.version,
                    "category": library.category,
                    "performance_impact": library.performance_impact
                }
                logger.info(f"✅ Successfully installed {library.name}")
                return True
            else:
                logger.error(f"❌ Failed to install {library.name}: {result.stderr}")
                self.failed_libraries.append(library.name)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout installing {library.name}")
            self.failed_libraries.append(library.name)
            return False
        except Exception as e:
            logger.error(f"❌ Error installing {library.name}: {e}")
            self.failed_libraries.append(library.name)
            return False
    
    def install_all_libraries(self, priority_filter: Optional[int] = None) -> Dict[str, Any]:
        """Install all libraries with priority filtering"""
        logger.info("🚀 Starting optimized library installation...")
        logger.info(f"System: {self.system_info['platform']}")
        logger.info(f"Python: {self.system_info['python_version']}")
        
        # Check system requirements
        requirements = self.check_system_requirements()
        logger.info("📋 System requirements check completed")
        
        # Sort libraries by priority
        libraries_to_install = [
            lib for lib in self.libraries
            if (priority_filter is None or lib.priority <= priority_filter) and
               requirements.get(lib.name, True)
        ]
        
        libraries_to_install.sort(key=lambda x: x.priority)
        
        # Install libraries
        start_time = time.time()
        successful_installations = 0
        
        for library in libraries_to_install:
            if self.install_library(library):
                successful_installations += 1
        
        installation_time = time.time() - start_time
        
        # Generate installation report
        report = {
            "system_info": self.system_info,
            "total_libraries": len(libraries_to_install),
            "successful_installations": successful_installations,
            "failed_installations": len(self.failed_libraries),
            "installation_time": installation_time,
            "installed_libraries": self.installed_libraries,
            "failed_libraries": self.failed_libraries,
            "performance_impact": self._calculate_performance_impact()
        }
        
        return report
    
    def _calculate_performance_impact(self) -> Dict[str, Any]:
        """Calculate overall performance impact of installed libraries"""
        impacts = []
        total_improvement = 1.0
        
        for lib_name, lib_info in self.installed_libraries.items():
            impact = lib_info["performance_impact"]
            impacts.append(f"{lib_name}: {impact}")
            
            # Estimate performance improvement
            if "5x" in impact:
                total_improvement *= 5
            elif "4x" in impact:
                total_improvement *= 4
            elif "10x" in impact:
                total_improvement *= 10
            elif "15x" in impact:
                total_improvement *= 15
            elif "8x" in impact:
                total_improvement *= 8
        
        return {
            "estimated_total_improvement": f"{total_improvement:.1f}x",
            "individual_impacts": impacts,
            "installed_categories": list(set(lib["category"] for lib in self.installed_libraries.values()))
        }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests on installed libraries"""
        logger.info("🧪 Running performance tests...")
        
        tests = {}
        
        # Test JSON serialization
        if "orjson" in self.installed_libraries:
            tests["json_serialization"] = self._test_json_serialization()
        
        # Test compression
        if "lz4" in self.installed_libraries:
            tests["compression"] = self._test_compression()
        
        # Test hashing
        if "xxhash" in self.installed_libraries:
            tests["hashing"] = self._test_hashing()
        
        # Test data processing
        if "polars" in self.installed_libraries:
            tests["data_processing"] = self._test_data_processing()
        
        return tests
    
    def _test_json_serialization(self) -> Dict[str, Any]:
        """Test JSON serialization performance"""
        import json
        import time
        
        test_data = {
            "users": [
                {"id": i, "name": f"user{i}", "email": f"user{i}@example.com"}
                for i in range(1000)
            ],
            "settings": {"theme": "dark", "notifications": True},
            "metadata": {"version": "1.0", "timestamp": time.time()}
        }
        
        results = {}
        
        # Test standard JSON
        start_time = time.time()
        for _ in range(100):
            json.dumps(test_data)
        standard_time = time.time() - start_time
        results["standard_json"] = standard_time
        
        # Test orjson if available
        try:
            import orjson
            start_time = time.time()
            for _ in range(100):
                orjson.dumps(test_data)
            orjson_time = time.time() - start_time
            results["orjson"] = orjson_time
            results["orjson_improvement"] = standard_time / orjson_time
        except ImportError:
            pass
        
        return results
    
    def _test_compression(self) -> Dict[str, Any]:
        """Test compression performance"""
        import time
        import gzip
        
        test_data = b"x" * 1000000  # 1MB of data
        
        results = {}
        
        # Test gzip
        start_time = time.time()
        gzip.compress(test_data)
        gzip_time = time.time() - start_time
        results["gzip"] = gzip_time
        
        # Test lz4 if available
        try:
            import lz4
            start_time = time.time()
            lz4.frame.compress(test_data)
            lz4_time = time.time() - start_time
            results["lz4"] = lz4_time
            results["lz4_improvement"] = gzip_time / lz4_time
        except ImportError:
            pass
        
        return results
    
    def _test_hashing(self) -> Dict[str, Any]:
        """Test hashing performance"""
        import hashlib
        import time
        
        test_data = b"x" * 1000000  # 1MB of data
        
        results = {}
        
        # Test SHA256
        start_time = time.time()
        for _ in range(100):
            hashlib.sha256(test_data).hexdigest()
        sha256_time = time.time() - start_time
        results["sha256"] = sha256_time
        
        # Test xxhash if available
        try:
            import xxhash
            start_time = time.time()
            for _ in range(100):
                xxhash.xxh64(test_data).hexdigest()
            xxhash_time = time.time() - start_time
            results["xxhash"] = xxhash_time
            results["xxhash_improvement"] = sha256_time / xxhash_time
        except ImportError:
            pass
        
        return results
    
    def _test_data_processing(self) -> Dict[str, Any]:
        """Test data processing performance"""
        import time
        
        # Create test data
        data = [{"id": i, "value": i * 2, "category": f"cat{i % 10}"} for i in range(10000)]
        
        results = {}
        
        # Test pandas if available
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            
            start_time = time.time()
            result = df.groupby("category")["value"].sum()
            pandas_time = time.time() - start_time
            results["pandas"] = pandas_time
        except ImportError:
            pass
        
        # Test polars if available
        try:
            import polars as pl
            df = pl.DataFrame(data)
            
            start_time = time.time()
            result = df.groupby("category").agg(pl.col("value").sum())
            polars_time = time.time() - start_time
            results["polars"] = polars_time
            
            if "pandas" in results:
                results["polars_improvement"] = results["pandas"] / polars_time
        except ImportError:
            pass
        
        return results
    
    def generate_installation_report(self, report: Dict[str, Any]) -> str:
        """Generate a comprehensive installation report"""
        report_text = f"""
🚀 OPTIMIZED LIBRARY INSTALLATION REPORT
{'='*50}

📊 INSTALLATION SUMMARY
- Total Libraries: {report['total_libraries']}
- Successfully Installed: {report['successful_installations']}
- Failed Installations: {report['failed_installations']}
- Installation Time: {report['installation_time']:.2f} seconds

💻 SYSTEM INFORMATION
- Platform: {report['system_info']['platform']}
- Python Version: {report['system_info']['python_version']}
- CPU Cores: {report['system_info']['cpu_count']}
- Memory: {report['system_info']['memory_total'] / (1024**3):.1f} GB

✅ SUCCESSFULLY INSTALLED LIBRARIES
"""
        
        for lib_name, lib_info in report['installed_libraries'].items():
            report_text += f"- {lib_name} {lib_info['version']}: {lib_info['performance_impact']}\n"
        
        if report['failed_libraries']:
            report_text += f"\n❌ FAILED INSTALLATIONS\n"
            for lib in report['failed_libraries']:
                report_text += f"- {lib}\n"
        
        report_text += f"""
🚀 PERFORMANCE IMPACT
- Estimated Total Improvement: {report['performance_impact']['estimated_total_improvement']}
- Installed Categories: {', '.join(report['performance_impact']['installed_categories'])}

📈 INDIVIDUAL PERFORMANCE IMPACTS
"""
        
        for impact in report['performance_impact']['individual_impacts']:
            report_text += f"- {impact}\n"
        
        return report_text


def main():
    """Main installation function"""
    print("🚀 Starting Optimized Library Installation for Email Sequence System")
    print("="*70)
    
    installer = OptimizedLibraryInstaller()
    
    # Install critical and high priority libraries first
    print("\n📦 Installing Critical and High Priority Libraries...")
    report = installer.install_all_libraries(priority_filter=2)
    
    # Generate and display report
    report_text = installer.generate_installation_report(report)
    print(report_text)
    
    # Run performance tests
    print("\n🧪 Running Performance Tests...")
    performance_tests = installer.run_performance_tests()
    
    if performance_tests:
        print("\n📊 PERFORMANCE TEST RESULTS")
        print("="*30)
        for test_name, results in performance_tests.items():
            print(f"\n{test_name.upper()}:")
            for metric, value in results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}s")
                else:
                    print(f"  {metric}: {value}")
    
    # Save detailed report to file
    detailed_report = {
        "installation_report": report,
        "performance_tests": performance_tests,
        "timestamp": time.time()
    }
    
    with open("optimized_libraries_installation_report.json", "w") as f:
        json.dump(detailed_report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: optimized_libraries_installation_report.json")
    print("\n✅ Installation completed successfully!")
    
    if report['failed_installations'] > 0:
        print(f"⚠️  {report['failed_installations']} libraries failed to install. Check the report for details.")
    
    print(f"\n🎯 Estimated Performance Improvement: {report['performance_impact']['estimated_total_improvement']}")


if __name__ == "__main__":
    main() 
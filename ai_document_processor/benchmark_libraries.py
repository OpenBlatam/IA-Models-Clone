#!/usr/bin/env python3
"""
Library Benchmark - Performance Testing for Optimized Libraries
==============================================================

Comprehensive benchmark suite for testing library performance and optimizations.
"""

import asyncio
import time
import statistics
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Result of a benchmark test"""
    library: str
    test_name: str
    execution_time: float
    memory_usage: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class LibraryBenchmark:
    """Comprehensive library benchmark suite"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            import psutil
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'python_version': sys.version,
                'platform': sys.platform
            }
        except ImportError:
            return {
                'cpu_count': 1,
                'memory_gb': 4.0,
                'python_version': sys.version,
                'platform': sys.platform
            }
    
    def _measure_memory(self, func, *args, **kwargs):
        """Measure memory usage of a function"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            return result, execution_time, memory_usage
            
        except ImportError:
            # Fallback without memory measurement
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            return result, execution_time, 0.0
    
    def benchmark_numpy(self) -> List[BenchmarkResult]:
        """Benchmark NumPy operations"""
        results = []
        
        try:
            import numpy as np
            
            # Test 1: Matrix multiplication
            def matrix_mult_test():
                a = np.random.rand(1000, 1000)
                b = np.random.rand(1000, 1000)
                return np.dot(a, b)
            
            _, exec_time, memory = self._measure_memory(matrix_mult_test)
            results.append(BenchmarkResult(
                library='numpy',
                test_name='matrix_multiplication_1000x1000',
                execution_time=exec_time,
                memory_usage=memory,
                success=True,
                metadata={'matrix_size': '1000x1000'}
            ))
            
            # Test 2: Array operations
            def array_ops_test():
                arr = np.random.rand(1000000)
                return np.sum(arr), np.mean(arr), np.std(arr)
            
            _, exec_time, memory = self._measure_memory(array_ops_test)
            results.append(BenchmarkResult(
                library='numpy',
                test_name='array_operations_1M_elements',
                execution_time=exec_time,
                memory_usage=memory,
                success=True,
                metadata={'array_size': 1000000}
            ))
            
            # Test 3: FFT
            def fft_test():
                signal = np.random.rand(10000)
                return np.fft.fft(signal)
            
            _, exec_time, memory = self._measure_memory(fft_test)
            results.append(BenchmarkResult(
                library='numpy',
                test_name='fft_10K_elements',
                execution_time=exec_time,
                memory_usage=memory,
                success=True,
                metadata={'signal_length': 10000}
            ))
            
            logger.info("✅ NumPy benchmarks completed")
            
        except ImportError:
            logger.warning("❌ NumPy not available for benchmarking")
            results.append(BenchmarkResult(
                library='numpy',
                test_name='import_test',
                execution_time=0.0,
                memory_usage=0.0,
                success=False,
                error="NumPy not installed"
            ))
        
        return results
    
    def benchmark_pandas(self) -> List[BenchmarkResult]:
        """Benchmark Pandas operations"""
        results = []
        
        try:
            import pandas as pd
            import numpy as np
            
            # Test 1: DataFrame creation and operations
            def dataframe_ops_test():
                df = pd.DataFrame(np.random.rand(100000, 10))
                df['sum'] = df.sum(axis=1)
                df['mean'] = df.mean(axis=1)
                return df.groupby(df.index % 100).sum()
            
            _, exec_time, memory = self._measure_memory(dataframe_ops_test)
            results.append(BenchmarkResult(
                library='pandas',
                test_name='dataframe_operations_100K_rows',
                execution_time=exec_time,
                memory_usage=memory,
                success=True,
                metadata={'rows': 100000, 'columns': 10}
            ))
            
            # Test 2: String operations
            def string_ops_test():
                df = pd.DataFrame({
                    'text': ['hello world'] * 50000,
                    'number': range(50000)
                })
                df['upper'] = df['text'].str.upper()
                df['length'] = df['text'].str.len()
                return df
            
            _, exec_time, memory = self._measure_memory(string_ops_test)
            results.append(BenchmarkResult(
                library='pandas',
                test_name='string_operations_50K_rows',
                execution_time=exec_time,
                memory_usage=memory,
                success=True,
                metadata={'rows': 50000}
            ))
            
            # Test 3: Merge operations
            def merge_test():
                df1 = pd.DataFrame({'key': range(10000), 'value1': np.random.rand(10000)})
                df2 = pd.DataFrame({'key': range(10000), 'value2': np.random.rand(10000)})
                return pd.merge(df1, df2, on='key')
            
            _, exec_time, memory = self._measure_memory(merge_test)
            results.append(BenchmarkResult(
                library='pandas',
                test_name='merge_operations_10K_rows',
                execution_time=exec_time,
                memory_usage=memory,
                success=True,
                metadata={'rows': 10000}
            ))
            
            logger.info("✅ Pandas benchmarks completed")
            
        except ImportError:
            logger.warning("❌ Pandas not available for benchmarking")
            results.append(BenchmarkResult(
                library='pandas',
                test_name='import_test',
                execution_time=0.0,
                memory_usage=0.0,
                success=False,
                error="Pandas not installed"
            ))
        
        return results
    
    def benchmark_torch(self) -> List[BenchmarkResult]:
        """Benchmark PyTorch operations"""
        results = []
        
        try:
            import torch
            
            # Test 1: Tensor operations
            def tensor_ops_test():
                a = torch.randn(1000, 1000)
                b = torch.randn(1000, 1000)
                c = torch.mm(a, b)
                return c.sum()
            
            _, exec_time, memory = self._measure_memory(tensor_ops_test)
            results.append(BenchmarkResult(
                library='torch',
                test_name='tensor_operations_1000x1000',
                execution_time=exec_time,
                memory_usage=memory,
                success=True,
                metadata={'tensor_size': '1000x1000', 'device': 'cpu'}
            ))
            
            # Test 2: CUDA operations (if available)
            if torch.cuda.is_available():
                def cuda_ops_test():
                    a = torch.randn(1000, 1000).cuda()
                    b = torch.randn(1000, 1000).cuda()
                    c = torch.mm(a, b)
                    return c.sum().cpu()
                
                _, exec_time, memory = self._measure_memory(cuda_ops_test)
                results.append(BenchmarkResult(
                    library='torch',
                    test_name='cuda_operations_1000x1000',
                    execution_time=exec_time,
                    memory_usage=memory,
                    success=True,
                    metadata={'tensor_size': '1000x1000', 'device': 'cuda'}
                ))
            
            # Test 3: Neural network forward pass
            def nn_forward_test():
                model = torch.nn.Sequential(
                    torch.nn.Linear(1000, 500),
                    torch.nn.ReLU(),
                    torch.nn.Linear(500, 100),
                    torch.nn.ReLU(),
                    torch.nn.Linear(100, 10)
                )
                x = torch.randn(100, 1000)
                return model(x)
            
            _, exec_time, memory = self._measure_memory(nn_forward_test)
            results.append(BenchmarkResult(
                library='torch',
                test_name='neural_network_forward_pass',
                execution_time=exec_time,
                memory_usage=memory,
                success=True,
                metadata={'batch_size': 100, 'input_size': 1000, 'output_size': 10}
            ))
            
            logger.info("✅ PyTorch benchmarks completed")
            
        except ImportError:
            logger.warning("❌ PyTorch not available for benchmarking")
            results.append(BenchmarkResult(
                library='torch',
                test_name='import_test',
                execution_time=0.0,
                memory_usage=0.0,
                success=False,
                error="PyTorch not installed"
            ))
        
        return results
    
    def benchmark_sklearn(self) -> List[BenchmarkResult]:
        """Benchmark Scikit-learn operations"""
        results = []
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            import numpy as np
            
            # Test 1: Random Forest training
            def rf_training_test():
                X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                return accuracy_score(y_test, y_pred)
            
            _, exec_time, memory = self._measure_memory(rf_training_test)
            results.append(BenchmarkResult(
                library='sklearn',
                test_name='random_forest_training_10K_samples',
                execution_time=exec_time,
                memory_usage=memory,
                success=True,
                metadata={'samples': 10000, 'features': 20, 'estimators': 100}
            ))
            
            # Test 2: Cross-validation
            def cv_test():
                from sklearn.model_selection import cross_val_score
                from sklearn.linear_model import LogisticRegression
                
                X, y = make_classification(n_samples=5000, n_features=10, random_state=42)
                model = LogisticRegression(random_state=42)
                scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
                return scores.mean()
            
            _, exec_time, memory = self._measure_memory(cv_test)
            results.append(BenchmarkResult(
                library='sklearn',
                test_name='cross_validation_5K_samples',
                execution_time=exec_time,
                memory_usage=memory,
                success=True,
                metadata={'samples': 5000, 'features': 10, 'cv_folds': 5}
            ))
            
            logger.info("✅ Scikit-learn benchmarks completed")
            
        except ImportError:
            logger.warning("❌ Scikit-learn not available for benchmarking")
            results.append(BenchmarkResult(
                library='sklearn',
                test_name='import_test',
                execution_time=0.0,
                memory_usage=0.0,
                success=False,
                error="Scikit-learn not installed"
            ))
        
        return results
    
    def benchmark_opencv(self) -> List[BenchmarkResult]:
        """Benchmark OpenCV operations"""
        results = []
        
        try:
            import cv2
            import numpy as np
            
            # Test 1: Image processing
            def image_processing_test():
                img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (15, 15), 0)
                edges = cv2.Canny(blurred, 50, 150)
                return edges
            
            _, exec_time, memory = self._measure_memory(image_processing_test)
            results.append(BenchmarkResult(
                library='opencv',
                test_name='image_processing_1000x1000',
                execution_time=exec_time,
                memory_usage=memory,
                success=True,
                metadata={'image_size': '1000x1000', 'operations': ['color_conversion', 'blur', 'edge_detection']}
            ))
            
            # Test 2: Feature detection
            def feature_detection_test():
                img = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
                detector = cv2.ORB_create()
                keypoints, descriptors = detector.detectAndCompute(img, None)
                return len(keypoints)
            
            _, exec_time, memory = self._measure_memory(feature_detection_test)
            results.append(BenchmarkResult(
                library='opencv',
                test_name='feature_detection_500x500',
                execution_time=exec_time,
                memory_usage=memory,
                success=True,
                metadata={'image_size': '500x500', 'detector': 'ORB'}
            ))
            
            logger.info("✅ OpenCV benchmarks completed")
            
        except ImportError:
            logger.warning("❌ OpenCV not available for benchmarking")
            results.append(BenchmarkResult(
                library='opencv',
                test_name='import_test',
                execution_time=0.0,
                memory_usage=0.0,
                success=False,
                error="OpenCV not installed"
            ))
        
        return results
    
    def benchmark_async_libraries(self) -> List[BenchmarkResult]:
        """Benchmark async libraries"""
        results = []
        
        try:
            import aiohttp
            import asyncio
            
            # Test 1: Async HTTP requests
            async def async_http_test():
                async with aiohttp.ClientSession() as session:
                    tasks = []
                    for i in range(10):
                        task = session.get('https://httpbin.org/delay/0.1')
                        tasks.append(task)
                    
                    responses = await asyncio.gather(*tasks)
                    return [await resp.text() for resp in responses]
            
            async def run_async_test():
                return await async_http_test()
            
            _, exec_time, memory = self._measure_memory(asyncio.run, run_async_test())
            results.append(BenchmarkResult(
                library='aiohttp',
                test_name='async_http_requests_10_concurrent',
                execution_time=exec_time,
                memory_usage=memory,
                success=True,
                metadata={'requests': 10, 'concurrent': True}
            ))
            
            logger.info("✅ Async libraries benchmarks completed")
            
        except ImportError:
            logger.warning("❌ Async libraries not available for benchmarking")
            results.append(BenchmarkResult(
                library='aiohttp',
                test_name='import_test',
                execution_time=0.0,
                memory_usage=0.0,
                success=False,
                error="Async libraries not installed"
            ))
        
        return results
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all available benchmarks"""
        logger.info("🚀 Starting comprehensive library benchmarks...")
        
        all_results = []
        
        # Run individual benchmarks
        benchmarks = [
            self.benchmark_numpy,
            self.benchmark_pandas,
            self.benchmark_torch,
            self.benchmark_sklearn,
            self.benchmark_opencv,
            self.benchmark_async_libraries
        ]
        
        for benchmark_func in benchmarks:
            try:
                results = benchmark_func()
                all_results.extend(results)
            except Exception as e:
                logger.error(f"❌ Benchmark failed: {e}")
        
        self.results = all_results
        logger.info(f"✅ Completed {len(all_results)} benchmark tests")
        
        return all_results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        if not self.results:
            return {}
        
        # Group results by library
        library_results = {}
        for result in self.results:
            if result.library not in library_results:
                library_results[result.library] = []
            library_results[result.library].append(result)
        
        # Calculate statistics
        report = {
            'timestamp': time.time(),
            'system_info': self.system_info,
            'total_tests': len(self.results),
            'successful_tests': sum(1 for r in self.results if r.success),
            'failed_tests': sum(1 for r in self.results if not r.success),
            'libraries_tested': len(library_results),
            'library_results': {}
        }
        
        for library, results in library_results.items():
            successful_results = [r for r in results if r.success]
            
            if successful_results:
                avg_time = statistics.mean(r.execution_time for r in successful_results)
                avg_memory = statistics.mean(r.memory_usage for r in successful_results)
                min_time = min(r.execution_time for r in successful_results)
                max_time = max(r.execution_time for r in successful_results)
            else:
                avg_time = avg_memory = min_time = max_time = 0.0
            
            report['library_results'][library] = {
                'total_tests': len(results),
                'successful_tests': len(successful_results),
                'success_rate': len(successful_results) / len(results) * 100,
                'avg_execution_time': avg_time,
                'avg_memory_usage': avg_memory,
                'min_execution_time': min_time,
                'max_execution_time': max_time,
                'tests': [
                    {
                        'name': r.test_name,
                        'execution_time': r.execution_time,
                        'memory_usage': r.memory_usage,
                        'success': r.success,
                        'error': r.error,
                        'metadata': r.metadata
                    }
                    for r in results
                ]
            }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted benchmark report"""
        print("\n" + "="*80)
        print("📊 LIBRARY BENCHMARK REPORT")
        print("="*80)
        
        print(f"System: {report['system_info']['platform']}")
        print(f"CPU Cores: {report['system_info']['cpu_count']}")
        print(f"Memory: {report['system_info']['memory_gb']} GB")
        print(f"Python: {report['system_info']['python_version'].split()[0]}")
        
        print(f"\n📈 Overall Results:")
        print(f"   Total Tests: {report['total_tests']}")
        print(f"   Successful: {report['successful_tests']}")
        print(f"   Failed: {report['failed_tests']}")
        print(f"   Libraries Tested: {report['libraries_tested']}")
        
        print(f"\n📚 Library Performance:")
        for library, lib_report in report['library_results'].items():
            print(f"\n   🔧 {library.upper()}:")
            print(f"      Success Rate: {lib_report['success_rate']:.1f}%")
            print(f"      Avg Execution Time: {lib_report['avg_execution_time']:.3f}s")
            print(f"      Avg Memory Usage: {lib_report['avg_memory_usage']:.1f} MB")
            
            # Show individual test results
            for test in lib_report['tests']:
                status = "✅" if test['success'] else "❌"
                print(f"      {status} {test['name']}: {test['execution_time']:.3f}s")
                if test['error']:
                    print(f"         Error: {test['error']}")
        
        print("="*80 + "\n")
    
    def save_report(self, report: Dict[str, Any], filename: str = "library_benchmark_report.json"):
        """Save benchmark report to file"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📝 Report saved to {filename}")

def main():
    """Main benchmark function"""
    benchmark = LibraryBenchmark()
    
    try:
        # Run all benchmarks
        results = benchmark.run_all_benchmarks()
        
        # Generate and print report
        report = benchmark.generate_report()
        benchmark.print_report(report)
        
        # Save report
        benchmark.save_report(report)
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")

if __name__ == "__main__":
    main()


















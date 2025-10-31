#!/usr/bin/env python3
"""
Enhanced Libraries Benchmark - Performance Testing
================================================

Comprehensive benchmark suite for enhanced libraries performance testing.
"""

import time
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import statistics
import psutil
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""
    name: str
    duration: float
    memory_usage: float
    cpu_usage: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Benchmark suite configuration."""
    name: str
    iterations: int = 5
    warmup_iterations: int = 2
    timeout_seconds: int = 300
    results: List[BenchmarkResult] = field(default_factory=list)


class EnhancedLibraryBenchmark:
    """Enhanced library benchmark suite."""
    
    def __init__(self):
        self.suites: Dict[str, BenchmarkSuite] = {}
        self.system_info = self._get_system_info()
        self.results_file = Path("enhanced_libraries_benchmark_results.json")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            return {
                'platform': psutil.WINDOWS if hasattr(psutil, 'WINDOWS') else 'unknown',
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'disk_usage': psutil.disk_usage('/').free if hasattr(psutil, 'disk_usage') else 0
            }
        except Exception as e:
            logger.warning(f"Could not get system info: {e}")
            return {}
    
    def create_benchmark_suite(self, name: str, iterations: int = 5, warmup_iterations: int = 2) -> BenchmarkSuite:
        """Create a new benchmark suite."""
        suite = BenchmarkSuite(
            name=name,
            iterations=iterations,
            warmup_iterations=warmup_iterations
        )
        self.suites[name] = suite
        return suite
    
    def run_benchmark(self, suite_name: str, benchmark_func, *args, **kwargs) -> BenchmarkResult:
        """Run a benchmark function."""
        if suite_name not in self.suites:
            raise ValueError(f"Benchmark suite '{suite_name}' not found")
        
        suite = self.suites[suite_name]
        results = []
        
        logger.info(f"Running benchmark: {benchmark_func.__name__}")
        
        # Warmup iterations
        for i in range(suite.warmup_iterations):
            try:
                benchmark_func(*args, **kwargs)
                gc.collect()
            except Exception as e:
                logger.warning(f"Warmup iteration {i} failed: {e}")
        
        # Actual benchmark iterations
        for i in range(suite.iterations):
            try:
                # Measure memory before
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                cpu_before = process.cpu_percent()
                
                # Run benchmark
                start_time = time.time()
                result = benchmark_func(*args, **kwargs)
                end_time = time.time()
                
                # Measure memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                cpu_after = process.cpu_percent()
                
                duration = end_time - start_time
                memory_usage = memory_after - memory_before
                cpu_usage = (cpu_before + cpu_after) / 2
                
                results.append({
                    'duration': duration,
                    'memory_usage': memory_usage,
                    'cpu_usage': cpu_usage,
                    'success': True
                })
                
                logger.info(f"Iteration {i+1}: {duration:.3f}s, {memory_usage:.1f}MB, {cpu_usage:.1f}% CPU")
                
                gc.collect()
                
            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed: {e}")
                results.append({
                    'duration': 0,
                    'memory_usage': 0,
                    'cpu_usage': 0,
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate statistics
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            avg_duration = statistics.mean([r['duration'] for r in successful_results])
            avg_memory = statistics.mean([r['memory_usage'] for r in successful_results])
            avg_cpu = statistics.mean([r['cpu_usage'] for r in successful_results])
            min_duration = min([r['duration'] for r in successful_results])
            max_duration = max([r['duration'] for r in successful_results])
            
            benchmark_result = BenchmarkResult(
                name=benchmark_func.__name__,
                duration=avg_duration,
                memory_usage=avg_memory,
                cpu_usage=avg_cpu,
                success=True,
                metadata={
                    'min_duration': min_duration,
                    'max_duration': max_duration,
                    'iterations': len(successful_results),
                    'total_iterations': suite.iterations
                }
            )
        else:
            benchmark_result = BenchmarkResult(
                name=benchmark_func.__name__,
                duration=0,
                memory_usage=0,
                cpu_usage=0,
                success=False,
                error="All iterations failed"
            )
        
        suite.results.append(benchmark_result)
        return benchmark_result
    
    def benchmark_json_serialization(self):
        """Benchmark JSON serialization libraries."""
        suite = self.create_benchmark_suite("json_serialization", iterations=10)
        
        # Test data
        test_data = {
            'numbers': list(range(10000)),
            'strings': [f"string_{i}" for i in range(1000)],
            'nested': {
                'level1': {
                    'level2': {
                        'level3': list(range(100))
                    }
                }
            }
        }
        
        # Standard json
        def json_serialize():
            import json
            return json.dumps(test_data)
        
        # OrJSON
        def orjson_serialize():
            import orjson
            return orjson.dumps(test_data)
        
        # MsgPack
        def msgpack_serialize():
            import msgpack
            return msgpack.packb(test_data)
        
        # Run benchmarks
        self.run_benchmark("json_serialization", json_serialize)
        self.run_benchmark("json_serialization", orjson_serialize)
        self.run_benchmark("json_serialization", msgpack_serialize)
    
    def benchmark_compression(self):
        """Benchmark compression libraries."""
        suite = self.create_benchmark_suite("compression", iterations=5)
        
        # Test data
        test_data = b"x" * 1000000  # 1MB of data
        
        # LZ4
        def lz4_compress():
            import lz4.frame
            return lz4.frame.compress(test_data)
        
        # Zstandard
        def zstd_compress():
            import zstandard as zstd
            cctx = zstd.ZstdCompressor()
            return cctx.compress(test_data)
        
        # Brotli
        def brotli_compress():
            import brotli
            return brotli.compress(test_data)
        
        # Run benchmarks
        self.run_benchmark("compression", lz4_compress)
        self.run_benchmark("compression", zstd_compress)
        self.run_benchmark("compression", brotli_compress)
    
    def benchmark_numpy_operations(self):
        """Benchmark NumPy operations."""
        suite = self.create_benchmark_suite("numpy_operations", iterations=5)
        
        # Matrix multiplication
        def matrix_mult():
            import numpy as np
            a = np.random.rand(2000, 2000)
            b = np.random.rand(2000, 2000)
            return np.dot(a, b)
        
        # FFT
        def fft_operation():
            import numpy as np
            data = np.random.rand(100000)
            return np.fft.fft(data)
        
        # Array operations
        def array_operations():
            import numpy as np
            arr = np.random.rand(1000000)
            return np.sum(arr), np.mean(arr), np.std(arr)
        
        # Run benchmarks
        self.run_benchmark("numpy_operations", matrix_mult)
        self.run_benchmark("numpy_operations", fft_operation)
        self.run_benchmark("numpy_operations", array_operations)
    
    def benchmark_pandas_operations(self):
        """Benchmark Pandas operations."""
        suite = self.create_benchmark_suite("pandas_operations", iterations=5)
        
        # DataFrame creation and operations
        def dataframe_operations():
            import pandas as pd
            import numpy as np
            
            df = pd.DataFrame({
                'A': np.random.rand(100000),
                'B': np.random.rand(100000),
                'C': np.random.randint(0, 100, 100000)
            })
            
            # Groupby operation
            result = df.groupby('C').agg({'A': 'mean', 'B': 'sum'})
            return result
        
        # Data processing
        def data_processing():
            import pandas as pd
            import numpy as np
            
            df = pd.DataFrame({
                'values': np.random.rand(1000000)
            })
            
            # Complex operations
            df['squared'] = df['values'] ** 2
            df['log'] = np.log(df['values'])
            df['normalized'] = (df['values'] - df['values'].mean()) / df['values'].std()
            
            return df
        
        # Run benchmarks
        self.run_benchmark("pandas_operations", dataframe_operations)
        self.run_benchmark("pandas_operations", data_processing)
    
    def benchmark_ai_libraries(self):
        """Benchmark AI libraries."""
        suite = self.create_benchmark_suite("ai_libraries", iterations=3)
        
        # PyTorch operations
        def torch_operations():
            import torch
            
            # Create tensors
            a = torch.randn(1000, 1000)
            b = torch.randn(1000, 1000)
            
            # Matrix multiplication
            c = torch.mm(a, b)
            
            # GPU operations if available
            if torch.cuda.is_available():
                a_gpu = a.cuda()
                b_gpu = b.cuda()
                c_gpu = torch.mm(a_gpu, b_gpu)
                return c_gpu.cpu()
            
            return c
        
        # Transformers tokenization
        def transformers_tokenization():
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            text = "This is a test sentence for tokenization benchmarking. " * 100
            
            return tokenizer(text, return_tensors="pt")
        
        # Sentence transformers
        def sentence_transformers():
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            sentences = [f"This is sentence number {i}" for i in range(100)]
            
            return model.encode(sentences)
        
        # Run benchmarks
        self.run_benchmark("ai_libraries", torch_operations)
        self.run_benchmark("ai_libraries", transformers_tokenization)
        self.run_benchmark("ai_libraries", sentence_transformers)
    
    def benchmark_document_processing(self):
        """Benchmark document processing libraries."""
        suite = self.create_benchmark_suite("document_processing", iterations=3)
        
        # PDF processing
        def pdf_processing():
            import PyPDF2
            import io
            
            # Create a simple PDF in memory
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            buffer = io.BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            p.drawString(100, 750, "Test PDF Document")
            p.showPage()
            p.save()
            
            buffer.seek(0)
            reader = PyPDF2.PdfReader(buffer)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            return text
        
        # Markdown processing
        def markdown_processing():
            import markdown
            
            md_text = "# Test Document\n\nThis is a **test** document with *markdown* formatting.\n\n" * 100
            
            html = markdown.markdown(md_text, extensions=['extra', 'codehilite'])
            return html
        
        # HTML processing
        def html_processing():
            from bs4 import BeautifulSoup
            
            html_content = "<html><body><h1>Test</h1><p>Content</p></body></html>" * 1000
            
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()
            return text
        
        # Run benchmarks
        self.run_benchmark("document_processing", pdf_processing)
        self.run_benchmark("document_processing", markdown_processing)
        self.run_benchmark("document_processing", html_processing)
    
    def benchmark_async_operations(self):
        """Benchmark async operations."""
        suite = self.create_benchmark_suite("async_operations", iterations=5)
        
        # Async HTTP requests
        async def async_http_requests():
            import aiohttp
            import asyncio
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                for i in range(10):
                    task = session.get('https://httpbin.org/delay/0.1')
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks)
                return [await resp.text() for resp in responses]
        
        # Async file operations
        async def async_file_operations():
            import aiofiles
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                async with aiofiles.open(tmp_path, 'w') as f:
                    for i in range(1000):
                        await f.write(f"Line {i}\n")
                
                async with aiofiles.open(tmp_path, 'r') as f:
                    content = await f.read()
                
                return content
            finally:
                os.unlink(tmp_path)
        
        # Run async benchmarks
        def run_async_http():
            return asyncio.run(async_http_requests())
        
        def run_async_file():
            return asyncio.run(async_file_operations())
        
        self.run_benchmark("async_operations", run_async_http)
        self.run_benchmark("async_operations", run_async_file)
    
    def benchmark_caching(self):
        """Benchmark caching libraries."""
        suite = self.create_benchmark_suite("caching", iterations=10)
        
        # Redis operations
        def redis_operations():
            import redis
            
            r = redis.Redis(host='localhost', port=6379, db=0)
            
            # Set operations
            for i in range(1000):
                r.set(f"key_{i}", f"value_{i}")
            
            # Get operations
            results = []
            for i in range(1000):
                results.append(r.get(f"key_{i}"))
            
            return results
        
        # DiskCache operations
        def diskcache_operations():
            import diskcache
            
            cache = diskcache.Cache('./test_cache')
            
            # Set operations
            for i in range(1000):
                cache[f"key_{i}"] = f"value_{i}"
            
            # Get operations
            results = []
            for i in range(1000):
                results.append(cache[f"key_{i}"])
            
            cache.clear()
            return results
        
        # Run benchmarks
        self.run_benchmark("caching", redis_operations)
        self.run_benchmark("caching", diskcache_operations)
    
    def run_all_benchmarks(self):
        """Run all benchmark suites."""
        logger.info("🚀 Starting Enhanced Libraries Benchmark Suite")
        
        # Run all benchmark suites
        self.benchmark_json_serialization()
        self.benchmark_compression()
        self.benchmark_numpy_operations()
        self.benchmark_pandas_operations()
        self.benchmark_ai_libraries()
        self.benchmark_document_processing()
        self.benchmark_async_operations()
        self.benchmark_caching()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def save_results(self):
        """Save benchmark results to file."""
        results_data = {
            'timestamp': time.time(),
            'system_info': self.system_info,
            'suites': {}
        }
        
        for suite_name, suite in self.suites.items():
            results_data['suites'][suite_name] = {
                'name': suite.name,
                'iterations': suite.iterations,
                'warmup_iterations': suite.warmup_iterations,
                'results': [
                    {
                        'name': result.name,
                        'duration': result.duration,
                        'memory_usage': result.memory_usage,
                        'cpu_usage': result.cpu_usage,
                        'success': result.success,
                        'error': result.error,
                        'metadata': result.metadata
                    }
                    for result in suite.results
                ]
            }
        
        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {self.results_file}")
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("📊 ENHANCED LIBRARIES BENCHMARK SUMMARY")
        print("="*80)
        
        total_benchmarks = 0
        successful_benchmarks = 0
        
        for suite_name, suite in self.suites.items():
            print(f"\n🔧 {suite.name.upper()}")
            print("-" * 40)
            
            for result in suite.results:
                total_benchmarks += 1
                if result.success:
                    successful_benchmarks += 1
                    print(f"✅ {result.name}: {result.duration:.3f}s, {result.memory_usage:.1f}MB, {result.cpu_usage:.1f}% CPU")
                else:
                    print(f"❌ {result.name}: FAILED - {result.error}")
        
        success_rate = (successful_benchmarks / total_benchmarks) * 100 if total_benchmarks > 0 else 0
        
        print(f"\n📈 OVERALL RESULTS")
        print(f"Total Benchmarks: {total_benchmarks}")
        print(f"Successful: {successful_benchmarks}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 95:
            print("🚀 Performance: ULTRA EXCELLENT")
        elif success_rate >= 90:
            print("🎉 Performance: EXCELLENT")
        elif success_rate >= 85:
            print("✅ Performance: VERY GOOD")
        elif success_rate >= 80:
            print("👍 Performance: GOOD")
        else:
            print("⚠️ Performance: NEEDS ATTENTION")
        
        print("="*80 + "\n")


def main():
    """Main benchmark function."""
    benchmark = EnhancedLibraryBenchmark()
    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()


















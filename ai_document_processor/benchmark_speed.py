#!/usr/bin/env python3
"""
Speed Benchmark - Measure Performance Improvements
=================================================

Benchmark script to measure and compare processing speeds.
"""

import asyncio
import time
import statistics
import json
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeedBenchmark:
    """Benchmark tool for measuring processing speed"""
    
    def __init__(self):
        self.results = []
        self.test_files = []
    
    def create_test_files(self) -> List[str]:
        """Create test files of different sizes and types"""
        test_files = []
        
        # Create test directory
        test_dir = Path("test_files")
        test_dir.mkdir(exist_ok=True)
        
        # Small text file (1KB)
        small_file = test_dir / "small.txt"
        with open(small_file, 'w', encoding='utf-8') as f:
            f.write("This is a small test document. " * 50)
        test_files.append(str(small_file))
        
        # Medium text file (10KB)
        medium_file = test_dir / "medium.txt"
        with open(medium_file, 'w', encoding='utf-8') as f:
            f.write("This is a medium test document with more content. " * 500)
        test_files.append(str(medium_file))
        
        # Large text file (100KB)
        large_file = test_dir / "large.txt"
        with open(large_file, 'w', encoding='utf-8') as f:
            f.write("This is a large test document with extensive content. " * 5000)
        test_files.append(str(large_file))
        
        # Markdown file
        md_file = test_dir / "test.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("""# Test Document

## Introduction
This is a test markdown document for benchmarking.

## Content
This document contains various markdown elements:
- Lists
- **Bold text**
- *Italic text*
- `Code blocks`

## Conclusion
This is the end of the test document.
""" * 20)
        test_files.append(str(md_file))
        
        self.test_files = test_files
        return test_files
    
    async def benchmark_old_processor(self, file_path: str) -> Dict[str, Any]:
        """Benchmark the old processor"""
        try:
            # Import old processor
            from services.document_processor import DocumentProcessor
            
            processor = DocumentProcessor()
            await processor.initialize()
            
            start_time = time.time()
            result = await processor.process_document(file_path, Path(file_path).name)
            processing_time = time.time() - start_time
            
            return {
                'processor': 'old',
                'file': Path(file_path).name,
                'file_size': os.path.getsize(file_path),
                'processing_time': processing_time,
                'success': result.success if hasattr(result, 'success') else True,
                'error': None
            }
            
        except Exception as e:
            return {
                'processor': 'old',
                'file': Path(file_path).name,
                'file_size': os.path.getsize(file_path),
                'processing_time': None,
                'success': False,
                'error': str(e)
            }
    
    async def benchmark_fast_processor(self, file_path: str) -> Dict[str, Any]:
        """Benchmark the fast processor"""
        try:
            # Import fast processor
            from services.fast_document_processor import FastDocumentProcessor
            
            processor = FastDocumentProcessor()
            await processor.initialize()
            
            start_time = time.time()
            result = await processor.process_document_fast(file_path, Path(file_path).name)
            processing_time = time.time() - start_time
            
            await processor.close()
            
            return {
                'processor': 'fast',
                'file': Path(file_path).name,
                'file_size': os.path.getsize(file_path),
                'processing_time': processing_time,
                'success': result.success if hasattr(result, 'success') else True,
                'error': None
            }
            
        except Exception as e:
            return {
                'processor': 'fast',
                'file': Path(file_path).name,
                'file_size': os.path.getsize(file_path),
                'processing_time': None,
                'success': False,
                'error': str(e)
            }
    
    async def run_single_benchmark(self, file_path: str, iterations: int = 3) -> Dict[str, Any]:
        """Run benchmark for a single file"""
        logger.info(f"Benchmarking {Path(file_path).name}...")
        
        old_times = []
        fast_times = []
        
        # Benchmark old processor
        for i in range(iterations):
            logger.info(f"  Old processor - iteration {i+1}/{iterations}")
            result = await self.benchmark_old_processor(file_path)
            if result['processing_time'] is not None:
                old_times.append(result['processing_time'])
        
        # Benchmark fast processor
        for i in range(iterations):
            logger.info(f"  Fast processor - iteration {i+1}/{iterations}")
            result = await self.benchmark_fast_processor(file_path)
            if result['processing_time'] is not None:
                fast_times.append(result['processing_time'])
        
        # Calculate statistics
        old_avg = statistics.mean(old_times) if old_times else 0
        fast_avg = statistics.mean(fast_times) if fast_times else 0
        speedup = old_avg / fast_avg if fast_avg > 0 else 0
        
        return {
            'file': Path(file_path).name,
            'file_size': os.path.getsize(file_path),
            'old_processor': {
                'times': old_times,
                'avg_time': old_avg,
                'min_time': min(old_times) if old_times else 0,
                'max_time': max(old_times) if old_times else 0
            },
            'fast_processor': {
                'times': fast_times,
                'avg_time': fast_avg,
                'min_time': min(fast_times) if fast_times else 0,
                'max_time': max(fast_times) if fast_times else 0
            },
            'speedup': speedup,
            'improvement_percent': ((old_avg - fast_avg) / old_avg * 100) if old_avg > 0 else 0
        }
    
    async def run_full_benchmark(self, iterations: int = 3) -> Dict[str, Any]:
        """Run full benchmark suite"""
        logger.info("🚀 Starting Speed Benchmark...")
        
        # Create test files
        test_files = self.create_test_files()
        
        results = []
        total_old_time = 0
        total_fast_time = 0
        
        for file_path in test_files:
            result = await self.run_single_benchmark(file_path, iterations)
            results.append(result)
            
            if result['old_processor']['avg_time'] > 0:
                total_old_time += result['old_processor']['avg_time']
            if result['fast_processor']['avg_time'] > 0:
                total_fast_time += result['fast_processor']['avg_time']
        
        # Calculate overall statistics
        overall_speedup = total_old_time / total_fast_time if total_fast_time > 0 else 0
        overall_improvement = ((total_old_time - total_fast_time) / total_old_time * 100) if total_old_time > 0 else 0
        
        benchmark_result = {
            'timestamp': time.time(),
            'iterations_per_test': iterations,
            'test_files': len(test_files),
            'results': results,
            'overall_statistics': {
                'total_old_time': total_old_time,
                'total_fast_time': total_fast_time,
                'overall_speedup': overall_speedup,
                'overall_improvement_percent': overall_improvement
            }
        }
        
        return benchmark_result
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results in a formatted way"""
        print("\n" + "="*80)
        print("📊 SPEED BENCHMARK RESULTS")
        print("="*80)
        
        print(f"Test Files: {results['test_files']}")
        print(f"Iterations per test: {results['iterations_per_test']}")
        print(f"Timestamp: {time.ctime(results['timestamp'])}")
        
        print("\n📈 Individual File Results:")
        print("-" * 80)
        
        for result in results['results']:
            print(f"\n📄 {result['file']} ({result['file_size']} bytes)")
            print(f"   Old Processor:  {result['old_processor']['avg_time']:.3f}s (avg)")
            print(f"   Fast Processor: {result['fast_processor']['avg_time']:.3f}s (avg)")
            print(f"   Speedup:        {result['speedup']:.2f}x")
            print(f"   Improvement:    {result['improvement_percent']:.1f}%")
        
        print("\n🏆 Overall Results:")
        print("-" * 80)
        overall = results['overall_statistics']
        print(f"Total Old Time:    {overall['total_old_time']:.3f}s")
        print(f"Total Fast Time:   {overall['total_fast_time']:.3f}s")
        print(f"Overall Speedup:   {overall['overall_speedup']:.2f}x")
        print(f"Overall Improvement: {overall['overall_improvement_percent']:.1f}%")
        
        # Performance rating
        if overall['overall_speedup'] >= 3:
            rating = "🚀 EXCELLENT"
        elif overall['overall_speedup'] >= 2:
            rating = "⚡ VERY GOOD"
        elif overall['overall_speedup'] >= 1.5:
            rating = "✅ GOOD"
        elif overall['overall_speedup'] >= 1.2:
            rating = "👍 IMPROVED"
        else:
            rating = "⚠️ MINIMAL IMPROVEMENT"
        
        print(f"\nPerformance Rating: {rating}")
        print("="*80)
    
    def save_results(self, results: Dict[str, Any], filename: str = "benchmark_results.json"):
        """Save benchmark results to file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filename}")
    
    def cleanup_test_files(self):
        """Clean up test files"""
        try:
            import shutil
            if Path("test_files").exists():
                shutil.rmtree("test_files")
            logger.info("Test files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup test files: {e}")

async def main():
    """Main benchmark function"""
    benchmark = SpeedBenchmark()
    
    try:
        # Run benchmark
        results = await benchmark.run_full_benchmark(iterations=3)
        
        # Print results
        benchmark.print_results(results)
        
        # Save results
        benchmark.save_results(results)
        
        # Cleanup
        benchmark.cleanup_test_files()
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
    finally:
        benchmark.cleanup_test_files()

if __name__ == "__main__":
    asyncio.run(main())


















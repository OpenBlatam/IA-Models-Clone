"""
Ultra Fast NLP Benchmark
========================

Benchmark ultra-rápido para probar el rendimiento máximo
del sistema NLP con optimizaciones extremas.
"""

import asyncio
import time
import statistics
import json
import psutil
import torch
from typing import Dict, List, Any
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Import ultra-fast NLP system
from .ultra_fast_nlp import ultra_fast_nlp

logger = logging.getLogger(__name__)

class UltraFastBenchmark:
    """Benchmark ultra-rápido del sistema NLP."""
    
    def __init__(self):
        """Initialize ultra-fast benchmark."""
        self.test_texts = self._generate_test_texts()
        self.system_info = self._get_system_info()
        self.results = {}
    
    def _generate_test_texts(self) -> List[str]:
        """Generate test texts for ultra-fast benchmarking."""
        return [
            # Short texts
            "This is a short test.",
            "I love this product!",
            "This is terrible.",
            "The weather is nice.",
            
            # Medium texts
            "Our company specializes in artificial intelligence solutions for healthcare. We develop machine learning algorithms that help doctors diagnose diseases more accurately.",
            "The implementation of sophisticated machine learning algorithms requires comprehensive understanding of statistical methodologies and computational complexity theory.",
            "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company is headquartered in Cupertino, California.",
            
            # Long texts
            "Artificial intelligence is revolutionizing the way we approach complex problems in various industries. From healthcare to finance, from transportation to entertainment, AI technologies are transforming traditional business models and creating new opportunities for innovation. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions with unprecedented accuracy. Deep learning networks are capable of understanding natural language, recognizing images, and even generating creative content. The integration of AI into everyday applications is becoming seamless, with voice assistants, recommendation systems, and autonomous vehicles becoming increasingly sophisticated.",
            
            # Technical texts
            "The transformer architecture, introduced in the paper 'Attention Is All You Need', has revolutionized natural language processing. The self-attention mechanism allows the model to focus on different parts of the input sequence, enabling better understanding of long-range dependencies. BERT and GPT are two prominent examples of transformer-based models that have achieved state-of-the-art performance on various NLP tasks.",
            
            # Business texts
            "Our quarterly financial results show strong growth across all business segments. Revenue increased by 25% year-over-year, driven by strong demand for our cloud computing services and AI-powered solutions. Customer acquisition costs decreased by 15% while customer lifetime value increased by 30%.",
        ]
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking."""
        return {
            'cpu_count': mp.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
            'platform': psutil.sys.platform
        }
    
    async def run_speed_benchmark(self) -> Dict[str, Any]:
        """Run speed benchmark for ultra-fast performance."""
        print("⚡ Running Ultra-Fast Speed Benchmark...")
        
        results = {
            'test_type': 'speed_benchmark',
            'processing_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'error_count': 0,
            'success_count': 0
        }
        
        try:
            # Initialize system
            start_time = time.time()
            await ultra_fast_nlp.initialize()
            init_time = time.time() - start_time
            results['initialization_time'] = init_time
            
            # Test each text
            for i, text in enumerate(self.test_texts):
                try:
                    start_time = time.time()
                    result = await ultra_fast_nlp.analyze_ultra_fast(
                        text=text,
                        use_cache=True,
                        parallel_processing=True
                    )
                    processing_time = time.time() - start_time
                    
                    results['processing_times'].append(processing_time)
                    results['success_count'] += 1
                    
                    if result.cache_hit:
                        results['cache_hits'] += 1
                    else:
                        results['cache_misses'] += 1
                    
                except Exception as e:
                    results['error_count'] += 1
                    logger.error(f"Speed benchmark error for text {i}: {e}")
            
            # Calculate statistics
            if results['processing_times']:
                results['average_processing_time'] = statistics.mean(results['processing_times'])
                results['min_processing_time'] = min(results['processing_times'])
                results['max_processing_time'] = max(results['processing_times'])
                results['p95_processing_time'] = statistics.quantiles(results['processing_times'], n=20)[18] if len(results['processing_times']) >= 20 else results['average_processing_time']
            
            results['success_rate'] = (results['success_count'] / len(self.test_texts)) * 100
            results['cache_hit_rate'] = (results['cache_hits'] / (results['cache_hits'] + results['cache_misses'])) * 100 if (results['cache_hits'] + results['cache_misses']) > 0 else 0
            
            print(f"✅ Speed benchmark completed: {results['average_processing_time']:.3f}s average")
            
        except Exception as e:
            logger.error(f"Speed benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    async def run_concurrent_benchmark(self) -> Dict[str, Any]:
        """Run concurrent processing benchmark."""
        print("🔄 Running Ultra-Fast Concurrent Benchmark...")
        
        concurrent_levels = [10, 25, 50, 100, 200]
        results = {}
        
        for concurrent in concurrent_levels:
            print(f"🔄 Testing {concurrent} concurrent requests...")
            
            try:
                concurrent_results = await self._run_concurrent_test(concurrent)
                results[f"{concurrent}_concurrent"] = concurrent_results
                
                print(f"✅ {concurrent} concurrent: {concurrent_results['throughput']:.2f} req/s")
                
            except Exception as e:
                logger.error(f"Concurrent benchmark failed for {concurrent}: {e}")
                results[f"{concurrent}_concurrent"] = {'error': str(e)}
        
        return results
    
    async def _run_concurrent_test(self, concurrent: int) -> Dict[str, Any]:
        """Run concurrent processing test."""
        test_text = "This is an ultra-fast concurrent test for maximum performance."
        
        start_time = time.time()
        
        # Create concurrent tasks
        tasks = []
        for i in range(concurrent):
            task = ultra_fast_nlp.analyze_ultra_fast(
                text=f"{test_text} Request #{i+1}",
                use_cache=True,
                parallel_processing=True
            )
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Analyze results
        success_count = len([r for r in results if not isinstance(r, Exception)])
        error_count = len(results) - success_count
        throughput = concurrent / total_time if total_time > 0 else 0
        
        return {
            'concurrent_requests': concurrent,
            'total_time': total_time,
            'throughput_per_second': throughput,
            'success_count': success_count,
            'error_count': error_count,
            'success_rate': (success_count / concurrent) * 100,
            'average_time_per_request': total_time / concurrent
        }
    
    async def run_memory_benchmark(self) -> Dict[str, Any]:
        """Run memory usage benchmark."""
        print("💾 Running Ultra-Fast Memory Benchmark...")
        
        # Get initial memory usage
        initial_memory = psutil.virtual_memory().percent
        
        results = {
            'initial_memory_percent': initial_memory,
            'memory_usage_over_time': [],
            'peak_memory_percent': initial_memory,
            'memory_efficiency': 0
        }
        
        # Test with different text sizes
        text_sizes = [50, 100, 200, 500, 1000]
        
        for size in text_sizes:
            print(f"💾 Testing with {size} character texts...")
            
            # Generate test text
            test_text = "This is an ultra-fast memory test. " * (size // 30)
            
            # Monitor memory during processing
            memory_samples = []
            
            for i in range(10):  # 10 iterations per size
                # Sample memory before
                memory_before = psutil.virtual_memory().percent
                
                try:
                    result = await ultra_fast_nlp.analyze_ultra_fast(
                        text=test_text,
                        use_cache=True,
                        parallel_processing=True
                    )
                    
                    # Sample memory after
                    memory_after = psutil.virtual_memory().percent
                    memory_samples.append(memory_after - memory_before)
                    
                except Exception as e:
                    logger.error(f"Memory benchmark error: {e}")
            
            if memory_samples:
                avg_memory_increase = statistics.mean(memory_samples)
                results['memory_usage_over_time'].append({
                    'text_size': size,
                    'average_memory_increase': avg_memory_increase
                })
                
                # Update peak memory
                current_memory = psutil.virtual_memory().percent
                results['peak_memory_percent'] = max(results['peak_memory_percent'], current_memory)
        
        # Calculate memory efficiency
        if results['memory_usage_over_time']:
            total_memory_increase = sum(sample['average_memory_increase'] for sample in results['memory_usage_over_time'])
            results['memory_efficiency'] = 1 / (total_memory_increase + 1)  # Higher is better
        
        return results
    
    async def run_batch_benchmark(self) -> Dict[str, Any]:
        """Run batch processing benchmark."""
        print("📦 Running Ultra-Fast Batch Benchmark...")
        
        batch_sizes = [10, 25, 50, 100, 200]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"📦 Testing batch size {batch_size}...")
            
            try:
                # Generate test texts
                test_texts = [f"Ultra-fast batch test text #{i+1}" for i in range(batch_size)]
                
                start_time = time.time()
                
                batch_results = await ultra_fast_nlp.batch_analyze_ultra_fast(
                    texts=test_texts,
                    use_cache=True,
                    parallel_processing=True
                )
                
                total_time = time.time() - start_time
                
                # Calculate metrics
                success_count = len([r for r in batch_results if r.processing_time > 0])
                cache_hits = len([r for r in batch_results if r.cache_hit])
                throughput = batch_size / total_time if total_time > 0 else 0
                
                results[f"batch_{batch_size}"] = {
                    'batch_size': batch_size,
                    'total_time': total_time,
                    'throughput_per_second': throughput,
                    'success_count': success_count,
                    'cache_hit_rate': cache_hits / batch_size if batch_size > 0 else 0,
                    'average_time_per_text': total_time / batch_size
                }
                
                print(f"✅ Batch {batch_size}: {throughput:.2f} texts/s")
                
            except Exception as e:
                logger.error(f"Batch benchmark failed for size {batch_size}: {e}")
                results[f"batch_{batch_size}"] = {'error': str(e)}
        
        return results
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive ultra-fast benchmark suite."""
        print("🚀 Starting Comprehensive Ultra-Fast NLP Benchmark...")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Run all benchmark suites
            speed_results = await self.run_speed_benchmark()
            concurrent_results = await self.run_concurrent_benchmark()
            memory_results = await self.run_memory_benchmark()
            batch_results = await self.run_batch_benchmark()
            
            total_time = time.time() - start_time
            
            # Compile comprehensive results
            comprehensive_results = {
                'benchmark_info': {
                    'total_time': total_time,
                    'system_info': self.system_info,
                    'timestamp': datetime.now().isoformat()
                },
                'speed_benchmark': speed_results,
                'concurrent_processing': concurrent_results,
                'memory_usage': memory_results,
                'batch_processing': batch_results,
                'summary': self._generate_summary(
                    speed_results,
                    concurrent_results,
                    memory_results,
                    batch_results
                )
            }
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Comprehensive benchmark failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _generate_summary(self, *results) -> Dict[str, Any]:
        """Generate benchmark summary."""
        summary = {
            'max_throughput': 0,
            'best_concurrent_level': 0,
            'memory_efficiency': 0,
            'best_batch_size': 0,
            'recommendations': []
        }
        
        try:
            # Find max throughput
            if 'concurrent_processing' in results[1]:
                concurrent_results = results[1]['concurrent_processing']
                max_throughput = 0
                best_concurrent = 0
                
                for key, result in concurrent_results.items():
                    if 'error' not in result and 'throughput_per_second' in result:
                        if result['throughput_per_second'] > max_throughput:
                            max_throughput = result['throughput_per_second']
                            best_concurrent = result['concurrent_requests']
                
                summary['max_throughput'] = max_throughput
                summary['best_concurrent_level'] = best_concurrent
            
            # Memory efficiency
            if 'memory_usage' in results[2]:
                memory_results = results[2]['memory_usage']
                summary['memory_efficiency'] = memory_results.get('memory_efficiency', 0)
            
            # Best batch size
            if 'batch_processing' in results[3]:
                batch_results = results[3]['batch_processing']
                best_batch_size = 0
                best_batch_throughput = 0
                
                for key, result in batch_results.items():
                    if 'error' not in result and 'throughput_per_second' in result:
                        if result['throughput_per_second'] > best_batch_throughput:
                            best_batch_throughput = result['throughput_per_second']
                            best_batch_size = result['batch_size']
                
                summary['best_batch_size'] = best_batch_size
            
            # Generate recommendations
            if summary['max_throughput'] > 100:
                summary['recommendations'].append("System shows excellent ultra-fast performance")
            
            if summary['memory_efficiency'] > 0.8:
                summary['recommendations'].append("Memory usage is highly optimized")
            
            if summary['best_concurrent_level'] > 50:
                summary['recommendations'].append("System handles high concurrency well")
            
            if summary['best_batch_size'] > 100:
                summary['recommendations'].append("System is optimized for large batch processing")
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            summary['error'] = str(e)
        
        return summary
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive ultra-fast benchmark report."""
        report = []
        report.append("# Ultra-Fast NLP System Benchmark Report")
        report.append(f"Generated: {results.get('benchmark_info', {}).get('timestamp', 'Unknown')}")
        report.append("")
        
        # System information
        system_info = results.get('benchmark_info', {}).get('system_info', {})
        report.append("## System Information")
        report.append(f"- **CPU Cores**: {system_info.get('cpu_count', 'Unknown')}")
        report.append(f"- **Memory**: {system_info.get('memory_gb', 0):.1f} GB")
        report.append(f"- **GPU Available**: {system_info.get('gpu_available', False)}")
        report.append(f"- **GPU Count**: {system_info.get('gpu_count', 0)}")
        report.append("")
        
        # Speed benchmark results
        if 'speed_benchmark' in results:
            report.append("## Speed Benchmark Results")
            report.append("")
            
            speed_results = results['speed_benchmark']
            report.append(f"- **Average Processing Time**: {speed_results.get('average_processing_time', 0):.3f}s")
            report.append(f"- **Min Processing Time**: {speed_results.get('min_processing_time', 0):.3f}s")
            report.append(f"- **Max Processing Time**: {speed_results.get('max_processing_time', 0):.3f}s")
            report.append(f"- **P95 Processing Time**: {speed_results.get('p95_processing_time', 0):.3f}s")
            report.append(f"- **Success Rate**: {speed_results.get('success_rate', 0):.1f}%")
            report.append(f"- **Cache Hit Rate**: {speed_results.get('cache_hit_rate', 0):.1f}%")
            report.append("")
        
        # Concurrent processing results
        if 'concurrent_processing' in results:
            report.append("## Concurrent Processing Results")
            report.append("")
            
            concurrent_results = results['concurrent_processing']
            for concurrent, data in concurrent_results.items():
                if 'error' not in data:
                    report.append(f"### {concurrent.replace('_', ' ').title()}")
                    report.append(f"- **Throughput**: {data.get('throughput_per_second', 0):.2f} req/s")
                    report.append(f"- **Success Rate**: {data.get('success_rate', 0):.1f}%")
                    report.append(f"- **Average Time per Request**: {data.get('average_time_per_request', 0):.3f}s")
                    report.append("")
        
        # Memory usage results
        if 'memory_usage' in results:
            report.append("## Memory Usage Analysis")
            report.append("")
            
            memory_results = results['memory_usage']
            report.append(f"- **Initial Memory**: {memory_results.get('initial_memory_percent', 0):.1f}%")
            report.append(f"- **Peak Memory**: {memory_results.get('peak_memory_percent', 0):.1f}%")
            report.append(f"- **Memory Efficiency**: {memory_results.get('memory_efficiency', 0):.3f}")
            report.append("")
        
        # Batch processing results
        if 'batch_processing' in results:
            report.append("## Batch Processing Results")
            report.append("")
            
            batch_results = results['batch_processing']
            for batch, data in batch_results.items():
                if 'error' not in data:
                    report.append(f"### {batch.replace('_', ' ').title()}")
                    report.append(f"- **Batch Size**: {data.get('batch_size', 0)}")
                    report.append(f"- **Throughput**: {data.get('throughput_per_second', 0):.2f} texts/s")
                    report.append(f"- **Cache Hit Rate**: {data.get('cache_hit_rate', 0):.1f}%")
                    report.append(f"- **Average Time per Text**: {data.get('average_time_per_text', 0):.3f}s")
                    report.append("")
        
        # Summary and recommendations
        if 'summary' in results:
            summary = results['summary']
            report.append("## Summary and Recommendations")
            report.append("")
            
            report.append(f"- **Maximum Throughput**: {summary.get('max_throughput', 0):.2f} req/s")
            report.append(f"- **Best Concurrent Level**: {summary.get('best_concurrent_level', 0)}")
            report.append(f"- **Memory Efficiency**: {summary.get('memory_efficiency', 0):.3f}")
            report.append(f"- **Best Batch Size**: {summary.get('best_batch_size', 0)}")
            
            report.append("")
            report.append("### Recommendations")
            for recommendation in summary.get('recommendations', []):
                report.append(f"- {recommendation}")
        
        return "\n".join(report)
    
    async def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ultra_fast_benchmark_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📁 Results saved to: {filename}")
        
        # Also save report
        report_filename = filename.replace('.json', '_report.md')
        report = self.generate_report(results)
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Report saved to: {report_filename}")

async def main():
    """Main ultra-fast benchmark function."""
    benchmark = UltraFastBenchmark()
    
    print("🚀 Starting Ultra-Fast NLP System Benchmark Suite")
    print("=" * 60)
    
    try:
        # Run comprehensive benchmark
        results = await benchmark.run_comprehensive_benchmark()
        
        # Generate and display report
        report = benchmark.generate_report(results)
        print("\n" + "=" * 60)
        print(report)
        
        # Save results
        await benchmark.save_results(results)
        
        print("\n🎉 Ultra-fast benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"Ultra-fast benchmark failed: {e}")
        print(f"\n❌ Ultra-fast benchmark failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())













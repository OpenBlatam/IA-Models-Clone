"""
Optimal NLP Benchmark
====================

Benchmark óptimo para probar el rendimiento máximo del sistema NLP
con diferentes niveles de optimización y configuraciones.
"""

import asyncio
import time
import statistics
import json
import psutil
import torch
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Import optimal NLP system
from .optimal_nlp_system import optimal_nlp_system, OptimalConfig, OptimizationLevel, ProcessingMode

logger = logging.getLogger(__name__)

class OptimalBenchmark:
    """Benchmark óptimo del sistema NLP."""
    
    def __init__(self):
        """Initialize optimal benchmark."""
        self.test_texts = self._generate_test_texts()
        self.results = {}
        self.system_info = self._get_system_info()
    
    def _generate_test_texts(self) -> List[str]:
        """Generate comprehensive test texts."""
        return [
            # Short texts
            "This is a simple test.",
            "I love this product!",
            "This is terrible.",
            "The weather is nice today.",
            
            # Medium texts
            "Our company specializes in artificial intelligence solutions for healthcare. We develop machine learning algorithms that help doctors diagnose diseases more accurately and efficiently.",
            "The implementation of sophisticated machine learning algorithms requires comprehensive understanding of statistical methodologies and computational complexity theory, along with practical experience in software engineering and data science.",
            "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company is headquartered in Cupertino, California, United States. Apple's current CEO is Tim Cook, and the company is worth over $3 trillion.",
            
            # Long texts
            "Artificial intelligence is revolutionizing the way we approach complex problems in various industries. From healthcare to finance, from transportation to entertainment, AI technologies are transforming traditional business models and creating new opportunities for innovation. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions with unprecedented accuracy. Deep learning networks are capable of understanding natural language, recognizing images, and even generating creative content. The integration of AI into everyday applications is becoming seamless, with voice assistants, recommendation systems, and autonomous vehicles becoming increasingly sophisticated. However, with these advances come important considerations about ethics, privacy, and the future of work. As AI systems become more powerful, we must ensure they are developed and deployed responsibly, with proper safeguards and regulations in place.",
            
            # Multilingual texts
            "Este es un texto en español para probar el análisis multilingüe del sistema NLP óptimo.",
            "Ceci est un texte en français pour tester l'analyse multilingue du système NLP optimal.",
            "Dies ist ein deutscher Text zur Prüfung der mehrsprachigen Analyse des optimalen NLP-Systems.",
            
            # Technical texts
            "The transformer architecture, introduced in the paper 'Attention Is All You Need', has revolutionized natural language processing. The self-attention mechanism allows the model to focus on different parts of the input sequence, enabling better understanding of long-range dependencies. BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer) are two prominent examples of transformer-based models that have achieved state-of-the-art performance on various NLP tasks.",
            
            # Business texts
            "Our quarterly financial results show strong growth across all business segments. Revenue increased by 25% year-over-year, driven by strong demand for our cloud computing services and AI-powered solutions. Customer acquisition costs decreased by 15% while customer lifetime value increased by 30%. We are well-positioned to capitalize on the growing market for enterprise AI solutions and expect continued growth in the coming quarters.",
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
    
    async def run_optimization_level_benchmark(self) -> Dict[str, Any]:
        """Benchmark different optimization levels."""
        print("🚀 Running Optimization Level Benchmark...")
        
        optimization_levels = [
            OptimizationLevel.MINIMAL,
            OptimizationLevel.BALANCED,
            OptimizationLevel.MAXIMUM,
            OptimizationLevel.ULTRA
        ]
        
        results = {}
        
        for level in optimization_levels:
            print(f"\n📊 Testing {level.value} optimization...")
            
            # Configure system
            optimal_nlp_system.config.optimization_level = level
            
            # Initialize system
            start_time = time.time()
            await optimal_nlp_system.initialize()
            init_time = time.time() - start_time
            
            # Run benchmark
            level_results = await self._run_level_benchmark(level, init_time)
            results[level.value] = level_results
            
            print(f"✅ {level.value} completed: {level_results['average_processing_time']:.3f}s average")
        
        return results
    
    async def _run_level_benchmark(self, level: OptimizationLevel, init_time: float) -> Dict[str, Any]:
        """Run benchmark for specific optimization level."""
        results = {
            'optimization_level': level.value,
            'initialization_time': init_time,
            'processing_times': [],
            'quality_scores': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'error_count': 0,
            'success_count': 0
        }
        
        # Test each text
        for i, text in enumerate(self.test_texts):
            try:
                start_time = time.time()
                result = await optimal_nlp_system.analyze_text_optimal(
                    text=text,
                    use_cache=True,
                    quality_check=True,
                    parallel_processing=True
                )
                processing_time = time.time() - start_time
                
                results['processing_times'].append(processing_time)
                results['success_count'] += 1
                
                if result.get('cache_hit'):
                    results['cache_hits'] += 1
                else:
                    results['cache_misses'] += 1
                
                if 'quality_score' in result:
                    results['quality_scores'].append(result['quality_score'])
                
            except Exception as e:
                results['error_count'] += 1
                logger.error(f"Benchmark error for text {i}: {e}")
        
        # Calculate statistics
        if results['processing_times']:
            results['average_processing_time'] = statistics.mean(results['processing_times'])
            results['min_processing_time'] = min(results['processing_times'])
            results['max_processing_time'] = max(results['processing_times'])
            results['p95_processing_time'] = statistics.quantiles(results['processing_times'], n=20)[18] if len(results['processing_times']) >= 20 else results['average_processing_time']
        
        if results['quality_scores']:
            results['average_quality'] = statistics.mean(results['quality_scores'])
            results['min_quality'] = min(results['quality_scores'])
            results['max_quality'] = max(results['quality_scores'])
        
        results['success_rate'] = (results['success_count'] / len(self.test_texts)) * 100
        results['cache_hit_rate'] = (results['cache_hits'] / (results['cache_hits'] + results['cache_misses'])) * 100 if (results['cache_hits'] + results['cache_misses']) > 0 else 0
        
        return results
    
    async def run_processing_mode_benchmark(self) -> Dict[str, Any]:
        """Benchmark different processing modes."""
        print("🔄 Running Processing Mode Benchmark...")
        
        processing_modes = [
            ProcessingMode.CPU_ONLY,
            ProcessingMode.GPU_ACCELERATED,
            ProcessingMode.HYBRID
        ]
        
        results = {}
        
        for mode in processing_modes:
            if mode == ProcessingMode.GPU_ACCELERATED and not torch.cuda.is_available():
                print(f"⚠️  Skipping {mode.value} - GPU not available")
                continue
            
            print(f"\n🔄 Testing {mode.value} processing...")
            
            # Configure system
            optimal_nlp_system.config.processing_mode = mode
            
            # Run benchmark
            mode_results = await self._run_mode_benchmark(mode)
            results[mode.value] = mode_results
            
            print(f"✅ {mode.value} completed: {mode_results['average_processing_time']:.3f}s average")
        
        return results
    
    async def _run_mode_benchmark(self, mode: ProcessingMode) -> Dict[str, Any]:
        """Run benchmark for specific processing mode."""
        results = {
            'processing_mode': mode.value,
            'processing_times': [],
            'throughput': 0,
            'error_count': 0,
            'success_count': 0
        }
        
        # Test with medium complexity text
        test_text = self.test_texts[4]  # Medium complexity text
        
        # Run multiple iterations
        iterations = 10
        for i in range(iterations):
            try:
                start_time = time.time()
                result = await optimal_nlp_system.analyze_text_optimal(
                    text=test_text,
                    use_cache=False,  # Disable cache for fair comparison
                    quality_check=True,
                    parallel_processing=True
                )
                processing_time = time.time() - start_time
                
                results['processing_times'].append(processing_time)
                results['success_count'] += 1
                
            except Exception as e:
                results['error_count'] += 1
                logger.error(f"Mode benchmark error: {e}")
        
        # Calculate statistics
        if results['processing_times']:
            results['average_processing_time'] = statistics.mean(results['processing_times'])
            results['throughput'] = 1 / results['average_processing_time']  # requests per second
        
        results['success_rate'] = (results['success_count'] / iterations) * 100
        
        return results
    
    async def run_concurrent_benchmark(self) -> Dict[str, Any]:
        """Benchmark concurrent processing performance."""
        print("⚡ Running Concurrent Processing Benchmark...")
        
        concurrent_levels = [1, 5, 10, 20, 50, 100]
        results = {}
        
        for concurrent in concurrent_levels:
            print(f"\n⚡ Testing {concurrent} concurrent requests...")
            
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
        test_text = self.test_texts[4]  # Medium complexity text
        
        start_time = time.time()
        
        # Create concurrent tasks
        tasks = []
        for i in range(concurrent):
            task = optimal_nlp_system.analyze_text_optimal(
                text=f"{test_text} Request #{i+1}",
                use_cache=True,
                quality_check=True,
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
        """Benchmark memory usage and optimization."""
        print("💾 Running Memory Benchmark...")
        
        # Get initial memory usage
        initial_memory = psutil.virtual_memory().percent
        
        results = {
            'initial_memory_percent': initial_memory,
            'memory_usage_over_time': [],
            'peak_memory_percent': initial_memory,
            'memory_efficiency': 0
        }
        
        # Test with different text sizes
        text_sizes = [100, 500, 1000, 2000, 5000]
        
        for size in text_sizes:
            print(f"💾 Testing with {size} character texts...")
            
            # Generate test text
            test_text = "This is a memory test text. " * (size // 30)
            
            # Monitor memory during processing
            memory_samples = []
            
            for i in range(5):  # 5 iterations per size
                # Sample memory before
                memory_before = psutil.virtual_memory().percent
                
                try:
                    result = await optimal_nlp_system.analyze_text_optimal(
                        text=test_text,
                        use_cache=True,
                        quality_check=True,
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
    
    async def run_quality_benchmark(self) -> Dict[str, Any]:
        """Benchmark quality assessment performance."""
        print("🎯 Running Quality Benchmark...")
        
        results = {
            'quality_scores': [],
            'quality_consistency': 0,
            'quality_accuracy': 0,
            'quality_recommendations': []
        }
        
        # Test with different text types
        text_types = [
            ('simple', "This is a simple text."),
            ('complex', "The implementation of sophisticated machine learning algorithms requires comprehensive understanding of statistical methodologies."),
            ('technical', "BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for natural language processing."),
            ('business', "Our quarterly financial results show strong growth across all business segments with revenue increasing by 25% year-over-year."),
            ('multilingual', "Este es un texto en español para probar la calidad del análisis multilingüe.")
        ]
        
        for text_type, text in text_types:
            print(f"🎯 Testing {text_type} text quality...")
            
            try:
                result = await optimal_nlp_system.analyze_text_optimal(
                    text=text,
                    use_cache=True,
                    quality_check=True,
                    parallel_processing=True
                )
                
                if 'quality_score' in result:
                    results['quality_scores'].append({
                        'text_type': text_type,
                        'quality_score': result['quality_score'],
                        'text_length': len(text)
                    })
                
                if 'quality_assessment' in result:
                    assessment = result['quality_assessment']
                    results['quality_recommendations'].extend(assessment.get('recommendations', []))
                
            except Exception as e:
                logger.error(f"Quality benchmark error for {text_type}: {e}")
        
        # Calculate quality statistics
        if results['quality_scores']:
            scores = [q['quality_score'] for q in results['quality_scores']]
            results['average_quality'] = statistics.mean(scores)
            results['min_quality'] = min(scores)
            results['max_quality'] = max(scores)
            results['quality_consistency'] = 1 - statistics.stdev(scores) if len(scores) > 1 else 1
        
        return results
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        print("🚀 Starting Comprehensive Optimal NLP Benchmark...")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Run all benchmark suites
            optimization_results = await self.run_optimization_level_benchmark()
            processing_results = await self.run_processing_mode_benchmark()
            concurrent_results = await self.run_concurrent_benchmark()
            memory_results = await self.run_memory_benchmark()
            quality_results = await self.run_quality_benchmark()
            
            total_time = time.time() - start_time
            
            # Compile comprehensive results
            comprehensive_results = {
                'benchmark_info': {
                    'total_time': total_time,
                    'system_info': self.system_info,
                    'timestamp': datetime.now().isoformat()
                },
                'optimization_levels': optimization_results,
                'processing_modes': processing_results,
                'concurrent_processing': concurrent_results,
                'memory_usage': memory_results,
                'quality_assessment': quality_results,
                'summary': self._generate_summary(
                    optimization_results,
                    processing_results,
                    concurrent_results,
                    memory_results,
                    quality_results
                )
            }
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Comprehensive benchmark failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _generate_summary(self, *results) -> Dict[str, Any]:
        """Generate benchmark summary."""
        summary = {
            'best_optimization_level': None,
            'best_processing_mode': None,
            'max_throughput': 0,
            'best_quality_score': 0,
            'memory_efficiency': 0,
            'recommendations': []
        }
        
        try:
            # Find best optimization level
            if 'optimization_levels' in results[0]:
                opt_results = results[0]['optimization_levels']
                best_opt = max(opt_results.items(), key=lambda x: x[1].get('average_quality', 0))
                summary['best_optimization_level'] = best_opt[0]
            
            # Find best processing mode
            if 'processing_modes' in results[1]:
                proc_results = results[1]['processing_modes']
                best_proc = max(proc_results.items(), key=lambda x: x[1].get('throughput', 0))
                summary['best_processing_mode'] = best_proc[0]
            
            # Find max throughput
            if 'concurrent_processing' in results[2]:
                concurrent_results = results[2]['concurrent_processing']
                max_throughput = max(
                    (result.get('throughput_per_second', 0) for result in concurrent_results.values()),
                    default=0
                )
                summary['max_throughput'] = max_throughput
            
            # Find best quality score
            if 'quality_assessment' in results[4]:
                quality_results = results[4]['quality_assessment']
                if 'average_quality' in quality_results:
                    summary['best_quality_score'] = quality_results['average_quality']
            
            # Memory efficiency
            if 'memory_usage' in results[3]:
                memory_results = results[3]['memory_usage']
                summary['memory_efficiency'] = memory_results.get('memory_efficiency', 0)
            
            # Generate recommendations
            if summary['best_optimization_level']:
                summary['recommendations'].append(f"Use {summary['best_optimization_level']} optimization level for best performance")
            
            if summary['best_processing_mode']:
                summary['recommendations'].append(f"Use {summary['best_processing_mode']} processing mode for maximum throughput")
            
            if summary['max_throughput'] > 10:
                summary['recommendations'].append("System shows excellent throughput performance")
            
            if summary['best_quality_score'] > 0.8:
                summary['recommendations'].append("System maintains high quality standards")
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            summary['error'] = str(e)
        
        return summary
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive benchmark report."""
        report = []
        report.append("# Optimal NLP System Benchmark Report")
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
        
        # Optimization levels comparison
        if 'optimization_levels' in results:
            report.append("## Optimization Levels Performance")
            report.append("")
            
            opt_results = results['optimization_levels']
            for level, data in opt_results.items():
                report.append(f"### {level.title()} Optimization")
                report.append(f"- **Average Processing Time**: {data.get('average_processing_time', 0):.3f}s")
                report.append(f"- **Success Rate**: {data.get('success_rate', 0):.1f}%")
                report.append(f"- **Cache Hit Rate**: {data.get('cache_hit_rate', 0):.1f}%")
                if 'average_quality' in data:
                    report.append(f"- **Average Quality**: {data['average_quality']:.3f}")
                report.append("")
        
        # Processing modes comparison
        if 'processing_modes' in results:
            report.append("## Processing Modes Performance")
            report.append("")
            
            proc_results = results['processing_modes']
            for mode, data in proc_results.items():
                report.append(f"### {mode.replace('_', ' ').title()} Mode")
                report.append(f"- **Average Processing Time**: {data.get('average_processing_time', 0):.3f}s")
                report.append(f"- **Throughput**: {data.get('throughput', 0):.2f} req/s")
                report.append(f"- **Success Rate**: {data.get('success_rate', 0):.1f}%")
                report.append("")
        
        # Concurrent processing results
        if 'concurrent_processing' in results:
            report.append("## Concurrent Processing Performance")
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
        
        # Quality assessment results
        if 'quality_assessment' in results:
            report.append("## Quality Assessment Results")
            report.append("")
            
            quality_results = results['quality_assessment']
            if 'average_quality' in quality_results:
                report.append(f"- **Average Quality Score**: {quality_results['average_quality']:.3f}")
                report.append(f"- **Quality Consistency**: {quality_results.get('quality_consistency', 0):.3f}")
                report.append("")
        
        # Summary and recommendations
        if 'summary' in results:
            summary = results['summary']
            report.append("## Summary and Recommendations")
            report.append("")
            
            if summary.get('best_optimization_level'):
                report.append(f"- **Best Optimization Level**: {summary['best_optimization_level']}")
            
            if summary.get('best_processing_mode'):
                report.append(f"- **Best Processing Mode**: {summary['best_processing_mode']}")
            
            if summary.get('max_throughput'):
                report.append(f"- **Maximum Throughput**: {summary['max_throughput']:.2f} req/s")
            
            if summary.get('best_quality_score'):
                report.append(f"- **Best Quality Score**: {summary['best_quality_score']:.3f}")
            
            if summary.get('memory_efficiency'):
                report.append(f"- **Memory Efficiency**: {summary['memory_efficiency']:.3f}")
            
            report.append("")
            report.append("### Recommendations")
            for recommendation in summary.get('recommendations', []):
                report.append(f"- {recommendation}")
        
        return "\n".join(report)
    
    async def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimal_nlp_benchmark_{timestamp}.json"
        
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
    """Main benchmark function."""
    benchmark = OptimalBenchmark()
    
    print("🚀 Starting Optimal NLP System Benchmark Suite")
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
        
        print("\n🎉 Optimal benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"Optimal benchmark failed: {e}")
        print(f"\n❌ Optimal benchmark failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())













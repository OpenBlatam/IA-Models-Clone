"""
NLP System Benchmark
====================

Script de benchmark para probar y comparar el rendimiento del sistema NLP
básico, avanzado y mejorado.
"""

import asyncio
import time
import statistics
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

# Import NLP systems
from .nlp_system import nlp_system
from .advanced_nlp_system import advanced_nlp_system
from .enhanced_nlp_system import enhanced_nlp_system

logger = logging.getLogger(__name__)

class NLPBenchmark:
    """Benchmark del sistema NLP."""
    
    def __init__(self):
        """Initialize benchmark."""
        self.test_texts = [
            "This is a simple test sentence for basic analysis.",
            "Our company specializes in artificial intelligence solutions for healthcare. We develop machine learning algorithms that help doctors diagnose diseases more accurately.",
            "The implementation of sophisticated machine learning algorithms requires comprehensive understanding of statistical methodologies and computational complexity theory, along with practical experience in software engineering and data science.",
            "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company is headquartered in Cupertino, California, United States. Apple's current CEO is Tim Cook, and the company is worth over $3 trillion.",
            "I love this product! It's amazing and works perfectly. The quality is outstanding and the customer service is excellent. I would definitely recommend it to anyone looking for a reliable solution.",
            "This is terrible. I hate it and want my money back. The product doesn't work as advertised and the support is non-existent. I regret purchasing this item.",
            "The product is okay, nothing special but it works. It meets the basic requirements but doesn't exceed expectations. The price is reasonable for what you get.",
            "Excelente producto, muy recomendado para todos. La calidad es excepcional y el servicio al cliente es excelente. Definitivamente lo recomendaría.",
            "Ce produit est fantastique, je le recommande vivement. La qualité est exceptionnelle et le service client est excellent.",
            "Dieses Produkt ist ausgezeichnet, ich empfehle es sehr. Die Qualität ist außergewöhnlich und der Kundenservice ist exzellent."
        ]
        
        self.results = {
            'basic': {},
            'advanced': {},
            'enhanced': {}
        }
    
    async def run_basic_benchmark(self) -> Dict[str, Any]:
        """Run benchmark for basic NLP system."""
        print("🔍 Running Basic NLP System Benchmark...")
        
        results = {
            'system': 'basic',
            'initialization_time': 0,
            'analysis_times': [],
            'success_rate': 0,
            'error_count': 0,
            'features_tested': []
        }
        
        try:
            # Initialize system
            start_time = time.time()
            await nlp_system.initialize()
            init_time = time.time() - start_time
            results['initialization_time'] = init_time
            
            # Test each text
            for i, text in enumerate(self.test_texts):
                try:
                    start_time = time.time()
                    result = await nlp_system.analyze_text(text)
                    analysis_time = time.time() - start_time
                    
                    results['analysis_times'].append(analysis_time)
                    results['features_tested'].append({
                        'text_id': i,
                        'text_length': len(text),
                        'processing_time': analysis_time,
                        'features': list(result.__dict__.keys()) if hasattr(result, '__dict__') else []
                    })
                    
                except Exception as e:
                    results['error_count'] += 1
                    logger.error(f"Basic analysis error for text {i}: {e}")
            
            # Calculate statistics
            if results['analysis_times']:
                results['average_time'] = statistics.mean(results['analysis_times'])
                results['min_time'] = min(results['analysis_times'])
                results['max_time'] = max(results['analysis_times'])
                results['success_rate'] = (len(results['analysis_times']) / len(self.test_texts)) * 100
            
            print(f"✅ Basic NLP Benchmark completed: {results['average_time']:.3f}s average")
            
        except Exception as e:
            logger.error(f"Basic benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    async def run_advanced_benchmark(self) -> Dict[str, Any]:
        """Run benchmark for advanced NLP system."""
        print("🚀 Running Advanced NLP System Benchmark...")
        
        results = {
            'system': 'advanced',
            'initialization_time': 0,
            'analysis_times': [],
            'success_rate': 0,
            'error_count': 0,
            'features_tested': [],
            'quality_scores': []
        }
        
        try:
            # Initialize system
            start_time = time.time()
            await advanced_nlp_system.initialize()
            init_time = time.time() - start_time
            results['initialization_time'] = init_time
            
            # Test each text
            for i, text in enumerate(self.test_texts):
                try:
                    start_time = time.time()
                    result = await advanced_nlp_system.analyze_text_advanced(text)
                    analysis_time = time.time() - start_time
                    
                    results['analysis_times'].append(analysis_time)
                    
                    # Extract quality metrics
                    quality_score = 0.0
                    if 'readability' in result and 'average_score' in result['readability']:
                        quality_score = result['readability']['average_score'] / 100
                    results['quality_scores'].append(quality_score)
                    
                    results['features_tested'].append({
                        'text_id': i,
                        'text_length': len(text),
                        'processing_time': analysis_time,
                        'quality_score': quality_score,
                        'features': list(result.keys())
                    })
                    
                except Exception as e:
                    results['error_count'] += 1
                    logger.error(f"Advanced analysis error for text {i}: {e}")
            
            # Calculate statistics
            if results['analysis_times']:
                results['average_time'] = statistics.mean(results['analysis_times'])
                results['min_time'] = min(results['analysis_times'])
                results['max_time'] = max(results['analysis_times'])
                results['success_rate'] = (len(results['analysis_times']) / len(self.test_texts)) * 100
                
            if results['quality_scores']:
                results['average_quality'] = statistics.mean(results['quality_scores'])
            
            print(f"✅ Advanced NLP Benchmark completed: {results['average_time']:.3f}s average")
            
        except Exception as e:
            logger.error(f"Advanced benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    async def run_enhanced_benchmark(self) -> Dict[str, Any]:
        """Run benchmark for enhanced NLP system."""
        print("⚡ Running Enhanced NLP System Benchmark...")
        
        results = {
            'system': 'enhanced',
            'initialization_time': 0,
            'analysis_times': [],
            'cache_hit_rate': 0,
            'success_rate': 0,
            'error_count': 0,
            'features_tested': [],
            'quality_scores': [],
            'confidence_scores': []
        }
        
        try:
            # Initialize system
            start_time = time.time()
            await enhanced_nlp_system.initialize()
            init_time = time.time() - start_time
            results['initialization_time'] = init_time
            
            # First pass (no cache)
            print("  First pass (cold cache)...")
            for i, text in enumerate(self.test_texts):
                try:
                    start_time = time.time()
                    result = await enhanced_nlp_system.analyze_text_enhanced(
                        text=text,
                        use_cache=True,
                        quality_check=True,
                        include_trends=False
                    )
                    analysis_time = time.time() - start_time
                    
                    results['analysis_times'].append(analysis_time)
                    results['quality_scores'].append(result.quality_score)
                    results['confidence_scores'].append(result.confidence)
                    
                    results['features_tested'].append({
                        'text_id': i,
                        'text_length': len(text),
                        'processing_time': analysis_time,
                        'quality_score': result.quality_score,
                        'confidence': result.confidence,
                        'cache_hit': result.cache_hit,
                        'features': list(result.analysis.keys())
                    })
                    
                except Exception as e:
                    results['error_count'] += 1
                    logger.error(f"Enhanced analysis error for text {i}: {e}")
            
            # Second pass (with cache)
            print("  Second pass (warm cache)...")
            cache_times = []
            cache_hits = 0
            
            for i, text in enumerate(self.test_texts):
                try:
                    start_time = time.time()
                    result = await enhanced_nlp_system.analyze_text_enhanced(
                        text=text,
                        use_cache=True,
                        quality_check=True,
                        include_trends=False
                    )
                    analysis_time = time.time() - start_time
                    
                    cache_times.append(analysis_time)
                    if result.cache_hit:
                        cache_hits += 1
                    
                except Exception as e:
                    logger.error(f"Enhanced cache analysis error for text {i}: {e}")
            
            # Calculate statistics
            if results['analysis_times']:
                results['average_time'] = statistics.mean(results['analysis_times'])
                results['min_time'] = min(results['analysis_times'])
                results['max_time'] = max(results['analysis_times'])
                results['success_rate'] = (len(results['analysis_times']) / len(self.test_texts)) * 100
                
                # Cache performance
                if cache_times:
                    results['cache_average_time'] = statistics.mean(cache_times)
                    results['cache_improvement'] = (results['average_time'] - results['cache_average_time']) / results['average_time'] * 100
                
                results['cache_hit_rate'] = (cache_hits / len(self.test_texts)) * 100
                
            if results['quality_scores']:
                results['average_quality'] = statistics.mean(results['quality_scores'])
                
            if results['confidence_scores']:
                results['average_confidence'] = statistics.mean(results['confidence_scores'])
            
            print(f"✅ Enhanced NLP Benchmark completed: {results['average_time']:.3f}s average")
            print(f"   Cache hit rate: {results['cache_hit_rate']:.1f}%")
            print(f"   Quality score: {results['average_quality']:.3f}")
            
        except Exception as e:
            logger.error(f"Enhanced benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    async def run_comparative_benchmark(self) -> Dict[str, Any]:
        """Run comparative benchmark across all systems."""
        print("📊 Running Comparative NLP Benchmark...")
        
        # Run all benchmarks
        basic_results = await self.run_basic_benchmark()
        advanced_results = await self.run_advanced_benchmark()
        enhanced_results = await self.run_enhanced_benchmark()
        
        # Compare results
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'systems': {
                'basic': basic_results,
                'advanced': advanced_results,
                'enhanced': enhanced_results
            },
            'comparison': {
                'initialization_times': {
                    'basic': basic_results.get('initialization_time', 0),
                    'advanced': advanced_results.get('initialization_time', 0),
                    'enhanced': enhanced_results.get('initialization_time', 0)
                },
                'processing_times': {
                    'basic': basic_results.get('average_time', 0),
                    'advanced': advanced_results.get('average_time', 0),
                    'enhanced': enhanced_results.get('average_time', 0)
                },
                'success_rates': {
                    'basic': basic_results.get('success_rate', 0),
                    'advanced': advanced_results.get('success_rate', 0),
                    'enhanced': enhanced_results.get('success_rate', 0)
                },
                'quality_scores': {
                    'advanced': advanced_results.get('average_quality', 0),
                    'enhanced': enhanced_results.get('average_quality', 0)
                }
            }
        }
        
        # Calculate improvements
        if basic_results.get('average_time') and enhanced_results.get('average_time'):
            speed_improvement = (basic_results['average_time'] - enhanced_results['average_time']) / basic_results['average_time'] * 100
            comparison['improvements'] = {
                'speed_improvement': speed_improvement,
                'cache_improvement': enhanced_results.get('cache_improvement', 0),
                'quality_improvement': enhanced_results.get('average_quality', 0) - advanced_results.get('average_quality', 0)
            }
        
        return comparison
    
    async def run_stress_test(self, concurrent_requests: int = 10) -> Dict[str, Any]:
        """Run stress test with concurrent requests."""
        print(f"🔥 Running Stress Test ({concurrent_requests} concurrent requests)...")
        
        test_text = "This is a stress test for the enhanced NLP system with multiple concurrent requests."
        
        start_time = time.time()
        
        # Create concurrent tasks
        tasks = []
        for i in range(concurrent_requests):
            task = enhanced_nlp_system.analyze_text_enhanced(
                text=f"{test_text} Request #{i+1}",
                use_cache=True,
                quality_check=True
            )
            tasks.append(task)
        
        # Execute concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            
            # Analyze results
            success_count = len([r for r in results if not isinstance(r, Exception)])
            error_count = len([r for r in results if isinstance(r, Exception)])
            
            # Calculate performance metrics
            throughput = concurrent_requests / total_time
            success_rate = (success_count / concurrent_requests) * 100
            
            stress_results = {
                'concurrent_requests': concurrent_requests,
                'total_time': total_time,
                'throughput': throughput,
                'success_count': success_count,
                'error_count': error_count,
                'success_rate': success_rate,
                'average_time_per_request': total_time / concurrent_requests
            }
            
            print(f"✅ Stress test completed: {throughput:.2f} requests/second")
            print(f"   Success rate: {success_rate:.1f}%")
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            return {'error': str(e)}
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive benchmark report."""
        report = []
        report.append("# NLP System Benchmark Report")
        report.append(f"Generated: {results.get('timestamp', 'Unknown')}")
        report.append("")
        
        # System comparison
        if 'comparison' in results:
            comparison = results['comparison']
            
            report.append("## Performance Comparison")
            report.append("")
            
            # Processing times
            report.append("### Processing Times (seconds)")
            for system, time in comparison['processing_times'].items():
                report.append(f"- **{system.title()}**: {time:.3f}s")
            report.append("")
            
            # Success rates
            report.append("### Success Rates (%)")
            for system, rate in comparison['success_rates'].items():
                report.append(f"- **{system.title()}**: {rate:.1f}%")
            report.append("")
            
            # Quality scores
            if 'quality_scores' in comparison:
                report.append("### Quality Scores")
                for system, score in comparison['quality_scores'].items():
                    if score > 0:
                        report.append(f"- **{system.title()}**: {score:.3f}")
                report.append("")
            
            # Improvements
            if 'improvements' in results:
                improvements = results['improvements']
                report.append("### Improvements")
                if 'speed_improvement' in improvements:
                    report.append(f"- **Speed Improvement**: {improvements['speed_improvement']:.1f}%")
                if 'cache_improvement' in improvements:
                    report.append(f"- **Cache Improvement**: {improvements['cache_improvement']:.1f}%")
                if 'quality_improvement' in improvements:
                    report.append(f"- **Quality Improvement**: {improvements['quality_improvement']:.3f}")
                report.append("")
        
        # System details
        report.append("## System Details")
        report.append("")
        
        for system_name, system_results in results.get('systems', {}).items():
            report.append(f"### {system_name.title()} System")
            report.append(f"- **Initialization Time**: {system_results.get('initialization_time', 0):.3f}s")
            report.append(f"- **Average Processing Time**: {system_results.get('average_time', 0):.3f}s")
            report.append(f"- **Success Rate**: {system_results.get('success_rate', 0):.1f}%")
            
            if 'average_quality' in system_results:
                report.append(f"- **Average Quality Score**: {system_results['average_quality']:.3f}")
            
            if 'cache_hit_rate' in system_results:
                report.append(f"- **Cache Hit Rate**: {system_results['cache_hit_rate']:.1f}%")
            
            if 'error_count' in system_results:
                report.append(f"- **Error Count**: {system_results['error_count']}")
            
            report.append("")
        
        return "\n".join(report)
    
    async def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nlp_benchmark_{timestamp}.json"
        
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
    benchmark = NLPBenchmark()
    
    print("🚀 Starting NLP System Benchmark Suite")
    print("=" * 50)
    
    try:
        # Run comparative benchmark
        results = await benchmark.run_comparative_benchmark()
        
        # Run stress test
        stress_results = await benchmark.run_stress_test(concurrent_requests=20)
        results['stress_test'] = stress_results
        
        # Generate and display report
        report = benchmark.generate_report(results)
        print("\n" + "=" * 50)
        print(report)
        
        # Save results
        await benchmark.save_results(results)
        
        print("\n🎉 Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        print(f"\n❌ Benchmark failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())













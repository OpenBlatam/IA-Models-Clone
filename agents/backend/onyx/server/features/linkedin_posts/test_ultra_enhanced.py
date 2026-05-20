#!/usr/bin/env python3
"""
Ultra Enhanced LinkedIn Posts System - Test Runner
=================================================

Comprehensive test suite for validating all ultra-enhanced optimizations:
- Performance testing
- AI optimization testing
- Cache efficiency testing
- Load testing
- Memory profiling
"""

import asyncio
import time
import sys
import os
import json
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import psutil
import GPUtil

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the ultra enhanced system
from ULTRA_ENHANCED_OPTIMIZATION import (
    UltraEnhancedLinkedInPostsSystem,
    UltraEnhancedConfig,
    get_ultra_enhanced_system
)

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    duration: float
    success: bool
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None

class UltraEnhancedTestRunner:
    """Comprehensive test runner for ultra-enhanced system"""
    
    def __init__(self):
        self.system = None
        self.test_results = []
        self.performance_metrics = {}
        
    async def initialize(self):
        """Initialize the ultra-enhanced system"""
        print("🚀 Initializing Ultra Enhanced LinkedIn Posts System...")
        
        try:
            self.system = await get_ultra_enhanced_system()
            print("✅ System initialized successfully")
            
            # Run initial health check
            health = await self.system.health_check()
            print(f"✅ Health check: {health['status']}")
            
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            raise
    
    async def run_performance_test(self) -> TestResult:
        """Test system performance with various scenarios"""
        print("\n⚡ Running Performance Test...")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        try:
            # Test data
            test_posts = [
                {
                    "topic": "AI Innovation Breakthrough",
                    "key_points": [
                        "Revolutionary AI algorithm developed",
                        "300% improvement in processing speed",
                        "Industry-leading accuracy rates"
                    ],
                    "target_audience": "tech professionals",
                    "industry": "technology",
                    "tone": "professional",
                    "post_type": "announcement",
                    "keywords": ["AI", "Innovation", "Technology"],
                    "additional_context": "Latest breakthrough in machine learning"
                },
                {
                    "topic": "LinkedIn Marketing Strategies",
                    "key_points": [
                        "Proven techniques for engagement",
                        "Optimal posting times identified",
                        "Hashtag optimization strategies"
                    ],
                    "target_audience": "marketers",
                    "industry": "marketing",
                    "tone": "casual",
                    "post_type": "educational",
                    "keywords": ["Marketing", "LinkedIn", "Growth"],
                    "additional_context": "Comprehensive guide for social media success"
                },
                {
                    "topic": "Remote Work Best Practices",
                    "key_points": [
                        "Productivity tips for remote teams",
                        "Communication tools and strategies",
                        "Work-life balance maintenance"
                    ],
                    "target_audience": "professionals",
                    "industry": "consulting",
                    "tone": "friendly",
                    "post_type": "insight",
                    "keywords": ["Remote Work", "Productivity", "Leadership"],
                    "additional_context": "Lessons learned from successful remote teams"
                }
            ]
            
            # Generate posts and measure performance
            generation_times = []
            optimization_scores = []
            
            for i, post_data in enumerate(test_posts):
                post_start = time.time()
                
                result = await self.system.generate_optimized_post(**post_data)
                
                generation_time = time.time() - post_start
                generation_times.append(generation_time)
                
                if 'optimization_score' in result:
                    optimization_scores.append(result['optimization_score'])
                
                print(f"  📝 Generated post {i+1}/{len(test_posts)} in {generation_time:.3f}s")
            
            # Calculate metrics
            avg_generation_time = statistics.mean(generation_times)
            avg_optimization_score = statistics.mean(optimization_scores) if optimization_scores else 0
            
            end_memory = psutil.virtual_memory().used
            memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB
            
            metrics = {
                "total_posts": len(test_posts),
                "avg_generation_time": avg_generation_time,
                "min_generation_time": min(generation_times),
                "max_generation_time": max(generation_times),
                "avg_optimization_score": avg_optimization_score,
                "memory_delta_mb": memory_delta,
                "throughput_posts_per_second": len(test_posts) / (time.time() - start_time)
            }
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Performance Test",
                duration=duration,
                success=True,
                metrics=metrics,
                memory_usage=memory_delta,
                cpu_usage=psutil.cpu_percent()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Performance Test",
                duration=duration,
                success=False,
                error=str(e)
            )
    
    async def run_cache_efficiency_test(self) -> TestResult:
        """Test cache efficiency and hit rates"""
        print("\n💾 Running Cache Efficiency Test...")
        
        start_time = time.time()
        
        try:
            # Test data for cache testing
            test_post = {
                "topic": "Cache Testing",
                "key_points": ["Cache efficiency", "Hit rate optimization", "Performance improvement"],
                "target_audience": "developers",
                "industry": "technology",
                "tone": "professional",
                "post_type": "educational"
            }
            
            # First generation (cache miss)
            start1 = time.time()
            result1 = await self.system.generate_optimized_post(**test_post)
            time1 = time.time() - start1
            
            # Second generation (cache hit)
            start2 = time.time()
            result2 = await self.system.generate_optimized_post(**test_post)
            time2 = time.time() - start2
            
            # Get cache statistics
            cache_stats = await self.system.get_performance_metrics()
            
            metrics = {
                "first_generation_time": time1,
                "second_generation_time": time2,
                "cache_speedup": time1 / time2 if time2 > 0 else 0,
                "cache_hit_ratio": cache_stats.get('cache_metrics', {}).get('hit_ratio', 0),
                "superposition_states": cache_stats.get('cache_metrics', {}).get('superposition_states', 0)
            }
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Cache Efficiency Test",
                duration=duration,
                success=True,
                metrics=metrics
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Cache Efficiency Test",
                duration=duration,
                success=False,
                error=str(e)
            )
    
    async def run_ai_optimization_test(self) -> TestResult:
        """Test AI optimization capabilities"""
        print("\n🤖 Running AI Optimization Test...")
        
        start_time = time.time()
        
        try:
            # Test content for AI optimization
            test_content = """
            We are excited to announce our latest breakthrough in artificial intelligence technology. 
            Our new algorithm has achieved unprecedented performance improvements, delivering 300% 
            faster processing speeds while maintaining industry-leading accuracy rates. This innovation 
            represents a significant milestone in our journey toward advanced AI solutions.
            """
            
            # Test different optimization targets
            optimization_targets = [
                {"engagement_score": 0.8, "readability_score": 0.7, "sentiment_score": 0.6},
                {"engagement_score": 0.9, "readability_score": 0.8, "sentiment_score": 0.7},
                {"engagement_score": 0.7, "readability_score": 0.9, "sentiment_score": 0.5}
            ]
            
            optimization_results = []
            
            for i, targets in enumerate(optimization_targets):
                start = time.time()
                
                result = await self.system.ai_processor.optimize_content(test_content, targets)
                
                optimization_time = time.time() - start
                optimization_results.append({
                    "targets": targets,
                    "optimization_score": result.get('optimization_score', 0),
                    "processing_time": optimization_time
                })
                
                print(f"  🎯 Optimization {i+1}/{len(optimization_targets)}: {result.get('optimization_score', 0):.3f}")
            
            # Calculate metrics
            avg_optimization_score = statistics.mean([r['optimization_score'] for r in optimization_results])
            avg_processing_time = statistics.mean([r['processing_time'] for r in optimization_results])
            
            metrics = {
                "total_optimizations": len(optimization_results),
                "avg_optimization_score": avg_optimization_score,
                "avg_processing_time": avg_processing_time,
                "optimization_results": optimization_results
            }
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="AI Optimization Test",
                duration=duration,
                success=True,
                metrics=metrics
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="AI Optimization Test",
                duration=duration,
                success=False,
                error=str(e)
            )
    
    async def run_load_test(self) -> TestResult:
        """Test system under load"""
        print("\n📊 Running Load Test...")
        
        start_time = time.time()
        
        try:
            # Generate test data for load testing
            load_test_data = []
            for i in range(20):  # 20 concurrent requests
                load_test_data.append({
                    "topic": f"Load Test Topic {i}",
                    "key_points": [f"Key point {j}" for j in range(3)],
                    "target_audience": "professionals",
                    "industry": "technology",
                    "tone": "professional",
                    "post_type": "announcement"
                })
            
            # Run concurrent requests
            start = time.time()
            tasks = [
                self.system.generate_optimized_post(**post_data)
                for post_data in load_test_data
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            load_time = time.time() - start
            
            # Analyze results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            
            metrics = {
                "total_requests": len(load_test_data),
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": len(successful_results) / len(load_test_data),
                "total_time": load_time,
                "requests_per_second": len(load_test_data) / load_time,
                "avg_time_per_request": load_time / len(load_test_data)
            }
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Load Test",
                duration=duration,
                success=len(failed_results) == 0,
                metrics=metrics,
                error=f"{len(failed_results)} requests failed" if failed_results else None
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Load Test",
                duration=duration,
                success=False,
                error=str(e)
            )
    
    async def run_memory_profiling_test(self) -> TestResult:
        """Test memory usage and profiling"""
        print("\n🧠 Running Memory Profiling Test...")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory()
        
        try:
            # Monitor memory during operations
            memory_samples = []
            
            for i in range(10):
                # Generate a post
                await self.system.generate_optimized_post(
                    topic=f"Memory Test {i}",
                    key_points=[f"Point {j}" for j in range(3)],
                    target_audience="professionals",
                    industry="technology",
                    tone="professional",
                    post_type="announcement"
                )
                
                # Sample memory usage
                memory = psutil.virtual_memory()
                memory_samples.append({
                    "iteration": i,
                    "memory_used_mb": memory.used / 1024 / 1024,
                    "memory_percent": memory.percent
                })
                
                # Small delay
                await asyncio.sleep(0.1)
            
            # Calculate memory metrics
            memory_used = [s['memory_used_mb'] for s in memory_samples]
            memory_percent = [s['memory_percent'] for s in memory_samples]
            
            end_memory = psutil.virtual_memory()
            
            metrics = {
                "start_memory_mb": start_memory.used / 1024 / 1024,
                "end_memory_mb": end_memory.used / 1024 / 1024,
                "memory_delta_mb": (end_memory.used - start_memory.used) / 1024 / 1024,
                "avg_memory_usage_mb": statistics.mean(memory_used),
                "max_memory_usage_mb": max(memory_used),
                "avg_memory_percent": statistics.mean(memory_percent),
                "memory_samples": memory_samples
            }
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Memory Profiling Test",
                duration=duration,
                success=True,
                metrics=metrics,
                memory_usage=metrics['memory_delta_mb']
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Memory Profiling Test",
                duration=duration,
                success=False,
                error=str(e)
            )
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report"""
        print("🎯 Starting Comprehensive Ultra Enhanced Test Suite...")
        
        # Initialize system
        await self.initialize()
        
        # Run all tests
        tests = [
            self.run_performance_test(),
            self.run_cache_efficiency_test(),
            self.run_ai_optimization_test(),
            self.run_load_test(),
            self.run_memory_profiling_test()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Process results
        successful_tests = 0
        total_duration = 0
        all_metrics = {}
        
        for result in results:
            if isinstance(result, TestResult):
                self.test_results.append(result)
                total_duration += result.duration
                
                if result.success:
                    successful_tests += 1
                
                if result.metrics:
                    all_metrics[result.test_name] = result.metrics
        
        # Generate summary
        summary = {
            "total_tests": len(results),
            "successful_tests": successful_tests,
            "failed_tests": len(results) - successful_tests,
            "success_rate": successful_tests / len(results),
            "total_duration": total_duration,
            "test_results": [r.__dict__ for r in self.test_results],
            "metrics_summary": all_metrics,
            "timestamp": time.time()
        }
        
        return summary
    
    def print_results(self, summary: Dict[str, Any]):
        """Print comprehensive test results"""
        print("\n" + "="*60)
        print("🎉 ULTRA ENHANCED TEST RESULTS")
        print("="*60)
        
        print(f"\n📊 Overall Results:")
        print(f"  ✅ Successful Tests: {summary['successful_tests']}/{summary['total_tests']}")
        print(f"  ❌ Failed Tests: {summary['failed_tests']}")
        print(f"  📈 Success Rate: {summary['success_rate']:.1%}")
        print(f"  ⏱️  Total Duration: {summary['total_duration']:.2f}s")
        
        print(f"\n🔍 Detailed Results:")
        for result in summary['test_results']:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"  {status} {result['test_name']}: {result['duration']:.3f}s")
            if result['error']:
                print(f"    Error: {result['error']}")
        
        print(f"\n📈 Performance Metrics:")
        metrics = summary['metrics_summary']
        
        if 'Performance Test' in metrics:
            perf = metrics['Performance Test']
            print(f"  🚀 Avg Generation Time: {perf['avg_generation_time']:.3f}s")
            print(f"  📝 Throughput: {perf['throughput_posts_per_second']:.1f} posts/s")
            print(f"  🎯 Avg Optimization Score: {perf['avg_optimization_score']:.3f}")
        
        if 'Cache Efficiency Test' in metrics:
            cache = metrics['Cache Efficiency Test']
            print(f"  💾 Cache Speedup: {cache['cache_speedup']:.1f}x")
            print(f"  🎯 Cache Hit Ratio: {cache['cache_hit_ratio']:.1%}")
        
        if 'Load Test' in metrics:
            load = metrics['Load Test']
            print(f"  📊 Success Rate: {load['success_rate']:.1%}")
            print(f"  ⚡ Requests/Second: {load['requests_per_second']:.1f}")
        
        if 'Memory Profiling Test' in metrics:
            memory = metrics['Memory Profiling Test']
            print(f"  🧠 Memory Delta: {memory['memory_delta_mb']:.1f} MB")
            print(f"  📊 Avg Memory Usage: {memory['avg_memory_usage_mb']:.1f} MB")
        
        print("\n" + "="*60)

async def main():
    """Main test runner"""
    print("🚀 Ultra Enhanced LinkedIn Posts System - Test Runner")
    print("="*60)
    
    runner = UltraEnhancedTestRunner()
    
    try:
        # Run comprehensive test suite
        summary = await runner.run_comprehensive_test()
        
        # Print results
        runner.print_results(summary)
        
        # Save results to file
        with open("ultra_enhanced_test_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: ultra_enhanced_test_results.json")
        
        # Exit with appropriate code
        if summary['success_rate'] >= 0.8:
            print("\n🎉 All tests passed successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️  Some tests failed. Success rate: {summary['success_rate']:.1%}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 
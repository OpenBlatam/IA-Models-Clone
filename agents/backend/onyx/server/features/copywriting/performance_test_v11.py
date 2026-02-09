#!/usr/bin/env python3
"""
Performance Testing Script v11
==============================

Comprehensive performance testing for the ultra-optimized copywriting system v11.
Tests various scenarios and measures performance improvements.
"""

import asyncio
import time
import json
import statistics
import psutil
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import aiohttp
import httpx
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Import the ultra-optimized engine
from ultra_optimized_engine_v11 import (
    UltraOptimizedEngineV11,
    PerformanceConfig,
    ModelConfig,
    CacheConfig,
    MonitoringConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """Test configuration settings"""
    num_requests: int = 100
    concurrent_requests: int = 10
    test_duration: int = 300  # 5 minutes
    warmup_requests: int = 10
    cache_test_requests: int = 50
    batch_test_requests: int = 20
    memory_test_requests: int = 100
    gpu_test_requests: int = 50
    stress_test_requests: int = 1000
    api_url: str = "http://localhost:8000"
    api_key: str = "your-secret-api-key"

@dataclass
class TestResult:
    """Test result data"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    memory_usage: Dict[str, float]
    cpu_usage: float
    gpu_usage: Optional[float] = None
    cache_hit_ratio: Optional[float] = None
    error_rate: float = 0.0

class PerformanceTester:
    """Comprehensive performance testing framework"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results = []
        self.engine = None
        self.session = None
        
    async def initialize(self):
        """Initialize testing environment"""
        logger.info("Initializing performance testing environment")
        
        # Initialize engine
        performance_config = PerformanceConfig(
            enable_gpu=True,
            enable_caching=True,
            enable_monitoring=True,
            max_workers=16,
            batch_size=64
        )
        
        model_config = ModelConfig(
            model_name="gpt2",
            max_length=512,
            temperature=0.7,
            num_return_sequences=3
        )
        
        cache_config = CacheConfig(
            redis_url="redis://localhost:6379",
            cache_ttl=3600
        )
        
        monitoring_config = MonitoringConfig(
            enable_prometheus=True,
            enable_monitoring=True
        )
        
        self.engine = UltraOptimizedEngineV11(
            performance_config=performance_config,
            model_config=model_config,
            cache_config=cache_config,
            monitoring_config=monitoring_config
        )
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession()
        
        logger.info("Performance testing environment initialized")
    
    async def cleanup(self):
        """Cleanup testing environment"""
        if self.engine:
            await self.engine.cleanup()
        
        if self.session:
            await self.session.close()
        
        logger.info("Performance testing environment cleaned up")
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        metrics = {
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'cpu_percent': cpu
        }
        
        # Try to get GPU metrics if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                metrics['gpu_memory_gb'] = gpu_memory
        except:
            pass
        
        return metrics
    
    async def make_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single request to the engine"""
        start_time = time.perf_counter()
        
        try:
            result = await self.engine.generate_copywriting(request_data)
            
            end_time = time.perf_counter()
            response_time = end_time - start_time
            
            return {
                'success': True,
                'response_time': response_time,
                'result': result
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            response_time = end_time - start_time
            
            return {
                'success': False,
                'response_time': response_time,
                'error': str(e)
            }
    
    async def make_api_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single request to the API"""
        start_time = time.perf_counter()
        
        try:
            headers = {
                'X-API-Key': self.config.api_key,
                'Content-Type': 'application/json'
            }
            
            async with self.session.post(
                f"{self.config.api_url}/generate",
                json=request_data,
                headers=headers
            ) as response:
                result = await response.json()
                
                end_time = time.perf_counter()
                response_time = end_time - start_time
                
                return {
                    'success': response.status == 200,
                    'response_time': response_time,
                    'result': result,
                    'status_code': response.status
                }
                
        except Exception as e:
            end_time = time.perf_counter()
            response_time = end_time - start_time
            
            return {
                'success': False,
                'response_time': response_time,
                'error': str(e)
            }
    
    def create_test_request(self, index: int = 0) -> Dict[str, Any]:
        """Create a test request with varying parameters"""
        products = [
            "Premium wireless headphones with noise cancellation",
            "Smart fitness tracker with heart rate monitoring",
            "Ultra-fast gaming laptop with RGB keyboard",
            "Professional camera with 4K video recording",
            "Wireless charging pad with fast charging technology"
        ]
        
        platforms = ["Instagram", "Facebook", "Twitter", "LinkedIn", "TikTok"]
        tones = ["inspirational", "professional", "casual", "luxury", "friendly"]
        audiences = ["Young professionals", "Tech enthusiasts", "Fitness lovers", "Gamers", "Photographers"]
        
        return {
            "product_description": products[index % len(products)],
            "target_platform": platforms[index % len(platforms)],
            "tone": tones[index % len(tones)],
            "target_audience": audiences[index % len(audiences)],
            "key_points": ["Quality", "Innovation", "Performance"],
            "instructions": f"Emphasize the unique features of product {index}",
            "restrictions": ["no price mentions"],
            "creativity_level": 0.7 + (index % 3) * 0.1,
            "language": "en"
        }
    
    async def run_basic_performance_test(self) -> TestResult:
        """Run basic performance test"""
        logger.info("Running basic performance test")
        
        start_time = time.perf_counter()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # Warmup
        for i in range(self.config.warmup_requests):
            request_data = self.create_test_request(i)
            result = await self.make_request(request_data)
            if result['success']:
                successful_requests += 1
            else:
                failed_requests += 1
        
        # Main test
        for i in range(self.config.num_requests):
            request_data = self.create_test_request(i)
            result = await self.make_request(request_data)
            
            if result['success']:
                successful_requests += 1
                response_times.append(result['response_time'])
            else:
                failed_requests += 1
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p50_response_time = statistics.median(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50_response_time = p95_response_time = p99_response_time = 0
        
        requests_per_second = successful_requests / total_time if total_time > 0 else 0
        error_rate = failed_requests / (successful_requests + failed_requests) if (successful_requests + failed_requests) > 0 else 0
        
        # Get system metrics
        system_metrics = self.get_system_metrics()
        
        result = TestResult(
            test_name="Basic Performance Test",
            total_requests=self.config.num_requests + self.config.warmup_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            memory_usage=system_metrics,
            cpu_usage=system_metrics.get('cpu_percent', 0),
            gpu_usage=system_metrics.get('gpu_memory_gb', 0),
            error_rate=error_rate
        )
        
        logger.info(f"Basic performance test completed: {requests_per_second:.2f} req/s")
        return result
    
    async def run_concurrent_performance_test(self) -> TestResult:
        """Run concurrent performance test"""
        logger.info("Running concurrent performance test")
        
        start_time = time.perf_counter()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # Create tasks for concurrent execution
        tasks = []
        for i in range(self.config.concurrent_requests):
            for j in range(self.config.num_requests // self.config.concurrent_requests):
                request_data = self.create_test_request(i * 100 + j)
                task = asyncio.create_task(self.make_request(request_data))
                tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                failed_requests += 1
            elif result['success']:
                successful_requests += 1
                response_times.append(result['response_time'])
            else:
                failed_requests += 1
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p50_response_time = statistics.median(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50_response_time = p95_response_time = p99_response_time = 0
        
        requests_per_second = successful_requests / total_time if total_time > 0 else 0
        error_rate = failed_requests / (successful_requests + failed_requests) if (successful_requests + failed_requests) > 0 else 0
        
        # Get system metrics
        system_metrics = self.get_system_metrics()
        
        result = TestResult(
            test_name="Concurrent Performance Test",
            total_requests=len(tasks),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            memory_usage=system_metrics,
            cpu_usage=system_metrics.get('cpu_percent', 0),
            gpu_usage=system_metrics.get('gpu_memory_gb', 0),
            error_rate=error_rate
        )
        
        logger.info(f"Concurrent performance test completed: {requests_per_second:.2f} req/s")
        return result
    
    async def run_cache_performance_test(self) -> TestResult:
        """Run cache performance test"""
        logger.info("Running cache performance test")
        
        start_time = time.perf_counter()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        cache_hits = 0
        
        # Create a fixed set of requests for cache testing
        test_requests = []
        for i in range(self.config.cache_test_requests):
            test_requests.append(self.create_test_request(i))
        
        # First pass - populate cache
        for request_data in test_requests:
            result = await self.make_request(request_data)
            if result['success']:
                successful_requests += 1
                response_times.append(result['response_time'])
            else:
                failed_requests += 1
        
        # Second pass - test cache hits
        cache_response_times = []
        for request_data in test_requests:
            result = await self.make_request(request_data)
            if result['success']:
                successful_requests += 1
                cache_response_times.append(result['response_time'])
                cache_hits += 1
            else:
                failed_requests += 1
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate cache statistics
        if response_times and cache_response_times:
            cache_hit_ratio = len(cache_response_times) / len(test_requests)
            avg_cache_hit_time = statistics.mean(cache_response_times)
            avg_cache_miss_time = statistics.mean(response_times)
            cache_improvement = avg_cache_miss_time / avg_cache_hit_time if avg_cache_hit_time > 0 else 0
        else:
            cache_hit_ratio = 0
            avg_cache_hit_time = avg_cache_miss_time = cache_improvement = 0
        
        # Calculate overall statistics
        all_response_times = response_times + cache_response_times
        if all_response_times:
            avg_response_time = statistics.mean(all_response_times)
            min_response_time = min(all_response_times)
            max_response_time = max(all_response_times)
            p50_response_time = statistics.median(all_response_times)
            p95_response_time = np.percentile(all_response_times, 95)
            p99_response_time = np.percentile(all_response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50_response_time = p95_response_time = p99_response_time = 0
        
        requests_per_second = successful_requests / total_time if total_time > 0 else 0
        error_rate = failed_requests / (successful_requests + failed_requests) if (successful_requests + failed_requests) > 0 else 0
        
        # Get system metrics
        system_metrics = self.get_system_metrics()
        
        result = TestResult(
            test_name="Cache Performance Test",
            total_requests=len(test_requests) * 2,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            memory_usage=system_metrics,
            cpu_usage=system_metrics.get('cpu_percent', 0),
            gpu_usage=system_metrics.get('gpu_memory_gb', 0),
            cache_hit_ratio=cache_hit_ratio,
            error_rate=error_rate
        )
        
        logger.info(f"Cache performance test completed: {cache_hit_ratio:.2%} hit ratio, {cache_improvement:.1f}x improvement")
        return result
    
    async def run_memory_stress_test(self) -> TestResult:
        """Run memory stress test"""
        logger.info("Running memory stress test")
        
        start_time = time.perf_counter()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        memory_usage = []
        
        # Monitor memory during test
        for i in range(self.config.memory_test_requests):
            # Get memory before request
            memory_before = self.get_system_metrics()
            
            request_data = self.create_test_request(i)
            result = await self.make_request(request_data)
            
            # Get memory after request
            memory_after = self.get_system_metrics()
            memory_usage.append({
                'before': memory_before,
                'after': memory_after,
                'difference': memory_after['memory_used_gb'] - memory_before['memory_used_gb']
            })
            
            if result['success']:
                successful_requests += 1
                response_times.append(result['response_time'])
            else:
                failed_requests += 1
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate memory statistics
        memory_differences = [m['difference'] for m in memory_usage]
        avg_memory_increase = statistics.mean(memory_differences) if memory_differences else 0
        max_memory_increase = max(memory_differences) if memory_differences else 0
        
        # Calculate response time statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p50_response_time = statistics.median(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50_response_time = p95_response_time = p99_response_time = 0
        
        requests_per_second = successful_requests / total_time if total_time > 0 else 0
        error_rate = failed_requests / (successful_requests + failed_requests) if (successful_requests + failed_requests) > 0 else 0
        
        # Get final system metrics
        final_metrics = self.get_system_metrics()
        final_metrics['avg_memory_increase_gb'] = avg_memory_increase
        final_metrics['max_memory_increase_gb'] = max_memory_increase
        
        result = TestResult(
            test_name="Memory Stress Test",
            total_requests=self.config.memory_test_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            memory_usage=final_metrics,
            cpu_usage=final_metrics.get('cpu_percent', 0),
            gpu_usage=final_metrics.get('gpu_memory_gb', 0),
            error_rate=error_rate
        )
        
        logger.info(f"Memory stress test completed: {avg_memory_increase:.3f}GB avg memory increase")
        return result
    
    async def run_api_performance_test(self) -> TestResult:
        """Run API performance test"""
        logger.info("Running API performance test")
        
        start_time = time.perf_counter()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        for i in range(self.config.num_requests):
            request_data = self.create_test_request(i)
            result = await self.make_api_request(request_data)
            
            if result['success']:
                successful_requests += 1
                response_times.append(result['response_time'])
            else:
                failed_requests += 1
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p50_response_time = statistics.median(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50_response_time = p95_response_time = p99_response_time = 0
        
        requests_per_second = successful_requests / total_time if total_time > 0 else 0
        error_rate = failed_requests / (successful_requests + failed_requests) if (successful_requests + failed_requests) > 0 else 0
        
        # Get system metrics
        system_metrics = self.get_system_metrics()
        
        result = TestResult(
            test_name="API Performance Test",
            total_requests=self.config.num_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            memory_usage=system_metrics,
            cpu_usage=system_metrics.get('cpu_percent', 0),
            gpu_usage=system_metrics.get('gpu_memory_gb', 0),
            error_rate=error_rate
        )
        
        logger.info(f"API performance test completed: {requests_per_second:.2f} req/s")
        return result
    
    def generate_report(self, results: List[TestResult]) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("# Performance Test Report v11")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        report.append("## Summary")
        report.append("")
        report.append("| Test | Requests | Success Rate | Avg Response Time | RPS | Error Rate |")
        report.append("|------|----------|--------------|-------------------|-----|------------|")
        
        for result in results:
            success_rate = result.successful_requests / result.total_requests * 100
            report.append(f"| {result.test_name} | {result.total_requests} | {success_rate:.1f}% | {result.avg_response_time:.3f}s | {result.requests_per_second:.1f} | {result.error_rate:.1%} |")
        
        report.append("")
        
        # Detailed results
        for result in results:
            report.append(f"## {result.test_name}")
            report.append("")
            report.append(f"- **Total Requests**: {result.total_requests}")
            report.append(f"- **Successful Requests**: {result.successful_requests}")
            report.append(f"- **Failed Requests**: {result.failed_requests}")
            report.append(f"- **Success Rate**: {result.successful_requests / result.total_requests * 100:.1f}%")
            report.append(f"- **Total Time**: {result.total_time:.2f}s")
            report.append(f"- **Requests Per Second**: {result.requests_per_second:.2f}")
            report.append(f"- **Error Rate**: {result.error_rate:.1%}")
            report.append("")
            report.append("### Response Time Statistics")
            report.append(f"- **Average**: {result.avg_response_time:.3f}s")
            report.append(f"- **Minimum**: {result.min_response_time:.3f}s")
            report.append(f"- **Maximum**: {result.max_response_time:.3f}s")
            report.append(f"- **Median (P50)**: {result.p50_response_time:.3f}s")
            report.append(f"- **P95**: {result.p95_response_time:.3f}s")
            report.append(f"- **P99**: {result.p99_response_time:.3f}s")
            report.append("")
            report.append("### System Metrics")
            report.append(f"- **CPU Usage**: {result.cpu_usage:.1f}%")
            report.append(f"- **Memory Usage**: {result.memory_usage.get('memory_percent', 0):.1f}%")
            report.append(f"- **Memory Used**: {result.memory_usage.get('memory_used_gb', 0):.2f}GB")
            if result.gpu_usage:
                report.append(f"- **GPU Memory**: {result.gpu_usage:.2f}GB")
            if result.cache_hit_ratio is not None:
                report.append(f"- **Cache Hit Ratio**: {result.cache_hit_ratio:.1%}")
            report.append("")
        
        # Performance comparison
        report.append("## Performance Comparison")
        report.append("")
        
        # Find best performing test
        best_test = max(results, key=lambda r: r.requests_per_second)
        report.append(f"**Best Performance**: {best_test.test_name} with {best_test.requests_per_second:.2f} requests/second")
        report.append("")
        
        # Improvement analysis
        if len(results) >= 2:
            basic_test = next((r for r in results if "Basic" in r.test_name), None)
            concurrent_test = next((r for r in results if "Concurrent" in r.test_name), None)
            
            if basic_test and concurrent_test:
                improvement = concurrent_test.requests_per_second / basic_test.requests_per_second
                report.append(f"**Concurrent vs Sequential Improvement**: {improvement:.1f}x")
                report.append("")
        
        return "\n".join(report)
    
    def save_results(self, results: List[TestResult], filename: str = None):
        """Save test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_test_results_v11_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                'test_name': result.test_name,
                'total_requests': result.total_requests,
                'successful_requests': result.successful_requests,
                'failed_requests': result.failed_requests,
                'total_time': result.total_time,
                'avg_response_time': result.avg_response_time,
                'min_response_time': result.min_response_time,
                'max_response_time': result.max_response_time,
                'p50_response_time': result.p50_response_time,
                'p95_response_time': result.p95_response_time,
                'p99_response_time': result.p99_response_time,
                'requests_per_second': result.requests_per_second,
                'memory_usage': result.memory_usage,
                'cpu_usage': result.cpu_usage,
                'gpu_usage': result.gpu_usage,
                'cache_hit_ratio': result.cache_hit_ratio,
                'error_rate': result.error_rate
            })
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all performance tests"""
        logger.info("Starting comprehensive performance testing")
        
        results = []
        
        try:
            # Run basic performance test
            result = await self.run_basic_performance_test()
            results.append(result)
            
            # Run concurrent performance test
            result = await self.run_concurrent_performance_test()
            results.append(result)
            
            # Run cache performance test
            result = await self.run_cache_performance_test()
            results.append(result)
            
            # Run memory stress test
            result = await self.run_memory_stress_test()
            results.append(result)
            
            # Run API performance test (if API is available)
            try:
                result = await self.run_api_performance_test()
                results.append(result)
            except Exception as e:
                logger.warning(f"API performance test skipped: {e}")
            
        except Exception as e:
            logger.error(f"Error during performance testing: {e}")
        
        return results

async def main():
    """Main function to run performance tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Testing Script v11")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of requests per test")
    parser.add_argument("--concurrent-requests", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL for testing")
    parser.add_argument("--api-key", default="your-secret-api-key", help="API key for testing")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Create test configuration
    config = TestConfig(
        num_requests=args.num_requests,
        concurrent_requests=args.concurrent_requests,
        api_url=args.api_url,
        api_key=args.api_key
    )
    
    # Create and run performance tester
    tester = PerformanceTester(config)
    
    try:
        await tester.initialize()
        
        # Run all tests
        results = await tester.run_all_tests()
        
        # Generate and print report
        report = tester.generate_report(results)
        print(report)
        
        # Save results
        tester.save_results(results, args.output)
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 
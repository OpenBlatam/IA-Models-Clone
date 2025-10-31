"""
Performance and Load Testing Suite for Ultimate Brand Voice AI System
====================================================================

This module provides comprehensive performance testing, load testing,
stress testing, and benchmarking for the Brand Voice AI system.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import pytest
import unittest
import time
import psutil
import os
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import tempfile
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from pathlib import Path

# Import Brand Voice AI modules
from brand_ai_transformer import AdvancedBrandTransformer
from brand_ai_serving import BrandAIServing
from brand_ai_computer_vision import AdvancedComputerVisionSystem
from brand_ai_sentiment_analysis import AdvancedSentimentAnalyzer
from brand_ai_voice_cloning import AdvancedVoiceCloningSystem
from brand_ai_performance_prediction import AdvancedPerformancePredictionSystem

# Import test utilities
from test_utils import (
    create_mock_config, create_test_data, create_test_images, 
    create_test_audio, measure_execution_time, measure_memory_usage,
    setup_test_environment, teardown_test_environment
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    operation_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    error_rate: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class LoadTestResult:
    """Load test result data class"""
    test_name: str
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float
    error_rate: float
    timestamp: datetime

class PerformanceMonitor:
    """Performance monitoring utility"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process(os.getpid())
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
        self.start_cpu = self.process.cpu_percent()
    
    def stop_monitoring(self, operation_name: str) -> PerformanceMetrics:
        """Stop performance monitoring and return metrics"""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        end_cpu = self.process.cpu_percent()
        
        execution_time = end_time - self.start_time
        memory_usage = end_memory - self.start_memory
        cpu_usage = (self.start_cpu + end_cpu) / 2
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            throughput=1.0 / execution_time if execution_time > 0 else 0,
            error_rate=0.0,
            timestamp=datetime.now()
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict()
        }

class TestBrandAIPerformance(unittest.TestCase):
    """Performance tests for Brand Voice AI system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = create_mock_config()
        self.monitor = PerformanceMonitor()
        self.temp_dirs = setup_test_environment()
        
        # Initialize systems
        self.transformer = AdvancedBrandTransformer(self.config)
        self.serving = BrandAIServing(self.config)
        self.computer_vision = AdvancedComputerVisionSystem(self.config)
        self.sentiment_analyzer = AdvancedSentimentAnalyzer(self.config)
        self.voice_cloning = AdvancedVoiceCloningSystem(self.config)
        self.performance_prediction = AdvancedPerformancePredictionSystem(self.config)
    
    def tearDown(self):
        """Clean up test fixtures"""
        teardown_test_environment(self.temp_dirs)
    
    @pytest.mark.asyncio
    async def test_transformer_performance(self):
        """Test transformer model performance"""
        try:
            await self.transformer.initialize_models()
            
            # Test with different input sizes
            input_sizes = [100, 500, 1000, 2000]
            results = []
            
            for size in input_sizes:
                test_data = create_test_data("text", size)
                
                self.monitor.start_monitoring()
                result = await self.transformer.analyze_brand_content(test_data)
                metrics = self.monitor.stop_monitoring(f"transformer_analysis_{size}")
                
                results.append(metrics)
                
                # Assert performance thresholds
                self.assertLess(metrics.execution_time, 10.0)  # Max 10 seconds
                self.assertLess(metrics.memory_usage, 500.0)   # Max 500MB
                
                logger.info(f"Transformer analysis for {size} items: {metrics.execution_time:.2f}s, {metrics.memory_usage:.2f}MB")
            
            # Generate performance report
            self._generate_performance_report("transformer", results)
            
        except Exception as e:
            logger.error(f"✗ Transformer performance test failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_computer_vision_performance(self):
        """Test computer vision system performance"""
        try:
            await self.computer_vision.initialize_models()
            
            # Test with different image sizes
            image_sizes = [(224, 224), (512, 512), (1024, 1024)]
            results = []
            
            for size in image_sizes:
                test_images = create_test_images(5, size)
                
                self.monitor.start_monitoring()
                for image_path in test_images:
                    result = await self.computer_vision.analyze_brand_image(image_path)
                metrics = self.monitor.stop_monitoring(f"vision_analysis_{size[0]}x{size[1]}")
                
                results.append(metrics)
                
                # Assert performance thresholds
                self.assertLess(metrics.execution_time, 15.0)  # Max 15 seconds
                self.assertLess(metrics.memory_usage, 800.0)   # Max 800MB
                
                logger.info(f"Vision analysis for {size}: {metrics.execution_time:.2f}s, {metrics.memory_usage:.2f}MB")
            
            # Generate performance report
            self._generate_performance_report("computer_vision", results)
            
        except Exception as e:
            logger.error(f"✗ Computer vision performance test failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis_performance(self):
        """Test sentiment analysis performance"""
        try:
            await self.sentiment_analyzer.initialize_models()
            
            # Test with different text lengths
            text_lengths = [50, 200, 500, 1000]
            results = []
            
            for length in text_lengths:
                test_text = " ".join(["test"] * length)
                
                self.monitor.start_monitoring()
                result = await self.sentiment_analyzer.analyze_text_sentiment(test_text)
                metrics = self.monitor.stop_monitoring(f"sentiment_analysis_{length}")
                
                results.append(metrics)
                
                # Assert performance thresholds
                self.assertLess(metrics.execution_time, 5.0)   # Max 5 seconds
                self.assertLess(metrics.memory_usage, 200.0)   # Max 200MB
                
                logger.info(f"Sentiment analysis for {length} chars: {metrics.execution_time:.2f}s, {metrics.memory_usage:.2f}MB")
            
            # Generate performance report
            self._generate_performance_report("sentiment_analysis", results)
            
        except Exception as e:
            logger.error(f"✗ Sentiment analysis performance test failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_voice_cloning_performance(self):
        """Test voice cloning system performance"""
        try:
            await self.voice_cloning.initialize_models()
            
            # Test with different audio durations
            audio_durations = [1.0, 3.0, 5.0, 10.0]
            results = []
            
            for duration in audio_durations:
                test_audio = create_test_audio(duration)
                
                self.monitor.start_monitoring()
                voice_profile = await self.voice_cloning.create_voice_profile("TestVoice", [test_audio])
                metrics = self.monitor.stop_monitoring(f"voice_cloning_{duration}s")
                
                results.append(metrics)
                
                # Assert performance thresholds
                self.assertLess(metrics.execution_time, 20.0)  # Max 20 seconds
                self.assertLess(metrics.memory_usage, 1000.0)  # Max 1GB
                
                logger.info(f"Voice cloning for {duration}s: {metrics.execution_time:.2f}s, {metrics.memory_usage:.2f}MB")
            
            # Generate performance report
            self._generate_performance_report("voice_cloning", results)
            
        except Exception as e:
            logger.error(f"✗ Voice cloning performance test failed: {e}")
            raise
    
    def _generate_performance_report(self, system_name: str, metrics: List[PerformanceMetrics]):
        """Generate performance report"""
        if not metrics:
            return
        
        # Calculate statistics
        execution_times = [m.execution_time for m in metrics]
        memory_usages = [m.memory_usage for m in metrics]
        
        report = {
            'system': system_name,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'execution_time': {
                    'mean': statistics.mean(execution_times),
                    'median': statistics.median(execution_times),
                    'std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                    'min': min(execution_times),
                    'max': max(execution_times)
                },
                'memory_usage': {
                    'mean': statistics.mean(memory_usages),
                    'median': statistics.median(memory_usages),
                    'std': statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0,
                    'min': min(memory_usages),
                    'max': max(memory_usages)
                }
            },
            'metrics': [m.__dict__ for m in metrics]
        }
        
        # Save report
        report_path = os.path.join(self.temp_dirs[0], f"{system_name}_performance_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved: {report_path}")

class TestBrandAILoadTesting(unittest.TestCase):
    """Load testing for Brand Voice AI system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = create_mock_config()
        self.temp_dirs = setup_test_environment()
        
        # Initialize serving system
        self.serving = BrandAIServing(self.config)
    
    def tearDown(self):
        """Clean up test fixtures"""
        teardown_test_environment(self.temp_dirs)
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test system performance under concurrent requests"""
        try:
            await self.serving.initialize()
            
            # Test with different concurrency levels
            concurrency_levels = [1, 5, 10, 20, 50]
            results = []
            
            for concurrency in concurrency_levels:
                result = await self._run_concurrent_test(concurrency, 100)
                results.append(result)
                
                # Assert performance thresholds
                self.assertLess(result.average_response_time, 5.0)  # Max 5 seconds
                self.assertLess(result.error_rate, 0.05)            # Max 5% error rate
                
                logger.info(f"Concurrency {concurrency}: {result.average_response_time:.2f}s avg, {result.error_rate:.2%} error rate")
            
            # Generate load test report
            self._generate_load_test_report("concurrent_requests", results)
            
        except Exception as e:
            logger.error(f"✗ Concurrent requests test failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_sustained_load(self):
        """Test system performance under sustained load"""
        try:
            await self.serving.initialize()
            
            # Run sustained load test for 5 minutes
            duration = 300  # 5 minutes
            concurrency = 10
            results = []
            
            start_time = time.time()
            while time.time() - start_time < duration:
                result = await self._run_concurrent_test(concurrency, 50)
                results.append(result)
                
                # Check system health
                system_metrics = psutil.virtual_memory()
                if system_metrics.percent > 90:
                    logger.warning("High memory usage detected during sustained load test")
                
                await asyncio.sleep(10)  # Wait 10 seconds between test batches
            
            # Analyze results
            avg_response_times = [r.average_response_time for r in results]
            error_rates = [r.error_rate for r in results]
            
            # Assert performance degradation is minimal
            self.assertLess(max(avg_response_times) - min(avg_response_times), 2.0)  # Max 2s variation
            self.assertLess(max(error_rates), 0.1)  # Max 10% error rate
            
            logger.info(f"Sustained load test completed: {len(results)} batches, avg response time: {statistics.mean(avg_response_times):.2f}s")
            
        except Exception as e:
            logger.error(f"✗ Sustained load test failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_stress_testing(self):
        """Test system behavior under extreme load"""
        try:
            await self.serving.initialize()
            
            # Gradually increase load until system fails
            max_concurrency = 100
            step = 10
            results = []
            
            for concurrency in range(step, max_concurrency + 1, step):
                try:
                    result = await self._run_concurrent_test(concurrency, 20)
                    results.append(result)
                    
                    # Stop if error rate is too high
                    if result.error_rate > 0.5:  # 50% error rate
                        logger.info(f"Stress test stopped at concurrency {concurrency} due to high error rate")
                        break
                        
                except Exception as e:
                    logger.info(f"Stress test failed at concurrency {concurrency}: {e}")
                    break
            
            # Find breaking point
            breaking_point = max([r.concurrent_users for r in results if r.error_rate < 0.5])
            logger.info(f"System breaking point: {breaking_point} concurrent users")
            
            # Assert system can handle reasonable load
            self.assertGreater(breaking_point, 20)  # Should handle at least 20 concurrent users
            
        except Exception as e:
            logger.error(f"✗ Stress testing failed: {e}")
            raise
    
    async def _run_concurrent_test(self, concurrency: int, requests_per_user: int) -> LoadTestResult:
        """Run concurrent load test"""
        async def make_request():
            try:
                start_time = time.time()
                test_data = {
                    "brand_name": "TestBrand",
                    "content": ["Test content for analysis"]
                }
                result = await self.serving.analyze_brand(test_data)
                end_time = time.time()
                return {
                    'success': True,
                    'response_time': end_time - start_time,
                    'error': None
                }
            except Exception as e:
                return {
                    'success': False,
                    'response_time': 0,
                    'error': str(e)
                }
        
        # Create concurrent tasks
        tasks = []
        for _ in range(concurrency):
            for _ in range(requests_per_user):
                tasks.append(make_request())
        
        # Execute all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        successful_requests = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
        failed_requests = len(results) - successful_requests
        response_times = [r['response_time'] for r in results if isinstance(r, dict) and r.get('success', False)]
        
        total_time = end_time - start_time
        total_requests = len(results)
        
        return LoadTestResult(
            test_name=f"concurrent_{concurrency}",
            concurrent_users=concurrency,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=statistics.mean(response_times) if response_times else 0,
            p95_response_time=np.percentile(response_times, 95) if response_times else 0,
            p99_response_time=np.percentile(response_times, 99) if response_times else 0,
            throughput=total_requests / total_time if total_time > 0 else 0,
            error_rate=failed_requests / total_requests if total_requests > 0 else 0,
            timestamp=datetime.now()
        )
    
    def _generate_load_test_report(self, test_name: str, results: List[LoadTestResult]):
        """Generate load test report"""
        if not results:
            return
        
        # Calculate statistics
        response_times = [r.average_response_time for r in results]
        error_rates = [r.error_rate for r in results]
        throughputs = [r.throughput for r in results]
        
        report = {
            'test_name': test_name,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': len(results),
                'max_concurrency': max([r.concurrent_users for r in results]),
                'average_response_time': statistics.mean(response_times),
                'max_response_time': max(response_times),
                'average_error_rate': statistics.mean(error_rates),
                'max_error_rate': max(error_rates),
                'average_throughput': statistics.mean(throughputs),
                'max_throughput': max(throughputs)
            },
            'results': [r.__dict__ for r in results]
        }
        
        # Save report
        report_path = os.path.join(self.temp_dirs[0], f"{test_name}_load_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Load test report saved: {report_path}")

class TestBrandAIBenchmarking(unittest.TestCase):
    """Benchmarking tests for Brand Voice AI system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = create_mock_config()
        self.temp_dirs = setup_test_environment()
        self.benchmark_data = self._create_benchmark_data()
    
    def tearDown(self):
        """Clean up test fixtures"""
        teardown_test_environment(self.temp_dirs)
    
    def _create_benchmark_data(self) -> Dict[str, Any]:
        """Create benchmark test data"""
        return {
            'text_analysis': {
                'small': create_test_data("text", 10),
                'medium': create_test_data("text", 100),
                'large': create_test_data("text", 1000)
            },
            'image_analysis': {
                'small': create_test_images(5, (224, 224)),
                'medium': create_test_images(5, (512, 512)),
                'large': create_test_images(5, (1024, 1024))
            },
            'audio_analysis': {
                'short': create_test_audio(1.0),
                'medium': create_test_audio(5.0),
                'long': create_test_audio(30.0)
            }
        }
    
    @pytest.mark.asyncio
    async def test_text_analysis_benchmark(self):
        """Benchmark text analysis performance"""
        try:
            transformer = AdvancedBrandTransformer(self.config)
            await transformer.initialize_models()
            
            results = {}
            for size, data in self.benchmark_data['text_analysis'].items():
                # Warm up
                await transformer.analyze_brand_content(data[:5])
                
                # Benchmark
                start_time = time.time()
                for _ in range(10):  # Run 10 iterations
                    await transformer.analyze_brand_content(data)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                results[size] = {
                    'avg_time': avg_time,
                    'throughput': len(data) / avg_time,
                    'data_size': len(data)
                }
                
                logger.info(f"Text analysis benchmark - {size}: {avg_time:.2f}s, {len(data)/avg_time:.2f} items/s")
            
            # Save benchmark results
            self._save_benchmark_results("text_analysis", results)
            
        except Exception as e:
            logger.error(f"✗ Text analysis benchmark failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_image_analysis_benchmark(self):
        """Benchmark image analysis performance"""
        try:
            computer_vision = AdvancedComputerVisionSystem(self.config)
            await computer_vision.initialize_models()
            
            results = {}
            for size, images in self.benchmark_data['image_analysis'].items():
                # Warm up
                await computer_vision.analyze_brand_image(images[0])
                
                # Benchmark
                start_time = time.time()
                for _ in range(5):  # Run 5 iterations
                    for image_path in images:
                        await computer_vision.analyze_brand_image(image_path)
                end_time = time.time()
                
                total_images = len(images) * 5
                avg_time = (end_time - start_time) / total_images
                results[size] = {
                    'avg_time': avg_time,
                    'throughput': 1 / avg_time,
                    'image_count': total_images,
                    'image_size': images[0].split('x')[-1] if 'x' in images[0] else 'unknown'
                }
                
                logger.info(f"Image analysis benchmark - {size}: {avg_time:.2f}s per image, {1/avg_time:.2f} images/s")
            
            # Save benchmark results
            self._save_benchmark_results("image_analysis", results)
            
        except Exception as e:
            logger.error(f"✗ Image analysis benchmark failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_audio_analysis_benchmark(self):
        """Benchmark audio analysis performance"""
        try:
            voice_cloning = AdvancedVoiceCloningSystem(self.config)
            await voice_cloning.initialize_models()
            
            results = {}
            for size, audio_path in self.benchmark_data['audio_analysis'].items():
                # Warm up
                await voice_cloning.create_voice_profile(f"TestVoice_{size}", [audio_path])
                
                # Benchmark
                start_time = time.time()
                for i in range(3):  # Run 3 iterations
                    await voice_cloning.create_voice_profile(f"TestVoice_{size}_{i}", [audio_path])
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 3
                results[size] = {
                    'avg_time': avg_time,
                    'throughput': 1 / avg_time,
                    'audio_duration': float(size.replace('short', '1').replace('medium', '5').replace('long', '30'))
                }
                
                logger.info(f"Audio analysis benchmark - {size}: {avg_time:.2f}s, {1/avg_time:.2f} profiles/s")
            
            # Save benchmark results
            self._save_benchmark_results("audio_analysis", results)
            
        except Exception as e:
            logger.error(f"✗ Audio analysis benchmark failed: {e}")
            raise
    
    def _save_benchmark_results(self, benchmark_type: str, results: Dict[str, Any]):
        """Save benchmark results"""
        report = {
            'benchmark_type': benchmark_type,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
                'python_version': os.sys.version
            }
        }
        
        report_path = os.path.join(self.temp_dirs[0], f"{benchmark_type}_benchmark.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved: {report_path}")

class TestBrandAIMemoryProfiling(unittest.TestCase):
    """Memory profiling tests for Brand Voice AI system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = create_mock_config()
        self.temp_dirs = setup_test_environment()
    
    def tearDown(self):
        """Clean up test fixtures"""
        teardown_test_environment(self.temp_dirs)
    
    @pytest.mark.asyncio
    async def test_memory_usage_patterns(self):
        """Test memory usage patterns during operations"""
        try:
            transformer = AdvancedBrandTransformer(self.config)
            
            # Monitor memory during initialization
            initial_memory = self._get_memory_usage()
            await transformer.initialize_models()
            init_memory = self._get_memory_usage()
            init_memory_increase = init_memory - initial_memory
            
            logger.info(f"Memory increase during initialization: {init_memory_increase:.2f}MB")
            
            # Monitor memory during operations
            test_data = create_test_data("text", 100)
            operation_memories = []
            
            for i in range(10):
                before_memory = self._get_memory_usage()
                result = await transformer.analyze_brand_content(test_data)
                after_memory = self._get_memory_usage()
                
                operation_memories.append(after_memory - before_memory)
                
                # Force garbage collection
                gc.collect()
            
            avg_operation_memory = statistics.mean(operation_memories)
            max_operation_memory = max(operation_memories)
            
            logger.info(f"Average memory per operation: {avg_operation_memory:.2f}MB")
            logger.info(f"Max memory per operation: {max_operation_memory:.2f}MB")
            
            # Assert memory usage is reasonable
            self.assertLess(init_memory_increase, 1000)  # Max 1GB for initialization
            self.assertLess(avg_operation_memory, 100)   # Max 100MB per operation
            
        except Exception as e:
            logger.error(f"✗ Memory usage patterns test failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Test for memory leaks during extended operations"""
        try:
            transformer = AdvancedBrandTransformer(self.config)
            await transformer.initialize_models()
            
            test_data = create_test_data("text", 50)
            memory_samples = []
            
            # Run operations and monitor memory
            for i in range(50):
                result = await transformer.analyze_brand_content(test_data)
                
                if i % 10 == 0:  # Sample every 10 operations
                    memory_samples.append(self._get_memory_usage())
                    gc.collect()  # Force garbage collection
            
            # Check for memory leak (increasing trend)
            if len(memory_samples) >= 3:
                # Calculate trend
                x = list(range(len(memory_samples)))
                y = memory_samples
                
                # Simple linear regression to detect trend
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x[i] * y[i] for i in range(n))
                sum_x2 = sum(x[i] ** 2 for i in range(n))
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                
                logger.info(f"Memory trend slope: {slope:.2f}MB per sample")
                
                # Assert no significant memory leak (slope < 10MB per sample)
                self.assertLess(slope, 10.0)
            
        except Exception as e:
            logger.error(f"✗ Memory leak detection test failed: {e}")
            raise
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

class TestBrandAIScalability(unittest.TestCase):
    """Scalability tests for Brand Voice AI system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = create_mock_config()
        self.temp_dirs = setup_test_environment()
    
    def tearDown(self):
        """Clean up test fixtures"""
        teardown_test_environment(self.temp_dirs)
    
    @pytest.mark.asyncio
    async def test_horizontal_scaling(self):
        """Test horizontal scaling capabilities"""
        try:
            # Simulate multiple instances
            instances = []
            for i in range(3):
                serving = BrandAIServing(self.config)
                await serving.initialize()
                instances.append(serving)
            
            # Test load distribution
            test_data = {
                "brand_name": f"TestBrand_{i}",
                "content": ["Test content for analysis"]
            }
            
            # Send requests to different instances
            results = []
            for i, instance in enumerate(instances):
                start_time = time.time()
                result = await instance.analyze_brand(test_data)
                end_time = time.time()
                
                results.append({
                    'instance': i,
                    'response_time': end_time - start_time,
                    'success': result is not None
                })
            
            # Assert all instances respond successfully
            success_rate = sum(1 for r in results if r['success']) / len(results)
            self.assertGreater(success_rate, 0.9)  # 90% success rate
            
            logger.info(f"Horizontal scaling test: {success_rate:.2%} success rate across {len(instances)} instances")
            
        except Exception as e:
            logger.error(f"✗ Horizontal scaling test failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_data_volume_scaling(self):
        """Test system performance with increasing data volumes"""
        try:
            transformer = AdvancedBrandTransformer(self.config)
            await transformer.initialize_models()
            
            # Test with increasing data volumes
            data_sizes = [10, 50, 100, 200, 500]
            results = []
            
            for size in data_sizes:
                test_data = create_test_data("text", size)
                
                start_time = time.time()
                result = await transformer.analyze_brand_content(test_data)
                end_time = time.time()
                
                execution_time = end_time - start_time
                throughput = size / execution_time
                
                results.append({
                    'data_size': size,
                    'execution_time': execution_time,
                    'throughput': throughput
                })
                
                logger.info(f"Data volume scaling - {size} items: {execution_time:.2f}s, {throughput:.2f} items/s")
            
            # Check if throughput scales reasonably
            throughputs = [r['throughput'] for r in results]
            min_throughput = min(throughputs)
            max_throughput = max(throughputs)
            
            # Throughput should not degrade significantly
            self.assertGreater(min_throughput / max_throughput, 0.5)  # Max 50% degradation
            
        except Exception as e:
            logger.error(f"✗ Data volume scaling test failed: {e}")
            raise

def run_performance_tests():
    """Run all performance tests"""
    test_suites = [
        TestBrandAIPerformance,
        TestBrandAILoadTesting,
        TestBrandAIBenchmarking,
        TestBrandAIMemoryProfiling,
        TestBrandAIScalability
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_suite in test_suites:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_suite)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        passed_tests += result.testsRun - len(result.failures) - len(result.errors)
        failed_tests += len(result.failures) + len(result.errors)
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"{'='*60}")
    
    return failed_tests == 0

if __name__ == "__main__":
    # Run performance tests
    success = run_performance_tests()
    
    if success:
        print("\n🎉 All performance tests passed successfully!")
        exit(0)
    else:
        print("\n❌ Some performance tests failed. Please check the output above.")
        exit(1)

























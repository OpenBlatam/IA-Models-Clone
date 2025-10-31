#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced SEO Engine
Tests all components including performance, error handling, and integration
"""

import unittest
import asyncio
import time
import threading
import tempfile
import os
import sys
from typing import Dict, List, Any
import json
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_seo_engine import (
    EnhancedSEOEngine, EnhancedSEOConfig, EnhancedSEOProcessor,
    ModelManager, LRUCache, CircuitBreaker, MetricsCollector,
    InputValidator, SEOError, ProcessingError, ValidationError
)

# ============================================================================
# TEST DATA
# ============================================================================

SAMPLE_TEXTS = [
    "This is a sample text for SEO analysis. It contains multiple sentences and should provide good insights for optimization.",
    "Another example text with different content and structure for comprehensive testing of the SEO engine capabilities.",
    "A third text to demonstrate batch processing and concurrent analysis features of the enhanced system.",
    "This is a longer text that should provide more comprehensive SEO analysis results. It includes various sentence structures and should test the system's ability to handle different content types effectively.",
    "Short text."
]

INVALID_TEXTS = [
    "",  # Empty text
    "   ",  # Whitespace only
    "a" * 15000,  # Too long
    123,  # Not a string
    None,  # None value
]

# ============================================================================
# BASE TEST CLASS
# ============================================================================

class BaseTestCase(unittest.TestCase):
    """Base test case with common setup and teardown."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = EnhancedSEOConfig(
            model_name="microsoft/DialoGPT-medium",
            enable_caching=True,
            enable_async=True,
            enable_profiling=True,
            batch_size=2,
            max_concurrent_requests=3,
            enable_logging=False,  # Disable logging for tests
            log_level="ERROR"
        )
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidator(BaseTestCase):
    """Test input validation functionality."""
    
    def test_valid_text_validation(self):
        """Test validation of valid texts."""
        for text in SAMPLE_TEXTS:
            validated = InputValidator.validate_text(text)
            self.assertEqual(validated, text.strip())
    
    def test_invalid_text_validation(self):
        """Test validation of invalid texts."""
        for text in INVALID_TEXTS:
            with self.assertRaises(ValidationError):
                InputValidator.validate_text(text)
    
    def test_text_normalization(self):
        """Test text normalization."""
        text = "  Multiple    spaces   and\ttabs  "
        normalized = InputValidator.validate_text(text)
        self.assertEqual(normalized, "Multiple spaces and tabs")
    
    def test_text_length_validation(self):
        """Test text length validation."""
        # Test maximum length
        long_text = "a" * 10001
        with self.assertRaises(ValidationError):
            InputValidator.validate_text(long_text)
        
        # Test maximum allowed length
        max_text = "a" * 10000
        validated = InputValidator.validate_text(max_text)
        self.assertEqual(validated, max_text)
    
    def test_texts_list_validation(self):
        """Test validation of text lists."""
        # Valid list
        validated = InputValidator.validate_texts(SAMPLE_TEXTS)
        self.assertEqual(len(validated), len(SAMPLE_TEXTS))
        
        # Invalid list
        with self.assertRaises(ValidationError):
            InputValidator.validate_texts("not a list")
        
        # Empty list
        with self.assertRaises(ValidationError):
            InputValidator.validate_texts([])

# ============================================================================
# CACHE TESTS
# ============================================================================

class TestLRUCache(BaseTestCase):
    """Test LRU cache functionality."""
    
    def setUp(self):
        super().setUp()
        self.cache = LRUCache(max_size=3)
    
    def test_basic_cache_operations(self):
        """Test basic cache get/set operations."""
        # Set values
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        # Get values
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.get("key2"), "value2")
        self.assertIsNone(self.cache.get("key3"))
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        # Fill cache
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")
        
        # Access key1 to make it most recently used
        self.cache.get("key1")
        
        # Add new key, should evict key2 (least recently used)
        self.cache.set("key4", "value4")
        
        # key2 should be evicted
        self.assertIsNone(self.cache.get("key2"))
        # Other keys should still be there
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.get("key3"), "value3")
        self.assertEqual(self.cache.get("key4"), "value4")
    
    def test_ttl_functionality(self):
        """Test TTL functionality."""
        self.cache.set("key1", "value1", ttl=1)  # 1 second TTL
        
        # Should be available immediately
        self.assertEqual(self.cache.get("key1"), "value1")
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        self.assertIsNone(self.cache.get("key1"))
    
    def test_cache_cleanup(self):
        """Test cache cleanup functionality."""
        # Add entries with TTL
        self.cache.set("key1", "value1", ttl=1)
        self.cache.set("key2", "value2", ttl=0)  # No TTL
        self.cache.set("key3", "value3", ttl=1)
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Cleanup expired entries
        self.cache.cleanup_expired()
        
        # key2 should still be there (no TTL)
        self.assertEqual(self.cache.get("key2"), "value2")
        # key1 and key3 should be gone
        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNone(self.cache.get("key3"))
    
    def test_thread_safety(self):
        """Test cache thread safety."""
        def worker(worker_id):
            for i in range(100):
                key = f"worker{worker_id}_key{i}"
                value = f"value{i}"
                self.cache.set(key, value)
                retrieved = self.cache.get(key)
                self.assertEqual(retrieved, value)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()

# ============================================================================
# CIRCUIT BREAKER TESTS
# ============================================================================

class TestCircuitBreaker(BaseTestCase):
    """Test circuit breaker functionality."""
    
    def setUp(self):
        super().setUp()
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    
    def test_successful_calls(self):
        """Test successful function calls."""
        def success_func():
            return "success"
        
        result = self.circuit_breaker.call(success_func)
        self.assertEqual(result, "success")
        self.assertEqual(self.circuit_breaker.state, "CLOSED")
    
    def test_failure_threshold(self):
        """Test failure threshold behavior."""
        def failing_func():
            raise Exception("Test failure")
        
        # Should fail 3 times before opening circuit
        for i in range(3):
            with self.assertRaises(Exception):
                self.circuit_breaker.call(failing_func)
        
        # Circuit should now be open
        self.assertEqual(self.circuit_breaker.state, "OPEN")
        
        # Next call should fail immediately
        with self.assertRaises(Exception):
            self.circuit_breaker.call(failing_func)
    
    def test_recovery_timeout(self):
        """Test recovery timeout behavior."""
        def failing_func():
            raise Exception("Test failure")
        
        # Open the circuit
        for i in range(3):
            with self.assertRaises(Exception):
                self.circuit_breaker.call(failing_func)
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Circuit should be half-open
        self.assertEqual(self.circuit_breaker.state, "HALF_OPEN")
        
        # Successful call should close circuit
        def success_func():
            return "success"
        
        result = self.circuit_breaker.call(success_func)
        self.assertEqual(result, "success")
        self.assertEqual(self.circuit_breaker.state, "CLOSED")
    
    def test_reset_on_success(self):
        """Test circuit reset on successful call."""
        def failing_func():
            raise Exception("Test failure")
        
        def success_func():
            return "success"
        
        # Fail twice
        for i in range(2):
            with self.assertRaises(Exception):
                self.circuit_breaker.call(failing_func)
        
        # Success should reset failure count
        result = self.circuit_breaker.call(success_func)
        self.assertEqual(result, "success")
        self.assertEqual(self.circuit_breaker.failure_count, 0)

# ============================================================================
# METRICS COLLECTOR TESTS
# ============================================================================

class TestMetricsCollector(BaseTestCase):
    """Test metrics collection functionality."""
    
    def setUp(self):
        super().setUp()
        self.metrics = MetricsCollector()
    
    def test_counter_increment(self):
        """Test counter increment functionality."""
        self.metrics.increment_counter("test_counter")
        self.metrics.increment_counter("test_counter", 5)
        
        stats = self.metrics.get_stats()
        self.assertEqual(stats['counters']['test_counter'], 6)
    
    def test_timing_recording(self):
        """Test timing recording functionality."""
        # Record some timings
        self.metrics.record_timing("test_timing", 1.0)
        self.metrics.record_timing("test_timing", 2.0)
        self.metrics.record_timing("test_timing", 3.0)
        
        stats = self.metrics.get_stats()
        timing_stats = stats['test_timing_timings']
        
        self.assertEqual(timing_stats['count'], 3)
        self.assertEqual(timing_stats['mean'], 2.0)
        self.assertEqual(timing_stats['min'], 1.0)
        self.assertEqual(timing_stats['max'], 3.0)
        self.assertEqual(timing_stats['std'], 1.0)
    
    def test_timer_context_manager(self):
        """Test timer context manager."""
        with self.metrics.timer("context_timer"):
            time.sleep(0.1)
        
        stats = self.metrics.get_stats()
        timing_stats = stats['context_timer_timings']
        
        self.assertEqual(timing_stats['count'], 1)
        self.assertGreater(timing_stats['mean'], 0.1)
    
    def test_value_recording(self):
        """Test value recording functionality."""
        self.metrics.record_value("test_value", 10.5)
        self.metrics.record_value("test_value", 20.5)
        
        stats = self.metrics.get_stats()
        value_stats = stats['test_value']
        
        self.assertEqual(value_stats['count'], 2)
        self.assertEqual(value_stats['mean'], 15.5)
    
    def test_thread_safety(self):
        """Test metrics collector thread safety."""
        def worker(worker_id):
            for i in range(100):
                self.metrics.increment_counter(f"worker{worker_id}_counter")
                self.metrics.record_timing(f"worker{worker_id}_timing", i)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all counters are correct
        stats = self.metrics.get_stats()
        for i in range(5):
            counter_name = f"worker{i}_counter"
            self.assertEqual(stats['counters'][counter_name], 100)

# ============================================================================
# SEO PROCESSOR TESTS
# ============================================================================

class TestEnhancedSEOProcessor(BaseTestCase):
    """Test enhanced SEO processor functionality."""
    
    def setUp(self):
        super().setUp()
        self.processor = EnhancedSEOProcessor(self.config)
    
    def tearDown(self):
        """Clean up processor."""
        if hasattr(self, 'processor'):
            self.processor.cleanup()
        super().tearDown()
    
    def test_single_text_processing(self):
        """Test single text processing."""
        text = SAMPLE_TEXTS[0]
        result = self.processor.process(text)
        
        # Check required fields
        required_fields = ['word_count', 'character_count', 'sentence_count', 'seo_score']
        for field in required_fields:
            self.assertIn(field, result)
        
        # Check SEO score range
        self.assertGreaterEqual(result['seo_score'], 0)
        self.assertLessEqual(result['seo_score'], 100)
        
        # Check word count
        self.assertEqual(result['word_count'], len(text.split()))
    
    def test_batch_processing(self):
        """Test batch text processing."""
        results = self.processor.batch_process(SAMPLE_TEXTS[:3])
        
        self.assertEqual(len(results), 3)
        
        for result in results:
            self.assertIn('seo_score', result)
            self.assertGreaterEqual(result['seo_score'], 0)
            self.assertLessEqual(result['seo_score'], 100)
    
    def test_caching_functionality(self):
        """Test caching functionality."""
        text = SAMPLE_TEXTS[0]
        
        # First call should cache miss
        result1 = self.processor.process(text)
        self.assertIn('seo_score', result1)
        
        # Second call should cache hit
        result2 = self.processor.process(text)
        self.assertEqual(result1['seo_score'], result2['seo_score'])
        
        # Check metrics
        metrics = self.processor.get_metrics()
        self.assertGreater(metrics['counters']['cache_hits'], 0)
    
    def test_error_handling(self):
        """Test error handling."""
        # Test with invalid input
        with self.assertRaises(ProcessingError):
            self.processor.process("")
        
        # Test with None input
        with self.assertRaises(ProcessingError):
            self.processor.process(None)
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        # Process some texts
        for text in SAMPLE_TEXTS[:2]:
            self.processor.process(text)
        
        metrics = self.processor.get_metrics()
        
        # Check that metrics are collected
        self.assertIn('counters', metrics)
        self.assertIn('processed_texts', metrics['counters'])
        self.assertGreater(metrics['counters']['processed_texts'], 0)

# ============================================================================
# ASYNC TESTS
# ============================================================================

class TestAsyncFunctionality(BaseTestCase):
    """Test async functionality."""
    
    def setUp(self):
        super().setUp()
        self.processor = EnhancedSEOProcessor(self.config)
    
    def tearDown(self):
        """Clean up processor."""
        if hasattr(self, 'processor'):
            self.processor.cleanup()
        super().tearDown()
    
    def test_async_single_processing(self):
        """Test async single text processing."""
        async def test_async():
            text = SAMPLE_TEXTS[0]
            result = await self.processor.process_async(text)
            
            self.assertIn('seo_score', result)
            self.assertGreaterEqual(result['seo_score'], 0)
            self.assertLessEqual(result['seo_score'], 100)
        
        asyncio.run(test_async())
    
    def test_async_batch_processing(self):
        """Test async batch processing."""
        async def test_async_batch():
            results = await self.processor.batch_process_async(SAMPLE_TEXTS[:3])
            
            self.assertEqual(len(results), 3)
            
            for result in results:
                self.assertIn('seo_score', result)
                self.assertGreaterEqual(result['seo_score'], 0)
                self.assertLessEqual(result['seo_score'], 100)
        
        asyncio.run(test_async_batch())
    
    def test_concurrent_processing(self):
        """Test concurrent processing with semaphore."""
        async def test_concurrent():
            # Create multiple concurrent requests
            tasks = []
            for i in range(5):
                text = f"Test text {i} for concurrent processing."
                task = self.processor.process_async(text)
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks)
            
            self.assertEqual(len(results), 5)
            
            for result in results:
                self.assertIn('seo_score', result)
        
        asyncio.run(test_concurrent())

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration(BaseTestCase):
    """Test full system integration."""
    
    def setUp(self):
        super().setUp()
        self.engine = EnhancedSEOEngine(self.config)
    
    def tearDown(self):
        """Clean up engine."""
        if hasattr(self, 'engine'):
            self.engine.cleanup()
        super().tearDown()
    
    def test_full_workflow(self):
        """Test complete workflow."""
        # Single text analysis
        result = self.engine.analyze_text(SAMPLE_TEXTS[0])
        self.assertIn('seo_score', result)
        
        # Batch analysis
        results = self.engine.analyze_texts(SAMPLE_TEXTS[:3])
        self.assertEqual(len(results), 3)
        
        # Get metrics
        metrics = self.engine.get_system_metrics()
        self.assertIn('processor_metrics', metrics)
        self.assertIn('engine_metrics', metrics)
        self.assertIn('system_info', metrics)
    
    def test_async_workflow(self):
        """Test async workflow."""
        async def test_async_workflow():
            # Async single analysis
            result = await self.engine.analyze_text_async(SAMPLE_TEXTS[0])
            self.assertIn('seo_score', result)
            
            # Async batch analysis
            results = await self.engine.analyze_texts_async(SAMPLE_TEXTS[:3])
            self.assertEqual(len(results), 3)
        
        asyncio.run(test_async_workflow())
    
    def test_performance_monitoring(self):
        """Test performance monitoring."""
        # Perform some operations
        for text in SAMPLE_TEXTS[:2]:
            self.engine.analyze_text(text)
        
        # Get metrics
        metrics = self.engine.get_system_metrics()
        
        # Check that timing metrics are collected
        engine_metrics = metrics['engine_metrics']
        self.assertIn('total_analysis_timings', engine_metrics)
        
        timing_stats = engine_metrics['total_analysis_timings']
        self.assertGreater(timing_stats['count'], 0)
        self.assertGreater(timing_stats['mean'], 0)

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance(BaseTestCase):
    """Test performance characteristics."""
    
    def setUp(self):
        super().setUp()
        self.engine = EnhancedSEOEngine(self.config)
    
    def tearDown(self):
        """Clean up engine."""
        if hasattr(self, 'engine'):
            self.engine.cleanup()
        super().tearDown()
    
    def test_processing_speed(self):
        """Test processing speed."""
        text = SAMPLE_TEXTS[0]
        
        # Measure processing time
        start_time = time.time()
        result = self.engine.analyze_text(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust as needed)
        self.assertLess(processing_time, 5.0)  # 5 seconds max
        self.assertIn('seo_score', result)
    
    def test_batch_processing_speed(self):
        """Test batch processing speed."""
        texts = SAMPLE_TEXTS * 2  # 10 texts
        
        # Measure batch processing time
        start_time = time.time()
        results = self.engine.analyze_texts(texts)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(processing_time, 10.0)  # 10 seconds max
        self.assertEqual(len(results), len(texts))
    
    def test_memory_usage(self):
        """Test memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process multiple texts
        for i in range(10):
            text = f"Test text {i} for memory usage testing. " * 10
            self.engine.analyze_text(text)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust as needed)
        self.assertLess(memory_increase, 500 * 1024 * 1024)  # 500MB max increase

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestInputValidator,
        TestLRUCache,
        TestCircuitBreaker,
        TestMetricsCollector,
        TestEnhancedSEOProcessor,
        TestAsyncFunctionality,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

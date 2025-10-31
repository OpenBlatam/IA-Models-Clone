#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Optimized SEO System
Unit tests, integration tests, and performance benchmarks
"""

import unittest
import asyncio
import time
import tempfile
import json
import yaml
import os
import sys
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Local imports
from core_config import (
    SEOConfig, SystemConfig, ModelConfig, PerformanceConfig, 
    MonitoringConfig, ConfigurationManager, DependencyContainer
)
from advanced_monitoring import (
    MonitoringSystem, MetricsCollector, PerformanceProfiler,
    AlertManager, MetricsVisualizer
)
from optimized_seo_engine import (
    OptimizedSEOEngine, AdvancedCacheManager, IntelligentModelManager,
    AdvancedSEOProcessor, KeywordAnalyzer, ContentAnalyzer,
    ReadabilityAnalyzer, TechnicalAnalyzer
)

warnings.filterwarnings("ignore")

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

class TestConfig:
    """Test configuration and utilities."""
    
    @staticmethod
    def create_test_config() -> SEOConfig:
        """Create a test configuration."""
        return SEOConfig(
            system=SystemConfig(
                debug=True,
                log_level="DEBUG",
                max_workers=2,
                temp_dir=tempfile.gettempdir()
            ),
            models=ModelConfig(
                default_model="microsoft/DialoGPT-medium",
                cache_enabled=True,
                cache_ttl=60
            ),
            performance=PerformanceConfig(
                batch_size=2,
                max_memory_usage=0.5,
                enable_mixed_precision=False,
                enable_compilation=False
            ),
            monitoring=MonitoringConfig(
                metrics_enabled=True,
                profiling_enabled=True,
                alerting_enabled=True
            )
        )
    
    @staticmethod
    def create_sample_texts() -> List[str]:
        """Create sample texts for testing."""
        return [
            """
            # SEO Optimization Guide
            
            This comprehensive guide covers the essential aspects of SEO optimization.
            We'll explore keyword research, content optimization, and technical SEO.
            
            ## Key Points
            - Keyword density optimization
            - Content quality improvement
            - Technical SEO best practices
            
            ## Conclusion
            Follow these guidelines for better search engine rankings.
            """,
            
            """
            # Machine Learning Basics
            
            Machine learning is a subset of artificial intelligence that enables
            computers to learn and improve from experience without being explicitly programmed.
            
            ## Types of ML
            - Supervised Learning
            - Unsupervised Learning
            - Reinforcement Learning
            
            This field continues to evolve rapidly.
            """,
            
            """
            # Web Development Fundamentals
            
            Web development involves creating websites and web applications.
            It encompasses frontend and backend development, database management,
            and server administration.
            
            ## Technologies
            - HTML, CSS, JavaScript
            - Python, Node.js, PHP
            - MySQL, MongoDB, PostgreSQL
            """
        ]

# ============================================================================
# UNIT TESTS - CORE CONFIGURATION
# ============================================================================

class TestCoreConfiguration(unittest.TestCase):
    """Test core configuration system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = TestConfig.create_test_config()
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise exceptions
        self.test_config.validate()
        
        # Test invalid configs
        invalid_config = SEOConfig()
        invalid_config.system.max_workers = 0
        with self.assertRaises(Exception):
            invalid_config.validate()
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        # Test to_dict
        config_dict = self.test_config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn('system', config_dict)
        self.assertIn('models', config_dict)
        
        # Test from_dict
        new_config = SEOConfig.from_dict(config_dict)
        self.assertEqual(new_config.system.max_workers, self.test_config.system.max_workers)
        self.assertEqual(new_config.models.default_model, self.test_config.models.default_model)
    
    def test_config_file_operations(self):
        """Test configuration file operations."""
        # Save config to file
        self.test_config.save_to_file(self.config_file)
        self.assertTrue(os.path.exists(self.config_file))
        
        # Load config from file
        loaded_config = SEOConfig.load_from_file(self.config_file)
        self.assertEqual(loaded_config.system.max_workers, self.test_config.system.max_workers)
    
    def test_config_manager(self):
        """Test configuration manager."""
        config_manager = ConfigurationManager()
        
        # Test get/set operations
        config_manager.set('test.key', 'test_value')
        self.assertEqual(config_manager.get('test.key'), 'test_value')
        
        # Test default values
        self.assertIsNone(config_manager.get('nonexistent.key'))
        self.assertEqual(config_manager.get('nonexistent.key', 'default'), 'default')
    
    def test_dependency_container(self):
        """Test dependency injection container."""
        container = DependencyContainer()
        
        # Test service registration
        container.register('test_service', 'test_value')
        self.assertTrue(container.has('test_service'))
        
        # Test service retrieval
        service = container.get('test_service')
        self.assertEqual(service, 'test_value')
        
        # Test singleton registration
        container.register_singleton('singleton', str, 'test')
        singleton1 = container.get('singleton')
        singleton2 = container.get('singleton')
        self.assertIs(singleton1, singleton2)

# ============================================================================
# UNIT TESTS - ADVANCED MONITORING
# ============================================================================

class TestAdvancedMonitoring(unittest.TestCase):
    """Test advanced monitoring system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitoring = MonitoringSystem()
        self.metrics_collector = MetricsCollector()
        self.profiler = PerformanceProfiler()
        self.alert_manager = AlertManager()
    
    def test_metrics_collector(self):
        """Test metrics collection."""
        # Test metric registration
        self.metrics_collector.register_metric('test_metric')
        self.assertIn('test_metric', self.metrics_collector.metrics)
        
        # Test metric recording
        self.metrics_collector.record('test_metric', 42.0)
        metric = self.metrics_collector.get_metric('test_metric')
        self.assertEqual(len(metric.points), 1)
        self.assertEqual(metric.points[0].value, 42.0)
        
        # Test batch recording
        self.metrics_collector.record_batch({'metric1': 1.0, 'metric2': 2.0})
        self.assertIn('metric1', self.metrics_collector.metrics)
        self.assertIn('metric2', self.metrics_collector.metrics)
    
    def test_metric_statistics(self):
        """Test metric statistics calculation."""
        metric = self.metrics_collector.metrics['test_metric']
        
        # Add multiple data points
        for i in range(10):
            metric.add_point(float(i))
        
        stats = metric.get_statistics()
        self.assertEqual(stats['count'], 10)
        self.assertEqual(stats['min'], 0.0)
        self.assertEqual(stats['max'], 9.0)
        self.assertAlmostEqual(stats['mean'], 4.5)
    
    def test_performance_profiler(self):
        """Test performance profiling."""
        # Test function profiling
        with self.profiler.profile_function('test_function'):
            time.sleep(0.01)  # Simulate work
        
        stats = self.profiler.get_profile_stats('test_function')
        self.assertIsNotNone(stats)
        self.assertEqual(stats['type'], 'cprofile')
        
        # Test memory tracing
        with self.profiler.memory_trace('test_memory'):
            _ = [i for i in range(1000)]  # Allocate memory
        
        trace = self.profiler.get_memory_trace('test_memory')
        self.assertIsNotNone(trace)
        self.assertIn('current_mb', trace)
        self.assertIn('peak_mb', trace)
    
    def test_alert_manager(self):
        """Test alert management."""
        # Test alert rule addition
        def test_condition(metrics):
            return True  # Always trigger
        
        self.alert_manager.add_rule('test_rule', test_condition, 'warning')
        self.assertEqual(len(self.alert_manager.rules), 1)
        
        # Test alert checking
        self.alert_manager.check_alerts(self.metrics_collector)
        alerts = self.alert_manager.get_recent_alerts()
        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0]['rule_name'], 'test_rule')

# ============================================================================
# UNIT TESTS - CACHE SYSTEM
# ============================================================================

class TestAdvancedCacheManager(unittest.TestCase):
    """Test advanced caching system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = AdvancedCacheManager(max_size=100, compression_threshold=50)
    
    def test_basic_operations(self):
        """Test basic cache operations."""
        # Test set and get
        self.cache.set('key1', 'value1')
        self.assertEqual(self.cache.get('key1'), 'value1')
        
        # Test cache miss
        self.assertIsNone(self.cache.get('nonexistent'))
        
        # Test invalidation
        self.cache.invalidate('key1')
        self.assertIsNone(self.cache.get('key1'))
    
    def test_size_management(self):
        """Test cache size management."""
        # Fill cache beyond capacity
        for i in range(150):
            self.cache.set(f'key{i}', f'value{i}')
        
        # Check that cache size is within limits
        stats = self.cache.get_stats()
        self.assertLessEqual(stats['total_items'], 100)
    
    def test_compression(self):
        """Test data compression."""
        # Large data should be compressed
        large_data = 'x' * 100
        self.cache.set('large_key', large_data)
        
        # Small data should not be compressed
        small_data = 'x' * 10
        self.cache.set('small_key', small_data)
        
        # Verify both can be retrieved
        self.assertEqual(self.cache.get('large_key'), large_data)
        self.assertEqual(self.cache.get('small_key'), small_data)

# ============================================================================
# UNIT TESTS - SEO ANALYSIS COMPONENTS
# ============================================================================

class TestSEOAnalysisComponents(unittest.TestCase):
    """Test SEO analysis components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_text = TestConfig.create_sample_texts()[0]
        self.keyword_analyzer = KeywordAnalyzer()
        self.content_analyzer = ContentAnalyzer()
        self.readability_analyzer = ReadabilityAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
    
    def test_keyword_analyzer(self):
        """Test keyword analysis."""
        result = self.keyword_analyzer.analyze(self.sample_text)
        
        self.assertIn('word_count', result)
        self.assertIn('unique_words', result)
        self.assertIn('keyword_density', result)
        self.assertIn('score', result)
        
        self.assertGreater(result['word_count'], 0)
        self.assertGreater(result['unique_words'], 0)
        self.assertGreaterEqual(result['score'], 0)
        self.assertLessEqual(result['score'], 100)
    
    def test_content_analyzer(self):
        """Test content analysis."""
        result = self.content_analyzer.analyze(self.sample_text)
        
        self.assertIn('sentence_count', result)
        self.assertIn('word_count', result)
        self.assertIn('avg_sentence_length', result)
        self.assertIn('has_headings', result)
        self.assertIn('has_lists', result)
        self.assertIn('score', result)
        
        self.assertGreater(result['sentence_count'], 0)
        self.assertTrue(result['has_headings'])  # Sample text has headings
        self.assertTrue(result['has_lists'])     # Sample text has lists
    
    def test_readability_analyzer(self):
        """Test readability analysis."""
        result = self.readability_analyzer.analyze(self.sample_text)
        
        self.assertIn('flesch_reading_ease', result)
        self.assertIn('sentence_count', result)
        self.assertIn('word_count', result)
        self.assertIn('syllable_count', result)
        self.assertIn('score', result)
        
        self.assertGreaterEqual(result['flesch_reading_ease'], 0)
        self.assertLessEqual(result['flesch_reading_ease'], 100)
        self.assertGreaterEqual(result['score'], 0)
        self.assertLessEqual(result['score'], 100)
    
    def test_technical_analyzer(self):
        """Test technical analysis."""
        result = self.technical_analyzer.analyze(self.sample_text)
        
        self.assertIn('has_meta_description', result)
        self.assertIn('has_title_tag', result)
        self.assertIn('has_alt_tags', result)
        self.assertIn('content_length', result)
        self.assertIn('score', result)
        
        self.assertGreater(result['content_length'], 0)
        self.assertGreaterEqual(result['score'], 0)
        self.assertLessEqual(result['score'], 100)

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSEOEngineIntegration(unittest.TestCase):
    """Test SEO engine integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TestConfig.create_test_config()
        self.engine = None
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.engine:
            self.engine.cleanup()
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        self.engine = OptimizedSEOEngine(self.config)
        
        # Check that all components are initialized
        self.assertIsNotNone(self.engine.model_manager)
        self.assertIsNotNone(self.engine.seo_processor)
        self.assertIsNotNone(self.engine.monitoring)
        
        # Check configuration
        self.assertEqual(self.engine.config.system.max_workers, 2)
        self.assertEqual(self.engine.config.models.cache_enabled, True)
    
    def test_text_analysis_integration(self):
        """Test complete text analysis workflow."""
        self.engine = OptimizedSEOEngine(self.config)
        sample_text = TestConfig.create_sample_texts()[0]
        
        # Perform analysis
        result = self.engine.analyze_text(sample_text)
        
        # Verify result structure
        self.assertIn('seo_score', result)
        self.assertIn('keyword_analysis', result)
        self.assertIn('content_analysis', result)
        self.assertIn('readability_analysis', result)
        self.assertIn('technical_analysis', result)
        self.assertIn('recommendations', result)
        self.assertIn('metadata', result)
        
        # Verify SEO score
        self.assertGreaterEqual(result['seo_score'], 0)
        self.assertLessEqual(result['seo_score'], 100)
        
        # Verify recommendations
        self.assertIsInstance(result['recommendations'], list)
    
    def test_batch_analysis_integration(self):
        """Test batch analysis workflow."""
        self.engine = OptimizedSEOEngine(self.config)
        sample_texts = TestConfig.create_sample_texts()
        
        # Perform batch analysis
        results = self.engine.analyze_texts(sample_texts)
        
        # Verify results
        self.assertEqual(len(results), len(sample_texts))
        
        for result in results:
            self.assertIn('seo_score', result)
            self.assertIn('metadata', result)
    
    def test_system_metrics_integration(self):
        """Test system metrics collection."""
        self.engine = OptimizedSEOEngine(self.config)
        
        # Wait for metrics to be collected
        time.sleep(3)
        
        # Get system metrics
        metrics = self.engine.get_system_metrics()
        
        # Verify metrics structure
        self.assertIn('system_health', metrics)
        self.assertIn('cache_stats', metrics)
        self.assertIn('model_info', metrics)
        self.assertIn('performance_stats', metrics)
        
        # Verify system health
        system_health = metrics['system_health']
        self.assertIn('status', system_health)
        self.assertIn('metrics', system_health)
        self.assertIn('alerts', system_health)

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance(unittest.TestCase):
    """Test system performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TestConfig.create_test_config()
        self.engine = None
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.engine:
            self.engine.cleanup()
    
    def test_analysis_performance(self):
        """Test analysis performance."""
        self.engine = OptimizedSEOEngine(self.config)
        sample_text = TestConfig.create_sample_texts()[0]
        
        # Measure analysis time
        start_time = time.time()
        result = self.engine.analyze_text(sample_text)
        analysis_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(analysis_time, 5.0)  # Should complete within 5 seconds
        self.assertIn('metadata', result)
        self.assertIn('analysis_time', result['metadata'])
        
        # Verify cached performance improvement
        start_time = time.time()
        cached_result = self.engine.analyze_text(sample_text)
        cached_time = time.time() - start_time
        
        # Cached result should be faster
        self.assertLess(cached_time, analysis_time)
    
    def test_batch_performance(self):
        """Test batch processing performance."""
        self.engine = OptimizedSEOEngine(self.config)
        sample_texts = TestConfig.create_sample_texts() * 5  # 15 texts
        
        # Measure batch processing time
        start_time = time.time()
        results = self.engine.analyze_texts(sample_texts)
        batch_time = time.time() - start_time
        
        # Performance assertions
        self.assertEqual(len(results), len(sample_texts))
        self.assertLess(batch_time, 30.0)  # Should complete within 30 seconds
        
        # Verify all results are valid
        for result in results:
            self.assertIn('seo_score', result)
            self.assertIn('metadata', result)
    
    def test_memory_usage(self):
        """Test memory usage optimization."""
        self.engine = OptimizedSEOEngine(self.config)
        
        # Get initial memory usage
        initial_memory = psutil.virtual_memory().percent
        
        # Perform multiple analyses
        sample_texts = TestConfig.create_sample_texts() * 10  # 30 texts
        
        for text in sample_texts:
            self.engine.analyze_text(text)
        
        # Get memory usage after processing
        final_memory = psutil.virtual_memory().percent
        
        # Memory usage should not increase dramatically
        memory_increase = final_memory - initial_memory
        self.assertLess(memory_increase, 20.0)  # Should not increase by more than 20%

# ============================================================================
# STRESS TESTS
# ============================================================================

class TestStress(unittest.TestCase):
    """Test system under stress."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TestConfig.create_test_config()
        self.engine = None
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.engine:
            self.engine.cleanup()
    
    def test_concurrent_requests(self):
        """Test concurrent request handling."""
        self.engine = OptimizedSEOEngine(self.config)
        sample_text = TestConfig.create_sample_texts()[0]
        
        def analyze_text():
            return self.engine.analyze_text(sample_text)
        
        # Create multiple threads
        threads = []
        results = []
        
        for i in range(10):
            thread = threading.Thread(target=lambda: results.append(analyze_text()))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all requests completed
        self.assertEqual(len(results), 10)
        
        for result in results:
            self.assertIn('seo_score', result)
            self.assertIn('metadata', result)
    
    def test_large_text_handling(self):
        """Test handling of very large texts."""
        self.engine = OptimizedSEOEngine(self.config)
        
        # Create very large text
        large_text = "This is a test sentence. " * 10000  # ~300k characters
        
        # Should handle without crashing
        start_time = time.time()
        result = self.engine.analyze_text(large_text)
        analysis_time = time.time() - start_time
        
        # Verify result
        self.assertIn('seo_score', result)
        self.assertIn('metadata', result)
        
        # Should complete within reasonable time
        self.assertLess(analysis_time, 30.0)
    
    def test_rapid_requests(self):
        """Test rapid successive requests."""
        self.engine = OptimizedSEOEngine(self.config)
        sample_text = TestConfig.create_sample_texts()[0]
        
        # Make rapid requests
        start_time = time.time()
        for i in range(50):
            result = self.engine.analyze_text(sample_text)
            self.assertIn('seo_score', result)
        
        total_time = time.time() - start_time
        
        # Should handle rapid requests efficiently
        self.assertLess(total_time, 60.0)  # Should complete within 60 seconds

# ============================================================================
# TEST RUNNER
# ============================================================================

def run_performance_benchmarks():
    """Run performance benchmarks."""
    print("\n" + "="*60)
    print("🚀 PERFORMANCE BENCHMARKS")
    print("="*60)
    
    config = TestConfig.create_test_config()
    engine = OptimizedSEOEngine(config)
    
    try:
        sample_texts = TestConfig.create_sample_texts() * 5  # 15 texts
        
        # Single text analysis benchmark
        print("\n📊 Single Text Analysis Benchmark:")
        start_time = time.time()
        result = engine.analyze_text(sample_texts[0])
        single_time = time.time() - start_time
        print(f"   Time: {single_time:.3f}s")
        print(f"   SEO Score: {result['seo_score']:.1f}")
        
        # Batch analysis benchmark
        print("\n📊 Batch Analysis Benchmark:")
        start_time = time.time()
        results = engine.analyze_texts(sample_texts)
        batch_time = time.time() - start_time
        print(f"   Time: {batch_time:.3f}s")
        print(f"   Texts processed: {len(results)}")
        print(f"   Average time per text: {batch_time/len(results):.3f}s")
        
        # System metrics
        print("\n📊 System Metrics:")
        metrics = engine.get_system_metrics()
        system_health = metrics['system_health']
        print(f"   System Status: {system_health['status']}")
        print(f"   Cache Stats: {metrics['cache_stats']['total_items']} items")
        print(f"   Models Loaded: {metrics['model_info']['total_models']}")
        
    finally:
        engine.cleanup()

def main():
    """Main test runner."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestCoreConfiguration,
        TestAdvancedMonitoring,
        TestAdvancedCacheManager,
        TestSEOAnalysisComponents,
        TestSEOEngineIntegration,
        TestPerformance,
        TestStress
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    print("🧪 Running Comprehensive Test Suite...")
    print("="*60)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Run performance benchmarks if tests pass
    if result.wasSuccessful():
        run_performance_benchmarks()
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



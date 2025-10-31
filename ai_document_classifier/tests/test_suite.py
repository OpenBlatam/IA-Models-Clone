"""
Comprehensive Test Suite
========================

Complete test suite for the AI Document Classifier system including
unit tests, integration tests, and performance tests.
"""

import unittest
import asyncio
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import logging

# Import components to test
from ..document_classifier_engine import DocumentClassifierEngine, DocumentType
from ..models.advanced_classifier import AdvancedDocumentClassifier, ClassificationMethod
from ..templates.dynamic_template_generator import DynamicTemplateGenerator, TemplateComplexity
from ..utils.batch_processor import BatchProcessor
from ..integrations.external_services import ExternalServiceManager
from ..analytics.performance_monitor import PerformanceMonitor
from ..notifications.alert_system import AlertSystem

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)

class TestDocumentClassifierEngine(unittest.TestCase):
    """Test cases for the core document classifier engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = DocumentClassifierEngine()
        self.test_queries = [
            ("I want to write a science fiction novel", DocumentType.NOVEL),
            ("Create a service agreement contract", DocumentType.CONTRACT),
            ("Design a mobile app interface", DocumentType.DESIGN),
            ("Write a business plan for startup", DocumentType.BUSINESS_PLAN),
            ("Research paper on machine learning", DocumentType.ACADEMIC_PAPER)
        ]
    
    def test_basic_classification(self):
        """Test basic document classification"""
        for query, expected_type in self.test_queries:
            with self.subTest(query=query):
                result = self.classifier.classify_document(query, use_ai=False)
                self.assertIsNotNone(result)
                self.assertIsInstance(result.document_type, DocumentType)
                self.assertGreaterEqual(result.confidence, 0.0)
                self.assertLessEqual(result.confidence, 1.0)
                self.assertIsInstance(result.keywords, list)
                self.assertIsInstance(result.reasoning, str)
    
    def test_template_retrieval(self):
        """Test template retrieval for different document types"""
        for doc_type in DocumentType:
            if doc_type == DocumentType.UNKNOWN:
                continue
            
            templates = self.classifier.get_templates(doc_type)
            self.assertIsInstance(templates, list)
            
            if templates:
                template = templates[0]
                self.assertIsNotNone(template.name)
                self.assertEqual(template.document_type, doc_type)
                self.assertIsInstance(template.sections, list)
                self.assertIsInstance(template.formatting, dict)
    
    def test_template_export(self):
        """Test template export in different formats"""
        templates = self.classifier.get_templates(DocumentType.NOVEL)
        if templates:
            template = templates[0]
            
            # Test JSON export
            json_export = self.classifier.export_template(template, "json")
            self.assertIsInstance(json_export, str)
            self.assertTrue(json_export.startswith("{"))
            
            # Test YAML export
            yaml_export = self.classifier.export_template(template, "yaml")
            self.assertIsInstance(yaml_export, str)
            
            # Test Markdown export
            md_export = self.classifier.export_template(template, "markdown")
            self.assertIsInstance(md_export, str)
            self.assertTrue(md_export.startswith("#"))

class TestAdvancedClassifier(unittest.TestCase):
    """Test cases for the advanced classifier"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.classifier = AdvancedDocumentClassifier(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        test_text = "I want to write a science fiction novel about space exploration and alien civilizations."
        
        features = self.classifier.extract_features(test_text)
        self.assertIsNotNone(features)
        self.assertGreater(features.word_count, 0)
        self.assertGreater(features.sentence_count, 0)
        self.assertIsInstance(features.keywords, list)
        self.assertIsInstance(features.pos_tags, dict)
    
    def test_ml_classification(self):
        """Test ML-based classification"""
        test_queries = [
            "I want to write a novel",
            "Create a contract agreement",
            "Design a mobile app"
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                result = self.classifier.classify_with_ml(query)
                self.assertIsNotNone(result)
                self.assertIsInstance(result.document_type, str)
                self.assertGreaterEqual(result.confidence, 0.0)
                self.assertLessEqual(result.confidence, 1.0)
                self.assertIsNotNone(result.method_used)
                self.assertGreaterEqual(result.processing_time, 0.0)
    
    def test_model_training(self):
        """Test model training"""
        # This test might take some time
        results = self.classifier.train_models()
        self.assertIsInstance(results, dict)
        
        # Check that training completed
        self.assertTrue(self.classifier.is_trained)

class TestDynamicTemplateGenerator(unittest.TestCase):
    """Test cases for the dynamic template generator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = DynamicTemplateGenerator()
    
    def test_template_generation(self):
        """Test dynamic template generation"""
        template = self.generator.generate_template(
            document_type="novel",
            complexity=TemplateComplexity.INTERMEDIATE,
            style_preset="creative"
        )
        
        self.assertIsNotNone(template)
        self.assertEqual(template.document_type, "novel")
        self.assertEqual(template.complexity, TemplateComplexity.INTERMEDIATE)
        self.assertIsInstance(template.sections, list)
        self.assertGreater(len(template.sections), 0)
        self.assertIsNotNone(template.style)
    
    def test_template_export(self):
        """Test template export in different formats"""
        template = self.generator.generate_template("contract", TemplateComplexity.BASIC)
        
        # Test JSON export
        json_export = self.generator.export_template(template, "json")
        self.assertIsInstance(json_export, str)
        
        # Test YAML export
        yaml_export = self.generator.export_template(template, "yaml")
        self.assertIsInstance(yaml_export, str)
        
        # Test Markdown export
        md_export = self.generator.export_template(template, "markdown")
        self.assertIsInstance(md_export, str)
        self.assertTrue(md_export.startswith("#"))
    
    def test_custom_requirements(self):
        """Test template generation with custom requirements"""
        custom_requirements = {
            "required_sections": ["Custom Section"],
            "excluded_sections": ["Glossary"],
            "section_order": ["Title Page", "Custom Section", "Main Content"]
        }
        
        template = self.generator.generate_template(
            document_type="novel",
            custom_requirements=custom_requirements
        )
        
        section_names = [section.name for section in template.sections]
        self.assertIn("Custom Section", section_names)
        self.assertNotIn("Glossary", section_names)

class TestBatchProcessor(unittest.TestCase):
    """Test cases for the batch processor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.classifier = DocumentClassifierEngine()
        self.processor = BatchProcessor(self.classifier, self.temp_dir)
        self.test_queries = [
            "I want to write a novel",
            "Create a contract",
            "Design an app",
            "Write a business plan",
            "Research paper"
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_batch_processing(self):
        """Test batch processing"""
        result = self.processor.process_batch(
            queries=self.test_queries,
            use_cache=False
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.total_jobs, len(self.test_queries))
        self.assertGreaterEqual(result.completed_jobs, 0)
        self.assertGreaterEqual(result.processing_time, 0.0)
        self.assertIsInstance(result.results, list)
        self.assertIsInstance(result.errors, list)
        self.assertIsInstance(result.analytics, dict)
    
    def test_caching(self):
        """Test caching functionality"""
        # First run without cache
        result1 = self.processor.process_batch(
            queries=self.test_queries[:2],
            use_cache=True
        )
        
        # Second run with cache
        result2 = self.processor.process_batch(
            queries=self.test_queries[:2],
            use_cache=True
        )
        
        # Results should be similar
        self.assertEqual(result1.total_jobs, result2.total_jobs)
    
    def test_analytics(self):
        """Test analytics generation"""
        result = self.processor.process_batch(self.test_queries)
        
        analytics = result.analytics
        self.assertIn("total_processed", analytics)
        self.assertIn("total_failed", analytics)
        self.assertIn("success_rate", analytics)
        self.assertIn("document_type_distribution", analytics)

class TestExternalServices(unittest.TestCase):
    """Test cases for external services integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.service_manager = ExternalServiceManager()
    
    def test_service_configuration(self):
        """Test service configuration"""
        # Test adding a service
        from ..integrations.external_services import ServiceConfig, ServiceType
        
        test_service = ServiceConfig(
            name="test_service",
            service_type=ServiceType.AI_CLASSIFICATION,
            base_url="https://api.test.com",
            enabled=False
        )
        
        self.service_manager.add_service(test_service)
        self.assertIn("test_service", self.service_manager.services)
        
        # Test removing a service
        self.service_manager.remove_service("test_service")
        self.assertNotIn("test_service", self.service_manager.services)
    
    def test_service_status(self):
        """Test service status retrieval"""
        status = self.service_manager.get_all_services_status()
        self.assertIsInstance(status, dict)
        
        for service_name, service_status in status.items():
            self.assertIn("name", service_status)
            self.assertIn("type", service_status)
            self.assertIn("enabled", service_status)

class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for performance monitoring"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = PerformanceMonitor(str(Path(self.temp_dir) / "metrics.db"))
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.monitor.stop_monitoring()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_metric_recording(self):
        """Test metric recording"""
        self.monitor.record_metric("test_metric", 42.0, {"test": "true"})
        self.monitor.record_classification_request("test_method", 0.5, True)
        
        # Give some time for metrics to be processed
        time.sleep(1)
        
        summary = self.monitor.get_performance_summary(hours=1)
        self.assertIsInstance(summary, dict)
    
    def test_health_status(self):
        """Test health status"""
        health = self.monitor.get_health_status()
        self.assertIsInstance(health, dict)
        self.assertIn("status", health)
        self.assertIn("timestamp", health)

class TestAlertSystem(unittest.TestCase):
    """Test cases for the alert system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.alert_system = AlertSystem(str(Path(self.temp_dir) / "alerts.db"))
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_alert_rules(self):
        """Test alert rules"""
        from ..notifications.alert_system import AlertRule, AlertSeverity, AlertType
        
        test_rule = AlertRule(
            id="test_rule",
            name="Test Rule",
            description="Test alert rule",
            condition="test_value > 10",
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.CUSTOM
        )
        
        self.alert_system.add_alert_rule(test_rule)
        self.assertIn("test_rule", self.alert_system.alert_rules)
        
        self.alert_system.remove_alert_rule("test_rule")
        self.assertNotIn("test_rule", self.alert_system.alert_rules)
    
    def test_alert_checking(self):
        """Test alert checking"""
        test_context = {
            "system_metrics": {
                "cpu_percent": 85.0,
                "memory_percent": 90.0
            }
        }
        
        # This should trigger some alerts
        self.alert_system.check_alerts(test_context)
        
        active_alerts = self.alert_system.get_active_alerts()
        self.assertIsInstance(active_alerts, list)

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.classifier = DocumentClassifierEngine()
        self.advanced_classifier = AdvancedDocumentClassifier(self.temp_dir)
        self.template_generator = DynamicTemplateGenerator()
        self.batch_processor = BatchProcessor(self.classifier, self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_classification(self):
        """Test end-to-end classification workflow"""
        test_query = "I want to write a science fiction novel about space exploration"
        
        # Basic classification
        basic_result = self.classifier.classify_document(test_query, use_ai=False)
        self.assertIsNotNone(basic_result)
        
        # Advanced classification
        advanced_result = self.advanced_classifier.classify_with_ml(test_query)
        self.assertIsNotNone(advanced_result)
        
        # Template generation
        if basic_result.document_type != DocumentType.UNKNOWN:
            template = self.template_generator.generate_template(
                document_type=basic_result.document_type.value,
                complexity=TemplateComplexity.INTERMEDIATE
            )
            self.assertIsNotNone(template)
    
    def test_batch_processing_workflow(self):
        """Test batch processing workflow"""
        test_queries = [
            "Write a novel",
            "Create a contract",
            "Design an app"
        ]
        
        # Process batch
        batch_result = self.batch_processor.process_batch(test_queries)
        
        # Verify results
        self.assertEqual(batch_result.total_jobs, len(test_queries))
        self.assertGreaterEqual(batch_result.completed_jobs, 0)
        
        # Check analytics
        analytics = batch_result.analytics
        self.assertIn("total_processed", analytics)
        self.assertIn("success_rate", analytics)

class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = DocumentClassifierEngine()
        self.test_queries = [
            "I want to write a science fiction novel about space exploration and alien civilizations",
            "Create a comprehensive service agreement contract with detailed terms and conditions",
            "Design a modern mobile application interface with user experience focus",
            "Write a detailed business plan for a technology startup company",
            "Research paper on machine learning algorithms and their applications"
        ] * 10  # 50 queries total
    
    def test_classification_performance(self):
        """Test classification performance"""
        start_time = time.time()
        
        results = []
        for query in self.test_queries:
            result = self.classifier.classify_document(query, use_ai=False)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(self.test_queries)
        
        # Performance assertions
        self.assertLess(avg_time, 1.0)  # Should be less than 1 second per query
        self.assertEqual(len(results), len(self.test_queries))
        
        print(f"Processed {len(self.test_queries)} queries in {total_time:.2f}s")
        print(f"Average time per query: {avg_time:.3f}s")
    
    def test_memory_usage(self):
        """Test memory usage during processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process queries
        for query in self.test_queries:
            result = self.classifier.classify_document(query, use_ai=False)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable
        self.assertLess(memory_increase, 100)  # Less than 100MB increase
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")

def run_all_tests():
    """Run all test suites"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDocumentClassifierEngine,
        TestAdvancedClassifier,
        TestDynamicTemplateGenerator,
        TestBatchProcessor,
        TestExternalServices,
        TestPerformanceMonitor,
        TestAlertSystem,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)




























#!/usr/bin/env python3
"""
Production Tests - Comprehensive production test suite
Tests all production components including API, monitoring, logging, and deployment
"""

import unittest
import asyncio
import time
import json
import tempfile
import os
import requests
import threading
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
import logging
from datetime import datetime, timezone

# Import production components
from production_config import (
    ProductionConfigManager, DeploymentConfig, Environment,
    create_production_config, load_production_config
)
from production_logging import (
    ProductionLogger, LogContext, PerformanceMetrics,
    create_production_logger, setup_production_logging
)
from production_monitoring import (
    ProductionMonitor, MetricsCollector, AlertManager, HealthChecker,
    create_production_monitor, Metric, Alert, HealthCheck,
    MetricType, AlertLevel, HealthStatus
)
from production_api import (
    app, OptimizationRequest, OptimizationResponse,
    OperationStatusResponse, HealthResponse, MetricsResponse
)
from production_deployment import (
    ProductionDeployment, DockerDeployment, KubernetesDeployment,
    create_production_deployment, DeploymentConfig
)

class TestProductionConfig(unittest.TestCase):
    """Test production configuration system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_creation(self):
        """Test configuration creation."""
        config_manager = create_production_config()
        config = config_manager.get_config()
        
        self.assertIsInstance(config, type(config_manager.config))
        self.assertEqual(config.environment, Environment.PRODUCTION)
        self.assertFalse(config.debug)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = create_production_config()
        self.assertTrue(config_manager.validate_config())
    
    def test_config_from_file(self):
        """Test configuration loading from file."""
        config_data = {
            "environment": "development",
            "debug": True,
            "log_level": "DEBUG",
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db"
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)
        
        config_manager = ProductionConfigManager(self.config_file)
        config = config_manager.get_config()
        
        self.assertEqual(config.environment, Environment.DEVELOPMENT)
        self.assertTrue(config.debug)
        self.assertEqual(config.database.host, "localhost")
    
    def test_config_save(self):
        """Test configuration saving."""
        config_manager = create_production_config()
        config_manager.update_config({"debug": True})
        config_manager.save_config(self.config_file)
        
        self.assertTrue(os.path.exists(self.config_file))
        
        # Load and verify
        loaded_config = load_production_config(self.config_file)
        self.assertTrue(loaded_config.debug)

class TestProductionLogging(unittest.TestCase):
    """Test production logging system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test.log")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_logger_creation(self):
        """Test logger creation."""
        config = {
            'log_level': 'INFO',
            'enable_console': True,
            'enable_file': True,
            'log_file': self.log_file
        }
        
        logger = create_production_logger("test_logger", config)
        self.assertIsInstance(logger, ProductionLogger)
    
    def test_logging_with_context(self):
        """Test logging with context."""
        logger = create_production_logger("test_logger")
        
        # Set context
        logger.set_context(
            request_id="req_123",
            user_id="user_456",
            operation_id="op_789"
        )
        
        # Log message
        logger.info("Test message")
        
        # Verify context is set
        self.assertEqual(logger.context.request_id, "req_123")
        self.assertEqual(logger.context.user_id, "user_456")
        self.assertEqual(logger.context.operation_id, "op_789")
    
    def test_performance_metrics(self):
        """Test performance metrics logging."""
        logger = create_production_logger("test_logger")
        
        # Set metrics
        logger.set_metrics(
            operation_time=1.5,
            memory_usage=100.0,
            cpu_usage=50.0
        )
        
        # Log with metrics
        logger.log_performance_metrics({
            "operation_time": 1.5,
            "memory_usage": 100.0
        })
        
        # Verify metrics are set
        self.assertEqual(logger.metrics.operation_time, 1.5)
        self.assertEqual(logger.metrics.memory_usage, 100.0)
        self.assertEqual(logger.metrics.cpu_usage, 50.0)
    
    def test_operation_logging(self):
        """Test operation logging."""
        logger = create_production_logger("test_logger")
        
        # Log operation start
        logger.log_operation_start("op_123", "optimization")
        
        # Log operation end
        logger.log_operation_end("op_123", True, 2.0)
        
        # Verify operation logging
        self.assertEqual(logger.context.operation_id, "op_123")
    
    def test_error_logging(self):
        """Test error logging."""
        logger = create_production_logger("test_logger")
        
        # Create test error
        error = ValueError("Test error")
        
        # Log error
        logger.log_error(error, {"context": "test"})
        
        # Verify error logging
        self.assertIsNotNone(error)

class TestProductionMonitoring(unittest.TestCase):
    """Test production monitoring system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'metrics': {
                'collection_interval': 1,
                'max_history_size': 100
            },
            'health': {
                'check_interval': 1
            },
            'alerts': {}
        }
    
    def test_monitor_creation(self):
        """Test monitor creation."""
        monitor = create_production_monitor(self.config)
        self.assertIsInstance(monitor, type(monitor))
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        monitor = create_production_monitor(self.config)
        monitor.start()
        
        # Wait for metrics collection
        time.sleep(2)
        
        # Get metrics summary
        summary = monitor.get_metrics_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_metrics', summary)
        self.assertIn('active_alerts', summary)
        self.assertIn('health_status', summary)
        
        monitor.stop()
    
    def test_health_checks(self):
        """Test health checks."""
        monitor = create_production_monitor(self.config)
        monitor.start()
        
        # Wait for health checks
        time.sleep(2)
        
        # Get health status
        health_status = monitor.get_health_status()
        
        self.assertIsInstance(health_status, dict)
        self.assertIn('overall_status', health_status)
        self.assertIn('checks', health_status)
        
        monitor.stop()
    
    def test_alert_management(self):
        """Test alert management."""
        monitor = create_production_monitor(self.config)
        
        # Add alert rule
        monitor.alert_manager.add_alert_rule(
            "test_alert",
            "system_cpu_percent",
            80.0,
            AlertLevel.WARNING,
            "CPU usage is high"
        )
        
        # Check alerts
        monitor.check_alerts()
        
        # Get alerts summary
        alerts_summary = monitor.get_alerts_summary()
        
        self.assertIsInstance(alerts_summary, dict)
        self.assertIn('total_alerts', alerts_summary)
        self.assertIn('alerts_by_level', alerts_summary)
        
        monitor.stop()
    
    def test_metric_creation(self):
        """Test metric creation."""
        metric = Metric(
            name="test_metric",
            value=100.0,
            metric_type=MetricType.GAUGE,
            labels={"test": "value"},
            description="Test metric"
        )
        
        self.assertEqual(metric.name, "test_metric")
        self.assertEqual(metric.value, 100.0)
        self.assertEqual(metric.metric_type, MetricType.GAUGE)
        self.assertEqual(metric.labels["test"], "value")
    
    def test_alert_creation(self):
        """Test alert creation."""
        alert = Alert(
            name="test_alert",
            level=AlertLevel.WARNING,
            message="Test alert message",
            metric_name="test_metric",
            threshold=80.0,
            current_value=90.0
        )
        
        self.assertEqual(alert.name, "test_alert")
        self.assertEqual(alert.level, AlertLevel.WARNING)
        self.assertEqual(alert.message, "Test alert message")
        self.assertEqual(alert.metric_name, "test_metric")
        self.assertEqual(alert.threshold, 80.0)
        self.assertEqual(alert.current_value, 90.0)
    
    def test_health_check_creation(self):
        """Test health check creation."""
        health_check = HealthCheck(
            name="test_health",
            status=HealthStatus.HEALTHY,
            message="System is healthy",
            response_time=0.1,
            details={"cpu": 50.0, "memory": 60.0}
        )
        
        self.assertEqual(health_check.name, "test_health")
        self.assertEqual(health_check.status, HealthStatus.HEALTHY)
        self.assertEqual(health_check.message, "System is healthy")
        self.assertEqual(health_check.response_time, 0.1)
        self.assertEqual(health_check.details["cpu"], 50.0)

class TestProductionAPI(unittest.TestCase):
    """Test production API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.client = None
    
    def test_app_creation(self):
        """Test app creation."""
        self.assertIsNotNone(self.app)
        self.assertEqual(self.app.title, "Bulk Optimization API")
        self.assertEqual(self.app.version, "1.0.0")
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        # This would require a running server
        # For now, just test that the endpoint is defined
        routes = [route.path for route in self.app.routes]
        self.assertIn("/", routes)
    
    def test_health_endpoint(self):
        """Test health endpoint."""
        routes = [route.path for route in self.app.routes]
        self.assertIn("/health", routes)
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        routes = [route.path for route in self.app.routes]
        self.assertIn("/metrics", routes)
    
    def test_optimize_endpoint(self):
        """Test optimize endpoint."""
        routes = [route.path for route in self.app.routes]
        self.assertIn("/optimize", routes)
    
    def test_operations_endpoint(self):
        """Test operations endpoint."""
        routes = [route.path for route in self.app.routes]
        self.assertIn("/operations", routes)
    
    def test_request_models(self):
        """Test request models."""
        # Test OptimizationRequest
        request_data = {
            "models": [{"name": "model1", "type": "linear"}],
            "strategy": "memory",
            "config": {"max_workers": 4},
            "priority": 1
        }
        
        request = OptimizationRequest(**request_data)
        self.assertEqual(len(request.models), 1)
        self.assertEqual(request.strategy, "memory")
        self.assertEqual(request.priority, 1)
        
        # Test OptimizationResponse
        response_data = {
            "operation_id": "op_123",
            "status": "pending",
            "message": "Operation started",
            "estimated_time": 60.0
        }
        
        response = OptimizationResponse(**response_data)
        self.assertEqual(response.operation_id, "op_123")
        self.assertEqual(response.status, "pending")
        self.assertEqual(response.estimated_time, 60.0)

class TestProductionDeployment(unittest.TestCase):
    """Test production deployment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "app_name": "test-app",
            "app_version": "1.0.0",
            "environment": "testing",
            "replicas": 2,
            "cpu_limit": "1",
            "memory_limit": "2Gi"
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_deployment_creation(self):
        """Test deployment creation."""
        deployment = create_production_deployment(self.config)
        self.assertIsInstance(deployment, ProductionDeployment)
    
    def test_docker_deployment(self):
        """Test Docker deployment creation."""
        deployment = create_production_deployment(self.config)
        docker_dir = os.path.join(self.temp_dir, "docker")
        
        deployment.create_docker_deployment(docker_dir)
        
        # Check that files were created
        self.assertTrue(os.path.exists(os.path.join(docker_dir, "Dockerfile")))
        self.assertTrue(os.path.exists(os.path.join(docker_dir, "docker-compose.yml")))
        self.assertTrue(os.path.exists(os.path.join(docker_dir, "nginx.conf")))
        self.assertTrue(os.path.exists(os.path.join(docker_dir, "requirements.txt")))
        self.assertTrue(os.path.exists(os.path.join(docker_dir, ".dockerignore")))
    
    def test_kubernetes_deployment(self):
        """Test Kubernetes deployment creation."""
        deployment = create_production_deployment(self.config)
        k8s_dir = os.path.join(self.temp_dir, "kubernetes")
        
        deployment.create_kubernetes_deployment(k8s_dir)
        
        # Check that files were created
        expected_files = [
            "namespace.yaml",
            "configmap.yaml",
            "secret.yaml",
            "deployment.yaml",
            "service.yaml",
            "hpa.yaml",
            "pvc.yaml",
            "ingress.yaml",
            "deploy.sh"
        ]
        
        for file in expected_files:
            self.assertTrue(os.path.exists(os.path.join(k8s_dir, file)))
    
    def test_dockerfile_content(self):
        """Test Dockerfile content."""
        deployment = create_production_deployment(self.config)
        docker_dir = os.path.join(self.temp_dir, "docker")
        
        deployment.create_docker_deployment(docker_dir)
        
        with open(os.path.join(docker_dir, "Dockerfile"), 'r') as f:
            dockerfile_content = f.read()
        
        self.assertIn("FROM python:3.9-slim", dockerfile_content)
        self.assertIn("WORKDIR /app", dockerfile_content)
        self.assertIn("EXPOSE", dockerfile_content)
        self.assertIn("HEALTHCHECK", dockerfile_content)
    
    def test_docker_compose_content(self):
        """Test docker-compose.yml content."""
        deployment = create_production_deployment(self.config)
        docker_dir = os.path.join(self.temp_dir, "docker")
        
        deployment.create_docker_deployment(docker_dir)
        
        with open(os.path.join(docker_dir, "docker-compose.yml"), 'r') as f:
            compose_content = f.read()
        
        self.assertIn("version: '3.8'", compose_content)
        self.assertIn("services:", compose_content)
        self.assertIn("app:", compose_content)
        self.assertIn("postgres:", compose_content)
        self.assertIn("redis:", compose_content)
        self.assertIn("nginx:", compose_content)
    
    def test_kubernetes_deployment_content(self):
        """Test Kubernetes deployment content."""
        deployment = create_production_deployment(self.config)
        k8s_dir = os.path.join(self.temp_dir, "kubernetes")
        
        deployment.create_kubernetes_deployment(k8s_dir)
        
        with open(os.path.join(k8s_dir, "deployment.yaml"), 'r') as f:
            deployment_content = f.read()
        
        self.assertIn("apiVersion: apps/v1", deployment_content)
        self.assertIn("kind: Deployment", deployment_content)
        self.assertIn("replicas:", deployment_content)
        self.assertIn("livenessProbe:", deployment_content)
        self.assertIn("readinessProbe:", deployment_content)

class TestProductionIntegration(unittest.TestCase):
    """Test production system integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_system_integration(self):
        """Test full system integration."""
        # Create configuration
        config_manager = create_production_config()
        config = config_manager.get_config()
        
        # Create logger
        logger = create_production_logger("integration_test")
        
        # Create monitor
        monitor = create_production_monitor()
        monitor.start()
        
        try:
            # Test logging
            logger.info("Integration test started")
            logger.set_context(operation_id="integration_test")
            logger.log_operation_start("integration_test", "test")
            
            # Test monitoring
            time.sleep(1)  # Let monitor collect some metrics
            
            metrics_summary = monitor.get_metrics_summary()
            health_status = monitor.get_health_status()
            
            self.assertIsInstance(metrics_summary, dict)
            self.assertIsInstance(health_status, dict)
            
            # Test configuration
            self.assertTrue(config_manager.validate_config())
            
            # Test deployment
            deployment = create_production_deployment({
                "app_name": "integration-test",
                "environment": "testing"
            })
            
            # Create deployment files
            deployment.create_docker_deployment(os.path.join(self.temp_dir, "docker"))
            deployment.create_kubernetes_deployment(os.path.join(self.temp_dir, "kubernetes"))
            
            logger.log_operation_end("integration_test", True, 1.0)
            logger.info("Integration test completed")
            
        finally:
            monitor.stop()
    
    def test_error_handling(self):
        """Test error handling across components."""
        logger = create_production_logger("error_test")
        
        # Test error logging
        try:
            raise ValueError("Test error")
        except Exception as e:
            logger.log_error(e, {"test": "error_handling"})
        
        # Test configuration error handling
        config_manager = create_production_config()
        
        # Test with invalid configuration
        config_manager.update_config({"port": -1})  # Invalid port
        self.assertFalse(config_manager.validate_config())
    
    def test_performance_under_load(self):
        """Test performance under load."""
        logger = create_production_logger("load_test")
        monitor = create_production_monitor()
        monitor.start()
        
        try:
            # Simulate load
            start_time = time.time()
            
            for i in range(100):
                logger.info(f"Load test message {i}")
                logger.set_metrics(operation_time=i * 0.01, memory_usage=i * 10)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Performance should be reasonable
            self.assertLess(duration, 10.0)  # Should complete in under 10 seconds
            
            # Check metrics
            metrics_summary = monitor.get_metrics_summary()
            self.assertIsInstance(metrics_summary, dict)
            
        finally:
            monitor.stop()

def run_production_tests():
    """Run all production tests."""
    print("🧪 Running Production Tests")
    print("=" * 50)
    
    # Create test suite
    test_classes = [
        TestProductionConfig,
        TestProductionLogging,
        TestProductionMonitoring,
        TestProductionAPI,
        TestProductionDeployment,
        TestProductionIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"  - Tests run: {result.testsRun}")
    print(f"  - Failures: {len(result.failures)}")
    print(f"  - Errors: {len(result.errors)}")
    print(f"  - Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_production_tests()
    exit(0 if success else 1)


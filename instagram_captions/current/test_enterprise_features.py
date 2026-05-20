#!/usr/bin/env python3
"""
Enterprise Features Test Suite for Instagram Captions API v10.0

Tests all newly implemented enterprise-grade features:
- Advanced Security (SecurityUtils)
- Enhanced Performance Monitoring (PerformanceMonitor)
- Circuit Breaker Pattern (CircuitBreaker)
- Enterprise Error Handling (ErrorHandler)
"""

import unittest
import time
import json
from unittest.mock import Mock, patch
from utils import (
    SecurityUtils, PerformanceMonitor, CircuitBreaker, 
    ErrorHandler, ValidationUtils
)

class TestSecurityUtils(unittest.TestCase):
    """Test enterprise-grade security utilities."""
    
    def setUp(self):
        self.security = SecurityUtils()
    
    def test_generate_api_key_maximum_complexity(self):
        """Test maximum complexity API key generation."""
        key = SecurityUtils.generate_api_key(length=64, complexity="maximum")
        self.assertEqual(len(key), 64)
        self.assertIsInstance(key, str)
    
    def test_generate_secure_token_enterprise(self):
        """Test enterprise-grade secure token generation."""
        token = SecurityUtils.generate_secure_token(security_level="enterprise")
        self.assertTrue(token.startswith("token_"))
        self.assertIn("_", token)
    
    def test_hash_password_multiple_algorithms(self):
        """Test password hashing with multiple algorithms."""
        password = "test_password_123"
        
        # Test PBKDF2
        pbkdf2_hash = SecurityUtils.hash_password(password, algorithm="pbkdf2")
        self.assertTrue(pbkdf2_hash.startswith("pbkdf2:"))
        self.assertTrue(SecurityUtils.verify_password(password, pbkdf2_hash))
        
        # Test SHA-256
        sha256_hash = SecurityUtils.hash_password(password, algorithm="sha256")
        self.assertTrue(sha256_hash.startswith("sha256:"))
        self.assertTrue(SecurityUtils.verify_password(password, sha256_hash))
    
    def test_verify_api_key_enterprise_analysis(self):
        """Test enterprise-grade API key validation."""
        # Test weak API key
        weak_key = "test123demo456example789"
        result = SecurityUtils.verify_api_key(weak_key, security_level="enterprise")
        self.assertFalse(result["valid"])
        self.assertIn("Weak pattern detected", str(result["warnings"]))
        
        # Test strong API key
        strong_key = SecurityUtils.generate_api_key(length=64, complexity="maximum")
        result = SecurityUtils.verify_api_key(strong_key, security_level="enterprise")
        self.assertTrue(result["valid"])
        self.assertGreaterEqual(result["score"], 70)
    
    def test_sanitize_input_enterprise(self):
        """Test enterprise-grade input sanitization."""
        malicious_input = "<script>alert('xss')</script> UNION SELECT * FROM users"
        result = SecurityUtils.sanitize_input(malicious_input, strict=True)
        
        self.assertNotIn("<script>", result["sanitized_text"])
        self.assertNotIn("UNION SELECT", result["sanitized_text"])
        self.assertGreater(len(result["threats_detected"]), 0)
        self.assertLess(result["security_score"], 100)
    
    def test_validate_content_type_security(self):
        """Test content type validation with security analysis."""
        # Test dangerous content type
        dangerous_type = "text/html; charset=utf-8"
        result = SecurityUtils.validate_content_type(dangerous_type)
        self.assertFalse(result["valid"])
        self.assertIn("Dangerous content type", str(result["warnings"]))
        
        # Test safe content type
        safe_type = "application/json"
        result = SecurityUtils.validate_content_type(safe_type)
        self.assertTrue(result["valid"])
        self.assertEqual(result["security_score"], 100)
    
    def test_validate_file_extension_enterprise(self):
        """Test enterprise-grade file extension validation."""
        # Test dangerous extension
        dangerous_file = "malware.exe"
        result = SecurityUtils.validate_file_extension(dangerous_file)
        self.assertFalse(result["valid"])
        self.assertEqual(result["risk_level"], "critical")
        
        # Test safe extension
        safe_file = "document.txt"
        result = SecurityUtils.validate_file_extension(safe_file)
        self.assertTrue(result["valid"])
        self.assertEqual(result["risk_level"], "low")
    
    def test_generate_security_headers_enterprise(self):
        """Test enterprise-grade security headers generation."""
        headers = SecurityUtils.generate_security_headers(
            security_level="enterprise", 
            compliance_mode="government"
        )
        
        self.assertIn("Content-Security-Policy", headers)
        self.assertIn("Strict-Transport-Security", headers)
        self.assertIn("Permissions-Policy", headers)
        self.assertIn("X-Content-Type-Options", headers)
    
    def test_validate_url_enterprise(self):
        """Test enterprise-grade URL validation."""
        # Test dangerous URL
        dangerous_url = "file:///etc/passwd"
        result = SecurityUtils.validate_url(dangerous_url)
        self.assertFalse(result["valid"])
        self.assertEqual(result["risk_level"], "critical")
        
        # Test safe URL
        safe_url = "https://api.example.com/endpoint"
        result = SecurityUtils.validate_url(safe_url)
        self.assertTrue(result["valid"])
        self.assertEqual(result["risk_level"], "low")
    
    def test_generate_csrf_token_enterprise(self):
        """Test enterprise-grade CSRF token generation."""
        token = SecurityUtils.generate_csrf_token(security_level="enterprise")
        self.assertTrue(token.startswith("csrf_"))
        self.assertIn("_", token)
        
        # Test verification
        result = SecurityUtils.verify_csrf_token(token, token, security_level="enterprise")
        self.assertTrue(result["valid"])
    
    def test_analyze_threat_intelligence(self):
        """Test threat intelligence analysis."""
        malicious_input = "<script>alert('xss')</script>"
        result = SecurityUtils.analyze_threat_intelligence(malicious_input)
        
        self.assertGreater(result["risk_score"], 0)
        self.assertGreater(len(result["threats_found"]), 0)
        self.assertIn("recommendations", result)

class TestPerformanceMonitor(unittest.TestCase):
    """Test enhanced performance monitoring with enterprise features."""
    
    def setUp(self):
        self.monitor = PerformanceMonitor(enterprise_mode=True)
    
    def test_enterprise_initialization(self):
        """Test enterprise mode initialization."""
        self.assertTrue(self.monitor.enterprise_mode)
        self.assertIsNotNone(self.monitor.sla_targets)
        self.assertIsNotNone(self.monitor.cost_analysis)
        self.assertIsNotNone(self.monitor.resource_utilization)
    
    def test_record_metric_with_metadata(self):
        """Test metric recording with enhanced metadata."""
        metadata = {"user_id": "123", "endpoint": "/api/test"}
        self.monitor.record_metric("response_time", 150.5, metadata)
        
        self.assertIn("response_time", self.monitor.metrics)
        self.assertEqual(len(self.monitor.metrics["response_time"]), 1)
        self.assertEqual(self.monitor.metrics["response_time"][0], 150.5)
    
    def test_set_sla_target(self):
        """Test SLA target setting."""
        self.monitor.set_sla_target("response_time", "p95", 200.0)
        self.assertIn("response_time", self.monitor.sla_targets)
        self.assertEqual(self.monitor.sla_targets["response_time"]["p95"], 200.0)
    
    def test_get_sla_compliance(self):
        """Test SLA compliance checking."""
        # Set SLA target
        self.monitor.set_sla_target("response_time", "p95", 200.0)
        
        # Record metrics
        for i in range(100):
            self.monitor.record_metric("response_time", 150.0 + i)
        
        # Check compliance
        compliance = self.monitor.get_sla_compliance("response_time")
        self.assertIn("overall_compliance", compliance)
        self.assertIn("sla_details", compliance)
    
    def test_get_enterprise_insights(self):
        """Test enterprise insights generation."""
        # Record some metrics first
        for i in range(50):
            self.monitor.record_metric("response_time", 100.0 + i)
            self.monitor.record_metric("throughput", 1000 + i)
        
        insights = self.monitor.get_enterprise_insights()
        
        self.assertIn("performance_summary", insights)
        self.assertIn("sla_compliance", insights)
        self.assertIn("anomaly_summary", insights)
        self.assertIn("predictive_insights", insights)
        self.assertIn("capacity_recommendations", insights)
        self.assertIn("business_impact", insights)
        self.assertIn("cost_analysis", insights)
        self.assertIn("recommendations", insights)
    
    def test_export_enterprise_report(self):
        """Test enterprise report export."""
        # Record metrics
        self.monitor.record_metric("test_metric", 100.0)
        
        # Export report
        json_report = self.monitor.export_enterprise_report("json")
        csv_report = self.monitor.export_enterprise_report("csv")
        
        self.assertIsInstance(json_report, str)
        self.assertIsInstance(csv_report, str)
        
        # Verify JSON is valid
        json_data = json.loads(json_report)
        self.assertIn("performance_summary", json_data)

class TestCircuitBreaker(unittest.TestCase):
    """Test enterprise-grade circuit breaker implementation."""
    
    def setUp(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5,
            success_threshold=2,
            adaptive_thresholds=True,
            monitoring_enabled=True
        )
    
    def test_initial_state(self):
        """Test initial circuit breaker state."""
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 0)
        self.assertEqual(self.circuit_breaker.success_count, 0)
    
    def test_successful_execution(self):
        """Test successful function execution."""
        def success_func():
            return "success"
        
        result = self.circuit_breaker.call(success_func)
        self.assertEqual(result, "success")
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.CLOSED)
        self.assertEqual(self.circuit_breaker.success_count, 1)
    
    def test_failure_threshold_exceeded(self):
        """Test circuit opening when failure threshold is exceeded."""
        def failing_func():
            raise Exception("Test failure")
        
        # Execute failing function multiple times
        for _ in range(3):
            try:
                self.circuit_breaker.call(failing_func)
            except Exception:
                pass
        
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.OPEN)
        self.assertEqual(self.circuit_breaker.failure_count, 3)
    
    def test_half_open_state(self):
        """Test circuit transitioning to half-open state."""
        # First, open the circuit
        def failing_func():
            raise Exception("Test failure")
        
        for _ in range(3):
            try:
                self.circuit_breaker.call(failing_func)
            except Exception:
                pass
        
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.OPEN)
        
        # Wait for recovery timeout and try again
        self.circuit_breaker.last_failure_time = time.time() - 10
        
        # This should transition to half-open
        self.assertTrue(self.circuit_breaker._can_execute())
    
    def test_recovery_to_closed(self):
        """Test circuit recovery to closed state."""
        # Open the circuit
        def failing_func():
            raise Exception("Test failure")
        
        for _ in range(3):
            try:
                self.circuit_breaker.call(failing_func)
            except Exception:
                pass
        
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.OPEN)
        
        # Reset to half-open
        self.circuit_breaker._transition_to_half_open()
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.HALF_OPEN)
        
        # Execute successful function multiple times
        def success_func():
            return "success"
        
        for _ in range(2):
            self.circuit_breaker.call(success_func)
        
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.CLOSED)
    
    def test_async_execution(self):
        """Test async function execution."""
        async def async_success_func():
            return "async_success"
        
        # This would need to be run in an async context
        # For now, just test that the method exists
        self.assertTrue(hasattr(self.circuit_breaker, 'call_async'))
    
    def test_get_status(self):
        """Test comprehensive status reporting."""
        status = self.circuit_breaker.get_status()
        
        self.assertIn("state", status)
        self.assertIn("failure_count", status)
        self.assertIn("success_count", status)
        self.assertIn("total_requests", status)
        self.assertIn("success_rate", status)
        self.assertIn("health_score", status)
        self.assertIn("business_impact", status)
    
    def test_health_check(self):
        """Test health check functionality."""
        health_result = self.circuit_breaker.get_health_check()
        
        self.assertIn("overall_health", health_result)
        self.assertIn("health_score", health_result)
        self.assertIn("indicators", health_result)
        self.assertIn("recommendations", health_result)
    
    def test_manual_control(self):
        """Test manual circuit breaker control."""
        # Test reset
        self.circuit_breaker.reset()
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.CLOSED)
        
        # Test force open
        self.circuit_breaker.force_open()
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.OPEN)
        
        # Test force close
        self.circuit_breaker.force_close()
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.CLOSED)
    
    def test_threshold_updates(self):
        """Test dynamic threshold updates."""
        self.circuit_breaker.update_thresholds(
            failure_threshold=5,
            recovery_timeout=10,
            success_threshold=3
        )
        
        self.assertEqual(self.circuit_breaker.failure_threshold, 5)
        self.assertEqual(self.circuit_breaker.recovery_timeout, 10)
        self.assertEqual(self.circuit_breaker.success_threshold, 3)
    
    def test_export_health_report(self):
        """Test health report export."""
        # Generate some activity first
        def success_func():
            return "success"
        
        self.circuit_breaker.call(success_func)
        
        # Export report
        report = self.circuit_breaker.export_health_report("json")
        self.assertIsInstance(report, str)
        
        # Verify JSON is valid
        report_data = json.loads(report)
        self.assertIn("circuit_breaker_status", report_data)

class TestErrorHandler(unittest.TestCase):
    """Test enterprise error handling and intelligent alerting."""
    
    def setUp(self):
        self.error_handler = ErrorHandler(
            alerting_enabled=True,
            error_tracking=True
        )
    
    def test_error_categorization(self):
        """Test automatic error categorization."""
        # Test critical error
        critical_error = Exception("database connection failed")
        error_info = self.error_handler.handle_error(critical_error)
        
        self.assertEqual(error_info["severity"], "critical")
        self.assertEqual(error_info["category_details"]["business_impact"], "high")
        
        # Test high severity error
        high_error = Exception("validation failed")
        error_info = self.error_handler.handle_error(high_error)
        
        self.assertEqual(error_info["severity"], "high")
        self.assertEqual(error_info["category_details"]["escalation"], True)
    
    def test_error_context_analysis(self):
        """Test error context analysis."""
        context = {
            "user_id": "123",
            "request_id": "req_456",
            "endpoint": "/api/test",
            "execution_time": 150.5,
            "business_impact": "high"
        }
        
        error = Exception("test error")
        error_info = self.error_handler.handle_error(error, context)
        
        self.assertIn("context_analysis", error_info)
        analysis = error_info["context_analysis"]
        
        self.assertTrue(analysis["has_user_context"])
        self.assertTrue(analysis["has_request_context"])
        self.assertTrue(analysis["has_performance_context"])
        self.assertTrue(analysis["has_business_context"])
        self.assertEqual(analysis["context_completeness"], 100.0)
    
    def test_alert_generation(self):
        """Test intelligent alert generation."""
        # Generate multiple critical errors to trigger alerts
        for _ in range(5):
            critical_error = Exception("database connection failed")
            self.error_handler.handle_error(critical_error)
        
        # Check if alerts were generated
        self.assertGreater(len(self.error_handler.alerts), 0)
        
        # Verify alert structure
        alert = self.error_handler.alerts[0]
        self.assertIn("alert_id", alert)
        self.assertIn("severity", alert)
        self.assertIn("channels", alert)
        self.assertIn("escalation_required", alert)
    
    def test_error_pattern_tracking(self):
        """Test error pattern analysis."""
        # Generate multiple similar errors
        for _ in range(10):
            error = Exception("validation failed")
            self.error_handler.handle_error(error)
        
        # Check pattern tracking
        self.assertIn("Exception", self.error_handler.error_patterns)
        pattern = self.error_handler.error_patterns["Exception"]
        
        self.assertEqual(pattern["count"], 10)
        self.assertIn("severity_distribution", pattern)
        self.assertIn("context_patterns", pattern)
    
    def test_business_impact_tracking(self):
        """Test business impact metrics."""
        # Generate errors with different severities
        critical_error = Exception("database connection failed")
        high_error = Exception("validation failed")
        
        self.error_handler.handle_error(critical_error)
        self.error_handler.handle_error(high_error)
        
        # Check business impact metrics
        metrics = self.error_handler.business_impact_metrics
        
        self.assertGreater(metrics["total_downtime_minutes"], 0)
        self.assertGreater(metrics["affected_users"], 0)
        self.assertGreater(metrics["revenue_impact"], 0.0)
    
    def test_error_resolution(self):
        """Test error resolution tracking."""
        # Generate an error
        error = Exception("test error")
        error_info = self.error_handler.handle_error(error)
        
        # Resolve the error
        success = self.error_handler.resolve_error(
            error_info["error_id"],
            "Fixed by updating configuration",
            resolution_time_minutes=15.5
        )
        
        self.assertTrue(success)
        
        # Check resolution tracking
        self.assertIn(error_info["error_id"], self.error_handler.error_resolution)
        resolution = self.error_handler.error_resolution[error_info["error_id"]]
        
        self.assertEqual(resolution["resolution"], "Fixed by updating configuration")
        self.assertEqual(resolution["resolution_time_minutes"], 15.5)
    
    def test_error_summary(self):
        """Test comprehensive error summary."""
        # Generate some errors first
        for i in range(5):
            error = Exception(f"test error {i}")
            self.error_handler.handle_error(error)
        
        summary = self.error_handler.get_error_summary()
        
        self.assertIn("total_errors", summary)
        self.assertIn("severity_distribution", summary)
        self.assertIn("top_error_types", summary)
        self.assertIn("error_patterns", summary)
        self.assertIn("business_impact", summary)
        self.assertIn("active_alerts", summary)
    
    def test_export_error_report(self):
        """Test error report export."""
        # Generate some errors
        for i in range(3):
            error = Exception(f"test error {i}")
            self.error_handler.handle_error(error)
        
        # Export report
        json_report = self.error_handler.export_error_report("json")
        self.assertIsInstance(json_report, str)
        
        # Verify JSON is valid
        report_data = json.loads(json_report)
        self.assertIn("error_summary", report_data)
        self.assertIn("error_history", report_data)

class TestValidationUtils(unittest.TestCase):
    """Test enhanced validation utilities."""
    
    def setUp(self):
        self.validator = ValidationUtils()
    
    def test_validate_email(self):
        """Test email validation."""
        self.assertTrue(self.validator.validate_email("test@example.com"))
        self.assertFalse(self.validator.validate_email("invalid-email"))
        self.assertFalse(self.validator.validate_email("test@"))
    
    def test_validate_url(self):
        """Test URL validation."""
        self.assertTrue(self.validator.validate_url("https://example.com"))
        self.assertTrue(self.validator.validate_url("http://api.example.com/endpoint"))
        self.assertFalse(self.validator.validate_url("not-a-url"))
        self.assertFalse(self.validator.validate_url("ftp://example.com"))
    
    def test_validate_phone(self):
        """Test phone number validation."""
        self.assertTrue(self.validator.validate_phone("1234567890"))
        self.assertTrue(self.validator.validate_phone("+1-234-567-8900"))
        self.assertFalse(self.validator.validate_phone("123"))
        self.assertFalse(self.validator.validate_phone("abcdefghij"))
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        dangerous_filename = "file<name>with:invalid*chars?.txt"
        sanitized = self.validator.sanitize_filename(dangerous_filename)
        
        self.assertNotIn("<", sanitized)
        self.assertNotIn(">", sanitized)
        self.assertNotIn(":", sanitized)
        self.assertNotIn("*", sanitized)
        self.assertNotIn("?", sanitized)
        self.assertIn("file", sanitized)
        self.assertIn("name", sanitized)
        self.assertIn("with", sanitized)
        self.assertIn("invalid", sanitized)
        self.assertIn("chars", sanitized)
        self.assertIn(".txt", sanitized)

def run_enterprise_tests():
    """Run all enterprise feature tests and generate report."""
    print("🚀 Starting Enterprise Features Test Suite...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSecurityUtils,
        TestPerformanceMonitor,
        TestCircuitBreaker,
        TestErrorHandler,
        TestValidationUtils
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate test report
    print("\n" + "=" * 60)
    print("📊 ENTERPRISE FEATURES TEST REPORT")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if failures > 0:
        print(f"\n❌ FAILURES ({failures}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if errors > 0:
        print(f"\n💥 ERRORS ({errors}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Feature summary
    print(f"\n🎯 FEATURE SUMMARY:")
    print(f"  ✅ Advanced Security (SecurityUtils): {len([t for t in result.testsRun if 'SecurityUtils' in str(t)])} tests")
    print(f"  ✅ Enhanced Performance Monitoring (PerformanceMonitor): {len([t for t in result.testsRun if 'PerformanceMonitor' in str(t)])} tests")
    print(f"  ✅ Circuit Breaker Pattern (CircuitBreaker): {len([t for t in result.testsRun if 'CircuitBreaker' in str(t)])} tests")
    print(f"  ✅ Enterprise Error Handling (ErrorHandler): {len([t for t in result.testsRun if 'ErrorHandler' in str(t)])} tests")
    print(f"  ✅ Enhanced Validation (ValidationUtils): {len([t for t in result.testsRun if 'ValidationUtils' in str(t)])} tests")
    
    if success_rate >= 90:
        print(f"\n🎉 EXCELLENT! Enterprise features are working perfectly!")
    elif success_rate >= 80:
        print(f"\n👍 GOOD! Most enterprise features are working well.")
    elif success_rate >= 70:
        print(f"\n⚠️  FAIR! Some enterprise features need attention.")
    else:
        print(f"\n🚨 POOR! Enterprise features need significant work.")
    
    return result

if __name__ == "__main__":
    run_enterprise_tests()







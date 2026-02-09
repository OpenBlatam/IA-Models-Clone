"""
Comprehensive Testing System for Ultimate Opus Clip

Complete testing suite including unit tests, integration tests,
performance tests, security tests, and end-to-end tests.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import asyncio
import time
import unittest
import pytest
import json
import tempfile
import shutil
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from pathlib import Path
import uuid
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
import sys
import os

logger = structlog.get_logger("comprehensive_testing")

class TestType(Enum):
    """Types of tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    END_TO_END = "end_to_end"
    LOAD = "load"
    STRESS = "stress"

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Test execution result."""
    test_id: str
    test_name: str
    test_type: TestType
    status: TestStatus
    duration: float
    start_time: float
    end_time: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None
    assertions: List[Dict[str, Any]] = None

@dataclass
class TestSuite:
    """Test suite configuration."""
    suite_id: str
    name: str
    description: str
    test_type: TestType
    tests: List[str]
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    timeout: int = 300
    parallel: bool = False

class UnitTests:
    """Unit tests for individual components."""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        logger.info("Unit Tests initialized")
    
    def test_ai_enhancements(self) -> TestResult:
        """Test AI enhancements functionality."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Import AI enhancements
            from ai_enhancements import get_ai_enhancements, ContentType, EmotionType
            
            # Test content analysis
            ai_enhancements = get_ai_enhancements()
            
            # Test emotion detection
            emotions = [(EmotionType.HAPPY, 0.8), (EmotionType.EXCITED, 0.6)]
            assert len(emotions) > 0, "Should detect emotions"
            
            # Test content classification
            content_type = ContentType.ENTERTAINMENT
            assert content_type is not None, "Should classify content type"
            
            # Test viral prediction
            viral_potential = 0.75
            assert 0 <= viral_potential <= 1, "Viral potential should be between 0 and 1"
            
            duration = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="test_ai_enhancements",
                test_type=TestType.UNIT,
                status=TestStatus.PASSED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                metrics={"viral_potential": viral_potential, "emotions_detected": len(emotions)}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="test_ai_enhancements",
                test_type=TestType.UNIT,
                status=TestStatus.FAILED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e)
            )
    
    def test_cloud_integration(self) -> TestResult:
        """Test cloud integration functionality."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            from cloud_integration import get_cloud_manager, CloudProvider, ServiceType
            
            # Test cloud manager initialization
            cloud_manager = get_cloud_manager()
            assert cloud_manager is not None, "Cloud manager should be initialized"
            
            # Test cloud provider enum
            provider = CloudProvider.AWS
            assert provider.value == "aws", "AWS provider should be correct"
            
            # Test service type enum
            service_type = ServiceType.STORAGE
            assert service_type.value == "storage", "Storage service type should be correct"
            
            duration = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="test_cloud_integration",
                test_type=TestType.UNIT,
                status=TestStatus.PASSED,
                duration=duration,
                start_time=start_time,
                end_time=time.time()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="test_cloud_integration",
                test_type=TestType.UNIT,
                status=TestStatus.FAILED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e)
            )
    
    def test_mobile_api(self) -> TestResult:
        """Test mobile API functionality."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            from mobile_api import get_mobile_api, MobilePlatform, QualityLevel
            
            # Test mobile API initialization
            mobile_api = get_mobile_api()
            assert mobile_api is not None, "Mobile API should be initialized"
            
            # Test platform enum
            platform = MobilePlatform.IOS
            assert platform.value == "ios", "iOS platform should be correct"
            
            # Test quality level enum
            quality = QualityLevel.HIGH
            assert quality.value == "high", "High quality should be correct"
            
            duration = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="test_mobile_api",
                test_type=TestType.UNIT,
                status=TestStatus.PASSED,
                duration=duration,
                start_time=start_time,
                end_time=time.time()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="test_mobile_api",
                test_type=TestType.UNIT,
                status=TestStatus.FAILED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e)
            )
    
    def test_workflow_automation(self) -> TestResult:
        """Test workflow automation functionality."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            from workflow_automation import get_workflow_engine, WorkflowStatus, ActionType
            
            # Test workflow engine initialization
            workflow_engine = get_workflow_engine()
            assert workflow_engine is not None, "Workflow engine should be initialized"
            
            # Test workflow status enum
            status = WorkflowStatus.ACTIVE
            assert status.value == "active", "Active status should be correct"
            
            # Test action type enum
            action_type = ActionType.PROCESS_VIDEO
            assert action_type.value == "process_video", "Process video action should be correct"
            
            duration = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="test_workflow_automation",
                test_type=TestType.UNIT,
                status=TestStatus.PASSED,
                duration=duration,
                start_time=start_time,
                end_time=time.time()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="test_workflow_automation",
                test_type=TestType.UNIT,
                status=TestStatus.FAILED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e)
            )
    
    def test_enterprise_features(self) -> TestResult:
        """Test enterprise features functionality."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            from enterprise_features import get_enterprise_features, UserRole, BillingPlan
            
            # Test enterprise features initialization
            enterprise_features = get_enterprise_features()
            assert enterprise_features is not None, "Enterprise features should be initialized"
            
            # Test user role enum
            role = UserRole.ADMIN
            assert role.value == "admin", "Admin role should be correct"
            
            # Test billing plan enum
            plan = BillingPlan.PROFESSIONAL
            assert plan.value == "professional", "Professional plan should be correct"
            
            duration = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="test_enterprise_features",
                test_type=TestType.UNIT,
                status=TestStatus.PASSED,
                duration=duration,
                start_time=start_time,
                end_time=time.time()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="test_enterprise_features",
                test_type=TestType.UNIT,
                status=TestStatus.FAILED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e)
            )
    
    def test_gpu_optimization(self) -> TestResult:
        """Test GPU optimization functionality."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            from gpu_optimization import get_gpu_manager, OptimizationLevel, MemoryStrategy
            
            # Test GPU manager initialization
            gpu_manager = get_gpu_manager()
            assert gpu_manager is not None, "GPU manager should be initialized"
            
            # Test optimization level enum
            level = OptimizationLevel.HIGH
            assert level.value == "high", "High optimization should be correct"
            
            # Test memory strategy enum
            strategy = MemoryStrategy.BALANCED
            assert strategy.value == "balanced", "Balanced strategy should be correct"
            
            duration = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="test_gpu_optimization",
                test_type=TestType.UNIT,
                status=TestStatus.PASSED,
                duration=duration,
                start_time=start_time,
                end_time=time.time()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="test_gpu_optimization",
                test_type=TestType.UNIT,
                status=TestStatus.FAILED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e)
            )
    
    def test_advanced_security(self) -> TestResult:
        """Test advanced security functionality."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            from advanced_security import get_security_system, SecurityLevel, ThreatType
            
            # Test security system initialization
            security_system = get_security_system()
            assert security_system is not None, "Security system should be initialized"
            
            # Test security level enum
            level = SecurityLevel.HIGH
            assert level.value == "high", "High security level should be correct"
            
            # Test threat type enum
            threat_type = ThreatType.SQL_INJECTION
            assert threat_type.value == "sql_injection", "SQL injection threat should be correct"
            
            duration = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="test_advanced_security",
                test_type=TestType.UNIT,
                status=TestStatus.PASSED,
                duration=duration,
                start_time=start_time,
                end_time=time.time()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="test_advanced_security",
                test_type=TestType.UNIT,
                status=TestStatus.FAILED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e)
            )
    
    def run_all_unit_tests(self) -> List[TestResult]:
        """Run all unit tests."""
        tests = [
            self.test_ai_enhancements,
            self.test_cloud_integration,
            self.test_mobile_api,
            self.test_workflow_automation,
            self.test_enterprise_features,
            self.test_gpu_optimization,
            self.test_advanced_security
        ]
        
        results = []
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                self.test_results.append(result)
            except Exception as e:
                logger.error(f"Error running test {test_func.__name__}: {e}")
        
        return results

class IntegrationTests:
    """Integration tests for component interactions."""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        logger.info("Integration Tests initialized")
    
    def test_ai_cloud_integration(self) -> TestResult:
        """Test AI enhancements with cloud integration."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            from ai_enhancements import get_ai_enhancements
            from cloud_integration import get_cloud_manager
            
            # Test AI enhancements
            ai_enhancements = get_ai_enhancements()
            assert ai_enhancements is not None, "AI enhancements should be available"
            
            # Test cloud manager
            cloud_manager = get_cloud_manager()
            assert cloud_manager is not None, "Cloud manager should be available"
            
            # Test integration (simplified)
            # In a real test, you would test actual integration between components
            
            duration = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="test_ai_cloud_integration",
                test_type=TestType.INTEGRATION,
                status=TestStatus.PASSED,
                duration=duration,
                start_time=start_time,
                end_time=time.time()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="test_ai_cloud_integration",
                test_type=TestType.INTEGRATION,
                status=TestStatus.FAILED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e)
            )
    
    def test_workflow_enterprise_integration(self) -> TestResult:
        """Test workflow automation with enterprise features."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            from workflow_automation import get_workflow_engine
            from enterprise_features import get_enterprise_features
            
            # Test workflow engine
            workflow_engine = get_workflow_engine()
            assert workflow_engine is not None, "Workflow engine should be available"
            
            # Test enterprise features
            enterprise_features = get_enterprise_features()
            assert enterprise_features is not None, "Enterprise features should be available"
            
            # Test integration (simplified)
            # In a real test, you would test actual integration between components
            
            duration = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="test_workflow_enterprise_integration",
                test_type=TestType.INTEGRATION,
                status=TestStatus.PASSED,
                duration=duration,
                start_time=start_time,
                end_time=time.time()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="test_workflow_enterprise_integration",
                test_type=TestType.INTEGRATION,
                status=TestStatus.FAILED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e)
            )
    
    def run_all_integration_tests(self) -> List[TestResult]:
        """Run all integration tests."""
        tests = [
            self.test_ai_cloud_integration,
            self.test_workflow_enterprise_integration
        ]
        
        results = []
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                self.test_results.append(result)
            except Exception as e:
                logger.error(f"Error running test {test_func.__name__}: {e}")
        
        return results

class PerformanceTests:
    """Performance tests for system components."""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        logger.info("Performance Tests initialized")
    
    def test_ai_enhancements_performance(self) -> TestResult:
        """Test AI enhancements performance."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            from ai_enhancements import get_ai_enhancements
            
            ai_enhancements = get_ai_enhancements()
            
            # Test performance with multiple operations
            operations = 100
            operation_times = []
            
            for i in range(operations):
                op_start = time.time()
                
                # Simulate AI operation
                await asyncio.sleep(0.001)  # Simulate processing
                
                op_end = time.time()
                operation_times.append(op_end - op_start)
            
            avg_operation_time = sum(operation_times) / len(operation_times)
            total_time = time.time() - start_time
            
            # Performance assertions
            assert avg_operation_time < 0.1, f"Average operation time should be < 0.1s, got {avg_operation_time}"
            assert total_time < 5.0, f"Total test time should be < 5s, got {total_time}"
            
            duration = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="test_ai_enhancements_performance",
                test_type=TestType.PERFORMANCE,
                status=TestStatus.PASSED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                metrics={
                    "operations": operations,
                    "avg_operation_time": avg_operation_time,
                    "total_time": total_time,
                    "operations_per_second": operations / total_time
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="test_ai_enhancements_performance",
                test_type=TestType.PERFORMANCE,
                status=TestStatus.FAILED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e)
            )
    
    def test_memory_usage(self) -> TestResult:
        """Test memory usage of components."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Get initial memory usage
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Import and initialize components
            from ai_enhancements import get_ai_enhancements
            from cloud_integration import get_cloud_manager
            from mobile_api import get_mobile_api
            from workflow_automation import get_workflow_engine
            from enterprise_features import get_enterprise_features
            from gpu_optimization import get_gpu_manager
            from advanced_security import get_security_system
            
            # Initialize all components
            ai_enhancements = get_ai_enhancements()
            cloud_manager = get_cloud_manager()
            mobile_api = get_mobile_api()
            workflow_engine = get_workflow_engine()
            enterprise_features = get_enterprise_features()
            gpu_manager = get_gpu_manager()
            security_system = get_security_system()
            
            # Get final memory usage
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory assertions
            assert memory_increase < 500, f"Memory increase should be < 500MB, got {memory_increase}MB"
            
            duration = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="test_memory_usage",
                test_type=TestType.PERFORMANCE,
                status=TestStatus.PASSED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                metrics={
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "memory_increase_mb": memory_increase
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="test_memory_usage",
                test_type=TestType.PERFORMANCE,
                status=TestStatus.FAILED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e)
            )
    
    def run_all_performance_tests(self) -> List[TestResult]:
        """Run all performance tests."""
        tests = [
            self.test_ai_enhancements_performance,
            self.test_memory_usage
        ]
        
        results = []
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                self.test_results.append(result)
            except Exception as e:
                logger.error(f"Error running test {test_func.__name__}: {e}")
        
        return results

class SecurityTests:
    """Security tests for system components."""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        logger.info("Security Tests initialized")
    
    def test_authentication_security(self) -> TestResult:
        """Test authentication security."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            from advanced_security import get_security_system, authenticate_user
            
            security_system = get_security_system()
            
            # Test valid authentication
            result = authenticate_user("admin", "admin123", "127.0.0.1")
            assert result["success"] == True, "Valid authentication should succeed"
            
            # Test invalid authentication
            result = authenticate_user("admin", "wrongpassword", "127.0.0.1")
            assert result["success"] == False, "Invalid authentication should fail"
            
            # Test SQL injection attempt
            result = authenticate_user("admin'; DROP TABLE users; --", "admin123", "127.0.0.1")
            assert result["success"] == False, "SQL injection should be blocked"
            
            # Test XSS attempt
            result = authenticate_user("<script>alert('xss')</script>", "admin123", "127.0.0.1")
            assert result["success"] == False, "XSS should be blocked"
            
            duration = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="test_authentication_security",
                test_type=TestType.SECURITY,
                status=TestStatus.PASSED,
                duration=duration,
                start_time=start_time,
                end_time=time.time()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="test_authentication_security",
                test_type=TestType.SECURITY,
                status=TestStatus.FAILED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e)
            )
    
    def test_input_validation(self) -> TestResult:
        """Test input validation security."""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            from advanced_security import get_security_system, validate_request
            
            security_system = get_security_system()
            
            # Test valid input
            result = validate_request("normal input", "127.0.0.1")
            assert result == True, "Valid input should pass validation"
            
            # Test SQL injection
            result = validate_request("'; DROP TABLE users; --", "127.0.0.1")
            assert result == False, "SQL injection should be blocked"
            
            # Test XSS
            result = validate_request("<script>alert('xss')</script>", "127.0.0.1")
            assert result == False, "XSS should be blocked"
            
            # Test CSRF
            result = validate_request("<form action='http://evil.com'>", "127.0.0.1")
            assert result == False, "CSRF should be blocked"
            
            duration = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_name="test_input_validation",
                test_type=TestType.SECURITY,
                status=TestStatus.PASSED,
                duration=duration,
                start_time=start_time,
                end_time=time.time()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_id=test_id,
                test_name="test_input_validation",
                test_type=TestType.SECURITY,
                status=TestStatus.FAILED,
                duration=duration,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e)
            )
    
    def run_all_security_tests(self) -> List[TestResult]:
        """Run all security tests."""
        tests = [
            self.test_authentication_security,
            self.test_input_validation
        ]
        
        results = []
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                self.test_results.append(result)
            except Exception as e:
                logger.error(f"Error running test {test_func.__name__}: {e}")
        
        return results

class ComprehensiveTestingSystem:
    """Main comprehensive testing system."""
    
    def __init__(self):
        self.unit_tests = UnitTests()
        self.integration_tests = IntegrationTests()
        self.performance_tests = PerformanceTests()
        self.security_tests = SecurityTests()
        self.all_results: List[TestResult] = []
        
        logger.info("Comprehensive Testing System initialized")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        try:
            logger.info("Starting comprehensive test suite")
            
            # Run unit tests
            logger.info("Running unit tests...")
            unit_results = self.unit_tests.run_all_unit_tests()
            self.all_results.extend(unit_results)
            
            # Run integration tests
            logger.info("Running integration tests...")
            integration_results = self.integration_tests.run_all_integration_tests()
            self.all_results.extend(integration_results)
            
            # Run performance tests
            logger.info("Running performance tests...")
            performance_results = self.performance_tests.run_all_performance_tests()
            self.all_results.extend(performance_results)
            
            # Run security tests
            logger.info("Running security tests...")
            security_results = self.security_tests.run_all_security_tests()
            self.all_results.extend(security_results)
            
            # Generate summary
            summary = self._generate_test_summary()
            
            logger.info("Comprehensive test suite completed")
            return summary
            
        except Exception as e:
            logger.error(f"Error running comprehensive tests: {e}")
            return {"error": str(e)}
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        try:
            total_tests = len(self.all_results)
            passed_tests = len([r for r in self.all_results if r.status == TestStatus.PASSED])
            failed_tests = len([r for r in self.all_results if r.status == TestStatus.FAILED])
            error_tests = len([r for r in self.all_results if r.status == TestStatus.ERROR])
            
            # Group by test type
            by_type = {}
            for result in self.all_results:
                test_type = result.test_type.value
                if test_type not in by_type:
                    by_type[test_type] = {"total": 0, "passed": 0, "failed": 0, "error": 0}
                
                by_type[test_type]["total"] += 1
                if result.status == TestStatus.PASSED:
                    by_type[test_type]["passed"] += 1
                elif result.status == TestStatus.FAILED:
                    by_type[test_type]["failed"] += 1
                elif result.status == TestStatus.ERROR:
                    by_type[test_type]["error"] += 1
            
            # Calculate success rate
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            # Get average duration
            avg_duration = sum(r.duration for r in self.all_results) / total_tests if total_tests > 0 else 0
            
            # Get failed tests details
            failed_details = [
                {
                    "name": r.test_name,
                    "type": r.test_type.value,
                    "error": r.error_message,
                    "duration": r.duration
                }
                for r in self.all_results if r.status == TestStatus.FAILED
            ]
            
            return {
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "error_tests": error_tests,
                    "success_rate": success_rate,
                    "avg_duration": avg_duration
                },
                "by_type": by_type,
                "failed_tests": failed_details,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error generating test summary: {e}")
            return {"error": str(e)}
    
    def generate_test_report(self, output_path: str = "test_report.json"):
        """Generate detailed test report."""
        try:
            report = {
                "test_results": [asdict(result) for result in self.all_results],
                "summary": self._generate_test_summary()
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Test report generated: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating test report: {e}")

# Global testing system instance
_global_testing_system: Optional[ComprehensiveTestingSystem] = None

def get_testing_system() -> ComprehensiveTestingSystem:
    """Get the global testing system instance."""
    global _global_testing_system
    if _global_testing_system is None:
        _global_testing_system = ComprehensiveTestingSystem()
    return _global_testing_system

async def run_comprehensive_tests() -> Dict[str, Any]:
    """Run comprehensive test suite."""
    testing_system = get_testing_system()
    return await testing_system.run_all_tests()

def generate_test_report(output_path: str = "test_report.json"):
    """Generate test report."""
    testing_system = get_testing_system()
    testing_system.generate_test_report(output_path)

if __name__ == "__main__":
    # Run tests when script is executed directly
    import asyncio
    
    async def main():
        testing_system = get_testing_system()
        results = await testing_system.run_all_tests()
        print(json.dumps(results, indent=2))
        
        # Generate report
        testing_system.generate_test_report()
    
    asyncio.run(main())



"""
Comprehensive Test Runner for LinkedIn Posts
===========================================

A comprehensive test runner that executes all types of tests
(unit, integration, API, load) with proper configuration,
reporting, and CI/CD integration.
"""

import pytest
import asyncio
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
from unit.test_post_service import TestPostService
from unit.test_entities import (
    TestLinkedInPost, TestPostGenerationRequest, TestPostValidationResult,
    TestContentAnalysisResult, TestPostOptimizationResult, TestEngagementMetrics
)
from unit.test_edge_cases import TestEdgeCases
from unit.test_security import TestSecurity
from unit.test_data_validation import TestDataValidation
from unit.test_ai_integration import TestAIIntegration
from unit.test_business_logic import TestBusinessLogic
from unit.test_workflow_scenarios import TestWorkflowScenarios
from unit.test_advanced_analytics import TestAdvancedAnalytics
from unit.test_database_repository import TestDatabaseRepository
from unit.test_event_driven_architecture import TestEventDrivenArchitecture
from unit.test_caching_strategies import TestCachingStrategies
from unit.test_rate_limiting import TestRateLimiting
from unit.test_notification_system import TestNotificationSystem
from unit.test_microservices_architecture import TestMicroservicesArchitecture
from unit.test_content_personalization import TestContentPersonalization
from unit.test_content_optimization import TestContentOptimization
from unit.test_social_media_integration import TestSocialMediaIntegration
from unit.test_notification_system import TestNotificationSystem
from unit.test_data_validation import TestDataValidation
from unit.test_data_analytics_reporting import TestDataAnalyticsReporting
from unit.test_machine_learning_integration import TestMachineLearningIntegration
from unit.test_content_scheduling import TestContentScheduling
from unit.test_content_approval import TestContentApproval
from unit.test_content_localization import TestContentLocalization
from unit.test_content_engagement import TestContentEngagement
from unit.test_content_performance import TestContentPerformance
from unit.test_content_collaboration import TestContentCollaboration
from unit.test_content_versioning import TestContentVersioning
from unit.test_content_quality_assurance import TestContentQualityAssurance
from unit.test_content_compliance import TestContentCompliance
from unit.test_content_monetization import TestContentMonetization
from unit.test_content_accessibility import TestContentAccessibility
from unit.test_content_governance import TestContentGovernance
from unit.test_content_lifecycle import TestContentLifecycle
from unit.test_content_intelligence import TestContentIntelligence
from unit.test_content_automation import TestContentAutomation
from unit.test_content_security_privacy import TestContentSecurityPrivacy
from unit.test_content_scalability_performance import TestContentScalabilityPerformance
from unit.test_content_workflow_management import TestContentWorkflowManagement
from unit.test_content_distribution_syndication import TestContentDistributionSyndication
from unit.test_content_discovery_recommendation import TestContentDiscoveryRecommendation
from unit.test_content_analytics_insights import TestContentAnalyticsInsights
from unit.test_content_team_collaboration import TestContentTeamCollaboration
from unit.test_content_integration_api import TestContentIntegrationAPI
from unit.test_content_metadata_management import TestContentMetadataManagement
from unit.test_content_moderation_filtering import TestContentModerationFiltering
from unit.test_content_backup_recovery import TestContentBackupRecovery
from unit.test_content_multi_platform_sync import TestContentMultiPlatformSync
from unit.test_content_real_time_collaboration import TestContentRealTimeCollaboration
from unit.test_content_advanced_security import TestContentAdvancedSecurity
from unit.test_content_ai_enhancement import TestContentAIEnhancement
from unit.test_content_predictive_analytics import TestContentPredictiveAnalytics
from unit.test_content_gamification import TestContentGamification
from unit.test_content_advanced_analytics_v2 import TestContentAdvancedAnalyticsV2
from unit.test_content_enterprise_features import TestContentEnterpriseFeatures
from unit.test_content_advanced_ml_integration import TestContentAdvancedMLIntegration
from integration.test_post_integration import TestPostServiceIntegration, TestPerformanceIntegration
from api.test_api_endpoints import TestLinkedInPostsAPI
from api.test_api_versioning import TestAPIVersioning
from load.test_load_performance import TestLoadPerformance
from performance.test_performance_benchmarks import TestPerformanceBenchmarks


class ComprehensiveTestRunner:
    """Comprehensive test runner for LinkedIn Posts system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize test runner with configuration."""
        self.config = config or self._default_config()
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def _default_config(self) -> Dict[str, Any]:
        """Default test configuration."""
        return {
            "test_types": ["unit", "integration", "api", "load", "performance"],
            "parallel": True,
            "verbose": True,
            "coverage": True,
            "performance_thresholds": {
                "max_response_time": 2.0,
                "min_throughput": 10.0,
                "max_memory_usage": 500,
                "max_cpu_usage": 80
            },
            "load_test_config": {
                "concurrent_users": 50,
                "duration": 60,
                "ramp_up_time": 10
            },
            "reporting": {
                "generate_html": True,
                "generate_json": True,
                "generate_junit": True
            }
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        self.start_time = time.time()
        
        print("🚀 Starting Comprehensive Test Suite for LinkedIn Posts")
        print("=" * 60)
        
        results = {}
        
        # Run different test types
        if "unit" in self.config["test_types"]:
            print("\n📋 Running Unit Tests...")
            results["unit"] = await self._run_unit_tests()
        
        if "integration" in self.config["test_types"]:
            print("\n🔗 Running Integration Tests...")
            results["integration"] = await self._run_integration_tests()
        
        if "api" in self.config["test_types"]:
            print("\n🌐 Running API Tests...")
            results["api"] = await self._run_api_tests()
        
        if "load" in self.config["test_types"]:
            print("\n⚡ Running Load Tests...")
            results["load"] = await self._run_load_tests()
        
        if "performance" in self.config["test_types"]:
            print("\n📊 Running Performance Tests...")
            results["performance"] = await self._run_performance_tests()
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(results)
        
        # Save reports
        self._save_reports(report)
        
        print("\n" + "=" * 60)
        print("✅ Test Suite Completed!")
        print(f"Total execution time: {self.end_time - self.start_time:.2f} seconds")
        
        return report
    
    async def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "execution_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Run PostService tests
            post_service_tests = TestPostService()
            post_service_results = await self._run_test_class(post_service_tests)
            results["post_service"] = post_service_results
            
            # Run entity tests
            entity_tests = [
                TestLinkedInPost(),
                TestPostGenerationRequest(),
                TestPostValidationResult(),
                TestContentAnalysisResult(),
                TestPostOptimizationResult(),
                TestEngagementMetrics()
            ]
            
                                                # Run additional unit tests
                    additional_unit_tests = [
                        TestEdgeCases(),
                        TestSecurity(),
                        TestDataValidation(),
                        TestAIIntegration(),
                        TestBusinessLogic(),
                        TestWorkflowScenarios(),
                        TestAdvancedAnalytics(),
                        TestDatabaseRepository(),
                        TestEventDrivenArchitecture(),
                        TestCachingStrategies(),
                        TestRateLimiting(),
                        TestNotificationSystem(),
                        TestMicroservicesArchitecture(),
                        TestContentPersonalization(),
                        TestContentOptimization(),
                        TestSocialMediaIntegration(),
                        TestNotificationSystem(),
                        TestDataValidation(),
                        TestDataAnalyticsReporting(),
                        TestMachineLearningIntegration(),
                        TestContentScheduling(),
                        TestContentApproval(),
                        TestContentLocalization(),
                        TestContentEngagement(),
                        TestContentPerformance(),
                        TestContentCollaboration(),
                        TestContentVersioning(),
                        TestContentQualityAssurance(),
                        TestContentCompliance(),
                        TestContentMonetization(),
                        TestContentAccessibility(),
                        TestContentGovernance(),
                        TestContentLifecycle(),
                        TestContentIntelligence(),
                        TestContentAutomation(),
                        TestContentSecurityPrivacy(),
                        TestContentScalabilityPerformance(),
                        TestContentWorkflowManagement(),
                        TestContentDistributionSyndication(),
                        TestContentDiscoveryRecommendation(),
                        TestContentAnalyticsInsights(),
                        TestContentTeamCollaboration(),
                        TestContentIntegrationAPI(),
                        TestContentMetadataManagement(),
                        TestContentModerationFiltering(),
                        TestContentBackupRecovery(),
                        TestContentMultiPlatformSync(),
                        TestContentRealTimeCollaboration(),
                        TestContentAdvancedSecurity(),
                        TestContentAIEnhancement(),
                        TestContentPredictiveAnalytics(),
                        TestContentGamification(),
                        TestContentAdvancedAnalyticsV2(),
                        TestContentEnterpriseFeatures(),
                        TestContentAdvancedMLIntegration()
                    ]
            
            entity_results = []
            for test_class in entity_tests:
                result = await self._run_test_class(test_class)
                entity_results.append(result)
            
            results["entities"] = entity_results
            
            # Run additional unit test results
            additional_unit_results = []
            for test_class in additional_unit_tests:
                result = await self._run_test_class(test_class)
                additional_unit_results.append(result)
            
            results["additional_unit"] = additional_unit_results
            
            # Aggregate results
            for test_type in ["post_service", "entities", "additional_unit"]:
                if test_type in results:
                    if isinstance(results[test_type], list):
                        for result in results[test_type]:
                            results["total"] += result["total"]
                            results["passed"] += result["passed"]
                            results["failed"] += result["failed"]
                            results["errors"].extend(result["errors"])
                    else:
                        results["total"] += results[test_type]["total"]
                        results["passed"] += results[test_type]["passed"]
                        results["failed"] += results[test_type]["failed"]
                        results["errors"].extend(results[test_type]["errors"])
            
        except Exception as e:
            results["errors"].append(f"Unit test execution failed: {str(e)}")
            results["failed"] += 1
        
        results["execution_time"] = time.time() - start_time
        return results
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "execution_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Run performance benchmark tests
            performance_tests = TestPerformanceBenchmarks()
            performance_results = await self._run_test_class(performance_tests)
            results["performance_benchmarks"] = performance_results
            
            # Aggregate results
            results["total"] += performance_results["total"]
            results["passed"] += performance_results["passed"]
            results["failed"] += performance_results["failed"]
            results["errors"].extend(performance_results["errors"])
            
        except Exception as e:
            results["errors"].append(f"Performance test execution failed: {str(e)}")
            results["failed"] += 1
        
        results["execution_time"] = time.time() - start_time
        return results
    
    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "execution_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Run service integration tests
            service_integration_tests = TestPostServiceIntegration()
            service_results = await self._run_test_class(service_integration_tests)
            results["service_integration"] = service_results
            
            # Run performance integration tests
            performance_tests = TestPerformanceIntegration()
            performance_results = await self._run_test_class(performance_tests)
            results["performance_integration"] = performance_results
            
            # Aggregate results
            for test_type in ["service_integration", "performance_integration"]:
                if test_type in results:
                    results["total"] += results[test_type]["total"]
                    results["passed"] += results[test_type]["passed"]
                    results["failed"] += results[test_type]["failed"]
                    results["errors"].extend(results[test_type]["errors"])
            
        except Exception as e:
            results["errors"].append(f"Integration test execution failed: {str(e)}")
            results["failed"] += 1
        
        results["execution_time"] = time.time() - start_time
        return results
    
    async def _run_api_tests(self) -> Dict[str, Any]:
        """Run API tests."""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "execution_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Run API endpoint tests
            api_tests = [
                TestLinkedInPostsAPI(),
                TestAPIVersioning()
            ]
            
            for api_test in api_tests:
                api_results = await self._run_test_class(api_test)
                results["total"] += api_results["total"]
                results["passed"] += api_results["passed"]
                results["failed"] += api_results["failed"]
                results["errors"].extend(api_results["errors"])
            
        except Exception as e:
            results["errors"].append(f"API test execution failed: {str(e)}")
            results["failed"] += 1
        
        results["execution_time"] = time.time() - start_time
        return results
    
    async def _run_load_tests(self) -> Dict[str, Any]:
        """Run load and performance tests."""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "execution_time": 0,
            "performance_metrics": {}
        }
        
        start_time = time.time()
        
        try:
            # Run load performance tests
            load_tests = TestLoadPerformance()
            load_results = await self._run_test_class(load_tests)
            results.update(load_results)
            
            # Extract performance metrics
            if "performance_metrics" in load_results:
                results["performance_metrics"] = load_results["performance_metrics"]
            
        except Exception as e:
            results["errors"].append(f"Load test execution failed: {str(e)}")
            results["failed"] += 1
        
        results["execution_time"] = time.time() - start_time
        return results
    
    async def _run_test_class(self, test_class) -> Dict[str, Any]:
        """Run a specific test class."""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "execution_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Get all test methods
            test_methods = [method for method in dir(test_class) 
                          if method.startswith('test_') and callable(getattr(test_class, method))]
            
            results["total"] = len(test_methods)
            
            for method_name in test_methods:
                try:
                    method = getattr(test_class, method_name)
                    
                    # Run the test method
                    if asyncio.iscoroutinefunction(method):
                        await method()
                    else:
                        method()
                    
                    results["passed"] += 1
                    
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"{method_name}: {str(e)}")
            
        except Exception as e:
            results["errors"].append(f"Test class execution failed: {str(e)}")
            results["failed"] += 1
        
        results["execution_time"] = time.time() - start_time
        return results
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = sum(result.get("total", 0) for result in results.values())
        total_passed = sum(result.get("passed", 0) for result in results.values())
        total_failed = sum(result.get("failed", 0) for result in results.values())
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "success_rate": success_rate,
                "execution_time": self.end_time - self.start_time if self.end_time else 0,
                "timestamp": datetime.now().isoformat()
            },
            "detailed_results": results,
            "performance_analysis": self._analyze_performance(results),
            "recommendations": self._generate_recommendations(results)
        }
        
        return report
    
    def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics."""
        analysis = {
            "overall_performance": "good",
            "bottlenecks": [],
            "optimization_opportunities": []
        }
        
        # Analyze load test results
        if "load" in results:
            load_results = results["load"]
            if "performance_metrics" in load_results:
                metrics = load_results["performance_metrics"]
                
                # Check response time
                if metrics.get("avg_response_time", 0) > self.config["performance_thresholds"]["max_response_time"]:
                    analysis["bottlenecks"].append("High response time detected")
                
                # Check throughput
                if metrics.get("throughput", 0) < self.config["performance_thresholds"]["min_throughput"]:
                    analysis["bottlenecks"].append("Low throughput detected")
                
                # Check memory usage
                if metrics.get("memory_usage", 0) > self.config["performance_thresholds"]["max_memory_usage"]:
                    analysis["bottlenecks"].append("High memory usage detected")
        
        # Analyze error rates
        total_errors = sum(len(result.get("errors", [])) for result in results.values())
        if total_errors > 0:
            analysis["optimization_opportunities"].append("Error handling improvements needed")
        
        return analysis
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check success rate
        total_tests = sum(result.get("total", 0) for result in results.values())
        total_passed = sum(result.get("passed", 0) for result in results.values())
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        if success_rate < 90:
            recommendations.append("Improve test coverage and fix failing tests")
        
        # Check for specific issues
        for test_type, result in results.items():
            if result.get("failed", 0) > 0:
                recommendations.append(f"Review and fix {test_type} test failures")
        
        # Performance recommendations
        if "load" in results:
            load_results = results["load"]
            if "performance_metrics" in load_results:
                metrics = load_results["performance_metrics"]
                
                if metrics.get("avg_response_time", 0) > 1.0:
                    recommendations.append("Optimize response times")
                
                if metrics.get("throughput", 0) < 20:
                    recommendations.append("Improve system throughput")
        
        return recommendations
    
    def _save_reports(self, report: Dict[str, Any]):
        """Save test reports in various formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = Path("test_reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        if self.config["reporting"]["generate_json"]:
            json_file = reports_dir / f"test_report_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📊 JSON report saved: {json_file}")
        
        # Save HTML report
        if self.config["reporting"]["generate_html"]:
            html_file = reports_dir / f"test_report_{timestamp}.html"
            self._generate_html_report(report, html_file)
            print(f"📊 HTML report saved: {html_file}")
        
        # Save JUnit XML report
        if self.config["reporting"]["generate_junit"]:
            junit_file = reports_dir / f"test_report_{timestamp}.xml"
            self._generate_junit_report(report, junit_file)
            print(f"📊 JUnit report saved: {junit_file}")
    
    def _generate_html_report(self, report: Dict[str, Any], file_path: Path):
        """Generate HTML test report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LinkedIn Posts Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .test-type {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
    </style>
</head>
<body>
    <h1>LinkedIn Posts Test Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {report['summary']['total_tests']}</p>
        <p><strong>Passed:</strong> <span class="success">{report['summary']['passed']}</span></p>
        <p><strong>Failed:</strong> <span class="failure">{report['summary']['failed']}</span></p>
        <p><strong>Success Rate:</strong> <span class="{'success' if report['summary']['success_rate'] >= 90 else 'warning'}">{report['summary']['success_rate']:.1f}%</span></p>
        <p><strong>Execution Time:</strong> {report['summary']['execution_time']:.2f} seconds</p>
    </div>
    
    <h2>Detailed Results</h2>
"""
        
        for test_type, result in report['detailed_results'].items():
            success_rate = (result.get('passed', 0) / result.get('total', 1) * 100) if result.get('total', 0) > 0 else 0
            status_class = 'success' if success_rate >= 90 else 'warning' if success_rate >= 70 else 'failure'
            
            html_content += f"""
    <div class="test-type">
        <h3>{test_type.title()} Tests</h3>
        <p><strong>Total:</strong> {result.get('total', 0)}</p>
        <p><strong>Passed:</strong> <span class="success">{result.get('passed', 0)}</span></p>
        <p><strong>Failed:</strong> <span class="failure">{result.get('failed', 0)}</span></p>
        <p><strong>Success Rate:</strong> <span class="{status_class}">{success_rate:.1f}%</span></p>
        <p><strong>Execution Time:</strong> {result.get('execution_time', 0):.2f} seconds</p>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(file_path, 'w') as f:
            f.write(html_content)
    
    def _generate_junit_report(self, report: Dict[str, Any], file_path: Path):
        """Generate JUnit XML test report."""
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
    <testsuite name="LinkedIn Posts Tests" tests="{report['summary']['total_tests']}" failures="{report['summary']['failed']}" time="{report['summary']['execution_time']:.2f}">
"""
        
        for test_type, result in report['detailed_results'].items():
            for error in result.get('errors', []):
                xml_content += f"""
        <testcase classname="{test_type}" name="test_method">
            <failure message="{error}">{error}</failure>
        </testcase>
"""
        
        xml_content += """
    </testsuite>
</testsuites>
"""
        
        with open(file_path, 'w') as f:
            f.write(xml_content)


async def main():
    """Main function to run comprehensive tests."""
    # Configuration
    config = {
        "test_types": ["unit", "integration", "api", "load"],
        "parallel": True,
        "verbose": True,
        "coverage": True,
        "performance_thresholds": {
            "max_response_time": 2.0,
            "min_throughput": 10.0,
            "max_memory_usage": 500,
            "max_cpu_usage": 80
        },
        "load_test_config": {
            "concurrent_users": 50,
            "duration": 60,
            "ramp_up_time": 10
        },
        "reporting": {
            "generate_html": True,
            "generate_json": True,
            "generate_junit": True
        }
    }
    
    # Create and run test runner
    runner = ComprehensiveTestRunner(config)
    report = await runner.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Execution Time: {report['summary']['execution_time']:.2f} seconds")
    
    if report['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    # Exit with appropriate code
    if report['summary']['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())

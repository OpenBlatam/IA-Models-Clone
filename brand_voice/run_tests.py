#!/usr/bin/env python3
"""
Comprehensive Test Runner for Ultimate Brand Voice AI System
===========================================================

This script provides a comprehensive test runner for the Brand Voice AI system,
supporting different test configurations, parallel execution, and detailed reporting.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import unittest
import subprocess
import tempfile
import shutil

# Import test modules
from test_brand_ai_comprehensive import (
    TestBrandAITransformer, TestBrandAITraining, TestBrandAIServing,
    TestBrandAIAdvancedModels, TestBrandAIOptimization, TestBrandAIDeployment,
    TestBrandAIComputerVision, TestBrandAIMonitoring, TestBrandAITrendPrediction,
    TestBrandAIMultilingual, TestBrandAISentimentAnalysis, TestBrandAICompetitiveIntelligence,
    TestBrandAIAutomation, TestBrandAIVoiceCloning, TestBrandAICollaboration,
    TestBrandAIPerformancePrediction, TestBrandAIBlockchainVerification,
    TestBrandAICrisisManagement, TestIntegration, TestPerformance, TestSecurity
)

from test_brand_ai_performance import (
    TestBrandAIPerformance, TestBrandAILoadTesting, TestBrandAIBenchmarking,
    TestBrandAIMemoryProfiling, TestBrandAIScalability
)

# Import test configuration
from test_config import (
    TestConfig, get_quick_test_config, get_comprehensive_test_config,
    get_performance_test_config, get_integration_test_config,
    get_security_test_config, get_deployment_test_config,
    get_monitoring_test_config, setup_test_environment, teardown_test_environment
)

# Import test utilities
from test_utils import create_mock_config

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_runner.log')
    ]
)
logger = logging.getLogger(__name__)

class TestRunner:
    """Comprehensive test runner for Brand Voice AI system"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results = {}
        self.start_time = None
        self.end_time = None
        self.test_suites = self._get_test_suites()
        self.report_data = {
            'config': config.to_dict(),
            'start_time': None,
            'end_time': None,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'test_results': {},
            'performance_metrics': {},
            'system_info': self._get_system_info()
        }
    
    def _get_test_suites(self) -> Dict[str, List[unittest.TestCase]]:
        """Get test suites based on configuration"""
        suites = {}
        
        # Core module tests
        if self.config.transformer_tests:
            suites['transformer'] = [TestBrandAITransformer]
        
        if self.config.computer_vision_tests:
            suites['computer_vision'] = [TestBrandAIComputerVision]
        
        if self.config.sentiment_analysis_tests:
            suites['sentiment_analysis'] = [TestBrandAISentimentAnalysis]
        
        if self.config.voice_cloning_tests:
            suites['voice_cloning'] = [TestBrandAIVoiceCloning]
        
        if self.config.collaboration_tests:
            suites['collaboration'] = [TestBrandAICollaboration]
        
        if self.config.automation_tests:
            suites['automation'] = [TestBrandAIAutomation]
        
        if self.config.blockchain_tests:
            suites['blockchain'] = [TestBrandAIBlockchainVerification]
        
        if self.config.crisis_management_tests:
            suites['crisis_management'] = [TestBrandAICrisisManagement]
        
        # System tests
        if self.config.training_tests:
            suites['training'] = [TestBrandAITraining]
        
        if self.config.serving_tests:
            suites['serving'] = [TestBrandAIServing]
        
        if self.config.advanced_models_tests:
            suites['advanced_models'] = [TestBrandAIAdvancedModels]
        
        if self.config.optimization_tests:
            suites['optimization'] = [TestBrandAIOptimization]
        
        if self.config.deployment_tests:
            suites['deployment'] = [TestBrandAIDeployment]
        
        if self.config.monitoring_tests:
            suites['monitoring'] = [TestBrandAIMonitoring]
        
        if self.config.trend_prediction_tests:
            suites['trend_prediction'] = [TestBrandAITrendPrediction]
        
        if self.config.multilingual_tests:
            suites['multilingual'] = [TestBrandAIMultilingual]
        
        if self.config.competitive_intelligence_tests:
            suites['competitive_intelligence'] = [TestBrandAICompetitiveIntelligence]
        
        if self.config.performance_prediction_tests:
            suites['performance_prediction'] = [TestBrandAIPerformancePrediction]
        
        # Integration tests
        if self.config.integration_tests:
            suites['integration'] = [TestIntegration]
        
        if self.config.end_to_end_tests:
            suites['end_to_end'] = [TestIntegration]
        
        if self.config.cross_module_tests:
            suites['cross_module'] = [TestIntegration]
        
        # Performance tests
        if self.config.performance_tests:
            suites['performance'] = [TestBrandAIPerformance]
        
        if self.config.load_tests:
            suites['load_testing'] = [TestBrandAILoadTesting]
        
        if self.config.stress_tests:
            suites['stress_testing'] = [TestBrandAILoadTesting]
        
        if self.config.memory_profiling:
            suites['memory_profiling'] = [TestBrandAIMemoryProfiling]
        
        if self.config.benchmark_tests:
            suites['benchmarking'] = [TestBrandAIBenchmarking]
        
        if self.config.scalability_tests:
            suites['scalability'] = [TestBrandAIScalability]
        
        # Security tests
        if self.config.security_tests:
            suites['security'] = [TestSecurity]
        
        if self.config.authentication_tests:
            suites['authentication'] = [TestSecurity]
        
        if self.config.authorization_tests:
            suites['authorization'] = [TestSecurity]
        
        if self.config.input_validation_tests:
            suites['input_validation'] = [TestSecurity]
        
        return suites
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            'disk_usage': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }
    
    async def run_tests(self) -> Dict[str, Any]:
        """Run all configured tests"""
        logger.info("Starting Brand Voice AI System Tests")
        logger.info(f"Configuration: {self.config.to_dict()}")
        
        self.start_time = datetime.now()
        self.report_data['start_time'] = self.start_time.isoformat()
        
        # Setup test environment
        setup_test_environment(self.config)
        
        try:
            if self.config.parallel_execution:
                await self._run_tests_parallel()
            else:
                await self._run_tests_sequential()
            
            self.end_time = datetime.now()
            self.report_data['end_time'] = self.end_time.isoformat()
            
            # Generate reports
            if self.config.generate_reports:
                self._generate_reports()
            
            # Cleanup
            if self.config.cleanup_after_tests:
                teardown_test_environment(self.config)
            
            return self.report_data
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _run_tests_sequential(self):
        """Run tests sequentially"""
        for suite_name, test_classes in self.test_suites.items():
            logger.info(f"Running {suite_name} tests...")
            
            suite_results = []
            for test_class in test_classes:
                try:
                    result = await self._run_test_class(test_class)
                    suite_results.append(result)
                except Exception as e:
                    logger.error(f"Test class {test_class.__name__} failed: {e}")
                    suite_results.append({
                        'class_name': test_class.__name__,
                        'success': False,
                        'error': str(e),
                        'tests_run': 0,
                        'failures': 0,
                        'errors': 0
                    })
            
            self.report_data['test_results'][suite_name] = suite_results
            self._update_summary_stats(suite_results)
    
    async def _run_tests_parallel(self):
        """Run tests in parallel"""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            tasks = []
            
            for suite_name, test_classes in self.test_suites.items():
                for test_class in test_classes:
                    task = executor.submit(self._run_test_class_sync, test_class)
                    tasks.append((suite_name, test_class, task))
            
            # Collect results
            for suite_name, test_class, task in tasks:
                try:
                    result = task.result(timeout=self.config.timeout)
                    if suite_name not in self.report_data['test_results']:
                        self.report_data['test_results'][suite_name] = []
                    self.report_data['test_results'][suite_name].append(result)
                    self._update_summary_stats([result])
                except Exception as e:
                    logger.error(f"Parallel test execution failed for {test_class.__name__}: {e}")
                    if suite_name not in self.report_data['test_results']:
                        self.report_data['test_results'][suite_name] = []
                    self.report_data['test_results'][suite_name].append({
                        'class_name': test_class.__name__,
                        'success': False,
                        'error': str(e),
                        'tests_run': 0,
                        'failures': 0,
                        'errors': 0
                    })
    
    async def _run_test_class(self, test_class) -> Dict[str, Any]:
        """Run a single test class"""
        try:
            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            
            # Run tests
            runner = unittest.TextTestRunner(
                verbosity=2,
                stream=open(os.path.join(self.config.test_logs_dir, f"{test_class.__name__}.log"), 'w')
            )
            
            result = runner.run(suite)
            
            return {
                'class_name': test_class.__name__,
                'success': result.wasSuccessful(),
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
                'execution_time': getattr(result, 'execution_time', 0),
                'failures_details': [str(f) for f in result.failures],
                'errors_details': [str(e) for e in result.errors]
            }
            
        except Exception as e:
            logger.error(f"Error running test class {test_class.__name__}: {e}")
            return {
                'class_name': test_class.__name__,
                'success': False,
                'error': str(e),
                'tests_run': 0,
                'failures': 0,
                'errors': 0
            }
    
    def _run_test_class_sync(self, test_class) -> Dict[str, Any]:
        """Run a single test class synchronously (for parallel execution)"""
        try:
            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            
            # Run tests
            runner = unittest.TextTestRunner(verbosity=0)
            result = runner.run(suite)
            
            return {
                'class_name': test_class.__name__,
                'success': result.wasSuccessful(),
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
                'execution_time': getattr(result, 'execution_time', 0),
                'failures_details': [str(f) for f in result.failures],
                'errors_details': [str(e) for e in result.errors]
            }
            
        except Exception as e:
            logger.error(f"Error running test class {test_class.__name__}: {e}")
            return {
                'class_name': test_class.__name__,
                'success': False,
                'error': str(e),
                'tests_run': 0,
                'failures': 0,
                'errors': 0
            }
    
    def _update_summary_stats(self, results: List[Dict[str, Any]]):
        """Update summary statistics"""
        for result in results:
            self.report_data['total_tests'] += result.get('tests_run', 0)
            if result.get('success', False):
                self.report_data['passed_tests'] += result.get('tests_run', 0)
            else:
                self.report_data['failed_tests'] += result.get('failures', 0) + result.get('errors', 0)
            self.report_data['skipped_tests'] += result.get('skipped', 0)
    
    def _generate_reports(self):
        """Generate test reports"""
        logger.info("Generating test reports...")
        
        # JSON report
        json_report_path = os.path.join(self.config.test_reports_dir, "test_report.json")
        with open(json_report_path, 'w') as f:
            json.dump(self.report_data, f, indent=2, default=str)
        
        # HTML report
        if self.config.report_format in ['html', 'both']:
            html_report_path = os.path.join(self.config.test_reports_dir, "test_report.html")
            self._generate_html_report(html_report_path)
        
        # Summary report
        summary_report_path = os.path.join(self.config.test_reports_dir, "test_summary.txt")
        self._generate_summary_report(summary_report_path)
        
        logger.info(f"Reports generated in: {self.config.test_reports_dir}")
    
    def _generate_html_report(self, output_path: str):
        """Generate HTML test report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Brand Voice AI System Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .test-suite {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
                .test-suite-header {{ background-color: #f5f5f5; padding: 10px; font-weight: bold; }}
                .test-suite-content {{ padding: 15px; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                .error {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Brand Voice AI System Test Report</h1>
                <p>Generated on: {self.report_data['start_time']}</p>
                <p>Test Duration: {self._get_test_duration()}</p>
            </div>
            
            <div class="summary">
                <h2>Test Summary</h2>
                <p><strong>Total Tests:</strong> {self.report_data['total_tests']}</p>
                <p><strong>Passed:</strong> <span class="success">{self.report_data['passed_tests']}</span></p>
                <p><strong>Failed:</strong> <span class="failure">{self.report_data['failed_tests']}</span></p>
                <p><strong>Skipped:</strong> {self.report_data['skipped_tests']}</p>
                <p><strong>Success Rate:</strong> {(self.report_data['passed_tests']/max(self.report_data['total_tests'], 1)*100):.1f}%</p>
            </div>
            
            <h2>Test Results by Suite</h2>
            {self._generate_test_suite_html()}
            
            <h2>System Information</h2>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Platform</td><td>{self.report_data['system_info']['platform']}</td></tr>
                <tr><td>Python Version</td><td>{self.report_data['system_info']['python_version']}</td></tr>
                <tr><td>CPU Count</td><td>{self.report_data['system_info']['cpu_count']}</td></tr>
                <tr><td>Memory Total</td><td>{self.report_data['system_info']['memory_total']:.1f} GB</td></tr>
            </table>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_test_suite_html(self) -> str:
        """Generate HTML for test suites"""
        html = ""
        for suite_name, results in self.report_data['test_results'].items():
            html += f"""
            <div class="test-suite">
                <div class="test-suite-header">{suite_name.title()} Tests</div>
                <div class="test-suite-content">
                    <table>
                        <tr>
                            <th>Test Class</th>
                            <th>Status</th>
                            <th>Tests Run</th>
                            <th>Failures</th>
                            <th>Errors</th>
                            <th>Execution Time</th>
                        </tr>
            """
            
            for result in results:
                status_class = "success" if result.get('success', False) else "failure"
                status_text = "PASS" if result.get('success', False) else "FAIL"
                
                html += f"""
                        <tr>
                            <td>{result.get('class_name', 'Unknown')}</td>
                            <td class="{status_class}">{status_text}</td>
                            <td>{result.get('tests_run', 0)}</td>
                            <td>{result.get('failures', 0)}</td>
                            <td>{result.get('errors', 0)}</td>
                            <td>{result.get('execution_time', 0):.2f}s</td>
                        </tr>
                """
            
            html += """
                    </table>
                </div>
            </div>
            """
        
        return html
    
    def _generate_summary_report(self, output_path: str):
        """Generate summary text report"""
        with open(output_path, 'w') as f:
            f.write("Brand Voice AI System Test Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Start Time: {self.report_data['start_time']}\n")
            f.write(f"Test End Time: {self.report_data['end_time']}\n")
            f.write(f"Test Duration: {self._get_test_duration()}\n\n")
            
            f.write("Test Summary:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Tests: {self.report_data['total_tests']}\n")
            f.write(f"Passed: {self.report_data['passed_tests']}\n")
            f.write(f"Failed: {self.report_data['failed_tests']}\n")
            f.write(f"Skipped: {self.report_data['skipped_tests']}\n")
            f.write(f"Success Rate: {(self.report_data['passed_tests']/max(self.report_data['total_tests'], 1)*100):.1f}%\n\n")
            
            f.write("Test Results by Suite:\n")
            f.write("-" * 30 + "\n")
            for suite_name, results in self.report_data['test_results'].items():
                f.write(f"\n{suite_name.title()} Tests:\n")
                for result in results:
                    status = "PASS" if result.get('success', False) else "FAIL"
                    f.write(f"  {result.get('class_name', 'Unknown')}: {status} "
                           f"({result.get('tests_run', 0)} tests, "
                           f"{result.get('failures', 0)} failures, "
                           f"{result.get('errors', 0)} errors)\n")
    
    def _get_test_duration(self) -> str:
        """Get test duration as string"""
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            return str(duration)
        return "Unknown"

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Brand Voice AI System Test Runner")
    parser.add_argument("--config", type=str, help="Path to test configuration file")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--security", action="store_true", help="Run security tests only")
    parser.add_argument("--deployment", action="store_true", help="Run deployment tests only")
    parser.add_argument("--monitoring", action="store_true", help="Run monitoring tests only")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--timeout", type=int, default=300, help="Test timeout in seconds")
    parser.add_argument("--output-dir", type=str, help="Output directory for test results")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't cleanup test files")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine configuration
    if args.config:
        config = TestConfig.from_file(args.config)
    elif args.quick:
        config = get_quick_test_config()
    elif args.comprehensive:
        config = get_comprehensive_test_config()
    elif args.performance:
        config = get_performance_test_config()
    elif args.integration:
        config = get_integration_test_config()
    elif args.security:
        config = get_security_test_config()
    elif args.deployment:
        config = get_deployment_test_config()
    elif args.monitoring:
        config = get_monitoring_test_config()
    else:
        config = get_quick_test_config()  # Default to quick tests
    
    # Override configuration with command line arguments
    if args.parallel:
        config.parallel_execution = True
    if args.workers:
        config.max_workers = args.workers
    if args.timeout:
        config.timeout = args.timeout
    if args.output_dir:
        config.test_reports_dir = args.output_dir
    if args.no_cleanup:
        config.cleanup_after_tests = False
    
    # Run tests
    try:
        runner = TestRunner(config)
        results = asyncio.run(runner.run_tests())
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Skipped: {results['skipped_tests']}")
        print(f"Success Rate: {(results['passed_tests']/max(results['total_tests'], 1)*100):.1f}%")
        print(f"Test Duration: {runner._get_test_duration()}")
        print("=" * 60)
        
        if results['failed_tests'] > 0:
            print("\n❌ Some tests failed. Check the detailed reports for more information.")
            sys.exit(1)
        else:
            print("\n🎉 All tests passed successfully!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

























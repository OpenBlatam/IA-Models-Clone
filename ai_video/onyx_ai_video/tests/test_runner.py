from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import sys
import os
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pytest
import coverage
                from onyx_ai_video.core.models import VideoRequest, VideoResponse
                from onyx_ai_video.config.config_manager import OnyxConfigManager
                from onyx_ai_video.api.main import OnyxAIVideoSystem
                        import shutil
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Comprehensive test runner for Onyx AI Video System.

This script runs all tests (unit, integration, system) and generates
detailed reports with coverage analysis.
"""



class TestRunner:
    """Comprehensive test runner for Onyx AI Video System."""
    
    def __init__(self, project_root: str = None):
        """Initialize test runner."""
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.test_results = {}
        self.coverage_data = {}
        self.start_time = None
        self.end_time = None
        
        # Test categories
        self.test_categories = {
            "unit": {
                "path": "tests/unit",
                "description": "Unit tests for individual components",
                "timeout": 300
            },
            "integration": {
                "path": "tests/integration", 
                "description": "Integration tests for component interactions",
                "timeout": 600
            },
            "system": {
                "path": "tests/system",
                "description": "System tests for end-to-end functionality",
                "timeout": 1200
            }
        }
    
    def setup_environment(self) -> bool:
        """Setup test environment."""
        print("🔧 Setting up test environment...")
        
        try:
            # Create test directories
            test_dirs = ["test_outputs", "test_logs", "test_reports"]
            for dir_name in test_dirs:
                dir_path = self.project_root / dir_name
                dir_path.mkdir(exist_ok=True)
            
            # Check Python environment
            if not self._check_python_environment():
                print("❌ Python environment check failed")
                return False
            
            # Check dependencies
            if not self._check_dependencies():
                print("❌ Dependencies check failed")
                return False
            
            # Setup test configuration
            self._setup_test_config()
            
            print("✅ Test environment setup completed")
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return False
    
    def _check_python_environment(self) -> bool:
        """Check Python environment requirements."""
        try:
            # Check Python version
            if sys.version_info < (3, 8):
                print("❌ Python 3.8+ required")
                return False
            
            # Check required packages
            required_packages = [
                "pytest", "pytest-asyncio", "pytest-cov", "pytest-html",
                "coverage", "asyncio", "pathlib", "typing"
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package.replace("-", "_"))
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                print(f"❌ Missing packages: {missing_packages}")
                return False
            
            print("✅ Python environment check passed")
            return True
            
        except Exception as e:
            print(f"❌ Python environment check failed: {e}")
            return False
    
    def _check_dependencies(self) -> bool:
        """Check system dependencies."""
        try:
            # Check if we can import the main modules
            sys.path.insert(0, str(self.project_root))
            
            try:
                print("✅ Core modules import check passed")
            except ImportError as e:
                print(f"❌ Core modules import failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Dependencies check failed: {e}")
            return False
    
    def _setup_test_config(self) -> Any:
        """Setup test configuration."""
        test_config = {
            "test_outputs_dir": str(self.project_root / "test_outputs"),
            "test_logs_dir": str(self.project_root / "test_logs"),
            "test_reports_dir": str(self.project_root / "test_reports"),
            "coverage_enabled": True,
            "parallel_tests": True,
            "verbose_output": True
        }
        
        config_file = self.project_root / "test_config.json"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(test_config, f, indent=2)
    
    def run_tests(self, categories: List[str] = None, coverage: bool = True) -> Dict[str, Any]:
        """Run tests for specified categories."""
        if categories is None:
            categories = list(self.test_categories.keys())
        
        self.start_time = time.time()
        print(f"🚀 Starting test execution for categories: {categories}")
        
        results = {}
        
        for category in categories:
            if category not in self.test_categories:
                print(f"⚠️  Unknown test category: {category}")
                continue
            
            print(f"\n📋 Running {category} tests...")
            category_result = self._run_category_tests(category, coverage)
            results[category] = category_result
        
        self.end_time = time.time()
        self.test_results = results
        
        return results
    
    def _run_category_tests(self, category: str, coverage: bool) -> Dict[str, Any]:
        """Run tests for a specific category."""
        category_info = self.test_categories[category]
        test_path = self.project_root / category_info["path"]
        
        if not test_path.exists():
            return {
                "status": "skipped",
                "reason": f"Test path not found: {test_path}",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_skipped": 0,
                "duration": 0,
                "coverage": None
            }
        
        # Prepare pytest arguments
        pytest_args = [
            str(test_path),
            "-v",
            "--tb=short",
            f"--timeout={category_info['timeout']}",
            "--asyncio-mode=auto"
        ]
        
        # Add coverage if requested
        if coverage:
            pytest_args.extend([
                "--cov=onyx_ai_video",
                "--cov-report=html:" + str(self.project_root / "test_reports" / f"coverage_{category}"),
                "--cov-report=json:" + str(self.project_root / "test_reports" / f"coverage_{category}.json"),
                "--cov-report=term-missing"
            ])
        
        # Add HTML report
        pytest_args.extend([
            "--html=" + str(self.project_root / "test_reports" / f"report_{category}.html"),
            "--self-contained-html"
        ])
        
        # Add JUnit XML report
        pytest_args.extend([
            "--junitxml=" + str(self.project_root / "test_reports" / f"junit_{category}.xml")
        ])
        
        try:
            print(f"Running: pytest {' '.join(pytest_args)}")
            
            # Run pytest
            start_time = time.time()
            result = pytest.main(pytest_args)
            end_time = time.time()
            
            # Parse results
            test_result = self._parse_pytest_result(result, end_time - start_time)
            
            # Load coverage data if available
            if coverage:
                coverage_file = self.project_root / "test_reports" / f"coverage_{category}.json"
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        test_result["coverage"] = json.load(f)
            
            return test_result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_skipped": 0,
                "duration": 0,
                "coverage": None
            }
    
    def _parse_pytest_result(self, exit_code: int, duration: float) -> Dict[str, Any]:
        """Parse pytest exit code and return structured result."""
        if exit_code == 0:
            status = "passed"
        elif exit_code == 1:
            status = "failed"
        elif exit_code == 2:
            status = "error"
        elif exit_code == 5:
            status = "no_tests"
        else:
            status = "unknown"
        
        # Note: In a real implementation, you'd parse the actual test output
        # to get exact counts. This is a simplified version.
        return {
            "status": status,
            "exit_code": exit_code,
            "duration": duration,
            "tests_run": 0,  # Would be parsed from output
            "tests_passed": 0,  # Would be parsed from output
            "tests_failed": 0,  # Would be parsed from output
            "tests_skipped": 0,  # Would be parsed from output
            "coverage": None
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate comprehensive test report."""
        if not self.test_results:
            print("❌ No test results available")
            return ""
        
        print("\n📊 Generating test report...")
        
        # Calculate summary statistics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_duration = 0
        
        for category, result in self.test_results.items():
            total_tests += result.get("tests_run", 0)
            total_passed += result.get("tests_passed", 0)
            total_failed += result.get("tests_failed", 0)
            total_skipped += result.get("tests_skipped", 0)
            total_duration += result.get("duration", 0)
        
        # Generate report
        report = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "duration": self.end_time - self.start_time if self.end_time else 0,
                "categories_run": list(self.test_results.keys())
            },
            "summary": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_skipped": total_skipped,
                "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
                "total_duration": total_duration
            },
            "categories": self.test_results,
            "coverage_summary": self._generate_coverage_summary(),
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        if output_file is None:
            output_file = self.project_root / "test_reports" / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report, f, indent=2)
        
        # Generate HTML report
        html_report = self._generate_html_report(report)
        html_file = Path(output_file).with_suffix('.html')
        with open(html_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(html_report)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        print(f"✅ Test report generated: {output_file}")
        print(f"✅ HTML report generated: {html_file}")
        
        return str(output_file)
    
    def _generate_coverage_summary(self) -> Dict[str, Any]:
        """Generate coverage summary from all categories."""
        coverage_summary = {
            "overall_coverage": 0,
            "modules": {},
            "missing_lines": []
        }
        
        total_coverage = 0
        coverage_count = 0
        
        for category, result in self.test_results.items():
            if result.get("coverage"):
                coverage_data = result["coverage"]
                if "totals" in coverage_data:
                    total_coverage += coverage_data["totals"].get("percent_covered", 0)
                    coverage_count += 1
        
        if coverage_count > 0:
            coverage_summary["overall_coverage"] = total_coverage / coverage_count
        
        return coverage_summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Analyze test results and generate recommendations
        for category, result in self.test_results.items():
            if result.get("status") == "failed":
                recommendations.append(f"Fix failing {category} tests")
            
            if result.get("tests_failed", 0) > 0:
                recommendations.append(f"Review and fix {result['tests_failed']} failed {category} tests")
            
            if result.get("tests_skipped", 0) > 0:
                recommendations.append(f"Review {result['tests_skipped']} skipped {category} tests")
        
        # Coverage recommendations
        coverage_summary = self._generate_coverage_summary()
        if coverage_summary["overall_coverage"] < 80:
            recommendations.append("Increase test coverage to at least 80%")
        
        if not recommendations:
            recommendations.append("All tests passing! Great job!")
        
        return recommendations
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML test report."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Onyx AI Video System - Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .summary-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .category { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .status-passed { color: green; }
        .status-failed { color: red; }
        .status-error { color: orange; }
        .recommendations { background: #fff3cd; padding: 20px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Onyx AI Video System - Test Report</h1>
        <p>Generated on: {timestamp}</p>
        <p>Duration: {duration:.2f} seconds</p>
    </div>
    
    <div class="summary">
        <div class="summary-card">
            <h3>Total Tests</h3>
            <p style="font-size: 2em;">{total_tests}</p>
        </div>
        <div class="summary-card">
            <h3>Passed</h3>
            <p style="font-size: 2em; color: green;">{total_passed}</p>
        </div>
        <div class="summary-card">
            <h3>Failed</h3>
            <p style="font-size: 2em; color: red;">{total_failed}</p>
        </div>
        <div class="summary-card">
            <h3>Success Rate</h3>
            <p style="font-size: 2em;">{success_rate:.1f}%</p>
        </div>
    </div>
    
    <h2>Test Categories</h2>
    {categories_html}
    
    <h2>Coverage Summary</h2>
    <p>Overall Coverage: {coverage:.1f}%</p>
    
    <div class="recommendations">
        <h3>Recommendations</h3>
        <ul>
            {recommendations_html}
        </ul>
    </div>
</body>
</html>
        """
        
        # Generate categories HTML
        categories_html = ""
        for category, result in report["categories"].items():
            status_class = f"status-{result.get('status', 'unknown')}"
            categories_html += f"""
            <div class="category">
                <h3>{category.title()} Tests</h3>
                <p class="{status_class}">Status: {result.get('status', 'unknown')}</p>
                <p>Duration: {result.get('duration', 0):.2f} seconds</p>
                <p>Tests: {result.get('tests_passed', 0)} passed, {result.get('tests_failed', 0)} failed, {result.get('tests_skipped', 0)} skipped</p>
            </div>
            """
        
        # Generate recommendations HTML
        recommendations_html = ""
        for rec in report.get("recommendations", []):
            recommendations_html += f"<li>{rec}</li>"f"
        
        return html_template"
    
    def cleanup(self) -> Any:
        """Cleanup test artifacts."""
        print("\n🧹 Cleaning up test artifacts...")
        
        try:
            # Remove temporary files
            temp_patterns = ["*.pyc", "__pycache__", "*.tmp"]
            for pattern in temp_patterns:
                for file_path in self.project_root.rglob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Onyx AI Video System Test Runner")
    parser.add_argument("--categories", nargs="+", choices=["unit", "integration", "system"],
                       default=["unit", "integration", "system"],
                       help="Test categories to run")
    parser.add_argument("--no-coverage", action="store_true",
                       help="Disable coverage reporting")
    parser.add_argument("--output", type=str,
                       help="Output file for test report")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up test artifacts after completion")
    parser.add_argument("--project-root", type=str,
                       help="Project root directory")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(args.project_root)
    
    try:
        # Setup environment
        if not runner.setup_environment():
            print("❌ Environment setup failed")
            sys.exit(1)
        
        # Run tests
        results = runner.run_tests(args.categories, not args.no_coverage)
        
        # Generate report
        report_file = runner.generate_report(args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("TEST EXECUTION SUMMARY")
        print("="*60)
        
        for category, result in results.items():
            status_emoji = "✅" if result.get("status") == "passed" else "❌"
            print(f"{status_emoji} {category.title()}: {result.get('status', 'unknown')}")
        
        print(f"\n📊 Detailed report: {report_file}")
        
        # Cleanup if requested
        if args.cleanup:
            runner.cleanup()
        
        # Exit with appropriate code
        all_passed = all(result.get("status") == "passed" for result in results.values())
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        sys.exit(1)


match __name__:
    case "__main__":
    main() 
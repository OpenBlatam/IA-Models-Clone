"""
Playwright Test Runner
======================
Test runner utilities for executing and managing Playwright tests.
"""

import pytest
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = None
    trace_path: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Test suite result data structure."""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration: float
    results: List[TestResult]
    timestamp: float


class PlaywrightTestRunner:
    """Test runner for Playwright tests."""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[TestResult] = []
    
    def run_tests(
        self,
        test_path: str = "tests/test_playwright*.py",
        markers: Optional[List[str]] = None,
        verbose: bool = True
    ) -> TestSuiteResult:
        """Run Playwright tests."""
        start_time = time.time()
        
        # Build pytest command
        cmd = [test_path]
        
        if markers:
            marker_expr = " and ".join([f"mark.{m}" for m in markers])
            cmd.extend(["-m", marker_expr])
        
        if verbose:
            cmd.append("-v")
        
        # Run tests
        exit_code = pytest.main(cmd)
        
        duration = time.time() - start_time
        
        # Parse results (simplified - in real implementation would parse pytest output)
        return TestSuiteResult(
            suite_name="Playwright Tests",
            total_tests=0,  # Would be parsed from pytest output
            passed=0,
            failed=0,
            skipped=0,
            duration=duration,
            results=self.results,
            timestamp=time.time()
        )
    
    def save_results(self, suite_result: TestSuiteResult, filename: str = "test_results.json"):
        """Save test results to file."""
        file_path = self.output_dir / filename
        file_path.write_text(json.dumps(asdict(suite_result), indent=2))
        return file_path
    
    def generate_report(self, suite_result: TestSuiteResult) -> str:
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Playwright Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .skipped {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>Playwright Test Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Suite:</strong> {suite_result.suite_name}</p>
                <p><strong>Total Tests:</strong> {suite_result.total_tests}</p>
                <p class="passed"><strong>Passed:</strong> {suite_result.passed}</p>
                <p class="failed"><strong>Failed:</strong> {suite_result.failed}</p>
                <p class="skipped"><strong>Skipped:</strong> {suite_result.skipped}</p>
                <p><strong>Duration:</strong> {suite_result.duration:.2f}s</p>
            </div>
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Duration</th>
                </tr>
        """
        
        for result in suite_result.results:
            status_class = result.status
            html += f"""
                <tr>
                    <td>{result.test_name}</td>
                    <td class="{status_class}">{result.status}</td>
                    <td>{result.duration:.3f}s</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
    
    def save_html_report(self, suite_result: TestSuiteResult, filename: str = "test_report.html"):
        """Save HTML report to file."""
        html = self.generate_report(suite_result)
        file_path = self.output_dir / filename
        file_path.write_text(html)
        return file_path


class PlaywrightTestFilter:
    """Filter tests based on criteria."""
    
    @staticmethod
    def filter_by_marker(markers: List[str]) -> str:
        """Create pytest marker filter."""
        return " and ".join([f"mark.{m}" for m in markers])
    
    @staticmethod
    def filter_by_name(pattern: str) -> str:
        """Create pytest name filter."""
        return f"-k {pattern}"
    
    @staticmethod
    def filter_by_file(files: List[str]) -> List[str]:
        """Filter by file paths."""
        return files


class PlaywrightTestExecutor:
    """Execute Playwright tests with options."""
    
    def __init__(self):
        self.runner = PlaywrightTestRunner()
    
    def run_smoke_tests(self) -> TestSuiteResult:
        """Run smoke tests."""
        return self.runner.run_tests(markers=["smoke"])
    
    def run_critical_tests(self) -> TestSuiteResult:
        """Run critical tests."""
        return self.runner.run_tests(markers=["critical"])
    
    def run_fast_tests(self) -> TestSuiteResult:
        """Run fast tests."""
        return self.runner.run_tests(markers=["fast"])
    
    def run_all_tests(self) -> TestSuiteResult:
        """Run all tests."""
        return self.runner.run_tests()
    
    def run_with_coverage(self) -> TestSuiteResult:
        """Run tests with coverage."""
        # Would add coverage options
        return self.runner.run_tests()
    
    def run_parallel(self, workers: int = 4) -> TestSuiteResult:
        """Run tests in parallel."""
        # Would add parallel execution
        return self.runner.run_tests()




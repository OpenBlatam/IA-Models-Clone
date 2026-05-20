"""
Playwright Analytics and Metrics
=================================
Utilities for analyzing test results and generating metrics.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from datetime import datetime
from collections import defaultdict


@dataclass
class TestMetrics:
    """Test execution metrics."""
    test_name: str
    duration: float
    status: str
    requests_count: int
    failed_requests: int
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None


@dataclass
class SuiteMetrics:
    """Test suite metrics."""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    total_duration: float
    avg_test_duration: float
    total_requests: int
    failed_requests: int
    avg_response_time: float
    timestamp: float
    test_metrics: List[TestMetrics]


class PlaywrightAnalytics:
    """Analytics for Playwright tests."""
    
    def __init__(self, output_dir: str = "analytics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics: List[TestMetrics] = []
        self.suite_metrics: Optional[SuiteMetrics] = None
    
    def record_test_metrics(
        self,
        test_name: str,
        duration: float,
        status: str,
        requests: List[Dict[str, Any]],
        memory_usage: Optional[float] = None,
        cpu_usage: Optional[float] = None
    ):
        """Record metrics for a single test."""
        response_times = [
            r.get("duration", 0) for r in requests
            if "duration" in r
        ]
        
        failed = len([r for r in requests if r.get("status", 0) >= 400])
        
        metrics = TestMetrics(
            test_name=test_name,
            duration=duration,
            status=status,
            requests_count=len(requests),
            failed_requests=failed,
            avg_response_time=sum(response_times) / len(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def calculate_suite_metrics(self, suite_name: str = "Playwright Tests") -> SuiteMetrics:
        """Calculate metrics for the entire suite."""
        total_tests = len(self.metrics)
        passed = len([m for m in self.metrics if m.status == "passed"])
        failed = len([m for m in self.metrics if m.status == "failed"])
        skipped = len([m for m in self.metrics if m.status == "skipped"])
        
        total_duration = sum(m.duration for m in self.metrics)
        avg_test_duration = total_duration / total_tests if total_tests > 0 else 0
        
        total_requests = sum(m.requests_count for m in self.metrics)
        failed_requests = sum(m.failed_requests for m in self.metrics)
        
        all_response_times = [
            m.avg_response_time for m in self.metrics
            if m.avg_response_time > 0
        ]
        avg_response_time = (
            sum(all_response_times) / len(all_response_times)
            if all_response_times else 0
        )
        
        self.suite_metrics = SuiteMetrics(
            suite_name=suite_name,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            total_duration=total_duration,
            avg_test_duration=avg_test_duration,
            total_requests=total_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            timestamp=time.time(),
            test_metrics=self.metrics
        )
        
        return self.suite_metrics
    
    def generate_report(self, format: str = "json") -> Path:
        """Generate analytics report."""
        if not self.suite_metrics:
            self.calculate_suite_metrics()
        
        if format == "json":
            return self._generate_json_report()
        elif format == "html":
            return self._generate_html_report()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_json_report(self) -> Path:
        """Generate JSON report."""
        file_path = self.output_dir / f"analytics_{int(time.time())}.json"
        file_path.write_text(json.dumps(asdict(self.suite_metrics), indent=2))
        return file_path
    
    def _generate_html_report(self) -> Path:
        """Generate HTML report."""
        if not self.suite_metrics:
            self.calculate_suite_metrics()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Playwright Analytics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 5px; }}
                .chart {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .skipped {{ color: orange; }}
            </style>
        </head>
        <body>
            <h1>Playwright Analytics Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <div class="metric">
                    <strong>Total Tests:</strong> {self.suite_metrics.total_tests}
                </div>
                <div class="metric passed">
                    <strong>Passed:</strong> {self.suite_metrics.passed}
                </div>
                <div class="metric failed">
                    <strong>Failed:</strong> {self.suite_metrics.failed}
                </div>
                <div class="metric skipped">
                    <strong>Skipped:</strong> {self.suite_metrics.skipped}
                </div>
                <div class="metric">
                    <strong>Total Duration:</strong> {self.suite_metrics.total_duration:.2f}s
                </div>
                <div class="metric">
                    <strong>Avg Test Duration:</strong> {self.suite_metrics.avg_test_duration:.3f}s
                </div>
                <div class="metric">
                    <strong>Total Requests:</strong> {self.suite_metrics.total_requests}
                </div>
                <div class="metric">
                    <strong>Failed Requests:</strong> {self.suite_metrics.failed_requests}
                </div>
                <div class="metric">
                    <strong>Avg Response Time:</strong> {self.suite_metrics.avg_response_time:.3f}ms
                </div>
            </div>
            <h2>Test Metrics</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Requests</th>
                    <th>Failed Requests</th>
                    <th>Avg Response Time</th>
                </tr>
        """
        
        for metric in self.metrics:
            status_class = metric.status
            html += f"""
                <tr>
                    <td>{metric.test_name}</td>
                    <td class="{status_class}">{metric.status}</td>
                    <td>{metric.duration:.3f}s</td>
                    <td>{metric.requests_count}</td>
                    <td>{metric.failed_requests}</td>
                    <td>{metric.avg_response_time:.3f}ms</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        file_path = self.output_dir / f"analytics_{int(time.time())}.html"
        file_path.write_text(html)
        return file_path
    
    def compare_with_baseline(self, baseline_path: Path) -> Dict[str, Any]:
        """Compare current metrics with baseline."""
        baseline_data = json.loads(baseline_path.read_text())
        baseline_metrics = SuiteMetrics(**baseline_data)
        
        if not self.suite_metrics:
            self.calculate_suite_metrics()
        
        current = self.suite_metrics
        
        return {
            "duration_diff": current.total_duration - baseline_metrics.total_duration,
            "duration_percent_change": (
                (current.total_duration - baseline_metrics.total_duration) /
                baseline_metrics.total_duration * 100
            ) if baseline_metrics.total_duration > 0 else 0,
            "response_time_diff": current.avg_response_time - baseline_metrics.avg_response_time,
            "response_time_percent_change": (
                (current.avg_response_time - baseline_metrics.avg_response_time) /
                baseline_metrics.avg_response_time * 100
            ) if baseline_metrics.avg_response_time > 0 else 0,
            "failure_rate_diff": (
                (current.failed / current.total_tests) -
                (baseline_metrics.failed / baseline_metrics.total_tests)
            ) if current.total_tests > 0 and baseline_metrics.total_tests > 0 else 0
        }
    
    def identify_slow_tests(self, threshold: float = 5.0) -> List[TestMetrics]:
        """Identify tests that take longer than threshold."""
        return [m for m in self.metrics if m.duration > threshold]
    
    def identify_flaky_tests(self, test_runs: List[List[TestMetrics]]) -> List[str]:
        """Identify flaky tests across multiple runs."""
        test_results = defaultdict(list)
        
        for run in test_runs:
            for metric in run:
                test_results[metric.test_name].append(metric.status)
        
        flaky_tests = []
        for test_name, statuses in test_results.items():
            unique_statuses = set(statuses)
            if len(unique_statuses) > 1:  # Test has both passed and failed
                flaky_tests.append(test_name)
        
        return flaky_tests
    
    def generate_trends(self, historical_data: List[SuiteMetrics]) -> Dict[str, Any]:
        """Generate trends from historical data."""
        if not historical_data:
            return {}
        
        durations = [m.total_duration for m in historical_data]
        response_times = [m.avg_response_time for m in historical_data]
        failure_rates = [
            m.failed / m.total_tests if m.total_tests > 0 else 0
            for m in historical_data
        ]
        
        return {
            "duration_trend": {
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
                "trend": "increasing" if durations[-1] > durations[0] else "decreasing"
            },
            "response_time_trend": {
                "min": min(response_times),
                "max": max(response_times),
                "avg": sum(response_times) / len(response_times),
                "trend": "increasing" if response_times[-1] > response_times[0] else "decreasing"
            },
            "failure_rate_trend": {
                "min": min(failure_rates),
                "max": max(failure_rates),
                "avg": sum(failure_rates) / len(failure_rates),
                "trend": "increasing" if failure_rates[-1] > failure_rates[0] else "decreasing"
            }
        }


def create_analytics(output_dir: str = "analytics") -> PlaywrightAnalytics:
    """Create analytics instance."""
    return PlaywrightAnalytics(output_dir)




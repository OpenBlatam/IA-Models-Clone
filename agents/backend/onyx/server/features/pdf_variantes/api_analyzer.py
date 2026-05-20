#!/usr/bin/env python3
"""
API Analyzer
============
Advanced analysis tool for API performance and behavior.

⚠️ DEPRECATED: This file is deprecated. Consider migrating to the new tools structure.

For new code, use:
    from tools.manager import ToolManager
    manager = ToolManager()
    # Tools can be extended with analysis capabilities
"""
import warnings

warnings.warn(
    "api_analyzer.py is deprecated. Consider migrating to the new tools structure in tools/.",
    DeprecationWarning,
    stacklevel=2
)

import json
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict


@dataclass
class AnalysisResult:
    """Analysis result."""
    metric: str
    value: float
    threshold: Optional[float]
    status: str  # good, warning, critical
    recommendation: Optional[str]


class APIAnalyzer:
    """Advanced API analyzer."""
    
    def __init__(self):
        self.analyses: List[AnalysisResult] = []
    
    def analyze_health_check(self, health_file: Path) -> List[AnalysisResult]:
        """Analyze health check results."""
        with open(health_file, "r") as f:
            data = json.load(f)
        
        results = []
        
        # Overall status
        overall_status = data.get("overall_status", "unknown")
        if overall_status == "healthy":
            results.append(AnalysisResult(
                metric="Overall Health",
                value=100,
                threshold=95,
                status="good",
                recommendation=None
            ))
        elif overall_status == "degraded":
            results.append(AnalysisResult(
                metric="Overall Health",
                value=75,
                threshold=95,
                status="warning",
                recommendation="Some endpoints are degraded. Review error logs."
            ))
        else:
            results.append(AnalysisResult(
                metric="Overall Health",
                value=0,
                threshold=95,
                status="critical",
                recommendation="API is unhealthy. Immediate action required."
            ))
        
        # Response times
        checks = data.get("checks", [])
        response_times = [c.get("response_time", 0) for c in checks if c.get("response_time", 0) > 0]
        
        if response_times:
            avg_time = statistics.mean(response_times)
            max_time = max(response_times)
            
            if avg_time > 1000:
                results.append(AnalysisResult(
                    metric="Average Response Time",
                    value=avg_time,
                    threshold=500,
                    status="critical",
                    recommendation="Response times are too high. Consider optimization."
                ))
            elif avg_time > 500:
                results.append(AnalysisResult(
                    metric="Average Response Time",
                    value=avg_time,
                    threshold=500,
                    status="warning",
                    recommendation="Response times are elevated. Monitor closely."
                ))
            else:
                results.append(AnalysisResult(
                    metric="Average Response Time",
                    value=avg_time,
                    threshold=500,
                    status="good",
                    recommendation=None
                ))
            
            if max_time > 2000:
                results.append(AnalysisResult(
                    metric="Max Response Time",
                    value=max_time,
                    threshold=1000,
                    status="warning",
                    recommendation="Some requests are very slow. Investigate outliers."
                ))
        
        # Error rate
        unhealthy_count = sum(1 for c in checks if c.get("status") == "unhealthy")
        total_checks = len(checks)
        if total_checks > 0:
            error_rate = (unhealthy_count / total_checks) * 100
            
            if error_rate > 10:
                results.append(AnalysisResult(
                    metric="Error Rate",
                    value=error_rate,
                    threshold=5,
                    status="critical",
                    recommendation="High error rate detected. Review error logs immediately."
                ))
            elif error_rate > 5:
                results.append(AnalysisResult(
                    metric="Error Rate",
                    value=error_rate,
                    threshold=5,
                    status="warning",
                    recommendation="Elevated error rate. Monitor and investigate."
                ))
        
        self.analyses.extend(results)
        return results
    
    def analyze_benchmark(self, benchmark_file: Path) -> List[AnalysisResult]:
        """Analyze benchmark results."""
        with open(benchmark_file, "r") as f:
            data = json.load(f)
        
        results = []
        
        for result_data in data.get("results", []):
            endpoint = result_data.get("endpoint", "unknown")
            avg_time = result_data.get("avg_time", 0)
            p95_time = result_data.get("p95_time", 0)
            success_rate = result_data.get("success_rate", 0)
            
            # Analyze average time
            if avg_time > 1000:
                results.append(AnalysisResult(
                    metric=f"{endpoint} - Average Time",
                    value=avg_time,
                    threshold=500,
                    status="critical",
                    recommendation=f"Endpoint {endpoint} is too slow. Optimize."
                ))
            elif avg_time > 500:
                results.append(AnalysisResult(
                    metric=f"{endpoint} - Average Time",
                    value=avg_time,
                    threshold=500,
                    status="warning",
                    recommendation=f"Endpoint {endpoint} response time is elevated."
                ))
            
            # Analyze P95
            if p95_time > 2000:
                results.append(AnalysisResult(
                    metric=f"{endpoint} - P95 Time",
                    value=p95_time,
                    threshold=1000,
                    status="warning",
                    recommendation=f"95% of requests to {endpoint} exceed 1s. Consider optimization."
                ))
            
            # Analyze success rate
            if success_rate < 95:
                results.append(AnalysisResult(
                    metric=f"{endpoint} - Success Rate",
                    value=success_rate,
                    threshold=99,
                    status="critical",
                    recommendation=f"Endpoint {endpoint} has low success rate. Investigate failures."
                ))
            elif success_rate < 99:
                results.append(AnalysisResult(
                    metric=f"{endpoint} - Success Rate",
                    value=success_rate,
                    threshold=99,
                    status="warning",
                    recommendation=f"Endpoint {endpoint} success rate is below optimal."
                ))
        
        self.analyses.extend(results)
        return results
    
    def analyze_test_results(self, test_file: Path) -> List[AnalysisResult]:
        """Analyze test results."""
        with open(test_file, "r") as f:
            data = json.load(f)
        
        results = []
        summary = data.get("summary", {})
        
        total = summary.get("total", 0)
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        
        if total > 0:
            pass_rate = (passed / total) * 100
            
            if pass_rate < 80:
                results.append(AnalysisResult(
                    metric="Test Pass Rate",
                    value=pass_rate,
                    threshold=95,
                    status="critical",
                    recommendation="Low test pass rate. Review failing tests."
                ))
            elif pass_rate < 95:
                results.append(AnalysisResult(
                    metric="Test Pass Rate",
                    value=pass_rate,
                    threshold=95,
                    status="warning",
                    recommendation="Some tests are failing. Review and fix."
                ))
            else:
                results.append(AnalysisResult(
                    metric="Test Pass Rate",
                    value=pass_rate,
                    threshold=95,
                    status="good",
                    recommendation=None
                ))
        
        self.analyses.extend(results)
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate analysis report."""
        critical = [a for a in self.analyses if a.status == "critical"]
        warnings = [a for a in self.analyses if a.status == "warning"]
        good = [a for a in self.analyses if a.status == "good"]
        
        return {
            "summary": {
                "total": len(self.analyses),
                "critical": len(critical),
                "warnings": len(warnings),
                "good": len(good)
            },
            "critical_issues": [asdict(a) for a in critical],
            "warnings": [asdict(a) for a in warnings],
            "good_metrics": [asdict(a) for a in good],
            "generated_at": datetime.now().isoformat()
        }
    
    def print_report(self):
        """Print analysis report."""
        report = self.generate_report()
        
        print("\n" + "=" * 70)
        print("📊 API Analysis Report")
        print("=" * 70)
        
        print(f"\nSummary:")
        print(f"  Total Metrics: {report['summary']['total']}")
        print(f"  ✅ Good: {report['summary']['good']}")
        print(f"  ⚠️  Warnings: {report['summary']['warnings']}")
        print(f"  ❌ Critical: {report['summary']['critical']}")
        
        if report["critical_issues"]:
            print(f"\n❌ Critical Issues ({len(report['critical_issues'])}):")
            for issue in report["critical_issues"]:
                print(f"  • {issue['metric']}: {issue['value']:.2f}")
                if issue.get("recommendation"):
                    print(f"    Recommendation: {issue['recommendation']}")
        
        if report["warnings"]:
            print(f"\n⚠️  Warnings ({len(report['warnings'])}):")
            for warning in report["warnings"]:
                print(f"  • {warning['metric']}: {warning['value']:.2f}")
                if warning.get("recommendation"):
                    print(f"    Recommendation: {warning['recommendation']}")
        
        if report["good_metrics"]:
            print(f"\n✅ Good Metrics ({len(report['good_metrics'])}):")
            for metric in report["good_metrics"][:5]:  # Show first 5
                print(f"  • {metric['metric']}: {metric['value']:.2f}")
        
        print("\n" + "=" * 70)
    
    def export_report(self, file_path: Path):
        """Export analysis report."""
        report = self.generate_report()
        
        with open(file_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Analysis report exported to {file_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Analyzer")
    parser.add_argument("--health", help="Health check JSON file")
    parser.add_argument("--benchmark", help="Benchmark JSON file")
    parser.add_argument("--tests", help="Test results JSON file")
    parser.add_argument("--export", help="Export report to file")
    
    args = parser.parse_args()
    
    analyzer = APIAnalyzer()
    
    if args.health:
        analyzer.analyze_health_check(Path(args.health))
    
    if args.benchmark:
        analyzer.analyze_benchmark(Path(args.benchmark))
    
    if args.tests:
        analyzer.analyze_test_results(Path(args.tests))
    
    analyzer.print_report()
    
    if args.export:
        analyzer.export_report(Path(args.export))


if __name__ == "__main__":
    main()




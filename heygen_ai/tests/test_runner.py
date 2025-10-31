from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import tempfile
import shutil
import platform
import sys
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
HeyGen AI Testing Framework - Comprehensive Test Runner
======================================================

A sophisticated test runner that provides comprehensive testing capabilities
for the HeyGen AI FastAPI service following clean architecture principles.

Features:
- Multiple test categories (unit, integration, performance, e2e)
- Parallel test execution
- Coverage reporting
- Performance benchmarking
- Test result analysis
- CI/CD integration support
- Custom test filtering
- Detailed reporting

Usage:
    python tests/test_runner.py                    # Run all tests
    python tests/test_runner.py --unit             # Unit tests only
    python tests/test_runner.py --integration      # Integration tests only
    python tests/test_runner.py --performance      # Performance tests only
    python tests/test_runner.py --coverage         # With coverage
    python tests/test_runner.py --parallel         # Parallel execution
    python tests/test_runner.py --verbose          # Verbose output
    python tests/test_runner.py --benchmark        # Include benchmarks
"""


# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Comprehensive test runner for HeyGen AI service."""
    
    @staticmethod
    def run_tests(
        categories: List[str] = None,
        parallel: bool = False,
        coverage: bool = False,
        verbose: bool = False,
        benchmark: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run tests with specified configuration.
        
        Args:
            categories: List of test categories to run
            parallel: Enable parallel test execution
            coverage: Enable coverage reporting
            verbose: Enable verbose output
            benchmark: Enable performance benchmarking
            output_dir: Output directory for reports
            
        Returns:
            Dictionary containing test results and metrics
        """
        start_time = time.time()
        
        if categories is None:
            categories = ['unit', 'integration', 'performance']
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = current_dir / "reports"
            output_path.mkdir(exist_ok=True)
        
        print("🚀 HeyGen AI Testing Framework")
        print("=" * 50)
        print(f"Categories: {', '.join(categories)}")
        print(f"Parallel: {parallel}")
        print(f"Coverage: {coverage}")
        print(f"Benchmark: {benchmark}")
        print(f"Output: {output_path}")
        print("=" * 50)
        
        results = {}
        coverage_data = {}
        performance_metrics = {}
        
        # Run each test category
        for category in categories:
            print(f"\n🧪 Running {category} tests...")
            category_results = TestRunner._run_category(
                category=category,
                parallel=parallel,
                coverage=coverage,
                verbose=verbose,
                benchmark=benchmark,
                output_dir=output_path
            )
            results[category] = category_results
        
        # Generate comprehensive report
        report = TestRunner._generate_report(output_path, results, coverage_data, performance_metrics)
        
        # Print summary
        TestRunner._print_summary(results, coverage_data, performance_metrics)
        
        return {
            'results': results,
            'coverage': coverage_data,
            'performance': performance_metrics,
            'report_path': str(output_path / "test_report.json"),
            'duration': time.time() - start_time
        }
    
    @staticmethod
    def _run_category(
        category: str,
        parallel: bool,
        coverage: bool,
        verbose: bool,
        benchmark: bool,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Run tests for a specific category."""
        category_start = time.time()
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        # Add test directory for category
        test_path = TestRunner._get_test_path(category)
        if test_path.exists():
            cmd.append(str(test_path))
        else:
            print(f"⚠️  Warning: Test path {test_path} does not exist")
            return {'status': 'skipped', 'reason': 'path_not_found'}
        
        # Add markers
        cmd.extend(["-m", category])
        
        # Add coverage if requested
        if coverage:
            cmd.extend([
                "--cov=heygen_ai",
                "--cov-report=html:" + str(output_dir / f"coverage_{category}"),
                "--cov-report=xml:" + str(output_dir / f"coverage_{category}.xml"),
                "--cov-report=term-missing"
            ])
        
        # Add parallel execution
        if parallel:
            cmd.extend(["-n", "auto"])
        
        # Add verbose output
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Add benchmark if requested
        if benchmark and category in ['unit', 'performance']:
            cmd.append("--benchmark-only")
            cmd.append("--benchmark-json=" + str(output_dir / f"benchmark_{category}.json"))
        
        # Add output files
        cmd.extend([
            "--junitxml=" + str(output_dir / f"junit_{category}.xml"),
            "--html=" + str(output_dir / f"report_{category}.html"),
            "--self-contained-html"
        ])
        
        # Add timeout
        cmd.extend(["--timeout=300"])
        
        # Run tests
        try:
            print(f"   Command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes max
            )
            
            duration = time.time() - category_start
            
            # Parse results
            return {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'returncode': result.returncode,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'duration': time.time() - category_start,
                'error': 'Test execution timed out'
            }
        except Exception as e:
            return {
                'status': 'error',
                'duration': time.time() - category_start,
                'error': str(e)
            }
    
    @staticmethod
    def _get_test_path(category: str) -> Path:
        """Get the test path for a category."""
        if category == 'unit':
            return current_dir / "unit"
        elif category == 'integration':
            return current_dir / "integration"
        elif category == 'performance':
            return current_dir / "performance"
        elif category == 'e2e':
            return current_dir / "e2e"
        else:
            return current_dir
    
    @staticmethod
    def _generate_report(output_dir: Path, results: Dict[str, Any], coverage_data: Dict[str, Any], performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'results': results,
            'summary': TestRunner._calculate_summary(results),
            'environment': TestRunner._get_environment_info(),
            'coverage': TestRunner._collect_coverage_data(output_dir, results),
            'performance': TestRunner._collect_performance_data(output_dir, results)
        }
        
        # Save JSON report
        report_file = output_dir / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate HTML report
        TestRunner._generate_html_report(report, output_dir)
        
        return report
    
    @staticmethod
    def _calculate_summary(results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate test summary statistics."""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        total_duration = 0
        
        for category, result in results.items():
            if result.get('status') == 'passed':
                passed_tests += 1
            elif result.get('status') == 'failed':
                failed_tests += 1
            elif result.get('status') == 'skipped':
                skipped_tests += 1
            
            total_tests += result.get('total', 0)
            total_duration += result.get('duration', 0)
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'skipped': skipped_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'total_duration': total_duration
        }
    
    @staticmethod
    def _get_environment_info() -> Dict[str, Any]:
        """Collect environment information."""
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'working_directory': str(Path.cwd()),
            'project_root': str(project_root)
        }
    
    @staticmethod
    def _collect_coverage_data(output_dir: Path, results: Dict[str, Any]) -> Dict[str, Any]:
        """Collect coverage data from reports."""
        final_coverage_data = {}
        
        for category in results.keys():
            coverage_file = output_dir / f"coverage_{category}.xml"
            if coverage_file.exists():
                # Parse coverage XML (simplified)
                final_coverage_data[category] = {
                    'file': str(coverage_file),
                    'exists': True
                }
        
        return final_coverage_data
    
    @staticmethod
    def _collect_performance_data(output_dir: Path, results: Dict[str, Any]) -> Dict[str, Any]:
        """Collect performance benchmark data."""
        final_performance_data = {}
        
        for category in results.keys():
            benchmark_file = output_dir / f"benchmark_{category}.json"
            if benchmark_file.exists():
                try:
                    with open(benchmark_file, 'r') as f:
                        data = json.load(f)
                        final_performance_data[category] = data
                except Exception as e:
                    final_performance_data[category] = {'error': str(e)}
        
        return final_performance_data
    
    @staticmethod
    def _generate_html_report(report: Dict[str, Any], output_dir: Path):
        """Generate HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HeyGen AI Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .category {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 HeyGen AI Test Report</h1>
                <p>Generated: {report['timestamp']}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {report['summary']['total_tests']}</p>
                <p>Passed: {report['summary']['passed']}</p>
                <p>Failed: {report['summary']['failed']}</p>
                <p>Success Rate: {report['summary']['success_rate']:.1f}%</p>
            </div>
            
            <div class="categories">
                <h2>Test Categories</h2>
                {''.join([f'<div class="category"><h3>{cat}</h3><p>Status: {result.get("status", "unknown")}</p></div>' for cat, result in report['results'].items()])}
            </div>
        </body>
        </html>
        """
        
        html_file = output_dir / "test_report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
    
    @staticmethod
    def _print_summary(results: Dict[str, Any], coverage_data: Dict[str, Any], performance_data: Dict[str, Any]):
        """Print test execution summary."""
        summary = TestRunner._calculate_summary(results)
        
        print("\n" + "=" * 50)
        print("📊 TEST EXECUTION SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⏭️ Skipped: {summary['skipped']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️ Total Duration: {summary['total_duration']:.2f}s")
        print("=" * 50)
        
        # Print detailed results
        for category, result in results.items():
            status_icon = {
                'passed': '✅',
                'failed': '❌',
                'skipped': '⏭️',
                'error': '⚠️'
            }.get(result.get('status', 'unknown'), '❓')
            
            print(f"{status_icon} {category.upper()}: {result.get('status', 'unknown')}")
            if result.get('total'):
                print(f"   Tests: {result.get('total')}")
            if result.get('duration'):
                print(f"   Duration: {result.get('duration'):.2f}s")
        
        print("=" * 50)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="HeyGen AI Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Test categories
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--e2e', action='store_true', help='Run end-to-end tests')
    parser.add_argument('--all', action='store_true', help='Run all test categories')
    
    # Execution options
    parser.add_argument('--parallel', action='store_true', help='Enable parallel execution')
    parser.add_argument('--coverage', action='store_true', help='Enable coverage reporting')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--benchmark', action='store_true', help='Enable performance benchmarking')
    
    # Output options
    parser.add_argument('--output-dir', type=str, help='Output directory for reports')
    parser.add_argument('--no-report', action='store_true', help='Skip report generation')
    
    # CI/CD options
    parser.add_argument('--ci', action='store_true', help='CI/CD mode (optimized for automation)')
    parser.add_argument('--fail-fast', action='store_true', help='Stop on first failure')
    
    args = parser.parse_args()
    
    # Determine categories to run
    categories = []
    if args.unit or args.all:
        categories.append('unit')
    if args.integration or args.all:
        categories.append('integration')
    if args.performance or args.all:
        categories.append('performance')
    if args.e2e or args.all:
        categories.append('e2e')
    
    # Default to unit tests if nothing specified
    if not categories:
        categories = ['unit']
    
    # CI mode adjustments
    if args.ci:
        args.parallel = True
        args.coverage = True
        if not args.output_dir:
            args.output_dir = "ci_reports"
    
    # Create and run test runner
    runner = TestRunner()
    
    try:
        results = runner.run_tests(
            categories=categories,
            parallel=args.parallel,
            coverage=args.coverage,
            verbose=args.verbose,
            benchmark=args.benchmark,
            output_dir=args.output_dir
        )
        
        # Exit with appropriate code
        summary = results.get('results', {})
        failed_categories = [cat for cat, result in summary.items() 
                           if result.get('status') in ['failed', 'error', 'timeout']]
        
        if failed_categories:
            print(f"\n❌ Tests failed in categories: {', '.join(failed_categories)}")
            sys.exit(1)
        else:
            print("\n✅ All tests passed!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
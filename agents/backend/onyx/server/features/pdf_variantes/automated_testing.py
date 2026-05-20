#!/usr/bin/env python3
"""
Automated Testing Pipeline
==========================
Complete automated testing pipeline for API.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class AutomatedTestingPipeline:
    """Automated testing pipeline."""
    
    def __init__(self, base_url: str = "http://localhost:8000", output_dir: Path = Path("test_results")):
        self.base_url = base_url
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results: Dict[str, Any] = {}
    
    def run_health_check(self) -> bool:
        """Run health check."""
        print("🔍 Running health check...")
        
        output_file = self.output_dir / f"health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        result = subprocess.run(
            [sys.executable, "api_health_checker.py", "--url", self.base_url, "--export", str(output_file)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            with open(output_file, "r") as f:
                data = json.load(f)
                self.results["health"] = {
                    "status": data.get("overall_status"),
                    "file": str(output_file)
                }
                return data.get("overall_status") == "healthy"
        else:
            print(f"❌ Health check failed: {result.stderr}")
            return False
    
    def run_tests(self, suite_file: Optional[Path] = None) -> bool:
        """Run test suite."""
        print("🧪 Running test suite...")
        
        output_file = self.output_dir / f"tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        cmd = [sys.executable, "api_test_suite.py", "--url", self.base_url, "--export", str(output_file)]
        if suite_file:
            cmd.extend(["--suite", str(suite_file)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            with open(output_file, "r") as f:
                data = json.load(f)
                summary = data.get("summary", {})
                self.results["tests"] = {
                    "passed": summary.get("passed", 0),
                    "failed": summary.get("failed", 0),
                    "total": summary.get("total", 0),
                    "file": str(output_file)
                }
                return summary.get("failed", 0) == 0
        else:
            print(f"❌ Tests failed: {result.stderr}")
            return False
    
    def run_benchmark(self, endpoint: str = "/health", iterations: int = 100) -> bool:
        """Run benchmark."""
        print(f"🔥 Running benchmark on {endpoint}...")
        
        output_file = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        result = subprocess.run(
            [
                sys.executable, "api_benchmark.py",
                "--url", self.base_url,
                "--endpoint", endpoint,
                "--iterations", str(iterations),
                "--export", str(output_file)
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            with open(output_file, "r") as f:
                data = json.load(f)
                self.results["benchmark"] = {
                    "file": str(output_file),
                    "results": data.get("results", [])
                }
                return True
        else:
            print(f"❌ Benchmark failed: {result.stderr}")
            return False
    
    def generate_report(self) -> Path:
        """Generate comprehensive report."""
        print("📄 Generating comprehensive report...")
        
        report_file = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        cmd = [sys.executable, "api_reporter.py", "--output", str(report_file)]
        
        if "health" in self.results:
            cmd.extend(["--health", self.results["health"]["file"]])
        
        if "tests" in self.results:
            cmd.extend(["--tests", self.results["tests"]["file"]])
        
        if "benchmark" in self.results:
            cmd.extend(["--benchmark", self.results["benchmark"]["file"]])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            self.results["report"] = str(report_file)
            return report_file
        else:
            print(f"❌ Report generation failed: {result.stderr}")
            return None
    
    def run_full_pipeline(self, suite_file: Optional[Path] = None) -> bool:
        """Run full testing pipeline."""
        print("=" * 70)
        print("🚀 Starting Automated Testing Pipeline")
        print("=" * 70)
        print(f"Base URL: {self.base_url}")
        print(f"Output Directory: {self.output_dir}")
        print()
        
        success = True
        
        # Health check
        if not self.run_health_check():
            print("⚠️  Health check failed, but continuing...")
            success = False
        
        print()
        
        # Tests
        if not self.run_tests(suite_file):
            print("❌ Tests failed")
            success = False
        
        print()
        
        # Benchmark
        if not self.run_benchmark():
            print("⚠️  Benchmark failed, but continuing...")
        
        print()
        
        # Generate report
        report_file = self.generate_report()
        if report_file:
            print(f"✅ Report generated: {report_file}")
        
        print()
        print("=" * 70)
        if success:
            print("✅ Pipeline completed successfully")
        else:
            print("⚠️  Pipeline completed with errors")
        print("=" * 70)
        
        return success


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Testing Pipeline")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--suite", help="Test suite JSON file")
    parser.add_argument("--output", default="test_results", help="Output directory")
    parser.add_argument("--health-only", action="store_true", help="Run only health check")
    parser.add_argument("--tests-only", action="store_true", help="Run only tests")
    parser.add_argument("--benchmark-only", action="store_true", help="Run only benchmark")
    
    args = parser.parse_args()
    
    pipeline = AutomatedTestingPipeline(
        base_url=args.url,
        output_dir=Path(args.output)
    )
    
    if args.health_only:
        pipeline.run_health_check()
    elif args.tests_only:
        pipeline.run_tests(Path(args.suite) if args.suite else None)
    elif args.benchmark_only:
        pipeline.run_benchmark()
    else:
        pipeline.run_full_pipeline(Path(args.suite) if args.suite else None)


if __name__ == "__main__":
    main()




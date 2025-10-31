#!/usr/bin/env python3
"""
Test Coverage Analyzer for HeyGen AI
====================================

Advanced coverage analysis and reporting system.
"""

import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET

@dataclass
class CoverageMetrics:
    """Coverage metrics for a module or file"""
    name: str
    statements: int
    missing: int
    covered: int
    percentage: float
    branches: Optional[int] = None
    branch_missing: Optional[int] = None
    branch_covered: Optional[int] = None
    branch_percentage: Optional[float] = None

@dataclass
class CoverageReport:
    """Complete coverage report"""
    total_statements: int
    total_missing: int
    total_covered: int
    total_percentage: float
    modules: List[CoverageMetrics]
    generated_at: datetime
    test_duration: float

class CoverageAnalyzer:
    """Advanced coverage analyzer for HeyGen AI"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.core_dir = self.base_dir / "core"
        self.test_dir = self.base_dir / "tests"
        self.coverage_data: Optional[Dict[str, Any]] = None
        self.xml_coverage_data: Optional[ET.Element] = None
    
    def run_coverage_analysis(self) -> CoverageReport:
        """Run comprehensive coverage analysis"""
        print("📊 Running Coverage Analysis...")
        print("=" * 50)
        
        start_time = time.time()
        
        # Run pytest with coverage
        print("🔬 Running tests with coverage...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                str(self.test_dir),
                "--cov=core",
                "--cov-report=json",
                "--cov-report=xml",
                "--cov-report=term-missing",
                "-v"
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"⚠️ Tests completed with issues (return code: {result.returncode})")
                print("Coverage data may be incomplete.")
            
        except subprocess.TimeoutExpired:
            print("⏰ Coverage analysis timed out after 10 minutes")
        except Exception as e:
            print(f"❌ Error running coverage analysis: {e}")
        
        end_time = time.time()
        test_duration = end_time - start_time
        
        # Load coverage data
        self.load_coverage_data()
        
        # Generate report
        report = self.generate_coverage_report(test_duration)
        
        return report
    
    def load_coverage_data(self):
        """Load coverage data from JSON and XML files"""
        print("📁 Loading coverage data...")
        
        # Load JSON coverage data
        json_file = self.base_dir / "coverage.json"
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.coverage_data = json.load(f)
                print("  ✅ JSON coverage data loaded")
            except Exception as e:
                print(f"  ❌ Error loading JSON coverage: {e}")
        else:
            print("  ⚠️ JSON coverage file not found")
        
        # Load XML coverage data
        xml_file = self.base_dir / "coverage.xml"
        if xml_file.exists():
            try:
                self.xml_coverage_data = ET.parse(xml_file).getroot()
                print("  ✅ XML coverage data loaded")
            except Exception as e:
                print(f"  ❌ Error loading XML coverage: {e}")
        else:
            print("  ⚠️ XML coverage file not found")
    
    def analyze_module_coverage(self, module_name: str) -> CoverageMetrics:
        """Analyze coverage for a specific module"""
        if not self.coverage_data or "files" not in self.coverage_data:
            return CoverageMetrics(
                name=module_name,
                statements=0,
                missing=0,
                covered=0,
                percentage=0.0
            )
        
        # Find the module in coverage data
        module_path = None
        for file_path in self.coverage_data["files"]:
            if module_name in file_path:
                module_path = file_path
                break
        
        if not module_path:
            return CoverageMetrics(
                name=module_name,
                statements=0,
                missing=0,
                covered=0,
                percentage=0.0
            )
        
        file_data = self.coverage_data["files"][module_path]
        
        statements = file_data.get("summary", {}).get("num_statements", 0)
        missing = file_data.get("summary", {}).get("missing_lines", 0)
        covered = statements - missing
        percentage = file_data.get("summary", {}).get("percent_covered", 0.0)
        
        return CoverageMetrics(
            name=module_name,
            statements=statements,
            missing=missing,
            covered=covered,
            percentage=percentage
        )
    
    def get_core_modules(self) -> List[str]:
        """Get list of core modules to analyze"""
        if not self.core_dir.exists():
            return []
        
        modules = []
        for py_file in self.core_dir.glob("*.py"):
            if py_file.name != "__init__.py":
                modules.append(py_file.stem)
        
        return modules
    
    def generate_coverage_report(self, test_duration: float) -> CoverageReport:
        """Generate comprehensive coverage report"""
        print("📋 Generating coverage report...")
        
        # Get overall coverage from JSON data
        total_statements = 0
        total_missing = 0
        total_covered = 0
        total_percentage = 0.0
        
        if self.coverage_data and "totals" in self.coverage_data:
            totals = self.coverage_data["totals"]
            total_statements = totals.get("num_statements", 0)
            total_missing = totals.get("missing_lines", 0)
            total_covered = total_statements - total_missing
            total_percentage = totals.get("percent_covered", 0.0)
        
        # Analyze individual modules
        modules = []
        core_modules = self.get_core_modules()
        
        for module_name in core_modules:
            module_metrics = self.analyze_module_coverage(module_name)
            modules.append(module_metrics)
        
        # Sort modules by coverage percentage (descending)
        modules.sort(key=lambda x: x.percentage, reverse=True)
        
        report = CoverageReport(
            total_statements=total_statements,
            total_missing=total_missing,
            total_covered=total_covered,
            total_percentage=total_percentage,
            modules=modules,
            generated_at=datetime.now(),
            test_duration=test_duration
        )
        
        return report
    
    def print_coverage_report(self, report: CoverageReport):
        """Print formatted coverage report"""
        print("\n📊 HeyGen AI Coverage Report")
        print("=" * 60)
        print(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test Duration: {report.test_duration:.2f} seconds")
        print("")
        
        # Overall coverage
        print("🎯 Overall Coverage:")
        print(f"  Total Statements: {report.total_statements:,}")
        print(f"  Covered: {report.total_covered:,}")
        print(f"  Missing: {report.total_missing:,}")
        print(f"  Coverage: {report.total_percentage:.1f}%")
        print("")
        
        # Coverage bar
        coverage_bar = self.create_coverage_bar(report.total_percentage)
        print(f"  [{coverage_bar}] {report.total_percentage:.1f}%")
        print("")
        
        # Module coverage
        print("📁 Module Coverage:")
        print("-" * 60)
        print(f"{'Module':<25} {'Statements':<12} {'Covered':<8} {'Missing':<8} {'%':<6}")
        print("-" * 60)
        
        for module in report.modules:
            coverage_bar = self.create_coverage_bar(module.percentage, width=20)
            print(f"{module.name:<25} {module.statements:<12} {module.covered:<8} {module.missing:<8} {module.percentage:<6.1f}")
            if module.percentage < 80:
                print(f"  {'':25} [{coverage_bar}]")
        
        print("-" * 60)
        print("")
        
        # Coverage recommendations
        self.print_coverage_recommendations(report)
    
    def create_coverage_bar(self, percentage: float, width: int = 50) -> str:
        """Create a visual coverage bar"""
        filled = int((percentage / 100) * width)
        empty = width - filled
        
        if percentage >= 90:
            char = "█"
        elif percentage >= 70:
            char = "▓"
        elif percentage >= 50:
            char = "▒"
        else:
            char = "░"
        
        return char * filled + "░" * empty
    
    def print_coverage_recommendations(self, report: CoverageReport):
        """Print coverage improvement recommendations"""
        print("💡 Coverage Recommendations:")
        print("-" * 40)
        
        # Find modules with low coverage
        low_coverage = [m for m in report.modules if m.percentage < 80]
        high_coverage = [m for m in report.modules if m.percentage >= 90]
        
        if low_coverage:
            print("🔍 Modules needing attention:")
            for module in low_coverage:
                print(f"  • {module.name}: {module.percentage:.1f}% ({module.missing} lines missing)")
        
        if high_coverage:
            print("✅ Well-tested modules:")
            for module in high_coverage:
                print(f"  • {module.name}: {module.percentage:.1f}%")
        
        # Overall recommendations
        if report.total_percentage < 70:
            print("\n📈 Overall Recommendations:")
            print("  • Add more unit tests for core functionality")
            print("  • Test edge cases and error conditions")
            print("  • Add integration tests for module interactions")
        elif report.total_percentage < 90:
            print("\n📈 Overall Recommendations:")
            print("  • Focus on testing remaining uncovered lines")
            print("  • Add tests for error handling paths")
            print("  • Consider adding property-based tests")
        else:
            print("\n🏆 Excellent coverage! Consider:")
            print("  • Adding performance tests")
            print("  • Testing with different data sets")
            print("  • Adding mutation testing")
    
    def save_coverage_report(self, report: CoverageReport, filename: str = "coverage_analysis.json"):
        """Save coverage report to JSON file"""
        report_data = {
            "generated_at": report.generated_at.isoformat(),
            "test_duration": report.test_duration,
            "overall": {
                "total_statements": report.total_statements,
                "total_missing": report.total_missing,
                "total_covered": report.total_covered,
                "total_percentage": report.total_percentage
            },
            "modules": [
                {
                    "name": module.name,
                    "statements": module.statements,
                    "missing": module.missing,
                    "covered": module.covered,
                    "percentage": module.percentage
                }
                for module in report.modules
            ]
        }
        
        report_file = self.base_dir / filename
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Coverage report saved to: {report_file}")
    
    def generate_html_report(self):
        """Generate HTML coverage report"""
        print("🌐 Generating HTML coverage report...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                str(self.test_dir),
                "--cov=core",
                "--cov-report=html",
                "-q"
            ], capture_output=True, text=True, timeout=300)
            
            html_dir = self.base_dir / "htmlcov"
            if html_dir.exists():
                print(f"  ✅ HTML report generated: {html_dir / 'index.html'}")
            else:
                print("  ⚠️ HTML report directory not found")
                
        except Exception as e:
            print(f"  ❌ Error generating HTML report: {e}")

def main():
    """Main coverage analysis function"""
    analyzer = CoverageAnalyzer()
    
    # Run coverage analysis
    report = analyzer.run_coverage_analysis()
    
    # Print report
    analyzer.print_coverage_report(report)
    
    # Save report
    analyzer.save_coverage_report(report)
    
    # Generate HTML report
    analyzer.generate_html_report()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())






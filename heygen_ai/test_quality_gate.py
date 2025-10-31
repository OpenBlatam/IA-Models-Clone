#!/usr/bin/env python3
"""
Test Quality Gate for HeyGen AI
===============================

Advanced quality gate system for continuous integration and deployment.
"""

import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class QualityLevel(Enum):
    """Quality levels for the quality gate"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"

@dataclass
class QualityMetric:
    """Individual quality metric"""
    name: str
    value: float
    threshold: float
    unit: str
    status: QualityLevel
    description: str

@dataclass
class QualityGateResult:
    """Result of quality gate evaluation"""
    overall_status: QualityLevel
    overall_score: float
    metrics: List[QualityMetric]
    passed_gates: int
    total_gates: int
    recommendations: List[str]
    generated_at: datetime

class QualityGate:
    """Advanced quality gate system for HeyGen AI"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.test_dir = self.base_dir / "tests"
        self.core_dir = self.base_dir / "core"
        self.metrics: List[QualityMetric] = []
        self.thresholds = {
            "test_coverage": 80.0,
            "test_success_rate": 95.0,
            "test_execution_time": 300.0,  # seconds
            "code_complexity": 10.0,
            "duplicate_code": 5.0,
            "security_issues": 0.0,
            "linting_errors": 0.0,
            "documentation_coverage": 70.0
        }
    
    def run_test_coverage_analysis(self) -> QualityMetric:
        """Analyze test coverage"""
        print("📊 Analyzing test coverage...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                str(self.test_dir),
                "--cov=core",
                "--cov-report=json",
                "-q"
            ], capture_output=True, text=True, timeout=300)
            
            # Load coverage data
            coverage_file = self.base_dir / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r', encoding='utf-8') as f:
                    coverage_data = json.load(f)
                
                total_percentage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
            else:
                total_percentage = 0.0
            
        except Exception as e:
            print(f"  ⚠️ Error analyzing coverage: {e}")
            total_percentage = 0.0
        
        threshold = self.thresholds["test_coverage"]
        status = self._evaluate_metric(total_percentage, threshold, higher_is_better=True)
        
        return QualityMetric(
            name="Test Coverage",
            value=total_percentage,
            threshold=threshold,
            unit="%",
            status=status,
            description="Percentage of code covered by tests"
        )
    
    def run_test_success_analysis(self) -> QualityMetric:
        """Analyze test success rate"""
        print("✅ Analyzing test success rate...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                str(self.test_dir),
                "--tb=no",
                "-q"
            ], capture_output=True, text=True, timeout=300)
            
            # Parse pytest output to get success rate
            output_lines = result.stdout.split('\n')
            total_tests = 0
            failed_tests = 0
            
            for line in output_lines:
                if 'failed' in line and 'passed' in line:
                    # Parse line like "5 failed, 95 passed in 2.34s"
                    parts = line.split(',')
                    for part in parts:
                        if 'failed' in part:
                            failed_tests = int(part.strip().split()[0])
                        elif 'passed' in part:
                            passed_tests = int(part.strip().split()[0])
                            total_tests = passed_tests + failed_tests
            
            if total_tests > 0:
                success_rate = ((total_tests - failed_tests) / total_tests) * 100
            else:
                success_rate = 0.0
                
        except Exception as e:
            print(f"  ⚠️ Error analyzing test success: {e}")
            success_rate = 0.0
        
        threshold = self.thresholds["test_success_rate"]
        status = self._evaluate_metric(success_rate, threshold, higher_is_better=True)
        
        return QualityMetric(
            name="Test Success Rate",
            value=success_rate,
            threshold=threshold,
            unit="%",
            status=status,
            description="Percentage of tests that pass"
        )
    
    def run_test_execution_time_analysis(self) -> QualityMetric:
        """Analyze test execution time"""
        print("⏱️ Analyzing test execution time...")
        
        try:
            start_time = time.time()
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                str(self.test_dir),
                "--tb=no",
                "-q"
            ], capture_output=True, text=True, timeout=600)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
        except subprocess.TimeoutExpired:
            execution_time = 600.0  # Timeout
        except Exception as e:
            print(f"  ⚠️ Error analyzing execution time: {e}")
            execution_time = 600.0
        
        threshold = self.thresholds["test_execution_time"]
        status = self._evaluate_metric(execution_time, threshold, higher_is_better=False)
        
        return QualityMetric(
            name="Test Execution Time",
            value=execution_time,
            threshold=threshold,
            unit="seconds",
            status=status,
            description="Total time to run all tests"
        )
    
    def run_linting_analysis(self) -> QualityMetric:
        """Analyze code linting"""
        print("🔍 Analyzing code linting...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "flake8",
                str(self.core_dir),
                "--count",
                "--statistics"
            ], capture_output=True, text=True, timeout=120)
            
            # Count linting errors
            error_count = 0
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'E' in line or 'W' in line or 'F' in line:
                        try:
                            count = int(line.split()[-1])
                            error_count += count
                        except (ValueError, IndexError):
                            continue
            
        except Exception as e:
            print(f"  ⚠️ Error analyzing linting: {e}")
            error_count = 999  # High error count if analysis fails
        
        threshold = self.thresholds["linting_errors"]
        status = self._evaluate_metric(error_count, threshold, higher_is_better=False)
        
        return QualityMetric(
            name="Linting Errors",
            value=error_count,
            threshold=threshold,
            unit="errors",
            status=status,
            description="Number of linting errors found"
        )
    
    def run_security_analysis(self) -> QualityMetric:
        """Analyze security issues"""
        print("🔒 Analyzing security issues...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "bandit",
                "-r", str(self.core_dir),
                "-f", "json"
            ], capture_output=True, text=True, timeout=120)
            
            # Parse bandit output
            security_issues = 0
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    security_issues = len(bandit_data.get("results", []))
                except json.JSONDecodeError:
                    security_issues = 0
            
        except Exception as e:
            print(f"  ⚠️ Error analyzing security: {e}")
            security_issues = 999  # High count if analysis fails
        
        threshold = self.thresholds["security_issues"]
        status = self._evaluate_metric(security_issues, threshold, higher_is_better=False)
        
        return QualityMetric(
            name="Security Issues",
            value=security_issues,
            threshold=threshold,
            unit="issues",
            status=status,
            description="Number of security issues found"
        )
    
    def run_documentation_analysis(self) -> QualityMetric:
        """Analyze documentation coverage"""
        print("📚 Analyzing documentation coverage...")
        
        try:
            total_functions = 0
            documented_functions = 0
            
            for py_file in self.core_dir.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    in_function = False
                    has_docstring = False
                    
                    for line in lines:
                        stripped = line.strip()
                        
                        if stripped.startswith('def ') or stripped.startswith('async def '):
                            if in_function and not has_docstring:
                                total_functions += 1
                            else:
                                total_functions += 1
                                if has_docstring:
                                    documented_functions += 1
                            
                            in_function = True
                            has_docstring = False
                        
                        elif in_function and stripped.startswith('"""') or stripped.startswith("'''"):
                            has_docstring = True
                        
                        elif in_function and stripped and not stripped.startswith('#'):
                            if not has_docstring:
                                total_functions += 1
                            else:
                                documented_functions += 1
                            in_function = False
                            has_docstring = False
            
            if total_functions > 0:
                doc_coverage = (documented_functions / total_functions) * 100
            else:
                doc_coverage = 0.0
                
        except Exception as e:
            print(f"  ⚠️ Error analyzing documentation: {e}")
            doc_coverage = 0.0
        
        threshold = self.thresholds["documentation_coverage"]
        status = self._evaluate_metric(doc_coverage, threshold, higher_is_better=True)
        
        return QualityMetric(
            name="Documentation Coverage",
            value=doc_coverage,
            threshold=threshold,
            unit="%",
            status=status,
            description="Percentage of functions with documentation"
        )
    
    def _evaluate_metric(self, value: float, threshold: float, higher_is_better: bool = True) -> QualityLevel:
        """Evaluate a metric against its threshold"""
        if higher_is_better:
            if value >= threshold:
                return QualityLevel.EXCELLENT
            elif value >= threshold * 0.8:
                return QualityLevel.GOOD
            elif value >= threshold * 0.6:
                return QualityLevel.FAIR
            elif value >= threshold * 0.4:
                return QualityLevel.POOR
            else:
                return QualityLevel.FAILED
        else:
            if value <= threshold:
                return QualityLevel.EXCELLENT
            elif value <= threshold * 1.2:
                return QualityLevel.GOOD
            elif value <= threshold * 1.5:
                return QualityLevel.FAIR
            elif value <= threshold * 2.0:
                return QualityLevel.POOR
            else:
                return QualityLevel.FAILED
    
    def run_quality_gate(self) -> QualityGateResult:
        """Run complete quality gate evaluation"""
        print("🚪 Running Quality Gate Evaluation")
        print("=" * 50)
        
        # Run all quality analyses
        self.metrics = [
            self.run_test_coverage_analysis(),
            self.run_test_success_analysis(),
            self.run_test_execution_time_analysis(),
            self.run_linting_analysis(),
            self.run_security_analysis(),
            self.run_documentation_analysis()
        ]
        
        # Calculate overall status
        passed_gates = sum(1 for metric in self.metrics if metric.status in [QualityLevel.EXCELLENT, QualityLevel.GOOD])
        total_gates = len(self.metrics)
        
        # Calculate overall score
        status_scores = {
            QualityLevel.EXCELLENT: 100,
            QualityLevel.GOOD: 80,
            QualityLevel.FAIR: 60,
            QualityLevel.POOR: 40,
            QualityLevel.FAILED: 0
        }
        
        overall_score = sum(status_scores[metric.status] for metric in self.metrics) / len(self.metrics)
        
        # Determine overall status
        if overall_score >= 90:
            overall_status = QualityLevel.EXCELLENT
        elif overall_score >= 80:
            overall_status = QualityLevel.GOOD
        elif overall_score >= 60:
            overall_status = QualityLevel.FAIR
        elif overall_score >= 40:
            overall_status = QualityLevel.POOR
        else:
            overall_status = QualityLevel.FAILED
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        result = QualityGateResult(
            overall_status=overall_status,
            overall_score=overall_score,
            metrics=self.metrics,
            passed_gates=passed_gates,
            total_gates=total_gates,
            recommendations=recommendations,
            generated_at=datetime.now()
        )
        
        return result
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        for metric in self.metrics:
            if metric.status == QualityLevel.FAILED:
                if metric.name == "Test Coverage":
                    recommendations.append("🚨 CRITICAL: Increase test coverage to at least 80%")
                elif metric.name == "Test Success Rate":
                    recommendations.append("🚨 CRITICAL: Fix failing tests to achieve 95% success rate")
                elif metric.name == "Linting Errors":
                    recommendations.append("🚨 CRITICAL: Fix all linting errors")
                elif metric.name == "Security Issues":
                    recommendations.append("🚨 CRITICAL: Address all security vulnerabilities")
            elif metric.status == QualityLevel.POOR:
                if metric.name == "Test Coverage":
                    recommendations.append("⚠️ Improve test coverage by adding more test cases")
                elif metric.name == "Test Execution Time":
                    recommendations.append("⚠️ Optimize test execution time by parallelizing tests")
                elif metric.name == "Documentation Coverage":
                    recommendations.append("⚠️ Add documentation to functions and classes")
        
        if not recommendations:
            recommendations.append("✅ All quality gates are passing! Keep up the excellent work!")
        
        return recommendations
    
    def print_quality_report(self, result: QualityGateResult):
        """Print formatted quality gate report"""
        print("\n🚪 HeyGen AI Quality Gate Report")
        print("=" * 60)
        print(f"Generated: {result.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Overall Status: {result.overall_status.value.upper()}")
        print(f"Overall Score: {result.overall_score:.1f}/100")
        print(f"Gates Passed: {result.passed_gates}/{result.total_gates}")
        print("")
        
        # Quality status indicator
        status_emoji = {
            QualityLevel.EXCELLENT: "🏆",
            QualityLevel.GOOD: "✅",
            QualityLevel.FAIR: "⚠️",
            QualityLevel.POOR: "❌",
            QualityLevel.FAILED: "🚨"
        }
        
        print(f"{status_emoji[result.overall_status]} Quality Gate: {result.overall_status.value.upper()}")
        print("")
        
        # Individual metrics
        print("📊 Quality Metrics:")
        print("-" * 60)
        print(f"{'Metric':<25} {'Value':<12} {'Threshold':<12} {'Status':<10}")
        print("-" * 60)
        
        for metric in result.metrics:
            status_icon = status_emoji[metric.status]
            print(f"{metric.name:<25} {metric.value:<12.1f} {metric.threshold:<12.1f} {status_icon} {metric.status.value}")
        
        print("-" * 60)
        print("")
        
        # Recommendations
        print("💡 Recommendations:")
        print("-" * 40)
        for recommendation in result.recommendations:
            print(f"  {recommendation}")
        print("")
        
        # Quality gate decision
        if result.overall_status in [QualityLevel.EXCELLENT, QualityLevel.GOOD]:
            print("🎉 QUALITY GATE PASSED - Ready for deployment!")
        else:
            print("🚫 QUALITY GATE FAILED - Address issues before deployment")
    
    def save_quality_report(self, result: QualityGateResult, filename: str = "quality_gate_report.json"):
        """Save quality gate report to JSON file"""
        report_data = {
            "generated_at": result.generated_at.isoformat(),
            "overall_status": result.overall_status.value,
            "overall_score": result.overall_score,
            "passed_gates": result.passed_gates,
            "total_gates": result.total_gates,
            "metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "threshold": metric.threshold,
                    "unit": metric.unit,
                    "status": metric.status.value,
                    "description": metric.description
                }
                for metric in result.metrics
            ],
            "recommendations": result.recommendations
        }
        
        report_file = self.base_dir / filename
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Quality gate report saved to: {report_file}")

def main():
    """Main quality gate function"""
    quality_gate = QualityGate()
    
    # Run quality gate evaluation
    result = quality_gate.run_quality_gate()
    
    # Print report
    quality_gate.print_quality_report(result)
    
    # Save report
    quality_gate.save_quality_report(result)
    
    # Return appropriate exit code
    if result.overall_status in [QualityLevel.EXCELLENT, QualityLevel.GOOD]:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())






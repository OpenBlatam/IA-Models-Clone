"""
Test summary generator for reporting
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import json


class TestSummaryGenerator:
    """Generate test summaries and reports"""
    
    @staticmethod
    def generate_test_summary(test_results: Dict[str, Any]) -> str:
        """Generate a human-readable test summary"""
        total = test_results.get("total", 0)
        passed = test_results.get("passed", 0)
        failed = test_results.get("failed", 0)
        skipped = test_results.get("skipped", 0)
        
        summary = f"""
{'='*70}
TEST SUMMARY
{'='*70}
Total Tests: {total}
Passed: {passed} ({passed/total*100:.1f}%)
Failed: {failed} ({failed/total*100:.1f}%)
Skipped: {skipped} ({skipped/total*100:.1f}%)
{'='*70}
"""
        return summary
    
    @staticmethod
    def generate_coverage_summary(coverage_data: Dict[str, float]) -> str:
        """Generate coverage summary"""
        summary = f"""
{'='*70}
COVERAGE SUMMARY
{'='*70}
"""
        for module, coverage in coverage_data.items():
            summary += f"{module}: {coverage:.2f}%\n"
        
        summary += f"{'='*70}\n"
        return summary
    
    @staticmethod
    def save_test_report(output_path: Path, report_data: Dict[str, Any]):
        """Save test report to file"""
        report_data["generated_at"] = datetime.now().isoformat()
        
        output_path.write_text(
            json.dumps(report_data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8"
        )


@pytest.fixture
def test_summary_generator():
    """Fixture for test summary generator"""
    return TestSummaryGenerator


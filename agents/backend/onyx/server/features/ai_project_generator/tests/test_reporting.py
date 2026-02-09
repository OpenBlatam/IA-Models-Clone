"""
Test reporting and analytics
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import json


class TestReporting:
    """Tests for reporting functionality"""
    
    def test_generate_test_report(self, temp_dir):
        """Test generating test report"""
        from .test_summary_generator import TestSummaryGenerator
        
        test_results = {
            "total": 100,
            "passed": 95,
            "failed": 3,
            "skipped": 2
        }
        
        summary = TestSummaryGenerator.generate_test_summary(test_results)
        
        assert "Total Tests: 100" in summary
        assert "Passed: 95" in summary
        assert "Failed: 3" in summary
    
    def test_generate_coverage_report(self, temp_dir):
        """Test generating coverage report"""
        from .test_summary_generator import TestSummaryGenerator
        
        coverage_data = {
            "overall": 99.5,
            "core": 99.8,
            "utils": 99.5,
            "api": 99.5
        }
        
        summary = TestSummaryGenerator.generate_coverage_summary(coverage_data)
        
        assert "overall: 99.50%" in summary
        assert "core: 99.80%" in summary
    
    def test_save_test_report(self, temp_dir):
        """Test saving test report"""
        from .test_summary_generator import TestSummaryGenerator
        
        report_data = {
            "test_results": {
                "total": 100,
                "passed": 95,
                "failed": 3,
                "skipped": 2
            },
            "coverage": {
                "overall": 99.5
            }
        }
        
        report_path = temp_dir / "test_report.json"
        TestSummaryGenerator.save_test_report(report_path, report_data)
        
        assert report_path.exists()
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        assert "test_results" in loaded
        assert "generated_at" in loaded


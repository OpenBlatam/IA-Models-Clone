"""
Export Testing Helpers
Specialized helpers for export functionality testing
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock
import json
import csv
import io
from datetime import datetime


class ExportTestHelpers:
    """Helpers for export testing"""
    
    @staticmethod
    def create_mock_exporter(
        export_format: str = "json"
    ) -> Mock:
        """Create mock exporter"""
        exporter = Mock()
        exporter.format = export_format
        
        async def export_side_effect(data: List[Dict[str, Any]], **kwargs):
            if export_format == "json":
                return json.dumps(data).encode()
            elif export_format == "csv":
                output = io.StringIO()
                if data:
                    writer = csv.DictWriter(output, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                return output.getvalue().encode()
            return b"exported_data"
        
        exporter.export = AsyncMock(side_effect=export_side_effect)
        return exporter
    
    @staticmethod
    def assert_export_valid(
        export_data: bytes,
        expected_format: str = "json"
    ):
        """Assert export data is valid"""
        assert export_data is not None, "Export data is None"
        assert len(export_data) > 0, "Export data is empty"
        
        if expected_format == "json":
            try:
                json.loads(export_data.decode())
            except json.JSONDecodeError:
                raise AssertionError("Export data is not valid JSON")
        elif expected_format == "csv":
            # Basic CSV validation
            assert b"," in export_data or b"\n" in export_data, \
                "Export data does not appear to be CSV"


class ReportHelpers:
    """Helpers for report generation testing"""
    
    @staticmethod
    def create_mock_report_generator(
        report_type: str = "analysis"
    ) -> Mock:
        """Create mock report generator"""
        generator = Mock()
        generator.report_type = report_type
        
        async def generate_side_effect(data: Dict[str, Any], format: str = "json"):
            return {
                "report_id": "report-123",
                "type": report_type,
                "format": format,
                "generated_at": datetime.utcnow().isoformat(),
                "data": data
            }
        
        generator.generate = AsyncMock(side_effect=generate_side_effect)
        return generator
    
    @staticmethod
    def assert_report_generated(
        generator: Mock,
        report_type: Optional[str] = None
    ):
        """Assert report was generated"""
        assert generator.generate.called, "Report was not generated"
        if report_type:
            # Additional validation can check report type
            pass


# Convenience exports
create_mock_exporter = ExportTestHelpers.create_mock_exporter
assert_export_valid = ExportTestHelpers.assert_export_valid

create_mock_report_generator = ReportHelpers.create_mock_report_generator
assert_report_generated = ReportHelpers.assert_report_generated




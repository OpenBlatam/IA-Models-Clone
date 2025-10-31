from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
import aiofiles
    import gzip
    import os
        import gzip
            import os
from typing import Any, List, Dict, Optional
import logging
"""
JSON reporting utilities for cybersecurity testing results.

Provides tools for:
- Structured data export
- Machine-readable reports
- API integration
- Data analysis support
"""


@dataclass
class JSONReportConfig:
    """Configuration for JSON reporting."""
    output_directory: str = "reports"
    pretty_print: bool = True
    include_metadata: bool = True
    compress_output: bool = False
    schema_validation: bool = True
    max_file_size: int = 100 * 1024 * 1024  # 100MB

@dataclass
class JSONReportResult:
    """Result of a JSON reporting operation."""
    success: bool = False
    file_path: Optional[str] = None
    file_size: int = 0
    record_count: int = 0
    time_taken: float = 0.0
    error_message: Optional[str] = None

# CPU-bound operations (use 'def')
def generate_metadata() -> Dict[str, Any]:
    """Generate report metadata - CPU intensive."""
    return {
        "report_type": "cybersecurity_assessment",
        "generated_at": datetime.now().isoformat(),
        "version": "1.0.0",
        "format": "json",
        "schema_version": "1.0"
    }

def validate_data_structure(data: Dict[str, Any]) -> List[str]:
    """Validate data structure for JSON export - CPU intensive."""
    errors = []
    
    required_fields = ['scan_results', 'vulnerabilities', 'summary']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if 'scan_results' in data and not isinstance(data['scan_results'], list):
        errors.append("scan_results must be a list")
    
    if 'vulnerabilities' in data and not isinstance(data['vulnerabilities'], list):
        errors.append("vulnerabilities must be a list")
    
    if 'summary' in data and not isinstance(data['summary'], dict):
        errors.append("summary must be a dictionary")
    
    return errors

def serialize_datetime(obj: Any) -> str:
    """Serialize datetime objects for JSON - CPU intensive."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def calculate_data_statistics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate statistics for the data - CPU intensive."""
    stats = {
        "total_records": 0,
        "scan_results_count": 0,
        "vulnerabilities_count": 0,
        "summary_fields_count": 0
    }
    
    if 'scan_results' in data:
        stats['scan_results_count'] = len(data['scan_results'])
        stats['total_records'] += stats['scan_results_count']
    
    if 'vulnerabilities' in data:
        stats['vulnerabilities_count'] = len(data['vulnerabilities'])
        stats['total_records'] += stats['vulnerabilities_count']
    
    if 'summary' in data:
        stats['summary_fields_count'] = len(data['summary'])
    
    return stats

def format_json_data(data: Dict[str, Any], config: JSONReportConfig) -> str:
    """Format data as JSON string - CPU intensive."""
    # Add metadata if requested
    if config.include_metadata:
        data['metadata'] = generate_metadata()
    
    # Validate data structure if requested
    if config.schema_validation:
        errors = validate_data_structure(data)
        if errors:
            raise ValueError(f"Data validation failed: {', '.join(errors)}")
    
    # Serialize to JSON
    if config.pretty_print:
        return json.dumps(data, indent=2, default=serialize_datetime, ensure_ascii=False)
    else:
        return json.dumps(data, default=serialize_datetime, ensure_ascii=False, separators=(',', ':'))

def compress_json_data(json_string: str) -> bytes:
    """Compress JSON data - CPU intensive."""
    return gzip.compress(json_string.encode('utf-8'))

def estimate_file_size(data: Dict[str, Any]) -> int:
    """Estimate file size for data - CPU intensive."""
    # Rough estimation: convert to JSON and measure
    try:
        json_string = json.dumps(data, default=serialize_datetime)
        return len(json_string.encode('utf-8'))
    except:
        return 0

# Async operations (use 'async def')
async def write_json_file_async(file_path: str, content: str, compress: bool = False) -> None:
    """Write JSON content to file asynchronously - I/O bound."""
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if compress:
        compressed_content = compress_json_data(content)
        async with aiofiles.open(file_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            await f.write(compressed_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    else:
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            await f.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")

async def read_json_file_async(file_path: str, compressed: bool = False) -> Dict[str, Any]:
    """Read JSON file asynchronously - I/O bound."""
    if compressed:
        async with aiofiles.open(file_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            compressed_content = await f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = gzip.decompress(compressed_content).decode('utf-8')
    else:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = await f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    return json.loads(content)

class JSONReporter:
    """JSON-based reporting tool."""
    
    def __init__(self, config: JSONReportConfig):
        
    """__init__ function."""
self.config = config
        self.start_time = None
    
    async def export_data(self, data: Dict[str, Any], filename: Optional[str] = None) -> JSONReportResult:
        """Export data to JSON file."""
        start_time = time.time()
        
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cybersecurity_report_{timestamp}.json"
                if self.config.compress_output:
                    filename += ".gz"
            
            # Format JSON data
            json_content = format_json_data(data, self.config)
            
            # Check file size limit
            estimated_size = estimate_file_size(data)
            if estimated_size > self.config.max_file_size:
                return JSONReportResult(
                    success=False,
                    time_taken=time.time() - start_time,
                    error_message=f"Estimated file size ({estimated_size} bytes) exceeds limit ({self.config.max_file_size} bytes)"
                )
            
            # Write to file
            file_path = f"{self.config.output_directory}/{filename}"
            await write_json_file_async(file_path, json_content, self.config.compress_output)
            
            # Get actual file size
            file_size = os.path.getsize(file_path)
            
            # Calculate record count
            stats = calculate_data_statistics(data)
            record_count = stats['total_records']
            
            return JSONReportResult(
                success=True,
                file_path=file_path,
                file_size=file_size,
                record_count=record_count,
                time_taken=time.time() - start_time
            )
            
        except Exception as e:
            return JSONReportResult(
                success=False,
                time_taken=time.time() - start_time,
                error_message=str(e)
            )
    
    async def export_scan_results(self, scan_results: List[Dict[str, Any]]) -> JSONReportResult:
        """Export scan results to JSON."""
        data = {
            "scan_results": scan_results,
            "summary": {
                "total_scans": len(scan_results),
                "successful_scans": len([r for r in scan_results if r.get('success', False)]),
                "failed_scans": len([r for r in scan_results if not r.get('success', False)]),
                "average_response_time": sum(r.get('response_time', 0) for r in scan_results) / len(scan_results) if scan_results else 0
            },
            "vulnerabilities": []
        }
        
        return await self.export_data(data, "scan_results.json")
    
    async def export_vulnerabilities(self, vulnerabilities: List[Dict[str, Any]]) -> JSONReportResult:
        """Export vulnerabilities to JSON."""
        data = {
            "vulnerabilities": vulnerabilities,
            "summary": {
                "total_vulnerabilities": len(vulnerabilities),
                "critical_count": len([v for v in vulnerabilities if v.get('severity') == 'critical']),
                "high_count": len([v for v in vulnerabilities if v.get('severity') == 'high']),
                "medium_count": len([v for v in vulnerabilities if v.get('severity') == 'medium']),
                "low_count": len([v for v in vulnerabilities if v.get('severity') == 'low']),
                "info_count": len([v for v in vulnerabilities if v.get('severity') == 'info'])
            },
            "scan_results": []
        }
        
        return await self.export_data(data, "vulnerabilities.json")
    
    async def export_comprehensive_report(self, scan_results: List[Dict[str, Any]], 
                                        vulnerabilities: List[Dict[str, Any]]) -> JSONReportResult:
        """Export comprehensive report with all data."""
        successful_scans = [r for r in scan_results if r.get('success', False)]
        failed_scans = [r for r in scan_results if not r.get('success', False)]
        
        data = {
            "scan_results": scan_results,
            "vulnerabilities": vulnerabilities,
            "summary": {
                "total_scans": len(scan_results),
                "successful_scans": len(successful_scans),
                "failed_scans": len(failed_scans),
                "success_rate": (len(successful_scans) / len(scan_results)) * 100 if scan_results else 0,
                "average_response_time": sum(r.get('response_time', 0) for r in scan_results) / len(scan_results) if scan_results else 0,
                "total_vulnerabilities": len(vulnerabilities),
                "critical_vulnerabilities": len([v for v in vulnerabilities if v.get('severity') == 'critical']),
                "high_vulnerabilities": len([v for v in vulnerabilities if v.get('severity') == 'high']),
                "medium_vulnerabilities": len([v for v in vulnerabilities if v.get('severity') == 'medium']),
                "low_vulnerabilities": len([v for v in vulnerabilities if v.get('severity') == 'low']),
                "info_vulnerabilities": len([v for v in vulnerabilities if v.get('severity') == 'info'])
            }
        }
        
        return await self.export_data(data, "comprehensive_report.json")
    
    async def merge_reports(self, report_files: List[str]) -> JSONReportResult:
        """Merge multiple JSON reports into one."""
        start_time = time.time()
        
        try:
            merged_data = {
                "scan_results": [],
                "vulnerabilities": [],
                "summary": {}
            }
            
            # Read and merge all reports
            for file_path in report_files:
                try:
                    file_data = await read_json_file_async(file_path)
                    
                    if 'scan_results' in file_data:
                        merged_data['scan_results'].extend(file_data['scan_results'])
                    
                    if 'vulnerabilities' in file_data:
                        merged_data['vulnerabilities'].extend(file_data['vulnerabilities'])
                    
                except Exception as e:
                    print(f"Warning: Could not read file {file_path}: {e}")
            
            # Recalculate summary
            successful_scans = [r for r in merged_data['scan_results'] if r.get('success', False)]
            merged_data['summary'] = {
                "total_scans": len(merged_data['scan_results']),
                "successful_scans": len(successful_scans),
                "failed_scans": len(merged_data['scan_results']) - len(successful_scans),
                "success_rate": (len(successful_scans) / len(merged_data['scan_results'])) * 100 if merged_data['scan_results'] else 0,
                "total_vulnerabilities": len(merged_data['vulnerabilities']),
                "critical_vulnerabilities": len([v for v in merged_data['vulnerabilities'] if v.get('severity') == 'critical']),
                "high_vulnerabilities": len([v for v in merged_data['vulnerabilities'] if v.get('severity') == 'high']),
                "medium_vulnerabilities": len([v for v in merged_data['vulnerabilities'] if v.get('severity') == 'medium']),
                "low_vulnerabilities": len([v for v in merged_data['vulnerabilities'] if v.get('severity') == 'low']),
                "info_vulnerabilities": len([v for v in merged_data['vulnerabilities'] if v.get('severity') == 'info'])
            }
            
            # Export merged data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"merged_report_{timestamp}.json"
            
            return await self.export_data(merged_data, filename)
            
        except Exception as e:
            return JSONReportResult(
                success=False,
                time_taken=time.time() - start_time,
                error_message=str(e)
            )
    
    async def validate_json_schema(self, data: Dict[str, Any]) -> List[str]:
        """Validate JSON schema - I/O bound for large datasets."""
        errors = []
        
        # Basic structure validation
        if not isinstance(data, dict):
            errors.append("Root data must be a dictionary")
            return errors
        
        # Required fields validation
        required_fields = ['scan_results', 'vulnerabilities', 'summary']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Type validation
        if 'scan_results' in data and not isinstance(data['scan_results'], list):
            errors.append("scan_results must be a list")
        
        if 'vulnerabilities' in data and not isinstance(data['vulnerabilities'], list):
            errors.append("vulnerabilities must be a list")
        
        if 'summary' in data and not isinstance(data['summary'], dict):
            errors.append("summary must be a dictionary")
        
        return errors 
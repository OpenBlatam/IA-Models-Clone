from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, field_validator
from datetime import datetime
import structlog
import json
import os
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
JSON reporting module for cybersecurity assessment results.
"""

logger = structlog.get_logger(__name__)

class JSONReportInput(BaseModel):
    """Input model for JSON report generation."""
    scan_results: Dict[str, Any]
    target_info: Dict[str, str]
    scan_metadata: Dict[str, Any]
    output_file: Optional[str] = None
    pretty_print: bool = True
    include_timestamps: bool = True
    
    @field_validator('scan_results')
    def validate_scan_results(cls, v) -> bool:
        if not v:
            raise ValueError("Scan results cannot be empty")
        return v
    
    @field_validator('target_info')
    def validate_target_info(cls, v) -> Optional[Dict[str, Any]]:
        if not v:
            raise ValueError("Target info cannot be empty")
        return v

class JSONReportResult(BaseModel):
    """Result model for JSON report generation."""
    json_content: str
    file_path: Optional[str] = None
    summary_stats: Dict[str, int]
    generation_time: float
    is_successful: bool
    error_message: Optional[str] = None

def generate_json_report(input_data: JSONReportInput) -> JSONReportResult:
    """
    RORO: Receive JSONReportInput, return JSONReportResult
    
    Generate structured JSON report for cybersecurity assessment results.
    """
    start_time = datetime.now()
    
    try:
        # Create JSON report structure
        report_data = create_json_report_structure(input_data)
        
        # Generate JSON content
        if input_data.pretty_print:
            json_content = json.dumps(report_data, indent=2, default=str)
        else:
            json_content = json.dumps(report_data, default=str)
        
        # Save to file if specified
        file_path = None
        if input_data.output_file:
            file_path = save_json_to_file(json_content, input_data.output_file)
        
        # Calculate summary statistics
        summary_stats = calculate_summary_stats(input_data.scan_results)
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("JSON report generated successfully", 
                   target=input_data.target_info.get('host', 'unknown'),
                   file_path=file_path,
                   generation_time=generation_time)
        
        return JSONReportResult(
            json_content=json_content,
            file_path=file_path,
            summary_stats=summary_stats,
            generation_time=generation_time,
            is_successful=True
        )
        
    except Exception as e:
        generation_time = (datetime.now() - start_time).total_seconds()
        logger.error("JSON report generation failed", error=str(e))
        
        return JSONReportResult(
            json_content="",
            file_path=None,
            summary_stats={},
            generation_time=generation_time,
            is_successful=False,
            error_message=str(e)
        )

def create_json_report_structure(input_data: JSONReportInput) -> Dict[str, Any]:
    """Create structured JSON report data."""
    report = {
        "report_metadata": {
            "report_type": "cybersecurity_assessment",
            "version": "1.0",
            "generator": "Key Messages Security Scanner",
            "generation_timestamp": datetime.now().isoformat() if input_data.include_timestamps else None
        },
        "target_information": input_data.target_info,
        "scan_metadata": input_data.scan_metadata,
        "summary": create_summary_data(input_data.scan_results),
        "detailed_results": create_detailed_results_data(input_data.scan_results),
        "statistics": create_statistics_data(input_data.scan_results)
    }
    
    return report

def create_summary_data(scan_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary data for JSON report."""
    summary = {
        "total_categories": len(scan_results),
        "total_findings": 0,
        "categories": {}
    }
    
    for category, data in scan_results.items():
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
            count = len(results)
            summary['total_findings'] += count
            summary['categories'][category] = {
                "findings_count": count,
                "status": "clean" if count == 0 else "issues_found"
            }
        elif isinstance(data, list):
            count = len(data)
            summary['total_findings'] += count
            summary['categories'][category] = {
                "findings_count": count,
                "status": "clean" if count == 0 else "issues_found"
            }
    
    return summary

def create_detailed_results_data(scan_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create detailed results data for JSON report."""
    detailed_results = {}
    
    for category, data in scan_results.items():
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
            detailed_results[category] = {
                "type": "structured",
                "results": results,
                "metadata": {k: v for k, v in data.items() if k != 'results'}
            }
        elif isinstance(data, list):
            detailed_results[category] = {
                "type": "list",
                "results": data
            }
        else:
            detailed_results[category] = {
                "type": "raw",
                "data": data
            }
    
    return detailed_results

def create_statistics_data(scan_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create statistics data for JSON report."""
    stats = {
        "severity_distribution": {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0
        },
        "category_distribution": {},
        "timeline_data": {
            "scan_start": None,
            "scan_end": None,
            "duration_seconds": 0
        }
    }
    
    for category, data in scan_results.items():
        category_count = 0
        
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
            category_count = len(results)
            
            for result in results:
                if isinstance(result, dict):
                    severity = result.get('severity', '').lower()
                    if severity in stats['severity_distribution']:
                        stats['severity_distribution'][severity] += 1
        
        elif isinstance(data, list):
            category_count = len(data)
        
        stats['category_distribution'][category] = category_count
    
    return stats

def save_json_to_file(json_content: str, file_path: str) -> str:
    """Save JSON content to file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(json_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        return file_path
        
    except Exception as e:
        logger.error("Failed to save JSON file", file_path=file_path, error=str(e))
        raise

def calculate_summary_stats(scan_results: Dict[str, Any]) -> Dict[str, int]:
    """Calculate summary statistics from scan results."""
    stats = {
        'total_categories': len(scan_results),
        'total_findings': 0,
        'critical_findings': 0,
        'high_findings': 0,
        'medium_findings': 0,
        'low_findings': 0
    }
    
    for category, data in scan_results.items():
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
            stats['total_findings'] += len(results)
            
            for result in results:
                if isinstance(result, dict):
                    severity = result.get('severity', '').lower()
                    if severity == 'critical':
                        stats['critical_findings'] += 1
                    elif severity == 'high':
                        stats['high_findings'] += 1
                    elif severity == 'medium':
                        stats['medium_findings'] += 1
                    elif severity == 'low':
                        stats['low_findings'] += 1
        
        elif isinstance(data, list):
            stats['total_findings'] += len(data)
    
    return stats 
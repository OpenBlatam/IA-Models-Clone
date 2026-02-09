from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, field_validator
from datetime import datetime
import structlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Console reporting module for cybersecurity assessment results.
"""

logger = structlog.get_logger(__name__)

class ConsoleReportInput(BaseModel):
    """Input model for console report generation."""
    scan_results: Dict[str, Any]
    target_info: Dict[str, str]
    scan_metadata: Dict[str, Any]
    include_details: bool = True
    color_output: bool = True
    
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

class ConsoleReportResult(BaseModel):
    """Result model for console report generation."""
    report_content: str
    summary_stats: Dict[str, int]
    generation_time: float
    is_successful: bool
    error_message: Optional[str] = None

def generate_console_report(input_data: ConsoleReportInput) -> ConsoleReportResult:
    """
    RORO: Receive ConsoleReportInput, return ConsoleReportResult
    
    Generate formatted console report for cybersecurity assessment results.
    """
    start_time = datetime.now()
    
    try:
        console = Console(color_system="auto" if input_data.color_output else None)
        
        # Create report sections
        report_sections = []
        
        # Header section
        header = create_header_section(input_data.target_info, console)
        report_sections.append(header)
        
        # Summary section
        summary = create_summary_section(input_data.scan_results, console)
        report_sections.append(summary)
        
        # Detailed results section
        if input_data.include_details:
            details = create_details_section(input_data.scan_results, console)
            report_sections.append(details)
        
        # Footer section
        footer = create_footer_section(input_data.scan_metadata, console)
        report_sections.append(footer)
        
        # Combine all sections
        report_content = "\n\n".join(report_sections)
        
        # Calculate summary statistics
        summary_stats = calculate_summary_stats(input_data.scan_results)
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("Console report generated successfully", 
                   target=input_data.target_info.get('host', 'unknown'),
                   generation_time=generation_time)
        
        return ConsoleReportResult(
            report_content=report_content,
            summary_stats=summary_stats,
            generation_time=generation_time,
            is_successful=True
        )
        
    except Exception as e:
        generation_time = (datetime.now() - start_time).total_seconds()
        logger.error("Console report generation failed", error=str(e))
        
        return ConsoleReportResult(
            report_content="",
            summary_stats={},
            generation_time=generation_time,
            is_successful=False,
            error_message=str(e)
        )

def create_header_section(target_info: Dict[str, str], console: Console) -> str:
    """Create header section with target information."""
    header_text = Text()
    header_text.append("🔒 CYBERSECURITY ASSESSMENT REPORT\n", style="bold blue")
    header_text.append(f"Target: {target_info.get('host', 'Unknown')}\n", style="bold")
    header_text.append(f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", style="dim")
    
    if 'ip' in target_info:
        header_text.append(f"IP Address: {target_info['ip']}\n", style="dim")
    
    if 'ports' in target_info:
        header_text.append(f"Ports Scanned: {target_info['ports']}\n", style="dim")
    
    panel = Panel(header_text, title="Assessment Header", border_style="blue")
    return console.render_str(panel)

def create_summary_section(scan_results: Dict[str, Any], console: Console) -> str:
    """Create summary section with key findings."""
    summary_table = Table(title="📊 Scan Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Category", style="cyan", no_wrap=True)
    summary_table.add_column("Count", justify="right", style="green")
    summary_table.add_column("Status", style="yellow")
    
    # Add summary rows
    for category, data in scan_results.items():
        if isinstance(data, dict):
            count = len(data.get('results', []))
            status = "✅ Clean" if count == 0 else "⚠️ Issues Found"
            summary_table.add_row(category.title(), str(count), status)
        elif isinstance(data, list):
            count = len(data)
            status = "✅ Clean" if count == 0 else "⚠️ Issues Found"
            summary_table.add_row(category.title(), str(count), status)
    
    return console.render_str(summary_table)

def create_details_section(scan_results: Dict[str, Any], console: Console) -> str:
    """Create detailed results section."""
    details_sections = []
    
    for category, data in scan_results.items():
        if not data or (isinstance(data, list) and len(data) == 0):
            continue
            
        category_table = Table(title=f"🔍 {category.title()} Details", show_header=True, header_style="bold")
        
        # Determine columns based on data structure
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
            if results and isinstance(results[0], dict):
                # Add columns based on first result
                for key in results[0].keys():
                    category_table.add_column(key.title(), style="cyan")
                
                # Add rows
                for result in results:
                    row_data = [str(result.get(key, '')) for key in results[0].keys()]
                    category_table.add_row(*row_data)
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # Add columns based on first item
            for key in data[0].keys():
                category_table.add_column(key.title(), style="cyan")
            
            # Add rows
            for item in data:
                row_data = [str(item.get(key, '')) for key in data[0].keys()]
                category_table.add_row(*row_data)
        
        details_sections.append(console.render_str(category_table))
    
    return "\n\n".join(details_sections)

def create_footer_section(scan_metadata: Dict[str, Any], console: Console) -> str:
    """Create footer section with scan metadata."""
    footer_text = Text()
    footer_text.append("📋 Scan Metadata\n", style="bold blue")
    
    for key, value in scan_metadata.items():
        footer_text.append(f"{key.title()}: {value}\n", style="dim")
    
    footer_text.append("\n🔐 Report generated by Key Messages Security Scanner", style="italic")
    
    panel = Panel(footer_text, title="Footer", border_style="green")
    return console.render_str(panel)

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
            
            # Count by severity if available
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
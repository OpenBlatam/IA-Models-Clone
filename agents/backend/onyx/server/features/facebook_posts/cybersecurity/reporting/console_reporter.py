from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
from typing import Any, List, Dict, Optional
import logging
"""
Console reporting utilities for cybersecurity testing results.

Provides tools for:
- Terminal-based result display
- Colored output formatting
- Progress indicators
- Real-time status updates
"""


@dataclass
class ConsoleReportConfig:
    """Configuration for console reporting."""
    enable_colors: bool = True
    show_progress: bool = True
    show_timestamps: bool = True
    max_line_length: int = 80
    indent_size: int = 2
    output_stream: Any = sys.stdout

@dataclass
class ConsoleReportResult:
    """Result of a console reporting operation."""
    success: bool = False
    lines_written: int = 0
    time_taken: float = 0.0
    error_message: Optional[str] = None

# CPU-bound operations (use 'def')
def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for console output - CPU intensive."""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to fit console width - CPU intensive."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def calculate_progress_percentage(current: int, total: int) -> float:
    """Calculate progress percentage - CPU intensive."""
    if total == 0:
        return 0.0
    return (current / total) * 100

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format - CPU intensive."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"

def generate_progress_bar(percentage: float, width: int = 50) -> str:
    """Generate ASCII progress bar - CPU intensive."""
    filled = int(width * percentage / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percentage:.1f}%"

def format_severity(severity: str) -> str:
    """Format severity with colors - CPU intensive."""
    severity_colors = {
        "critical": "\033[91m",  # Red
        "high": "\033[31m",      # Dark red
        "medium": "\033[33m",    # Yellow
        "low": "\033[32m",       # Green
        "info": "\033[36m"       # Cyan
    }
    reset_color = "\033[0m"
    
    color = severity_colors.get(severity.lower(), "")
    return f"{color}{severity.upper()}{reset_color}"

def format_status(status: str) -> str:
    """Format status with colors - CPU intensive."""
    status_colors = {
        "success": "\033[92m",   # Green
        "warning": "\033[93m",   # Yellow
        "error": "\033[91m",     # Red
        "info": "\033[94m"       # Blue
    }
    reset_color = "\033[0m"
    
    color = status_colors.get(status.lower(), "")
    return f"{color}{status.upper()}{reset_color}"

# Async operations (use 'async def')
async def write_to_console_async(text: str, config: ConsoleReportConfig) -> None:
    """Write text to console asynchronously - I/O bound."""
    if config.output_stream:
        config.output_stream.write(text)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        config.output_stream.flush()
    await asyncio.sleep(0)  # Yield control

async def clear_console_async(config: ConsoleReportConfig) -> None:
    """Clear console screen asynchronously - I/O bound."""
    if config.output_stream:
        config.output_stream.write("\033[2J\033[H")  # Clear screen
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        config.output_stream.flush()
    await asyncio.sleep(0)

class ConsoleReporter:
    """Console-based reporting tool."""
    
    def __init__(self, config: ConsoleReportConfig):
        
    """__init__ function."""
self.config = config
        self.start_time = None
    
    def start_reporting(self) -> None:
        """Start reporting session."""
        self.start_time = time.time()
        if self.config.show_timestamps:
            timestamp = format_timestamp(datetime.now())
            print(f"[{timestamp}] Starting cybersecurity report...")
    
    def end_reporting(self) -> ConsoleReportResult:
        """End reporting session."""
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        if self.config.show_timestamps:
            timestamp = format_timestamp(datetime.now())
            print(f"[{timestamp}] Report completed in {format_duration(duration)}")
        
        return ConsoleReportResult(
            success=True,
            time_taken=duration
        )
    
    async def report_scan_results(self, results: List[Dict[str, Any]]) -> ConsoleReportResult:
        """Report scan results to console."""
        start_time = time.time()
        lines_written = 0
        
        try:
            if not results:
                await write_to_console_async("No scan results to report.\n", self.config)
                return ConsoleReportResult(success=True, lines_written=1)
            
            # Summary header
            await write_to_console_async("\n" + "="*60 + "\n", self.config)
            await write_to_console_async("CYBERSECURITY SCAN RESULTS\n", self.config)
            await write_to_console_async("="*60 + "\n\n", self.config)
            lines_written += 4
            
            # Statistics
            total_scans = len(results)
            successful_scans = sum(1 for r in results if r.get('success', False))
            failed_scans = total_scans - successful_scans
            
            await write_to_console_async(f"Total Scans: {total_scans}\n", self.config)
            await write_to_console_async(f"Successful: {successful_scans}\n", self.config)
            await write_to_console_async(f"Failed: {failed_scans}\n", self.config)
            await write_to_console_async(f"Success Rate: {(successful_scans/total_scans)*100:.1f}%\n\n", self.config)
            lines_written += 4
            
            # Detailed results
            for i, result in enumerate(results, 1):
                await self._report_single_result(result, i, total_scans)
                lines_written += 1
            
            return ConsoleReportResult(
                success=True,
                lines_written=lines_written,
                time_taken=time.time() - start_time
            )
            
        except Exception as e:
            return ConsoleReportResult(
                success=False,
                lines_written=lines_written,
                time_taken=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _report_single_result(self, result: Dict[str, Any], index: int, total: int) -> None:
        """Report a single scan result."""
        target = result.get('target', 'Unknown')
        success = result.get('success', False)
        status = format_status('success' if success else 'error')
        
        # Progress indicator
        if self.config.show_progress:
            percentage = calculate_progress_percentage(index, total)
            progress_bar = generate_progress_bar(percentage, 30)
            await write_to_console_async(f"{progress_bar} ", self.config)
        
        # Result line
        await write_to_console_async(f"[{index:3d}/{total:3d}] {target}: {status}\n", self.config)
        
        # Additional details
        if 'response_time' in result:
            duration = format_duration(result['response_time'])
            await write_to_console_async(f"{' ' * self.config.indent_size}Duration: {duration}\n", self.config)
        
        if 'error_message' in result and result['error_message']:
            error_msg = truncate_text(result['error_message'], self.config.max_line_length)
            await write_to_console_async(f"{' ' * self.config.indent_size}Error: {error_msg}\n", self.config)
    
    async def report_vulnerabilities(self, vulnerabilities: List[Dict[str, Any]]) -> ConsoleReportResult:
        """Report vulnerability findings."""
        start_time = time.time()
        lines_written = 0
        
        try:
            if not vulnerabilities:
                await write_to_console_async("No vulnerabilities found.\n", self.config)
                return ConsoleReportResult(success=True, lines_written=1)
            
            # Vulnerability header
            await write_to_console_async("\n" + "="*60 + "\n", self.config)
            await write_to_console_async("VULNERABILITY FINDINGS\n", self.config)
            await write_to_console_async("="*60 + "\n\n", self.config)
            lines_written += 4
            
            # Group by severity
            severity_groups = {}
            for vuln in vulnerabilities:
                severity = vuln.get('severity', 'info').lower()
                if severity not in severity_groups:
                    severity_groups[severity] = []
                severity_groups[severity].append(vuln)
            
            # Report by severity (critical first)
            severity_order = ['critical', 'high', 'medium', 'low', 'info']
            for severity in severity_order:
                if severity in severity_groups:
                    await write_to_console_async(f"\n{format_severity(severity)} VULNERABILITIES ({len(severity_groups[severity])})\n", self.config)
                    await write_to_console_async("-" * 40 + "\n", self.config)
                    lines_written += 2
                    
                    for vuln in severity_groups[severity]:
                        await self._report_single_vulnerability(vuln)
                        lines_written += 1
            
            return ConsoleReportResult(
                success=True,
                lines_written=lines_written,
                time_taken=time.time() - start_time
            )
            
        except Exception as e:
            return ConsoleReportResult(
                success=False,
                lines_written=lines_written,
                time_taken=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _report_single_vulnerability(self, vuln: Dict[str, Any]) -> None:
        """Report a single vulnerability."""
        title = vuln.get('title', 'Unknown Vulnerability')
        description = vuln.get('description', 'No description available')
        target = vuln.get('target', 'Unknown Target')
        
        # Truncate long descriptions
        description = truncate_text(description, self.config.max_line_length - 10)
        
        await write_to_console_async(f"• {title}\n", self.config)
        await write_to_console_async(f"{' ' * self.config.indent_size}Target: {target}\n", self.config)
        await write_to_console_async(f"{' ' * self.config.indent_size}Description: {description}\n", self.config)
        
        if 'cvss_score' in vuln:
            await write_to_console_async(f"{' ' * self.config.indent_size}CVSS Score: {vuln['cvss_score']}\n", self.config)
        
        if 'remediation' in vuln:
            remediation = truncate_text(vuln['remediation'], self.config.max_line_length - 10)
            await write_to_console_async(f"{' ' * self.config.indent_size}Remediation: {remediation}\n", self.config)
    
    async def report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress during operations."""
        if not self.config.show_progress:
            return
        
        percentage = calculate_progress_percentage(current, total)
        progress_bar = generate_progress_bar(percentage, 30)
        
        progress_text = f"\r{progress_bar} {current}/{total}"
        if message:
            progress_text += f" - {message}"
        
        await write_to_console_async(progress_text, self.config)
        
        if current == total:
            await write_to_console_async("\n", self.config)
    
    async def report_summary(self, summary: Dict[str, Any]) -> ConsoleReportResult:
        """Report summary statistics."""
        start_time = time.time()
        lines_written = 0
        
        try:
            await write_to_console_async("\n" + "="*60 + "\n", self.config)
            await write_to_console_async("SCAN SUMMARY\n", self.config)
            await write_to_console_async("="*60 + "\n\n", self.config)
            lines_written += 4
            
            for key, value in summary.items():
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                elif isinstance(value, dict):
                    formatted_value = f"{len(value)} items"
                else:
                    formatted_value = str(value)
                
                await write_to_console_async(f"{key.replace('_', ' ').title()}: {formatted_value}\n", self.config)
                lines_written += 1
            
            return ConsoleReportResult(
                success=True,
                lines_written=lines_written,
                time_taken=time.time() - start_time
            )
            
        except Exception as e:
            return ConsoleReportResult(
                success=False,
                lines_written=lines_written,
                time_taken=time.time() - start_time,
                error_message=str(e)
            ) 
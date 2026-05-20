"""
Formatting Utilities
====================

Utilities for formatting data, responses, and messages.
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal


def format_response(
    data: Any,
    success: bool = True,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format a standardized API response.
    
    Args:
        data: Response data
        success: Whether operation was successful
        message: Optional message
        metadata: Optional metadata
        
    Returns:
        Formatted response dictionary
    """
    response = {
        "success": success,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    
    if message:
        response["message"] = message
    
    if metadata:
        response["metadata"] = metadata
    
    return response


def format_error(
    error: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format an error response.
    
    Args:
        error: Error message
        error_code: Optional error code
        details: Optional error details
        
    Returns:
        Formatted error response dictionary
    """
    response = {
        "success": False,
        "error": error,
        "timestamp": datetime.now().isoformat()
    }
    
    if error_code:
        response["error_code"] = error_code
    
    if details:
        response["details"] = details
    
    return response


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {secs:.2f}s"
    
    hours = int(minutes // 60)
    mins = minutes % 60
    
    if hours < 24:
        return f"{hours}h {mins}m {secs:.2f}s"
    
    days = int(hours // 24)
    hrs = hours % 24
    
    return f"{days}d {hrs}h {mins}m {secs:.2f}s"


def format_file_size(bytes_size: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def format_percentage(value: float, total: float, decimals: int = 2) -> str:
    """
    Format percentage value.
    
    Args:
        value: Current value
        total: Total value
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if total == 0:
        return "0.00%"
    
    percentage = (value / total) * 100
    return f"{percentage:.{decimals}f}%"


def format_json_safe(obj: Any) -> Any:
    """
    Format object to be JSON-safe (handles Decimal, datetime, etc.).
    
    Args:
        obj: Object to format
        
    Returns:
        JSON-safe object
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: format_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_json_safe(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(format_json_safe(item) for item in obj)
    elif isinstance(obj, set):
        return list(format_json_safe(item) for item in obj)
    else:
        return obj


def format_prompt_summary(prompt: str, max_length: int = 100) -> str:
    """
    Format prompt summary for display.
    
    Args:
        prompt: Prompt text
        max_length: Maximum length
        
    Returns:
        Formatted prompt summary
    """
    if len(prompt) <= max_length:
        return prompt
    
    return prompt[:max_length - 3] + "..."


def format_url_safe(url: str) -> str:
    """
    Format URL for safe display (truncate if too long).
    
    Args:
        url: URL string
        
    Returns:
        Formatted URL
    """
    if len(url) <= 80:
        return url
    
    return url[:40] + "..." + url[-37:]


def format_batch_summary(
    total: int,
    completed: int,
    failed: int
) -> str:
    """
    Format batch processing summary.
    
    Args:
        total: Total items
        completed: Completed items
        failed: Failed items
        
    Returns:
        Formatted summary string
    """
    return f"{completed}/{total} completed, {failed} failed ({format_percentage(completed, total)} success rate)"


def format_workflow_status(status: str) -> str:
    """
    Format workflow status for display.
    
    Args:
        status: Status string
        
    Returns:
        Formatted status string
    """
    status_map = {
        "queued": "⏳ Queued",
        "processing": "🔄 Processing",
        "completed": "✅ Completed",
        "failed": "❌ Failed",
        "cancelled": "🚫 Cancelled"
    }
    
    return status_map.get(status.lower(), status.title())


def format_metrics_summary(metrics: Dict[str, Any]) -> str:
    """
    Format metrics summary for display.
    
    Args:
        metrics: Metrics dictionary
        
    Returns:
        Formatted summary string
    """
    total = metrics.get("total_operations", 0)
    successful = metrics.get("successful_operations", 0)
    success_rate = metrics.get("success_rate", 0.0)
    avg_duration = metrics.get("average_duration", 0.0)
    
    return (
        f"Total: {total} operations, "
        f"Success: {successful} ({success_rate:.1f}%), "
        f"Avg duration: {format_duration(avg_duration)}"
    )


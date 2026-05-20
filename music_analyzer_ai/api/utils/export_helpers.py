"""
Export helper functions
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_export_method(export_service: Any, format: str):
    """
    Get the appropriate export method based on format
    
    Args:
        export_service: Export service instance
        format: Export format (json, text, markdown)
    
    Returns:
        Export method function
    
    Raises:
        ValueError: If format is not supported
    """
    format_map = {
        "json": export_service.export_json,
        "text": export_service.export_text,
        "markdown": export_service.export_markdown
    }
    
    if format not in format_map:
        raise ValueError(f"Formato no soportado: {format}")
    
    return format_map[format]


def export_analysis(
    export_service: Any,
    analysis: Dict[str, Any],
    format: str
) -> str:
    """
    Export analysis in the specified format
    
    Args:
        export_service: Export service instance
        analysis: Analysis data to export
        format: Export format (json, text, markdown)
    
    Returns:
        Exported content as string
    
    Raises:
        ValueError: If format is not supported
    """
    export_method = get_export_method(export_service, format)
    return export_method(analysis)


"""
Export Module - Enhanced export utilities for results and data.

Provides:
- Multiple export formats (JSON, CSV, YAML, Markdown, HTML)
- Batch export with progress tracking
- Custom formatters and transformers
- Data validation and transformation
- Error handling and recovery
"""

import logging
import json
import csv
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyYAML not available. YAML export will be disabled.")

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Export formats."""
    JSON = "json"
    CSV = "csv"
    YAML = "yaml"
    MARKDOWN = "markdown"
    HTML = "html"


class ExportError(Exception):
    """Export-related errors."""
    pass


class Exporter:
    """
    Enhanced export utility for benchmark data.
    
    Features:
    - Multiple export formats
    - Data transformation
    - Progress tracking
    - Error handling
    - Validation
    """
    
    def __init__(
        self,
        default_format: ExportFormat = ExportFormat.JSON,
        validate_data: bool = True,
    ):
        """
        Initialize exporter.
        
        Args:
            default_format: Default export format
            validate_data: Whether to validate data before export
        """
        self.default_format = default_format
        self.validate_data = validate_data
    
    def export_results(
        self,
        results: List[Any],
        output_path: Path,
        format: Optional[ExportFormat] = None,
        transformer: Optional[Callable[[Any], Dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """
        Export results to file with enhanced features.
        
        Args:
            results: List of results to export
            output_path: Output file path
            format: Export format (defaults to instance default)
            transformer: Optional function to transform each result
            progress_callback: Optional callback(current, total) for progress
        
        Returns:
            Path to exported file
        
        Raises:
            ExportError: If export fails
        """
        if not results:
            logger.warning("No results to export")
            return output_path
        
        format = format or self.default_format
        
        # Validate data
        if self.validate_data:
            self._validate_results(results)
        
        # Transform data if transformer provided
        if transformer:
            results = [transformer(r) for r in results]
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        try:
            if format == ExportFormat.JSON:
                return self._export_json(results, output_path, progress_callback)
            elif format == ExportFormat.CSV:
                return self._export_csv(results, output_path, progress_callback)
            elif format == ExportFormat.YAML:
                if not YAML_AVAILABLE:
                    raise ExportError("YAML export requires PyYAML. Install with: pip install pyyaml")
                return self._export_yaml(results, output_path, progress_callback)
            elif format == ExportFormat.MARKDOWN:
                return self._export_markdown(results, output_path, progress_callback)
            elif format == ExportFormat.HTML:
                return self._export_html(results, output_path, progress_callback)
            else:
                raise ExportError(f"Unsupported format: {format}")
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            raise ExportError(f"Failed to export results: {e}") from e
    
    def _validate_results(self, results: List[Any]) -> None:
        """
        Validate results before export.
        
        Args:
            results: List of results to validate
        
        Raises:
            ExportError: If validation fails
        """
        if not isinstance(results, list):
            raise ExportError("Results must be a list")
        
        if not results:
            raise ExportError("Results list is empty")
        
        # Check if all results have to_dict method or are dicts
        for i, result in enumerate(results):
            if not isinstance(result, dict) and not hasattr(result, 'to_dict'):
                raise ExportError(
                    f"Result at index {i} must be a dict or have a to_dict method"
                )
    
    def _export_json(
        self,
        results: List[Any],
        output_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """Export to JSON with progress tracking."""
        data = []
        total = len(results)
        
        for i, result in enumerate(results):
            if hasattr(result, 'to_dict'):
                data.append(result.to_dict())
            else:
                data.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"Exported {len(data)} results to JSON: {output_path}")
        return output_path
    
    def _export_csv(
        self,
        results: List[Any],
        output_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """Export to CSV with progress tracking."""
        if not results:
            return output_path
        
        # Convert first result to dict to get fieldnames
        first = results[0]
        if hasattr(first, 'to_dict'):
            first = first.to_dict()
        elif not isinstance(first, dict):
            raise ExportError("Results must be dicts or have to_dict method")
        
        # Get all unique fieldnames from all results
        fieldnames = set(first.keys())
        for result in results[1:]:
            if hasattr(result, 'to_dict'):
                result = result.to_dict()
            if isinstance(result, dict):
                fieldnames.update(result.keys())
        
        fieldnames = sorted(fieldnames)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, result in enumerate(results):
                if hasattr(result, 'to_dict'):
                    result = result.to_dict()
                
                # Ensure all fields are present
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)
                
                if progress_callback:
                    progress_callback(i + 1, len(results))
        
        logger.info(f"Exported {len(results)} results to CSV: {output_path}")
        return output_path
    
    def _export_yaml(
        self,
        results: List[Any],
        output_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """Export to YAML with progress tracking."""
        if not YAML_AVAILABLE:
            raise ExportError("YAML export requires PyYAML")
        
        data = []
        total = len(results)
        
        for i, result in enumerate(results):
            if hasattr(result, 'to_dict'):
                data.append(result.to_dict())
            else:
                data.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Exported {len(data)} results to YAML: {output_path}")
        return output_path
    
    def _export_markdown(
        self,
        results: List[Any],
        output_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """Export to Markdown with progress tracking."""
        if not results:
            return output_path
        
        md_lines = [
            "# Benchmark Results",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
        ]
        
        # Convert first result to get keys
        first = results[0]
        if hasattr(first, 'to_dict'):
            first = first.to_dict()
        
        keys = list(first.keys())
        
        # Table header
        md_lines.append("| " + " | ".join(keys) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
        
        # Table rows
        for i, result in enumerate(results):
            if hasattr(result, 'to_dict'):
                result = result.to_dict()
            
            values = [str(result.get(k, "")).replace("|", "\\|") for k in keys]
            md_lines.append("| " + " | ".join(values) + " |")
            
            if progress_callback:
                progress_callback(i + 1, len(results))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))
        
        logger.info(f"Exported {len(results)} results to Markdown: {output_path}")
        return output_path
    
    def _export_html(
        self,
        results: List[Any],
        output_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """Export to HTML with progress tracking."""
        if not results:
            return output_path
        
        # Convert first result to get keys
        first = results[0]
        if hasattr(first, 'to_dict'):
            first = first.to_dict()
        
        keys = list(first.keys())
        
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>Benchmark Results</title>",
            "    <meta charset='utf-8'>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        table { border-collapse: collapse; width: 100%; margin-top: 20px; }",
            "        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }",
            "        th { background-color: #4CAF50; color: white; font-weight: bold; }",
            "        tr:nth-child(even) { background-color: #f2f2f2; }",
            "        tr:hover { background-color: #f5f5f5; }",
            "        .header { margin-bottom: 20px; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <div class='header'>",
            f"        <h1>Benchmark Results</h1>",
            f"        <p>Generated: {datetime.now().isoformat()}</p>",
            f"        <p>Total Results: {len(results)}</p>",
            "    </div>",
            "    <table>",
            "        <tr>",
        ]
        
        # Table header
        for key in keys:
            html_parts.append(f"            <th>{key}</th>")
        html_parts.append("        </tr>")
        
        # Table rows
        for i, result in enumerate(results):
            if hasattr(result, 'to_dict'):
                result = result.to_dict()
            
            html_parts.append("        <tr>")
            for key in keys:
                value = str(result.get(key, "")).replace("<", "&lt;").replace(">", "&gt;")
                html_parts.append(f"            <td>{value}</td>")
            html_parts.append("        </tr>")
            
            if progress_callback:
                progress_callback(i + 1, len(results))
        
        html_parts.extend([
            "    </table>",
            "</body>",
            "</html>",
        ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_parts))
        
        logger.info(f"Exported {len(results)} results to HTML: {output_path}")
        return output_path
    
    def export_batch(
        self,
        results: List[Any],
        output_dir: Path,
        formats: List[ExportFormat],
        transformer: Optional[Callable[[Any], Dict[str, Any]]] = None,
    ) -> Dict[ExportFormat, Path]:
        """
        Export results in multiple formats.
        
        Args:
            results: List of results to export
            output_dir: Output directory
            formats: List of formats to export
            transformer: Optional transformer function
        
        Returns:
            Dictionary mapping format to output path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exported = {}
        for format in formats:
            output_path = output_dir / f"results_{timestamp}.{format.value}"
            exported[format] = self.export_results(
                results,
                output_path,
                format=format,
                transformer=transformer,
            )
        
        return exported


__all__ = [
    "ExportFormat",
    "ExportError",
    "Exporter",
]

"""
Export utilities for Imagen Video Enhancer AI
============================================

Export results to various formats.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultExporter:
    """
    Exports enhancement results to various formats.
    
    Supported formats:
    - JSON
    - Markdown
    - CSV (for batch results)
    - HTML (for reports)
    """
    
    @staticmethod
    def export_json(
        results: List[Dict[str, Any]],
        output_path: str,
        indent: int = 2
    ) -> str:
        """
        Export results to JSON.
        
        Args:
            results: List of result dictionaries
            output_path: Output file path
            indent: JSON indentation
            
        Returns:
            Path to exported file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "total_results": len(results),
            "results": results
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=indent, ensure_ascii=False)
        
        logger.info(f"Exported {len(results)} results to {output_file}")
        return str(output_file)
    
    @staticmethod
    def export_markdown(
        results: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        Export results to Markdown.
        
        Args:
            results: List of result dictionaries
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        lines = [
            "# Enhancement Results",
            "",
            f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Results:** {len(results)}",
            "",
            "---",
            ""
        ]
        
        for i, result in enumerate(results, 1):
            lines.extend([
                f"## Result {i}",
                "",
                f"**Task ID:** `{result.get('task_id', 'N/A')}`",
                f"**Service Type:** {result.get('service_type', 'N/A')}",
                f"**File Path:** `{result.get('file_path', 'N/A')}`",
                f"**Status:** {result.get('status', 'N/A')}",
                "",
            ])
            
            if result.get("enhancement_guide"):
                lines.extend([
                    "### Enhancement Guide",
                    "",
                    result["enhancement_guide"],
                    ""
                ])
            
            if result.get("error"):
                lines.extend([
                    "### Error",
                    "",
                    f"```\n{result['error']}\n```",
                    ""
                ])
            
            lines.append("---")
            lines.append("")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        logger.info(f"Exported {len(results)} results to {output_file}")
        return str(output_file)
    
    @staticmethod
    def export_csv(
        results: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        Export results to CSV.
        
        Args:
            results: List of result dictionaries
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        import csv
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not results:
            # Create empty CSV with headers
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "task_id", "service_type", "file_path", "status",
                    "tokens_used", "model", "timestamp"
                ])
            return str(output_file)
        
        # Extract all possible keys
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        # Standardize keys
        fieldnames = [
            "task_id", "service_type", "file_path", "status",
            "tokens_used", "model", "timestamp", "error"
        ]
        
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            
            for result in results:
                writer.writerow(result)
        
        logger.info(f"Exported {len(results)} results to {output_file}")
        return str(output_file)
    
    @staticmethod
    def export_html(
        results: List[Dict[str, Any]],
        output_path: str,
        title: str = "Enhancement Results"
    ) -> str:
        """
        Export results to HTML report.
        
        Args:
            results: List of result dictionaries
            output_path: Output file path
            title: Report title
            
        Returns:
            Path to exported file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{title}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1 { color: #333; }",
            ".result { border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px; }",
            ".success { background-color: #d4edda; }",
            ".error { background-color: #f8d7da; }",
            ".metadata { color: #666; font-size: 0.9em; }",
            "pre { background-color: #f4f4f4; padding: 10px; border-radius: 3px; overflow-x: auto; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{title}</h1>",
            f"<p><strong>Exported:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            f"<p><strong>Total Results:</strong> {len(results)}</p>",
            "<hr>",
        ]
        
        for i, result in enumerate(results, 1):
            status_class = "success" if result.get("status") == "completed" else "error"
            html.extend([
                f'<div class="result {status_class}">',
                f"<h2>Result {i}</h2>",
                '<div class="metadata">',
                f"<p><strong>Task ID:</strong> {result.get('task_id', 'N/A')}</p>",
                f"<p><strong>Service Type:</strong> {result.get('service_type', 'N/A')}</p>",
                f"<p><strong>File Path:</strong> <code>{result.get('file_path', 'N/A')}</code></p>",
                f"<p><strong>Status:</strong> {result.get('status', 'N/A')}</p>",
                "</div>",
            ])
            
            if result.get("enhancement_guide"):
                html.extend([
                    "<h3>Enhancement Guide</h3>",
                    f"<pre>{result['enhancement_guide']}</pre>",
                ])
            
            if result.get("error"):
                html.extend([
                    "<h3>Error</h3>",
                    f"<pre>{result['error']}</pre>",
                ])
            
            html.append("</div>")
        
        html.extend([
            "</body>",
            "</html>"
        ])
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        
        logger.info(f"Exported {len(results)} results to {output_file}")
        return str(output_file)





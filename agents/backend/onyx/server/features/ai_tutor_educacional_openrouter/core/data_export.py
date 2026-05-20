"""
Data export system for analytics and reporting.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DataExporter:
    """
    Exports data in various formats for analysis.
    """
    
    def __init__(self):
        self.export_history: List[Dict[str, Any]] = []
    
    def export_students_data(
        self,
        students_data: List[Dict[str, Any]],
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export students data.
        
        Args:
            students_data: List of student data dictionaries
            format: Export format (json, csv)
            output_path: Optional output path
        
        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"exports/students_{timestamp}.{format}"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(students_data, f, indent=2, ensure_ascii=False, default=str)
        
        elif format == "csv":
            import csv
            if students_data:
                with open(output_file, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=students_data[0].keys())
                    writer.writeheader()
                    writer.writerows(students_data)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(students_data)} students to {output_file}")
        return str(output_file)
    
    def export_conversations_data(
        self,
        conversations_data: Dict[str, List[Dict[str, Any]]],
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """Export conversations data."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"exports/conversations_{timestamp}.{format}"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(conversations_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Exported conversations to {output_file}")
        return str(output_file)
    
    def export_metrics_data(
        self,
        metrics_data: Dict[str, Any],
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """Export metrics data."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"exports/metrics_{timestamp}.{format}"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Exported metrics to {output_file}")
        return str(output_file)

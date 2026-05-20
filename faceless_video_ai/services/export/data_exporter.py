"""
Data Exporter
Export data to various formats
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import csv
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataExporter:
    """Export data to various formats"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video/exports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_csv(
        self,
        data: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> Path:
        """
        Export data to CSV
        
        Args:
            data: List of dictionaries
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        if not data:
            raise ValueError("No data to export")
        
        if filename is None:
            filename = f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        
        output_path = self.output_dir / filename
        
        # Get all keys from all dictionaries
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        
        fieldnames = sorted(all_keys)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in data:
                # Flatten nested dictionaries
                flattened = self._flatten_dict(item)
                writer.writerow(flattened)
        
        logger.info(f"Exported {len(data)} records to CSV: {output_path}")
        return output_path
    
    def export_to_json(
        self,
        data: List[Dict[str, Any]],
        filename: Optional[str] = None,
        pretty: bool = True
    ) -> Path:
        """
        Export data to JSON
        
        Args:
            data: List of dictionaries
            filename: Output filename (optional)
            pretty: Pretty print JSON
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        
        logger.info(f"Exported {len(data)} records to JSON: {output_path}")
        return output_path
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert list to string representation
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)


_data_exporter: Optional[DataExporter] = None


def get_data_exporter(output_dir: Optional[str] = None) -> DataExporter:
    """Get data exporter instance (singleton)"""
    global _data_exporter
    if _data_exporter is None:
        _data_exporter = DataExporter(output_dir=output_dir)
    return _data_exporter


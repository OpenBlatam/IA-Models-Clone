"""
Export system for analysis results
"""

import json
import csv
import time
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import io
import zipfile

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Export format types"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    TXT = "txt"
    ZIP = "zip"


@dataclass
class ExportRequest:
    """Export request configuration"""
    id: str
    format: ExportFormat
    data: List[Dict[str, Any]]
    filename: Optional[str] = None
    include_metadata: bool = True
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    file_size: Optional[int] = None


class ExportManager:
    """Export management system"""
    
    def __init__(self):
        self._exports: Dict[str, ExportRequest] = {}
    
    def export_to_json(self, data: List[Dict[str, Any]], filename: Optional[str] = None) -> bytes:
        """Export data to JSON format"""
        export_data = {
            "metadata": {
                "exported_at": time.time(),
                "total_records": len(data),
                "format": "json"
            },
            "data": data
        }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        return json_str.encode('utf-8')
    
    def export_to_csv(self, data: List[Dict[str, Any]], filename: Optional[str] = None) -> bytes:
        """Export data to CSV format"""
        if not data:
            return b""
        
        # Get all unique keys from all records
        all_keys = set()
        for record in data:
            all_keys.update(record.keys())
        
        # Sort keys for consistent output
        fieldnames = sorted(all_keys)
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in data:
            # Flatten nested dictionaries
            flattened_record = self._flatten_dict(record)
            writer.writerow(flattened_record)
        
        return output.getvalue().encode('utf-8')
    
    def export_to_xml(self, data: List[Dict[str, Any]], filename: Optional[str] = None) -> bytes:
        """Export data to XML format"""
        xml_content = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_content.append('<export>')
        xml_content.append(f'  <metadata>')
        xml_content.append(f'    <exported_at>{time.time()}</exported_at>')
        xml_content.append(f'    <total_records>{len(data)}</total_records>')
        xml_content.append(f'    <format>xml</format>')
        xml_content.append(f'  </metadata>')
        xml_content.append(f'  <data>')
        
        for i, record in enumerate(data):
            xml_content.append(f'    <record id="{i}">')
            self._dict_to_xml(record, xml_content, indent=6)
            xml_content.append(f'    </record>')
        
        xml_content.append(f'  </data>')
        xml_content.append('</export>')
        
        return '\n'.join(xml_content).encode('utf-8')
    
    def export_to_txt(self, data: List[Dict[str, Any]], filename: Optional[str] = None) -> bytes:
        """Export data to plain text format"""
        lines = []
        lines.append("Content Redundancy Detector - Export Report")
        lines.append("=" * 50)
        lines.append(f"Exported at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        lines.append(f"Total records: {len(data)}")
        lines.append("")
        
        for i, record in enumerate(data, 1):
            lines.append(f"Record {i}:")
            lines.append("-" * 20)
            for key, value in record.items():
                lines.append(f"{key}: {value}")
            lines.append("")
        
        return '\n'.join(lines).encode('utf-8')
    
    def export_to_zip(self, data: List[Dict[str, Any]], filename: Optional[str] = None) -> bytes:
        """Export data to ZIP format with multiple files"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add JSON file
            json_data = self.export_to_json(data)
            zip_file.writestr("export.json", json_data)
            
            # Add CSV file
            csv_data = self.export_to_csv(data)
            zip_file.writestr("export.csv", csv_data)
            
            # Add XML file
            xml_data = self.export_to_xml(data)
            zip_file.writestr("export.xml", xml_data)
            
            # Add TXT file
            txt_data = self.export_to_txt(data)
            zip_file.writestr("export.txt", txt_data)
            
            # Add metadata file
            metadata = {
                "export_info": {
                    "exported_at": time.time(),
                    "total_records": len(data),
                    "files_included": ["export.json", "export.csv", "export.xml", "export.txt"]
                }
            }
            metadata_json = json.dumps(metadata, indent=2)
            zip_file.writestr("metadata.json", metadata_json.encode('utf-8'))
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert list to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _dict_to_xml(self, d: Dict[str, Any], xml_content: List[str], indent: int = 0) -> None:
        """Convert dictionary to XML"""
        indent_str = ' ' * indent
        for key, value in d.items():
            # Clean key name for XML
            clean_key = key.replace(' ', '_').replace('-', '_')
            if isinstance(value, dict):
                xml_content.append(f'{indent_str}<{clean_key}>')
                self._dict_to_xml(value, xml_content, indent + 2)
                xml_content.append(f'{indent_str}</{clean_key}>')
            elif isinstance(value, list):
                xml_content.append(f'{indent_str}<{clean_key}>')
                for item in value:
                    if isinstance(item, dict):
                        xml_content.append(f'{indent_str}  <item>')
                        self._dict_to_xml(item, xml_content, indent + 4)
                        xml_content.append(f'{indent_str}  </item>')
                    else:
                        xml_content.append(f'{indent_str}  <item>{item}</item>')
                xml_content.append(f'{indent_str}</{clean_key}>')
            else:
                xml_content.append(f'{indent_str}<{clean_key}>{value}</{clean_key}>')
    
    def create_export(self, data: List[Dict[str, Any]], format_type: ExportFormat, 
                     filename: Optional[str] = None) -> ExportRequest:
        """Create an export request"""
        export_id = f"export_{int(time.time() * 1000)}"
        
        export_request = ExportRequest(
            id=export_id,
            format=format_type,
            data=data,
            filename=filename or f"export_{export_id}.{format_type.value}"
        )
        
        self._exports[export_id] = export_request
        return export_request
    
    def process_export(self, export_request: ExportRequest) -> bytes:
        """Process export request and return file content"""
        try:
            if export_request.format == ExportFormat.JSON:
                content = self.export_to_json(export_request.data, export_request.filename)
            elif export_request.format == ExportFormat.CSV:
                content = self.export_to_csv(export_request.data, export_request.filename)
            elif export_request.format == ExportFormat.XML:
                content = self.export_to_xml(export_request.data, export_request.filename)
            elif export_request.format == ExportFormat.TXT:
                content = self.export_to_txt(export_request.data, export_request.filename)
            elif export_request.format == ExportFormat.ZIP:
                content = self.export_to_zip(export_request.data, export_request.filename)
            else:
                raise ValueError(f"Unsupported export format: {export_request.format}")
            
            # Update export request
            export_request.completed_at = time.time()
            export_request.file_size = len(content)
            
            logger.info(f"Export completed: {export_request.id} - {len(content)} bytes")
            return content
            
        except Exception as e:
            logger.error(f"Export failed: {export_request.id} - {e}")
            raise
    
    def get_export(self, export_id: str) -> Optional[ExportRequest]:
        """Get export request by ID"""
        return self._exports.get(export_id)
    
    def get_all_exports(self) -> List[ExportRequest]:
        """Get all export requests"""
        return list(self._exports.values())
    
    def cleanup_old_exports(self, max_age_hours: int = 24) -> int:
        """Clean up old export requests"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        to_remove = []
        for export_id, export_request in self._exports.items():
            if current_time - export_request.created_at > max_age_seconds:
                to_remove.append(export_id)
        
        for export_id in to_remove:
            del self._exports[export_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old exports")
        return len(to_remove)


# Global export manager
export_manager = ExportManager()


def create_export(data: List[Dict[str, Any]], format_type: ExportFormat, 
                 filename: Optional[str] = None) -> ExportRequest:
    """Create an export request"""
    return export_manager.create_export(data, format_type, filename)


def process_export(export_request: ExportRequest) -> bytes:
    """Process export request"""
    return export_manager.process_export(export_request)


def get_export(export_id: str) -> Optional[ExportRequest]:
    """Get export request"""
    return export_manager.get_export(export_id)


def get_all_exports() -> List[ExportRequest]:
    """Get all exports"""
    return export_manager.get_all_exports()


def cleanup_old_exports(max_age_hours: int = 24) -> int:
    """Clean up old exports"""
    return export_manager.cleanup_old_exports(max_age_hours)



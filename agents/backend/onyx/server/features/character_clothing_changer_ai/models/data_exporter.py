"""
Data Exporter for Flux2 Clothing Changer
========================================

Advanced data export system with multiple formats.
"""

import json
import csv
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataExporter:
    """Advanced data export system."""
    
    def __init__(self):
        """Initialize data exporter."""
        self.export_history: List[Dict[str, Any]] = []
    
    def export_json(
        self,
        data: Any,
        output_path: Path,
        indent: int = 2,
    ) -> bool:
        """
        Export data to JSON.
        
        Args:
            data: Data to export
            output_path: Output file path
            indent: JSON indentation
            
        Returns:
            True if exported
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            logger.info(f"Exported JSON to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
            return False
    
    def export_csv(
        self,
        data: List[Dict[str, Any]],
        output_path: Path,
        fieldnames: Optional[List[str]] = None,
    ) -> bool:
        """
        Export data to CSV.
        
        Args:
            data: List of dictionaries to export
            output_path: Output file path
            fieldnames: Optional field names
            
        Returns:
            True if exported
        """
        if not data:
            return False
        
        try:
            if fieldnames is None:
                fieldnames = list(data[0].keys())
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            logger.info(f"Exported CSV to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return False
    
    def export_xml(
        self,
        data: Dict[str, Any],
        output_path: Path,
        root_element: str = "root",
    ) -> bool:
        """
        Export data to XML.
        
        Args:
            data: Data dictionary to export
            output_path: Output file path
            root_element: Root element name
            
        Returns:
            True if exported
        """
        try:
            def dict_to_xml(parent, d):
                for key, value in d.items():
                    if isinstance(value, dict):
                        elem = ET.SubElement(parent, key)
                        dict_to_xml(elem, value)
                    elif isinstance(value, list):
                        for item in value:
                            elem = ET.SubElement(parent, key)
                            if isinstance(item, dict):
                                dict_to_xml(elem, item)
                            else:
                                elem.text = str(item)
                    else:
                        elem = ET.SubElement(parent, key)
                        elem.text = str(value)
            
            root = ET.Element(root_element)
            dict_to_xml(root, data)
            
            tree = ET.ElementTree(root)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tree.write(output_path, encoding="utf-8", xml_declaration=True)
            logger.info(f"Exported XML to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export XML: {e}")
            return False
    
    def export_markdown(
        self,
        data: Any,
        output_path: Path,
        title: Optional[str] = None,
    ) -> bool:
        """
        Export data to Markdown.
        
        Args:
            data: Data to export
            output_path: Output file path
            title: Optional title
            
        Returns:
            True if exported
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                if title:
                    f.write(f"# {title}\n\n")
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        f.write(f"## {key}\n\n")
                        f.write(f"{value}\n\n")
                elif isinstance(data, list):
                    for item in data:
                        f.write(f"- {item}\n")
                else:
                    f.write(str(data))
            
            logger.info(f"Exported Markdown to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export Markdown: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get exporter statistics."""
        return {
            "total_exports": len(self.export_history),
        }



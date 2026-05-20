"""
XML export handler.
"""

from typing import Dict, Any
from datetime import datetime

from .base import BaseExporter


class XMLExporter(BaseExporter):
    """XML export handler with structured markup."""
    
    def get_supported_features(self) -> list:
        """Get supported XML features."""
        return [
            "Structured data",
            "Validation support",
            "Industry standard",
            "Schema validation"
        ]
    
    async def export(
        self, 
        content: Dict[str, Any], 
        config: Any, 
        output_path: str
    ) -> Dict[str, Any]:
        """Export content to XML format."""
        try:
            # Preprocess content
            processed_content = self.preprocess_content(content)
            
            xml_content = ['<?xml version="1.0" encoding="UTF-8"?>']
            xml_content.append('<document>')
            xml_content.append(f'  <metadata>')
            xml_content.append(f'    <exported_at>{datetime.now().isoformat()}</exported_at>')
            xml_content.append(f'    <format>{config.format.value if hasattr(config.format, "value") else str(config.format)}</format>')
            xml_content.append(f'    <document_type>{config.document_type.value if hasattr(config.document_type, "value") else str(config.document_type)}</document_type>')
            xml_content.append(f'    <quality_level>{config.quality_level.value if hasattr(config.quality_level, "value") else str(config.quality_level)}</quality_level>')
            xml_content.append(f'  </metadata>')
            
            if "title" in processed_content:
                xml_content.append(f'  <title>{processed_content["title"]}</title>')
            
            if "sections" in processed_content:
                xml_content.append('  <sections>')
                for section in processed_content["sections"]:
                    xml_content.append('    <section>')
                    if "heading" in section:
                        xml_content.append(f'      <heading>{section["heading"]}</heading>')
                    if "content" in section:
                        xml_content.append(f'      <content>{section["content"]}</content>')
                    xml_content.append('    </section>')
                xml_content.append('  </sections>')
            
            xml_content.append('</document>')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(xml_content))
            
            result = {
                "format": "xml",
                "sections": len(processed_content.get("sections", [])),
                "professional_features": {
                    "xml_structure": True,
                    "validation_support": True
                }
            }
            
            return self.postprocess_result(result)
            
        except Exception as e:
            self.logger.error(f"XML export failed: {e}")
            raise
    
    def postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess XML export result."""
        result["file_type"] = "application/xml"
        result["mime_type"] = "application/xml"
        return result





"""
JSON export handler.
"""

import json
from typing import Dict, Any
from datetime import datetime

from .base import BaseExporter


class JSONExporter(BaseExporter):
    """JSON export handler with structured data."""
    
    def get_supported_features(self) -> list:
        """Get supported JSON features."""
        return [
            "Structured data",
            "API friendly",
            "Machine readable",
            "Programmatic access"
        ]
    
    async def export(
        self, 
        content: Dict[str, Any], 
        config: Any, 
        output_path: str
    ) -> Dict[str, Any]:
        """Export content to JSON format."""
        try:
            # Preprocess content
            processed_content = self.preprocess_content(content)
            
            export_data = {
                "metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "format": config.format.value if hasattr(config.format, 'value') else str(config.format),
                    "document_type": config.document_type.value if hasattr(config.document_type, 'value') else str(config.document_type),
                    "quality_level": config.quality_level.value if hasattr(config.quality_level, 'value') else str(config.quality_level)
                },
                "content": processed_content
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            result = {
                "format": "json",
                "sections": len(processed_content.get("sections", [])),
                "professional_features": {
                    "structured_data": True,
                    "api_friendly": True
                }
            }
            
            return self.postprocess_result(result)
            
        except Exception as e:
            self.logger.error(f"JSON export failed: {e}")
            raise
    
    def postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess JSON export result."""
        result["file_type"] = "application/json"
        result["mime_type"] = "application/json"
        return result





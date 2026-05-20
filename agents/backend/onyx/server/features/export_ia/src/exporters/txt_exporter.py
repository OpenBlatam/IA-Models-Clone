"""
Plain text export handler.
"""

from typing import Dict, Any

from .base import BaseExporter


class TXTExporter(BaseExporter):
    """Plain text export handler with clean formatting."""
    
    def get_supported_features(self) -> list:
        """Get supported TXT features."""
        return [
            "Universal compatibility",
            "Lightweight",
            "Fast processing",
            "No formatting dependencies"
        ]
    
    async def export(
        self, 
        content: Dict[str, Any], 
        config: Any, 
        output_path: str
    ) -> Dict[str, Any]:
        """Export content to plain text format."""
        try:
            # Preprocess content
            processed_content = self.preprocess_content(content)
            
            txt_content = []
            
            if "title" in processed_content:
                txt_content.append(processed_content['title'])
                txt_content.append("=" * len(processed_content['title']))
                txt_content.append("")
            
            if "sections" in processed_content:
                for section in processed_content["sections"]:
                    if "heading" in section:
                        txt_content.append(section['heading'])
                        txt_content.append("-" * len(section['heading']))
                    
                    if "content" in section:
                        txt_content.append(section['content'])
                        txt_content.append("")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(txt_content))
            
            result = {
                "format": "txt",
                "sections": len(processed_content.get("sections", [])),
                "professional_features": {
                    "plain_text_formatting": True,
                    "universal_compatibility": True
                }
            }
            
            return self.postprocess_result(result)
            
        except Exception as e:
            self.logger.error(f"TXT export failed: {e}")
            raise
    
    def postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess TXT export result."""
        result["file_type"] = "text/plain"
        result["mime_type"] = "text/plain"
        return result





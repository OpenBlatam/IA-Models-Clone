"""
RTF export handler.
"""

from typing import Dict, Any

from .base import BaseExporter


class RTFExporter(BaseExporter):
    """RTF export handler with rich formatting."""
    
    def get_supported_features(self) -> list:
        """Get supported RTF features."""
        return [
            "Cross platform",
            "Rich formatting",
            "Legacy support",
            "Microsoft Word compatible"
        ]
    
    async def export(
        self, 
        content: Dict[str, Any], 
        config: Any, 
        output_path: str
    ) -> Dict[str, Any]:
        """Export content to RTF format."""
        try:
            # Preprocess content
            processed_content = self.preprocess_content(content)
            
            # Basic RTF implementation
            rtf_content = "{\\rtf1\\ansi\\deff0 {\\fonttbl {\\f0 Calibri;}}\n"
            
            if "title" in processed_content:
                rtf_content += f"\\f0\\fs24\\b {processed_content['title']}\\b0\\par\\par\n"
            
            if "sections" in processed_content:
                for section in processed_content["sections"]:
                    if "heading" in section:
                        rtf_content += f"\\f0\\fs20\\b {section['heading']}\\b0\\par\n"
                    
                    if "content" in section:
                        rtf_content += f"\\f0\\fs18 {section['content']}\\par\\par\n"
            
            rtf_content += "}"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(rtf_content)
            
            result = {
                "format": "rtf",
                "sections": len(processed_content.get("sections", [])),
                "professional_features": {
                    "rtf_formatting": True,
                    "cross_platform": True
                }
            }
            
            return self.postprocess_result(result)
            
        except Exception as e:
            self.logger.error(f"RTF export failed: {e}")
            raise
    
    def postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess RTF export result."""
        result["file_type"] = "application/rtf"
        result["mime_type"] = "application/rtf"
        return result





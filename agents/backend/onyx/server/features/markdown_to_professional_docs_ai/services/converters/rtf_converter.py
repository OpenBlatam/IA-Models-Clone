"""RTF Converter - Convert Markdown to RTF format"""
from typing import Dict, Any, Optional
import re

from .base_converter import BaseConverter


class RTFConverter(BaseConverter):
    """Convert Markdown to RTF (Rich Text Format)"""
    
    async def convert(
        self,
        parsed_content: Dict[str, Any],
        output_path: str,
        include_charts: bool = True,
        include_tables: bool = True,
        custom_styling: Optional[Dict[str, Any]] = None
    ) -> None:
        """Convert to RTF format"""
        rtf_parts = []
        
        # RTF header
        rtf_parts.append("{\\rtf1\\ansi\\deff0\n")
        rtf_parts.append("{\\fonttbl{\\f0 Times New Roman;}}\n")
        rtf_parts.append("{\\colortbl;\\red0\\green0\\blue0;\\red54\\green96\\blue146;}\n")
        rtf_parts.append("\\f0\\fs24\n\n")
        
        # Title
        if parsed_content.get("title"):
            rtf_parts.append(f"\\b\\fs32 {self._escape_rtf(parsed_content['title'])}\n")
            rtf_parts.append("\\b0\\fs24\\par\n\\par\n")
        
        # Headings
        for heading in parsed_content.get("headings", []):
            level = heading["level"]
            text = self._escape_rtf(heading["text"])
            size = 28 - (level * 2)
            
            rtf_parts.append(f"\\b\\fs{size} {text}\n")
            rtf_parts.append("\\b0\\fs24\\par\n\\par\n")
        
        # Tables
        if include_tables:
            for table in parsed_content.get("tables", []):
                rtf_parts.append("\\trowd\\trgaph108\\trleft-108\n")
                
                # Column widths
                col_count = len(table.get("headers", []))
                col_width = 5000 // col_count if col_count > 0 else 5000
                for _ in range(col_count):
                    rtf_parts.append(f"\\cellx{col_width}\n")
                
                # Headers
                headers = table.get("headers", [])
                if headers:
                    rtf_parts.append("\\intbl\\b\\cf2 ")
                    for header in headers:
                        rtf_parts.append(f"{self._escape_rtf(header)}\\cell ")
                    rtf_parts.append("\\b0\\cf1\\row\n")
                
                # Rows
                for row in table.get("rows", []):
                    rtf_parts.append("\\trowd\\trgaph108\\trleft-108\n")
                    for _ in range(col_count):
                        rtf_parts.append(f"\\cellx{col_width}\n")
                    rtf_parts.append("\\intbl ")
                    for cell in row:
                        rtf_parts.append(f"{self._escape_rtf(str(cell))}\\cell ")
                    rtf_parts.append("\\row\n")
                
                rtf_parts.append("\\par\n")
        
        # Paragraphs
        for para in parsed_content.get("paragraphs", []):
            # Convert markdown to RTF
            para = re.sub(r'\*\*([^\*]+)\*\*', r'\\b \1\\b0 ', para)
            para = re.sub(r'\*([^\*]+)\*', r'\\i \1\\i0 ', para)
            
            rtf_parts.append(f"{self._escape_rtf(para)}\\par\n")
        
        # Lists
        for list_data in parsed_content.get("lists", []):
            for item in list_data["items"]:
                if list_data["type"] == "unordered":
                    rtf_parts.append(f"\\bullet {self._escape_rtf(item)}\\par\n")
                else:
                    rtf_parts.append(f"{self._escape_rtf(item)}\\par\n")
            rtf_parts.append("\\par\n")
        
        rtf_parts.append("}\n")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(rtf_parts))
    
    def _escape_rtf(self, text: str) -> str:
        """Escape special RTF characters"""
        replacements = {
            '\\': '\\\\',
            '{': '\\{',
            '}': '\\}',
            '\n': '\\par\n',
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text


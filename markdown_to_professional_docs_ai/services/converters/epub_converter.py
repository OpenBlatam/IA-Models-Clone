"""EPUB Converter - Convert Markdown to EPUB format"""
from typing import Dict, Any, Optional
import zipfile
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

from .base_converter import BaseConverter


class EPUBConverter(BaseConverter):
    """Convert Markdown to EPUB format"""
    
    async def convert(
        self,
        parsed_content: Dict[str, Any],
        output_path: str,
        include_charts: bool = True,
        include_tables: bool = True,
        custom_styling: Optional[Dict[str, Any]] = None
    ) -> None:
        """Convert to EPUB format"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create EPUB structure
            mimetype_file = temp_path / "mimetype"
            mimetype_file.write_text("application/epub+zip")
            
            # META-INF directory
            meta_inf = temp_path / "META-INF"
            meta_inf.mkdir()
            
            # container.xml
            container = ET.Element("container", version="1.0", xmlns="urn:oasis:names:tc:opendocument:xmlns:container")
            rootfiles = ET.SubElement(container, "rootfiles")
            rootfile = ET.SubElement(rootfiles, "rootfile", {
                "full-path": "OEBPS/content.opf",
                "media-type": "application/oebps-package+xml"
            })
            
            container_file = meta_inf / "container.xml"
            tree = ET.ElementTree(container)
            ET.indent(tree)
            tree.write(container_file, encoding='utf-8', xml_declaration=True)
            
            # OEBPS directory
            oebps = temp_path / "OEBPS"
            oebps.mkdir()
            
            # Create content.opf (package file)
            package = ET.Element("package", {
                "xmlns": "http://www.idpf.org/2007/opf",
                "version": "3.0",
                "unique-identifier": "book-id"
            })
            
            metadata = ET.SubElement(package, "metadata", xmlns="http://www.idpf.org/2007/opf")
            title_elem = ET.SubElement(metadata, "dc:title")
            title_elem.text = parsed_content.get("title", "Document")
            
            identifier = ET.SubElement(metadata, "dc:identifier", id="book-id")
            identifier.text = "urn:uuid:12345678-1234-1234-1234-123456789012"
            
            manifest = ET.SubElement(package, "manifest")
            item1 = ET.SubElement(manifest, "item", {
                "id": "nav",
                "href": "nav.xhtml",
                "media-type": "application/xhtml+xml",
                "properties": "nav"
            })
            item2 = ET.SubElement(manifest, "item", {
                "id": "content",
                "href": "content.xhtml",
                "media-type": "application/xhtml+xml"
            })
            
            spine = ET.SubElement(package, "spine", toc="nav")
            itemref = ET.SubElement(spine, "itemref", idref="content")
            
            opf_file = oebps / "content.opf"
            tree = ET.ElementTree(package)
            ET.indent(tree)
            tree.write(opf_file, encoding='utf-8', xml_declaration=True)
            
            # Create content.xhtml
            html_content = self._create_html_content(parsed_content)
            content_file = oebps / "content.xhtml"
            content_file.write_text(html_content, encoding='utf-8')
            
            # Create nav.xhtml
            nav_content = self._create_nav_content(parsed_content)
            nav_file = oebps / "nav.xhtml"
            nav_file.write_text(nav_content, encoding='utf-8')
            
            # Create EPUB (ZIP file)
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as epub:
                # Add mimetype first (must be uncompressed)
                epub.write(mimetype_file, "mimetype", compress_type=zipfile.ZIP_STORED)
                
                # Add other files
                for file_path in temp_path.rglob("*"):
                    if file_path.is_file() and file_path != mimetype_file:
                        arcname = file_path.relative_to(temp_path)
                        epub.write(file_path, str(arcname))
    
    def _create_html_content(self, parsed_content: Dict[str, Any]) -> str:
        """Create HTML content for EPUB"""
        html_parts = []
        html_parts.append("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
    <meta charset="UTF-8"/>
    <title>""")
        html_parts.append(parsed_content.get("title", "Document"))
        html_parts.append("""</title>
    <style>
        body { font-family: serif; padding: 1em; }
        h1 { font-size: 2em; margin-top: 1em; }
        h2 { font-size: 1.5em; margin-top: 0.8em; }
        p { margin: 1em 0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
""")
        
        # Add title
        if parsed_content.get("title"):
            html_parts.append(f"<h1>{parsed_content['title']}</h1>\n")
        
        # Add headings
        for heading in parsed_content.get("headings", []):
            level = heading["level"]
            html_parts.append(f"<h{level}>{heading['text']}</h{level}>\n")
        
        # Add tables
        for table in parsed_content.get("tables", []):
            html_parts.append("<table>\n")
            # Headers
            html_parts.append("<tr>")
            for header in table.get("headers", []):
                html_parts.append(f"<th>{header}</th>")
            html_parts.append("</tr>\n")
            # Rows
            for row in table.get("rows", []):
                html_parts.append("<tr>")
                for cell in row:
                    html_parts.append(f"<td>{cell}</td>")
                html_parts.append("</tr>\n")
            html_parts.append("</table>\n")
        
        # Add paragraphs
        for para in parsed_content.get("paragraphs", []):
            html_parts.append(f"<p>{para}</p>\n")
        
        html_parts.append("</body>\n</html>")
        
        return ''.join(html_parts)
    
    def _create_nav_content(self, parsed_content: Dict[str, Any]) -> str:
        """Create navigation content for EPUB"""
        nav_parts = []
        nav_parts.append("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
    <meta charset="UTF-8"/>
    <title>Navigation</title>
</head>
<body>
    <nav epub:type="toc">
        <ol>
""")
        
        # Add navigation items from headings
        for heading in parsed_content.get("headings", []):
            if heading["level"] <= 2:
                nav_parts.append(f'            <li><a href="content.xhtml#{heading["id"]}">{heading["text"]}</a></li>\n')
        
        nav_parts.append("""        </ol>
    </nav>
</body>
</html>""")
        
        return ''.join(nav_parts)


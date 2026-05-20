"""
Export Engine
=============

Core engine for document export and processing.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import os
from pathlib import Path
import tempfile

# Document processing libraries
import markdown
from jinja2 import Template
import json as json_lib
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class ExportFormat(Enum):
    """Supported export formats."""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"
    TXT = "txt"

class ExportStatus(Enum):
    """Export status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ExportRequest:
    """Export request data."""
    id: str
    content: Dict[str, Any]
    format: ExportFormat
    options: Dict[str, Any] = field(default_factory=dict)
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    status: ExportStatus = ExportStatus.PENDING
    file_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExportResult:
    """Export result data."""
    request_id: str
    status: ExportStatus
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    download_url: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)

class ExportEngine:
    """Core export engine for document processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.export_requests: Dict[str, ExportRequest] = {}
        self.export_results: Dict[str, ExportResult] = {}
        
        # Export settings
        self.output_directory = self.config.get("output_directory", "exports")
        self.max_file_size = self.config.get("max_file_size", 50 * 1024 * 1024)  # 50MB
        self.supported_formats = [fmt.value for fmt in ExportFormat]
        
        # Create output directory
        os.makedirs(self.output_directory, exist_ok=True)
        
        logger.info("Export Engine initialized")
    
    async def create_export_request(
        self,
        content: Dict[str, Any],
        format: ExportFormat,
        options: Optional[Dict[str, Any]] = None,
        created_by: str = "system"
    ) -> ExportRequest:
        """Create a new export request."""
        
        request_id = str(uuid.uuid4())
        
        request = ExportRequest(
            id=request_id,
            content=content,
            format=format,
            options=options or {},
            created_by=created_by
        )
        
        self.export_requests[request_id] = request
        
        logger.info(f"Created export request: {request_id}")
        
        return request
    
    async def process_export_request(self, request_id: str) -> ExportResult:
        """Process an export request."""
        
        if request_id not in self.export_requests:
            raise ValueError(f"Export request {request_id} not found")
        
        request = self.export_requests[request_id]
        request.status = ExportStatus.PROCESSING
        
        start_time = datetime.now()
        
        try:
            # Process based on format
            if request.format == ExportFormat.HTML:
                file_path = await self._export_to_html(request)
            elif request.format == ExportFormat.MARKDOWN:
                file_path = await self._export_to_markdown(request)
            elif request.format == ExportFormat.JSON:
                file_path = await self._export_to_json(request)
            elif request.format == ExportFormat.XML:
                file_path = await self._export_to_xml(request)
            elif request.format == ExportFormat.TXT:
                file_path = await self._export_to_txt(request)
            elif request.format == ExportFormat.PDF:
                file_path = await self._export_to_pdf(request)
            elif request.format == ExportFormat.DOCX:
                file_path = await self._export_to_docx(request)
            else:
                raise ValueError(f"Unsupported export format: {request.format}")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Get file size
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            # Create result
            result = ExportResult(
                request_id=request_id,
                status=ExportStatus.COMPLETED,
                file_path=file_path,
                file_size=file_size,
                download_url=f"/api/v1/exports/{request_id}/download",
                processing_time=processing_time
            )
            
            # Update request
            request.status = ExportStatus.COMPLETED
            request.file_path = file_path
            
            self.export_results[request_id] = result
            
            logger.info(f"Export completed: {request_id} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            # Handle error
            processing_time = (datetime.now() - start_time).total_seconds()
            error_message = str(e)
            
            result = ExportResult(
                request_id=request_id,
                status=ExportStatus.FAILED,
                error_message=error_message,
                processing_time=processing_time
            )
            
            # Update request
            request.status = ExportStatus.FAILED
            request.error_message = error_message
            
            self.export_results[request_id] = result
            
            logger.error(f"Export failed: {request_id} - {error_message}")
            
            return result
    
    async def _export_to_html(self, request: ExportRequest) -> str:
        """Export content to HTML format."""
        
        # Generate filename
        filename = f"export_{request.id}.html"
        file_path = os.path.join(self.output_directory, filename)
        
        # Get HTML template
        html_template = self._get_html_template()
        
        # Render content
        template = Template(html_template)
        html_content = template.render(
            title=request.content.get("title", "Document"),
            content=request.content.get("content", ""),
            metadata=request.content.get("metadata", {}),
            styles=request.options.get("styles", {}),
            timestamp=datetime.now().isoformat()
        )
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return file_path
    
    async def _export_to_markdown(self, request: ExportRequest) -> str:
        """Export content to Markdown format."""
        
        # Generate filename
        filename = f"export_{request.id}.md"
        file_path = os.path.join(self.output_directory, filename)
        
        # Convert content to markdown
        markdown_content = self._content_to_markdown(request.content)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return file_path
    
    async def _export_to_json(self, request: ExportRequest) -> str:
        """Export content to JSON format."""
        
        # Generate filename
        filename = f"export_{request.id}.json"
        file_path = os.path.join(self.output_directory, filename)
        
        # Prepare JSON data
        json_data = {
            "metadata": {
                "export_id": request.id,
                "format": request.format.value,
                "created_at": request.created_at.isoformat(),
                "created_by": request.created_by
            },
            "content": request.content,
            "options": request.options
        }
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json_lib.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return file_path
    
    async def _export_to_xml(self, request: ExportRequest) -> str:
        """Export content to XML format."""
        
        # Generate filename
        filename = f"export_{request.id}.xml"
        file_path = os.path.join(self.output_directory, filename)
        
        # Create XML structure
        root = ET.Element("export")
        root.set("id", request.id)
        root.set("format", request.format.value)
        root.set("created_at", request.created_at.isoformat())
        root.set("created_by", request.created_by)
        
        # Add content
        content_elem = ET.SubElement(root, "content")
        self._dict_to_xml(request.content, content_elem)
        
        # Add options
        options_elem = ET.SubElement(root, "options")
        self._dict_to_xml(request.options, options_elem)
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
        
        return file_path
    
    async def _export_to_txt(self, request: ExportRequest) -> str:
        """Export content to plain text format."""
        
        # Generate filename
        filename = f"export_{request.id}.txt"
        file_path = os.path.join(self.output_directory, filename)
        
        # Convert content to plain text
        text_content = self._content_to_text(request.content)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        return file_path
    
    async def _export_to_pdf(self, request: ExportRequest) -> str:
        """Export content to PDF format."""
        
        # Generate filename
        filename = f"export_{request.id}.pdf"
        file_path = os.path.join(self.output_directory, filename)
        
        # For now, create a placeholder PDF
        # In a real implementation, you would use libraries like reportlab or weasyprint
        with open(file_path, 'w') as f:
            f.write("PDF export not implemented yet")
        
        return file_path
    
    async def _export_to_docx(self, request: ExportRequest) -> str:
        """Export content to DOCX format."""
        
        # Generate filename
        filename = f"export_{request.id}.docx"
        file_path = os.path.join(self.output_directory, filename)
        
        # For now, create a placeholder DOCX
        # In a real implementation, you would use python-docx
        with open(file_path, 'w') as f:
            f.write("DOCX export not implemented yet")
        
        return file_path
    
    def _content_to_markdown(self, content: Dict[str, Any]) -> str:
        """Convert content to markdown format."""
        
        markdown_parts = []
        
        # Add title
        if "title" in content:
            markdown_parts.append(f"# {content['title']}\n")
        
        # Add metadata
        if "metadata" in content:
            markdown_parts.append("## Metadata\n")
            for key, value in content["metadata"].items():
                markdown_parts.append(f"- **{key}**: {value}")
            markdown_parts.append("")
        
        # Add main content
        if "content" in content:
            markdown_parts.append("## Content\n")
            markdown_parts.append(content["content"])
        
        # Add sections
        if "sections" in content:
            for section in content["sections"]:
                if "heading" in section:
                    markdown_parts.append(f"\n## {section['heading']}\n")
                if "content" in section:
                    markdown_parts.append(section["content"])
        
        return "\n".join(markdown_parts)
    
    def _content_to_text(self, content: Dict[str, Any]) -> str:
        """Convert content to plain text format."""
        
        text_parts = []
        
        # Add title
        if "title" in content:
            text_parts.append(content["title"])
            text_parts.append("=" * len(content["title"]))
            text_parts.append("")
        
        # Add main content
        if "content" in content:
            text_parts.append(content["content"])
            text_parts.append("")
        
        # Add sections
        if "sections" in content:
            for section in content["sections"]:
                if "heading" in section:
                    text_parts.append(section["heading"])
                    text_parts.append("-" * len(section["heading"]))
                if "content" in section:
                    text_parts.append(section["content"])
                text_parts.append("")
        
        return "\n".join(text_parts)
    
    def _dict_to_xml(self, data: Dict[str, Any], parent: ET.Element):
        """Convert dictionary to XML elements."""
        
        for key, value in data.items():
            if isinstance(value, dict):
                elem = ET.SubElement(parent, key)
                self._dict_to_xml(value, elem)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        elem = ET.SubElement(parent, key)
                        self._dict_to_xml(item, elem)
                    else:
                        elem = ET.SubElement(parent, key)
                        elem.text = str(item)
            else:
                elem = ET.SubElement(parent, key)
                elem.text = str(value)
    
    def _get_html_template(self) -> str:
        """Get HTML template for export."""
        
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .metadata {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .content {
            margin-top: 20px;
        }
        .timestamp {
            font-size: 0.9em;
            color: #666;
            text-align: right;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    
    {% if metadata %}
    <div class="metadata">
        <h3>Metadata</h3>
        {% for key, value in metadata.items() %}
        <p><strong>{{ key }}:</strong> {{ value }}</p>
        {% endfor %}
    </div>
    {% endif %}
    
    <div class="content">
        {{ content | safe }}
    </div>
    
    <div class="timestamp">
        Generated on: {{ timestamp }}
    </div>
</body>
</html>
        """
    
    def get_export_request(self, request_id: str) -> Optional[ExportRequest]:
        """Get export request by ID."""
        return self.export_requests.get(request_id)
    
    def get_export_result(self, request_id: str) -> Optional[ExportResult]:
        """Get export result by ID."""
        return self.export_results.get(request_id)
    
    def list_export_requests(
        self,
        status: Optional[ExportStatus] = None,
        created_by: Optional[str] = None
    ) -> List[ExportRequest]:
        """List export requests with optional filtering."""
        
        requests = list(self.export_requests.values())
        
        if status:
            requests = [r for r in requests if r.status == status]
        
        if created_by:
            requests = [r for r in requests if r.created_by == created_by]
        
        return sorted(requests, key=lambda x: x.created_at, reverse=True)
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export statistics."""
        
        total_requests = len(self.export_requests)
        completed_requests = len([r for r in self.export_requests.values() if r.status == ExportStatus.COMPLETED])
        failed_requests = len([r for r in self.export_requests.values() if r.status == ExportStatus.FAILED])
        processing_requests = len([r for r in self.export_requests.values() if r.status == ExportStatus.PROCESSING])
        
        # Format distribution
        format_distribution = {}
        for request in self.export_requests.values():
            format_name = request.format.value
            format_distribution[format_name] = format_distribution.get(format_name, 0) + 1
        
        return {
            "total_requests": total_requests,
            "completed_requests": completed_requests,
            "failed_requests": failed_requests,
            "processing_requests": processing_requests,
            "success_rate": completed_requests / total_requests if total_requests > 0 else 0,
            "format_distribution": format_distribution
        }



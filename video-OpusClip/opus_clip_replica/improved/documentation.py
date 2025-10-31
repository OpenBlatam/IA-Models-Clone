"""
Advanced Documentation System for OpusClip Improved
=================================================

Comprehensive documentation system with API docs, user guides, and interactive help.
"""

import asyncio
import logging
import json
import yaml
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID, uuid4
from pathlib import Path
import markdown
from jinja2 import Template, Environment, FileSystemLoader
import aiofiles

from .schemas import get_settings
from .exceptions import DocumentationError, create_documentation_error

logger = logging.getLogger(__name__)


class DocumentationType(str, Enum):
    """Documentation types"""
    API_REFERENCE = "api_reference"
    USER_GUIDE = "user_guide"
    DEVELOPER_GUIDE = "developer_guide"
    TUTORIAL = "tutorial"
    FAQ = "faq"
    CHANGELOG = "changelog"
    RELEASE_NOTES = "release_notes"


class DocumentationStatus(str, Enum):
    """Documentation status"""
    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class ContentFormat(str, Enum):
    """Content formats"""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    YAML = "yaml"
    TEXT = "text"


@dataclass
class DocumentationPage:
    """Documentation page"""
    page_id: str
    title: str
    content: str
    doc_type: DocumentationType
    format: ContentFormat
    status: DocumentationStatus
    tags: List[str]
    categories: List[str]
    author: str
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None
    version: str = "1.0.0"
    slug: str = ""
    meta_description: str = ""
    keywords: List[str] = None


@dataclass
class DocumentationSection:
    """Documentation section"""
    section_id: str
    title: str
    description: str
    pages: List[str]  # Page IDs
    order: int
    parent_section: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class DocumentationTemplate:
    """Documentation template"""
    template_id: str
    name: str
    description: str
    template_content: str
    variables: List[str]
    doc_type: DocumentationType
    created_at: datetime = None
    updated_at: datetime = None


class MarkdownProcessor:
    """Markdown processing and conversion"""
    
    def __init__(self):
        self.markdown_extensions = [
            'markdown.extensions.codehilite',
            'markdown.extensions.fenced_code',
            'markdown.extensions.tables',
            'markdown.extensions.toc',
            'markdown.extensions.attr_list',
            'markdown.extensions.def_list',
            'markdown.extensions.footnotes',
            'markdown.extensions.md_in_html'
        ]
        self.md = markdown.Markdown(extensions=self.markdown_extensions)
    
    def process_markdown(self, content: str) -> str:
        """Process markdown content to HTML"""
        try:
            html = self.md.convert(content)
            return html
        except Exception as e:
            logger.error(f"Markdown processing failed: {e}")
            raise create_documentation_error("markdown_processing", "content", e)
    
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from markdown frontmatter"""
        try:
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = parts[1].strip()
                    content_body = parts[2].strip()
                    
                    # Parse YAML frontmatter
                    metadata = yaml.safe_load(frontmatter) or {}
                    return metadata, content_body
            
            return {}, content
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {}, content
    
    def generate_table_of_contents(self, content: str) -> List[Dict[str, Any]]:
        """Generate table of contents from markdown"""
        try:
            toc = []
            lines = content.split('\n')
            
            for line in lines:
                if line.startswith('#'):
                    level = len(line) - len(line.lstrip('#'))
                    title = line.lstrip('# ').strip()
                    
                    # Generate slug
                    slug = title.lower().replace(' ', '-').replace('_', '-')
                    slug = ''.join(c for c in slug if c.isalnum() or c == '-')
                    
                    toc.append({
                        'level': level,
                        'title': title,
                        'slug': slug
                    })
            
            return toc
            
        except Exception as e:
            logger.error(f"TOC generation failed: {e}")
            return []


class TemplateEngine:
    """Documentation template engine"""
    
    def __init__(self):
        self.jinja_env = Environment(loader=FileSystemLoader('templates'))
        self.templates: Dict[str, DocumentationTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default documentation templates"""
        try:
            # API Reference Template
            api_template = DocumentationTemplate(
                template_id="api_reference_template",
                name="API Reference Template",
                description="Template for API reference documentation",
                template_content="""
# {{ title }}

{{ description }}

## Endpoints

{% for endpoint in endpoints %}
### {{ endpoint.method }} {{ endpoint.path }}

{{ endpoint.description }}

**Parameters:**
{% for param in endpoint.parameters %}
- `{{ param.name }}` ({{ param.type }}) - {{ param.description }}
{% endfor %}

**Response:**
```json
{{ endpoint.response_example }}
```

{% endfor %}
""",
                variables=["title", "description", "endpoints"],
                doc_type=DocumentationType.API_REFERENCE
            )
            self.templates["api_reference_template"] = api_template
            
            # User Guide Template
            user_guide_template = DocumentationTemplate(
                template_id="user_guide_template",
                name="User Guide Template",
                description="Template for user guide documentation",
                template_content="""
# {{ title }}

{{ description }}

## Getting Started

{{ getting_started }}

## Features

{% for feature in features %}
### {{ feature.name }}

{{ feature.description }}

{% endfor %}

## Troubleshooting

{{ troubleshooting }}
""",
                variables=["title", "description", "getting_started", "features", "troubleshooting"],
                doc_type=DocumentationType.USER_GUIDE
            )
            self.templates["user_guide_template"] = user_guide_template
            
            logger.info("Default templates loaded")
            
        except Exception as e:
            logger.error(f"Default template loading failed: {e}")
    
    def render_template(self, template_id: str, variables: Dict[str, Any]) -> str:
        """Render template with variables"""
        try:
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            jinja_template = Template(template.template_content)
            return jinja_template.render(**variables)
            
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            raise create_documentation_error("template_rendering", template_id, e)
    
    def create_template(self, template: DocumentationTemplate) -> bool:
        """Create new template"""
        try:
            self.templates[template.template_id] = template
            logger.info(f"Created template: {template.template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Template creation failed: {e}")
            raise create_documentation_error("template_creation", template.template_id, e)


class APIDocumentationGenerator:
    """API documentation generator"""
    
    def __init__(self):
        self.settings = get_settings()
        self.endpoints: List[Dict[str, Any]] = []
    
    def add_endpoint(self, method: str, path: str, description: str,
                    parameters: List[Dict[str, Any]] = None,
                    response_example: str = None, tags: List[str] = None):
        """Add API endpoint to documentation"""
        try:
            endpoint = {
                "method": method.upper(),
                "path": path,
                "description": description,
                "parameters": parameters or [],
                "response_example": response_example or "{}",
                "tags": tags or []
            }
            
            self.endpoints.append(endpoint)
            logger.info(f"Added endpoint: {method} {path}")
            
        except Exception as e:
            logger.error(f"Endpoint addition failed: {e}")
            raise create_documentation_error("endpoint_addition", path, e)
    
    def generate_api_docs(self) -> str:
        """Generate API documentation"""
        try:
            # Group endpoints by tags
            grouped_endpoints = {}
            for endpoint in self.endpoints:
                for tag in endpoint["tags"]:
                    if tag not in grouped_endpoints:
                        grouped_endpoints[tag] = []
                    grouped_endpoints[tag].append(endpoint)
            
            # Generate documentation
            docs = f"# OpusClip Improved API Documentation\n\n"
            docs += f"Version: {self.settings.app_version}\n"
            docs += f"Base URL: {self.settings.host}:{self.settings.port}{self.settings.api_prefix}\n\n"
            
            docs += "## Overview\n\n"
            docs += "The OpusClip Improved API provides comprehensive video processing, AI analysis, and content management capabilities.\n\n"
            
            # Add endpoints by group
            for tag, tag_endpoints in grouped_endpoints.items():
                docs += f"## {tag.title()}\n\n"
                
                for endpoint in tag_endpoints:
                    docs += f"### {endpoint['method']} {endpoint['path']}\n\n"
                    docs += f"{endpoint['description']}\n\n"
                    
                    if endpoint['parameters']:
                        docs += "**Parameters:**\n\n"
                        for param in endpoint['parameters']:
                            docs += f"- `{param['name']}` ({param.get('type', 'string')}) - {param['description']}\n"
                        docs += "\n"
                    
                    docs += "**Response:**\n"
                    docs += f"```json\n{endpoint['response_example']}\n```\n\n"
            
            return docs
            
        except Exception as e:
            logger.error(f"API documentation generation failed: {e}")
            raise create_documentation_error("api_docs_generation", "api", e)
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI specification"""
        try:
            openapi_spec = {
                "openapi": "3.0.0",
                "info": {
                    "title": "OpusClip Improved API",
                    "version": self.settings.app_version,
                    "description": "Advanced video processing and AI-powered content creation platform",
                    "contact": {
                        "name": "OpusClip Support",
                        "email": "support@opusclip.com"
                    }
                },
                "servers": [
                    {
                        "url": f"http://{self.settings.host}:{self.settings.port}{self.settings.api_prefix}",
                        "description": "Development server"
                    }
                ],
                "paths": {},
                "components": {
                    "schemas": {},
                    "securitySchemes": {
                        "bearerAuth": {
                            "type": "http",
                            "scheme": "bearer",
                            "bearerFormat": "JWT"
                        }
                    }
                }
            }
            
            # Add paths
            for endpoint in self.endpoints:
                path = endpoint['path']
                method = endpoint['method'].lower()
                
                if path not in openapi_spec['paths']:
                    openapi_spec['paths'][path] = {}
                
                openapi_spec['paths'][path][method] = {
                    "summary": endpoint['description'],
                    "tags": endpoint['tags'],
                    "parameters": [
                        {
                            "name": param['name'],
                            "in": param.get('in', 'query'),
                            "description": param['description'],
                            "required": param.get('required', False),
                            "schema": {
                                "type": param.get('type', 'string')
                            }
                        }
                        for param in endpoint['parameters']
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "example": json.loads(endpoint['response_example'])
                                }
                            }
                        }
                    }
                }
            
            return openapi_spec
            
        except Exception as e:
            logger.error(f"OpenAPI spec generation failed: {e}")
            raise create_documentation_error("openapi_generation", "api", e)


class DocumentationManager:
    """Main documentation management system"""
    
    def __init__(self):
        self.settings = get_settings()
        self.pages: Dict[str, DocumentationPage] = {}
        self.sections: Dict[str, DocumentationSection] = {}
        self.markdown_processor = MarkdownProcessor()
        self.template_engine = TemplateEngine()
        self.api_generator = APIDocumentationGenerator()
        
        self._initialize_documentation()
    
    def _initialize_documentation(self):
        """Initialize documentation system"""
        try:
            # Create documentation directories
            docs_path = Path("docs")
            docs_path.mkdir(exist_ok=True)
            
            (docs_path / "api").mkdir(exist_ok=True)
            (docs_path / "user-guide").mkdir(exist_ok=True)
            (docs_path / "developer-guide").mkdir(exist_ok=True)
            (docs_path / "tutorials").mkdir(exist_ok=True)
            
            # Load existing documentation
            self._load_existing_documentation()
            
            logger.info("Documentation system initialized")
            
        except Exception as e:
            logger.error(f"Documentation initialization failed: {e}")
            raise create_documentation_error("docs_init", "system", e)
    
    def _load_existing_documentation(self):
        """Load existing documentation files"""
        try:
            docs_path = Path("docs")
            
            for doc_file in docs_path.rglob("*.md"):
                try:
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract metadata
                    metadata, content_body = self.markdown_processor.extract_metadata(content)
                    
                    # Create page
                    page = DocumentationPage(
                        page_id=str(uuid4()),
                        title=metadata.get('title', doc_file.stem),
                        content=content_body,
                        doc_type=DocumentationType(metadata.get('type', 'user_guide')),
                        format=ContentFormat.MARKDOWN,
                        status=DocumentationStatus(metadata.get('status', 'published')),
                        tags=metadata.get('tags', []),
                        categories=metadata.get('categories', []),
                        author=metadata.get('author', 'system'),
                        created_at=datetime.fromtimestamp(doc_file.stat().st_ctime),
                        updated_at=datetime.fromtimestamp(doc_file.stat().st_mtime),
                        slug=metadata.get('slug', doc_file.stem),
                        meta_description=metadata.get('description', ''),
                        keywords=metadata.get('keywords', [])
                    )
                    
                    self.pages[page.page_id] = page
                    
                except Exception as e:
                    logger.warning(f"Failed to load documentation file {doc_file}: {e}")
            
            logger.info(f"Loaded {len(self.pages)} documentation pages")
            
        except Exception as e:
            logger.error(f"Documentation loading failed: {e}")
    
    def create_page(self, title: str, content: str, doc_type: DocumentationType,
                   author: str, tags: List[str] = None, categories: List[str] = None,
                   format: ContentFormat = ContentFormat.MARKDOWN) -> DocumentationPage:
        """Create new documentation page"""
        try:
            page = DocumentationPage(
                page_id=str(uuid4()),
                title=title,
                content=content,
                doc_type=doc_type,
                format=format,
                status=DocumentationStatus.DRAFT,
                tags=tags or [],
                categories=categories or [],
                author=author,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                slug=self._generate_slug(title)
            )
            
            self.pages[page.page_id] = page
            logger.info(f"Created documentation page: {page.page_id}")
            return page
            
        except Exception as e:
            logger.error(f"Documentation page creation failed: {e}")
            raise create_documentation_error("page_creation", title, e)
    
    def _generate_slug(self, title: str) -> str:
        """Generate URL slug from title"""
        slug = title.lower()
        slug = ''.join(c for c in slug if c.isalnum() or c == ' ')
        slug = slug.replace(' ', '-')
        return slug
    
    def get_page(self, page_id: str) -> Optional[DocumentationPage]:
        """Get documentation page"""
        return self.pages.get(page_id)
    
    def update_page(self, page_id: str, updates: Dict[str, Any]) -> bool:
        """Update documentation page"""
        try:
            page = self.get_page(page_id)
            if not page:
                raise ValueError(f"Page {page_id} not found")
            
            # Update fields
            for key, value in updates.items():
                if hasattr(page, key):
                    setattr(page, key, value)
            
            page.updated_at = datetime.utcnow()
            
            logger.info(f"Updated documentation page: {page_id}")
            return True
            
        except Exception as e:
            logger.error(f"Documentation page update failed: {e}")
            raise create_documentation_error("page_update", page_id, e)
    
    def publish_page(self, page_id: str) -> bool:
        """Publish documentation page"""
        try:
            page = self.get_page(page_id)
            if not page:
                raise ValueError(f"Page {page_id} not found")
            
            page.status = DocumentationStatus.PUBLISHED
            page.published_at = datetime.utcnow()
            page.updated_at = datetime.utcnow()
            
            logger.info(f"Published documentation page: {page_id}")
            return True
            
        except Exception as e:
            logger.error(f"Documentation page publishing failed: {e}")
            raise create_documentation_error("page_publishing", page_id, e)
    
    def render_page(self, page_id: str) -> str:
        """Render documentation page to HTML"""
        try:
            page = self.get_page(page_id)
            if not page:
                raise ValueError(f"Page {page_id} not found")
            
            if page.format == ContentFormat.MARKDOWN:
                html_content = self.markdown_processor.process_markdown(page.content)
            else:
                html_content = page.content
            
            # Generate table of contents
            toc = self.markdown_processor.generate_table_of_contents(page.content)
            
            # Create HTML template
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{{ title }}</title>
                <meta name="description" content="{{ meta_description }}">
                <meta name="keywords" content="{{ keywords }}">
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    .toc { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }
                    .toc h3 { margin-top: 0; }
                    .toc ul { list-style: none; padding-left: 0; }
                    .toc li { margin: 5px 0; }
                    .toc a { text-decoration: none; color: #333; }
                    .content { line-height: 1.6; }
                    .meta { color: #666; font-size: 0.9em; margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <h1>{{ title }}</h1>
                <div class="meta">
                    <p>Author: {{ author }} | Created: {{ created_at }} | Updated: {{ updated_at }}</p>
                    <p>Tags: {{ tags | join(', ') }}</p>
                </div>
                
                {% if toc %}
                <div class="toc">
                    <h3>Table of Contents</h3>
                    <ul>
                    {% for item in toc %}
                        <li style="margin-left: {{ (item.level - 1) * 20 }}px;">
                            <a href="#{{ item.slug }}">{{ item.title }}</a>
                        </li>
                    {% endfor %}
                    </ul>
                </div>
                {% endif %}
                
                <div class="content">
                    {{ content | safe }}
                </div>
            </body>
            </html>
            """
            
            template = Template(html_template)
            return template.render(
                title=page.title,
                meta_description=page.meta_description,
                keywords=', '.join(page.keywords or []),
                author=page.author,
                created_at=page.created_at.strftime('%Y-%m-%d %H:%M'),
                updated_at=page.updated_at.strftime('%Y-%m-%d %H:%M'),
                tags=page.tags,
                toc=toc,
                content=html_content
            )
            
        except Exception as e:
            logger.error(f"Page rendering failed: {e}")
            raise create_documentation_error("page_rendering", page_id, e)
    
    def search_documentation(self, query: str, doc_type: Optional[DocumentationType] = None) -> List[DocumentationPage]:
        """Search documentation"""
        try:
            results = []
            query_lower = query.lower()
            
            for page in self.pages.values():
                # Check if doc type matches
                if doc_type and page.doc_type != doc_type:
                    continue
                
                # Search in title, content, tags
                if (query_lower in page.title.lower() or
                    query_lower in page.content.lower() or
                    any(query_lower in tag.lower() for tag in page.tags)):
                    results.append(page)
            
            # Sort by relevance (title matches first)
            results.sort(key=lambda x: (
                query_lower not in x.title.lower(),
                x.updated_at
            ))
            
            return results
            
        except Exception as e:
            logger.error(f"Documentation search failed: {e}")
            return []
    
    def generate_api_documentation(self) -> str:
        """Generate comprehensive API documentation"""
        try:
            # Add all API endpoints
            self._add_api_endpoints()
            
            # Generate documentation
            api_docs = self.api_generator.generate_api_docs()
            
            # Create API documentation page
            page = self.create_page(
                title="API Reference",
                content=api_docs,
                doc_type=DocumentationType.API_REFERENCE,
                author="system",
                tags=["api", "reference", "endpoints"],
                categories=["developer"]
            )
            
            # Publish the page
            self.publish_page(page.page_id)
            
            return api_docs
            
        except Exception as e:
            logger.error(f"API documentation generation failed: {e}")
            raise create_documentation_error("api_docs_generation", "api", e)
    
    def _add_api_endpoints(self):
        """Add all API endpoints to documentation"""
        try:
            # Video Analysis Endpoints
            self.api_generator.add_endpoint(
                method="POST",
                path="/analyze",
                description="Analyze video content with AI",
                parameters=[
                    {"name": "video_url", "type": "string", "description": "URL of the video to analyze"},
                    {"name": "analysis_type", "type": "string", "description": "Type of analysis to perform"}
                ],
                response_example='{"analysis_id": "uuid", "status": "completed", "results": {}}',
                tags=["video", "analysis"]
            )
            
            # Add more endpoints...
            # (This would include all 28 API endpoints)
            
        except Exception as e:
            logger.error(f"API endpoints addition failed: {e}")
    
    def export_documentation(self, format: str = "html", output_path: str = "docs_export") -> bool:
        """Export documentation in specified format"""
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(exist_ok=True)
            
            for page in self.pages.values():
                if page.status == DocumentationStatus.PUBLISHED:
                    if format == "html":
                        html_content = self.render_page(page.page_id)
                        file_path = output_dir / f"{page.slug}.html"
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                    
                    elif format == "markdown":
                        file_path = output_dir / f"{page.slug}.md"
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(f"# {page.title}\n\n{page.content}")
            
            logger.info(f"Documentation exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Documentation export failed: {e}")
            raise create_documentation_error("docs_export", format, e)
    
    def get_documentation_statistics(self) -> Dict[str, Any]:
        """Get documentation statistics"""
        try:
            total_pages = len(self.pages)
            published_pages = len([p for p in self.pages.values() if p.status == DocumentationStatus.PUBLISHED])
            
            # Count by type
            type_counts = {}
            for page in self.pages.values():
                doc_type = page.doc_type.value
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            # Count by author
            author_counts = {}
            for page in self.pages.values():
                author = page.author
                author_counts[author] = author_counts.get(author, 0) + 1
            
            return {
                "total_pages": total_pages,
                "published_pages": published_pages,
                "draft_pages": total_pages - published_pages,
                "type_distribution": type_counts,
                "author_distribution": author_counts,
                "templates_count": len(self.template_engine.templates),
                "api_endpoints_count": len(self.api_generator.endpoints)
            }
            
        except Exception as e:
            logger.error(f"Documentation statistics failed: {e}")
            return {}


# Global documentation manager
documentation_manager = DocumentationManager()






























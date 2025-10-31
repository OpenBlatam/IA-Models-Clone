"""
API Explorer and Documentation Generator
========================================

Advanced API documentation generator with interactive explorer,
endpoint testing, and automatic documentation updates.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import yaml
import requests
from jinja2 import Template
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class APIEndpoint:
    """API endpoint definition"""
    path: str
    method: str
    summary: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    security: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class APISchema:
    """Complete API schema"""
    title: str
    version: str
    description: str
    base_url: str
    endpoints: List[APIEndpoint] = field(default_factory=list)
    security_schemes: Dict[str, Any] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

class APIDocumentationGenerator:
    """
    Advanced API documentation generator
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API documentation generator
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.schema = APISchema(
            title="AI Document Classifier API",
            version="2.0.0",
            description="Advanced AI-powered document classification and template generation system",
            base_url=base_url
        )
        
        # Load API endpoints
        self._load_api_endpoints()
        
        # Templates for documentation
        self.templates = self._load_templates()
    
    def _load_api_endpoints(self):
        """Load API endpoints from the application"""
        # Standard endpoints
        standard_endpoints = [
            APIEndpoint(
                path="/classify",
                method="POST",
                summary="Classify Document Type",
                description="Classify a document type from text description using AI or pattern matching",
                parameters=[],
                request_body={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Text query describing the document"},
                        "use_ai": {"type": "boolean", "description": "Whether to use AI classification", "default": True}
                    },
                    "required": ["query"]
                },
                responses={
                    "200": {
                        "description": "Successful classification",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "document_type": {"type": "string"},
                                        "confidence": {"type": "number"},
                                        "keywords": {"type": "array", "items": {"type": "string"}},
                                        "reasoning": {"type": "string"},
                                        "template_suggestions": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            }
                        }
                    }
                },
                tags=["Classification"],
                examples=[
                    {
                        "name": "Novel Classification",
                        "request": {"query": "I want to write a science fiction novel", "use_ai": True},
                        "response": {
                            "document_type": "novel",
                            "confidence": 0.95,
                            "keywords": ["novel", "science fiction", "story"],
                            "reasoning": "Clear indicators of fictional narrative content",
                            "template_suggestions": ["Standard Novel", "Science Fiction Novel"]
                        }
                    }
                ]
            ),
            APIEndpoint(
                path="/templates/{document_type}",
                method="GET",
                summary="Get Templates",
                description="Get all available templates for a specific document type",
                parameters=[
                    {
                        "name": "document_type",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Document type (novel, contract, design, etc.)"
                    }
                ],
                responses={
                    "200": {
                        "description": "List of templates",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "document_type": {"type": "string"},
                                        "templates": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "sections": {"type": "array"},
                                                    "formatting": {"type": "object"},
                                                    "metadata": {"type": "object"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                tags=["Templates"]
            ),
            APIEndpoint(
                path="/export-template",
                method="POST",
                summary="Export Template",
                description="Export a template in specified format",
                request_body={
                    "type": "object",
                    "properties": {
                        "document_type": {"type": "string", "description": "Document type"},
                        "template_name": {"type": "string", "description": "Template name"},
                        "format": {"type": "string", "enum": ["json", "yaml", "markdown"], "default": "json"}
                    },
                    "required": ["document_type"]
                },
                responses={
                    "200": {
                        "description": "Exported template",
                        "content": {
                            "application/json": {"schema": {"type": "object"}},
                            "text/yaml": {"schema": {"type": "string"}},
                            "text/markdown": {"schema": {"type": "string"}}
                        }
                    }
                },
                tags=["Templates"]
            )
        ]
        
        # Enhanced endpoints
        enhanced_endpoints = [
            APIEndpoint(
                path="/ai-document-classifier/v2/classify/enhanced",
                method="POST",
                summary="Enhanced Document Classification",
                description="Advanced document classification with ML models and feature extraction",
                request_body={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Text query describing the document"},
                        "use_ai": {"type": "boolean", "description": "Whether to use AI classification", "default": True},
                        "use_advanced": {"type": "boolean", "description": "Whether to use advanced ML classification", "default": False},
                        "classification_method": {"type": "string", "description": "Specific classification method"},
                        "extract_features": {"type": "boolean", "description": "Whether to extract detailed features", "default": True},
                        "external_service": {"type": "string", "description": "External service to use"},
                        "language": {"type": "string", "description": "Language of the query", "default": "en"},
                        "context": {"type": "object", "description": "Additional context for classification"}
                    },
                    "required": ["query"]
                },
                responses={
                    "200": {
                        "description": "Enhanced classification result",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "document_type": {"type": "string"},
                                        "confidence": {"type": "number"},
                                        "method_used": {"type": "string"},
                                        "keywords": {"type": "array", "items": {"type": "string"}},
                                        "reasoning": {"type": "string"},
                                        "features": {"type": "object"},
                                        "processing_time": {"type": "number"},
                                        "alternative_types": {"type": "array"},
                                        "template_suggestions": {"type": "array", "items": {"type": "string"}},
                                        "external_service_used": {"type": "string"},
                                        "timestamp": {"type": "string", "format": "date-time"}
                                    }
                                }
                            }
                        }
                    }
                },
                tags=["Enhanced Classification"]
            ),
            APIEndpoint(
                path="/ai-document-classifier/v2/classify/batch",
                method="POST",
                summary="Batch Document Classification",
                description="Classify multiple documents in batch with analytics",
                request_body={
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of text queries to classify"
                        },
                        "batch_id": {"type": "string", "description": "Optional batch identifier"},
                        "use_cache": {"type": "boolean", "description": "Whether to use cached results", "default": True},
                        "use_advanced": {"type": "boolean", "description": "Whether to use advanced classification", "default": False},
                        "max_workers": {"type": "integer", "description": "Maximum number of worker threads"},
                        "progress_callback_url": {"type": "string", "description": "URL for progress callbacks"}
                    },
                    "required": ["queries"]
                },
                responses={
                    "200": {
                        "description": "Batch processing result",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "batch_id": {"type": "string"},
                                        "total_jobs": {"type": "integer"},
                                        "completed_jobs": {"type": "integer"},
                                        "failed_jobs": {"type": "integer"},
                                        "processing_time": {"type": "number"},
                                        "success_rate": {"type": "number"},
                                        "analytics": {"type": "object"},
                                        "results": {"type": "array"},
                                        "errors": {"type": "array"}
                                    }
                                }
                            }
                        }
                    }
                },
                tags=["Batch Processing"]
            ),
            APIEndpoint(
                path="/ai-document-classifier/v2/templates/generate",
                method="POST",
                summary="Generate Dynamic Template",
                description="Generate a dynamic template based on requirements",
                request_body={
                    "type": "object",
                    "properties": {
                        "document_type": {"type": "string", "description": "Type of document"},
                        "complexity": {"type": "string", "enum": ["basic", "intermediate", "advanced", "professional"], "default": "intermediate"},
                        "style_preset": {"type": "string", "enum": ["academic", "business", "creative", "technical"], "default": "business"},
                        "custom_requirements": {"type": "object", "description": "Custom requirements"},
                        "genre": {"type": "string", "description": "Document genre"},
                        "industry": {"type": "string", "description": "Target industry"},
                        "language": {"type": "string", "description": "Template language", "default": "en"}
                    },
                    "required": ["document_type"]
                },
                responses={
                    "200": {
                        "description": "Generated template",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "template": {"type": "object"},
                                        "generated_at": {"type": "string", "format": "date-time"},
                                        "template_id": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                tags=["Dynamic Templates"]
            ),
            APIEndpoint(
                path="/ai-document-classifier/v2/analytics",
                method="GET",
                summary="Get Analytics",
                description="Get comprehensive analytics and performance metrics",
                responses={
                    "200": {
                        "description": "Analytics data",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "batch_processing": {"type": "object"},
                                        "classifier_performance": {"type": "object"},
                                        "services": {"type": "object"},
                                        "system": {"type": "object"}
                                    }
                                }
                            }
                        }
                    }
                },
                tags=["Analytics"]
            ),
            APIEndpoint(
                path="/ai-document-classifier/v2/health/enhanced",
                method="GET",
                summary="Enhanced Health Check",
                description="Comprehensive health check with detailed system status",
                responses={
                    "200": {
                        "description": "Health status",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                                        "version": {"type": "string"},
                                        "components": {"type": "object"},
                                        "timestamp": {"type": "string", "format": "date-time"},
                                        "uptime": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                tags=["Health"]
            )
        ]
        
        self.schema.endpoints = standard_endpoints + enhanced_endpoints
    
    def _load_templates(self) -> Dict[str, str]:
        """Load HTML templates for documentation"""
        return {
            "main": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ schema.title }} - API Documentation</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 8px 8px 0 0; }
        .header h1 { margin: 0; font-size: 2.5em; }
        .header p { margin: 10px 0 0 0; opacity: 0.9; }
        .content { padding: 40px; }
        .endpoint { margin-bottom: 40px; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }
        .endpoint-header { background: #f8f9fa; padding: 20px; border-bottom: 1px solid #e0e0e0; }
        .method { display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; margin-right: 10px; }
        .method.post { background: #28a745; color: white; }
        .method.get { background: #007bff; color: white; }
        .method.put { background: #ffc107; color: black; }
        .method.delete { background: #dc3545; color: white; }
        .path { font-family: monospace; font-size: 1.2em; }
        .description { margin: 10px 0; color: #666; }
        .endpoint-body { padding: 20px; }
        .section { margin-bottom: 20px; }
        .section h3 { margin: 0 0 10px 0; color: #333; }
        .code-block { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 4px; padding: 15px; font-family: monospace; overflow-x: auto; }
        .parameter { margin-bottom: 10px; padding: 10px; background: #f8f9fa; border-radius: 4px; }
        .parameter-name { font-weight: bold; color: #007bff; }
        .parameter-type { color: #6c757d; font-size: 0.9em; }
        .example { margin-top: 15px; }
        .example h4 { margin: 0 0 10px 0; color: #333; }
        .tag { display: inline-block; background: #e9ecef; color: #495057; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 5px; }
        .toc { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
        .toc h2 { margin: 0 0 15px 0; }
        .toc ul { list-style: none; padding: 0; }
        .toc li { margin: 5px 0; }
        .toc a { text-decoration: none; color: #007bff; }
        .toc a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ schema.title }}</h1>
            <p>{{ schema.description }}</p>
            <p>Version: {{ schema.version }} | Base URL: {{ schema.base_url }}</p>
        </div>
        
        <div class="content">
            <div class="toc">
                <h2>Table of Contents</h2>
                <ul>
                    {% for endpoint in schema.endpoints %}
                    <li><a href="#{{ endpoint.path.replace('/', '').replace('{', '').replace('}', '') }}-{{ endpoint.method.lower() }}">{{ endpoint.method }} {{ endpoint.path }}</a></li>
                    {% endfor %}
                </ul>
            </div>
            
            {% for endpoint in schema.endpoints %}
            <div class="endpoint" id="{{ endpoint.path.replace('/', '').replace('{', '').replace('}', '') }}-{{ endpoint.method.lower() }}">
                <div class="endpoint-header">
                    <span class="method {{ endpoint.method.lower() }}">{{ endpoint.method }}</span>
                    <span class="path">{{ endpoint.path }}</span>
                    <div class="description">{{ endpoint.summary }}</div>
                    {% if endpoint.tags %}
                    <div style="margin-top: 10px;">
                        {% for tag in endpoint.tags %}
                        <span class="tag">{{ tag }}</span>
                        {% endfor %}
                    </div>
                    {% endif %}
                </div>
                
                <div class="endpoint-body">
                    <div class="section">
                        <h3>Description</h3>
                        <p>{{ endpoint.description }}</p>
                    </div>
                    
                    {% if endpoint.parameters %}
                    <div class="section">
                        <h3>Parameters</h3>
                        {% for param in endpoint.parameters %}
                        <div class="parameter">
                            <div class="parameter-name">{{ param.name }}</div>
                            <div class="parameter-type">{{ param.schema.type }} {% if param.required %}(required){% endif %}</div>
                            <div>{{ param.description }}</div>
                        </div>
                        {% endfor %}
                    </div>
                    {% endif %}
                    
                    {% if endpoint.request_body %}
                    <div class="section">
                        <h3>Request Body</h3>
                        <div class="code-block">
                            {{ endpoint.request_body | tojson(indent=2) }}
                        </div>
                    </div>
                    {% endif %}
                    
                    <div class="section">
                        <h3>Responses</h3>
                        {% for status_code, response in endpoint.responses.items() %}
                        <div class="parameter">
                            <div class="parameter-name">{{ status_code }} - {{ response.description }}</div>
                            {% if response.content %}
                            <div class="code-block">
                                {{ response.content | tojson(indent=2) }}
                            </div>
                            {% endif %}
                        </div>
                        {% endfor %}
                    </div>
                    
                    {% if endpoint.examples %}
                    <div class="section">
                        <h3>Examples</h3>
                        {% for example in endpoint.examples %}
                        <div class="example">
                            <h4>{{ example.name }}</h4>
                            <div class="code-block">
                                <strong>Request:</strong><br>
                                {{ example.request | tojson(indent=2) }}<br><br>
                                <strong>Response:</strong><br>
                                {{ example.response | tojson(indent=2) }}
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
            """,
            "openapi": """
openapi: 3.0.3
info:
  title: {{ schema.title }}
  version: {{ schema.version }}
  description: {{ schema.description }}
  contact:
    name: AI Document Classifier Support
    email: support@example.com
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT

servers:
  - url: {{ schema.base_url }}
    description: Production server

paths:
{% for endpoint in schema.endpoints %}
  {{ endpoint.path }}:
    {{ endpoint.method.lower() }}:
      summary: {{ endpoint.summary }}
      description: {{ endpoint.description }}
      tags: {{ endpoint.tags }}
      {% if endpoint.parameters %}
      parameters:
        {% for param in endpoint.parameters %}
        - name: {{ param.name }}
          in: {{ param.in }}
          required: {{ param.required | lower }}
          schema: {{ param.schema | tojson }}
          description: {{ param.description }}
        {% endfor %}
      {% endif %}
      {% if endpoint.request_body %}
      requestBody:
        required: true
        content:
          application/json:
            schema: {{ endpoint.request_body | tojson }}
      {% endif %}
      responses:
        {% for status_code, response in endpoint.responses.items() %}
        {{ status_code }}:
          description: {{ response.description }}
          {% if response.content %}
          content:
            {% for content_type, content in response.content.items() %}
            {{ content_type }}:
              schema: {{ content.schema | tojson }}
            {% endfor %}
          {% endif %}
        {% endfor %}
{% endfor %}

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
    apiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
            """
        }
    
    def generate_html_documentation(self) -> str:
        """Generate HTML documentation"""
        template = Template(self.templates["main"])
        return template.render(schema=self.schema)
    
    def generate_openapi_spec(self) -> str:
        """Generate OpenAPI specification"""
        template = Template(self.templates["openapi"])
        return template.render(schema=self.schema)
    
    def generate_markdown_documentation(self) -> str:
        """Generate Markdown documentation"""
        md = f"# {self.schema.title}\n\n"
        md += f"**Version:** {self.schema.version}\n\n"
        md += f"**Base URL:** {self.schema.base_url}\n\n"
        md += f"{self.schema.description}\n\n"
        
        # Table of contents
        md += "## Table of Contents\n\n"
        for endpoint in self.schema.endpoints:
            anchor = endpoint.path.replace("/", "").replace("{", "").replace("}", "").replace("-", "")
            md += f"- [{endpoint.method} {endpoint.path}](#{anchor}-{endpoint.method.lower()})\n"
        md += "\n"
        
        # Endpoints
        for endpoint in self.schema.endpoints:
            anchor = endpoint.path.replace("/", "").replace("{", "").replace("}", "").replace("-", "")
            md += f"## {endpoint.method} {endpoint.path} {{#{anchor}-{endpoint.method.lower()}}}\n\n"
            md += f"**Summary:** {endpoint.summary}\n\n"
            md += f"{endpoint.description}\n\n"
            
            if endpoint.tags:
                md += f"**Tags:** {', '.join(endpoint.tags)}\n\n"
            
            if endpoint.parameters:
                md += "### Parameters\n\n"
                for param in endpoint.parameters:
                    md += f"- **{param['name']}** (`{param['schema']['type']}`) - {param['description']}"
                    if param.get('required'):
                        md += " *(required)*"
                    md += "\n"
                md += "\n"
            
            if endpoint.request_body:
                md += "### Request Body\n\n"
                md += "```json\n"
                md += json.dumps(endpoint.request_body, indent=2)
                md += "\n```\n\n"
            
            md += "### Responses\n\n"
            for status_code, response in endpoint.responses.items():
                md += f"#### {status_code} - {response['description']}\n\n"
                if response.get('content'):
                    md += "```json\n"
                    md += json.dumps(response['content'], indent=2)
                    md += "\n```\n\n"
            
            if endpoint.examples:
                md += "### Examples\n\n"
                for example in endpoint.examples:
                    md += f"#### {example['name']}\n\n"
                    md += "**Request:**\n"
                    md += "```json\n"
                    md += json.dumps(example['request'], indent=2)
                    md += "\n```\n\n"
                    md += "**Response:**\n"
                    md += "```json\n"
                    md += json.dumps(example['response'], indent=2)
                    md += "\n```\n\n"
            
            md += "---\n\n"
        
        return md
    
    def save_documentation(self, output_dir: str = "docs"):
        """Save documentation in multiple formats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save HTML documentation
        html_doc = self.generate_html_documentation()
        with open(output_path / "index.html", "w", encoding="utf-8") as f:
            f.write(html_doc)
        
        # Save OpenAPI specification
        openapi_spec = self.generate_openapi_spec()
        with open(output_path / "openapi.yaml", "w", encoding="utf-8") as f:
            f.write(openapi_spec)
        
        # Save Markdown documentation
        md_doc = self.generate_markdown_documentation()
        with open(output_path / "README.md", "w", encoding="utf-8") as f:
            f.write(md_doc)
        
        # Save JSON schema
        schema_dict = {
            "title": self.schema.title,
            "version": self.schema.version,
            "description": self.schema.description,
            "base_url": self.schema.base_url,
            "endpoints": [
                {
                    "path": endpoint.path,
                    "method": endpoint.method,
                    "summary": endpoint.summary,
                    "description": endpoint.description,
                    "parameters": endpoint.parameters,
                    "request_body": endpoint.request_body,
                    "responses": endpoint.responses,
                    "tags": endpoint.tags,
                    "examples": endpoint.examples
                }
                for endpoint in self.schema.endpoints
            ],
            "generated_at": self.schema.generated_at.isoformat()
        }
        
        with open(output_path / "api_schema.json", "w", encoding="utf-8") as f:
            json.dump(schema_dict, f, indent=2)
        
        logger.info(f"Documentation saved to {output_path}")
    
    async def test_endpoints(self) -> Dict[str, Any]:
        """Test all API endpoints"""
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for endpoint in self.schema.endpoints:
                if endpoint.method == "GET":
                    try:
                        url = f"{self.base_url}{endpoint.path}"
                        # Replace path parameters with test values
                        url = url.replace("{document_type}", "novel")
                        
                        async with session.get(url) as response:
                            results[endpoint.path] = {
                                "status": response.status,
                                "method": endpoint.method,
                                "success": response.status < 400,
                                "response_time": response.headers.get("X-Response-Time", "N/A")
                            }
                    except Exception as e:
                        results[endpoint.path] = {
                            "status": "error",
                            "method": endpoint.method,
                            "success": False,
                            "error": str(e)
                        }
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize documentation generator
    doc_generator = APIDocumentationGenerator("http://localhost:8000")
    
    # Generate and save documentation
    doc_generator.save_documentation("docs")
    
    # Test endpoints
    import asyncio
    test_results = asyncio.run(doc_generator.test_endpoints())
    print("Endpoint test results:")
    for endpoint, result in test_results.items():
        print(f"  {endpoint}: {result}")
    
    print("API documentation generated successfully")




























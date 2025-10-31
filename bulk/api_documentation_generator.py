"""
BUL API Documentation Generator
==============================

Automated API documentation generator for the BUL system.
"""

import os
import sys
import json
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from modules.api_handler import APIHandler, DocumentRequest, DocumentResponse, TaskStatus
from bul_optimized import BULSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIDocumentationGenerator:
    """Generate comprehensive API documentation."""
    
    def __init__(self):
        self.api_info = {
            'title': 'BUL - Business Universal Language API',
            'version': '3.0.0',
            'description': 'AI-powered document generation system for SMEs',
            'base_url': 'http://localhost:8000',
            'contact': {
                'name': 'BUL Support',
                'email': 'support@bul-system.com'
            }
        }
        
        self.endpoints = []
        self.models = []
        self.examples = []
    
    def analyze_api_endpoints(self) -> List[Dict[str, Any]]:
        """Analyze API endpoints from the BUL system."""
        print("🔍 Analyzing API endpoints...")
        
        # Define endpoint information
        endpoints = [
            {
                'path': '/',
                'method': 'GET',
                'summary': 'System Information',
                'description': 'Get system information and status',
                'response_model': 'SystemInfo',
                'tags': ['System']
            },
            {
                'path': '/health',
                'method': 'GET',
                'summary': 'Health Check',
                'description': 'Check system health status',
                'response_model': 'HealthStatus',
                'tags': ['System']
            },
            {
                'path': '/documents/generate',
                'method': 'POST',
                'summary': 'Generate Document',
                'description': 'Generate a business document based on query',
                'request_model': 'DocumentRequest',
                'response_model': 'DocumentResponse',
                'tags': ['Documents']
            },
            {
                'path': '/tasks/{task_id}/status',
                'method': 'GET',
                'summary': 'Get Task Status',
                'description': 'Get the status of a document generation task',
                'path_parameters': [
                    {
                        'name': 'task_id',
                        'type': 'string',
                        'description': 'Task identifier',
                        'required': True
                    }
                ],
                'response_model': 'TaskStatus',
                'tags': ['Tasks']
            },
            {
                'path': '/tasks',
                'method': 'GET',
                'summary': 'List Tasks',
                'description': 'List all document generation tasks',
                'response_model': 'TaskList',
                'tags': ['Tasks']
            },
            {
                'path': '/tasks/{task_id}',
                'method': 'DELETE',
                'summary': 'Delete Task',
                'description': 'Delete a document generation task',
                'path_parameters': [
                    {
                        'name': 'task_id',
                        'type': 'string',
                        'description': 'Task identifier',
                        'required': True
                    }
                ],
                'response_model': 'DeleteResponse',
                'tags': ['Tasks']
            },
            {
                'path': '/agents',
                'method': 'GET',
                'summary': 'Get Agents',
                'description': 'Get all available business area agents',
                'response_model': 'AgentList',
                'tags': ['Agents']
            },
            {
                'path': '/agents/{area}',
                'method': 'GET',
                'summary': 'Get Agent Info',
                'description': 'Get information about a specific agent',
                'path_parameters': [
                    {
                        'name': 'area',
                        'type': 'string',
                        'description': 'Business area name',
                        'required': True,
                        'enum': ['marketing', 'sales', 'operations', 'hr', 'finance']
                    }
                ],
                'response_model': 'AgentInfo',
                'tags': ['Agents']
            }
        ]
        
        self.endpoints = endpoints
        return endpoints
    
    def analyze_data_models(self) -> List[Dict[str, Any]]:
        """Analyze Pydantic models for documentation."""
        print("📋 Analyzing data models...")
        
        models = [
            {
                'name': 'DocumentRequest',
                'description': 'Request model for document generation',
                'fields': [
                    {
                        'name': 'query',
                        'type': 'string',
                        'description': 'Business query for document generation',
                        'required': True,
                        'example': 'Create a marketing strategy for a new restaurant'
                    },
                    {
                        'name': 'business_area',
                        'type': 'string',
                        'description': 'Specific business area',
                        'required': False,
                        'enum': ['marketing', 'sales', 'operations', 'hr', 'finance'],
                        'example': 'marketing'
                    },
                    {
                        'name': 'document_type',
                        'type': 'string',
                        'description': 'Type of document to generate',
                        'required': False,
                        'example': 'strategy'
                    },
                    {
                        'name': 'priority',
                        'type': 'integer',
                        'description': 'Processing priority (1-5)',
                        'required': False,
                        'minimum': 1,
                        'maximum': 5,
                        'default': 1,
                        'example': 1
                    },
                    {
                        'name': 'metadata',
                        'type': 'object',
                        'description': 'Additional metadata',
                        'required': False,
                        'example': {'client_id': '12345', 'project': 'Q1_2024'}
                    }
                ]
            },
            {
                'name': 'DocumentResponse',
                'description': 'Response model for document generation',
                'fields': [
                    {
                        'name': 'task_id',
                        'type': 'string',
                        'description': 'Unique task identifier',
                        'required': True,
                        'example': 'task_20241216_143022_1'
                    },
                    {
                        'name': 'status',
                        'type': 'string',
                        'description': 'Task status',
                        'required': True,
                        'enum': ['queued', 'processing', 'completed', 'failed'],
                        'example': 'queued'
                    },
                    {
                        'name': 'message',
                        'type': 'string',
                        'description': 'Status message',
                        'required': True,
                        'example': 'Document generation started'
                    },
                    {
                        'name': 'estimated_time',
                        'type': 'integer',
                        'description': 'Estimated processing time in seconds',
                        'required': False,
                        'example': 60
                    }
                ]
            },
            {
                'name': 'TaskStatus',
                'description': 'Task status response model',
                'fields': [
                    {
                        'name': 'task_id',
                        'type': 'string',
                        'description': 'Task identifier',
                        'required': True,
                        'example': 'task_20241216_143022_1'
                    },
                    {
                        'name': 'status',
                        'type': 'string',
                        'description': 'Current task status',
                        'required': True,
                        'enum': ['queued', 'processing', 'completed', 'failed'],
                        'example': 'processing'
                    },
                    {
                        'name': 'progress',
                        'type': 'integer',
                        'description': 'Progress percentage (0-100)',
                        'required': True,
                        'minimum': 0,
                        'maximum': 100,
                        'example': 75
                    },
                    {
                        'name': 'result',
                        'type': 'object',
                        'description': 'Task result (when completed)',
                        'required': False,
                        'example': {
                            'document_id': 'doc_20241216_143022',
                            'title': 'Marketing Strategy Document',
                            'content': 'Generated document content...',
                            'format': 'markdown'
                        }
                    },
                    {
                        'name': 'error',
                        'type': 'string',
                        'description': 'Error message (when failed)',
                        'required': False,
                        'example': 'Processing timeout'
                    }
                ]
            }
        ]
        
        self.models = models
        return models
    
    def generate_examples(self) -> List[Dict[str, Any]]:
        """Generate API usage examples."""
        print("📝 Generating API examples...")
        
        examples = [
            {
                'title': 'Generate Marketing Strategy',
                'description': 'Create a marketing strategy document for a new restaurant',
                'endpoint': 'POST /documents/generate',
                'request': {
                    'query': 'Create a comprehensive marketing strategy for a new Italian restaurant',
                    'business_area': 'marketing',
                    'document_type': 'strategy',
                    'priority': 1,
                    'metadata': {
                        'client': 'Restaurant Group',
                        'project': 'Q1_2024_Launch'
                    }
                },
                'response': {
                    'task_id': 'task_20241216_143022_1',
                    'status': 'queued',
                    'message': 'Document generation started',
                    'estimated_time': 60
                }
            },
            {
                'title': 'Check Task Status',
                'description': 'Check the status of a document generation task',
                'endpoint': 'GET /tasks/{task_id}/status',
                'request': {
                    'task_id': 'task_20241216_143022_1'
                },
                'response': {
                    'task_id': 'task_20241216_143022_1',
                    'status': 'completed',
                    'progress': 100,
                    'result': {
                        'document_id': 'doc_20241216_143022',
                        'title': 'Marketing Strategy for Italian Restaurant',
                        'content': '# Marketing Strategy for Italian Restaurant\n\n## Executive Summary\n...',
                        'format': 'markdown',
                        'business_area': 'marketing',
                        'document_type': 'strategy',
                        'created_at': '2024-12-16T14:30:22Z'
                    }
                }
            },
            {
                'title': 'List All Tasks',
                'description': 'Get a list of all document generation tasks',
                'endpoint': 'GET /tasks',
                'request': {},
                'response': {
                    'tasks': [
                        {
                            'task_id': 'task_20241216_143022_1',
                            'status': 'completed',
                            'progress': 100,
                            'created_at': '2024-12-16T14:30:22Z'
                        },
                        {
                            'task_id': 'task_20241216_143015_2',
                            'status': 'processing',
                            'progress': 75,
                            'created_at': '2024-12-16T14:30:15Z'
                        }
                    ]
                }
            },
            {
                'title': 'Get Available Agents',
                'description': 'Get information about all available business area agents',
                'endpoint': 'GET /agents',
                'request': {},
                'response': {
                    'agents': {
                        'marketing': {
                            'area': 'marketing',
                            'supported_document_types': ['strategy', 'campaign', 'content', 'analysis'],
                            'priority': 1
                        },
                        'sales': {
                            'area': 'sales',
                            'supported_document_types': ['proposal', 'presentation', 'playbook', 'forecast'],
                            'priority': 1
                        }
                    }
                }
            }
        ]
        
        self.examples = examples
        return examples
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI 3.0 specification."""
        print("📋 Generating OpenAPI specification...")
        
        openapi_spec = {
            'openapi': '3.0.0',
            'info': {
                'title': self.api_info['title'],
                'version': self.api_info['version'],
                'description': self.api_info['description'],
                'contact': self.api_info['contact']
            },
            'servers': [
                {
                    'url': self.api_info['base_url'],
                    'description': 'BUL API Server'
                }
            ],
            'paths': {},
            'components': {
                'schemas': {},
                'examples': {}
            },
            'tags': [
                {'name': 'System', 'description': 'System information and health'},
                {'name': 'Documents', 'description': 'Document generation'},
                {'name': 'Tasks', 'description': 'Task management'},
                {'name': 'Agents', 'description': 'Business area agents'}
            ]
        }
        
        # Add paths
        for endpoint in self.endpoints:
            path_spec = {
                endpoint['method'].lower(): {
                    'summary': endpoint['summary'],
                    'description': endpoint['description'],
                    'tags': endpoint['tags']
                }
            }
            
            # Add parameters
            if 'path_parameters' in endpoint:
                path_spec[endpoint['method'].lower()]['parameters'] = endpoint['path_parameters']
            
            # Add request body
            if 'request_model' in endpoint:
                path_spec[endpoint['method'].lower()]['requestBody'] = {
                    'required': True,
                    'content': {
                        'application/json': {
                            'schema': {'$ref': f"#/components/schemas/{endpoint['request_model']}"}
                        }
                    }
                }
            
            # Add responses
            path_spec[endpoint['method'].lower()]['responses'] = {
                '200': {
                    'description': 'Successful response',
                    'content': {
                        'application/json': {
                            'schema': {'$ref': f"#/components/schemas/{endpoint['response_model']}"}
                        }
                    }
                },
                '404': {
                    'description': 'Not found'
                },
                '500': {
                    'description': 'Internal server error'
                }
            }
            
            openapi_spec['paths'][endpoint['path']] = path_spec
        
        # Add schemas
        for model in self.models:
            schema = {
                'type': 'object',
                'properties': {},
                'required': []
            }
            
            for field in model['fields']:
                field_schema = {'type': field['type']}
                
                if 'description' in field:
                    field_schema['description'] = field['description']
                
                if 'example' in field:
                    field_schema['example'] = field['example']
                
                if 'enum' in field:
                    field_schema['enum'] = field['enum']
                
                if 'minimum' in field:
                    field_schema['minimum'] = field['minimum']
                
                if 'maximum' in field:
                    field_schema['maximum'] = field['maximum']
                
                if 'default' in field:
                    field_schema['default'] = field['default']
                
                schema['properties'][field['name']] = field_schema
                
                if field.get('required', False):
                    schema['required'].append(field['name'])
            
            openapi_spec['components']['schemas'][model['name']] = schema
        
        return openapi_spec
    
    def generate_markdown_docs(self) -> str:
        """Generate Markdown documentation."""
        print("📝 Generating Markdown documentation...")
        
        docs = f"""# {self.api_info['title']}

**Version:** {self.api_info['version']}  
**Base URL:** {self.api_info['base_url']}  
**Description:** {self.api_info['description']}

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Data Models](#data-models)
- [Examples](#examples)
- [Error Handling](#error-handling)

## Overview

The BUL API provides endpoints for generating business documents using AI-powered analysis and specialized business area agents.

### Key Features

- **Document Generation**: Create professional business documents
- **Task Management**: Track document generation progress
- **Business Area Specialization**: Dedicated agents for different business areas
- **Real-time Processing**: Asynchronous task processing
- **Comprehensive API**: RESTful API with full documentation

## Authentication

Currently, the API does not require authentication. In production environments, consider implementing API key authentication.

## Endpoints

"""
        
        # Group endpoints by tag
        endpoints_by_tag = {}
        for endpoint in self.endpoints:
            for tag in endpoint['tags']:
                if tag not in endpoints_by_tag:
                    endpoints_by_tag[tag] = []
                endpoints_by_tag[tag].append(endpoint)
        
        # Add endpoint documentation
        for tag, tag_endpoints in endpoints_by_tag.items():
            docs += f"### {tag}\n\n"
            
            for endpoint in tag_endpoints:
                docs += f"#### {endpoint['method']} {endpoint['path']}\n\n"
                docs += f"**{endpoint['summary']}**\n\n"
                docs += f"{endpoint['description']}\n\n"
                
                if 'path_parameters' in endpoint:
                    docs += "**Path Parameters:**\n\n"
                    for param in endpoint['path_parameters']:
                        docs += f"- `{param['name']}` ({param['type']}): {param['description']}\n"
                    docs += "\n"
                
                if 'request_model' in endpoint:
                    docs += f"**Request Body:** `{endpoint['request_model']}`\n\n"
                
                docs += f"**Response:** `{endpoint['response_model']}`\n\n"
                docs += "---\n\n"
        
        # Add data models
        docs += "## Data Models\n\n"
        for model in self.models:
            docs += f"### {model['name']}\n\n"
            docs += f"{model['description']}\n\n"
            docs += "| Field | Type | Required | Description |\n"
            docs += "|-------|------|----------|-------------|\n"
            
            for field in model['fields']:
                required = "Yes" if field.get('required', False) else "No"
                docs += f"| `{field['name']}` | {field['type']} | {required} | {field['description']} |\n"
            
            docs += "\n"
        
        # Add examples
        docs += "## Examples\n\n"
        for example in self.examples:
            docs += f"### {example['title']}\n\n"
            docs += f"{example['description']}\n\n"
            docs += f"**Endpoint:** `{example['endpoint']}`\n\n"
            
            if example['request']:
                docs += "**Request:**\n```json\n"
                docs += json.dumps(example['request'], indent=2)
                docs += "\n```\n\n"
            
            docs += "**Response:**\n```json\n"
            docs += json.dumps(example['response'], indent=2)
            docs += "\n```\n\n"
        
        # Add error handling
        docs += """## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses include a JSON object with error details:

```json
{
  "detail": "Error message description"
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse. Default limits:
- 100 requests per minute per IP address
- 5 concurrent document generation tasks

## Support

For API support and questions:
- Check the interactive API documentation at `/docs`
- Review system health at `/health`
- Contact support at support@bul-system.com
"""
        
        return docs
    
    def save_documentation(self, output_dir: str = "docs") -> Dict[str, str]:
        """Save all documentation files."""
        print("💾 Saving documentation files...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        try:
            # Save OpenAPI specification
            openapi_spec = self.generate_openapi_spec()
            openapi_file = output_path / "openapi.json"
            with open(openapi_file, 'w') as f:
                json.dump(openapi_spec, f, indent=2)
            saved_files['openapi'] = str(openapi_file)
            
            # Save Markdown documentation
            markdown_docs = self.generate_markdown_docs()
            markdown_file = output_path / "API_Documentation.md"
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(markdown_docs)
            saved_files['markdown'] = str(markdown_file)
            
            # Save examples
            examples_file = output_path / "api_examples.json"
            with open(examples_file, 'w') as f:
                json.dump(self.examples, f, indent=2)
            saved_files['examples'] = str(examples_file)
            
            print("✅ Documentation files saved successfully")
            for file_type, file_path in saved_files.items():
                print(f"   {file_type.title()}: {file_path}")
            
            return saved_files
            
        except Exception as e:
            print(f"❌ Error saving documentation: {e}")
            return {}

def main():
    """Main documentation generator function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL API Documentation Generator")
    parser.add_argument("--output", default="docs", help="Output directory for documentation")
    parser.add_argument("--format", choices=['all', 'openapi', 'markdown', 'examples'],
                       default='all', help="Documentation format to generate")
    parser.add_argument("--analyze", action="store_true", help="Analyze API without generating files")
    
    args = parser.parse_args()
    
    generator = APIDocumentationGenerator()
    
    print("📚 BUL API Documentation Generator")
    print("=" * 50)
    
    # Analyze API
    endpoints = generator.analyze_api_endpoints()
    models = generator.analyze_data_models()
    examples = generator.generate_examples()
    
    print(f"✅ Analyzed {len(endpoints)} endpoints")
    print(f"✅ Analyzed {len(models)} data models")
    print(f"✅ Generated {len(examples)} examples")
    
    if args.analyze:
        print("\n📊 Analysis Summary:")
        print(f"   Endpoints: {len(endpoints)}")
        print(f"   Models: {len(models)}")
        print(f"   Examples: {len(examples)}")
        return 0
    
    # Generate documentation
    if args.format in ['all', 'openapi']:
        openapi_spec = generator.generate_openapi_spec()
        print(f"✅ Generated OpenAPI specification")
    
    if args.format in ['all', 'markdown']:
        markdown_docs = generator.generate_markdown_docs()
        print(f"✅ Generated Markdown documentation")
    
    # Save files
    saved_files = generator.save_documentation(args.output)
    
    if saved_files:
        print(f"\n📄 Documentation saved to: {args.output}")
        print("💡 Access interactive docs at: http://localhost:8000/docs")
    else:
        print("❌ Failed to save documentation files")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

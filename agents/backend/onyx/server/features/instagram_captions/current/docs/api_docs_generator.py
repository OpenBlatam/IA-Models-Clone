"""
API Documentation Generator for Instagram Captions API v10.0

Automatic generation of comprehensive API documentation.
"""

import inspect
import ast
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class EndpointInfo:
    """Information about an API endpoint."""
    path: str
    method: str
    function_name: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    responses: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    security: List[str] = field(default_factory=list)

@dataclass
class APIDocumentation:
    """Complete API documentation."""
    title: str
    version: str
    description: str
    base_url: str
    endpoints: List[EndpointInfo] = field(default_factory=list)
    models: Dict[str, Any] = field(default_factory=dict)
    security_schemes: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

class APIDocsGenerator:
    """Automatic API documentation generator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.api_info = {
            'title': 'Instagram Captions API v10.0',
            'version': '10.0.0',
            'description': 'Advanced Instagram Captions Generation API with Enterprise Features',
            'base_url': 'http://localhost:8000'
        }
        self.endpoints: List[EndpointInfo] = []
        self.models: Dict[str, Any] = {}
        self.security_schemes = {
            'api_key': {
                'type': 'apiKey',
                'in': 'header',
                'name': 'X-API-Key'
            },
            'bearer': {
                'type': 'http',
                'scheme': 'bearer',
                'bearerFormat': 'JWT'
            }
        }
    
    def scan_fastapi_app(self, app_file_path: str) -> List[EndpointInfo]:
        """Scan FastAPI application file for endpoints."""
        try:
            with open(app_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file content
            tree = ast.parse(content)
            
            # Find FastAPI route decorators
            endpoints = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    endpoint_info = self._extract_endpoint_info(node, content)
                    if endpoint_info:
                        endpoints.append(endpoint_info)
            
            self.endpoints.extend(endpoints)
            return endpoints
            
        except Exception as e:
            print(f"Error scanning FastAPI app: {e}")
            return []
    
    def _extract_endpoint_info(self, func_node: ast.FunctionDef, content: str) -> Optional[EndpointInfo]:
        """Extract endpoint information from a function definition."""
        # Look for route decorators
        route_info = self._find_route_decorator(func_node, content)
        if not route_info:
            return None
        
        # Extract function documentation
        docstring = ast.get_docstring(func_node) or ""
        
        # Parse parameters
        parameters = self._extract_parameters(func_node)
        
        # Parse responses
        responses = self._extract_responses(func_node, content)
        
        # Extract tags from decorators
        tags = self._extract_tags(func_node, content)
        
        # Check for security requirements
        security = self._extract_security(func_node, content)
        
        return EndpointInfo(
            path=route_info['path'],
            method=route_info['method'],
            function_name=func_node.name,
            description=docstring.strip(),
            parameters=parameters,
            responses=responses,
            tags=tags,
            security=security
        )
    
    def _find_route_decorator(self, func_node: ast.FunctionDef, content: str) -> Optional[Dict[str, str]]:
        """Find route decorator information."""
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    # Handle cases like app.get("/path")
                    if decorator.func.attr in ['get', 'post', 'put', 'delete', 'patch']:
                        method = decorator.func.attr.upper()
                        if decorator.args:
                            path = self._extract_string_literal(decorator.args[0])
                            if path:
                                return {'path': path, 'method': method}
        
        return None
    
    def _extract_string_literal(self, node: ast.AST) -> Optional[str]:
        """Extract string literal from AST node."""
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Str):  # Python < 3.8
            return node.s
        return None
    
    def _extract_parameters(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function parameters."""
        parameters = []
        
        for arg in func_node.args.args:
            if arg.arg != 'self':
                param_info = {
                    'name': arg.arg,
                    'type': 'string',  # Default type
                    'required': True,
                    'description': ''
                }
                
                # Try to extract type annotation
                if arg.annotation:
                    param_info['type'] = self._extract_type_annotation(arg.annotation)
                
                parameters.append(param_info)
        
        return parameters
    
    def _extract_type_annotation(self, annotation: ast.AST) -> str:
        """Extract type information from annotation."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Str):  # Python < 3.8
            return annotation.s
        return 'string'
    
    def _extract_responses(self, func_node: ast.FunctionDef, content: str) -> List[Dict[str, Any]]:
        """Extract response information."""
        responses = []
        
        # Look for common response patterns
        response_patterns = [
            r'HTTPException\(status_code=(\d+)\)',
            r'status\.HTTP_(\d+)',
            r'@responses\((\d+)'
        ]
        
        for pattern in response_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                status_code = int(match)
                responses.append({
                    'status_code': status_code,
                    'description': self._get_status_description(status_code)
                })
        
        # Add default responses if none found
        if not responses:
            responses = [
                {'status_code': 200, 'description': 'Success'},
                {'status_code': 400, 'description': 'Bad Request'},
                {'status_code': 500, 'description': 'Internal Server Error'}
            ]
        
        return responses
    
    def _get_status_description(self, status_code: int) -> str:
        """Get description for HTTP status code."""
        descriptions = {
            200: 'Success',
            201: 'Created',
            400: 'Bad Request',
            401: 'Unauthorized',
            403: 'Forbidden',
            404: 'Not Found',
            422: 'Validation Error',
            500: 'Internal Server Error',
            503: 'Service Unavailable'
        }
        return descriptions.get(status_code, f'Status {status_code}')
    
    def _extract_tags(self, func_node: ast.FunctionDef, content: str) -> List[str]:
        """Extract tags from function decorators."""
        tags = []
        
        # Look for @tags decorator
        tag_pattern = r'@tags\(\[([^\]]+)\]\)'
        matches = re.findall(tag_pattern, content)
        
        for match in matches:
            # Parse tag list
            tag_list = [tag.strip().strip('"\'') for tag in match.split(',')]
            tags.extend(tag_list)
        
        # Add default tags based on function name
        if not tags:
            if 'auth' in func_node.name.lower():
                tags.append('Authentication')
            elif 'user' in func_node.name.lower():
                tags.append('Users')
            elif 'caption' in func_node.name.lower():
                tags.append('Captions')
            else:
                tags.append('General')
        
        return tags
    
    def _extract_security(self, func_node: ast.FunctionDef, content: str) -> List[str]:
        """Extract security requirements."""
        security = []
        
        # Look for security decorators
        if '@requires_auth' in content or '@secure' in content:
            security.append('api_key')
        
        if '@jwt_required' in content or '@bearer_token' in content:
            security.append('bearer')
        
        return security
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI specification."""
        openapi_spec = {
            'openapi': '3.0.0',
            'info': {
                'title': self.api_info['title'],
                'version': self.api_info['version'],
                'description': self.api_info['description']
            },
            'servers': [
                {
                    'url': self.api_info['base_url'],
                    'description': 'Development server'
                }
            ],
            'paths': {},
            'components': {
                'securitySchemes': self.security_schemes,
                'schemas': self.models
            },
            'tags': self._generate_tags()
        }
        
        # Generate paths
        for endpoint in self.endpoints:
            path_item = openapi_spec['paths'].setdefault(endpoint.path, {})
            
            operation = {
                'summary': endpoint.function_name.replace('_', ' ').title(),
                'description': endpoint.description,
                'tags': endpoint.tags,
                'parameters': self._generate_parameters(endpoint),
                'responses': self._generate_responses(endpoint),
                'security': self._generate_security(endpoint)
            }
            
            path_item[endpoint.method.lower()] = operation
        
        return openapi_spec
    
    def _generate_tags(self) -> List[Dict[str, str]]:
        """Generate tag definitions."""
        all_tags = set()
        for endpoint in self.endpoints:
            all_tags.update(endpoint.tags)
        
        return [{'name': tag, 'description': f'{tag} operations'} for tag in sorted(all_tags)]
    
    def _generate_parameters(self, endpoint: EndpointInfo) -> List[Dict[str, Any]]:
        """Generate OpenAPI parameters."""
        parameters = []
        
        for param in endpoint.parameters:
            param_spec = {
                'name': param['name'],
                'in': 'query',  # Default to query parameter
                'required': param['required'],
                'schema': {
                    'type': param['type']
                }
            }
            
            if param['description']:
                param_spec['description'] = param['description']
            
            parameters.append(param_spec)
        
        return parameters
    
    def _generate_responses(self, endpoint: EndpointInfo) -> Dict[str, Any]:
        """Generate OpenAPI responses."""
        responses = {}
        
        for response in endpoint.responses:
            responses[str(response['status_code'])] = {
                'description': response['description']
            }
        
        return responses
    
    def _generate_security(self, endpoint: EndpointInfo) -> List[Dict[str, List[str]]]:
        """Generate OpenAPI security requirements."""
        if not endpoint.security:
            return []
        
        security_requirements = []
        for scheme in endpoint.security:
            security_requirements.append({scheme: []})
        
        return security_requirements
    
    def generate_markdown_docs(self) -> str:
        """Generate Markdown documentation."""
        md_content = f"""# {self.api_info['title']} Documentation

**Version:** {self.api_info['version']}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

{self.api_info['description']}

## Base URL

```
{self.api_info['base_url']}
```

## Authentication

### API Key Authentication
Include your API key in the request header:
```
X-API-Key: your_api_key_here
```

### Bearer Token Authentication
Include your JWT token in the Authorization header:
```
Authorization: Bearer your_jwt_token_here
```

## Endpoints

"""
        
        # Group endpoints by tags
        endpoints_by_tag = {}
        for endpoint in self.endpoints:
            for tag in endpoint.tags:
                if tag not in endpoints_by_tag:
                    endpoints_by_tag[tag] = []
                endpoints_by_tag[tag].append(endpoint)
        
        # Generate documentation for each tag
        for tag, tag_endpoints in sorted(endpoints_by_tag.items()):
            md_content += f"### {tag}\n\n"
            
            for endpoint in tag_endpoints:
                md_content += self._generate_endpoint_markdown(endpoint)
                md_content += "\n"
        
        return md_content
    
    def _generate_endpoint_markdown(self, endpoint: EndpointInfo) -> str:
        """Generate Markdown for a single endpoint."""
        md = f"#### {endpoint.method} {endpoint.path}\n\n"
        
        if endpoint.description:
            md += f"{endpoint.description}\n\n"
        
        # Parameters
        if endpoint.parameters:
            md += "**Parameters:**\n\n"
            md += "| Name | Type | Required | Description |\n"
            md += "|------|------|----------|-------------|\n"
            
            for param in endpoint.parameters:
                md += f"| {param['name']} | {param['type']} | {'Yes' if param['required'] else 'No'} | {param['description']} |\n"
            
            md += "\n"
        
        # Responses
        if endpoint.responses:
            md += "**Responses:**\n\n"
            md += "| Status Code | Description |\n"
            md += "|-------------|-------------|\n"
            
            for response in endpoint.responses:
                md += f"| {response['status_code']} | {response['description']} |\n"
            
            md += "\n"
        
        # Security
        if endpoint.security:
            md += "**Security:**\n"
            for scheme in endpoint.security:
                md += f"- {scheme}\n"
            md += "\n"
        
        return md
    
    def save_documentation(self, output_dir: str, formats: List[str] = None):
        """Save documentation in multiple formats."""
        if formats is None:
            formats = ['json', 'yaml', 'markdown']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for format_type in formats:
            try:
                if format_type == 'json':
                    import json
                    openapi_spec = self.generate_openapi_spec()
                    json_file = output_path / 'openapi.json'
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(openapi_spec, f, indent=2)
                    saved_files.append(str(json_file))
                
                elif format_type == 'yaml':
                    import yaml
                    openapi_spec = self.generate_openapi_spec()
                    yaml_file = output_path / 'openapi.yaml'
                    with open(yaml_file, 'w', encoding='utf-8') as f:
                        yaml.dump(openapi_spec, f, default_flow_style=False, indent=2)
                    saved_files.append(str(yaml_file))
                
                elif format_type == 'markdown':
                    md_content = self.generate_markdown_docs()
                    md_file = output_path / 'README.md'
                    with open(md_file, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                    saved_files.append(str(md_file))
                
            except Exception as e:
                print(f"Error saving {format_type} documentation: {e}")
        
        print(f"Documentation saved to: {output_dir}")
        print(f"Files created: {', '.join(saved_files)}")
        
        return saved_files







"""
Interactive Documentation for Flux2 Clothing Changer
====================================================

Interactive documentation and API explorer.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class APIEndpoint:
    """API endpoint documentation."""
    path: str
    method: str
    description: str
    parameters: Dict[str, Any] = None
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Any] = None
    examples: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.responses is None:
            self.responses = {}
        if self.examples is None:
            self.examples = []


@dataclass
class CodeExample:
    """Code example."""
    language: str
    title: str
    code: str
    description: Optional[str] = None


class InteractiveDocs:
    """Interactive documentation system."""
    
    def __init__(
        self,
        docs_dir: Path = Path("docs"),
    ):
        """
        Initialize interactive docs.
        
        Args:
            docs_dir: Documentation directory
        """
        self.docs_dir = docs_dir
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.examples: List[CodeExample] = []
        self.tutorials: List[Dict[str, Any]] = []
    
    def register_endpoint(
        self,
        path: str,
        method: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        request_body: Optional[Dict[str, Any]] = None,
        responses: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
    ) -> APIEndpoint:
        """
        Register API endpoint.
        
        Args:
            path: Endpoint path
            method: HTTP method
            description: Endpoint description
            parameters: Optional parameters
            request_body: Optional request body
            responses: Optional responses
            examples: Optional examples
            
        Returns:
            Created endpoint
        """
        endpoint = APIEndpoint(
            path=path,
            method=method,
            description=description,
            parameters=parameters or {},
            request_body=request_body,
            responses=responses or {},
            examples=examples or [],
        )
        
        key = f"{method}:{path}"
        self.endpoints[key] = endpoint
        
        logger.info(f"Registered endpoint: {key}")
        return endpoint
    
    def add_example(
        self,
        language: str,
        title: str,
        code: str,
        description: Optional[str] = None,
    ) -> CodeExample:
        """
        Add code example.
        
        Args:
            language: Programming language
            title: Example title
            code: Example code
            description: Optional description
            
        Returns:
            Created example
        """
        example = CodeExample(
            language=language,
            title=title,
            code=code,
            description=description,
        )
        
        self.examples.append(example)
        logger.info(f"Added example: {title}")
        return example
    
    def add_tutorial(
        self,
        title: str,
        content: str,
        steps: List[Dict[str, Any]],
    ) -> None:
        """
        Add tutorial.
        
        Args:
            title: Tutorial title
            content: Tutorial content
            steps: Tutorial steps
        """
        tutorial = {
            "title": title,
            "content": content,
            "steps": steps,
        }
        
        self.tutorials.append(tutorial)
        logger.info(f"Added tutorial: {title}")
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI specification."""
        paths = {}
        
        for key, endpoint in self.endpoints.items():
            method, path = key.split(":", 1)
            
            if path not in paths:
                paths[path] = {}
            
            paths[path][method.lower()] = {
                "summary": endpoint.description,
                "parameters": [
                    {
                        "name": name,
                        "in": param.get("in", "query"),
                        "schema": param.get("schema", {}),
                        "required": param.get("required", False),
                        "description": param.get("description", ""),
                    }
                    for name, param in endpoint.parameters.items()
                ],
                "requestBody": endpoint.request_body,
                "responses": endpoint.responses,
            }
        
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Flux2 Clothing Changer API",
                "version": "1.0.0",
            },
            "paths": paths,
        }
    
    def export_docs(self, format: str = "json") -> str:
        """
        Export documentation.
        
        Args:
            format: Export format (json, markdown)
            
        Returns:
            Exported documentation
        """
        if format == "json":
            return json.dumps({
                "endpoints": {
                    key: {
                        "path": endpoint.path,
                        "method": endpoint.method,
                        "description": endpoint.description,
                        "parameters": endpoint.parameters,
                        "request_body": endpoint.request_body,
                        "responses": endpoint.responses,
                        "examples": endpoint.examples,
                    }
                    for key, endpoint in self.endpoints.items()
                },
                "examples": [
                    {
                        "language": ex.language,
                        "title": ex.title,
                        "code": ex.code,
                        "description": ex.description,
                    }
                    for ex in self.examples
                ],
                "tutorials": self.tutorials,
            }, indent=2)
        elif format == "markdown":
            return self._generate_markdown()
        else:
            return ""
    
    def _generate_markdown(self) -> str:
        """Generate markdown documentation."""
        md = "# Flux2 Clothing Changer API Documentation\n\n"
        
        # Endpoints
        md += "## API Endpoints\n\n"
        for key, endpoint in self.endpoints.items():
            md += f"### {endpoint.method} {endpoint.path}\n\n"
            md += f"{endpoint.description}\n\n"
            
            if endpoint.parameters:
                md += "#### Parameters\n\n"
                for name, param in endpoint.parameters.items():
                    md += f"- `{name}`: {param.get('description', '')}\n"
                md += "\n"
            
            if endpoint.examples:
                md += "#### Examples\n\n"
                for example in endpoint.examples:
                    md += f"```json\n{json.dumps(example, indent=2)}\n```\n\n"
        
        # Code Examples
        if self.examples:
            md += "## Code Examples\n\n"
            for ex in self.examples:
                md += f"### {ex.title}\n\n"
                if ex.description:
                    md += f"{ex.description}\n\n"
                md += f"```{ex.language}\n{ex.code}\n```\n\n"
        
        # Tutorials
        if self.tutorials:
            md += "## Tutorials\n\n"
            for tutorial in self.tutorials:
                md += f"### {tutorial['title']}\n\n"
                md += f"{tutorial['content']}\n\n"
                for i, step in enumerate(tutorial['steps'], 1):
                    md += f"#### Step {i}: {step.get('title', '')}\n\n"
                    md += f"{step.get('content', '')}\n\n"
        
        return md
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get documentation statistics."""
        return {
            "total_endpoints": len(self.endpoints),
            "total_examples": len(self.examples),
            "total_tutorials": len(self.tutorials),
        }



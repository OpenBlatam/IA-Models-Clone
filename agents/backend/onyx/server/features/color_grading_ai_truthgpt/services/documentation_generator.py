"""
Documentation Generator for Color Grading AI
=============================================

Automatic documentation generation from code and services.
"""

import logging
import inspect
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ServiceDocumentation:
    """Service documentation."""
    name: str
    description: str
    methods: List[Dict[str, Any]] = field(default_factory=list)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    examples: List[str] = field(default_factory=list)


@dataclass
class APIDocumentation:
    """API documentation."""
    endpoint: str
    method: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    responses: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


class DocumentationGenerator:
    """
    Documentation generator.
    
    Features:
    - Service documentation
    - API documentation
    - Code documentation
    - Markdown generation
    - HTML generation
    - Auto-discovery
    """
    
    def __init__(self, output_dir: str = "docs"):
        """
        Initialize documentation generator.
        
        Args:
            output_dir: Output directory for documentation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._services: Dict[str, Any] = {}
    
    def register_service(self, name: str, service: Any):
        """
        Register service for documentation.
        
        Args:
            name: Service name
            service: Service instance
        """
        self._services[name] = service
        logger.info(f"Registered service for documentation: {name}")
    
    def generate_service_docs(self, service_name: str) -> ServiceDocumentation:
        """
        Generate documentation for a service.
        
        Args:
            service_name: Service name
            
        Returns:
            Service documentation
        """
        service = self._services.get(service_name)
        if not service:
            raise ValueError(f"Service not found: {service_name}")
        
        doc = inspect.getdoc(service) or ""
        methods = []
        
        # Extract methods
        for name, method in inspect.getmembers(service, predicate=inspect.ismethod):
            if name.startswith("_"):
                continue
            
            method_doc = inspect.getdoc(method) or ""
            sig = inspect.signature(method)
            params = []
            
            for param_name, param in sig.parameters.items():
                params.append({
                    "name": param_name,
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                    "default": str(param.default) if param.default != inspect.Parameter.empty else None,
                    "required": param.default == inspect.Parameter.empty
                })
            
            methods.append({
                "name": name,
                "description": method_doc,
                "parameters": params,
                "return_type": str(sig.return_annotation) if sig.return_annotation != inspect.Parameter.empty else "Any"
            })
        
        return ServiceDocumentation(
            name=service_name,
            description=doc,
            methods=methods
        )
    
    def generate_markdown(
        self,
        service_docs: List[ServiceDocumentation],
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate markdown documentation.
        
        Args:
            service_docs: List of service documentation
            output_file: Optional output file path
            
        Returns:
            Markdown content
        """
        lines = [
            "# Service Documentation",
            "",
            f"Generated on: {datetime.now().isoformat()}",
            "",
            "## Services",
            ""
        ]
        
        for doc in service_docs:
            lines.extend([
                f"### {doc.name}",
                "",
                doc.description,
                "",
                "#### Methods",
                ""
            ])
            
            for method in doc.methods:
                lines.extend([
                    f"##### {method['name']}",
                    "",
                    method.get("description", "No description"),
                    "",
                    "**Parameters:**",
                    ""
                ])
                
                for param in method.get("parameters", []):
                    req = "required" if param.get("required") else "optional"
                    default = f" (default: {param.get('default')})" if param.get("default") else ""
                    lines.append(f"- `{param['name']}` ({param['type']}): {req}{default}")
                
                lines.extend([
                    "",
                    f"**Returns:** `{method.get('return_type', 'Any')}`",
                    "",
                    "---",
                    ""
                ])
        
        content = "\n".join(lines)
        
        if output_file:
            output_path = self.output_dir / output_file
            output_path.write_text(content, encoding="utf-8")
            logger.info(f"Generated markdown documentation: {output_path}")
        
        return content
    
    def generate_api_docs(
        self,
        endpoints: List[APIDocumentation],
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate API documentation.
        
        Args:
            endpoints: List of API endpoints
            output_file: Optional output file path
            
        Returns:
            Markdown content
        """
        lines = [
            "# API Documentation",
            "",
            f"Generated on: {datetime.now().isoformat()}",
            "",
            "## Endpoints",
            ""
        ]
        
        for endpoint in endpoints:
            lines.extend([
                f"### {endpoint.method} {endpoint.endpoint}",
                "",
                endpoint.description,
                "",
                "**Parameters:**",
                ""
            ])
            
            for param in endpoint.parameters:
                req = "required" if param.get("required") else "optional"
                lines.append(f"- `{param['name']}` ({param.get('type', 'Any')}): {req} - {param.get('description', '')}")
            
            lines.extend([
                "",
                "**Responses:**",
                ""
            ])
            
            for response in endpoint.responses:
                lines.append(f"- `{response.get('status_code', 200)}`: {response.get('description', '')}")
            
            if endpoint.examples:
                lines.extend([
                    "",
                    "**Examples:**",
                    ""
                ])
                for example in endpoint.examples:
                    lines.append(f"```\n{example}\n```")
            
            lines.extend(["", "---", ""])
        
        content = "\n".join(lines)
        
        if output_file:
            output_path = self.output_dir / output_file
            output_path.write_text(content, encoding="utf-8")
            logger.info(f"Generated API documentation: {output_path}")
        
        return content
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get documentation statistics."""
        return {
            "registered_services": len(self._services),
            "output_directory": str(self.output_dir),
        }



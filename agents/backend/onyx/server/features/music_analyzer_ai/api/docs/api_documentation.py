"""
API documentation utilities
"""

from typing import Dict, Any, List


def generate_router_docs(router_name: str, endpoints: List[Dict[str, Any]]) -> str:
    """Generate documentation for a router"""
    docs = f"# {router_name} Router\n\n"
    docs += "## Endpoints\n\n"
    
    for endpoint in endpoints:
        docs += f"### {endpoint.get('method', 'GET')} {endpoint.get('path', '')}\n\n"
        docs += f"{endpoint.get('description', 'No description')}\n\n"
        
        if endpoint.get('parameters'):
            docs += "**Parameters:**\n\n"
            for param in endpoint['parameters']:
                docs += f"- `{param.get('name')}`: {param.get('description', '')}\n"
            docs += "\n"
        
        if endpoint.get('responses'):
            docs += "**Responses:**\n\n"
            for status, response in endpoint['responses'].items():
                docs += f"- `{status}`: {response.get('description', '')}\n"
            docs += "\n"
    
    return docs


def generate_api_summary(routers: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate API summary documentation"""
    summary = "# Music Analyzer AI - API Documentation\n\n"
    summary += f"Total Routers: {len(routers)}\n\n"
    
    for router_name, endpoints in routers.items():
        summary += f"## {router_name}\n"
        summary += f"Endpoints: {len(endpoints)}\n\n"
    
    return summary


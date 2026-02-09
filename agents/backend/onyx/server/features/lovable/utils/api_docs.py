"""
API documentation utilities.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


def generate_endpoint_doc(
    method: str,
    path: str,
    summary: str,
    description: Optional[str] = None,
    parameters: Optional[List[Dict[str, Any]]] = None,
    responses: Optional[Dict[int, Dict[str, Any]]] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate OpenAPI documentation for an endpoint.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        path: Endpoint path
        summary: Short summary
        description: Detailed description
        parameters: List of parameter definitions
        responses: Dictionary of status code -> response definition
        tags: List of tags
        
    Returns:
        OpenAPI endpoint documentation
    """
    doc = {
        "method": method.upper(),
        "path": path,
        "summary": summary,
        "tags": tags or []
    }
    
    if description:
        doc["description"] = description
    
    if parameters:
        doc["parameters"] = parameters
    
    if responses:
        doc["responses"] = responses
    
    return doc


def generate_schema_doc(
    name: str,
    fields: Dict[str, Dict[str, Any]],
    required: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate schema documentation.
    
    Args:
        name: Schema name
        fields: Dictionary of field_name -> field_definition
        required: List of required field names
        
    Returns:
        Schema documentation
    """
    return {
        "type": "object",
        "properties": fields,
        "required": required or []
    }


def generate_example_response(
    data: Any,
    status_code: int = 200,
    message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate example API response.
    
    Args:
        data: Response data
        status_code: HTTP status code
        message: Optional message
        
    Returns:
        Example response dictionary
    """
    response = {
        "status_code": status_code,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    
    if message:
        response["message"] = message
    
    return response






    return response


def generate_endpoint_docs(endpoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate documentation for multiple endpoints.
    
    Args:
        endpoints: List of endpoint definitions
        
    Returns:
        List of endpoint documentation
    """
    return [
        generate_endpoint_doc(
            method=ep.get("method", "GET"),
            path=ep.get("path", "/"),
            summary=ep.get("summary", ""),
            description=ep.get("description"),
            parameters=ep.get("parameters"),
            responses=ep.get("responses"),
            tags=ep.get("tags")
        )
        for ep in endpoints
    ]

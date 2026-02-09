"""
API documentation helpers
Utilities for generating consistent API documentation
"""

from typing import Dict, Any, List, Optional


def create_endpoint_summary(
    summary: str,
    description: Optional[str] = None
) -> Dict[str, str]:
    """
    Create endpoint summary and description
    
    Args:
        summary: Short summary
        description: Optional detailed description
    
    Returns:
        Dictionary with summary and description
    """
    result = {"summary": summary}
    
    if description:
        result["description"] = description
    
    return result


def create_response_examples(
    success_example: Dict[str, Any],
    error_examples: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Create response examples for documentation
    
    Args:
        success_example: Success response example
        error_examples: Optional list of error examples
    
    Returns:
        Dictionary with examples
    """
    result = {
        "success": success_example
    }
    
    if error_examples:
        result["errors"] = error_examples
    
    return result


def create_parameter_description(
    name: str,
    param_type: str,
    description: str,
    required: bool = True,
    example: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Create parameter description for documentation
    
    Args:
        name: Parameter name
        param_type: Parameter type
        description: Parameter description
        required: Whether parameter is required
        example: Optional example value
    
    Returns:
        Dictionary with parameter description
    """
    result = {
        "name": name,
        "type": param_type,
        "description": description,
        "required": required
    }
    
    if example is not None:
        result["example"] = example
    
    return result


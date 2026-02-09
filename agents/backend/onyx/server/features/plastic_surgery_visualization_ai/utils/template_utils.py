"""Template utilities."""

from typing import Dict, Any
import re


def render_template(template: str, context: Dict[str, Any]) -> str:
    """
    Render template with context.
    
    Args:
        template: Template string with {variable} placeholders
        context: Context dictionary
        
    Returns:
        Rendered string
    """
    return template.format(**context)


def render_template_safe(template: str, context: Dict[str, Any], default: str = "") -> str:
    """
    Render template safely (missing keys use default).
    
    Args:
        template: Template string
        context: Context dictionary
        default: Default value for missing keys
        
    Returns:
        Rendered string
    """
    class SafeDict(dict):
        def __missing__(self, key):
            return default
    
    return template.format_map(SafeDict(context))


def extract_template_variables(template: str) -> list:
    """
    Extract variable names from template.
    
    Args:
        template: Template string
        
    Returns:
        List of variable names
    """
    pattern = r'\{(\w+)\}'
    return re.findall(pattern, template)


def validate_template(template: str, context: Dict[str, Any]) -> bool:
    """
    Validate template has all required variables.
    
    Args:
        template: Template string
        context: Context dictionary
        
    Returns:
        True if all variables present
    """
    variables = extract_template_variables(template)
    return all(var in context for var in variables)


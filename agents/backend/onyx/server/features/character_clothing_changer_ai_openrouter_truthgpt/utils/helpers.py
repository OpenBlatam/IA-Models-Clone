"""
Helper utilities for the Character Clothing Changer service.
"""

import hashlib
import time
from typing import Dict, Any, Optional
from datetime import datetime


def generate_prompt_id(image_url: str, prompt: str, seed: Optional[int] = None) -> str:
    """
    Generate a unique prompt ID based on inputs.
    
    Args:
        image_url: Image URL
        prompt: Prompt text
        seed: Optional seed value
        
    Returns:
        Unique prompt ID string
    """
    data = f"{image_url}:{prompt}:{seed or 'random'}:{time.time()}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def format_workflow_summary(workflow: Dict[str, Any]) -> str:
    """
    Format a human-readable summary of a workflow.
    
    Args:
        workflow: Workflow dictionary
        
    Returns:
        Formatted summary string
    """
    nodes = workflow.get("nodes", [])
    links = workflow.get("links", [])
    
    summary = f"Workflow: {len(nodes)} nodes, {len(links)} links"
    
    # Count node types
    node_types = {}
    for node in nodes:
        node_type = node.get("type", "unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    if node_types:
        type_summary = ", ".join(f"{k}: {v}" for k, v in sorted(node_types.items()))
        summary += f" ({type_summary})"
    
    return summary


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + ('.' + ext if ext else '')
    return filename


def format_timestamp(timestamp: Optional[float] = None) -> str:
    """
    Format timestamp as ISO string.
    
    Args:
        timestamp: Unix timestamp (default: current time)
        
    Returns:
        ISO formatted timestamp string
    """
    if timestamp is None:
        timestamp = time.time()
    
    return datetime.fromtimestamp(timestamp).isoformat()


def extract_image_info(image_url: str) -> Dict[str, Any]:
    """
    Extract information from image URL.
    
    Args:
        image_url: Image URL or path
        
    Returns:
        Dictionary with image information:
        - is_url: bool - Whether it's a URL
        - scheme: str - URL scheme (http, https, file, or empty for path)
        - filename: str - Filename if available
        - extension: str - File extension if available
    """
    from urllib.parse import urlparse
    
    info = {
        "is_url": False,
        "scheme": "",
        "filename": "",
        "extension": ""
    }
    
    try:
        parsed = urlparse(image_url)
        if parsed.scheme:
            info["is_url"] = True
            info["scheme"] = parsed.scheme
            path = parsed.path
        else:
            path = image_url
        
        # Extract filename and extension
        if path:
            filename = path.split('/')[-1]
            if '.' in filename:
                name, ext = filename.rsplit('.', 1)
                info["filename"] = name
                info["extension"] = ext.lower()
            else:
                info["filename"] = filename
        
    except Exception:
        pass
    
    return info


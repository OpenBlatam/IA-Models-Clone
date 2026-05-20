"""
Base Classes for API Tools
==========================
Base classes and common functionality for all API tools.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import requests


@dataclass
class ToolResult:
    """Standard result structure for tools."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class BaseAPITool(ABC):
    """Base class for all API tools."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        self.results: list = []
    
    def set_auth_token(self, token: str):
        """Set authentication token."""
        self.session.headers["Authorization"] = f"Bearer {token}"
    
    def make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> requests.Response:
        """Make HTTP request."""
        url = f"{self.base_url}{endpoint}"
        
        if method.upper() == "GET":
            return self.session.get(url, timeout=self.timeout, **kwargs)
        elif method.upper() == "POST":
            return self.session.post(url, timeout=self.timeout, **kwargs)
        elif method.upper() == "PUT":
            return self.session.put(url, timeout=self.timeout, **kwargs)
        elif method.upper() == "DELETE":
            return self.session.delete(url, timeout=self.timeout, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Run the tool."""
        pass
    
    def export_results(self, file_path: str, format: str = "json"):
        """Export results to file."""
        from pathlib import Path
        import json
        
        file = Path(file_path)
        
        if format == "json":
            data = {
                "results": self.results,
                "exported_at": datetime.now().isoformat()
            }
            file.write_text(json.dumps(data, indent=2))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return file


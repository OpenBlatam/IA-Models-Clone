from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Request Context Entity
=====================

Domain entity representing the context of an API request.
"""



@dataclass
class RequestContext:
    """Request context with tracing information."""
    
    request_id: str
    method: str
    path: str
    client_ip: str
    user_agent: Optional[str]
    timestamp: datetime
    headers: Dict[str, str]
    query_params: Dict[str, Any]
    
    @classmethod
    def create(cls, method: str, path: str, client_ip: str, 
               user_agent: Optional[str] = None,
               headers: Optional[Dict[str, str]] = None,
               query_params: Optional[Dict[str, Any]] = None) -> "RequestContext":
        """Create a new request context."""
        return cls(
            request_id=str(uuid.uuid4()),
            method=method,
            path=path,
            client_ip=client_ip,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            headers=headers or {},
            query_params=query_params or {}
        )
    
    def get_identifier(self) -> str:
        """Get unique identifier for rate limiting."""
        return f"{self.client_ip}:{self.path}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "request_id": self.request_id,
            "method": self.method,
            "path": self.path,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "timestamp": self.timestamp.isoformat(),
            "headers": self.headers,
            "query_params": self.query_params
        } 
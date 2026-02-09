"""
Model Security
Security measures for model deployment and inference
"""

from typing import Dict, Any, Optional, List
import logging
import hashlib
import time

logger = logging.getLogger(__name__)


class ModelSecurity:
    """
    Security measures for models:
    - Input sanitization
    - Rate limiting
    - Model versioning
    - Access control
    """
    
    def __init__(self):
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.access_tokens: Dict[str, Dict[str, Any]] = {}
        self.model_checksums: Dict[str, str] = {}
    
    def sanitize_input(self, input_data: Any) -> tuple[bool, Optional[str], Any]:
        """Sanitize and validate input"""
        # Check for malicious patterns
        if isinstance(input_data, str):
            # Check for SQL injection patterns
            dangerous_patterns = ["'", '"', ";", "--", "/*", "*/", "xp_", "sp_"]
            if any(pattern in input_data for pattern in dangerous_patterns):
                return False, "Potentially dangerous input detected", None
            
            # Check length
            if len(input_data) > 10000:
                return False, "Input too long", None
        
        # Check for file paths (potential path traversal)
        if isinstance(input_data, str) and ("../" in input_data or "..\\" in input_data):
            return False, "Invalid path detected", None
        
        return True, None, input_data
    
    def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        max_requests: int = 100,
        window_seconds: int = 60
    ) -> bool:
        """Check rate limit for user/endpoint"""
        key = f"{user_id}:{endpoint}"
        current_time = time.time()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {
                "requests": [],
                "max_requests": max_requests,
                "window": window_seconds
            }
        
        limit_info = self.rate_limits[key]
        
        # Remove old requests
        limit_info["requests"] = [
            req_time for req_time in limit_info["requests"]
            if current_time - req_time < limit_info["window"]
        ]
        
        # Check limit
        if len(limit_info["requests"]) >= limit_info["max_requests"]:
            return False
        
        # Add current request
        limit_info["requests"].append(current_time)
        return True
    
    def verify_model_integrity(self, model_path: str, expected_checksum: Optional[str] = None) -> bool:
        """Verify model file integrity"""
        try:
            with open(model_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            if expected_checksum:
                return file_hash == expected_checksum
            
            # Store checksum if not provided
            self.model_checksums[model_path] = file_hash
            return True
        
        except Exception as e:
            logger.error(f"Model integrity check failed: {str(e)}")
            return False
    
    def generate_access_token(self, user_id: str, permissions: List[str]) -> str:
        """Generate access token"""
        token_data = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": time.time(),
            "expires_at": time.time() + 3600  # 1 hour
        }
        
        token = hashlib.sha256(f"{user_id}:{time.time()}".encode()).hexdigest()
        self.access_tokens[token] = token_data
        
        return token
    
    def validate_access_token(self, token: str, required_permission: Optional[str] = None) -> bool:
        """Validate access token"""
        if token not in self.access_tokens:
            return False
        
        token_data = self.access_tokens[token]
        
        # Check expiration
        if time.time() > token_data["expires_at"]:
            del self.access_tokens[token]
            return False
        
        # Check permission
        if required_permission:
            return required_permission in token_data["permissions"]
        
        return True


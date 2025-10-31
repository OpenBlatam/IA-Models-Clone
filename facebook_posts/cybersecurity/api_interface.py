from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import sys
import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging
from urllib.parse import urlparse
    from security_implementation import (
    from cybersecurity.security_implementation import (
    from fastapi import FastAPI, HTTPException, Depends, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    import argparse
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
RESTful API Interface for Cybersecurity Toolkit
Implements RORO (Receive an Object, Return an Object) pattern for tool control.
"""


# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
        SecurityConfig, SecureNetworkScanner, SecurityError, 
        create_secure_config, RateLimiter, AdaptiveRateLimiter,
        NetworkScanRateLimiter, SecureSecretManager
    )
except ImportError:
    # Fallback for different directory structure
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        SecurityConfig, SecureNetworkScanner, SecurityError,
        create_secure_config, RateLimiter, AdaptiveRateLimiter,
        NetworkScanRateLimiter, SecureSecretManager
    )

# Try to import FastAPI and related libraries
try:
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️  FastAPI not available. Install with: pip install fastapi uvicorn")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class APIRequest:
    """RORO pattern request object for API operations."""
    endpoint: str
    method: str
    data: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    user: Optional[str] = None
    session_id: Optional[str] = None
    api_key: Optional[str] = None

@dataclass
class APIResponse:
    """RORO pattern response object for API operations."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    message: Optional[str] = None
    status_code: int = 200
    timestamp: Optional[str] = None
    execution_time: Optional[float] = None

# Pydantic models for API requests/responses
class ScanRequest(BaseModel):
    target: str = Field(..., description="Target to scan")
    scan_type: str = Field(default="port_scan", description="Type of scan")
    user: Optional[str] = Field(default=None, description="User performing scan")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Additional options")

class RateLimitRequest(BaseModel):
    target: str = Field(..., description="Target to check rate limits for")

class SecretRequest(BaseModel):
    secret_name: str = Field(..., description="Name of secret to retrieve")
    source: str = Field(default="env", description="Source for secret")
    required: bool = Field(default=True, description="Whether secret is required")

class ConfigRequest(BaseModel):
    include_secrets: bool = Field(default=False, description="Include secret information")

class APIResponseModel(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    message: Optional[str] = None
    status_code: int = 200
    timestamp: Optional[str] = None
    execution_time: Optional[float] = None

class CybersecurityAPI:
    """RESTful API interface for cybersecurity toolkit using RORO pattern."""
    
    def __init__(self) -> Any:
        self.config = create_secure_config()
        self.scanner = SecureNetworkScanner(self.config)
        self.rate_limiter = RateLimiter()
        self.adaptive_limiter = AdaptiveRateLimiter()
        self.scan_limiter = NetworkScanRateLimiter()
        self.secret_manager = SecureSecretManager()
        
        # Initialize authorization for demo
        self.scanner.auth_checker.add_authorized_target("127.0.0.1", "api_user", 
                                                       int(time.time()) + 3600, ["scan"])
        self.scanner.auth_checker.record_consent("api_user", True, "network_scanning")
        
        # API security
        self.security = HTTPBearer()
        self.valid_api_keys = {"demo_api_key_12345", "test_api_key_67890"}
    
    async async def execute_api_request(self, request: APIRequest) -> APIResponse:
        """Execute API request using RORO pattern."""
        start_time = time.time()
        
        try:
            # Validate API key if required
            if request.api_key and request.api_key not in self.valid_api_keys:
                return APIResponse(
                    success=False,
                    error="Invalid API key",
                    error_code="INVALID_API_KEY",
                    status_code=401
                )
            
            # Route request based on endpoint
            if request.endpoint == "/scan":
                result = await self._handle_scan_endpoint(request)
            elif request.endpoint == "/rate-limit":
                result = await self._handle_rate_limit_endpoint(request)
            elif request.endpoint == "/secrets":
                result = await self._handle_secrets_endpoint(request)
            elif request.endpoint == "/config":
                result = await self._handle_config_endpoint(request)
            elif request.endpoint == "/health":
                result = await self._handle_health_endpoint(request)
            else:
                result = APIResponse(
                    success=False,
                    error=f"Unknown endpoint: {request.endpoint}",
                    error_code="UNKNOWN_ENDPOINT",
                    status_code=404
                )
            
            # Add execution time
            result.execution_time = time.time() - start_time
            result.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return result
            
        except Exception as e:
            logger.error(f"API execution error: {e}")
            return APIResponse(
                success=False,
                error=str(e),
                error_code="EXECUTION_ERROR",
                status_code=500,
                execution_time=time.time() - start_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
    
    async def _handle_scan_endpoint(self, request: APIRequest) -> APIResponse:
        """Handle scan endpoint."""
        try:
            data = request.data or {}
            target = data.get('target')
            scan_type = data.get('scan_type', 'port_scan')
            user = data.get('user', 'api_user')
            session_id = data.get('session_id', 'api_session')
            
            if not target:
                return APIResponse(
                    success=False,
                    error="Target is required for scan",
                    error_code="MISSING_TARGET",
                    status_code=400
                )
            
            result = await self.scanner.secure_scan(
                target=target,
                user=user,
                session_id=session_id,
                scan_type=scan_type
            )
            
            return APIResponse(
                success=result.get('success', False),
                data=result,
                message=f"Scan completed for {target}",
                status_code=200 if result.get('success', False) else 400
            )
            
        except SecurityError as e:
            return APIResponse(
                success=False,
                error=e.message,
                error_code=e.code,
                status_code=400
            )
    
    async def _handle_rate_limit_endpoint(self, request: APIRequest) -> APIResponse:
        """Handle rate limit endpoint."""
        try:
            data = request.data or {}
            target = data.get('target')
            
            if not target:
                return APIResponse(
                    success=False,
                    error="Target is required for rate-limit check",
                    error_code="MISSING_TARGET",
                    status_code=400
                )
            
            # Get rate limit statistics
            basic_stats = self.rate_limiter.get_rate_limit_stats(target)
            adaptive_stats = self.adaptive_limiter.get_adaptive_stats(target)
            scan_stats = self.scan_limiter.get_scan_stats()
            
            data = {
                "target": target,
                "basic_rate_limiter": basic_stats,
                "adaptive_rate_limiter": adaptive_stats,
                "scan_rate_limiter": scan_stats
            }
            
            return APIResponse(
                success=True,
                data=data,
                message=f"Rate limit statistics for {target}",
                status_code=200
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                error_code="RATE_LIMIT_ERROR",
                status_code=500
            )
    
    async def _handle_secrets_endpoint(self, request: APIRequest) -> APIResponse:
        """Handle secrets endpoint."""
        try:
            data = request.data or {}
            secret_name = data.get('secret_name')
            source = data.get('source', 'env')
            required = data.get('required', True)
            
            if not secret_name:
                return APIResponse(
                    success=False,
                    error="Secret name is required",
                    error_code="MISSING_SECRET_NAME",
                    status_code=400
                )
            
            secret = self.secret_manager.get_secret(secret_name, source, required=required)
            
            if secret:
                # Mask the secret for API response
                masked_secret = secret[:4] + '*' * (len(secret) - 8) + secret[-4:] if len(secret) > 8 else '***'
                
                # Validate secret strength
                validation = self.secret_manager.validate_secret_strength(secret, secret_name)
                
                response_data = {
                    "secret_name": secret_name,
                    "source": source,
                    "masked_value": masked_secret,
                    "length": len(secret),
                    "strength_validation": validation
                }
                
                return APIResponse(
                    success=True,
                    data=response_data,
                    message=f"Secret '{secret_name}' loaded successfully",
                    status_code=200
                )
            else:
                return APIResponse(
                    success=False,
                    error=f"Secret '{secret_name}' not found in {source}",
                    error_code="SECRET_NOT_FOUND",
                    status_code=404
                )
                
        except SecurityError as e:
            return APIResponse(
                success=False,
                error=e.message,
                error_code=e.code,
                status_code=400
            )
    
    async def _handle_config_endpoint(self, request: APIRequest) -> APIResponse:
        """Handle config endpoint."""
        try:
            data = request.data or {}
            include_secrets = data.get('include_secrets', False)
            
            # Get configuration information
            config_data = {
                "api_key_configured": bool(self.config.api_key),
                "encryption_key_configured": bool(self.config.encryption_key),
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout,
                "rate_limit": self.config.rate_limit,
                "session_timeout": self.config.session_timeout,
                "tls_version": self.config.tls_version,
                "verify_ssl": self.config.verify_ssl
            }
            
            # Add secret information if requested
            if include_secrets:
                config_data["secrets"] = {
                    "api_key_masked": self.config.api_key[:4] + "***" + self.config.api_key[-4:] if self.config.api_key else None,
                    "encryption_key_length": len(self.config.encryption_key) if self.config.encryption_key else 0
                }
            
            # Validate configuration
            try:
                self.config.validate()
                config_data["validation"] = "PASSED"
            except SecurityError as e:
                config_data["validation"] = f"FAILED: {e.message}"
            
            return APIResponse(
                success=True,
                data=config_data,
                message="Configuration information retrieved",
                status_code=200
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                error_code="CONFIG_ERROR",
                status_code=500
            )
    
    async def _handle_health_endpoint(self, request: APIRequest) -> APIResponse:
        """Handle health check endpoint."""
        try:
            health_data = {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "scanner": "operational",
                    "rate_limiter": "operational",
                    "secret_manager": "operational",
                    "config": "operational"
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return APIResponse(
                success=True,
                data=health_data,
                message="API is healthy",
                status_code=200
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                error_code="HEALTH_CHECK_ERROR",
                status_code=500
            )

# FastAPI application setup
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Cybersecurity Toolkit API",
        description="RESTful API for cybersecurity toolkit using RORO pattern",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize API
    api = CybersecurityAPI()
    
    # Dependency for API key validation
    async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if credentials.credentials not in api.valid_api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        return credentials.credentials
    
    @app.get("/health", response_model=APIResponseModel)
    async def health_check():
        """Health check endpoint."""
        request = APIRequest(endpoint="/health", method="GET")
        response = await api.execute_api_request(request)
        return APIResponseModel(**asdict(response))
    
    @app.post("/scan", response_model=APIResponseModel)
    async def scan_target(scan_request: ScanRequest, api_key: str = Depends(verify_api_key)):
        """Perform network scan."""
        request = APIRequest(
            endpoint="/scan",
            method="POST",
            data=scan_request.dict(),
            api_key=api_key
        )
        response = await api.execute_api_request(request)
        return APIResponseModel(**asdict(response))
    
    @app.post("/rate-limit", response_model=APIResponseModel)
    async def check_rate_limit(rate_limit_request: RateLimitRequest, api_key: str = Depends(verify_api_key)):
        """Check rate limit statistics."""
        request = APIRequest(
            endpoint="/rate-limit",
            method="POST",
            data=rate_limit_request.dict(),
            api_key=api_key
        )
        response = await api.execute_api_request(request)
        return APIResponseModel(**asdict(response))
    
    @app.post("/secrets", response_model=APIResponseModel)
    async def get_secret(secret_request: SecretRequest, api_key: str = Depends(verify_api_key)):
        """Get secret from secure store."""
        request = APIRequest(
            endpoint="/secrets",
            method="POST",
            data=secret_request.dict(),
            api_key=api_key
        )
        response = await api.execute_api_request(request)
        return APIResponseModel(**asdict(response))
    
    @app.get("/config", response_model=APIResponseModel)
    async def get_config(config_request: ConfigRequest = Depends(), api_key: str = Depends(verify_api_key)):
        """Get configuration information."""
        request = APIRequest(
            endpoint="/config",
            method="GET",
            data=config_request.dict(),
            api_key=api_key
        )
        response = await api.execute_api_request(request)
        return APIResponseModel(**asdict(response))
    
    @app.get("/docs")
    async def get_documentation():
        """Get API documentation."""
        return {
            "title": "Cybersecurity Toolkit API",
            "version": "1.0.0",
            "endpoints": {
                "/health": {
                    "method": "GET",
                    "description": "Health check endpoint",
                    "authentication": "Required"
                },
                "/scan": {
                    "method": "POST",
                    "description": "Perform network scan",
                    "authentication": "Required",
                    "body": "ScanRequest"
                },
                "/rate-limit": {
                    "method": "POST",
                    "description": "Check rate limit statistics",
                    "authentication": "Required",
                    "body": "RateLimitRequest"
                },
                "/secrets": {
                    "method": "POST",
                    "description": "Get secret from secure store",
                    "authentication": "Required",
                    "body": "SecretRequest"
                },
                "/config": {
                    "method": "GET",
                    "description": "Get configuration information",
                    "authentication": "Required",
                    "query": "ConfigRequest"
                }
            }
        }

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available. Install with: pip install fastapi uvicorn")
        return
    
    print(f"🚀 Starting Cybersecurity Toolkit API server on {host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Cybersecurity Toolkit API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    run_api_server(args.host, args.port) 
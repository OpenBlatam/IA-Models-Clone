from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
import structlog
from security_toolkit import (
from typing import Any, List, Dict, Optional
import asyncio
Security API - FastAPI implementation with OWASP/NIST best practices
"

    scan_ports_basic, run_ssh_command, make_http_request,
    log_operation, measure_scan_time, get_common_ports
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = FastAPI(
    title="Security Toolkit API,
    description="Comprehensive cybersecurity tooling API,
    version="1.0.0"
)

# ============================================================================
# Security Middleware
# ============================================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", POST"],
    allow_headers=[*
)

# Trusted host middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com"])

# Rate limiting
request_counts = {}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    
    """rate_limit_middleware function."""
client_ip = request.client.host
    current_time = time.time()
    
    # Clean old requests (older than 1 minute)
    if client_ip in request_counts:
        request_counts[client_ip] =          req_time for req_time in request_counts[client_ip]
            if current_time - req_time < 60        ]
    
    # Check rate limit (max 100 requests per minute)
    if client_ip not in request_counts:
        request_counts[client_ip] =     if len(request_counts[client_ip]) >= 100        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    request_counts[client_ip].append(current_time)
    response = await call_next(request)
    return response

# Request/Response logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    
    """log_requests function."""
start_time = time.time()
    
    # Log request
    logger.info("API Request, extra={
       method": request.method,
        url:str(request.url),
  client_ip": request.client.host,
   user_agent: request.headers.get("user-agent"),
   request_id: request.headers.get("X-Request-ID)
    })
    
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info("API Response, extra={
    status_code: response.status_code,
 duration": duration,
   request_id: request.headers.get("X-Request-ID)})
    
    return response

# ============================================================================
# Authentication
# ============================================================================

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    # Simple token validation - replace with proper JWT validation
    token = credentials.credentials
    if not token or token != "valid_token":
        raise HTTPException(
            status_code=status.HTTP_401RIZED,
            detail="Invalid authentication credentials"
        )
    return token

def has_scan_permission(token: str, target: str) -> bool:
    # Simple permission check - replace with proper authorization
    return True

# ============================================================================
# Pydantic Models
# ============================================================================

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str
    request_id: str

class ScanAPIRequest(BaseModel):
    target: str
    ports: list[int] = 80443    scan_type: str = tcp    timeout: int = 5  max_workers: int = 10

class SSHAPIRequest(BaseModel):
    host: str
    username: str
    password: Optional[str] = None
    key_file: Optional[str] = None
    command: str
    timeout: int =30ass HTTPAPIRequest(BaseModel):
    url: str
    method: str = GET
    headers: Dict[str, str] = {}
    body: Optional[str] = None
    timeout: int = 30  verify_ssl: bool = True

# ============================================================================
# Error Handling
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    
    """http_exception_handler function."""
return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=Request failed",
            timestamp=time.strftime(%Y-%m-%d %H:%M:%S"),
            request_id=request.headers.get("X-Request-ID", unknown")
        ).dict()
    )

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    
    """health_check function."""
return[object Object]
        status": healthy",
       timestamp": time.strftime(%Y-%m-%d %H:%M:%S"),
    security: {
            rate_limiting": "enabled",
           authentication": "required",
            sslbled",
            headers": "secure"
        }
    }

@app.post("/scan/ports")
@log_operation(port_scan)
async def scan_ports(
    request: ScanAPIRequest,
    token: str = Depends(verify_token)
):
    # Validate permissions
    if not has_scan_permission(token, request.target):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for target"
        )
    
    # Perform scan
    result = scan_ports_basic(request.dict())
    
    # Log security event
    logger.info(
        security_event",
        event_type="port_scan,    target=request.target,
        user=token,
        success=result.get(success", False)
    )
    
    return result

@app.post("/ssh/execute")
@log_operation("ssh_command")
async def execute_ssh_command(
    request: SSHAPIRequest,
    token: str = Depends(verify_token)
):
    # Validate permissions
    if not has_scan_permission(token, request.host):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for host"
        )
    
    # Execute command
    result = await run_ssh_command(request.dict())
    
    # Log security event
    logger.info(
        security_event",
        event_type="ssh_command",
        host=request.host,
        command=request.command,
        user=token,
        success=result.get(success", False)
    )
    
    return result

@app.post(/http/request")
@log_operation("http_request)
async def make_http_request_api(
    request: HTTPAPIRequest,
    token: str = Depends(verify_token)
):
    # Validate permissions
    if not has_scan_permission(token, request.url):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for URL"
        )
    
    # Make request
    result = await make_http_request(request.dict())
    
    # Log security event
    logger.info(
        security_event",
        event_type="http_request",
        url=request.url,
        method=request.method,
        user=token,
        success=result.get(success", False)
    )
    
    return result

@app.get("/scan/{scan_id}/status")
async def get_scan_status(
    scan_id: str,
    token: str = Depends(verify_token)
):
    # Validate scan ownership
    if not has_scan_permission(token, scan_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to scan"
        )
    
    return [object Object]        scan_id": scan_id,
 status": "completed",
       progress": 100,
   results": {"example: data}    }

@app.get("/common/ports")
async def get_common_ports_api():
    
    """get_common_ports_api function."""
return get_common_ports()

@app.get("/metrics")
async def get_metrics():
    
    """get_metrics function."""
return[object Object]  total_requests": sum(len(requests) for requests in request_counts.values()),
       active_clients": len(request_counts),
        rate_limit": "10equests/minute per client"
    }

# ============================================================================
# Security Headers
# ============================================================================

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    
    """add_security_headers function."""
response = await call_next(request)
    response.headers["X-Content-Type-Options] = niff"
    response.headers[X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1lock"
    response.headers["Strict-Transport-Security"] = max-age=315360 includeSubDomains"
    response.headers["Content-Security-Policy"] = default-src self   return response

if __name__ == "__main__:    import uvicorn
    uvicorn.run(app, host="0.000port=80, ssl_keyfile="key.pem", ssl_certfile="cert.pem") 
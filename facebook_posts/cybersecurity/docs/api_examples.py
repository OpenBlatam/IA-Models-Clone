from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import uuid
import json
from enum import Enum
            import jwt
        import time
    import uvicorn
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
FastAPI Security Tooling API Examples
Practical examples demonstrating best practices for API-driven security tooling.
"""


# ============================================================================
# MODELS
# ============================================================================

class ScanType(str, Enum):
    TCP = "tcp"
    UDP = "udp"
    SYN = "syn"

class ScanStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ScanRequest(BaseModel):
    targets: List[str] = Field(..., min_items=1, max_items=100, description="Target IPs or hostnames")
    ports: Optional[List[int]] = Field(None, ge=1, le=65535, description="Ports to scan")
    scan_type: ScanType = Field(ScanType.TCP, description="Type of scan")
    timeout: float = Field(5.0, ge=0.1, le=60.0, description="Scan timeout in seconds")
    banner_grab: bool = Field(True, description="Attempt banner grabbing")
    ssl_check: bool = Field(True, description="Check SSL certificates")
    
    @validator('targets')
    def validate_targets(cls, v) -> Optional[Dict[str, Any]]:
        for target in v:
            if not target or len(target.strip()) == 0:
                raise ValueError("Target cannot be empty")
        return v

class ScanResponse(BaseModel):
    scan_id: str
    status: ScanStatus
    progress: float = Field(..., ge=0.0, le=100.0)
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime] = None

class ScanResult(BaseModel):
    target: str
    port: int
    is_open: bool
    service_name: Optional[str] = None
    banner: Optional[str] = None
    ssl_info: Optional[Dict[str, Any]] = None
    response_time: float
    protocol: str = "tcp"

class VulnerabilityReport(BaseModel):
    scan_id: str
    vulnerabilities: List[Dict[str, Any]]
    risk_score: float = Field(..., ge=0.0, le=10.0)
    summary: str
    recommendations: List[str]
    generated_at: datetime

# ============================================================================
# SECURITY
# ============================================================================

security = HTTPBearer()

class SecurityService:
    def __init__(self) -> Any:
        self.secret_key = "your-secret-key-here"
        self.algorithm = "HS256"
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated user."""
    security_service = SecurityService()
    payload = security_service.verify_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return user_id

# ============================================================================
# SERVICES
# ============================================================================

class ScanService:
    def __init__(self) -> Any:
        self.active_scans = {}
        self.scan_results = {}
    
    async def create_scan(self, scan_request: ScanRequest, user_id: str) -> str:
        """Create a new scan."""
        scan_id = str(uuid.uuid4())
        
        scan_data = {
            "scan_id": scan_id,
            "user_id": user_id,
            "request": scan_request.dict(),
            "status": ScanStatus.PENDING,
            "progress": 0.0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "results": None,
            "error": None
        }
        
        self.active_scans[scan_id] = scan_data
        return scan_id
    
    async def start_scan(self, scan_id: str) -> None:
        """Start a scan in background."""
        if scan_id not in self.active_scans:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        scan_data = self.active_scans[scan_id]
        scan_data["status"] = ScanStatus.RUNNING
        scan_data["updated_at"] = datetime.utcnow()
        
        # Simulate scan execution
        asyncio.create_task(self._execute_scan(scan_id))
    
    async def _execute_scan(self, scan_id: str) -> None:
        """Execute scan in background."""
        scan_data = self.active_scans[scan_id]
        request_data = scan_data["request"]
        
        try:
            # Simulate scanning process
            total_targets = len(request_data["targets"])
            results = {}
            
            for i, target in enumerate(request_data["targets"]):
                target_results = []
                
                ports = request_data.get("ports", [80, 443, 22, 21, 23, 25, 53, 110, 143, 993, 995])
                
                for port in ports:
                    # Simulate port scan
                    await asyncio.sleep(0.1)  # Simulate network delay
                    
                    result = ScanResult(
                        target=target,
                        port=port,
                        is_open=port in [80, 443, 22],  # Simulate open ports
                        service_name=self._get_service_name(port),
                        banner=f"Service on port {port}" if port in [80, 443, 22] else None,
                        response_time=0.1,
                        protocol=request_data["scan_type"]
                    )
                    target_results.append(result)
                
                results[target] = target_results
                
                # Update progress
                progress = (i + 1) / total_targets * 100
                scan_data["progress"] = progress
                scan_data["updated_at"] = datetime.utcnow()
            
            # Complete scan
            scan_data["status"] = ScanStatus.COMPLETED
            scan_data["progress"] = 100.0
            scan_data["results"] = results
            scan_data["updated_at"] = datetime.utcnow()
            
        except Exception as e:
            scan_data["status"] = ScanStatus.FAILED
            scan_data["error"] = str(e)
            scan_data["updated_at"] = datetime.utcnow()
    
    def _get_service_name(self, port: int) -> Optional[str]:
        """Get service name for port."""
        services = {
            21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
            80: "http", 110: "pop3", 143: "imap", 443: "https",
            993: "imaps", 995: "pop3s"
        }
        return services.get(port)
    
    def get_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get scan by ID."""
        return self.active_scans.get(scan_id)
    
    def get_user_scans(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get scans for user."""
        user_scans = [
            scan for scan in self.active_scans.values()
            if scan["user_id"] == user_id
        ]
        
        # Sort by creation date (newest first)
        user_scans.sort(key=lambda x: x["created_at"], reverse=True)
        
        return user_scans[offset:offset + limit]

class VulnerabilityService:
    def __init__(self) -> Any:
        self.vulnerability_db = {
            "open_ports": {
                "risk": "medium",
                "description": "Open ports may expose services to attack",
                "recommendation": "Close unnecessary ports and use firewall rules"
            },
            "weak_ssl": {
                "risk": "high",
                "description": "Weak SSL configuration detected",
                "recommendation": "Update SSL configuration and use strong ciphers"
            },
            "default_credentials": {
                "risk": "critical",
                "description": "Default credentials detected",
                "recommendation": "Change default passwords immediately"
            }
        }
    
    def analyze_scan_results(self, scan_results: Dict[str, Any]) -> VulnerabilityReport:
        """Analyze scan results for vulnerabilities."""
        vulnerabilities = []
        risk_score = 0.0
        
        for target, target_results in scan_results.items():
            for result in target_results:
                if result.is_open:
                    # Check for common vulnerabilities
                    if result.port in [21, 23]:  # FTP, Telnet
                        vulnerabilities.append({
                            "target": target,
                            "port": result.port,
                            "type": "insecure_protocol",
                            "risk": "high",
                            "description": f"Insecure protocol on port {result.port}",
                            "recommendation": "Use secure alternatives (SFTP, SSH)"
                        })
                        risk_score += 2.0
                    
                    if result.port == 22 and result.banner:
                        if "OpenSSH" in result.banner and "7.0" in result.banner:
                            vulnerabilities.append({
                                "target": target,
                                "port": result.port,
                                "type": "outdated_service",
                                "risk": "medium",
                                "description": "Outdated SSH version detected",
                                "recommendation": "Update SSH to latest version"
                            })
                            risk_score += 1.5
        
        # Normalize risk score to 0-10 scale
        risk_score = min(risk_score, 10.0)
        
        return VulnerabilityReport(
            scan_id=str(uuid.uuid4()),
            vulnerabilities=vulnerabilities,
            risk_score=risk_score,
            summary=f"Found {len(vulnerabilities)} potential vulnerabilities",
            recommendations=[
                "Implement network segmentation",
                "Use strong authentication",
                "Regular security updates",
                "Monitor for suspicious activity"
            ],
            generated_at=datetime.utcnow()
        )

# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    def __init__(self) -> Any:
        self.requests = {}
        self.limits = {
            "default": {"requests": 100, "window": 3600},
            "scan": {"requests": 10, "window": 3600},
            "admin": {"requests": 1000, "window": 3600}
        }
    
    def is_allowed(self, client_id: str, endpoint: str = "default") -> bool:
        """Check if request is allowed."""
        now = time.time()
        limit = self.limits.get(endpoint, self.limits["default"])
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < limit["window"]
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= limit["requests"]:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True

# ============================================================================
# MIDDLEWARE
# ============================================================================

async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    client_id = request.client.host
    endpoint = "scan" if "scan" in request.url.path else "default"
    
    rate_limiter = RateLimiter()
    
    if not rate_limiter.is_allowed(client_id, endpoint):
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded for {endpoint}"}
        )
    
    response = await call_next(request)
    return response

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Security Scanning API",
    description="Comprehensive security scanning and vulnerability assessment API",
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

# Add rate limiting middleware
app.middleware("http")(rate_limit_middleware)

# ============================================================================
# DEPENDENCIES
# ============================================================================

def get_scan_service() -> ScanService:
    """Get scan service dependency."""
    return ScanService()

def get_vulnerability_service() -> VulnerabilityService:
    """Get vulnerability service dependency."""
    return VulnerabilityService()

# ============================================================================
# ROUTES
# ============================================================================

@app.post(
    "/api/v1/scans",
    response_model=ScanResponse,
    status_code=201,
    summary="Create a new security scan",
    description="""
    Create a new security scan with the specified targets and parameters.
    
    The scan will run asynchronously and can be monitored using the returned scan ID.
    """,
    responses={
        201: {"description": "Scan created successfully"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"}
    }
)
async def create_scan(
    scan_request: ScanRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
    scan_service: ScanService = Depends(get_scan_service)
):
    """Create a new security scan."""
    try:
        # Create scan
        scan_id = await scan_service.create_scan(scan_request, current_user)
        
        # Start scan in background
        background_tasks.add_task(scan_service.start_scan, scan_id)
        
        # Return scan response
        scan_data = scan_service.get_scan(scan_id)
        return ScanResponse(
            scan_id=scan_id,
            status=scan_data["status"],
            progress=scan_data["progress"],
            created_at=scan_data["created_at"],
            updated_at=scan_data["updated_at"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/api/v1/scans/{scan_id}",
    response_model=ScanResponse,
    summary="Get scan status and results",
    description="Retrieve the current status and results of a scan.",
    responses={
        200: {"description": "Scan found"},
        404: {"description": "Scan not found"}
    }
)
async def get_scan(
    scan_id: str,
    current_user: str = Depends(get_current_user),
    scan_service: ScanService = Depends(get_scan_service)
):
    """Get scan by ID."""
    scan_data = scan_service.get_scan(scan_id)
    
    if not scan_data:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    if scan_data["user_id"] != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return ScanResponse(
        scan_id=scan_id,
        status=scan_data["status"],
        progress=scan_data["progress"],
        results=scan_data["results"],
        error=scan_data["error"],
        created_at=scan_data["created_at"],
        updated_at=scan_data["updated_at"]
    )

@app.get(
    "/api/v1/scans",
    response_model=List[ScanResponse],
    summary="List user scans",
    description="Get a list of scans for the authenticated user."
)
async def list_scans(
    limit: int = Field(50, ge=1, le=100),
    offset: int = Field(0, ge=0),
    current_user: str = Depends(get_current_user),
    scan_service: ScanService = Depends(get_scan_service)
):
    """List scans for user."""
    scans = scan_service.get_user_scans(current_user, limit, offset)
    
    return [
        ScanResponse(
            scan_id=scan["scan_id"],
            status=scan["status"],
            progress=scan["progress"],
            results=scan["results"],
            error=scan["error"],
            created_at=scan["created_at"],
            updated_at=scan["updated_at"]
        )
        for scan in scans
    ]

@app.post(
    "/api/v1/scans/{scan_id}/vulnerability-report",
    response_model=VulnerabilityReport,
    summary="Generate vulnerability report",
    description="Analyze scan results and generate a vulnerability report.",
    responses={
        200: {"description": "Vulnerability report generated"},
        404: {"description": "Scan not found or not completed"}
    }
)
async def generate_vulnerability_report(
    scan_id: str,
    current_user: str = Depends(get_current_user),
    scan_service: ScanService = Depends(get_scan_service),
    vulnerability_service: VulnerabilityService = Depends(get_vulnerability_service)
):
    """Generate vulnerability report for completed scan."""
    scan_data = scan_service.get_scan(scan_id)
    
    if not scan_data:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    if scan_data["user_id"] != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if scan_data["status"] != ScanStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Scan must be completed to generate report")
    
    if not scan_data["results"]:
        raise HTTPException(status_code=400, detail="No scan results available")
    
    # Generate vulnerability report
    report = vulnerability_service.analyze_scan_results(scan_data["results"])
    report.scan_id = scan_id
    
    return report

@app.delete(
    "/api/v1/scans/{scan_id}",
    summary="Cancel scan",
    description="Cancel a running scan.",
    responses={
        200: {"description": "Scan cancelled successfully"},
        404: {"description": "Scan not found"},
        400: {"description": "Scan cannot be cancelled"}
    }
)
async def cancel_scan(
    scan_id: str,
    current_user: str = Depends(get_current_user),
    scan_service: ScanService = Depends(get_scan_service)
):
    """Cancel a scan."""
    scan_data = scan_service.get_scan(scan_id)
    
    if not scan_data:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    if scan_data["user_id"] != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if scan_data["status"] not in [ScanStatus.PENDING, ScanStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="Scan cannot be cancelled")
    
    # Cancel scan
    scan_data["status"] = ScanStatus.CANCELLED
    scan_data["updated_at"] = datetime.utcnow()
    
    return {"message": "Scan cancelled successfully"}

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get(
    "/health",
    summary="Health check",
    description="Check API health status."
)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }

@app.get(
    "/ready",
    summary="Readiness check",
    description="Check if API is ready to handle requests."
)
async def readiness_check():
    """Readiness check endpoint."""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow(),
        "services": {
            "scan_service": "available",
            "vulnerability_service": "available"
        }
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ============================================================================
# MAIN
# ============================================================================

match __name__:
    case "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
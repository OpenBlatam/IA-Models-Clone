"""
Security Routes - API endpoints for content security and threat detection
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator

from ..core.content_security_engine import (
    analyze_content_security,
    encrypt_content,
    decrypt_content,
    create_security_policy,
    perform_security_audit,
    get_security_engine_health,
    ContentSecurity,
    SecurityPolicy,
    SecurityAudit
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/security", tags=["Content Security"])


# Request/Response Models
class SecurityAnalysisRequest(BaseModel):
    """Request model for security analysis"""
    content: str = Field(..., description="Content to analyze for security threats")
    content_id: Optional[str] = Field(None, description="Unique identifier for the content")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context information")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Content cannot be empty')
        if len(v) > 100000:  # 100KB limit
            raise ValueError('Content too large (max 100KB)')
        return v


class EncryptionRequest(BaseModel):
    """Request model for content encryption"""
    content: str = Field(..., description="Content to encrypt")
    password: Optional[str] = Field(None, description="Password for encryption (optional)")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Content cannot be empty')
        return v


class DecryptionRequest(BaseModel):
    """Request model for content decryption"""
    encrypted_data: Dict[str, Any] = Field(..., description="Encrypted data to decrypt")
    password: Optional[str] = Field(None, description="Password for decryption (optional)")


class SecurityPolicyRequest(BaseModel):
    """Request model for creating security policy"""
    policy_name: str = Field(..., description="Name of the security policy")
    policy_type: str = Field(..., description="Type of security policy")
    rules: List[Dict[str, Any]] = Field(..., description="Security rules for the policy")
    
    @validator('policy_name')
    def validate_policy_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Policy name cannot be empty')
        return v


class SecurityAuditRequest(BaseModel):
    """Request model for security audit"""
    content_list: List[str] = Field(..., description="List of content to audit")
    audit_type: str = Field("comprehensive", description="Type of audit to perform")
    
    @validator('content_list')
    def validate_content_list(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Content list cannot be empty')
        if len(v) > 100:  # Limit to 100 items
            raise ValueError('Too many content items (max 100)')
        return v


class SecurityAnalysisResponse(BaseModel):
    """Response model for security analysis"""
    success: bool
    data: Optional[ContentSecurity] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class EncryptionResponse(BaseModel):
    """Response model for encryption"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class DecryptionResponse(BaseModel):
    """Response model for decryption"""
    success: bool
    data: Optional[str] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class SecurityPolicyResponse(BaseModel):
    """Response model for security policy creation"""
    success: bool
    data: Optional[SecurityPolicy] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class SecurityAuditResponse(BaseModel):
    """Response model for security audit"""
    success: bool
    data: Optional[SecurityAudit] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class HealthResponse(BaseModel):
    """Response model for health check"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime


# Route Handlers
@router.post("/analyze", response_model=SecurityAnalysisResponse)
async def analyze_content_security_endpoint(
    request: SecurityAnalysisRequest,
    background_tasks: BackgroundTasks
) -> SecurityAnalysisResponse:
    """Analyze content for security threats and vulnerabilities"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting security analysis for content: {request.content_id or 'unknown'}")
        
        # Perform security analysis
        security_analysis = await analyze_content_security(
            content=request.content,
            content_id=request.content_id or "",
            context=request.context or {}
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log analysis results
        background_tasks.add_task(
            log_security_analysis,
            request.content_id or "unknown",
            security_analysis.security_score,
            security_analysis.threat_count
        )
        
        return SecurityAnalysisResponse(
            success=True,
            data=security_analysis,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Security analysis failed: {e}")
        
        return SecurityAnalysisResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.post("/encrypt", response_model=EncryptionResponse)
async def encrypt_content_endpoint(
    request: EncryptionRequest,
    background_tasks: BackgroundTasks
) -> EncryptionResponse:
    """Encrypt content with optional password protection"""
    start_time = datetime.now()
    
    try:
        logger.info("Starting content encryption")
        
        # Encrypt content
        encrypted_data = await encrypt_content(
            content=request.content,
            password=request.password
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log encryption
        background_tasks.add_task(
            log_encryption_operation,
            "encrypt",
            len(request.content)
        )
        
        return EncryptionResponse(
            success=True,
            data=encrypted_data,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Content encryption failed: {e}")
        
        return EncryptionResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.post("/decrypt", response_model=DecryptionResponse)
async def decrypt_content_endpoint(
    request: DecryptionRequest,
    background_tasks: BackgroundTasks
) -> DecryptionResponse:
    """Decrypt content with optional password"""
    start_time = datetime.now()
    
    try:
        logger.info("Starting content decryption")
        
        # Decrypt content
        decrypted_content = await decrypt_content(
            encrypted_data=request.encrypted_data,
            password=request.password
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log decryption
        background_tasks.add_task(
            log_encryption_operation,
            "decrypt",
            len(decrypted_content)
        )
        
        return DecryptionResponse(
            success=True,
            data=decrypted_content,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Content decryption failed: {e}")
        
        return DecryptionResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.post("/policy", response_model=SecurityPolicyResponse)
async def create_security_policy_endpoint(
    request: SecurityPolicyRequest,
    background_tasks: BackgroundTasks
) -> SecurityPolicyResponse:
    """Create a new security policy"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Creating security policy: {request.policy_name}")
        
        # Create security policy
        security_policy = await create_security_policy(
            policy_name=request.policy_name,
            policy_type=request.policy_type,
            rules=request.rules
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log policy creation
        background_tasks.add_task(
            log_policy_operation,
            "create",
            request.policy_name
        )
        
        return SecurityPolicyResponse(
            success=True,
            data=security_policy,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Security policy creation failed: {e}")
        
        return SecurityPolicyResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.post("/audit", response_model=SecurityAuditResponse)
async def perform_security_audit_endpoint(
    request: SecurityAuditRequest,
    background_tasks: BackgroundTasks
) -> SecurityAuditResponse:
    """Perform comprehensive security audit on multiple content pieces"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting security audit for {len(request.content_list)} content pieces")
        
        # Perform security audit
        security_audit = await perform_security_audit(
            content_list=request.content_list,
            audit_type=request.audit_type
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log audit results
        background_tasks.add_task(
            log_audit_operation,
            request.audit_type,
            len(request.content_list),
            security_audit.threats_detected,
            security_audit.security_score
        )
        
        return SecurityAuditResponse(
            success=True,
            data=security_audit,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Security audit failed: {e}")
        
        return SecurityAuditResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.get("/health", response_model=HealthResponse)
async def get_security_health() -> HealthResponse:
    """Get security engine health status"""
    try:
        logger.info("Checking security engine health")
        
        # Get health status
        health_data = await get_security_engine_health()
        
        return HealthResponse(
            success=True,
            data=health_data,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        
        return HealthResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )


@router.get("/threats/types")
async def get_threat_types() -> Dict[str, Any]:
    """Get available threat types and their descriptions"""
    return {
        "threat_types": {
            "sql_injection": {
                "description": "SQL injection attacks attempt to manipulate database queries",
                "severity": "high",
                "examples": ["'; DROP TABLE users; --", "1' OR '1'='1"]
            },
            "xss_attack": {
                "description": "Cross-site scripting attacks inject malicious scripts",
                "severity": "high",
                "examples": ["<script>alert('XSS')</script>", "javascript:alert('XSS')"]
            },
            "path_traversal": {
                "description": "Path traversal attacks attempt to access files outside intended directory",
                "severity": "medium",
                "examples": ["../../../etc/passwd", "..\\..\\..\\windows\\system32"]
            },
            "command_injection": {
                "description": "Command injection attacks attempt to execute system commands",
                "severity": "critical",
                "examples": ["; cat /etc/passwd", "| dir", "`whoami`"]
            },
            "malicious_content": {
                "description": "Malicious content contains dangerous code execution patterns",
                "severity": "critical",
                "examples": ["eval(", "exec(", "system("]
            }
        },
        "severity_levels": {
            "critical": "Immediate action required",
            "high": "High priority action required",
            "medium": "Medium priority action required",
            "low": "Low priority monitoring"
        }
    }


@router.get("/compliance/standards")
async def get_compliance_standards() -> Dict[str, Any]:
    """Get supported compliance standards and their requirements"""
    return {
        "compliance_standards": {
            "gdpr": {
                "name": "General Data Protection Regulation",
                "description": "EU regulation for data protection and privacy",
                "requirements": [
                    "Data minimization",
                    "Consent management",
                    "Right to erasure",
                    "Data portability",
                    "Privacy by design"
                ],
                "penalties": "Up to 4% of annual global turnover or €20 million"
            },
            "hipaa": {
                "name": "Health Insurance Portability and Accountability Act",
                "description": "US regulation for healthcare data protection",
                "requirements": [
                    "Administrative safeguards",
                    "Physical safeguards",
                    "Technical safeguards",
                    "Access controls",
                    "Audit controls"
                ],
                "penalties": "Up to $1.5 million per violation"
            },
            "pci_dss": {
                "name": "Payment Card Industry Data Security Standard",
                "description": "Security standard for payment card data",
                "requirements": [
                    "Secure network and systems",
                    "Protect cardholder data",
                    "Vulnerability management",
                    "Access control measures",
                    "Regular monitoring and testing"
                ],
                "penalties": "Up to $500,000 per incident"
            }
        }
    }


@router.get("/policies")
async def get_security_policies() -> Dict[str, Any]:
    """Get available security policies"""
    return {
        "security_policies": {
            "content_validation": {
                "description": "Validates content for security issues",
                "rules": [
                    "Maximum content length",
                    "Allowed file types",
                    "Forbidden patterns"
                ]
            },
            "data_protection": {
                "description": "Protects sensitive data",
                "rules": [
                    "Encrypt sensitive data",
                    "Hash passwords",
                    "Sanitize input",
                    "Validate encoding"
                ]
            },
            "access_control": {
                "description": "Controls access to content",
                "rules": [
                    "Require authentication",
                    "Rate limit requests",
                    "Log all access",
                    "Session timeout"
                ]
            },
            "threat_detection": {
                "description": "Detects and prevents threats",
                "rules": [
                    "Scan for malware",
                    "Detect injection attacks",
                    "Monitor suspicious patterns",
                    "Block known threats"
                ]
            }
        }
    }


# Background Tasks
async def log_security_analysis(content_id: str, security_score: float, threat_count: int) -> None:
    """Log security analysis results"""
    try:
        logger.info(f"Security analysis completed - Content: {content_id}, Score: {security_score:.2f}, Threats: {threat_count}")
    except Exception as e:
        logger.warning(f"Failed to log security analysis: {e}")


async def log_encryption_operation(operation: str, content_length: int) -> None:
    """Log encryption/decryption operations"""
    try:
        logger.info(f"Encryption operation completed - Type: {operation}, Length: {content_length}")
    except Exception as e:
        logger.warning(f"Failed to log encryption operation: {e}")


async def log_policy_operation(operation: str, policy_name: str) -> None:
    """Log security policy operations"""
    try:
        logger.info(f"Security policy operation completed - Type: {operation}, Policy: {policy_name}")
    except Exception as e:
        logger.warning(f"Failed to log policy operation: {e}")


async def log_audit_operation(
    audit_type: str,
    content_count: int,
    threats_detected: int,
    security_score: float
) -> None:
    """Log security audit results"""
    try:
        logger.info(f"Security audit completed - Type: {audit_type}, Content: {content_count}, Threats: {threats_detected}, Score: {security_score:.2f}")
    except Exception as e:
        logger.warning(f"Failed to log audit operation: {e}")



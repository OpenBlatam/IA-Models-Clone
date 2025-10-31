"""
Cybersecurity API Endpoints
===========================

REST API endpoints for cybersecurity integration,
threat detection, and security monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.cybersecurity_service import (
    CybersecurityService, ThreatLevel, ThreatType, SecurityEventType, SecurityStatus,
    SecurityThreat, SecurityEvent, SecurityAlert, SecurityPolicy
)
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/cybersecurity", tags=["Cybersecurity"])

# Pydantic models
class ThreatDetectionRequest(BaseModel):
    threat_type: str = Field(..., description="Type of threat")
    source_ip: str = Field(..., description="Source IP address")
    target_ip: str = Field(..., description="Target IP address")
    indicators: List[str] = Field(..., description="Threat indicators")
    description: str = Field(..., description="Threat description")

class SecurityEventRequest(BaseModel):
    event_type: str = Field(..., description="Type of security event")
    source_ip: str = Field(..., description="Source IP address")
    target_ip: str = Field(..., description="Target IP address")
    description: str = Field(..., description="Event description")
    user_id: Optional[str] = Field(None, description="User ID")
    severity: str = Field("low", description="Event severity")

class AlertAcknowledgeRequest(BaseModel):
    alert_id: str = Field(..., description="Alert ID to acknowledge")

class AlertResolveRequest(BaseModel):
    alert_id: str = Field(..., description="Alert ID to resolve")
    actions_taken: List[str] = Field(..., description="Actions taken to resolve")

class DataEncryptionRequest(BaseModel):
    data: str = Field(..., description="Data to encrypt")

class DataDecryptionRequest(BaseModel):
    encrypted_data: str = Field(..., description="Encrypted data to decrypt")

class DigitalSignatureRequest(BaseModel):
    data: str = Field(..., description="Data to sign")

class DigitalSignatureVerificationRequest(BaseModel):
    data: str = Field(..., description="Data to verify")
    signature: str = Field(..., description="Digital signature")

# Global cybersecurity service instance
cybersecurity_service = None

def get_cybersecurity_service() -> CybersecurityService:
    """Get global cybersecurity service instance."""
    global cybersecurity_service
    if cybersecurity_service is None:
        cybersecurity_service = CybersecurityService({
            "cybersecurity": {
                "threat_detection_enabled": True,
                "real_time_monitoring": True,
                "auto_response_enabled": True,
                "encryption_enabled": True,
                "log_retention_days": 90,
                "alert_threshold": 0.7
            }
        })
    return cybersecurity_service

# API Endpoints

@router.post("/initialize", response_model=Dict[str, str])
async def initialize_cybersecurity_service(
    current_user: User = Depends(require_permission("cybersecurity:manage"))
):
    """Initialize the cybersecurity service."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        await cybersecurity_service.initialize()
        return {"message": "Cybersecurity Service initialized successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize cybersecurity service: {str(e)}")

@router.get("/status", response_model=Dict[str, Any])
async def get_cybersecurity_status(
    current_user: User = Depends(require_permission("cybersecurity:view"))
):
    """Get cybersecurity service status."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        status = await cybersecurity_service.get_service_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cybersecurity status: {str(e)}")

@router.post("/threats/detect", response_model=Dict[str, Any])
async def detect_threat(
    request: ThreatDetectionRequest,
    current_user: User = Depends(require_permission("cybersecurity:detect"))
):
    """Detect security threat."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        # Convert string to enum
        threat_type = ThreatType(request.threat_type)
        
        # Detect threat
        threat = await cybersecurity_service.detect_threat(
            threat_type=threat_type,
            source_ip=request.source_ip,
            target_ip=request.target_ip,
            indicators=request.indicators,
            description=request.description
        )
        
        return {
            "threat_id": threat.threat_id,
            "threat_type": threat.threat_type.value,
            "threat_level": threat.threat_level.value,
            "source_ip": threat.source_ip,
            "target_ip": threat.target_ip,
            "description": threat.description,
            "indicators": threat.indicators,
            "timestamp": threat.timestamp.isoformat(),
            "status": threat.status,
            "mitigation": threat.mitigation,
            "metadata": threat.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect threat: {str(e)}")

@router.get("/threats", response_model=List[Dict[str, Any]])
async def get_security_threats(
    threat_type: Optional[str] = Query(None, description="Filter by threat type"),
    threat_level: Optional[str] = Query(None, description="Filter by threat level"),
    limit: int = Query(100, description="Maximum number of threats"),
    current_user: User = Depends(require_permission("cybersecurity:view"))
):
    """Get security threats."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        # Convert string to enum if provided
        threat_type_enum = ThreatType(threat_type) if threat_type else None
        threat_level_enum = ThreatLevel(threat_level) if threat_level else None
        
        # Get threats
        threats = await cybersecurity_service.get_security_threats(
            threat_type_enum, threat_level_enum, limit
        )
        
        result = []
        for threat in threats:
            threat_dict = {
                "threat_id": threat.threat_id,
                "threat_type": threat.threat_type.value,
                "threat_level": threat.threat_level.value,
                "source_ip": threat.source_ip,
                "target_ip": threat.target_ip,
                "description": threat.description,
                "indicators": threat.indicators,
                "timestamp": threat.timestamp.isoformat(),
                "status": threat.status,
                "mitigation": threat.mitigation,
                "metadata": threat.metadata
            }
            result.append(threat_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get security threats: {str(e)}")

@router.get("/threats/{threat_id}", response_model=Dict[str, Any])
async def get_security_threat(
    threat_id: str,
    current_user: User = Depends(require_permission("cybersecurity:view"))
):
    """Get specific security threat."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        threats = await cybersecurity_service.get_security_threats()
        threat = next((t for t in threats if t.threat_id == threat_id), None)
        
        if not threat:
            raise HTTPException(status_code=404, detail="Security threat not found")
        
        return {
            "threat_id": threat.threat_id,
            "threat_type": threat.threat_type.value,
            "threat_level": threat.threat_level.value,
            "source_ip": threat.source_ip,
            "target_ip": threat.target_ip,
            "description": threat.description,
            "indicators": threat.indicators,
            "timestamp": threat.timestamp.isoformat(),
            "status": threat.status,
            "mitigation": threat.mitigation,
            "metadata": threat.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get security threat: {str(e)}")

@router.post("/events/create", response_model=Dict[str, Any])
async def create_security_event(
    request: SecurityEventRequest,
    current_user: User = Depends(require_permission("cybersecurity:create"))
):
    """Create security event."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        # Convert string to enum
        event_type = SecurityEventType(request.event_type)
        severity = ThreatLevel(request.severity)
        
        # Create event
        event = await cybersecurity_service.create_security_event(
            event_type=event_type,
            source_ip=request.source_ip,
            target_ip=request.target_ip,
            description=request.description,
            user_id=request.user_id,
            severity=severity
        )
        
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "user_id": event.user_id,
            "source_ip": event.source_ip,
            "target_ip": event.target_ip,
            "description": event.description,
            "severity": event.severity.value,
            "timestamp": event.timestamp.isoformat(),
            "status": event.status,
            "metadata": event.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create security event: {str(e)}")

@router.get("/events", response_model=List[Dict[str, Any]])
async def get_security_events(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(100, description="Maximum number of events"),
    current_user: User = Depends(require_permission("cybersecurity:view"))
):
    """Get security events."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        # Convert string to enum if provided
        event_type_enum = SecurityEventType(event_type) if event_type else None
        severity_enum = ThreatLevel(severity) if severity else None
        
        # Get events
        events = await cybersecurity_service.get_security_events(
            event_type_enum, severity_enum, limit
        )
        
        result = []
        for event in events:
            event_dict = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "user_id": event.user_id,
                "source_ip": event.source_ip,
                "target_ip": event.target_ip,
                "description": event.description,
                "severity": event.severity.value,
                "timestamp": event.timestamp.isoformat(),
                "status": event.status,
                "metadata": event.metadata
            }
            result.append(event_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get security events: {str(e)}")

@router.get("/alerts", response_model=List[Dict[str, Any]])
async def get_security_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledgment status"),
    resolved: Optional[bool] = Query(None, description="Filter by resolution status"),
    limit: int = Query(100, description="Maximum number of alerts"),
    current_user: User = Depends(require_permission("cybersecurity:view"))
):
    """Get security alerts."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        # Convert string to enum if provided
        severity_enum = ThreatLevel(severity) if severity else None
        
        # Get alerts
        alerts = await cybersecurity_service.get_security_alerts(
            severity_enum, acknowledged, resolved, limit
        )
        
        result = []
        for alert in alerts:
            alert_dict = {
                "alert_id": alert.alert_id,
                "threat_id": alert.threat_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved,
                "actions_taken": alert.actions_taken,
                "metadata": alert.metadata
            }
            result.append(alert_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get security alerts: {str(e)}")

@router.post("/alerts/acknowledge", response_model=Dict[str, str])
async def acknowledge_alert(
    request: AlertAcknowledgeRequest,
    current_user: User = Depends(require_permission("cybersecurity:manage"))
):
    """Acknowledge security alert."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        success = await cybersecurity_service.acknowledge_alert(request.alert_id)
        
        if success:
            return {"message": f"Alert {request.alert_id} acknowledged successfully"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")

@router.post("/alerts/resolve", response_model=Dict[str, str])
async def resolve_alert(
    request: AlertResolveRequest,
    current_user: User = Depends(require_permission("cybersecurity:manage"))
):
    """Resolve security alert."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        success = await cybersecurity_service.resolve_alert(
            request.alert_id, request.actions_taken
        )
        
        if success:
            return {"message": f"Alert {request.alert_id} resolved successfully"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")

@router.post("/encryption/encrypt", response_model=Dict[str, str])
async def encrypt_data(
    request: DataEncryptionRequest,
    current_user: User = Depends(require_permission("cybersecurity:encrypt"))
):
    """Encrypt data."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        encrypted_data = await cybersecurity_service.encrypt_data(request.data)
        
        return {
            "encrypted_data": encrypted_data,
            "message": "Data encrypted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to encrypt data: {str(e)}")

@router.post("/encryption/decrypt", response_model=Dict[str, str])
async def decrypt_data(
    request: DataDecryptionRequest,
    current_user: User = Depends(require_permission("cybersecurity:decrypt"))
):
    """Decrypt data."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        decrypted_data = await cybersecurity_service.decrypt_data(request.encrypted_data)
        
        return {
            "decrypted_data": decrypted_data,
            "message": "Data decrypted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to decrypt data: {str(e)}")

@router.post("/signature/sign", response_model=Dict[str, str])
async def generate_digital_signature(
    request: DigitalSignatureRequest,
    current_user: User = Depends(require_permission("cybersecurity:sign"))
):
    """Generate digital signature."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        signature = await cybersecurity_service.generate_digital_signature(request.data)
        
        return {
            "signature": signature,
            "message": "Digital signature generated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate digital signature: {str(e)}")

@router.post("/signature/verify", response_model=Dict[str, Any])
async def verify_digital_signature(
    request: DigitalSignatureVerificationRequest,
    current_user: User = Depends(require_permission("cybersecurity:verify"))
):
    """Verify digital signature."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        is_valid = await cybersecurity_service.verify_digital_signature(
            request.data, request.signature
        )
        
        return {
            "is_valid": is_valid,
            "message": "Digital signature verification completed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify digital signature: {str(e)}")

@router.get("/threat-types", response_model=List[Dict[str, Any]])
async def get_threat_types(
    current_user: User = Depends(require_permission("cybersecurity:view"))
):
    """Get available threat types."""
    
    try:
        threat_types = [
            {
                "type": "malware",
                "name": "Malware",
                "description": "Malicious software including viruses, trojans, and worms",
                "severity": "high",
                "examples": ["virus", "trojan", "worm", "rootkit"]
            },
            {
                "type": "phishing",
                "name": "Phishing",
                "description": "Social engineering attacks to steal sensitive information",
                "severity": "medium",
                "examples": ["email_phishing", "spear_phishing", "whaling"]
            },
            {
                "type": "ddos",
                "name": "DDoS Attack",
                "description": "Distributed Denial of Service attacks",
                "severity": "high",
                "examples": ["volumetric", "protocol", "application_layer"]
            },
            {
                "type": "brute_force",
                "name": "Brute Force Attack",
                "description": "Automated attempts to guess passwords or credentials",
                "severity": "medium",
                "examples": ["password_attack", "credential_stuffing", "dictionary_attack"]
            },
            {
                "type": "sql_injection",
                "name": "SQL Injection",
                "description": "Code injection attacks targeting SQL databases",
                "severity": "high",
                "examples": ["union_based", "boolean_based", "time_based"]
            },
            {
                "type": "xss",
                "name": "Cross-Site Scripting",
                "description": "Client-side code injection attacks",
                "severity": "medium",
                "examples": ["stored_xss", "reflected_xss", "dom_xss"]
            },
            {
                "type": "data_breach",
                "name": "Data Breach",
                "description": "Unauthorized access to sensitive data",
                "severity": "critical",
                "examples": ["database_breach", "file_breach", "api_breach"]
            },
            {
                "type": "ransomware",
                "name": "Ransomware",
                "description": "Malware that encrypts data and demands ransom",
                "severity": "critical",
                "examples": ["crypto_ransomware", "locker_ransomware", "doxware"]
            }
        ]
        
        return threat_types
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get threat types: {str(e)}")

@router.get("/event-types", response_model=List[Dict[str, Any]])
async def get_event_types(
    current_user: User = Depends(require_permission("cybersecurity:view"))
):
    """Get available event types."""
    
    try:
        event_types = [
            {
                "type": "login_attempt",
                "name": "Login Attempt",
                "description": "User login attempts and authentication events",
                "severity": "low",
                "examples": ["successful_login", "failed_login", "suspicious_login"]
            },
            {
                "type": "file_access",
                "name": "File Access",
                "description": "File system access and modification events",
                "severity": "medium",
                "examples": ["file_read", "file_write", "file_delete", "file_execute"]
            },
            {
                "type": "network_connection",
                "name": "Network Connection",
                "description": "Network connection and communication events",
                "severity": "medium",
                "examples": ["connection_established", "connection_terminated", "suspicious_connection"]
            },
            {
                "type": "system_change",
                "name": "System Change",
                "description": "System configuration and state changes",
                "severity": "high",
                "examples": ["config_change", "service_start", "service_stop", "user_creation"]
            },
            {
                "type": "data_access",
                "name": "Data Access",
                "description": "Database and data access events",
                "severity": "high",
                "examples": ["database_query", "data_export", "data_import", "sensitive_data_access"]
            },
            {
                "type": "privilege_escalation",
                "name": "Privilege Escalation",
                "description": "Attempts to gain higher privileges",
                "severity": "critical",
                "examples": ["sudo_usage", "admin_access", "root_access", "privilege_change"]
            }
        ]
        
        return event_types
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get event types: {str(e)}")

@router.get("/analytics", response_model=Dict[str, Any])
async def get_cybersecurity_analytics(
    current_user: User = Depends(require_permission("cybersecurity:view"))
):
    """Get cybersecurity analytics."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        # Get security metrics
        metrics = await cybersecurity_service.get_security_metrics()
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cybersecurity analytics: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def cybersecurity_health_check():
    """Cybersecurity service health check."""
    
    cybersecurity_service = get_cybersecurity_service()
    
    try:
        # Check if service is initialized
        initialized = hasattr(cybersecurity_service, 'security_threats') and len(cybersecurity_service.security_threats) >= 0
        
        # Get service status
        status = await cybersecurity_service.get_service_status()
        
        return {
            "status": "healthy" if initialized else "initializing",
            "initialized": initialized,
            "threat_detection_enabled": status.get("threat_detection_enabled", False),
            "real_time_monitoring": status.get("real_time_monitoring", False),
            "auto_response_enabled": status.get("auto_response_enabled", False),
            "encryption_enabled": status.get("encryption_enabled", False),
            "total_threats": status.get("total_threats", 0),
            "total_events": status.get("total_events", 0),
            "total_alerts": status.get("total_alerts", 0),
            "active_policies": status.get("active_policies", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }





























#!/usr/bin/env python3
"""
Schemas Module for Video-OpusClip
API schemas and request/response models
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Literal
from pydantic import BaseModel, Field, validator, root_validator, ConfigDict, EmailStr, HttpUrl
from typing_extensions import Annotated
from datetime import datetime, timedelta
from enum import Enum

# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class ScanRequestSchema(BaseModel):
    """Schema for scan request"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=True
    )
    
    scan_type: str = Field(..., description="Type of scan to perform")
    targets: List[Dict[str, Any]] = Field(..., min_items=1, description="Targets to scan")
    timeout: Optional[float] = Field(30.0, gt=0, description="Timeout in seconds")
    max_concurrent: Optional[int] = Field(10, gt=0, le=100, description="Maximum concurrent operations")
    custom_headers: Optional[Dict[str, str]] = Field(default_factory=dict, description="Custom HTTP headers")
    user_agent: Optional[str] = Field("Video-OpusClip-Scanner/1.0", description="User agent string")
    verify_ssl: Optional[bool] = Field(True, description="Verify SSL certificates")
    
    @validator('scan_type')
    def validate_scan_type(cls, v):
        valid_types = ["port_scan", "vulnerability_scan", "web_scan", "network_scan", "comprehensive_scan"]
        if v not in valid_types:
            raise ValueError(f"Scan type must be one of: {valid_types}")
        return v
    
    @validator('targets')
    def validate_targets(cls, v):
        if not v:
            raise ValueError("At least one target must be specified")
        
        for target in v:
            if 'host' not in target:
                raise ValueError("Each target must have a 'host' field")
            if not target['host'] or len(target['host'].strip()) == 0:
                raise ValueError("Host cannot be empty")
        
        return v

class EnumerationRequestSchema(BaseModel):
    """Schema for enumeration request"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=True
    )
    
    enumeration_type: str = Field(..., description="Type of enumeration to perform")
    target: Dict[str, Any] = Field(..., description="Target to enumerate")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Enumeration options")
    timeout: Optional[float] = Field(30.0, gt=0, description="Timeout in seconds")
    
    @validator('enumeration_type')
    def validate_enumeration_type(cls, v):
        valid_types = ["dns_enumeration", "smb_enumeration", "ssh_enumeration", "user_enumeration", "service_enumeration"]
        if v not in valid_types:
            raise ValueError(f"Enumeration type must be one of: {valid_types}")
        return v
    
    @validator('target')
    def validate_target(cls, v):
        if 'host' not in v:
            raise ValueError("Target must have a 'host' field")
        if not v['host'] or len(v['host'].strip()) == 0:
            raise ValueError("Host cannot be empty")
        return v

class AttackRequestSchema(BaseModel):
    """Schema for attack request"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=True
    )
    
    attack_type: str = Field(..., description="Type of attack to perform")
    target: Dict[str, Any] = Field(..., description="Target to attack")
    credentials: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="Credentials to test")
    payloads: Optional[List[str]] = Field(default_factory=list, description="Attack payloads")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Attack options")
    timeout: Optional[float] = Field(30.0, gt=0, description="Timeout in seconds")
    max_attempts: Optional[int] = Field(1000, gt=0, description="Maximum attack attempts")
    
    @validator('attack_type')
    def validate_attack_type(cls, v):
        valid_types = ["brute_force", "exploitation", "social_engineering", "phishing", "denial_of_service"]
        if v not in valid_types:
            raise ValueError(f"Attack type must be one of: {valid_types}")
        return v
    
    @validator('target')
    def validate_target(cls, v):
        if 'host' not in v:
            raise ValueError("Target must have a 'host' field")
        if not v['host'] or len(v['host'].strip()) == 0:
            raise ValueError("Host cannot be empty")
        return v

class UserRegistrationSchema(BaseModel):
    """Schema for user registration"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True
    )
    
    username: Annotated[str, Field(min_length=3, max_length=50, regex=r"^[a-zA-Z0-9_]+$")] = Field(..., description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: Annotated[str, Field(min_length=8, max_length=128)] = Field(..., description="Password")
    confirm_password: Annotated[str, Field(min_length=8, max_length=128)] = Field(..., description="Password confirmation")
    first_name: Optional[str] = Field(None, max_length=50, description="First name")
    last_name: Optional[str] = Field(None, max_length=50, description="Last name")
    organization: Optional[str] = Field(None, max_length=100, description="Organization")
    role: Optional[str] = Field("user", description="User role")
    
    @root_validator
    def validate_passwords_match(cls, values):
        password = values.get("password")
        confirm_password = values.get("confirm_password")
        
        if password and confirm_password and password != confirm_password:
            raise ValueError("Passwords do not match")
        
        return values
    
    @validator('password')
    def validate_password_strength(cls, v):
        import re
        
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        if not re.search(r'[A-Z]', v):
            raise ValueError("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', v):
            raise ValueError("Password must contain at least one lowercase letter")
        
        if not re.search(r'\d', v):
            raise ValueError("Password must contain at least one digit")
        
        if not re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', v):
            raise ValueError("Password must contain at least one special character")
        
        return v

class UserLoginSchema(BaseModel):
    """Schema for user login"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True
    )
    
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")
    remember_me: Optional[bool] = Field(False, description="Remember login session")
    two_factor_code: Optional[str] = Field(None, description="Two-factor authentication code")

class PasswordChangeSchema(BaseModel):
    """Schema for password change"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True
    )
    
    current_password: str = Field(..., description="Current password")
    new_password: Annotated[str, Field(min_length=8, max_length=128)] = Field(..., description="New password")
    confirm_new_password: Annotated[str, Field(min_length=8, max_length=128)] = Field(..., description="Confirm new password")
    
    @root_validator
    def validate_passwords_match(cls, values):
        new_password = values.get("new_password")
        confirm_new_password = values.get("confirm_new_password")
        
        if new_password and confirm_new_password and new_password != confirm_new_password:
            raise ValueError("New passwords do not match")
        
        return values

class FileUploadSchema(BaseModel):
    """Schema for file upload"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    filename: str = Field(..., description="File name")
    file_type: str = Field(..., description="File type")
    file_size: int = Field(..., gt=0, description="File size in bytes")
    checksum: Optional[str] = Field(None, description="File checksum")
    description: Optional[str] = Field(None, description="File description")
    tags: Optional[List[str]] = Field(default_factory=list, description="File tags")
    
    @validator('file_size')
    def validate_file_size(cls, v):
        max_size = 100 * 1024 * 1024  # 100MB
        if v > max_size:
            raise ValueError(f"File size cannot exceed {max_size} bytes")
        return v
    
    @validator('file_type')
    def validate_file_type(cls, v):
        allowed_types = ["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm", "m4v"]
        if v.lower() not in allowed_types:
            raise ValueError(f"File type must be one of: {allowed_types}")
        return v.lower()

class ConfigurationUpdateSchema(BaseModel):
    """Schema for configuration update"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    section: str = Field(..., description="Configuration section")
    key: str = Field(..., description="Configuration key")
    value: Any = Field(..., description="Configuration value")
    description: Optional[str] = Field(None, description="Configuration description")
    
    @validator('section')
    def validate_section(cls, v):
        valid_sections = ["database", "security", "network", "application", "logging"]
        if v not in valid_sections:
            raise ValueError(f"Section must be one of: {valid_sections}")
        return v

# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class BaseResponseSchema(BaseModel):
    """Base response schema"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True
    )
    
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")

class ErrorResponseSchema(BaseResponseSchema):
    """Error response schema"""
    success: Literal[False] = False
    error_code: str = Field(..., description="Error code")
    error_type: str = Field(..., description="Error type")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    stack_trace: Optional[str] = Field(None, description="Stack trace (in debug mode)")

class SuccessResponseSchema(BaseResponseSchema):
    """Success response schema"""
    success: Literal[True] = True
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Response metadata")

class ScanResponseSchema(SuccessResponseSchema):
    """Scan response schema"""
    data: Dict[str, Any] = Field(..., description="Scan results")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Scan completed successfully",
                "timestamp": "2024-01-01T12:00:00Z",
                "data": {
                    "scan_id": "scan_12345",
                    "scan_type": "port_scan",
                    "target": "192.168.1.100",
                    "status": "completed",
                    "ports_scanned": 1000,
                    "ports_open": 5,
                    "vulnerabilities_found": 2,
                    "duration": 45.2
                }
            }
        }

class EnumerationResponseSchema(SuccessResponseSchema):
    """Enumeration response schema"""
    data: Dict[str, Any] = Field(..., description="Enumeration results")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Enumeration completed successfully",
                "timestamp": "2024-01-01T12:00:00Z",
                "data": {
                    "enumeration_id": "enum_12345",
                    "enumeration_type": "dns_enumeration",
                    "target": "example.com",
                    "status": "completed",
                    "dns_records": 10,
                    "subdomains": 5,
                    "duration": 12.5
                }
            }
        }

class AttackResponseSchema(SuccessResponseSchema):
    """Attack response schema"""
    data: Dict[str, Any] = Field(..., description="Attack results")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Attack completed successfully",
                "timestamp": "2024-01-01T12:00:00Z",
                "data": {
                    "attack_id": "attack_12345",
                    "attack_type": "brute_force",
                    "target": "192.168.1.100",
                    "status": "completed",
                    "credentials_found": 2,
                    "attempts_made": 1000,
                    "duration": 180.5
                }
            }
        }

class UserResponseSchema(SuccessResponseSchema):
    """User response schema"""
    data: Dict[str, Any] = Field(..., description="User information")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "User operation completed successfully",
                "timestamp": "2024-01-01T12:00:00Z",
                "data": {
                    "user_id": "user_12345",
                    "username": "john_doe",
                    "email": "john@example.com",
                    "role": "user",
                    "created_at": "2024-01-01T10:00:00Z",
                    "last_login": "2024-01-01T11:30:00Z"
                }
            }
        }

class AuthenticationResponseSchema(SuccessResponseSchema):
    """Authentication response schema"""
    data: Dict[str, Any] = Field(..., description="Authentication information")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Authentication successful",
                "timestamp": "2024-01-01T12:00:00Z",
                "data": {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "bearer",
                    "expires_in": 3600,
                    "user": {
                        "user_id": "user_12345",
                        "username": "john_doe",
                        "email": "john@example.com",
                        "role": "user"
                    }
                }
            }
        }

class PaginatedResponseSchema(BaseResponseSchema):
    """Paginated response schema"""
    success: Literal[True] = True
    data: Dict[str, Any] = Field(..., description="Paginated data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Data retrieved successfully",
                "timestamp": "2024-01-01T12:00:00Z",
                "data": {
                    "items": [],
                    "total": 100,
                    "page": 1,
                    "per_page": 10,
                    "total_pages": 10,
                    "has_next": True,
                    "has_prev": False
                }
            }
        }

# ============================================================================
# QUERY PARAMETER SCHEMAS
# ============================================================================

class PaginationQuerySchema(BaseModel):
    """Schema for pagination query parameters"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    page: Optional[int] = Field(1, ge=1, description="Page number")
    per_page: Optional[int] = Field(10, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field("created_at", description="Sort field")
    sort_order: Optional[Literal["asc", "desc"]] = Field("desc", description="Sort order")

class ScanQuerySchema(PaginationQuerySchema):
    """Schema for scan query parameters"""
    scan_type: Optional[str] = Field(None, description="Filter by scan type")
    target: Optional[str] = Field(None, description="Filter by target")
    status: Optional[str] = Field(None, description="Filter by status")
    start_date: Optional[datetime] = Field(None, description="Filter by start date")
    end_date: Optional[datetime] = Field(None, description="Filter by end date")

class UserQuerySchema(PaginationQuerySchema):
    """Schema for user query parameters"""
    username: Optional[str] = Field(None, description="Filter by username")
    email: Optional[str] = Field(None, description="Filter by email")
    role: Optional[str] = Field(None, description="Filter by role")
    status: Optional[str] = Field(None, description="Filter by status")

class SecurityQuerySchema(PaginationQuerySchema):
    """Schema for security query parameters"""
    severity: Optional[str] = Field(None, description="Filter by severity")
    category: Optional[str] = Field(None, description="Filter by category")
    status: Optional[str] = Field(None, description="Filter by status")
    cve_id: Optional[str] = Field(None, description="Filter by CVE ID")

# ============================================================================
# FILTER SCHEMAS
# ============================================================================

class DateRangeFilterSchema(BaseModel):
    """Schema for date range filtering"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    start_date: Optional[datetime] = Field(None, description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date")
    
    @root_validator
    def validate_date_range(cls, values):
        start_date = values.get("start_date")
        end_date = values.get("end_date")
        
        if start_date and end_date and start_date > end_date:
            raise ValueError("Start date cannot be after end date")
        
        return values

class SeverityFilterSchema(BaseModel):
    """Schema for severity filtering"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    min_severity: Optional[str] = Field(None, description="Minimum severity level")
    max_severity: Optional[str] = Field(None, description="Maximum severity level")
    include_info: Optional[bool] = Field(True, description="Include info level findings")
    
    @validator('min_severity', 'max_severity')
    def validate_severity(cls, v):
        if v is not None:
            valid_severities = ["info", "low", "medium", "high", "critical"]
            if v not in valid_severities:
                raise ValueError(f"Severity must be one of: {valid_severities}")
        return v

class TargetFilterSchema(BaseModel):
    """Schema for target filtering"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    hosts: Optional[List[str]] = Field(None, description="Filter by specific hosts")
    ports: Optional[List[int]] = Field(None, description="Filter by specific ports")
    protocols: Optional[List[str]] = Field(None, description="Filter by protocols")
    domains: Optional[List[str]] = Field(None, description="Filter by domains")
    
    @validator('ports')
    def validate_ports(cls, v):
        if v is not None:
            for port in v:
                if port < 1 or port > 65535:
                    raise ValueError("Port must be between 1 and 65535")
        return v

# ============================================================================
# EXPORT SCHEMAS
# ============================================================================

class ExportRequestSchema(BaseModel):
    """Schema for export request"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    export_type: str = Field(..., description="Type of export")
    format: str = Field(..., description="Export format")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Export filters")
    include_metadata: Optional[bool] = Field(True, description="Include metadata in export")
    compression: Optional[bool] = Field(False, description="Compress export file")
    
    @validator('export_type')
    def validate_export_type(cls, v):
        valid_types = ["scan_results", "enumeration_results", "attack_results", "security_findings", "user_data"]
        if v not in valid_types:
            raise ValueError(f"Export type must be one of: {valid_types}")
        return v
    
    @validator('format')
    def validate_format(cls, v):
        valid_formats = ["json", "csv", "xml", "pdf", "html"]
        if v not in valid_formats:
            raise ValueError(f"Export format must be one of: {valid_formats}")
        return v

class ExportResponseSchema(SuccessResponseSchema):
    """Schema for export response"""
    data: Dict[str, Any] = Field(..., description="Export information")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Export completed successfully",
                "timestamp": "2024-01-01T12:00:00Z",
                "data": {
                    "export_id": "export_12345",
                    "export_type": "scan_results",
                    "format": "json",
                    "file_size": 1024000,
                    "download_url": "/api/exports/export_12345/download",
                    "expires_at": "2024-01-02T12:00:00Z"
                }
            }
        }

# ============================================================================
# WEBHOOK SCHEMAS
# ============================================================================

class WebhookRequestSchema(BaseModel):
    """Schema for webhook request"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    url: HttpUrl = Field(..., description="Webhook URL")
    events: List[str] = Field(..., min_items=1, description="Events to subscribe to")
    secret: Optional[str] = Field(None, description="Webhook secret for verification")
    description: Optional[str] = Field(None, description="Webhook description")
    active: Optional[bool] = Field(True, description="Whether webhook is active")
    
    @validator('events')
    def validate_events(cls, v):
        valid_events = [
            "scan.completed", "scan.failed", "enumeration.completed", "enumeration.failed",
            "attack.completed", "attack.failed", "vulnerability.found", "user.created",
            "user.updated", "user.deleted", "security.alert"
        ]
        
        for event in v:
            if event not in valid_events:
                raise ValueError(f"Event must be one of: {valid_events}")
        
        return v

class WebhookPayloadSchema(BaseModel):
    """Schema for webhook payload"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    event: str = Field(..., description="Event type")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    data: Dict[str, Any] = Field(..., description="Event data")
    webhook_id: str = Field(..., description="Webhook identifier")
    signature: Optional[str] = Field(None, description="Webhook signature")

# ============================================================================
# NOTIFICATION SCHEMAS
# ============================================================================

class NotificationRequestSchema(BaseModel):
    """Schema for notification request"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    type: str = Field(..., description="Notification type")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    recipients: List[str] = Field(..., min_items=1, description="Recipient list")
    priority: Optional[str] = Field("normal", description="Notification priority")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional notification data")
    
    @validator('type')
    def validate_type(cls, v):
        valid_types = ["email", "sms", "push", "webhook", "slack"]
        if v not in valid_types:
            raise ValueError(f"Notification type must be one of: {valid_types}")
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ["low", "normal", "high", "urgent"]
        if v not in valid_priorities:
            raise ValueError(f"Priority must be one of: {valid_priorities}")
        return v

class NotificationResponseSchema(SuccessResponseSchema):
    """Schema for notification response"""
    data: Dict[str, Any] = Field(..., description="Notification information")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Notification sent successfully",
                "timestamp": "2024-01-01T12:00:00Z",
                "data": {
                    "notification_id": "notif_12345",
                    "type": "email",
                    "status": "sent",
                    "recipients": ["user@example.com"],
                    "sent_at": "2024-01-01T12:00:00Z"
                }
            }
        }

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_api_key(api_key: str) -> bool:
    """Validate API key format"""
    import re
    # API key should be 32-64 characters, alphanumeric with optional hyphens
    pattern = r'^[a-zA-Z0-9\-]{32,64}$'
    return bool(re.match(pattern, api_key))

def validate_jwt_token(token: str) -> bool:
    """Validate JWT token format"""
    import re
    # JWT token has three parts separated by dots
    pattern = r'^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*$'
    return bool(re.match(pattern, token))

def validate_uuid(uuid_str: str) -> bool:
    """Validate UUID format"""
    import re
    pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(pattern, uuid_str.lower()))

# ============================================================================
# SCHEMA UTILITIES
# ============================================================================

def create_error_response(error_code: str, message: str, error_type: str = "validation_error", details: Optional[Dict[str, Any]] = None) -> ErrorResponseSchema:
    """Create a standardized error response"""
    return ErrorResponseSchema(
        success=False,
        message=message,
        error_code=error_code,
        error_type=error_type,
        error_details=details
    )

def create_success_response(data: Optional[Dict[str, Any]] = None, message: str = "Operation completed successfully", metadata: Optional[Dict[str, Any]] = None) -> SuccessResponseSchema:
    """Create a standardized success response"""
    return SuccessResponseSchema(
        success=True,
        message=message,
        data=data,
        metadata=metadata
    )

def create_paginated_response(items: List[Any], total: int, page: int, per_page: int, message: str = "Data retrieved successfully") -> PaginatedResponseSchema:
    """Create a standardized paginated response"""
    total_pages = (total + per_page - 1) // per_page
    has_next = page < total_pages
    has_prev = page > 1
    
    return PaginatedResponseSchema(
        success=True,
        message=message,
        data={
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev
        }
    )

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of schemas
    print("📋 Video-OpusClip Schemas Example")
    
    # Create scan request
    scan_request = ScanRequestSchema(
        scan_type="port_scan",
        targets=[
            {"host": "192.168.1.100", "port": 80, "protocol": "tcp"},
            {"host": "192.168.1.101", "port": 443, "protocol": "tcp"}
        ],
        timeout=30.0,
        max_concurrent=5
    )
    print(f"Scan request: {scan_request}")
    
    # Create user registration
    user_registration = UserRegistrationSchema(
        username="john_doe",
        email="john@example.com",
        password="SecureP@ssw0rd123",
        confirm_password="SecureP@ssw0rd123",
        first_name="John",
        last_name="Doe"
    )
    print(f"User registration: {user_registration}")
    
    # Create error response
    error_response = create_error_response(
        error_code="VALIDATION_ERROR",
        message="Invalid input data",
        error_type="validation_error",
        details={"field": "email", "issue": "Invalid email format"}
    )
    print(f"Error response: {error_response}")
    
    # Create success response
    success_response = create_success_response(
        data={"user_id": "12345", "username": "john_doe"},
        message="User created successfully"
    )
    print(f"Success response: {success_response}")
    
    # Create paginated response
    paginated_response = create_paginated_response(
        items=[{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}],
        total=100,
        page=1,
        per_page=10
    )
    print(f"Paginated response: {paginated_response}")
    
    # Validate formats
    print(f"API key validation: {validate_api_key('abc123-def456-ghi789-jkl012-mno345-pqr678')}")
    print(f"JWT validation: {validate_jwt_token('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c')}")
    print(f"UUID validation: {validate_uuid('550e8400-e29b-41d4-a716-446655440000')}") 
#!/usr/bin/env python3
"""
Custom Exceptions for Video-OpusClip
Defines all custom exceptions used throughout the system
"""

from typing import Optional, Dict, Any, List
from datetime import datetime


class VideoOpusClipException(Exception):
    """Base exception for all Video-OpusClip errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "GENERAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = timestamp or datetime.utcnow()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "exception_type": self.__class__.__name__
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message} (Code: {self.error_code})"


# ============================================================================
# VALIDATION EXCEPTIONS
# ============================================================================

class ValidationError(VideoOpusClipException):
    """Base exception for validation errors"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field
        self.value = value
        self.expected_type = expected_type
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "field": self.field,
            "value": str(self.value) if self.value is not None else None,
            "expected_type": self.expected_type
        })
        return base_dict


class InputValidationError(ValidationError):
    """Exception for input validation errors"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        validation_rules: Optional[List[str]] = None
    ):
        super().__init__(message, field, value, "input_validation")
        self.validation_rules = validation_rules or []
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["validation_rules"] = self.validation_rules
        return base_dict


class SchemaValidationError(ValidationError):
    """Exception for schema validation errors"""
    
    def __init__(
        self,
        message: str,
        schema_name: Optional[str] = None,
        field_errors: Optional[Dict[str, List[str]]] = None
    ):
        super().__init__(message, None, None, "schema_validation")
        self.schema_name = schema_name
        self.field_errors = field_errors or {}
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "schema_name": self.schema_name,
            "field_errors": self.field_errors
        })
        return base_dict


class TypeValidationError(ValidationError):
    """Exception for type validation errors"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        actual_type: Optional[str] = None
    ):
        super().__init__(message, field, value, expected_type)
        self.actual_type = actual_type
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["actual_type"] = self.actual_type
        return base_dict


# ============================================================================
# SECURITY EXCEPTIONS
# ============================================================================

class SecurityError(VideoOpusClipException):
    """Base exception for security-related errors"""
    
    def __init__(
        self,
        message: str,
        security_level: str = "medium",
        threat_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "SECURITY_ERROR", details)
        self.security_level = security_level
        self.threat_type = threat_type
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "security_level": self.security_level,
            "threat_type": self.threat_type
        })
        return base_dict


class AuthenticationError(SecurityError):
    """Exception for authentication failures"""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        auth_method: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ):
        super().__init__(message, "high", "authentication_failure")
        self.auth_method = auth_method
        self.user_id = user_id
        self.ip_address = ip_address
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "auth_method": self.auth_method,
            "user_id": self.user_id,
            "ip_address": self.ip_address
        })
        return base_dict


class AuthorizationError(SecurityError):
    """Exception for authorization failures"""
    
    def __init__(
        self,
        message: str = "Access denied",
        required_permission: Optional[str] = None,
        user_permissions: Optional[List[str]] = None,
        resource: Optional[str] = None
    ):
        super().__init__(message, "high", "authorization_failure")
        self.required_permission = required_permission
        self.user_permissions = user_permissions or []
        self.resource = resource
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "required_permission": self.required_permission,
            "user_permissions": self.user_permissions,
            "resource": self.resource
        })
        return base_dict


class RateLimitError(SecurityError):
    """Exception for rate limiting violations"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit_type: Optional[str] = None,
        current_count: Optional[int] = None,
        limit_value: Optional[int] = None,
        reset_time: Optional[datetime] = None
    ):
        super().__init__(message, "medium", "rate_limit_violation")
        self.limit_type = limit_type
        self.current_count = current_count
        self.limit_value = limit_value
        self.reset_time = reset_time
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "limit_type": self.limit_type,
            "current_count": self.current_count,
            "limit_value": self.limit_value,
            "reset_time": self.reset_time.isoformat() if self.reset_time else None
        })
        return base_dict


class IntrusionDetectionError(SecurityError):
    """Exception for intrusion detection alerts"""
    
    def __init__(
        self,
        message: str,
        threat_level: str = "high",
        attack_type: Optional[str] = None,
        source_ip: Optional[str] = None,
        indicators: Optional[List[str]] = None
    ):
        super().__init__(message, threat_level, "intrusion_detection")
        self.attack_type = attack_type
        self.source_ip = source_ip
        self.indicators = indicators or []
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "attack_type": self.attack_type,
            "source_ip": self.source_ip,
            "indicators": self.indicators
        })
        return base_dict


# ============================================================================
# SCANNING EXCEPTIONS
# ============================================================================

class ScanningError(VideoOpusClipException):
    """Base exception for scanning-related errors"""
    
    def __init__(
        self,
        message: str,
        scan_type: Optional[str] = None,
        target: Optional[str] = None,
        scan_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "SCANNING_ERROR", details)
        self.scan_type = scan_type
        self.target = target
        self.scan_id = scan_id
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "scan_type": self.scan_type,
            "target": self.target,
            "scan_id": self.scan_id
        })
        return base_dict


class PortScanError(ScanningError):
    """Exception for port scanning errors"""
    
    def __init__(
        self,
        message: str,
        target: Optional[str] = None,
        port: Optional[int] = None,
        protocol: Optional[str] = None,
        scan_id: Optional[str] = None
    ):
        super().__init__(message, "port_scan", target, scan_id)
        self.port = port
        self.protocol = protocol
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "port": self.port,
            "protocol": self.protocol
        })
        return base_dict


class VulnerabilityScanError(ScanningError):
    """Exception for vulnerability scanning errors"""
    
    def __init__(
        self,
        message: str,
        target: Optional[str] = None,
        vulnerability_type: Optional[str] = None,
        scan_id: Optional[str] = None
    ):
        super().__init__(message, "vulnerability_scan", target, scan_id)
        self.vulnerability_type = vulnerability_type
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["vulnerability_type"] = self.vulnerability_type
        return base_dict


class WebScanError(ScanningError):
    """Exception for web scanning errors"""
    
    def __init__(
        self,
        message: str,
        target: Optional[str] = None,
        url: Optional[str] = None,
        http_status: Optional[int] = None,
        scan_id: Optional[str] = None
    ):
        super().__init__(message, "web_scan", target, scan_id)
        self.url = url
        self.http_status = http_status
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "url": self.url,
            "http_status": self.http_status
        })
        return base_dict


# ============================================================================
# ENUMERATION EXCEPTIONS
# ============================================================================

class EnumerationError(VideoOpusClipException):
    """Base exception for enumeration-related errors"""
    
    def __init__(
        self,
        message: str,
        enumeration_type: Optional[str] = None,
        target: Optional[str] = None,
        enumeration_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "ENUMERATION_ERROR", details)
        self.enumeration_type = enumeration_type
        self.target = target
        self.enumeration_id = enumeration_id
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "enumeration_type": self.enumeration_type,
            "target": self.target,
            "enumeration_id": self.enumeration_id
        })
        return base_dict


class DNSEnumerationError(EnumerationError):
    """Exception for DNS enumeration errors"""
    
    def __init__(
        self,
        message: str,
        target: Optional[str] = None,
        record_type: Optional[str] = None,
        enumeration_id: Optional[str] = None
    ):
        super().__init__(message, "dns_enumeration", target, enumeration_id)
        self.record_type = record_type
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["record_type"] = self.record_type
        return base_dict


class SMBEnumerationError(EnumerationError):
    """Exception for SMB enumeration errors"""
    
    def __init__(
        self,
        message: str,
        target: Optional[str] = None,
        share_name: Optional[str] = None,
        enumeration_id: Optional[str] = None
    ):
        super().__init__(message, "smb_enumeration", target, enumeration_id)
        self.share_name = share_name
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["share_name"] = self.share_name
        return base_dict


class SSHEnumerationError(EnumerationError):
    """Exception for SSH enumeration errors"""
    
    def __init__(
        self,
        message: str,
        target: Optional[str] = None,
        port: Optional[int] = None,
        enumeration_id: Optional[str] = None
    ):
        super().__init__(message, "ssh_enumeration", target, enumeration_id)
        self.port = port
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["port"] = self.port
        return base_dict


# ============================================================================
# ATTACK EXCEPTIONS
# ============================================================================

class AttackError(VideoOpusClipException):
    """Base exception for attack-related errors"""
    
    def __init__(
        self,
        message: str,
        attack_type: Optional[str] = None,
        target: Optional[str] = None,
        attack_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "ATTACK_ERROR", details)
        self.attack_type = attack_type
        self.target = target
        self.attack_id = attack_id
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "attack_type": self.attack_type,
            "target": self.target,
            "attack_id": self.attack_id
        })
        return base_dict


class BruteForceError(AttackError):
    """Exception for brute force attack errors"""
    
    def __init__(
        self,
        message: str,
        target: Optional[str] = None,
        service: Optional[str] = None,
        attack_id: Optional[str] = None,
        attempts_made: Optional[int] = None
    ):
        super().__init__(message, "brute_force", target, attack_id)
        self.service = service
        self.attempts_made = attempts_made
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "service": self.service,
            "attempts_made": self.attempts_made
        })
        return base_dict


class ExploitationError(AttackError):
    """Exception for exploitation attack errors"""
    
    def __init__(
        self,
        message: str,
        target: Optional[str] = None,
        exploit_type: Optional[str] = None,
        payload: Optional[str] = None,
        attack_id: Optional[str] = None
    ):
        super().__init__(message, "exploitation", target, attack_id)
        self.exploit_type = exploit_type
        self.payload = payload
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "exploit_type": self.exploit_type,
            "payload": self.payload
        })
        return base_dict


# ============================================================================
# DATABASE EXCEPTIONS
# ============================================================================

class DatabaseError(VideoOpusClipException):
    """Base exception for database-related errors"""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "DATABASE_ERROR", details)
        self.operation = operation
        self.table = table
        self.query = query
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "operation": self.operation,
            "table": self.table,
            "query": self.query
        })
        return base_dict


class ConnectionError(DatabaseError):
    """Exception for database connection errors"""
    
    def __init__(
        self,
        message: str = "Database connection failed",
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None
    ):
        super().__init__(message, "connect")
        self.host = host
        self.port = port
        self.database = database
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "host": self.host,
            "port": self.port,
            "database": self.database
        })
        return base_dict


class QueryError(DatabaseError):
    """Exception for database query errors"""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        sql_error: Optional[str] = None
    ):
        super().__init__(message, operation, table, query)
        self.sql_error = sql_error
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["sql_error"] = self.sql_error
        return base_dict


class TransactionError(DatabaseError):
    """Exception for database transaction errors"""
    
    def __init__(
        self,
        message: str,
        transaction_id: Optional[str] = None,
        rollback_successful: bool = False
    ):
        super().__init__(message, "transaction")
        self.transaction_id = transaction_id
        self.rollback_successful = rollback_successful
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "transaction_id": self.transaction_id,
            "rollback_successful": self.rollback_successful
        })
        return base_dict


# ============================================================================
# NETWORK EXCEPTIONS
# ============================================================================

class NetworkError(VideoOpusClipException):
    """Base exception for network-related errors"""
    
    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        protocol: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "NETWORK_ERROR", details)
        self.host = host
        self.port = port
        self.protocol = protocol
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol
        })
        return base_dict


class ConnectionTimeoutError(NetworkError):
    """Exception for connection timeout errors"""
    
    def __init__(
        self,
        message: str = "Connection timeout",
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[float] = None
    ):
        super().__init__(message, host, port)
        self.timeout = timeout
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["timeout"] = self.timeout
        return base_dict


class DNSResolutionError(NetworkError):
    """Exception for DNS resolution errors"""
    
    def __init__(
        self,
        message: str,
        domain: Optional[str] = None,
        record_type: Optional[str] = None
    ):
        super().__init__(message)
        self.domain = domain
        self.record_type = record_type
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "domain": self.domain,
            "record_type": self.record_type
        })
        return base_dict


class HTTPError(NetworkError):
    """Exception for HTTP-related errors"""
    
    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        method: Optional[str] = None
    ):
        super().__init__(message)
        self.url = url
        self.status_code = status_code
        self.method = method
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "url": self.url,
            "status_code": self.status_code,
            "method": self.method
        })
        return base_dict


# ============================================================================
# FILE SYSTEM EXCEPTIONS
# ============================================================================

class FileSystemError(VideoOpusClipException):
    """Base exception for file system-related errors"""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "FILE_SYSTEM_ERROR", details)
        self.file_path = file_path
        self.operation = operation
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "file_path": self.file_path,
            "operation": self.operation
        })
        return base_dict


class FileNotFoundError(FileSystemError):
    """Exception for file not found errors"""
    
    def __init__(
        self,
        message: str = "File not found",
        file_path: Optional[str] = None
    ):
        super().__init__(message, file_path, "read")


class FilePermissionError(FileSystemError):
    """Exception for file permission errors"""
    
    def __init__(
        self,
        message: str = "Permission denied",
        file_path: Optional[str] = None,
        required_permission: Optional[str] = None
    ):
        super().__init__(message, file_path, "access")
        self.required_permission = required_permission
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["required_permission"] = self.required_permission
        return base_dict


class FileSizeError(FileSystemError):
    """Exception for file size errors"""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_size: Optional[int] = None,
        max_size: Optional[int] = None
    ):
        super().__init__(message, file_path, "size_check")
        self.file_size = file_size
        self.max_size = max_size
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "file_size": self.file_size,
            "max_size": self.max_size
        })
        return base_dict


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================

class ConfigurationError(VideoOpusClipException):
    """Base exception for configuration-related errors"""
    
    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "CONFIGURATION_ERROR", details)
        self.config_section = config_section
        self.config_key = config_key
        self.config_value = config_value
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "config_section": self.config_section,
            "config_key": self.config_key,
            "config_value": str(self.config_value) if self.config_value is not None else None
        })
        return base_dict


class MissingConfigurationError(ConfigurationError):
    """Exception for missing configuration errors"""
    
    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        config_key: Optional[str] = None
    ):
        super().__init__(message, config_section, config_key)


class InvalidConfigurationError(ConfigurationError):
    """Exception for invalid configuration errors"""
    
    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        expected_format: Optional[str] = None
    ):
        super().__init__(message, config_section, config_key, config_value)
        self.expected_format = expected_format
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["expected_format"] = self.expected_format
        return base_dict


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_exception_from_dict(exception_data: Dict[str, Any]) -> VideoOpusClipException:
    """Create an exception instance from dictionary data"""
    exception_type = exception_data.get("exception_type", "VideoOpusClipException")
    message = exception_data.get("message", "Unknown error")
    error_code = exception_data.get("error_code", "GENERAL_ERROR")
    details = exception_data.get("details", {})
    timestamp_str = exception_data.get("timestamp")
    
    timestamp = None
    if timestamp_str:
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except ValueError:
            pass
    
    # Map exception types to classes
    exception_classes = {
        "ValidationError": ValidationError,
        "InputValidationError": InputValidationError,
        "SchemaValidationError": SchemaValidationError,
        "TypeValidationError": TypeValidationError,
        "SecurityError": SecurityError,
        "AuthenticationError": AuthenticationError,
        "AuthorizationError": AuthorizationError,
        "RateLimitError": RateLimitError,
        "IntrusionDetectionError": IntrusionDetectionError,
        "ScanningError": ScanningError,
        "PortScanError": PortScanError,
        "VulnerabilityScanError": VulnerabilityScanError,
        "WebScanError": WebScanError,
        "EnumerationError": EnumerationError,
        "DNSEnumerationError": DNSEnumerationError,
        "SMBEnumerationError": SMBEnumerationError,
        "SSHEnumerationError": SSHEnumerationError,
        "AttackError": AttackError,
        "BruteForceError": BruteForceError,
        "ExploitationError": ExploitationError,
        "DatabaseError": DatabaseError,
        "ConnectionError": ConnectionError,
        "QueryError": QueryError,
        "TransactionError": TransactionError,
        "NetworkError": NetworkError,
        "ConnectionTimeoutError": ConnectionTimeoutError,
        "DNSResolutionError": DNSResolutionError,
        "HTTPError": HTTPError,
        "FileSystemError": FileSystemError,
        "FileNotFoundError": FileNotFoundError,
        "FilePermissionError": FilePermissionError,
        "FileSizeError": FileSizeError,
        "ConfigurationError": ConfigurationError,
        "MissingConfigurationError": MissingConfigurationError,
        "InvalidConfigurationError": InvalidConfigurationError
    }
    
    exception_class = exception_classes.get(exception_type, VideoOpusClipException)
    
    # Create exception with appropriate parameters
    if exception_class == VideoOpusClipException:
        return exception_class(message, error_code, details, timestamp)
    else:
        # For specific exception types, we need to handle their specific parameters
        # This is a simplified approach - in practice, you might need more complex logic
        return exception_class(message)


def get_exception_hierarchy() -> Dict[str, List[str]]:
    """Get the hierarchy of custom exceptions"""
    return {
        "VideoOpusClipException": [
            "ValidationError",
            "SecurityError",
            "ScanningError",
            "EnumerationError",
            "AttackError",
            "DatabaseError",
            "NetworkError",
            "FileSystemError",
            "ConfigurationError"
        ],
        "ValidationError": [
            "InputValidationError",
            "SchemaValidationError",
            "TypeValidationError"
        ],
        "SecurityError": [
            "AuthenticationError",
            "AuthorizationError",
            "RateLimitError",
            "IntrusionDetectionError"
        ],
        "ScanningError": [
            "PortScanError",
            "VulnerabilityScanError",
            "WebScanError"
        ],
        "EnumerationError": [
            "DNSEnumerationError",
            "SMBEnumerationError",
            "SSHEnumerationError"
        ],
        "AttackError": [
            "BruteForceError",
            "ExploitationError"
        ],
        "DatabaseError": [
            "ConnectionError",
            "QueryError",
            "TransactionError"
        ],
        "NetworkError": [
            "ConnectionTimeoutError",
            "DNSResolutionError",
            "HTTPError"
        ],
        "FileSystemError": [
            "FileNotFoundError",
            "FilePermissionError",
            "FileSizeError"
        ],
        "ConfigurationError": [
            "MissingConfigurationError",
            "InvalidConfigurationError"
        ]
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage of custom exceptions
    print("🚨 Custom Exceptions Example")
    
    # Create a validation error
    try:
        raise InputValidationError(
            message="Invalid email format",
            field="email",
            value="invalid-email",
            validation_rules=["must be valid email format"]
        )
    except InputValidationError as e:
        print(f"Validation Error: {e}")
        print(f"Error Details: {e.to_dict()}")
    
    # Create a security error
    try:
        raise AuthenticationError(
            message="Invalid credentials",
            auth_method="password",
            user_id="user123",
            ip_address="192.168.1.100"
        )
    except AuthenticationError as e:
        print(f"Authentication Error: {e}")
        print(f"Error Details: {e.to_dict()}")
    
    # Create a scanning error
    try:
        raise PortScanError(
            message="Connection refused",
            target="192.168.1.100",
            port=80,
            protocol="tcp",
            scan_id="scan_12345"
        )
    except PortScanError as e:
        print(f"Port Scan Error: {e}")
        print(f"Error Details: {e.to_dict()}")
    
    # Test exception hierarchy
    hierarchy = get_exception_hierarchy()
    print(f"Exception Hierarchy: {hierarchy}")
    
    # Test exception creation from dict
    exception_data = {
        "exception_type": "ValidationError",
        "message": "Test error",
        "error_code": "TEST_ERROR",
        "details": {"test": "data"},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    exception = create_exception_from_dict(exception_data)
    print(f"Created Exception: {exception}")
    print(f"Exception Type: {type(exception)}") 
"""
Types Module for Video-OpusClip
Data models, schemas, and type definitions
"""

from .models import (
    # Enums
    ScanType, EnumerationType, AttackType, SeverityLevel, StatusType,
    ProtocolType, EncryptionAlgorithm, HashAlgorithm,
    
    # Base Models
    BaseVideoOpusClipModel, TimestampedModel,
    
    # Scanning Models
    ScanTarget, ScanConfiguration, PortResult, VulnerabilityResult, ScanResult,
    
    # Enumeration Models
    DNSRecord, SubdomainResult, SMBShare, SSHHostKey, EnumerationResult,
    
    # Attack Models
    Credential, ExploitResult, AttackResult,
    
    # Security Models
    SecurityFinding, SecurityAssessment,
    
    # Configuration Models
    DatabaseConfig, SecurityConfig, NetworkConfig, ApplicationConfig,
    
    # Response Models
    BaseResponse, ErrorResponse, SuccessResponse, PaginatedResponse,
    
    # Utility Models
    FileInfo, SystemInfo,
    
    # Type Aliases
    HostPort, IPAddress, DomainName, URL, FilePath, JSONData,
    ListOfStrings, OptionalString, OptionalInt, OptionalFloat, OptionalBool, OptionalDateTime,
    
    # Validation Functions
    validate_ip_address, validate_domain_name, validate_url, validate_file_path,
    
    # Model Factories
    create_scan_target, create_security_finding, create_credential
)

from .schemas import (
    # Request Schemas
    ScanRequestSchema, EnumerationRequestSchema, AttackRequestSchema,
    UserRegistrationSchema, UserLoginSchema, PasswordChangeSchema,
    FileUploadSchema, ConfigurationUpdateSchema,
    
    # Response Schemas
    BaseResponseSchema, ErrorResponseSchema, SuccessResponseSchema,
    ScanResponseSchema, EnumerationResponseSchema, AttackResponseSchema,
    UserResponseSchema, AuthenticationResponseSchema, PaginatedResponseSchema,
    
    # Query Parameter Schemas
    PaginationQuerySchema, ScanQuerySchema, UserQuerySchema, SecurityQuerySchema,
    
    # Filter Schemas
    DateRangeFilterSchema, SeverityFilterSchema, TargetFilterSchema,
    
    # Export Schemas
    ExportRequestSchema, ExportResponseSchema,
    
    # Webhook Schemas
    WebhookRequestSchema, WebhookPayloadSchema,
    
    # Notification Schemas
    NotificationRequestSchema, NotificationResponseSchema,
    
    # Validation Functions
    validate_api_key, validate_jwt_token, validate_uuid,
    
    # Schema Utilities
    create_error_response, create_success_response, create_paginated_response
)

__all__ = [
    # Models
    'ScanType', 'EnumerationType', 'AttackType', 'SeverityLevel', 'StatusType',
    'ProtocolType', 'EncryptionAlgorithm', 'HashAlgorithm',
    'BaseVideoOpusClipModel', 'TimestampedModel',
    'ScanTarget', 'ScanConfiguration', 'PortResult', 'VulnerabilityResult', 'ScanResult',
    'DNSRecord', 'SubdomainResult', 'SMBShare', 'SSHHostKey', 'EnumerationResult',
    'Credential', 'ExploitResult', 'AttackResult',
    'SecurityFinding', 'SecurityAssessment',
    'DatabaseConfig', 'SecurityConfig', 'NetworkConfig', 'ApplicationConfig',
    'BaseResponse', 'ErrorResponse', 'SuccessResponse', 'PaginatedResponse',
    'FileInfo', 'SystemInfo',
    
    # Type Aliases
    'HostPort', 'IPAddress', 'DomainName', 'URL', 'FilePath', 'JSONData',
    'ListOfStrings', 'OptionalString', 'OptionalInt', 'OptionalFloat', 'OptionalBool', 'OptionalDateTime',
    
    # Validation Functions
    'validate_ip_address', 'validate_domain_name', 'validate_url', 'validate_file_path',
    
    # Model Factories
    'create_scan_target', 'create_security_finding', 'create_credential',
    
    # Schemas
    'ScanRequestSchema', 'EnumerationRequestSchema', 'AttackRequestSchema',
    'UserRegistrationSchema', 'UserLoginSchema', 'PasswordChangeSchema',
    'FileUploadSchema', 'ConfigurationUpdateSchema',
    'BaseResponseSchema', 'ErrorResponseSchema', 'SuccessResponseSchema',
    'ScanResponseSchema', 'EnumerationResponseSchema', 'AttackResponseSchema',
    'UserResponseSchema', 'AuthenticationResponseSchema', 'PaginatedResponseSchema',
    'PaginationQuerySchema', 'ScanQuerySchema', 'UserQuerySchema', 'SecurityQuerySchema',
    'DateRangeFilterSchema', 'SeverityFilterSchema', 'TargetFilterSchema',
    'ExportRequestSchema', 'ExportResponseSchema',
    'WebhookRequestSchema', 'WebhookPayloadSchema',
    'NotificationRequestSchema', 'NotificationResponseSchema',
    'validate_api_key', 'validate_jwt_token', 'validate_uuid',
    'create_error_response', 'create_success_response', 'create_paginated_response'
]

# Type conversion utilities
def convert_model_to_dict(model_instance) -> Dict[str, Any]:
    """
    Convert a Pydantic model instance to dictionary
    
    Args:
        model_instance: Pydantic model instance
        
    Returns:
        Dictionary representation of the model
    """
    if hasattr(model_instance, 'model_dump'):
        return model_instance.model_dump()
    elif hasattr(model_instance, 'dict'):
        return model_instance.dict()
    else:
        return dict(model_instance)

def convert_dict_to_model(data: Dict[str, Any], model_class) -> Any:
    """
    Convert dictionary to Pydantic model instance
    
    Args:
        data: Dictionary data
        model_class: Pydantic model class
        
    Returns:
        Model instance
    """
    return model_class(**data)

def validate_model_data(data: Dict[str, Any], model_class) -> Tuple[bool, Optional[str]]:
    """
    Validate data against a Pydantic model
    
    Args:
        data: Data to validate
        model_class: Pydantic model class
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        model_class(**data)
        return True, None
    except Exception as e:
        return False, str(e)

# Schema generation utilities
def generate_openapi_schema(model_class) -> Dict[str, Any]:
    """
    Generate OpenAPI schema for a Pydantic model
    
    Args:
        model_class: Pydantic model class
        
    Returns:
        OpenAPI schema dictionary
    """
    if hasattr(model_class, 'model_json_schema'):
        return model_class.model_json_schema()
    elif hasattr(model_class, 'schema'):
        return model_class.schema()
    else:
        raise ValueError("Model class does not support schema generation")

def generate_example_data(model_class) -> Dict[str, Any]:
    """
    Generate example data for a Pydantic model
    
    Args:
        model_class: Pydantic model class
        
    Returns:
        Example data dictionary
    """
    schema = generate_openapi_schema(model_class)
    return schema.get('example', {})

# Type checking utilities
def is_valid_scan_type(scan_type: str) -> bool:
    """Check if scan type is valid"""
    return scan_type in [e.value for e in ScanType]

def is_valid_enumeration_type(enumeration_type: str) -> bool:
    """Check if enumeration type is valid"""
    return enumeration_type in [e.value for e in EnumerationType]

def is_valid_attack_type(attack_type: str) -> bool:
    """Check if attack type is valid"""
    return attack_type in [e.value for e in AttackType]

def is_valid_severity_level(severity: str) -> bool:
    """Check if severity level is valid"""
    return severity in [e.value for e in SeverityLevel]

def is_valid_protocol_type(protocol: str) -> bool:
    """Check if protocol type is valid"""
    return protocol in [e.value for e in ProtocolType]

# Data transformation utilities
def transform_scan_result_to_response(scan_result: ScanResult) -> Dict[str, Any]:
    """
    Transform scan result to API response format
    
    Args:
        scan_result: ScanResult instance
        
    Returns:
        API response dictionary
    """
    return {
        "scan_id": scan_result.scan_id,
        "scan_type": scan_result.scan_type.value,
        "target": {
            "host": scan_result.target.host,
            "port": scan_result.target.port,
            "protocol": scan_result.target.protocol.value
        },
        "status": scan_result.status.value,
        "start_time": scan_result.start_time.isoformat(),
        "end_time": scan_result.end_time.isoformat() if scan_result.end_time else None,
        "duration": scan_result.duration,
        "ports_scanned": scan_result.ports_scanned,
        "ports_open": scan_result.ports_open,
        "vulnerabilities_found": scan_result.vulnerabilities_found,
        "port_results": [convert_model_to_dict(port) for port in scan_result.port_results],
        "vulnerability_results": [convert_model_to_dict(vuln) for vuln in scan_result.vulnerability_results]
    }

def transform_enumeration_result_to_response(enum_result: EnumerationResult) -> Dict[str, Any]:
    """
    Transform enumeration result to API response format
    
    Args:
        enum_result: EnumerationResult instance
        
    Returns:
        API response dictionary
    """
    return {
        "enumeration_id": enum_result.enumeration_id,
        "enumeration_type": enum_result.enumeration_type.value,
        "target": {
            "host": enum_result.target.host,
            "port": enum_result.target.port,
            "protocol": enum_result.target.protocol.value
        },
        "status": enum_result.status.value,
        "start_time": enum_result.start_time.isoformat(),
        "end_time": enum_result.end_time.isoformat() if enum_result.end_time else None,
        "duration": enum_result.duration,
        "dns_records": [convert_model_to_dict(record) for record in enum_result.dns_records],
        "subdomains": [convert_model_to_dict(subdomain) for subdomain in enum_result.subdomains],
        "smb_shares": [convert_model_to_dict(share) for share in enum_result.smb_shares],
        "ssh_host_keys": [convert_model_to_dict(key) for key in enum_result.ssh_host_keys]
    }

def transform_attack_result_to_response(attack_result: AttackResult) -> Dict[str, Any]:
    """
    Transform attack result to API response format
    
    Args:
        attack_result: AttackResult instance
        
    Returns:
        API response dictionary
    """
    return {
        "attack_id": attack_result.attack_id,
        "attack_type": attack_result.attack_type.value,
        "target": {
            "host": attack_result.target.host,
            "port": attack_result.target.port,
            "protocol": attack_result.target.protocol.value
        },
        "status": attack_result.status.value,
        "start_time": attack_result.start_time.isoformat(),
        "end_time": attack_result.end_time.isoformat() if attack_result.end_time else None,
        "duration": attack_result.duration,
        "credentials_found": [convert_model_to_dict(cred) for cred in attack_result.credentials_found],
        "attempts_made": attack_result.attempts_made,
        "total_combinations": attack_result.total_combinations,
        "exploits_attempted": attack_result.exploits_attempted,
        "exploits_successful": attack_result.exploits_successful,
        "exploit_results": [convert_model_to_dict(exploit) for exploit in attack_result.exploit_results]
    }

# Bulk operations utilities
def validate_bulk_scan_requests(requests: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Validate multiple scan requests
    
    Args:
        requests: List of scan request dictionaries
        
    Returns:
        Tuple of (valid_requests, invalid_requests_with_errors)
    """
    valid_requests = []
    invalid_requests = []
    
    for i, request_data in enumerate(requests):
        try:
            validated_request = ScanRequestSchema(**request_data)
            valid_requests.append(convert_model_to_dict(validated_request))
        except Exception as e:
            invalid_requests.append({
                "index": i,
                "data": request_data,
                "error": str(e)
            })
    
    return valid_requests, invalid_requests

def validate_bulk_user_registrations(registrations: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Validate multiple user registration requests
    
    Args:
        registrations: List of user registration dictionaries
        
    Returns:
        Tuple of (valid_registrations, invalid_registrations_with_errors)
    """
    valid_registrations = []
    invalid_registrations = []
    
    for i, registration_data in enumerate(registrations):
        try:
            validated_registration = UserRegistrationSchema(**registration_data)
            valid_registrations.append(convert_model_to_dict(validated_registration))
        except Exception as e:
            invalid_registrations.append({
                "index": i,
                "data": registration_data,
                "error": str(e)
            })
    
    return valid_registrations, invalid_registrations

# Example usage
def main():
    """Example usage of types module"""
    print("📋 Types Module Example")
    
    # Create scan target
    target = create_scan_target("192.168.1.100", 80, ProtocolType.TCP)
    print(f"Scan target: {target}")
    
    # Validate scan type
    print(f"Valid scan type: {is_valid_scan_type('port_scan')}")
    print(f"Invalid scan type: {is_valid_scan_type('invalid_scan')}")
    
    # Create scan result
    scan_result = ScanResult(
        scan_id="scan_12345",
        scan_type=ScanType.PORT_SCAN,
        target=target,
        status=StatusType.COMPLETED,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        duration=45.2,
        ports_scanned=1000,
        ports_open=5,
        vulnerabilities_found=2
    )
    
    # Transform to response format
    response_data = transform_scan_result_to_response(scan_result)
    print(f"Response data: {response_data}")
    
    # Validate model data
    is_valid, error = validate_model_data(
        {"host": "192.168.1.100", "port": 80, "protocol": "tcp"},
        ScanTarget
    )
    print(f"Model validation: {is_valid}, Error: {error}")
    
    # Generate OpenAPI schema
    schema = generate_openapi_schema(ScanTarget)
    print(f"OpenAPI schema keys: {list(schema.keys())}")

if __name__ == "__main__":
    from datetime import datetime
    main() 
from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import re
import ipaddress
from pathlib import Path
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Cybersecurity Toolkit - Validation Schemas
==========================================

Validation schemas with guard clauses for comprehensive error handling
and input validation.
"""


def validate_scan_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate scan configuration with guard clauses.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validation result dictionary
    """
    # Guard clause: Check if config is provided
    if not config:
        return {
            "is_valid": False,
            "error": "Configuration dictionary is required",
            "error_type": "MissingConfiguration"
        }
    
    # Guard clause: Check if config is a dictionary
    if not isinstance(config, dict):
        return {
            "is_valid": False,
            "error": "Configuration must be a dictionary",
            "error_type": "InvalidConfigurationType"
        }
    
    # Guard clause: Check required fields
    required_fields = ["target_host", "target_ports"]
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        return {
            "is_valid": False,
            "error": f"Missing required fields: {missing_fields}",
            "error_type": "MissingRequiredFields"
        }
    
    # Guard clause: Validate target host
    target_host = config.get("target_host")
    if not target_host or not isinstance(target_host, str):
        return {
            "is_valid": False,
            "error": "target_host must be a non-empty string",
            "error_type": "InvalidTargetHost"
        }
    
    # Guard clause: Validate target ports
    target_ports = config.get("target_ports")
    if not isinstance(target_ports, list):
        return {
            "is_valid": False,
            "error": "target_ports must be a list",
            "error_type": "InvalidTargetPorts"
        }
    
    if not target_ports:
        return {
            "is_valid": False,
            "error": "target_ports list cannot be empty",
            "error_type": "EmptyTargetPorts"
        }
    
    # Guard clause: Validate individual ports
    for port in target_ports:
        if not isinstance(port, int):
            return {
                "is_valid": False,
                "error": f"Port {port} must be an integer",
                "error_type": "InvalidPortType"
            }
        if port < 1 or port > 65535:
            return {
                "is_valid": False,
                "error": f"Port {port} is out of valid range (1-65535)",
                "error_type": "InvalidPortRange"
            }
    
    # Guard clause: Validate optional fields
    if "scan_timeout" in config:
        timeout = config["scan_timeout"]
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            return {
                "is_valid": False,
                "error": "scan_timeout must be a positive number",
                "error_type": "InvalidScanTimeout"
            }
    
    if "max_concurrent_scans" in config:
        max_scans = config["max_concurrent_scans"]
        if not isinstance(max_scans, int) or max_scans <= 0:
            return {
                "is_valid": False,
                "error": "max_concurrent_scans must be a positive integer",
                "error_type": "InvalidMaxConcurrentScans"
            }
    
    return {
        "is_valid": True,
        "message": "Scan configuration is valid",
        "validated_config": config
    }

def validate_vulnerability_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate vulnerability scan configuration with guard clauses.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validation result dictionary
    """
    # Guard clause: Check if config is provided
    if not config:
        return {
            "is_valid": False,
            "error": "Configuration dictionary is required",
            "error_type": "MissingConfiguration"
        }
    
    # Guard clause: Check if config is a dictionary
    if not isinstance(config, dict):
        return {
            "is_valid": False,
            "error": "Configuration must be a dictionary",
            "error_type": "InvalidConfigurationType"
        }
    
    # Guard clause: Check required fields
    if "target_url" not in config:
        return {
            "is_valid": False,
            "error": "target_url is required",
            "error_type": "MissingTargetUrl"
        }
    
    # Guard clause: Validate target URL
    target_url = config.get("target_url")
    if not target_url or not isinstance(target_url, str):
        return {
            "is_valid": False,
            "error": "target_url must be a non-empty string",
            "error_type": "InvalidTargetUrl"
        }
    
    # Guard clause: Validate URL format
    url_pattern = r'^https?://[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*(:[0-9]{1,5})?(/.*)?$'
    if not re.match(url_pattern, target_url):
        return {
            "is_valid": False,
            "error": "Invalid URL format",
            "error_type": "InvalidUrlFormat"
        }
    
    # Guard clause: Check URL length
    if len(target_url) > 2048:
        return {
            "is_valid": False,
            "error": "URL too long (max 2048 characters)",
            "error_type": "UrlTooLong"
        }
    
    # Guard clause: Validate scan depth
    if "scan_depth" in config:
        scan_depth = config["scan_depth"]
        valid_depths = ["low", "medium", "high"]
        if scan_depth not in valid_depths:
            return {
                "is_valid": False,
                "error": f"scan_depth must be one of: {valid_depths}",
                "error_type": "InvalidScanDepth"
            }
    
    # Guard clause: Validate boolean fields
    boolean_fields = ["include_ssl_checks", "include_header_checks", "include_content_checks", "follow_redirects"]
    for field in boolean_fields:
        if field in config and not isinstance(config[field], bool):
            return {
                "is_valid": False,
                "error": f"{field} must be a boolean",
                "error_type": "InvalidBooleanField"
            }
    
    # Guard clause: Validate custom headers
    if "custom_headers" in config:
        custom_headers = config["custom_headers"]
        if not isinstance(custom_headers, dict):
            return {
                "is_valid": False,
                "error": "custom_headers must be a dictionary",
                "error_type": "InvalidCustomHeaders"
            }
        
        for header_name, header_value in custom_headers.items():
            if not isinstance(header_name, str) or not header_name.strip():
                return {
                    "is_valid": False,
                    "error": "Header name cannot be empty",
                    "error_type": "InvalidHeaderName"
                }
            if not isinstance(header_value, str):
                return {
                    "is_valid": False,
                    "error": "Header value must be a string",
                    "error_type": "InvalidHeaderValue"
                }
    
    return {
        "is_valid": True,
        "message": "Vulnerability configuration is valid",
        "validated_config": config
    }

def validate_enumeration_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate enumeration configuration with guard clauses.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validation result dictionary
    """
    # Guard clause: Check if config is provided
    if not config:
        return {
            "is_valid": False,
            "error": "Configuration dictionary is required",
            "error_type": "MissingConfiguration"
        }
    
    # Guard clause: Check if config is a dictionary
    if not isinstance(config, dict):
        return {
            "is_valid": False,
            "error": "Configuration must be a dictionary",
            "error_type": "InvalidConfigurationType"
        }
    
    # Guard clause: Check required fields
    if "target_domain" not in config:
        return {
            "is_valid": False,
            "error": "target_domain is required",
            "error_type": "MissingTargetDomain"
        }
    
    if "enumeration_type" not in config:
        return {
            "is_valid": False,
            "error": "enumeration_type is required",
            "error_type": "MissingEnumerationType"
        }
    
    # Guard clause: Validate target domain
    target_domain = config.get("target_domain")
    if not target_domain or not isinstance(target_domain, str):
        return {
            "is_valid": False,
            "error": "target_domain must be a non-empty string",
            "error_type": "InvalidTargetDomain"
        }
    
    # Guard clause: Validate domain format
    domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
    if not re.match(domain_pattern, target_domain):
        return {
            "is_valid": False,
            "error": "Invalid domain format",
            "error_type": "InvalidDomainFormat"
        }
    
    # Guard clause: Check domain length
    if len(target_domain) > 253:
        return {
            "is_valid": False,
            "error": "Domain name too long (max 253 characters)",
            "error_type": "DomainTooLong"
        }
    
    # Guard clause: Validate enumeration type
    enumeration_type = config.get("enumeration_type")
    valid_types = ["dns", "smb", "ssh", "ftp", "http"]
    if enumeration_type not in valid_types:
        return {
            "is_valid": False,
            "error": f"enumeration_type must be one of: {valid_types}",
            "error_type": "InvalidEnumerationType"
        }
    
    # Guard clause: Validate lists if provided
    list_fields = ["subdomain_list", "username_list", "password_list"]
    for field in list_fields:
        if field in config:
            field_value = config[field]
            if not isinstance(field_value, list):
                return {
                    "is_valid": False,
                    "error": f"{field} must be a list",
                    "error_type": "InvalidListField"
                }
            
            # Check list size limits
            if len(field_value) > 10000:
                return {
                    "is_valid": False,
                    "error": f"{field} too large (max 10000 items)",
                    "error_type": "ListTooLarge"
                }
            
            # Validate list items
            for item in field_value:
                if not isinstance(item, str) or not item.strip():
                    return {
                        "is_valid": False,
                        "error": f"Items in {field} cannot be empty",
                        "error_type": "EmptyListItem"
                    }
                if len(item) > 100:
                    return {
                        "is_valid": False,
                        "error": f"Items in {field} too long (max 100 characters)",
                        "error_type": "ListItemTooLong"
                    }
    
    # Guard clause: Validate wordlist file if provided
    if "wordlist_file" in config:
        wordlist_file = config["wordlist_file"]
        if not isinstance(wordlist_file, str):
            return {
                "is_valid": False,
                "error": "wordlist_file must be a string",
                "error_type": "InvalidWordlistFile"
            }
        
        file_path = Path(wordlist_file)
        if not file_path.exists():
            return {
                "is_valid": False,
                "error": f"Wordlist file not found: {wordlist_file}",
                "error_type": "WordlistFileNotFound"
            }
        
        if not file_path.is_file():
            return {
                "is_valid": False,
                "error": f"Wordlist path is not a file: {wordlist_file}",
                "error_type": "WordlistNotAFile"
            }
    
    return {
        "is_valid": True,
        "message": "Enumeration configuration is valid",
        "validated_config": config
    }

def validate_attack_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate attack configuration with guard clauses.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validation result dictionary
    """
    # Guard clause: Check if config is provided
    if not config:
        return {
            "is_valid": False,
            "error": "Configuration dictionary is required",
            "error_type": "MissingConfiguration"
        }
    
    # Guard clause: Check if config is a dictionary
    if not isinstance(config, dict):
        return {
            "is_valid": False,
            "error": "Configuration must be a dictionary",
            "error_type": "InvalidConfigurationType"
        }
    
    # Guard clause: Check required fields
    required_fields = ["target_host", "target_port", "attack_type"]
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        return {
            "is_valid": False,
            "error": f"Missing required fields: {missing_fields}",
            "error_type": "MissingRequiredFields"
        }
    
    # Guard clause: Validate target host
    target_host = config.get("target_host")
    if not target_host or not isinstance(target_host, str):
        return {
            "is_valid": False,
            "error": "target_host must be a non-empty string",
            "error_type": "InvalidTargetHost"
        }
    
    # Guard clause: Validate target port
    target_port = config.get("target_port")
    if not isinstance(target_port, int):
        return {
            "is_valid": False,
            "error": "target_port must be an integer",
            "error_type": "InvalidTargetPort"
        }
    
    if target_port < 1 or target_port > 65535:
        return {
            "is_valid": False,
            "error": "target_port is out of valid range (1-65535)",
            "error_type": "InvalidPortRange"
        }
    
    # Guard clause: Validate attack type
    attack_type = config.get("attack_type")
    valid_types = ["brute_force", "exploit", "dos", "phishing"]
    if attack_type not in valid_types:
        return {
            "is_valid": False,
            "error": f"attack_type must be one of: {valid_types}",
            "error_type": "InvalidAttackType"
        }
    
    # Guard clause: Validate credential lists for brute force attacks
    if attack_type == "brute_force":
        if "username_list" not in config or "password_list" not in config:
            return {
                "is_valid": False,
                "error": "username_list and password_list are required for brute force attacks",
                "error_type": "MissingCredentials"
            }
        
        username_list = config.get("username_list")
        password_list = config.get("password_list")
        
        if not isinstance(username_list, list) or not username_list:
            return {
                "is_valid": False,
                "error": "username_list must be a non-empty list",
                "error_type": "InvalidUsernameList"
            }
        
        if not isinstance(password_list, list) or not password_list:
            return {
                "is_valid": False,
                "error": "password_list must be a non-empty list",
                "error_type": "InvalidPasswordList"
            }
        
        # Validate credential list sizes
        if len(username_list) > 10000:
            return {
                "is_valid": False,
                "error": "username_list too large (max 10000 items)",
                "error_type": "UsernameListTooLarge"
            }
        
        if len(password_list) > 10000:
            return {
                "is_valid": False,
                "error": "password_list too large (max 10000 items)",
                "error_type": "PasswordListTooLarge"
            }
    
    # Guard clause: Validate numeric fields
    numeric_fields = ["max_attempts", "delay_between_attempts", "timeout_per_attempt"]
    for field in numeric_fields:
        if field in config:
            field_value = config[field]
            if not isinstance(field_value, (int, float)):
                return {
                    "is_valid": False,
                    "error": f"{field} must be a number",
                    "error_type": "InvalidNumericField"
                }
            
            if field_value <= 0:
                return {
                    "is_valid": False,
                    "error": f"{field} must be positive",
                    "error_type": "NonPositiveNumericField"
                }
    
    # Guard clause: Validate max attempts limit
    if "max_attempts" in config:
        max_attempts = config["max_attempts"]
        if max_attempts > 100000:
            return {
                "is_valid": False,
                "error": "max_attempts cannot exceed 100000",
                "error_type": "MaxAttemptsTooHigh"
            }
    
    return {
        "is_valid": True,
        "message": "Attack configuration is valid",
        "validated_config": config
    }

def validate_network_target(target: str) -> Dict[str, Any]:
    """
    Validate network target (IP address or hostname) with guard clauses.
    
    Args:
        target: Target string to validate
        
    Returns:
        Validation result dictionary
    """
    # Guard clause: Check if target is provided
    if not target:
        return {
            "is_valid": False,
            "error": "Target is required",
            "error_type": "MissingTarget"
        }
    
    # Guard clause: Check if target is a string
    if not isinstance(target, str):
        return {
            "is_valid": False,
            "error": "Target must be a string",
            "error_type": "InvalidTargetType"
        }
    
    # Guard clause: Check target length
    if len(target) > 253:
        return {
            "is_valid": False,
            "error": "Target too long (max 253 characters)",
            "error_type": "TargetTooLong"
        }
    
    # Guard clause: Try to validate as IP address
    try:
        ipaddress.ip_address(target)
        return {
            "is_valid": True,
            "target_type": "ip_address",
            "message": "Valid IP address",
            "target": target
        }
    except ValueError:
        # Not an IP address, validate as hostname
        hostname_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
        if re.match(hostname_pattern, target):
            return {
                "is_valid": True,
                "target_type": "hostname",
                "message": "Valid hostname",
                "target": target.lower()
            }
        else:
            return {
                "is_valid": False,
                "error": "Invalid hostname format",
                "error_type": "InvalidHostnameFormat"
            }

def validate_file_path(file_path: str, must_exist: bool = False) -> Dict[str, Any]:
    """
    Validate file path with guard clauses.
    
    Args:
        file_path: File path to validate
        must_exist: Whether the file must exist
        
    Returns:
        Validation result dictionary
    """
    # Guard clause: Check if file path is provided
    if not file_path:
        return {
            "is_valid": False,
            "error": "File path is required",
            "error_type": "MissingFilePath"
        }
    
    # Guard clause: Check if file path is a string
    if not isinstance(file_path, str):
        return {
            "is_valid": False,
            "error": "File path must be a string",
            "error_type": "InvalidFilePathType"
        }
    
    # Guard clause: Check file path length
    if len(file_path) > 4096:
        return {
            "is_valid": False,
            "error": "File path too long (max 4096 characters)",
            "error_type": "FilePathTooLong"
        }
    
    # Guard clause: Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    for char in invalid_chars:
        if char in file_path:
            return {
                "is_valid": False,
                "error": f"File path contains invalid character: {char}",
                "error_type": "InvalidFilePathCharacters"
            }
    
    # Guard clause: Check if file exists (if required)
    if must_exist:
        path_obj = Path(file_path)
        if not path_obj.exists():
            return {
                "is_valid": False,
                "error": f"File does not exist: {file_path}",
                "error_type": "FileNotFound"
            }
        
        if not path_obj.is_file():
            return {
                "is_valid": False,
                "error": f"Path is not a file: {file_path}",
                "error_type": "NotAFile"
            }
    
    return {
        "is_valid": True,
        "message": "File path is valid",
        "file_path": file_path
    }

# --- Named Exports ---

__all__ = [
    'validate_scan_configuration',
    'validate_vulnerability_configuration',
    'validate_enumeration_configuration',
    'validate_attack_configuration',
    'validate_network_target',
    'validate_file_path'
] 
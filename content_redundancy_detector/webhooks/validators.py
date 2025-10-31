"""
Webhook Validators - Enhanced validation for webhook endpoints
Enterprise-grade validation with security checks and best practices
"""

import re
import logging
import json
from typing import Tuple, Optional, Dict, Any, List, Union
from urllib.parse import urlparse
import ipaddress

logger = logging.getLogger(__name__)


class WebhookValidator:
    """
    Enterprise-grade validator for webhook endpoints and payloads
    
    Features:
    - URL validation with security checks
    - Secret strength validation
    - Payload size validation
    - IP address filtering
    - Configuration validation
    """
    
    # Constants
    MAX_URL_LENGTH = 2048
    MIN_SECRET_LENGTH = 16
    MAX_SECRET_LENGTH = 256
    DEFAULT_MAX_PAYLOAD_SIZE_MB = 1.0
    MAX_PAYLOAD_SIZE_MB = 10.0
    MIN_TIMEOUT = 1
    MAX_TIMEOUT = 300
    MIN_RETRY_COUNT = 0
    MAX_RETRY_COUNT = 10
    
    @staticmethod
    def validate_endpoint_url(url: str, allow_localhost: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Validate webhook endpoint URL with comprehensive security checks
        
        Args:
            url: URL to validate
            allow_localhost: Allow localhost/internal IPs (for development only)
        
        Returns:
            (is_valid, error_message)
        """
        if not url:
            return False, "URL is required"
        
        if not isinstance(url, str):
            return False, "URL must be a string"
        
        url = url.strip()
        
        if len(url) > WebhookValidator.MAX_URL_LENGTH:
            return False, f"URL too long. Maximum length is {WebhookValidator.MAX_URL_LENGTH} characters"
        
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ["http", "https"]:
                return False, f"Invalid URL scheme. Only http and https are allowed, got {parsed.scheme}"
            
            # HTTPS is preferred
            if parsed.scheme == "http":
                logger.warning(f"Webhook endpoint using insecure HTTP: {url}")
            
            # Check hostname
            if not parsed.netloc:
                return False, "Invalid URL. Missing hostname"
            
            # Validate hostname format
            hostname = parsed.hostname or ""
            
            # Check for localhost/internal IPs (security concern)
            if not allow_localhost:
                if hostname in ["localhost", "127.0.0.1", "0.0.0.0"]:
                    return False, "Localhost/internal IPs are not allowed for webhook endpoints (set allow_localhost=True for development)"
                
                try:
                    ip = ipaddress.ip_address(hostname)
                    if ip.is_private or ip.is_loopback or ip.is_reserved:
                        return False, "Private/reserved/internal IPs are not allowed for webhook endpoints (set allow_localhost=True for development)"
                except ValueError:
                # Not an IP, check hostname format
                if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$', hostname):
                    return False, "Invalid hostname format"
            
            # Check path (optional but warn if empty)
            if not parsed.path or parsed.path == "/":
                logger.debug("Webhook endpoint has no path specified")
            
            return True, None
        
        except Exception as e:
            return False, f"Invalid URL format: {str(e)}"
    
    @staticmethod
    def validate_endpoint_secret(secret: Optional[str], require_strong: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Validate webhook secret with strength checks
        
        Args:
            secret: Secret string to validate
            require_strong: Require strong secret (default: False, only warns)
        
        Returns:
            (is_valid, error_message)
        """
        if secret is None:
            return True, None  # Secret is optional
        
        if not isinstance(secret, str):
            return False, "Secret must be a string"
        
        secret = secret.strip()
        
        if len(secret) < WebhookValidator.MIN_SECRET_LENGTH:
            return False, f"Secret must be at least {WebhookValidator.MIN_SECRET_LENGTH} characters long"
        
        if len(secret) > WebhookValidator.MAX_SECRET_LENGTH:
            return False, f"Secret too long. Maximum length is {WebhookValidator.MAX_SECRET_LENGTH} characters"
        
        # Check for strong secret (complexity requirements)
        has_upper = bool(re.search(r'[A-Z]', secret))
        has_lower = bool(re.search(r'[a-z]', secret))
        has_digit = bool(re.search(r'[0-9]', secret))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', secret))
        
        complexity_score = sum([has_upper, has_lower, has_digit, has_special])
        
        if require_strong:
            if complexity_score < 3:
                return False, "Secret must contain at least 3 of: uppercase, lowercase, digits, special characters"
        else:
            if complexity_score < 2:
                logger.warning(f"Weak webhook secret detected. Consider using uppercase, lowercase, digits, and special characters")
        
        # Check for common weak patterns
        if secret.lower() in ['password', 'secret', 'webhook', 'api', 'key', 'token']:
            logger.warning("Webhook secret appears to be a common weak value")
        
        return True, None
    
    @staticmethod
    def validate_payload_size(payload: Dict[str, Any], max_size_mb: float = None) -> Tuple[bool, Optional[str]]:
        """
        Validate webhook payload size and structure
        
        Args:
            payload: Payload dictionary to validate
            max_size_mb: Maximum size in MB (default: DEFAULT_MAX_PAYLOAD_SIZE_MB)
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(payload, dict):
            return False, "Payload must be a dictionary"
        
        max_size = max_size_mb or WebhookValidator.DEFAULT_MAX_PAYLOAD_SIZE_MB
        
        if max_size > WebhookValidator.MAX_PAYLOAD_SIZE_MB:
            return False, f"Max size limit cannot exceed {WebhookValidator.MAX_PAYLOAD_SIZE_MB}MB"
        
        try:
            payload_json = json.dumps(payload, ensure_ascii=False)
            size_bytes = len(payload_json.encode('utf-8'))
            size_mb = size_bytes / (1024 * 1024)
            
            if size_mb > max_size:
                return False, f"Payload too large: {size_mb:.2f}MB (max: {max_size}MB)"
            
            # Check for dangerous keys (security)
            dangerous_keys = ['__class__', '__dict__', '__getattr__', '__setattr__', '__reduce__']
            found_dangerous = [key for key in dangerous_keys if key in payload]
            if found_dangerous:
                logger.warning(f"Potentially dangerous keys found in payload: {found_dangerous}")
            
            return True, None
        except TypeError as e:
            return False, f"Payload contains non-serializable data: {str(e)}"
        except Exception as e:
            return False, f"Failed to validate payload size: {str(e)}"
    
    @staticmethod
    def validate_endpoint_config(
        url: str,
        secret: Optional[str] = None,
        timeout: int = 30,
        retry_count: int = 3,
        allow_localhost: bool = False,
        require_strong_secret: bool = False
    ) -> Tuple[bool, Optional[str], Optional[dict]]:
        """
        Validate complete webhook endpoint configuration
        
        Args:
            url: Endpoint URL
            secret: Webhook secret (optional)
            timeout: Request timeout in seconds
            retry_count: Number of retries
            allow_localhost: Allow localhost URLs (dev only)
            require_strong_secret: Require strong secret
        
        Returns:
            (is_valid, error_message, validation_details)
        """
        details = {
            "url": url,
            "timeout": timeout,
            "retry_count": retry_count,
            "has_secret": secret is not None
        }
        
        # Validate URL
        is_valid, error = WebhookValidator.validate_endpoint_url(url, allow_localhost=allow_localhost)
        if not is_valid:
            return False, error, None
        details["url_valid"] = True
        parsed = urlparse(url)
        details["scheme"] = parsed.scheme
        details["hostname"] = parsed.hostname
        
        # Validate secret
        is_valid, error = WebhookValidator.validate_endpoint_secret(secret, require_strong=require_strong_secret)
        if not is_valid:
            return False, error, None
        details["secret_valid"] = True
        if secret:
            details["secret_length"] = len(secret)
        
        # Validate timeout
        if timeout < WebhookValidator.MIN_TIMEOUT or timeout > WebhookValidator.MAX_TIMEOUT:
            return False, f"Timeout must be between {WebhookValidator.MIN_TIMEOUT} and {WebhookValidator.MAX_TIMEOUT} seconds", None
        details["timeout"] = timeout
        
        # Validate retry count
        if retry_count < WebhookValidator.MIN_RETRY_COUNT or retry_count > WebhookValidator.MAX_RETRY_COUNT:
            return False, f"Retry count must be between {WebhookValidator.MIN_RETRY_COUNT} and {WebhookValidator.MAX_RETRY_COUNT}", None
        details["retry_count"] = retry_count
        
        return True, None, details
    
    @staticmethod
    def validate_event_type(event: Union[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate webhook event type
        
        Args:
            event: Event type (string or enum)
        
        Returns:
            (is_valid, error_message)
        """
        if not event:
            return False, "Event type is required"
        
        # Allow enum or string
        if hasattr(event, 'value'):
            event_str = event.value
        elif isinstance(event, str):
            event_str = event
        else:
            return False, "Event must be a string or WebhookEvent enum"
        
        # Validate format (alphanumeric, underscore, dash)
        if not re.match(r'^[a-zA-Z0-9_-]+$', event_str):
            return False, "Event type must contain only alphanumeric characters, underscores, and dashes"
        
        if len(event_str) > 100:
            return False, "Event type too long (max 100 characters)"
        
        return True, None
    
    @staticmethod
    def sanitize_endpoint_id(endpoint_id: str) -> str:
        """
        Sanitize and normalize endpoint ID
        
        Args:
            endpoint_id: Raw endpoint ID
        
        Returns:
            Sanitized endpoint ID
        """
        if not endpoint_id:
            raise ValueError("Endpoint ID cannot be empty")
        
        # Lowercase and strip
        sanitized = endpoint_id.lower().strip()
        
        # Replace spaces with underscores
        sanitized = re.sub(r'\s+', '_', sanitized)
        
        # Remove invalid characters
        sanitized = re.sub(r'[^a-z0-9_-]', '', sanitized)
        
        # Ensure it starts/ends with alphanumeric
        sanitized = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', sanitized)
        
        if not sanitized:
            raise ValueError("Endpoint ID contains no valid characters")
        
        return sanitized


def validate_webhook_endpoint(
    endpoint: Union[Any, Dict[str, Any]],
    allow_localhost: bool = False,
    require_strong_secret: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate webhook endpoint (convenience function)
    
    Args:
        endpoint: WebhookEndpoint instance or dict with endpoint data
        allow_localhost: Allow localhost URLs (development only)
        require_strong_secret: Require strong secret
    
    Returns:
        (is_valid, error_message)
    """
    try:
        if hasattr(endpoint, 'url'):
            # Object with attributes (WebhookEndpoint)
            url = endpoint.url
            secret = getattr(endpoint, 'secret', None)
            timeout = getattr(endpoint, 'timeout', 30)
            retry_count = getattr(endpoint, 'retry_count', 3)
        elif isinstance(endpoint, dict):
            # Dictionary
            url = endpoint.get('url', '')
            secret = endpoint.get('secret')
            timeout = endpoint.get('timeout', 30)
            retry_count = endpoint.get('retry_count', 3)
        else:
            return False, "Invalid endpoint format. Must be WebhookEndpoint object or dict with 'url' key"
        
        if not url:
            return False, "Endpoint URL is required"
        
        is_valid, error, details = WebhookValidator.validate_endpoint_config(
            url,
            secret=secret,
            timeout=timeout,
            retry_count=retry_count,
            allow_localhost=allow_localhost,
            require_strong_secret=require_strong_secret
        )
        return is_valid, error
    except Exception as e:
        logger.error(f"Error validating webhook endpoint: {e}", exc_info=True)
        return False, f"Validation error: {str(e)}"

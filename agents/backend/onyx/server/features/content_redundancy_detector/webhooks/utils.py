"""
Webhook Utilities - Common helper functions
"""

import time
import hashlib
import hmac
import json
import logging
from typing import Dict, Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def generate_webhook_signature(payload: Dict[str, Any], secret: str) -> str:
    """
    Generate HMAC-SHA256 signature for webhook payload
    
    Args:
        payload: Webhook payload dictionary
        secret: Secret key for signing
        
    Returns:
        Signature string in format: sha256=<hex_digest>
    """
    # Sort keys for consistent signing. Use orjson if available for speed.
    try:
        import orjson  # type: ignore
        payload_bytes = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    except Exception:
        # Fallback to stdlib json (slower). default=str to handle non-serializable types.
        payload_bytes = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    
    # Generate HMAC signature (use bytes directly)
    secret_bytes = secret.encode('utf-8')
    signature = hmac.new(secret_bytes, payload_bytes, hashlib.sha256).hexdigest()
    
    return f"sha256={signature}"


def verify_webhook_signature(payload: Dict[str, Any], signature: str, secret: str) -> bool:
    """
    Verify webhook signature
    
    Args:
        payload: Webhook payload dictionary
        signature: Signature string from header
        secret: Secret key for verification
        
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        expected_signature = generate_webhook_signature(payload, secret)
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(expected_signature, signature)
    except Exception as e:
        logger.error(f"Signature verification error: {e}")
        return False


def normalize_endpoint_url(url: str) -> str:
    """
    Normalize webhook endpoint URL
    
    Args:
        url: Webhook URL to normalize
        
    Returns:
        Normalized URL
    """
    # Remove trailing slash
    url = url.rstrip('/')
    
    # Parse and reconstruct
    parsed = urlparse(url)
    
    # Ensure HTTPS (preferred)
    scheme = parsed.scheme.lower()
    if scheme not in ['http', 'https']:
        logger.warning(f"Invalid URL scheme: {scheme}, defaulting to https")
        scheme = 'https'
    
    # Reconstruct URL
    normalized = f"{scheme}://{parsed.netloc}{parsed.path}"
    
    if parsed.query:
        normalized += f"?{parsed.query}"
    
    if parsed.fragment:
        normalized += f"#{parsed.fragment}"
    
    return normalized


def calculate_retry_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, multiplier: float = 2.0) -> float:
    """
    Calculate retry delay with exponential backoff
    
    Args:
        attempt: Current attempt number (1-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        multiplier: Exponential multiplier
        
    Returns:
        Delay in seconds
    """
    # Use pow for speed, clamp to max_delay
    delay = min(base_delay * pow(multiplier, max(0, attempt - 1)), max_delay)
    return delay


def should_retry(status_code: int, attempt: int, max_attempts: int = 3) -> bool:
    """
    Determine if webhook should be retried
    
    Args:
        status_code: HTTP status code
        attempt: Current attempt number
        max_attempts: Maximum number of attempts
        
    Returns:
        True if should retry, False otherwise
    """
    if attempt >= max_attempts:
        return False
    
    # Retry on transient errors
    retryable_status_codes = [
        408,  # Request Timeout
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    ]
    
    # Also retry on network errors (status_code 0)
    return status_code == 0 or status_code in retryable_status_codes


def create_webhook_headers(
    delivery_id: str,
    event: str,
    timestamp: float,
    signature: Optional[str] = None
) -> Dict[str, str]:
    """
    Create standard webhook headers
    
    Args:
        delivery_id: Webhook delivery ID
        event: Webhook event type
        timestamp: Event timestamp
        signature: Optional signature
        
    Returns:
        Dictionary of HTTP headers
    """
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Content-Redundancy-Detector/2.0.0",
        "X-Webhook-Id": delivery_id,
        "X-Webhook-Event": event,
        "X-Webhook-Timestamp": str(int(timestamp)),
        "X-Webhook-Version": "2.0"
    }
    
    if signature:
        headers["X-Webhook-Signature"] = signature
    
    return headers


def parse_webhook_headers(headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Parse webhook headers from request
    
    Args:
        headers: Dictionary of HTTP headers
        
    Returns:
        Parsed webhook information
    """
    return {
        "webhook_id": headers.get("X-Webhook-Id"),
        "event": headers.get("X-Webhook-Event"),
        "timestamp": headers.get("X-Webhook-Timestamp"),
        "signature": headers.get("X-Webhook-Signature"),
        "version": headers.get("X-Webhook-Version", "1.0")
    }


def format_webhook_payload(
    event: str,
    data: Dict[str, Any],
    request_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format webhook payload with standard structure
    
    Args:
        event: Event type
        data: Event data
        request_id: Optional request ID
        user_id: Optional user ID
        
    Returns:
        Formatted payload dictionary
    """
    payload = {
        "event": event,
        "timestamp": time.time(),
        "data": data
    }
    
    if request_id:
        payload["request_id"] = request_id
    
    if user_id:
        payload["user_id"] = user_id
    
    return payload


def is_valid_webhook_url(url: str) -> bool:
    """
    Quick validation of webhook URL format
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL format is valid
    """
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme in ['http', 'https'] and
            parsed.netloc and
            len(url) <= 2048
        )
    except Exception:
        return False


def calculate_payload_size(payload: Dict[str, Any]) -> int:
    """
    Calculate payload size in bytes
    
    Args:
        payload: Payload dictionary
        
    Returns:
        Size in bytes
    """
    payload_json = json.dumps(payload, default=str)
    return len(payload_json.encode('utf-8'))


def sanitize_endpoint_url(url: str) -> str:
    """
    Sanitize endpoint URL for logging (remove credentials)
    
    Args:
        url: URL to sanitize
        
    Returns:
        Sanitized URL with credentials masked
    """
    try:
        parsed = urlparse(url)
        
        # Mask credentials if present
        if parsed.username or parsed.password:
            masked_netloc = f"{parsed.hostname or ''}"
            if parsed.port:
                masked_netloc += f":{parsed.port}"
        else:
            masked_netloc = parsed.netloc
        
        # Reconstruct URL
        sanitized = f"{parsed.scheme}://{masked_netloc}{parsed.path}"
        
        if parsed.query:
            sanitized += f"?{parsed.query}"
        
        return sanitized
    except Exception:
        # If parsing fails, just return truncated version
        return url[:100] + "..." if len(url) > 100 else url


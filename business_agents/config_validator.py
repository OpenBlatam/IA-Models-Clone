"""Configuration validation on startup."""
import logging
from typing import List, Tuple
from .settings import settings

logger = logging.getLogger(__name__)


def validate_config() -> Tuple[bool, List[str]]:
    """
    Validate configuration settings.
    
    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Security warnings
    if settings.enforce_auth:
        if not settings.api_key and not settings.jwt_secret:
            warnings.append(
                "ENFORCE_AUTH=true but neither API_KEY nor JWT_SECRET is set. "
                "All requests will be rejected!"
            )
    
    if settings.jwt_secret and len(settings.jwt_secret) < 16:
        warnings.append(
            "JWT_SECRET is too short (< 16 chars). Use a longer, secure secret."
        )
    
    # Performance warnings
    if settings.rps_limit > 1000:
        warnings.append(
            f"RPS_LIMIT is very high ({settings.rps_limit}). "
            "Ensure your infrastructure can handle this load."
        )
    
    if settings.max_body_bytes > 50 * 1024 * 1024:  # 50MB
        warnings.append(
            f"MAX_BODY_BYTES is very large ({settings.max_body_bytes / 1024 / 1024:.1f}MB). "
            "This may impact memory usage."
        )
    
    # Cache warnings
    if settings.cache_ttl_seconds < 1:
        warnings.append(
            f"CACHE_TTL_SECONDS is very short ({settings.cache_ttl_seconds}s). "
            "This may reduce cache effectiveness."
        )
    
    # Feature flags info
    if settings.features:
        logger.info(f"Feature flags enabled: {', '.join(settings.features)}")
    else:
        logger.info("All features enabled (no feature flags set)")
    
    # Log warnings
    for warning in warnings:
        logger.warning(f"Config validation: {warning}")
    
    return len(warnings) == 0, warnings



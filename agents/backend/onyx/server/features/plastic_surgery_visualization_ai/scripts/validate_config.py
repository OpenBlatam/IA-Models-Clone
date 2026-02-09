"""Script to validate configuration."""

from pathlib import Path
import sys

from config.settings import settings
from utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def validate_settings() -> bool:
    """
    Validate application settings.
    
    Returns:
        True if valid, False otherwise
    """
    errors = []
    
    # Validate directories
    upload_dir = Path(settings.upload_dir)
    output_dir = Path(settings.output_dir)
    
    if not upload_dir.exists():
        errors.append(f"Upload directory does not exist: {upload_dir}")
    
    if not output_dir.exists():
        errors.append(f"Output directory does not exist: {output_dir}")
    
    # Validate image settings
    if settings.max_image_size_mb <= 0:
        errors.append("max_image_size_mb must be positive")
    
    if not settings.supported_formats:
        errors.append("supported_formats cannot be empty")
    
    # Validate server settings
    if settings.port < 1 or settings.port > 65535:
        errors.append(f"Invalid port: {settings.port}")
    
    # Validate AI settings
    if settings.model_provider and not settings.api_key:
        logger.warning("API key not set but model provider is configured")
    
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("Configuration is valid!")
    return True


if __name__ == "__main__":
    success = validate_settings()
    sys.exit(0 if success else 1)


from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import sys
from pathlib import Path
from typing import Optional
    from dotenv import load_dotenv
import uvicorn
from api.routes.__main__ import create_app, create_development_app, create_production_app
    import structlog
        import logging
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
HeyGen AI FastAPI Main Entry Point
FastAPI best practices for main application entry point with proper configuration.
"""


try:
    load_dotenv()
except ImportError:
    pass


# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


try:
except ImportError:
    structlog = None

# =============================================================================
# Logging Configuration
# =============================================================================
def configure_logging():
    
    """configure_logging function."""
if structlog:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger()
    else:
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
        return logging.getLogger("heygen_ai")

logger = configure_logging()

# =============================================================================
# Environment Configuration
# =============================================================================
def get_env(key: str, default: Optional[str] = None) -> str:
    return os.getenv(key, default)

def get_environment() -> str:
    return get_env("ENVIRONMENT", "development").lower()

def get_host() -> str:
    return get_env("HOST", "0.0.0.0")

def get_port() -> int:
    return int(get_env("PORT", "8000"))

def get_workers() -> int:
    return int(get_env("WORKERS", "1"))

def get_log_level() -> str:
    return get_env("LOG_LEVEL", "info").lower()

def get_reload() -> bool:
    return get_env("RELOAD", "false").lower() == "true"

# =============================================================================
# Application Configuration
# =============================================================================
def configure_application():
    
    """configure_application function."""
environment = get_environment()
    if environment == "production":
        return create_production_app()
    elif environment == "development":
        return create_development_app()
    return create_app()

# =============================================================================
# Server Configuration
# =============================================================================
def get_server_config():
    
    """get_server_config function."""
return {
        "host": get_host(),
        "port": get_port(),
        "workers": get_workers(),
        "log_level": get_log_level(),
        "reload": get_reload(),
        "access_log": True,
        "proxy_headers": True,
        "forwarded_allow_ips": "*",
        "timeout_keep_alive": 30,
        "timeout_graceful_shutdown": 30
    }

# =============================================================================
# Main Function
# =============================================================================
def main():
    """Main entrypoint for HeyGen AI FastAPI server."""
    environment = get_environment()
    logger.info(
        "Starting HeyGen AI FastAPI server",
        extra={
            "environment": environment,
            "host": get_host(),
            "port": get_port(),
            "workers": get_workers(),
            "log_level": get_log_level(),
            "reload": get_reload()
        }
    )
    app = configure_application()
    server_config = get_server_config()
    try:
        uvicorn.run(
            "main:app",
            **server_config
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error("Server error", exc_info=True)
        sys.exit(1)

# =============================================================================
# Application Instance
# =============================================================================
app = configure_application()

# =============================================================================
# Entry Point
# =============================================================================
match __name__:
    case "__main__":
    main() 
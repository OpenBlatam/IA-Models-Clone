from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
from pydantic import BaseModel, Field, validator
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Onyx AI Video System - Settings

Onyx-specific settings and configuration for the AI Video system.
"""




@dataclass
class OnyxSettings:
    """Onyx-specific settings for AI Video system."""
    
    # Onyx integration settings
    use_onyx_logging: bool = True
    use_onyx_llm: bool = True
    use_onyx_telemetry: bool = True
    use_onyx_encryption: bool = True
    use_onyx_threading: bool = True
    use_onyx_retry: bool = True
    use_onyx_gpu: bool = True
    
    # Onyx paths
    onyx_root_path: Optional[str] = None
    onyx_config_path: Optional[str] = None
    onyx_plugins_path: Optional[str] = None
    
    # Onyx LLM settings
    onyx_default_llm: str = "gpt-4"
    onyx_vision_llm: Optional[str] = "gpt-4-vision-preview"
    onyx_temperature: float = 0.7
    onyx_max_tokens: int = 4000
    
    # Onyx threading settings
    onyx_max_workers: int = 10
    onyx_thread_timeout: int = 300
    
    # Onyx security settings
    onyx_encryption_key: Optional[str] = None
    onyx_validate_access: bool = True
    
    # Onyx performance settings
    onyx_enable_gpu: bool = True
    onyx_cache_enabled: bool = True
    onyx_metrics_enabled: bool = True
    
    def __post_init__(self) -> Any:
        """Post-initialization setup."""
        # Set default paths if not provided
        if not self.onyx_root_path:
            self.onyx_root_path = self._find_onyx_root()
        
        if not self.onyx_config_path:
            self.onyx_config_path = os.path.join(self.onyx_root_path, "config") if self.onyx_root_path else None
        
        if not self.onyx_plugins_path:
            self.onyx_plugins_path = os.path.join(self.onyx_root_path, "plugins") if self.onyx_root_path else None
    
    def _find_onyx_root(self) -> Optional[str]:
        """Find Onyx root directory."""
        # Common Onyx installation paths
        possible_paths = [
            os.getenv('ONYX_ROOT'),
            os.getenv('ONYX_HOME'),
            "/opt/onyx",
            "/usr/local/onyx",
            os.path.expanduser("~/onyx"),
            os.path.join(os.getcwd(), "onyx"),
            os.path.join(os.getcwd(), "..", "onyx"),
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "use_onyx_logging": self.use_onyx_logging,
            "use_onyx_llm": self.use_onyx_llm,
            "use_onyx_telemetry": self.use_onyx_telemetry,
            "use_onyx_encryption": self.use_onyx_encryption,
            "use_onyx_threading": self.use_onyx_threading,
            "use_onyx_retry": self.use_onyx_retry,
            "use_onyx_gpu": self.use_onyx_gpu,
            "onyx_root_path": self.onyx_root_path,
            "onyx_config_path": self.onyx_config_path,
            "onyx_plugins_path": self.onyx_plugins_path,
            "onyx_default_llm": self.onyx_default_llm,
            "onyx_vision_llm": self.onyx_vision_llm,
            "onyx_temperature": self.onyx_temperature,
            "onyx_max_tokens": self.onyx_max_tokens,
            "onyx_max_workers": self.onyx_max_workers,
            "onyx_thread_timeout": self.onyx_thread_timeout,
            "onyx_encryption_key": self.onyx_encryption_key,
            "onyx_validate_access": self.onyx_validate_access,
            "onyx_enable_gpu": self.onyx_enable_gpu,
            "onyx_cache_enabled": self.onyx_cache_enabled,
            "onyx_metrics_enabled": self.onyx_metrics_enabled,
        }


class OnyxEnvironmentConfig(BaseModel):
    """Onyx environment configuration."""
    
    # Environment variables
    ONYX_ROOT: Optional[str] = Field(default=None, description="Onyx root directory")
    ONYX_HOME: Optional[str] = Field(default=None, description="Onyx home directory")
    ONYX_CONFIG: Optional[str] = Field(default=None, description="Onyx config directory")
    ONYX_PLUGINS: Optional[str] = Field(default=None, description="Onyx plugins directory")
    
    # LLM environment variables
    ONYX_DEFAULT_LLM: str = Field(default="gpt-4", description="Default LLM model")
    ONYX_VISION_LLM: Optional[str] = Field(default="gpt-4-vision-preview", description="Vision LLM model")
    ONYX_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    ONYX_MAX_TOKENS: int = Field(default=4000, ge=1, le=32000, description="LLM max tokens")
    
    # Performance environment variables
    ONYX_MAX_WORKERS: int = Field(default=10, ge=1, le=50, description="Maximum workers")
    ONYX_THREAD_TIMEOUT: int = Field(default=300, ge=30, le=3600, description="Thread timeout")
    ONYX_ENABLE_GPU: bool = Field(default=True, description="Enable GPU")
    ONYX_CACHE_ENABLED: bool = Field(default=True, description="Enable caching")
    
    # Security environment variables
    ONYX_ENCRYPTION_KEY: Optional[str] = Field(default=None, description="Encryption key")
    ONYX_VALIDATE_ACCESS: bool = Field(default=True, description="Validate access")
    
    # Integration flags
    ONYX_USE_LOGGING: bool = Field(default=True, description="Use Onyx logging")
    ONYX_USE_LLM: bool = Field(default=True, description="Use Onyx LLM")
    ONYX_USE_TELEMETRY: bool = Field(default=True, description="Use Onyx telemetry")
    ONYX_USE_ENCRYPTION: bool = Field(default=True, description="Use Onyx encryption")
    ONYX_USE_THREADING: bool = Field(default=True, description="Use Onyx threading")
    ONYX_USE_RETRY: bool = Field(default=True, description="Use Onyx retry")
    ONYX_USE_GPU: bool = Field(default=True, description="Use Onyx GPU")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "ONYX_"
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_onyx_settings() -> OnyxSettings:
    """Load Onyx settings from environment and configuration."""
    # Load environment configuration
    env_config = OnyxEnvironmentConfig()
    
    # Create settings from environment
    settings = OnyxSettings(
        use_onyx_logging=env_config.ONYX_USE_LOGGING,
        use_onyx_llm=env_config.ONYX_USE_LLM,
        use_onyx_telemetry=env_config.ONYX_USE_TELEMETRY,
        use_onyx_encryption=env_config.ONYX_USE_ENCRYPTION,
        use_onyx_threading=env_config.ONYX_USE_THREADING,
        use_onyx_retry=env_config.ONYX_USE_RETRY,
        use_onyx_gpu=env_config.ONYX_USE_GPU,
        onyx_root_path=env_config.ONYX_ROOT,
        onyx_config_path=env_config.ONYX_CONFIG,
        onyx_plugins_path=env_config.ONYX_PLUGINS,
        onyx_default_llm=env_config.ONYX_DEFAULT_LLM,
        onyx_vision_llm=env_config.ONYX_VISION_LLM,
        onyx_temperature=env_config.ONYX_TEMPERATURE,
        onyx_max_tokens=env_config.ONYX_MAX_TOKENS,
        onyx_max_workers=env_config.ONYX_MAX_WORKERS,
        onyx_thread_timeout=env_config.ONYX_THREAD_TIMEOUT,
        onyx_encryption_key=env_config.ONYX_ENCRYPTION_KEY,
        onyx_validate_access=env_config.ONYX_VALIDATE_ACCESS,
        onyx_enable_gpu=env_config.ONYX_ENABLE_GPU,
        onyx_cache_enabled=env_config.ONYX_CACHE_ENABLED,
        onyx_metrics_enabled=True,  # Always enabled for Onyx
    )
    
    return settings


def get_onyx_settings() -> OnyxSettings:
    """Get Onyx settings singleton."""
    if not hasattr(get_onyx_settings, '_instance'):
        get_onyx_settings._instance = load_onyx_settings()
    return get_onyx_settings._instance


def update_onyx_settings(updates: Dict[str, Any]) -> OnyxSettings:
    """Update Onyx settings."""
    settings = get_onyx_settings()
    
    for key, value in updates.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    
    return settings


def validate_onyx_integration() -> Dict[str, Any]:
    """Validate Onyx integration and return status."""
    settings = get_onyx_settings()
    status = {
        "onyx_available": False,
        "modules_available": {},
        "paths_valid": {},
        "integration_ready": False,
        "errors": []
    }
    
    try:
        # Check if Onyx modules are available
        onyx_modules = [
            "onyx.core.functions",
            "onyx.utils.logger",
            "onyx.utils.threadpool_concurrency",
            "onyx.utils.timing",
            "onyx.utils.retry_wrapper",
            "onyx.utils.telemetry",
            "onyx.utils.encryption",
            "onyx.utils.file",
            "onyx.utils.text_processing",
            "onyx.utils.gpu_utils",
            "onyx.utils.error_handling",
            "onyx.llm.factory",
            "onyx.llm.interfaces",
            "onyx.llm.utils",
            "onyx.db.engine",
            "onyx.db.models"
        ]
        
        for module in onyx_modules:
            try:
                __import__(module)
                status["modules_available"][module] = True
            except ImportError:
                status["modules_available"][module] = False
                status["errors"].append(f"Module not available: {module}")
        
        # Check paths
        if settings.onyx_root_path:
            status["paths_valid"]["root"] = os.path.exists(settings.onyx_root_path)
            if not status["paths_valid"]["root"]:
                status["errors"].append(f"Onyx root path not found: {settings.onyx_root_path}")
        
        if settings.onyx_config_path:
            status["paths_valid"]["config"] = os.path.exists(settings.onyx_config_path)
            if not status["paths_valid"]["config"]:
                status["errors"].append(f"Onyx config path not found: {settings.onyx_config_path}")
        
        if settings.onyx_plugins_path:
            status["paths_valid"]["plugins"] = os.path.exists(settings.onyx_plugins_path)
            if not status["paths_valid"]["plugins"]:
                status["errors"].append(f"Onyx plugins path not found: {settings.onyx_plugins_path}")
        
        # Determine overall availability
        core_modules_available = all([
            status["modules_available"].get("onyx.core.functions", False),
            status["modules_available"].get("onyx.utils.logger", False),
            status["modules_available"].get("onyx.llm.factory", False)
        ])
        
        status["onyx_available"] = core_modules_available
        status["integration_ready"] = core_modules_available and len(status["errors"]) == 0
        
    except Exception as e:
        status["errors"].append(f"Validation error: {e}")
    
    return status


def get_onyx_environment_info() -> Dict[str, Any]:
    """Get Onyx environment information."""
    settings = get_onyx_settings()
    
    info = {
        "settings": settings.to_dict(),
        "environment_variables": {
            "ONYX_ROOT": os.getenv("ONYX_ROOT"),
            "ONYX_HOME": os.getenv("ONYX_HOME"),
            "ONYX_CONFIG": os.getenv("ONYX_CONFIG"),
            "ONYX_PLUGINS": os.getenv("ONYX_PLUGINS"),
            "ONYX_DEFAULT_LLM": os.getenv("ONYX_DEFAULT_LLM"),
            "ONYX_VISION_LLM": os.getenv("ONYX_VISION_LLM"),
            "ONYX_TEMPERATURE": os.getenv("ONYX_TEMPERATURE"),
            "ONYX_MAX_TOKENS": os.getenv("ONYX_MAX_TOKENS"),
            "ONYX_MAX_WORKERS": os.getenv("ONYX_MAX_WORKERS"),
            "ONYX_THREAD_TIMEOUT": os.getenv("ONYX_THREAD_TIMEOUT"),
            "ONYX_ENABLE_GPU": os.getenv("ONYX_ENABLE_GPU"),
            "ONYX_CACHE_ENABLED": os.getenv("ONYX_CACHE_ENABLED"),
            "ONYX_ENCRYPTION_KEY": "***" if os.getenv("ONYX_ENCRYPTION_KEY") else None,
            "ONYX_VALIDATE_ACCESS": os.getenv("ONYX_VALIDATE_ACCESS"),
            "ONYX_USE_LOGGING": os.getenv("ONYX_USE_LOGGING"),
            "ONYX_USE_LLM": os.getenv("ONYX_USE_LLM"),
            "ONYX_USE_TELEMETRY": os.getenv("ONYX_USE_TELEMETRY"),
            "ONYX_USE_ENCRYPTION": os.getenv("ONYX_USE_ENCRYPTION"),
            "ONYX_USE_THREADING": os.getenv("ONYX_USE_THREADING"),
            "ONYX_USE_RETRY": os.getenv("ONYX_USE_RETRY"),
            "ONYX_USE_GPU": os.getenv("ONYX_USE_GPU"),
        },
        "current_working_directory": os.getcwd(),
        "python_path": os.getenv("PYTHONPATH", "").split(os.pathsep),
        "sys_path": [str(p) for p in __import__("sys").path],
    }
    
    return info


def setup_onyx_environment() -> bool:
    """Setup Onyx environment for AI Video system."""
    try:
        settings = get_onyx_settings()
        
        # Set environment variables if not already set
        if settings.onyx_root_path and not os.getenv("ONYX_ROOT"):
            os.environ["ONYX_ROOT"] = settings.onyx_root_path
        
        if settings.onyx_config_path and not os.getenv("ONYX_CONFIG"):
            os.environ["ONYX_CONFIG"] = settings.onyx_config_path
        
        if settings.onyx_plugins_path and not os.getenv("ONYX_PLUGINS"):
            os.environ["ONYX_PLUGINS"] = settings.onyx_plugins_path
        
        # Set default LLM if not set
        if not os.getenv("ONYX_DEFAULT_LLM"):
            os.environ["ONYX_DEFAULT_LLM"] = settings.onyx_default_llm
        
        # Set vision LLM if not set
        if settings.onyx_vision_llm and not os.getenv("ONYX_VISION_LLM"):
            os.environ["ONYX_VISION_LLM"] = settings.onyx_vision_llm
        
        # Set performance settings
        if not os.getenv("ONYX_MAX_WORKERS"):
            os.environ["ONYX_MAX_WORKERS"] = str(settings.onyx_max_workers)
        
        if not os.getenv("ONYX_THREAD_TIMEOUT"):
            os.environ["ONYX_THREAD_TIMEOUT"] = str(settings.onyx_thread_timeout)
        
        # Set integration flags
        if not os.getenv("ONYX_USE_LOGGING"):
            os.environ["ONYX_USE_LOGGING"] = str(settings.use_onyx_logging).lower()
        
        if not os.getenv("ONYX_USE_LLM"):
            os.environ["ONYX_USE_LLM"] = str(settings.use_onyx_llm).lower()
        
        if not os.getenv("ONYX_USE_TELEMETRY"):
            os.environ["ONYX_USE_TELEMETRY"] = str(settings.use_onyx_telemetry).lower()
        
        if not os.getenv("ONYX_USE_ENCRYPTION"):
            os.environ["ONYX_USE_ENCRYPTION"] = str(settings.use_onyx_encryption).lower()
        
        if not os.getenv("ONYX_USE_THREADING"):
            os.environ["ONYX_USE_THREADING"] = str(settings.use_onyx_threading).lower()
        
        if not os.getenv("ONYX_USE_RETRY"):
            os.environ["ONYX_USE_RETRY"] = str(settings.use_onyx_retry).lower()
        
        if not os.getenv("ONYX_USE_GPU"):
            os.environ["ONYX_USE_GPU"] = str(settings.use_onyx_gpu).lower()
        
        return True
        
    except Exception as e:
        print(f"Failed to setup Onyx environment: {e}")
        return False


# Default Onyx settings
DEFAULT_ONYX_SETTINGS = {
    "use_onyx_logging": True,
    "use_onyx_llm": True,
    "use_onyx_telemetry": True,
    "use_onyx_encryption": True,
    "use_onyx_threading": True,
    "use_onyx_retry": True,
    "use_onyx_gpu": True,
    "onyx_default_llm": "gpt-4",
    "onyx_vision_llm": "gpt-4-vision-preview",
    "onyx_temperature": 0.7,
    "onyx_max_tokens": 4000,
    "onyx_max_workers": 10,
    "onyx_thread_timeout": 300,
    "onyx_validate_access": True,
    "onyx_enable_gpu": True,
    "onyx_cache_enabled": True,
    "onyx_metrics_enabled": True,
}


def create_onyx_env_file(env_path: str = ".env.onyx") -> None:
    """Create Onyx environment file with default settings."""
    env_content = """# Onyx AI Video System Environment Configuration

# Onyx Paths
ONYX_ROOT=
ONYX_HOME=
ONYX_CONFIG=
ONYX_PLUGINS=

# Onyx LLM Configuration
ONYX_DEFAULT_LLM=gpt-4
ONYX_VISION_LLM=gpt-4-vision-preview
ONYX_TEMPERATURE=0.7
ONYX_MAX_TOKENS=4000

# Onyx Performance Configuration
ONYX_MAX_WORKERS=10
ONYX_THREAD_TIMEOUT=300
ONYX_ENABLE_GPU=true
ONYX_CACHE_ENABLED=true

# Onyx Security Configuration
ONYX_ENCRYPTION_KEY=
ONYX_VALIDATE_ACCESS=true

# Onyx Integration Flags
ONYX_USE_LOGGING=true
ONYX_USE_LLM=true
ONYX_USE_TELEMETRY=true
ONYX_USE_ENCRYPTION=true
ONYX_USE_THREADING=true
ONYX_USE_RETRY=true
ONYX_USE_GPU=true
"""
    
    try:
        with open(env_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(env_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        print(f"Onyx environment file created: {env_path}")
    except Exception as e:
        print(f"Failed to create Onyx environment file: {e}")


# Initialize settings on module import
_onyx_settings = None

def get_onyx_settings_singleton() -> OnyxSettings:
    """Get Onyx settings singleton instance."""
    global _onyx_settings
    if _onyx_settings is None:
        _onyx_settings = load_onyx_settings()
    return _onyx_settings 
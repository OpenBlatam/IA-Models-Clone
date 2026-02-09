from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import time
import json
import logging
from typing import Any, List, Dict, Optional
"""
Core cybersecurity utilities and base classes.
Shared components used across all cybersecurity modules.
"""


@dataclass
class BaseConfig:
    """Base configuration class for all cybersecurity tools."""
    timeout: float = 10.0
    max_retries: int = 3
    enable_logging: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self) -> Any:
        """Initialize logging if enabled."""
        if self.enable_logging:
            logging.basicConfig(
                level=getattr(logging, self.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

@dataclass
class SecurityEvent:
    """Base security event class."""
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""
    severity: str = "info"
    description: str = ""
    source: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScanResult:
    """Base scan result class."""
    target: str
    success: bool = False
    response_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Export all classes for proper module imports
__all__ = [
    'BaseConfig',
    'SecurityEvent', 
    'ScanResult',
    'BaseScanner',
    'BaseValidator',
    'BaseMonitor'
]

class BaseScanner:
    """Base scanner class with common functionality."""
    
    def __init__(self, config: BaseConfig):
        
    """__init__ function."""
self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def scan_with_retry(self, scan_func, *args, **kwargs) -> Any:
        """Execute scan with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                result = await scan_func(*args, **kwargs)
                result.response_time = time.time() - start_time
                return result
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    return ScanResult(
                        target=str(args[0]) if args else "unknown",
                        success=False,
                        error_message=str(e)
                    )
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

class BaseValidator:
    """Base validator class with common validation logic."""
    
    def __init__(self, config: BaseConfig):
        
    """__init__ function."""
self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_required_fields(self, data: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """Validate that required fields are present."""
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)
        return missing_fields
    
    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """Basic string sanitization."""
        if not isinstance(value, str):
            return ""
        return value.strip()[:max_length]

class BaseMonitor:
    """Base monitor class with common monitoring functionality."""
    
    def __init__(self, config: BaseConfig):
        
    """__init__ function."""
self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.events: List[SecurityEvent] = []
    
    def add_event(self, event: SecurityEvent):
        """Add security event to monitor."""
        self.events.append(event)
        if self.config.enable_logging:
            self.logger.info(f"Security event: {event.event_type} - {event.description}")
    
    def get_events_by_type(self, event_type: str) -> List[SecurityEvent]:
        """Get events filtered by type."""
        return [event for event in self.events if event.event_type == event_type]
    
    def get_events_by_severity(self, severity: str) -> List[SecurityEvent]:
        """Get events filtered by severity."""
        return [event for event in self.events if event.severity == severity]

# Utility functions
def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for logging."""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def calculate_percentage(current: float, total: float) -> float:
    """Calculate percentage safely."""
    if total == 0:
        return 0.0
    return (current / total) * 100

def is_valid_ip(ip: str) -> bool:
    """Basic IP validation."""
    try:
        parts = ip.split('.')
        return len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts)
    except (ValueError, AttributeError):
        return False

def is_valid_port(port: int) -> bool:
    """Validate port number."""
    return 1 <= port <= 65535

# Named exports
__all__ = [
    'BaseConfig',
    'SecurityEvent', 
    'ScanResult',
    'BaseScanner',
    'BaseValidator',
    'BaseMonitor',
    'format_timestamp',
    'calculate_percentage',
    'is_valid_ip',
    'is_valid_port'
] 
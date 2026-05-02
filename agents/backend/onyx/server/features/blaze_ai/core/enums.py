from enum import Enum, auto

class SystemMode(Enum):
    """System operation modes."""
    DEVELOPMENT = auto()
    TESTING = auto()
    STAGING = auto()
    PRODUCTION = auto()
    PERFORMANCE = auto()

class OptimizationLevel(Enum):
    """Performance optimization levels."""
    BASIC = auto()
    STANDARD = auto()
    ADVANCED = auto()
    TURBO = auto()
    MARAREAL = auto()
    QUANTUM = auto()

class ComponentStatus(Enum):
    """Component operational status."""
    INITIALIZING = auto()
    ACTIVE = auto()
    IDLE = auto()
    ERROR = auto()
    SHUTDOWN = auto()

class PerformanceLevel(Enum):
    """Performance target levels."""
    STANDARD = "standard"
    ENHANCED = "enhanced"
    TURBO = "turbo"
    MARAREAL = "marareal"
    QUANTUM = "quantum"

# From .health import ComponentType (needs to be handled carefully to avoid circular imports if health imports this)
# Assuming ComponentType is in health.py and can be imported here or kept there.
# The original __init__.py imported it: from .health import ComponentType

"""
Logging utilities
"""

from utils.categories import register_utility

try:
    from utils.logging_utils import LoggingUtils
    from utils.logging_config import LoggingConfig
    from utils.advanced_logging import AdvancedLogging
    
    def register_utilities():
        register_utility("logging", "utils", LoggingUtils)
        register_utility("logging", "config", LoggingConfig)
        register_utility("logging", "advanced", AdvancedLogging)
except ImportError:
    pass




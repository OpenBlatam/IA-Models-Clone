"""
Serialization and compression utilities
"""

from utils.categories import register_utility

try:
    from utils.serialization import Serialization
    from utils.compression import Compression
    from utils.advanced_compression import AdvancedCompression
    from utils.parsers import Parsers
    from utils.formatters import Formatters
    from utils.type_converters import TypeConverters
    
    def register_utilities():
        register_utility("serialization", "serialization", Serialization)
        register_utility("serialization", "compression", Compression)
        register_utility("serialization", "advanced_compression", AdvancedCompression)
        register_utility("serialization", "parsers", Parsers)
        register_utility("serialization", "formatters", Formatters)
        register_utility("serialization", "type_converters", TypeConverters)
except ImportError:
    pass




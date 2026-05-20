"""
Security utilities
"""

from utils.categories import register_utility

try:
    from utils.security import Security
    from utils.hashers import Hashers
    from utils.encoders import Encoders
    from utils.sanitizers import Sanitizers
    
    def register_utilities():
        register_utility("security", "security", Security)
        register_utility("security", "hashers", Hashers)
        register_utility("security", "encoders", Encoders)
        register_utility("security", "sanitizers", Sanitizers)
except ImportError:
    pass




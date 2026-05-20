"""
Rate limiting and throttling utilities
"""

from utils.categories import register_utility

try:
    from utils.rate_limiter_advanced import RateLimiterAdvanced
    from utils.throttlers import Throttlers
    
    def register_utilities():
        register_utility("rate_limiting", "advanced", RateLimiterAdvanced)
        register_utility("rate_limiting", "throttlers", Throttlers)
except ImportError:
    pass




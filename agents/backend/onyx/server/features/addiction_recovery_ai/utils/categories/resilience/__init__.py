"""
Resilience and reliability utilities
"""

from utils.categories import register_utility

try:
    from utils.circuit_breakers import CircuitBreakers
    from utils.retry_strategies import RetryStrategies
    from utils.backpressure import Backpressure
    from utils.error_recovery import ErrorRecovery
    from utils.error_handler import ErrorHandler
    from utils.errors import Errors
    
    def register_utilities():
        register_utility("resilience", "circuit_breakers", CircuitBreakers)
        register_utility("resilience", "retry", RetryStrategies)
        register_utility("resilience", "backpressure", Backpressure)
        register_utility("resilience", "error_recovery", ErrorRecovery)
        register_utility("resilience", "error_handler", ErrorHandler)
        register_utility("resilience", "errors", Errors)
except ImportError:
    pass




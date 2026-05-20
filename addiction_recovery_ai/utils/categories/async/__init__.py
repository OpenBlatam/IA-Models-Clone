"""
Async and concurrency utilities
"""

from utils.categories import register_utility

try:
    from utils.async_helpers import AsyncHelpers
    from utils.async_composers import AsyncComposers
    from utils.async_inference import AsyncInference
    from utils.futures import Futures
    from utils.promises import Promises
    from utils.workers import Workers
    from utils.pools import Pools
    from utils.semaphores import Semaphores
    
    def register_utilities():
        register_utility("async", "helpers", AsyncHelpers)
        register_utility("async", "composers", AsyncComposers)
        register_utility("async", "inference", AsyncInference)
        register_utility("async", "futures", Futures)
        register_utility("async", "promises", Promises)
        register_utility("async", "workers", Workers)
        register_utility("async", "pools", Pools)
        register_utility("async", "semaphores", Semaphores)
except ImportError:
    pass




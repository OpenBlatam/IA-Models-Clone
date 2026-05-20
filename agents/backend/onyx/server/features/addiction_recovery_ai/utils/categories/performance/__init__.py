"""
Performance optimization utilities
"""

from utils.categories import register_utility

try:
    from utils.performance_helpers import PerformanceHelpers
    from utils.fast_inference import FastInference
    from utils.cache import Cache
    from utils.advanced_caching import AdvancedCaching
    from utils.cache_advanced import CacheAdvanced
    from utils.memoization import Memoization
    from utils.precomputation import Precomputation
    from utils.profiler import Profiler
    from utils.benchmarking import Benchmarking
    
    def register_utilities():
        register_utility("performance", "helpers", PerformanceHelpers)
        register_utility("performance", "fast_inference", FastInference)
        register_utility("performance", "cache", Cache)
        register_utility("performance", "advanced_caching", AdvancedCaching)
        register_utility("performance", "cache_advanced", CacheAdvanced)
        register_utility("performance", "memoization", Memoization)
        register_utility("performance", "precomputation", Precomputation)
        register_utility("performance", "profiler", Profiler)
        register_utility("performance", "benchmarking", Benchmarking)
except ImportError:
    pass




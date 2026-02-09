#!/usr/bin/env python3
"""
Performance Profiling Script
Profiles application code to identify bottlenecks
"""

import cProfile
import pstats
import io
from pathlib import Path


def profile_function(func, *args, **kwargs):
    """Profile a function"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    
    return result, profiler


def print_profile_stats(profiler, sort_by='cumulative', lines=50):
    """Print profiling statistics"""
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats(sort_by)
    ps.print_stats(lines)
    
    print(s.getvalue())


def save_profile_stats(profiler, output_file: str):
    """Save profiling statistics to file"""
    with open(output_file, 'w') as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.sort_stats('cumulative')
        ps.print_stats()
    
    print(f"Profile saved to {output_file}")


def profile_endpoint(endpoint_func):
    """Decorator to profile an endpoint"""
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = endpoint_func(*args, **kwargs)
        
        profiler.disable()
        
        # Save to file
        output_file = f"profile_{endpoint_func.__name__}.txt"
        save_profile_stats(profiler, output_file)
        
        return result
    
    return wrapper


if __name__ == "__main__":
    print("Profiling utilities loaded.")
    print("Import this module to use profiling decorators and functions.")



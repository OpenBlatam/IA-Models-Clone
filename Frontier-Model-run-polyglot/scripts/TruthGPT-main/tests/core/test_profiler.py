"""
Test Profiler
Profiles test execution to identify slow tests
"""

import time
import cProfile
import pstats
import io
from pathlib import Path
from typing import Dict, List, Any, Optional
from functools import wraps

class TestProfiler:
    """Profile test execution to find performance bottlenecks"""
    
    def __init__(self):
        self.profiles: Dict[str, cProfile.Profile] = {}
        self.test_timings: Dict[str, float] = {}
    
    def profile_test(self, test_name: str):
        """Decorator to profile a test method"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                profiler = cProfile.Profile()
                start_time = time.time()
                
                profiler.enable()
                try:
                    result = func(*args, **kwargs)
                finally:
                    profiler.disable()
                    end_time = time.time()
                
                self.profiles[test_name] = profiler
                self.test_timings[test_name] = end_time - start_time
                
                return result
            return wrapper
        return decorator
    
    def get_slow_tests(self, threshold: float = 1.0) -> List[tuple]:
        """Get tests that take longer than threshold seconds"""
        return sorted(
            [(name, time) for name, time in self.test_timings.items() if time > threshold],
            key=lambda x: x[1],
            reverse=True
        )
    
    def generate_profile_report(self, test_name: str, output_file: Optional[str] = None) -> str:
        """Generate profiling report for a test"""
        if test_name not in self.profiles:
            return f"No profile data for {test_name}"
        
        profiler = self.profiles[test_name]
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        report = s.getvalue()
        
        if output_file:
            Path(output_file).write_text(report)
        
        return report
    
    def generate_summary_report(self) -> str:
        """Generate summary of all profiled tests"""
        if not self.test_timings:
            return "No tests profiled yet"
        
        lines = ["Test Profiling Summary", "=" * 60, ""]
        
        # Sort by execution time
        sorted_tests = sorted(self.test_timings.items(), key=lambda x: x[1], reverse=True)
        
        lines.append(f"{'Test Name':<50} {'Time (s)':>10}")
        lines.append("-" * 60)
        
        for test_name, exec_time in sorted_tests:
            lines.append(f"{test_name:<50} {exec_time:>10.3f}")
        
        total_time = sum(self.test_timings.values())
        lines.append("-" * 60)
        lines.append(f"{'Total':<50} {total_time:>10.3f}")
        lines.append(f"{'Average':<50} {total_time/len(self.test_timings):>10.3f}")
        
        return "\n".join(lines)








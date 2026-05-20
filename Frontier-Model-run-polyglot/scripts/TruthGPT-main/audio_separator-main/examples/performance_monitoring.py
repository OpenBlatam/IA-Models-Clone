"""
Examples of performance monitoring.
"""

from audio_separator import AudioSeparator
from audio_separator.utils.performance import (
    timeit,
    timer,
    PerformanceMonitor,
    performance_monitor
)


@timeit
def example_timed_function():
    """Example function with timing decorator."""
    import time
    time.sleep(0.1)  # Simulate work
    return "done"


def example_context_manager():
    """Example using timer context manager."""
    with timer("Processing audio"):
        import time
        time.sleep(0.1)  # Simulate work


def example_performance_monitor():
    """Example using performance monitor."""
    print("Example: Performance Monitoring")
    print("-" * 50)
    
    monitor = PerformanceMonitor()
    
    # Monitor multiple operations
    for i in range(3):
        monitor.start("operation_1")
        import time
        time.sleep(0.05)
        monitor.stop("operation_1")
        
        monitor.start("operation_2")
        time.sleep(0.03)
        monitor.stop("operation_2")
    
    # Print statistics
    monitor.print_summary()
    print()


def example_separation_with_monitoring():
    """Example of separation with performance monitoring."""
    print("Example: Separation with Monitoring")
    print("-" * 50)
    
    separator = AudioSeparator(model_type="demucs")
    
    with timer("Full separation"):
        with timer("Loading audio"):
            # This would normally load audio
            import time
            time.sleep(0.05)
        
        with timer("Model inference"):
            # This would normally run model
            time.sleep(0.1)
        
        with timer("Saving outputs"):
            # This would normally save files
            time.sleep(0.03)
    
    print()


if __name__ == "__main__":
    example_timed_function()
    example_context_manager()
    example_performance_monitor()
    example_separation_with_monitoring()


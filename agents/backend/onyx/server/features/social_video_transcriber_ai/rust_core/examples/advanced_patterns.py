"""
Advanced Design Patterns Examples

Demonstrates event-driven architecture, middleware, observer pattern, and plugins.
"""

from transcriber_core import (
    EventBus, MiddlewareChain, Observable, PluginManager,
    CacheService, Profiler
)

def example_event_system():
    """Example: Event-driven architecture"""
    bus = EventBus()
    
    # Subscribe to events
    def on_cache_hit(event):
        print(f"Cache hit: {event}")
    
    def on_cache_miss(event):
        print(f"Cache miss: {event}")
    
    bus.on("cache_hit", on_cache_hit)
    bus.on("cache_miss", on_cache_miss)
    
    # Use cache with events
    cache = CacheService(1000, 3600)
    
    # Emit events
    cache.set("key1", "value1")
    bus.emit({"type": "cache_hit", "key": "key1"})
    
    cache.get("key2")  # Miss
    bus.emit({"type": "cache_miss", "key": "key2"})

def example_middleware():
    """Example: Middleware pattern"""
    chain = MiddlewareChain()
    
    # Add middleware
    def logging_middleware(context, next_fn):
        print(f"Request started: {context.request_id}")
        result = next_fn(context)
        print(f"Request completed: {context.request_id}")
        return result
    
    def timing_middleware(context, next_fn):
        import time
        start = time.time()
        result = next_fn(context)
        elapsed = time.time() - start
        context.metadata["duration_ms"] = elapsed * 1000
        return result
    
    chain.add(logging_middleware)
    chain.add(timing_middleware)
    
    # Execute with middleware
    context = {"request_id": "req-123", "metadata": {}}
    result = chain.execute(context)
    print(f"Duration: {context['metadata'].get('duration_ms')}ms")

def example_observer():
    """Example: Observer pattern"""
    # Create observable
    observable = Observable({"count": 0})
    
    # Create observers
    def observer1(data):
        print(f"Observer 1: count is now {data.get('count')}")
    
    def observer2(data):
        print(f"Observer 2: count is now {data.get('count')}")
    
    observable.attach(observer1)
    observable.attach(observer2)
    
    # Update state (triggers observers)
    observable.set_state({"count": 1})
    observable.set_state({"count": 2})
    observable.set_state({"count": 3})

def example_plugins():
    """Example: Plugin system"""
    manager = PluginManager()
    
    # Define a plugin
    class CachePlugin:
        def __init__(self):
            self.name = "cache_plugin"
            self.version = "1.0.0"
        
        def execute(self, data):
            # Process data
            return {"processed": True, "data": data}
    
    # Register plugin
    manager.register(CachePlugin())
    
    # List plugins
    plugins = manager.list_plugins()
    print(f"Registered plugins: {plugins}")
    
    # Execute plugin
    result = manager.execute_plugin("cache_plugin", {"key": "value"})
    print(f"Plugin result: {result}")

def example_integrated():
    """Example: Integrated patterns"""
    # Create components
    bus = EventBus()
    chain = MiddlewareChain()
    observable = Observable({"status": "idle"})
    profiler = Profiler()
    
    # Setup event handlers
    def on_status_change(event):
        print(f"Status changed: {event}")
    
    bus.on("status_change", on_status_change)
    
    # Setup middleware
    def profiling_middleware(context, next_fn):
        profiler.start_timer("operation")
        result = next_fn(context)
        profiler.record_time("operation", 10.5)
        return result
    
    chain.add(profiling_middleware)
    
    # Use together
    context = {"request_id": "req-1", "metadata": {}}
    chain.execute(context)
    
    # Update observable
    observable.set_state({"status": "processing"})
    bus.emit({"type": "status_change", "status": "processing"})
    
    # Get profiling stats
    stats = profiler.get_stats()
    print(f"Profiling stats: {stats}")

if __name__ == "__main__":
    print("=== Event System Example ===")
    example_event_system()
    
    print("\n=== Middleware Example ===")
    example_middleware()
    
    print("\n=== Observer Pattern Example ===")
    example_observer()
    
    print("\n=== Plugin System Example ===")
    example_plugins()
    
    print("\n=== Integrated Patterns Example ===")
    example_integrated()













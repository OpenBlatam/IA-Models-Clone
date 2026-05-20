
import sys
import os
import asyncio

# Setup path to import blaze_ai as a package
# We are in .../features/blaze_ai/core
# We want to add .../features to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
features_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, features_dir)

print(f"Added {features_dir} to sys.path")

try:
    from blaze_ai.core import (
        BlazeAISystem, 
        create_development_config,
        SystemMode,
        PerformanceLevel
    )
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

async def test_system_initialization():
    print("Creating config...")
    config = create_development_config()
    print(f"Config created: {config.system_mode}, {config.performance_target}")
    
    print("Instantiating system...")
    system = BlazeAISystem(config)
    print("System instantiated.")
    
    print("Initializing system...")
    # mocking utility optimizations to avoid dependency issues during this specific test if they don't exist
    # But system.py handles import errors gracefully, so we should be fine.
    success = await system.initialize()
    if success:
        print("System initialized successfully.")
    else:
        print("System initialization failed.")
        
    print("System Status:", await system.get_status())
    
    print("Shutting down...")
    await system.shutdown()
    print("Shutdown complete.")

if __name__ == "__main__":
    asyncio.run(test_system_initialization())

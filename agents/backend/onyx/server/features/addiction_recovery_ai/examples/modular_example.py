"""
Example: Using the Modular Architecture
Demonstrates how to use the module system
"""

import asyncio
from core.module_loader import get_loader
from core.module_registry import get_registry


async def example_basic_usage():
    """Basic module usage example"""
    print("=== Basic Module Usage ===\n")
    
    # Get loader and registry
    loader = get_loader()
    registry = get_registry()
    
    # Load default modules
    print("Loading modules...")
    loader.load_default_modules()
    
    # List loaded modules
    print(f"Loaded modules: {loader.get_loaded_modules()}\n")
    
    # Initialize all modules
    print("Initializing modules...")
    loader.initialize_all()
    
    # Get a specific module
    storage_module = registry.get_module("storage")
    if storage_module:
        print(f"Storage module: {storage_module.name} v{storage_module.version}")
        print(f"Initialized: {storage_module.is_initialized()}\n")
    
    # Use module services
    if storage_module:
        storage = storage_module.get_storage_service()
        print(f"Storage service: {type(storage).__name__}\n")
    
    # Shutdown
    print("Shutting down modules...")
    loader.shutdown_all()
    print("Done!\n")


async def example_custom_module():
    """Example of creating and using a custom module"""
    print("=== Custom Module Example ===\n")
    
    from modules.base_module import BaseModule
    from typing import List
    
    # Create custom module
    class CustomFeatureModule(BaseModule):
        def __init__(self):
            super().__init__("custom_feature", "1.0.0")
            self._data = {}
        
        def get_dependencies(self) -> List[str]:
            return ["storage", "cache"]
        
        def _on_initialize(self) -> None:
            print(f"Custom module {self.name} initializing...")
            self._data["initialized"] = True
        
        def _on_shutdown(self) -> None:
            print(f"Custom module {self.name} shutting down...")
            self._data.clear()
        
        def get_data(self):
            return self._data
    
    # Register and use
    registry = get_registry()
    custom_module = CustomFeatureModule()
    registry.register(custom_module)
    
    # Initialize (dependencies will be initialized first)
    registry.initialize_all()
    
    # Use module
    print(f"Custom module data: {custom_module.get_data()}\n")
    
    # Shutdown
    registry.shutdown_all()
    print("Done!\n")


async def example_selective_loading():
    """Example of loading only specific modules"""
    print("=== Selective Module Loading ===\n")
    
    loader = get_loader()
    
    # Load only storage and cache
    loader.load_module("modules.storage_module.StorageModule")
    loader.load_module("modules.cache_module.CacheModule")
    
    print(f"Loaded modules: {loader.get_loaded_modules()}\n")
    
    # Initialize
    loader.initialize_all()
    
    # Shutdown
    loader.shutdown_all()
    print("Done!\n")


async def example_module_dependencies():
    """Example showing dependency resolution"""
    print("=== Module Dependencies Example ===\n")
    
    from modules.base_module import BaseModule
    from typing import List
    
    # Module A depends on B
    class ModuleA(BaseModule):
        def __init__(self):
            super().__init__("module_a", "1.0.0")
        
        def get_dependencies(self) -> List[str]:
            return ["module_b"]
    
    # Module B depends on C
    class ModuleB(BaseModule):
        def __init__(self):
            super().__init__("module_b", "1.0.0")
        
        def get_dependencies(self) -> List[str]:
            return ["module_c"]
    
    # Module C has no dependencies
    class ModuleC(BaseModule):
        def __init__(self):
            super().__init__("module_c", "1.0.0")
    
    # Register in any order
    registry = get_registry()
    registry.register(ModuleA())
    registry.register(ModuleB())
    registry.register(ModuleC())
    
    # Initialize - will resolve dependencies automatically
    print("Initializing modules (dependencies will be resolved)...")
    registry.initialize_all()
    
    # Check initialization order
    print(f"Initialization order: {registry._load_order}\n")
    # Should be: ['module_c', 'module_b', 'module_a']
    
    # Shutdown
    registry.shutdown_all()
    print("Done!\n")


async def main():
    """Run all examples"""
    print("=" * 60)
    print("Modular Architecture Examples")
    print("=" * 60)
    print()
    
    await example_basic_usage()
    await example_custom_module()
    await example_selective_loading()
    await example_module_dependencies()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
















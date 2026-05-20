from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any
from plugins import (
from plugins.examples import WebExtractorPlugin
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Plugin System Demo

This script demonstrates the complete plugin system with:
- Plugin discovery and loading
- Configuration management
- Error handling and recovery
- Performance monitoring
- Event handling
- Health reporting

Usage:
    python demo.py
"""


# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

    PluginManager, 
    ManagerConfig, 
    ValidationLevel,
    quick_start,
    create_plugin_manager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('plugin_demo.log')
    ]
)

logger = logging.getLogger(__name__)


class DemoPluginManager:
    """Demo class that showcases the plugin system capabilities."""
    
    def __init__(self) -> Any:
        self.manager = None
        self.demo_plugins = []
    
    async def run_demo(self) -> Any:
        """Run the complete plugin system demo."""
        print("🎯 AI Video Plugin System Demo")
        print("=" * 50)
        
        try:
            # Step 1: Create and start plugin manager
            await self._demo_manager_creation()
            
            # Step 2: Plugin discovery
            await self._demo_plugin_discovery()
            
            # Step 3: Plugin loading and configuration
            await self._demo_plugin_loading()
            
            # Step 4: Plugin lifecycle management
            await self._demo_lifecycle_management()
            
            # Step 5: Error handling and recovery
            await self._demo_error_handling()
            
            # Step 6: Performance monitoring
            await self._demo_performance_monitoring()
            
            # Step 7: Event handling
            await self._demo_event_handling()
            
            # Step 8: Health reporting
            await self._demo_health_reporting()
            
            # Step 9: Advanced features
            await self._demo_advanced_features()
            
            print("\n✅ Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"\n❌ Demo failed: {e}")
        
        finally:
            # Cleanup
            await self._cleanup()
    
    async def _demo_manager_creation(self) -> Any:
        """Demonstrate plugin manager creation."""
        print("\n1️⃣ Plugin Manager Creation")
        print("-" * 30)
        
        # Method 1: Quick start
        print("Creating plugin manager with quick_start()...")
        self.manager = await quick_start()
        print("✅ Plugin manager created and started")
        
        # Show configuration
        config = self.manager.config
        print(f"Configuration:")
        print(f"  - Auto discover: {config.auto_discover}")
        print(f"  - Auto load: {config.auto_load}")
        print(f"  - Auto initialize: {config.auto_initialize}")
        print(f"  - Validation level: {config.validation_level.value}")
        print(f"  - Plugin directories: {config.plugin_dirs}")
    
    async def _demo_plugin_discovery(self) -> Any:
        """Demonstrate plugin discovery."""
        print("\n2️⃣ Plugin Discovery")
        print("-" * 30)
        
        # Discover plugins
        print("Discovering plugins...")
        discovered = await self.manager.discover_plugins()
        
        print(f"Found {len(discovered)} plugins:")
        for plugin_info in discovered:
            print(f"  📦 {plugin_info.name} v{plugin_info.version}")
            print(f"     Description: {plugin_info.description}")
            print(f"     Category: {plugin_info.category}")
            print(f"     Author: {plugin_info.author}")
            print()
    
    async def _demo_plugin_loading(self) -> Any:
        """Demonstrate plugin loading and configuration."""
        print("\n3️⃣ Plugin Loading and Configuration")
        print("-" * 40)
        
        # Create a custom plugin
        print("Creating custom web extractor plugin...")
        web_extractor = WebExtractorPlugin({
            "timeout": 15,
            "max_retries": 2,
            "extraction_methods": ["newspaper3k", "trafilatura"],
            "enable_metadata": True
        })
        
        # Load the plugin
        print("Loading web extractor plugin...")
        loaded_plugin = await self.manager.load_plugin("web_extractor", {
            "timeout": 20,
            "enable_images": True
        })
        
        self.demo_plugins.append("web_extractor")
        print("✅ Web extractor plugin loaded successfully")
        
        # Show plugin information
        info = self.manager.get_plugin_info("web_extractor")
        if info:
            print(f"Plugin info:")
            print(f"  - Name: {info.name}")
            print(f"  - Version: {info.version}")
            print(f"  - Description: {info.description}")
            print(f"  - Category: {info.category}")
        
        # Show plugin configuration
        config = self.manager.get_plugin_config("web_extractor")
        print(f"Plugin configuration: {config}")
    
    async def _demo_lifecycle_management(self) -> Any:
        """Demonstrate plugin lifecycle management."""
        print("\n4️⃣ Plugin Lifecycle Management")
        print("-" * 35)
        
        # Initialize plugin
        print("Initializing web extractor plugin...")
        await self.manager.initialize_plugin("web_extractor")
        print("✅ Plugin initialized")
        
        # Start plugin
        print("Starting web extractor plugin...")
        await self.manager.start_plugin("web_extractor")
        print("✅ Plugin started")
        
        # Show plugin state
        state = self.manager.get_plugin_state("web_extractor")
        print(f"Plugin state: {state.value}")
        
        # Stop plugin
        print("Stopping web extractor plugin...")
        await self.manager.stop_plugin("web_extractor")
        print("✅ Plugin stopped")
    
    async def _demo_error_handling(self) -> Any:
        """Demonstrate error handling and recovery."""
        print("\n5️⃣ Error Handling and Recovery")
        print("-" * 35)
        
        # Try to load a non-existent plugin
        print("Attempting to load non-existent plugin...")
        try:
            await self.manager.load_plugin("non_existent_plugin")
        except Exception as e:
            print(f"❌ Expected error: {e}")
        
        # Try to load plugin with invalid configuration
        print("Attempting to load plugin with invalid configuration...")
        try:
            await self.manager.load_plugin("web_extractor", {
                "timeout": -1,  # Invalid timeout
                "max_retries": 999  # Invalid retries
            })
        except Exception as e:
            print(f"❌ Expected error: {e}")
        
        # Show error recovery
        print("Demonstrating error recovery...")
        plugin = self.manager.get_plugin("web_extractor")
        if plugin:
            # Update configuration to fix errors
            success = plugin.update_config({
                "timeout": 30,
                "max_retries": 3
            })
            print(f"Configuration update successful: {success}")
    
    async def _demo_performance_monitoring(self) -> Any:
        """Demonstrate performance monitoring."""
        print("\n6️⃣ Performance Monitoring")
        print("-" * 30)
        
        # Get plugin statistics
        stats = self.manager.get_stats()
        print("Plugin manager statistics:")
        print(f"  - Total plugins: {stats['total_plugins']}")
        print(f"  - Loaded plugins: {stats['loaded_plugins']}")
        print(f"  - Initialized plugins: {stats['initialized_plugins']}")
        print(f"  - Running plugins: {stats['running_plugins']}")
        print(f"  - Failed plugins: {stats['failed_plugins']}")
        
        # Get plugin-specific statistics
        plugin = self.manager.get_plugin("web_extractor")
        if plugin and hasattr(plugin, 'get_stats'):
            plugin_stats = plugin.get_stats()
            print("\nWeb extractor plugin statistics:")
            for key, value in plugin_stats.items():
                print(f"  - {key}: {value}")
    
    async def _demo_event_handling(self) -> Any:
        """Demonstrate event handling."""
        print("\n7️⃣ Event Handling")
        print("-" * 20)
        
        # Add custom event handlers
        def on_plugin_loaded(plugin_name, plugin) -> Any:
            print(f"🎉 Custom event: Plugin '{plugin_name}' loaded!")
        
        def on_plugin_started(plugin_name, plugin) -> Any:
            print(f"🚀 Custom event: Plugin '{plugin_name}' started!")
        
        self.manager.add_event_handler("plugin_loaded", on_plugin_loaded)
        self.manager.add_event_handler("plugin_started", on_plugin_started)
        
        # Trigger events by reloading plugin
        print("Reloading plugin to trigger events...")
        await self.manager.reload_plugin("web_extractor")
        await self.manager.start_plugin("web_extractor")
    
    async def _demo_health_reporting(self) -> Any:
        """Demonstrate health reporting."""
        print("\n8️⃣ Health Reporting")
        print("-" * 20)
        
        # Get comprehensive health report
        health = self.manager.get_health_report()
        
        # Display overall health status with emojis
        status_emoji = "🟢" if health['overall_status'] == 'healthy' else "🔴"
        print(f"Plugin Health Report {status_emoji}")
        print("=" * 40)
        print(f"📊 Overall Status: {health['overall_status'].upper()}")
        print(f"📈 Total Plugins: {health['total_plugins']}")
        print(f"✅ Healthy Plugins: {health['healthy_plugins']}")
        print(f"❌ Unhealthy Plugins: {health['unhealthy_plugins']}")
        
        # Calculate health percentage
        if health['total_plugins'] > 0:
            health_percentage = (health['healthy_plugins'] / health['total_plugins']) * 100
            print(f"📊 Health Score: {health_percentage:.1f}%")
        
        # Show detailed plugin health
        print("\nPlugin details:")
        for plugin_name, details in health['plugin_details'].items():
            status_emoji = "✅" if details['status'] == 'healthy' else "❌"
            print(f"  {status_emoji} {plugin_name}: {details['status']}")
            print(f"    - State: {details['state']}")
            print(f"    - Error count: {details['error_count']}")
            if details['last_error']:
                print(f"    - Last error: {details['last_error']}")
    
    async def _demo_advanced_features(self) -> Any:
        """Demonstrate advanced features."""
        print("\n9️⃣ Advanced Features")
        print("-" * 20)
        
        # Plugin summary
        summary = self.manager.get_plugin_summary()
        print("Plugin summary:")
        print(f"  - Total plugins: {summary['total_plugins']}")
        print(f"  - By state: {summary['by_state']}")
        print(f"  - By category: {summary['by_category']}")
        
        # List plugins by category
        extractors = self.manager.list_plugins_by_category("extractor")
        print(f"  - Extractor plugins: {extractors}")
        
        # Configuration management
        print("\nConfiguration management:")
        success = self.manager.update_plugin_config("web_extractor", {
            "timeout": 25,
            "enable_images": True
        })
        print(f"  - Configuration update: {'✅' if success else '❌'}")
        
        # Plugin restart
        print("\nPlugin restart:")
        await self.manager.restart_plugin("web_extractor")
        print("  - Plugin restarted successfully")
    
    async def _cleanup(self) -> Any:
        """Cleanup resources."""
        print("\n🧹 Cleanup")
        print("-" * 10)
        
        if self.manager:
            await self.manager.shutdown()
            print("✅ Plugin manager shutdown complete")


async def interactive_demo():
    """Interactive demo with user input."""
    print("🎮 Interactive Plugin Demo")
    print("=" * 30)
    
    demo = DemoPluginManager()
    
    while True:
        print("\nAvailable commands:")
        print("1. Run full demo")
        print("2. Quick start demo")
        print("3. Plugin discovery only")
        print("4. Load specific plugin")
        print("5. Show plugin stats")
        print("6. Show health report")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            await demo.run_demo()
        elif choice == "2":
            print("Running quick start demo...")
            manager = await quick_start()
            plugins = manager.list_plugins()
            print(f"Loaded plugins: {plugins}")
            await manager.shutdown()
        elif choice == "3":
            print("Running plugin discovery...")
            manager = await create_plugin_manager(auto_load=False)
            discovered = await manager.discover_plugins()
            print(f"Discovered {len(discovered)} plugins")
            for plugin in discovered:
                print(f"  - {plugin.name} v{plugin.version}")
            await manager.shutdown()
        elif choice == "4":
            plugin_name = input("Enter plugin name: ").strip()
            if plugin_name:
                try:
                    manager = await create_plugin_manager()
                    await manager.start()
                    plugin = await manager.load_plugin(plugin_name)
                    print(f"✅ Plugin '{plugin_name}' loaded successfully")
                    await manager.shutdown()
                except Exception as e:
                    print(f"❌ Failed to load plugin: {e}")
        elif choice == "5":
            try:
                manager = await create_plugin_manager()
                await manager.start()
                stats = manager.get_stats()
                print("Plugin stats:")
                for key, value in stats.items():
                    print(f"  - {key}: {value}")
                await manager.shutdown()
            except Exception as e:
                print(f"❌ Failed to get stats: {e}")
        elif choice == "6":
            try:
                manager = await create_plugin_manager()
                await manager.start()
                health = manager.get_health_report()
                print("Health report:")
                print(f"  - Overall status: {health['overall_status']}")
                print(f"  - Healthy plugins: {health['healthy_plugins']}")
                print(f"  - Unhealthy plugins: {health['unhealthy_plugins']}")
                await manager.shutdown()
            except Exception as e:
                print(f"❌ Failed to get health report: {e}")
        elif choice == "7":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")


def main():
    """Main function."""
    print("🎯 AI Video Plugin System Demo")
    print("=" * 50)
    
    # Check if running in interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_demo())
    else:
        # Run full demo
        demo = DemoPluginManager()
        asyncio.run(demo.run_demo())


match __name__:
    case "__main__":
    main() 
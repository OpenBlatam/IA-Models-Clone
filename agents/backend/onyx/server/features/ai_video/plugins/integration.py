from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from .base import BasePlugin, PluginMetadata
from .manager import PluginManager, ManagerConfig
from ..web_extract import WebContentExtractor, ExtractedContent
from ..suggestions import ContentSuggestions, SuggestionEngine
from ..video_generator import VideoGenerator, VideoGenerationResult
from ..models import AIVideo
from typing import Any, List, Dict, Optional
"""
Plugin Integration Module

This module provides integration between the plugin system and the existing
AI video components, including adapters, bridges, and compatibility layers.
"""



logger = logging.getLogger(__name__)


@dataclass
class PluginAdapter:
    """Adapter for connecting plugins to existing components."""
    plugin: BasePlugin
    adapter_type: str
    priority: int = 0
    is_active: bool = True


class PluginIntegrationManager:
    """
    Manages integration between plugins and existing AI video components.
    
    This class provides:
    - Adapter creation and management
    - Component bridging
    - Plugin discovery and registration
    - Compatibility layers
    """
    
    def __init__(self, plugin_manager: PluginManager):
        
    """__init__ function."""
self.plugin_manager = plugin_manager
        self.adapters: Dict[str, List[PluginAdapter]] = {
            'extractor': [],
            'suggestion_engine': [],
            'video_generator': [],
            'processor': [],
            'analyzer': []
        }
        
        logger.info("PluginIntegrationManager initialized")
    
    async def register_plugins(self) -> Dict[str, int]:
        """
        Register all loaded plugins as adapters.
        
        Returns:
            Dictionary with counts of registered plugins by type
        """
        counts = {adapter_type: 0 for adapter_type in self.adapters.keys()}
        
        for plugin_name in self.plugin_manager.list_plugins():
            plugin = self.plugin_manager.get_plugin(plugin_name)
            if plugin:
                adapter = await self._create_adapter(plugin)
                if adapter:
                    self.adapters[adapter.adapter_type].append(adapter)
                    counts[adapter.adapter_type] += 1
                    logger.info(f"Registered plugin {plugin_name} as {adapter.adapter_type}")
        
        # Sort adapters by priority
        for adapter_type in self.adapters:
            self.adapters[adapter_type].sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Plugin registration complete: {counts}")
        return counts
    
    async def _create_adapter(self, plugin: BasePlugin) -> Optional[PluginAdapter]:
        """Create an adapter for a plugin based on its metadata."""
        try:
            metadata = plugin.get_metadata()
            category = metadata.category.lower()
            
            # Map plugin categories to adapter types
            category_mapping = {
                'extractor': 'extractor',
                'extractors': 'extractor',
                'web_extractor': 'extractor',
                'content_extractor': 'extractor',
                
                'suggestion': 'suggestion_engine',
                'suggestions': 'suggestion_engine',
                'suggestion_engine': 'suggestion_engine',
                'content_suggestions': 'suggestion_engine',
                
                'generator': 'video_generator',
                'generators': 'video_generator',
                'video_generator': 'video_generator',
                'video_creator': 'video_generator',
                
                'processor': 'processor',
                'content_processor': 'processor',
                'media_processor': 'processor',
                
                'analyzer': 'analyzer',
                'content_analyzer': 'analyzer',
                'video_analyzer': 'analyzer'
            }
            
            adapter_type = category_mapping.get(category, 'processor')
            
            # Determine priority based on plugin metadata
            priority = self._calculate_priority(metadata)
            
            return PluginAdapter(
                plugin=plugin,
                adapter_type=adapter_type,
                priority=priority
            )
            
        except Exception as e:
            logger.error(f"Failed to create adapter for plugin {plugin.name}: {e}")
            return None
    
    def _calculate_priority(self, metadata: PluginMetadata) -> int:
        """Calculate priority for a plugin based on its metadata."""
        priority = 0
        
        # Version-based priority (higher versions get higher priority)
        try:
            version_parts = metadata.version.split('.')
            if len(version_parts) >= 1:
                priority += int(version_parts[0]) * 100
            if len(version_parts) >= 2:
                priority += int(version_parts[1]) * 10
        except (ValueError, AttributeError):
            pass
        
        # Category-based priority
        category_priority = {
            'extractor': 50,
            'suggestion_engine': 40,
            'video_generator': 30,
            'processor': 20,
            'analyzer': 10
        }
        
        priority += category_priority.get(metadata.category.lower(), 0)
        
        return priority
    
    def get_adapters(self, adapter_type: str) -> List[PluginAdapter]:
        """Get all adapters of a specific type."""
        return [adapter for adapter in self.adapters.get(adapter_type, []) if adapter.is_active]
    
    def get_best_adapter(self, adapter_type: str) -> Optional[PluginAdapter]:
        """Get the highest priority adapter of a specific type."""
        adapters = self.get_adapters(adapter_type)
        return adapters[0] if adapters else None
    
    async def create_integrated_extractor(self) -> WebContentExtractor:
        """Create an integrated extractor that uses plugin adapters."""
        class IntegratedExtractor(WebContentExtractor):
            def __init__(self, adapters: List[PluginAdapter]):
                
    """__init__ function."""
super().__init__()
                self.adapters = adapters
                self.last_used = None
                self.stats = {
                    'total_requests': 0,
                    'successful_extractions': 0,
                    'failed_extractions': 0,
                    'adapter_usage': {}
                }
            
            async def extract(self, url: str) -> Optional[ExtractedContent]:
                self.stats['total_requests'] += 1
                
                # Try each adapter in priority order
                for adapter in self.adapters:
                    try:
                        if hasattr(adapter.plugin, 'extract_content'):
                            result = await adapter.plugin.extract_content(url)
                            if result:
                                self.last_used = adapter.plugin.name
                                self.stats['successful_extractions'] += 1
                                self.stats['adapter_usage'][adapter.plugin.name] = \
                                    self.stats['adapter_usage'].get(adapter.plugin.name, 0) + 1
                                return result
                    except Exception as e:
                        logger.warning(f"Extractor adapter {adapter.plugin.name} failed: {e}")
                        continue
                
                # Fallback to default extraction
                try:
                    result = await super().extract(url)
                    if result:
                        self.stats['successful_extractions'] += 1
                        self.last_used = 'default'
                        return result
                except Exception as e:
                    logger.error(f"Default extraction failed: {e}")
                
                self.stats['failed_extractions'] += 1
                return None
            
            def get_last_used_extractor(self) -> Optional[str]:
                return self.last_used
            
            def get_stats(self) -> Dict[str, Any]:
                return self.stats.copy()
        
        extractor_adapters = self.get_adapters('extractor')
        return IntegratedExtractor(extractor_adapters)
    
    async def create_integrated_suggestion_engine(self) -> SuggestionEngine:
        """Create an integrated suggestion engine that uses plugin adapters."""
        class IntegratedSuggestionEngine(SuggestionEngine):
            def __init__(self, adapters: List[PluginAdapter]):
                
    """__init__ function."""
super().__init__()
                self.adapters = adapters
                self.stats = {
                    'total_requests': 0,
                    'successful_suggestions': 0,
                    'failed_suggestions': 0,
                    'adapter_usage': {}
                }
            
            async def generate_suggestions(self, content: ExtractedContent) -> ContentSuggestions:
                self.stats['total_requests'] += 1
                
                # Try each adapter in priority order
                for adapter in self.adapters:
                    try:
                        if hasattr(adapter.plugin, 'generate_suggestions'):
                            result = await adapter.plugin.generate_suggestions(content)
                            if result:
                                self.stats['successful_suggestions'] += 1
                                self.stats['adapter_usage'][adapter.plugin.name] = \
                                    self.stats['adapter_usage'].get(adapter.plugin.name, 0) + 1
                                return result
                    except Exception as e:
                        logger.warning(f"Suggestion adapter {adapter.plugin.name} failed: {e}")
                        continue
                
                # Fallback to default suggestions
                try:
                    result = await super().generate_suggestions(content)
                    if result:
                        self.stats['successful_suggestions'] += 1
                        return result
                except Exception as e:
                    logger.error(f"Default suggestions failed: {e}")
                
                self.stats['failed_suggestions'] += 1
                return ContentSuggestions()  # Return empty suggestions
            
            def get_stats(self) -> Dict[str, Any]:
                return self.stats.copy()
        
        suggestion_adapters = self.get_adapters('suggestion_engine')
        return IntegratedSuggestionEngine(suggestion_adapters)
    
    async def create_integrated_generator(self) -> VideoGenerator:
        """Create an integrated video generator that uses plugin adapters."""
        class IntegratedVideoGenerator(VideoGenerator):
            def __init__(self, adapters: List[PluginAdapter]):
                
    """__init__ function."""
super().__init__()
                self.adapters = adapters
                self.stats = {
                    'total_requests': 0,
                    'successful_generations': 0,
                    'failed_generations': 0,
                    'adapter_usage': {}
                }
            
            async def generate_video(
                self, 
                content: ExtractedContent, 
                suggestions: ContentSuggestions, 
                avatar: Optional[str] = None
            ) -> VideoGenerationResult:
                self.stats['total_requests'] += 1
                
                # Try each adapter in priority order
                for adapter in self.adapters:
                    try:
                        if hasattr(adapter.plugin, 'generate_video'):
                            result = await adapter.plugin.generate_video(content, suggestions, avatar)
                            if result:
                                self.stats['successful_generations'] += 1
                                self.stats['adapter_usage'][adapter.plugin.name] = \
                                    self.stats['adapter_usage'].get(adapter.plugin.name, 0) + 1
                                return result
                    except Exception as e:
                        logger.warning(f"Generator adapter {adapter.plugin.name} failed: {e}")
                        continue
                
                # Fallback to default generation
                try:
                    result = await super().generate_video(content, suggestions, avatar)
                    if result:
                        self.stats['successful_generations'] += 1
                        return result
                except Exception as e:
                    logger.error(f"Default generation failed: {e}")
                
                self.stats['failed_generations'] += 1
                return VideoGenerationResult(success=False, error="All generators failed")
            
            def get_stats(self) -> Dict[str, Any]:
                return self.stats.copy()
        
        generator_adapters = self.get_adapters('video_generator')
        return IntegratedVideoGenerator(generator_adapters)
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        stats = {
            'total_adapters': sum(len(adapters) for adapters in self.adapters.values()),
            'adapter_counts': {adapter_type: len(adapters) for adapter_type, adapters in self.adapters.items()},
            'active_adapters': {adapter_type: len([a for a in adapters if a.is_active]) 
                               for adapter_type, adapters in self.adapters.items()},
            'plugin_manager_stats': self.plugin_manager.get_stats() if self.plugin_manager else {}
        }
        
        return stats
    
    def enable_adapter(self, plugin_name: str, adapter_type: str) -> bool:
        """Enable a specific adapter."""
        for adapter in self.adapters.get(adapter_type, []):
            if adapter.plugin.name == plugin_name:
                adapter.is_active = True
                logger.info(f"Enabled adapter: {plugin_name} ({adapter_type})")
                return True
        return False
    
    def disable_adapter(self, plugin_name: str, adapter_type: str) -> bool:
        """Disable a specific adapter."""
        for adapter in self.adapters.get(adapter_type, []):
            if adapter.plugin.name == plugin_name:
                adapter.is_active = False
                logger.info(f"Disabled adapter: {plugin_name} ({adapter_type})")
                return True
        return False


class PluginBridge:
    """
    Bridge for connecting existing components to the plugin system.
    
    This class provides compatibility layers and adapters for existing
    AI video components to work with the plugin system.
    """
    
    def __init__(self, integration_manager: PluginIntegrationManager):
        
    """__init__ function."""
self.integration_manager = integration_manager
        self.components = {}
        
        logger.info("PluginBridge initialized")
    
    async def create_bridged_components(self) -> Dict[str, Any]:
        """Create bridged components that integrate with the plugin system."""
        components = {}
        
        # Create integrated extractor
        components['extractor'] = await self.integration_manager.create_integrated_extractor()
        
        # Create integrated suggestion engine
        components['suggestion_engine'] = await self.integration_manager.create_integrated_suggestion_engine()
        
        # Create integrated video generator
        components['video_generator'] = await self.integration_manager.create_integrated_generator()
        
        self.components = components
        logger.info("Bridged components created successfully")
        
        return components
    
    def get_component(self, component_type: str) -> Optional[Any]:
        """Get a specific bridged component."""
        return self.components.get(component_type)
    
    def get_all_components(self) -> Dict[str, Any]:
        """Get all bridged components."""
        return self.components.copy()
    
    async def update_component_config(self, component_type: str, config: Dict[str, Any]) -> bool:
        """Update configuration for a specific component."""
        component = self.get_component(component_type)
        if component and hasattr(component, 'update_config'):
            try:
                return component.update_config(config)
            except Exception as e:
                logger.error(f"Failed to update component config: {e}")
                return False
        return False
    
    def get_bridge_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        stats = {
            'components_available': list(self.components.keys()),
            'integration_stats': self.integration_manager.get_integration_stats(),
            'component_stats': {}
        }
        
        # Collect component-specific stats
        for component_type, component in self.components.items():
            if hasattr(component, 'get_stats'):
                stats['component_stats'][component_type] = component.get_stats()
        
        return stats


# Convenience functions

async def create_plugin_integration(
    plugin_config: Optional[ManagerConfig] = None
) -> tuple[PluginIntegrationManager, PluginBridge]:
    """
    Create and initialize plugin integration.
    
    Args:
        plugin_config: Plugin manager configuration
        
    Returns:
        Tuple of (PluginIntegrationManager, PluginBridge)
    """
    # Create plugin manager
    plugin_manager = PluginManager(plugin_config or ManagerConfig())
    await plugin_manager.start()
    
    # Create integration manager
    integration_manager = PluginIntegrationManager(plugin_manager)
    await integration_manager.register_plugins()
    
    # Create bridge
    bridge = PluginBridge(integration_manager)
    await bridge.create_bridged_components()
    
    return integration_manager, bridge


async def get_integrated_components(
    plugin_config: Optional[ManagerConfig] = None
) -> Dict[str, Any]:
    """
    Get integrated components with plugin support.
    
    Args:
        plugin_config: Plugin manager configuration
        
    Returns:
        Dictionary of integrated components
    """
    _, bridge = await create_plugin_integration(plugin_config)
    return bridge.get_all_components()


# Example usage

async def example_integration():
    """Example of how to use the plugin integration."""
    # Create integration
    integration_manager, bridge = await create_plugin_integration()
    
    # Get integrated components
    components = bridge.get_all_components()
    
    # Use integrated extractor
    extractor = components['extractor']
    content = await extractor.extract("https://example.com")
    
    if content:
        # Use integrated suggestion engine
        suggestion_engine = components['suggestion_engine']
        suggestions = await suggestion_engine.generate_suggestions(content)
        
        # Use integrated video generator
        generator = components['video_generator']
        result = await generator.generate_video(content, suggestions)
        
        print(f"Video generation result: {result}")
    
    # Get statistics
    stats = bridge.get_bridge_stats()
    print(f"Integration stats: {stats}")


match __name__:
    case "__main__":
    asyncio.run(example_integration()) 
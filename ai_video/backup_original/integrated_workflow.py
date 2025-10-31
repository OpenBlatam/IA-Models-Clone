from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from .video_workflow import VideoWorkflow, WorkflowState, WorkflowStatus, WorkflowHooks
from .web_extract import WebContentExtractor, ExtractedContent
from .suggestions import ContentSuggestions, SuggestionEngine
from .video_generator import VideoGenerator, VideoGenerationResult
from .state_repository import StateRepository, FileStateRepository
from .metrics import record_extraction_metrics, record_generation_metrics, record_workflow_metrics
from .models import AIVideo
from .plugins import (
    import argparse
        import json
from typing import Any, List, Dict, Optional
"""
Integrated AI Video Workflow with Plugin System

This module provides a unified interface that integrates the existing video workflow
with the new plugin system, offering enhanced extensibility, monitoring, and
management capabilities.
"""


# Import existing components

# Import plugin system
    PluginManager, 
    ManagerConfig, 
    ValidationLevel,
    BasePlugin,
    PluginMetadata,
    quick_start
)

logger = logging.getLogger(__name__)


class IntegratedWorkflowStatus(Enum):
    """Enhanced workflow status with plugin management."""
    INITIALIZING = "initializing"
    PLUGINS_LOADING = "plugins_loading"
    PLUGINS_READY = "plugins_ready"
    EXTRACTING = "extracting"
    SUGGESTING = "suggesting"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PLUGIN_ERROR = "plugin_error"


@dataclass
class PluginWorkflowState:
    """Enhanced workflow state with plugin information."""
    workflow_id: str
    source_url: str
    status: IntegratedWorkflowStatus
    avatar: Optional[str] = None
    
    # Content and processing results
    content: Optional[ExtractedContent] = None
    suggestions: Optional[ContentSuggestions] = None
    video_url: Optional[str] = None
    ai_video: Optional[AIVideo] = None
    
    # Plugin information
    loaded_plugins: List[str] = field(default_factory=list)
    active_plugins: List[str] = field(default_factory=list)
    plugin_errors: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Performance tracking
    plugin_load_time: Optional[float] = None
    extraction_time: Optional[float] = None
    suggestions_time: Optional[float] = None
    generation_time: Optional[float] = None
    total_time: Optional[float] = None
    
    # Error handling
    error: Optional[str] = None
    error_stage: Optional[str] = None
    
    # User customizations
    user_edits: Dict[str, Any] = field(default_factory=dict)
    
    # Plugin configuration
    plugin_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedWorkflowHooks:
    """Enhanced hooks with plugin events."""
    # Original workflow hooks
    on_extraction_complete: Optional[Callable[[ExtractedContent], None]] = None
    on_suggestions_complete: Optional[Callable[[ContentSuggestions], None]] = None
    on_generation_complete: Optional[Callable[[VideoGenerationResult], None]] = None
    on_workflow_complete: Optional[Callable[[PluginWorkflowState], None]] = None
    on_workflow_failed: Optional[Callable[[PluginWorkflowState, Exception], None]] = None
    
    # Plugin-specific hooks
    on_plugin_loaded: Optional[Callable[[str, BasePlugin], None]] = None
    on_plugin_error: Optional[Callable[[str, Exception], None]] = None
    on_plugins_ready: Optional[Callable[[List[str]], None]] = None


class IntegratedVideoWorkflow:
    """
    Integrated workflow that combines the existing video workflow with the plugin system.
    
    This class provides:
    - Unified interface for video generation
    - Plugin management and lifecycle
    - Enhanced monitoring and metrics
    - Extensible architecture
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        plugin_config: Optional[ManagerConfig] = None,
        workflow_config: Optional[Dict[str, Any]] = None,
        hooks: Optional[IntegratedWorkflowHooks] = None
    ):
        
    """__init__ function."""
self.plugin_config = plugin_config or ManagerConfig(
            auto_discover=True,
            auto_load=True,
            validation_level=ValidationLevel.STANDARD,
            enable_events=True,
            enable_metrics=True
        )
        
        self.workflow_config = workflow_config or {}
        self.hooks = hooks or IntegratedWorkflowHooks()
        
        # Core components
        self.plugin_manager: Optional[PluginManager] = None
        self.workflow: Optional[VideoWorkflow] = None
        self.state_repository: Optional[StateRepository] = None
        
        # Plugin components
        self.extractors: Dict[str, BasePlugin] = {}
        self.suggestion_engines: Dict[str, BasePlugin] = {}
        self.generators: Dict[str, BasePlugin] = {}
        
        logger.info("IntegratedVideoWorkflow initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the integrated workflow system.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            logger.info("🚀 Initializing Integrated Video Workflow...")
            
            # Initialize plugin manager
            self.plugin_manager = PluginManager(self.plugin_config)
            success = await self.plugin_manager.start()
            
            if not success:
                raise Exception("Failed to start plugin manager")
            
            # Load and categorize plugins
            await self._load_and_categorize_plugins()
            
            # Initialize core workflow components
            await self._initialize_workflow_components()
            
            # Setup event handlers
            self._setup_plugin_events()
            
            logger.info("✅ Integrated Video Workflow initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Integrated Video Workflow: {e}")
            return False
    
    async def _load_and_categorize_plugins(self) -> Any:
        """Load and categorize plugins by type."""
        logger.info("📦 Loading and categorizing plugins...")
        
        # Discover and load plugins
        discovered = await self.plugin_manager.discover_plugins()
        loaded = await self.plugin_manager.load_all_plugins()
        
        # Categorize plugins
        for plugin_name in loaded:
            plugin = self.plugin_manager.get_plugin(plugin_name)
            if plugin:
                metadata = plugin.get_metadata()
                category = metadata.category.lower()
                
                if category in ['extractor', 'extractors']:
                    self.extractors[plugin_name] = plugin
                    logger.info(f"  📥 Extractor plugin: {plugin_name}")
                    
                elif category in ['suggestion', 'suggestions', 'suggestion_engine']:
                    self.suggestion_engines[plugin_name] = plugin
                    logger.info(f"  💡 Suggestion plugin: {plugin_name}")
                    
                elif category in ['generator', 'generators', 'video_generator']:
                    self.generators[plugin_name] = plugin
                    logger.info(f"  🎬 Generator plugin: {plugin_name}")
                    
                else:
                    logger.info(f"  🔧 Utility plugin: {plugin_name} (category: {category})")
        
        logger.info(f"✅ Loaded {len(self.extractors)} extractors, {len(self.suggestion_engines)} suggestion engines, {len(self.generators)} generators")
    
    async def _initialize_workflow_components(self) -> Any:
        """Initialize core workflow components with plugin integration."""
        # Create state repository
        self.state_repository = FileStateRepository()
        
        # Create integrated extractor
        integrated_extractor = await self._create_integrated_extractor()
        
        # Create integrated suggestion engine
        integrated_suggestion_engine = await self._create_integrated_suggestion_engine()
        
        # Create integrated video generator
        integrated_generator = await self._create_integrated_generator()
        
        # Create workflow hooks
        workflow_hooks = WorkflowHooks(
            on_extraction_complete=self.hooks.on_extraction_complete,
            on_suggestions_complete=self.hooks.on_suggestions_complete,
            on_generation_complete=self.hooks.on_generation_complete,
            on_workflow_complete=lambda state: self.hooks.on_workflow_complete(state) if self.hooks.on_workflow_complete else None,
            on_workflow_failed=lambda state, error: self.hooks.on_workflow_failed(state, error) if self.hooks.on_workflow_failed else None
        )
        
        # Create workflow
        self.workflow = VideoWorkflow(
            extractor=integrated_extractor,
            suggestion_engine=integrated_suggestion_engine,
            video_generator=integrated_generator,
            state_repository=self.state_repository,
            hooks=workflow_hooks
        )
    
    async def _create_integrated_extractor(self) -> WebContentExtractor:
        """Create an integrated extractor that uses available plugins."""
        class IntegratedExtractor(WebContentExtractor):
            def __init__(self, extractors: Dict[str, BasePlugin]):
                
    """__init__ function."""
super().__init__()
                self.extractors = extractors
                self.last_used = None
            
            async def extract(self, url: str) -> Optional[ExtractedContent]:
                # Try each extractor in order of preference
                for name, extractor in self.extractors.items():
                    try:
                        if hasattr(extractor, 'extract_content'):
                            result = await extractor.extract_content(url)
                            if result:
                                self.last_used = name
                                return result
                    except Exception as e:
                        logger.warning(f"Extractor {name} failed: {e}")
                        continue
                
                # Fallback to default extraction
                return await super().extract(url)
            
            def get_last_used_extractor(self) -> Optional[str]:
                return self.last_used
        
        return IntegratedExtractor(self.extractors)
    
    async def _create_integrated_suggestion_engine(self) -> SuggestionEngine:
        """Create an integrated suggestion engine that uses available plugins."""
        class IntegratedSuggestionEngine(SuggestionEngine):
            def __init__(self, engines: Dict[str, BasePlugin]):
                
    """__init__ function."""
super().__init__()
                self.engines = engines
            
            async def generate_suggestions(self, content: ExtractedContent) -> ContentSuggestions:
                # Use the first available suggestion engine
                for name, engine in self.engines.items():
                    try:
                        if hasattr(engine, 'generate_suggestions'):
                            result = await engine.generate_suggestions(content)
                            if result:
                                return result
                    except Exception as e:
                        logger.warning(f"Suggestion engine {name} failed: {e}")
                        continue
                
                # Fallback to default suggestions
                return await super().generate_suggestions(content)
        
        return IntegratedSuggestionEngine(self.suggestion_engines)
    
    async def _create_integrated_generator(self) -> VideoGenerator:
        """Create an integrated video generator that uses available plugins."""
        class IntegratedVideoGenerator(VideoGenerator):
            def __init__(self, generators: Dict[str, BasePlugin]):
                
    """__init__ function."""
super().__init__()
                self.generators = generators
            
            async def generate_video(self, content: ExtractedContent, suggestions: ContentSuggestions, avatar: Optional[str] = None) -> VideoGenerationResult:
                # Use the first available generator
                for name, generator in self.generators.items():
                    try:
                        if hasattr(generator, 'generate_video'):
                            result = await generator.generate_video(content, suggestions, avatar)
                            if result:
                                return result
                    except Exception as e:
                        logger.warning(f"Video generator {name} failed: {e}")
                        continue
                
                # Fallback to default generation
                return await super().generate_video(content, suggestions, avatar)
        
        return IntegratedVideoGenerator(self.generators)
    
    def _setup_plugin_events(self) -> Any:
        """Setup plugin event handlers."""
        if self.plugin_manager:
            self.plugin_manager.add_event_handler("plugin_loaded", self._on_plugin_loaded)
            self.plugin_manager.add_event_handler("plugin_error", self._on_plugin_error)
    
    def _on_plugin_loaded(self, plugin_name: str, plugin: BasePlugin):
        """Handle plugin loaded event."""
        logger.info(f"🎉 Plugin loaded: {plugin_name}")
        if self.hooks.on_plugin_loaded:
            self.hooks.on_plugin_loaded(plugin_name, plugin)
    
    def _on_plugin_error(self, plugin_name: str, error: Exception):
        """Handle plugin error event."""
        logger.error(f"❌ Plugin error: {plugin_name} - {error}")
        if self.hooks.on_plugin_error:
            self.hooks.on_plugin_error(plugin_name, error)
    
    async def execute_workflow(
        self,
        url: str,
        workflow_id: Optional[str] = None,
        avatar: Optional[str] = None,
        user_edits: Optional[Dict[str, Any]] = None,
        plugin_config: Optional[Dict[str, Any]] = None
    ) -> PluginWorkflowState:
        """
        Execute the complete integrated workflow.
        
        Args:
            url: Source URL for content extraction
            workflow_id: Unique identifier for this workflow
            avatar: Avatar to use for video generation
            user_edits: Optional user customizations
            plugin_config: Optional plugin-specific configuration
            
        Returns:
            PluginWorkflowState: Complete workflow state with results
        """
        if not self.workflow:
            raise Exception("Workflow not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        # Create enhanced workflow state
        state = PluginWorkflowState(
            workflow_id=workflow_id or f"workflow_{int(time.time())}",
            source_url=url,
            status=IntegratedWorkflowStatus.INITIALIZING,
            avatar=avatar,
            user_edits=user_edits or {},
            plugin_config=plugin_config or {}
        )
        
        try:
            # Update plugin configuration if provided
            if plugin_config:
                await self._update_plugin_configurations(plugin_config)
            
            # Execute the core workflow
            workflow_state = await self.workflow.execute(url, state.workflow_id, avatar, user_edits)
            
            # Convert to enhanced state
            state.status = IntegratedWorkflowStatus.COMPLETED
            state.content = workflow_state.content
            state.suggestions = workflow_state.suggestions
            state.video_url = workflow_state.video_url
            state.extraction_time = workflow_state.timings.extraction
            state.suggestions_time = workflow_state.timings.suggestions
            state.generation_time = workflow_state.timings.generation
            state.total_time = time.time() - start_time
            
            # Create AIVideo model
            if state.content and state.video_url:
                state.ai_video = AIVideo.from_web_url(
                    url=url,
                    title=state.content.title or "Generated Video",
                    description=state.content.text[:200] + "..." if len(state.content.text) > 200 else state.content.text,
                    prompts=[state.content.text],
                    ai_model="integrated_workflow",
                    duration=30.0,
                    resolution="1920x1080",
                    avatar_id=avatar
                )
            
            logger.info(f"✅ Workflow completed successfully in {state.total_time:.2f}s")
            return state
            
        except Exception as e:
            state.status = IntegratedWorkflowStatus.FAILED
            state.error = str(e)
            state.error_stage = "workflow_execution"
            state.total_time = time.time() - start_time
            
            logger.error(f"❌ Workflow failed: {e}")
            raise
    
    async def _update_plugin_configurations(self, plugin_config: Dict[str, Any]):
        """Update plugin configurations."""
        for plugin_name, config in plugin_config.items():
            if self.plugin_manager:
                success = self.plugin_manager.update_plugin_config(plugin_name, config)
                if success:
                    logger.info(f"Updated configuration for plugin: {plugin_name}")
                else:
                    logger.warning(f"Failed to update configuration for plugin: {plugin_name}")
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[PluginWorkflowState]:
        """Get the status of a specific workflow."""
        if self.state_repository:
            state = await self.state_repository.load(workflow_id)
            if state:
                # Convert to enhanced state
                return PluginWorkflowState(
                    workflow_id=state.workflow_id,
                    source_url=state.source_url,
                    status=IntegratedWorkflowStatus(state.status.value),
                    avatar=state.avatar,
                    content=state.content,
                    suggestions=state.suggestions,
                    video_url=state.video_url,
                    created_at=state.created_at,
                    updated_at=state.updated_at,
                    extraction_time=state.timings.extraction,
                    suggestions_time=state.timings.suggestions,
                    generation_time=state.timings.generation,
                    error=state.error,
                    error_stage=state.error_stage,
                    user_edits=state.user_edits
                )
        return None
    
    def get_plugin_stats(self) -> Dict[str, Any]:
        """Get comprehensive plugin statistics."""
        if not self.plugin_manager:
            return {}
        
        stats = self.plugin_manager.get_stats()
        stats.update({
            'extractors': len(self.extractors),
            'suggestion_engines': len(self.suggestion_engines),
            'generators': len(self.generators),
            'total_plugins': len(self.extractors) + len(self.suggestion_engines) + len(self.generators)
        })
        
        return stats
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        if not self.plugin_manager:
            return {'status': 'not_initialized'}
        
        health = self.plugin_manager.get_health_report()
        health.update({
            'workflow_initialized': self.workflow is not None,
            'state_repository_available': self.state_repository is not None,
            'plugin_categories': {
                'extractors': list(self.extractors.keys()),
                'suggestion_engines': list(self.suggestion_engines.keys()),
                'generators': list(self.generators.keys())
            }
        })
        
        return health
    
    async def shutdown(self) -> Any:
        """Shutdown the integrated workflow system."""
        logger.info("🔄 Shutting down Integrated Video Workflow...")
        
        if self.plugin_manager:
            await self.plugin_manager.shutdown()
        
        logger.info("✅ Integrated Video Workflow shutdown complete")


# Convenience functions for easy usage

async def create_integrated_workflow(
    plugin_config: Optional[ManagerConfig] = None,
    workflow_config: Optional[Dict[str, Any]] = None
) -> IntegratedVideoWorkflow:
    """
    Create and initialize an integrated workflow.
    
    Args:
        plugin_config: Plugin manager configuration
        workflow_config: Workflow configuration
        
    Returns:
        Initialized IntegratedVideoWorkflow instance
    """
    workflow = IntegratedVideoWorkflow(plugin_config, workflow_config)
    success = await workflow.initialize()
    
    if not success:
        raise Exception("Failed to initialize integrated workflow")
    
    return workflow


async def quick_video_generation(
    url: str,
    avatar: Optional[str] = None,
    plugin_config: Optional[Dict[str, Any]] = None
) -> PluginWorkflowState:
    """
    Quick video generation with default settings.
    
    Args:
        url: Source URL
        avatar: Avatar to use
        plugin_config: Optional plugin configuration
        
    Returns:
        Workflow state with results
    """
    workflow = await create_integrated_workflow()
    
    try:
        result = await workflow.execute_workflow(url, avatar=avatar, plugin_config=plugin_config)
        return result
    finally:
        await workflow.shutdown()


# Example usage and CLI integration

async def main():
    """Example usage of the integrated workflow."""
    
    parser = argparse.ArgumentParser(description="Integrated AI Video Workflow")
    parser.add_argument("url", help="Source URL for video generation")
    parser.add_argument("--avatar", help="Avatar to use for video generation")
    parser.add_argument("--workflow-id", help="Custom workflow ID")
    parser.add_argument("--config", help="Plugin configuration file")
    parser.add_argument("--stats", action="store_true", help="Show plugin statistics")
    parser.add_argument("--health", action="store_true", help="Show health report")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    plugin_config = None
    if args.config:
        with open(args.config, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            plugin_config = json.load(f)
    
    # Create workflow
    workflow = await create_integrated_workflow()
    
    try:
        # Show statistics if requested
        if args.stats:
            stats = workflow.get_plugin_stats()
            print("Plugin Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return
        
        # Show health report if requested
        if args.health:
            health = workflow.get_health_report()
            print("Health Report:")
            for key, value in health.items():
                print(f"  {key}: {value}")
            return
        
        # Execute workflow
        print(f"🎬 Starting video generation for: {args.url}")
        result = await workflow.execute_workflow(
            args.url,
            workflow_id=args.workflow_id,
            avatar=args.avatar,
            plugin_config=plugin_config
        )
        
        print(f"✅ Video generation completed!")
        print(f"  Workflow ID: {result.workflow_id}")
        print(f"  Total time: {result.total_time:.2f}s")
        print(f"  Video URL: {result.video_url}")
        
        if result.ai_video:
            print(f"  Title: {result.ai_video.title}")
            print(f"  Duration: {result.ai_video.duration}s")
            print(f"  Resolution: {result.ai_video.resolution}")
        
    finally:
        await workflow.shutdown()


match __name__:
    case "__main__":
    asyncio.run(main()) 
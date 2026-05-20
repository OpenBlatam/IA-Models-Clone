from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime
from ...video_workflow import VideoWorkflow, WorkflowHooks
from ...state_repository import FileStateRepository
from ...models import AIVideo
from ...plugins import PluginManager, ManagerConfig, ValidationLevel
from ..components.extractors import IntegratedExtractor
from ..components.suggestions import IntegratedSuggestionEngine
from ..components.generators import IntegratedVideoGenerator
from .models import (
from typing import Any, List, Dict, Optional
import asyncio
"""
Integrated Workflow - Core Workflow Engine

Main integrated workflow engine that combines video workflow with plugin system.
"""


# Import existing components

# Import plugin system

# Import integrated components

# Import models
    IntegratedWorkflowStatus,
    PluginWorkflowState,
    IntegratedWorkflowHooks,
    WorkflowConfiguration,
    WorkflowStatistics,
    HealthReport
)

logger = logging.getLogger(__name__)


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
        workflow_config: Optional[WorkflowConfiguration] = None,
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
        
        self.workflow_config = workflow_config or WorkflowConfiguration()
        self.hooks = hooks or IntegratedWorkflowHooks()
        
        # Core components
        self.plugin_manager: Optional[PluginManager] = None
        self.workflow: Optional[VideoWorkflow] = None
        self.state_repository: Optional[FileStateRepository] = None
        
        # Plugin components
        self.extractors: Dict[str, Any] = {}
        self.suggestion_engines: Dict[str, Any] = {}
        self.generators: Dict[str, Any] = {}
        
        # Statistics
        self.statistics = WorkflowStatistics()
        
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
        integrated_extractor = IntegratedExtractor(self.extractors)
        
        # Create integrated suggestion engine
        integrated_suggestion_engine = IntegratedSuggestionEngine(self.suggestion_engines)
        
        # Create integrated video generator
        integrated_generator = IntegratedVideoGenerator(self.generators)
        
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
    
    def _setup_plugin_events(self) -> Any:
        """Setup plugin event handlers."""
        if self.plugin_manager:
            self.plugin_manager.add_event_handler("plugin_loaded", self._on_plugin_loaded)
            self.plugin_manager.add_event_handler("plugin_error", self._on_plugin_error)
    
    def _on_plugin_loaded(self, plugin_name: str, plugin: Any):
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
        self.statistics.last_workflow_start = datetime.now()
        self.statistics.total_workflows += 1
        
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
            
            # Update statistics
            self.statistics.successful_workflows += 1
            self.statistics.last_workflow_complete = datetime.now()
            self.statistics.total_extraction_time += state.extraction_time or 0
            self.statistics.total_suggestions_time += state.suggestions_time or 0
            self.statistics.total_generation_time += state.generation_time or 0
            
            logger.info(f"✅ Workflow completed successfully in {state.total_time:.2f}s")
            return state
            
        except Exception as e:
            state.status = IntegratedWorkflowStatus.FAILED
            state.error = str(e)
            state.error_stage = "workflow_execution"
            state.total_time = time.time() - start_time
            
            # Update statistics
            self.statistics.failed_workflows += 1
            
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
    
    def get_health_report(self) -> HealthReport:
        """Get comprehensive health report."""
        if not self.plugin_manager:
            return HealthReport(status="not_initialized")
        
        plugin_health = self.plugin_manager.get_health_report()
        
        health = HealthReport(
            status="healthy" if plugin_health.get('status') == 'healthy' else "warning",
            plugin_manager_healthy=plugin_health.get('status') == 'healthy',
            workflow_initialized=self.workflow is not None,
            state_repository_available=self.state_repository is not None,
            plugins_loaded=len(self.extractors) + len(self.suggestion_engines) + len(self.generators),
            plugins_healthy=plugin_health.get('healthy_plugins', 0),
            plugins_failed=plugin_health.get('failed_plugins', 0)
        )
        
        # Add plugin categories
        health.recommendations = []
        if len(self.extractors) == 0:
            health.recommendations.append("No extractor plugins loaded")
        if len(self.suggestion_engines) == 0:
            health.recommendations.append("No suggestion engine plugins loaded")
        if len(self.generators) == 0:
            health.recommendations.append("No video generator plugins loaded")
        
        return health
    
    def get_statistics(self) -> WorkflowStatistics:
        """Get workflow statistics."""
        return self.statistics
    
    async def shutdown(self) -> Any:
        """Shutdown the integrated workflow system."""
        logger.info("🔄 Shutting down Integrated Video Workflow...")
        
        if self.plugin_manager:
            await self.plugin_manager.shutdown()
        
        logger.info("✅ Integrated Video Workflow shutdown complete") 
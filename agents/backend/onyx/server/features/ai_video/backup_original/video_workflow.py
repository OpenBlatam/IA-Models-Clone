from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from .web_extract import WebContentExtractor, ExtractedContent
from .suggestions import ContentSuggestions, SuggestionEngine
from .video_generator import VideoGenerator, VideoGenerationResult
from .state_repository import StateRepository, FileStateRepository
from .metrics import record_extraction_metrics, record_generation_metrics, record_workflow_metrics
            from urllib.parse import urlparse
        import uuid
    import argparse
    import sys
    import asyncio
    from pathlib import Path
    from datetime import datetime
            import traceback
from typing import Any, List, Dict, Optional
"""
AI Video Workflow - Complete Pipeline for Video Generation

This module orchestrates the entire video generation pipeline:
1. Web content extraction
2. AI-powered suggestions
3. User editing capabilities
4. Video generation with avatars
5. Comprehensive metrics and monitoring
"""



logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    EXTRACTING = "extracting"
    SUGGESTING = "suggesting"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowTimings:
    """Timing information for workflow stages."""
    extraction: Optional[float] = None
    suggestions: Optional[float] = None
    generation: Optional[float] = None
    total: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for metrics."""
        return {
            'extraction': self.extraction or 0.0,
            'suggestions': self.suggestions or 0.0,
            'generation': self.generation or 0.0,
            'total': self.total or 0.0
        }


@dataclass
class WorkflowState:
    """Complete state of a video generation workflow."""
    workflow_id: str
    source_url: str
    status: WorkflowStatus
    avatar: Optional[str] = None
    
    # Content and processing results
    content: Optional[ExtractedContent] = None
    suggestions: Optional[ContentSuggestions] = None
    video_url: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    timings: WorkflowTimings = field(default_factory=WorkflowTimings)
    
    # Performance tracking
    extractor_used: Optional[str] = None
    generator_used: Optional[str] = None
    
    # Error handling
    error: Optional[str] = None
    error_stage: Optional[str] = None
    
    # User customizations
    user_edits: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowHooks:
    """Hooks for custom workflow behavior."""
    on_extraction_complete: Optional[Callable[[ExtractedContent], None]] = None
    on_suggestions_complete: Optional[Callable[[ContentSuggestions], None]] = None
    on_generation_complete: Optional[Callable[[VideoGenerationResult], None]] = None
    on_workflow_complete: Optional[Callable[[WorkflowState], None]] = None
    on_workflow_failed: Optional[Callable[[WorkflowState, Exception], None]] = None


class VideoWorkflow:
    """
    Main workflow orchestrator for AI video generation.
    
    This class manages the complete pipeline from web content extraction
    to final video generation, with comprehensive state management,
    error handling, and extensibility through hooks.
    """
    
    def __init__(
        self,
        extractor: WebContentExtractor,
        suggestion_engine: SuggestionEngine,
        video_generator: VideoGenerator,
        state_repository: StateRepository,
        hooks: Optional[WorkflowHooks] = None
    ):
        
    """__init__ function."""
self.extractor = extractor
        self.suggestion_engine = suggestion_engine
        self.video_generator = video_generator
        self.state_repository = state_repository
        self.hooks = hooks or WorkflowHooks()
        
        logger.info("VideoWorkflow initialized with all components")
    
    async def execute(
        self,
        url: str,
        workflow_id: str,
        avatar: Optional[str] = None,
        user_edits: Optional[Dict[str, Any]] = None
    ) -> WorkflowState:
        """
        Execute the complete video generation workflow.
        
        Args:
            url: Source URL for content extraction
            workflow_id: Unique identifier for this workflow
            avatar: Avatar to use for video generation
            user_edits: Optional user customizations
            
        Returns:
            WorkflowState: Complete workflow state with results
        """
        start_time = time.time()
        state = WorkflowState(
            workflow_id=workflow_id,
            source_url=url,
            status=WorkflowStatus.PENDING,
            avatar=avatar,
            user_edits=user_edits or {}
        )
        
        try:
            # Save initial state
            await self.state_repository.save(state)
            
            # Stage 1: Content Extraction
            logger.info(f"Starting content extraction for {url}")
            state.status = WorkflowStatus.EXTRACTING
            state.updated_at = datetime.now()
            await self.state_repository.save(state)
            
            extraction_start = time.time()
            content = await self.extractor.extract(url)
            extraction_time = time.time() - extraction_start
            
            state.content = content
            state.extractor_used = self.extractor.get_last_used_extractor()
            state.timings.extraction = extraction_time
            
            # Record extraction metrics
            record_extraction_metrics(
                extractor_name=state.extractor_used or "unknown",
                success=content is not None,
                duration=extraction_time,
                domain=self._extract_domain(url)
            )
            
            if self.hooks.on_extraction_complete:
                self.hooks.on_extraction_complete(content)
            
            if not content:
                raise Exception("Failed to extract content from URL")
            
            logger.info(f"Content extraction completed in {extraction_time:.2f}s")
            
            # Stage 2: AI Suggestions
            logger.info("Generating AI suggestions")
            state.status = WorkflowStatus.SUGGESTING
            state.updated_at = datetime.now()
            await self.state_repository.save(state)
            
            suggestions_start = time.time()
            suggestions = await self.suggestion_engine.generate_suggestions(content)
            suggestions_time = time.time() - suggestions_start
            
            state.suggestions = suggestions
            state.timings.suggestions = suggestions_time
            
            if self.hooks.on_suggestions_complete:
                self.hooks.on_suggestions_complete(suggestions)
            
            logger.info(f"Suggestions generated in {suggestions_time:.2f}s")
            
            # Stage 3: Video Generation
            logger.info("Starting video generation")
            state.status = WorkflowStatus.GENERATING
            state.updated_at = datetime.now()
            await self.state_repository.save(state)
            
            generation_start = time.time()
            result = await self.video_generator.generate_video(
                content=content,
                suggestions=suggestions,
                avatar=avatar,
                user_edits=user_edits
            )
            generation_time = time.time() - generation_start
            
            state.video_url = result.video_url
            state.generator_used = self.video_generator.get_name()
            state.timings.generation = generation_time
            
            # Record generation metrics
            record_generation_metrics(
                generator_name=state.generator_used,
                success=result.success,
                duration=generation_time,
                quality_score=result.quality_score
            )
            
            if self.hooks.on_generation_complete:
                self.hooks.on_generation_complete(result)
            
            # Complete workflow
            total_time = time.time() - start_time
            state.status = WorkflowStatus.COMPLETED
            state.timings.total = total_time
            state.updated_at = datetime.now()
            
            # Record workflow metrics
            record_workflow_metrics(
                success=True,
                total_duration=total_time,
                stage_timings=state.timings.to_dict()
            )
            
            await self.state_repository.save(state)
            
            if self.hooks.on_workflow_complete:
                self.hooks.on_workflow_complete(state)
            
            logger.info(f"Workflow completed successfully in {total_time:.2f}s")
            return state
            
        except Exception as e:
            # Handle workflow failure
            total_time = time.time() - start_time
            state.status = WorkflowStatus.FAILED
            state.error = str(e)
            state.error_stage = state.status.value
            state.timings.total = total_time
            state.updated_at = datetime.now()
            
            # Record failed workflow metrics
            record_workflow_metrics(
                success=False,
                total_duration=total_time,
                stage_timings=state.timings.to_dict()
            )
            
            await self.state_repository.save(state)
            
            if self.hooks.on_workflow_failed:
                self.hooks.on_workflow_failed(state, e)
            
            logger.error(f"Workflow failed after {total_time:.2f}s: {e}")
            raise
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for metrics tracking."""
        try:
            return urlparse(url).netloc
        except:
            return "unknown"


async def run_full_workflow(
    url: str,
    workflow_id: Optional[str] = None,
    avatar: Optional[str] = None,
    debug: bool = False
) -> WorkflowState:
    """
    Convenience function to run a complete workflow with default components.
    
    This function creates all necessary components and executes the workflow,
    making it easy to test and use the system.
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    # Generate workflow ID if not provided
    if not workflow_id:
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
    
    # Create components
    extractor = WebContentExtractor()
    suggestion_engine = SuggestionEngine()
    video_generator = VideoGenerator()
    state_repository = FileStateRepository(directory='.workflow_state')
    
    # Create workflow
    workflow = VideoWorkflow(
        extractor=extractor,
        suggestion_engine=suggestion_engine,
        video_generator=video_generator,
        state_repository=state_repository
    )
    
    # Execute workflow
    return await workflow.execute(url, workflow_id, avatar)


# --- CLI for Testing ---
if __name__ == "__main__":

    async def start_workflow(args) -> Any:
        """Start a new workflow."""
        print(f"🚀 Starting new workflow for URL: {args.url}")
        print(f"📝 Workflow ID: {args.workflow_id or 'auto-generated'}")
        print(f"🤖 Avatar: {args.avatar or 'default'}")
        print("-" * 50)
        
        try:
            state = await run_full_workflow(
                url=args.url,
                workflow_id=args.workflow_id,
                avatar=args.avatar,
                debug=args.debug
            )
            
            print(f"\n✅ Workflow completed!")
            print(f"📊 Status: {state.status}")
            print(f"⏱️  Total time: {state.timings.total:.2f}s")
            print(f"🎬 Video URL: {state.video_url}")
            print(f"🔧 Extractor used: {state.extractor_used}")
            print(f"🎥 Generator used: {state.generator_used}")
            
        except Exception as e:
            print(f"❌ Workflow failed: {e}")
            sys.exit(1)

    async def resume_workflow(args) -> Any:
        """Resume an existing workflow."""
        print(f"🔄 Resuming workflow: {args.workflow_id}")
        print("-" * 50)
        
        try:
            state = await run_full_workflow(
                url="",  # Will be loaded from state
                workflow_id=args.workflow_id,
                debug=args.debug
            )
            
            print(f"\n✅ Workflow resumed and completed!")
            print(f"📊 Status: {state.status}")
            print(f"⏱️  Total time: {state.timings.total:.2f}s")
            print(f"🎬 Video URL: {state.video_url}")
            
        except Exception as e:
            print(f"❌ Resume failed: {e}")
            sys.exit(1)

    def list_workflows(args) -> List[Any]:
        """List all workflows."""
        repo = FileStateRepository(directory='.workflow_state')
        state_dir = Path('.workflow_state')
        
        if not state_dir.exists():
            print("📁 No workflows found.")
            return
        
        workflows = list(state_dir.glob('workflow_*.json'))
        if not workflows:
            print("📁 No workflows found.")
            return
        
        print(f"📋 Found {len(workflows)} workflow(s):")
        print("-" * 80)
        print(f"{'ID':<36} {'Status':<12} {'URL':<30} {'Created':<20}")
        print("-" * 80)
        
        for workflow_file in sorted(workflows, key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                state = repo.load(workflow_file.stem.replace('workflow_', ''))
                if state:
                    created = datetime.fromtimestamp(workflow_file.stat().st_mtime)
                    url_preview = str(state.source_url)[:27] + "..." if len(str(state.source_url)) > 30 else str(state.source_url)
                    print(f"{state.workflow_id:<36} {state.status.value:<12} {url_preview:<30} {created.strftime('%Y-%m-%d %H:%M:%S'):<20}")
            except Exception as e:
                print(f"❌ Error reading {workflow_file.name}: {e}")

    def show_status(args) -> Any:
        """Show detailed status of a workflow."""
        repo = FileStateRepository(directory='.workflow_state')
        state = repo.load(args.workflow_id)
        
        if not state:
            print(f"❌ Workflow '{args.workflow_id}' not found.")
            sys.exit(1)
        
        print(f"📊 Workflow Status: {args.workflow_id}")
        print("=" * 50)
        print(f"🔗 URL: {state.source_url}")
        print(f"📈 Status: {state.status}")
        print(f"🤖 Avatar: {state.avatar or 'default'}")
        print(f"⏱️  Timings:")
        print(f"   - Extraction: {state.timings.extraction:.2f}s" if state.timings.extraction else "   - Extraction: Not started")
        print(f"   - Suggestions: {state.timings.suggestions:.2f}s" if state.timings.suggestions else "   - Suggestions: Not started")
        print(f"   - Generation: {state.timings.generation:.2f}s" if state.timings.generation else "   - Generation: Not started")
        print(f"   - Total: {state.timings.total:.2f}s" if state.timings.total else "   - Total: Not completed")
        print(f"🔧 Extractor: {state.extractor_used or 'Not used'}")
        print(f"🎥 Generator: {state.generator_used or 'Not used'}")
        
        if state.content:
            print(f"📝 Content:")
            print(f"   - Title: {state.content.title or 'N/A'}")
            print(f"   - Images: {len(state.content.images)}")
            print(f"   - Links: {len(state.content.links)}")
        
        if state.suggestions:
            print(f"💡 Suggestions:")
            print(f"   - Script length: {len(state.suggestions.script or '')} chars")
            print(f"   - Suggested images: {len(state.suggestions.images)}")
            print(f"   - Style: {state.suggestions.style or 'N/A'}")
        
        if state.error:
            print(f"❌ Error: {state.error}")
        
        if state.video_url:
            print(f"🎬 Video: {state.video_url}")

    def clean_workflows(args) -> Any:
        """Clean old workflow states."""
        repo = FileStateRepository(directory='.workflow_state')
        state_dir = Path('.workflow_state')
        
        if not state_dir.exists():
            print("📁 No workflows to clean.")
            return
        
        workflows = list(state_dir.glob('workflow_*.json'))
        if not workflows:
            print("📁 No workflows to clean.")
            return
        
        if args.days:
            cutoff_time = datetime.now().timestamp() - (args.days * 24 * 3600)
            old_workflows = [w for w in workflows if w.stat().st_mtime < cutoff_time]
            print(f"🗑️  Cleaning workflows older than {args.days} days...")
        else:
            old_workflows = workflows
            print("🗑️  Cleaning all workflows...")
        
        if not old_workflows:
            print("✅ No workflows to clean.")
            return
        
        if not args.force:
            print(f"⚠️  This will delete {len(old_workflows)} workflow(s). Use --force to confirm.")
            return
        
        deleted = 0
        for workflow_file in old_workflows:
            try:
                workflow_file.unlink()
                deleted += 1
                print(f"🗑️  Deleted: {workflow_file.name}")
            except Exception as e:
                print(f"❌ Failed to delete {workflow_file.name}: {e}")
        
        print(f"✅ Cleaned {deleted} workflow(s).")

    # CLI Setup
    parser = argparse.ArgumentParser(description="AI Video Workflow CLI", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start a new workflow')
    start_parser.add_argument('url', help='URL to extract content from')
    start_parser.add_argument('--workflow-id', help='Custom workflow ID')
    start_parser.add_argument('--avatar', help='Avatar to use for video generation')
    
    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume an existing workflow')
    resume_parser.add_argument('workflow_id', help='Workflow ID to resume')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all workflows')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show workflow status')
    status_parser.add_argument('workflow_id', help='Workflow ID to check')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean old workflows')
    clean_parser.add_argument('--days', type=int, help='Clean workflows older than N days')
    clean_parser.add_argument('--force', action='store_true', help='Force deletion without confirmation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        print("\n📖 Examples:")
        print("  python video_workflow.py start https://example.com")
        print("  python video_workflow.py start https://example.com --avatar 'john' --workflow-id my-video")
        print("  python video_workflow.py resume abc123-def456")
        print("  python video_workflow.py list")
        print("  python video_workflow.py status abc123-def456")
        print("  python video_workflow.py clean --days 7 --force")
        sys.exit(0)
    
    # Execute command
    try:
        if args.command == 'start':
            asyncio.run(start_workflow(args))
        elif args.command == 'resume':
            asyncio.run(resume_workflow(args))
        elif args.command == 'list':
            list_workflows(args)
        elif args.command == 'status':
            show_status(args)
        elif args.command == 'clean':
            clean_workflows(args)
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1) 
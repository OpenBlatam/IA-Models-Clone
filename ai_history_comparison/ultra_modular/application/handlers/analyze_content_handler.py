"""
Analyze Content Handler
======================

Single responsibility: Handle analyze content commands.
"""

from typing import Any, Dict
from datetime import datetime
import logging

from ..commands.analyze_content_command import AnalyzeContentCommand
from ...domain.entities.history_entry import HistoryEntry
from ...domain.services.content_analyzer import ContentAnalyzer
from ...infrastructure.persistence.history_repository import HistoryRepository
from ...infrastructure.events.event_bus import EventBus
from ...domain.events.analysis_completed_event import AnalysisCompletedEvent

logger = logging.getLogger(__name__)


class AnalyzeContentHandler:
    """
    Handler for analyze content commands.
    
    Single Responsibility: Process analyze content commands and create history entries.
    """
    
    def __init__(
        self,
        content_analyzer: ContentAnalyzer,
        history_repository: HistoryRepository,
        event_bus: EventBus
    ):
        """
        Initialize the handler.
        
        Args:
            content_analyzer: Service for analyzing content
            history_repository: Repository for history entries
            event_bus: Event bus for publishing events
        """
        self._content_analyzer = content_analyzer
        self._history_repository = history_repository
        self._event_bus = event_bus
    
    async def handle(self, command: AnalyzeContentCommand) -> Dict[str, Any]:
        """
        Handle analyze content command.
        
        Args:
            command: Analyze content command
            
        Returns:
            Result dictionary with entry details
            
        Raises:
            ValueError: If command is invalid
            RuntimeError: If analysis fails
        """
        try:
            # Validate command
            if not command.is_valid():
                raise ValueError("Invalid analyze content command")
            
            logger.info(f"Processing analyze content command: {command.command_id}")
            
            # Analyze content
            metrics = self._content_analyzer.analyze(command.content)
            
            # Create history entry
            entry = HistoryEntry.create(
                content=command.content,
                model_version=command.model_version,
                metrics=metrics,
                metadata=command.metadata
            )
            
            # Save to repository
            saved_entry = await self._history_repository.save(entry)
            
            # Publish event
            event = AnalysisCompletedEvent(
                entry_id=saved_entry.id,
                analysis_type="content_analysis",
                results=metrics.to_dict(),
                command_id=command.command_id,
                user_id=command.user_id
            )
            await self._event_bus.publish(event)
            
            logger.info(f"Successfully analyzed content: {saved_entry.id}")
            
            return {
                "success": True,
                "entry_id": saved_entry.id,
                "metrics": metrics.to_dict(),
                "quality_score": saved_entry.calculate_quality_score(),
                "timestamp": saved_entry.timestamp.isoformat(),
                "command_id": command.command_id
            }
            
        except Exception as e:
            logger.error(f"Error handling analyze content command {command.command_id}: {e}")
            raise RuntimeError(f"Failed to analyze content: {str(e)}")
    
    def can_handle(self, command: Any) -> bool:
        """
        Check if handler can handle the command.
        
        Args:
            command: Command to check
            
        Returns:
            True if handler can handle the command
        """
        return isinstance(command, AnalyzeContentCommand)
    
    def get_handler_name(self) -> str:
        """
        Get handler name.
        
        Returns:
            Handler name
        """
        return "AnalyzeContentHandler"
    
    def get_supported_command_type(self) -> type:
        """
        Get supported command type.
        
        Returns:
            Command type this handler supports
        """
        return AnalyzeContentCommand





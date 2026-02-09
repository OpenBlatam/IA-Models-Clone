"""
Export Service Interface

Defines contract for export functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class IExportService(ABC):
    """Interface for export service"""
    
    @abstractmethod
    async def export_analysis(
        self,
        track_id: str,
        analysis_data: Dict[str, Any],
        format: str = "json",
        include_coaching: bool = False
    ) -> str:
        """
        Export analysis data in specified format.
        
        Args:
            track_id: Track ID
            analysis_data: Analysis data to export
            format: Export format (json, text, markdown, csv)
            include_coaching: Whether to include coaching data
        
        Returns:
            Exported data as string
        """
        pass
    
    @abstractmethod
    async def export_to_json(
        self,
        analysis_data: Dict[str, Any]
    ) -> str:
        """Export to JSON format"""
        pass
    
    @abstractmethod
    async def export_to_text(
        self,
        analysis_data: Dict[str, Any]
    ) -> str:
        """Export to plain text format"""
        pass
    
    @abstractmethod
    async def export_to_markdown(
        self,
        analysis_data: Dict[str, Any]
    ) -> str:
        """Export to Markdown format"""
        pass
    
    @abstractmethod
    async def export_to_csv(
        self,
        analysis_data: Dict[str, Any]
    ) -> str:
        """Export to CSV format"""
        pass





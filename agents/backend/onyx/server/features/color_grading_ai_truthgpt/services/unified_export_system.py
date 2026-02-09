"""
Unified Export System for Color Grading AI
===========================================

Consolidates export services:
- ParameterExporter (parameter export)
- ComparisonGenerator (comparison generation)

Features:
- Unified export interface
- Multiple export formats
- Comparison generation
- Custom export templates
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .parameter_exporter import ParameterExporter, ColorParameters
from .comparison_generator import ComparisonGenerator

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Export formats."""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    YAML = "yaml"
    LUT = "lut"
    PRESET = "preset"


@dataclass
class ExportResult:
    """Export result."""
    success: bool
    format: ExportFormat
    output_path: Optional[str] = None
    data: Any = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class UnifiedExportSystem:
    """
    Unified export system.
    
    Consolidates:
    - ParameterExporter: Parameter export
    - ComparisonGenerator: Comparison generation
    
    Features:
    - Unified export interface
    - Multiple export formats
    - Comparison generation
    """
    
    def __init__(self, output_dir: str = "exports"):
        """
        Initialize unified export system.
        
        Args:
            output_dir: Output directory for exports
        """
        from pathlib import Path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.parameter_exporter = ParameterExporter()
        self.comparison_generator = ComparisonGenerator()
        
        logger.info("Initialized UnifiedExportSystem")
    
    async def export_parameters(
        self,
        color_params: Dict[str, Any],
        format: ExportFormat = ExportFormat.JSON,
        output_path: Optional[str] = None
    ) -> ExportResult:
        """
        Export color parameters.
        
        Args:
            color_params: Color parameters
            format: Export format
            output_path: Optional output path
            
        Returns:
            Export result
        """
        try:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = str(self.output_dir / f"params_{timestamp}.{format.value}")
            
            # Convert to ColorParameters
            color_parameters = ColorParameters(**color_params)
            
            # Export based on format
            if format == ExportFormat.JSON:
                data = self.parameter_exporter.export_json(color_parameters, output_path)
            elif format == ExportFormat.XML:
                data = self.parameter_exporter.export_xml(color_parameters, output_path)
            elif format == ExportFormat.CSV:
                data = self.parameter_exporter.export_csv(color_parameters, output_path)
            elif format == ExportFormat.YAML:
                data = self.parameter_exporter.export_yaml(color_parameters, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return ExportResult(
                success=True,
                format=format,
                output_path=output_path,
                data=data
            )
        
        except Exception as e:
            logger.error(f"Export error: {e}")
            return ExportResult(
                success=False,
                format=format,
                error=str(e)
            )
    
    async def generate_comparison(
        self,
        original_path: str,
        graded_path: str,
        output_path: Optional[str] = None,
        layout: str = "side_by_side"
    ) -> ExportResult:
        """
        Generate comparison image/video.
        
        Args:
            original_path: Original media path
            graded_path: Graded media path
            output_path: Optional output path
            layout: Comparison layout
            
        Returns:
            Export result
        """
        try:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = str(self.output_dir / f"comparison_{timestamp}.jpg")
            
            result = await self.comparison_generator.generate_comparison(
                original_path=original_path,
                graded_path=graded_path,
                output_path=output_path,
                layout=layout
            )
            
            return ExportResult(
                success=result.get("success", False),
                format=ExportFormat.JSON,  # Comparison is image/video, not a format
                output_path=output_path,
                data=result
            )
        
        except Exception as e:
            logger.error(f"Comparison generation error: {e}")
            return ExportResult(
                success=False,
                format=ExportFormat.JSON,
                error=str(e)
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get export statistics."""
        return {
            "output_directory": str(self.output_dir),
            "parameter_exporter_available": self.parameter_exporter is not None,
            "comparison_generator_available": self.comparison_generator is not None,
        }



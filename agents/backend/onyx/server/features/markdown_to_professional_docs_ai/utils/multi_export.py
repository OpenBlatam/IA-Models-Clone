"""Multi-format export utilities"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import logging

logger = logging.getLogger(__name__)


class MultiFormatExporter:
    """Export to multiple formats simultaneously"""
    
    def __init__(self):
        from services.converter_service import ConverterService
        self.converter_service = ConverterService()
    
    async def export_to_multiple_formats(
        self,
        parsed_content: Dict[str, Any],
        output_formats: List[str],
        base_filename: str,
        include_charts: bool = True,
        include_tables: bool = True,
        custom_styling: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Export document to multiple formats simultaneously
        
        Args:
            parsed_content: Parsed Markdown content
            output_formats: List of formats to export to
            base_filename: Base filename (without extension)
            include_charts: Include charts
            include_tables: Include tables
            custom_styling: Custom styling
            
        Returns:
            Export results
        """
        results = {
            "formats": {},
            "successful": 0,
            "failed": 0,
            "total": len(output_formats)
        }
        
        # Export to all formats in parallel
        async def export_format(format_name):
            try:
                output_path = await self.converter_service.convert(
                    parsed_content=parsed_content,
                    output_format=format_name,
                    include_charts=include_charts,
                    include_tables=include_tables,
                    custom_styling=custom_styling,
                    filename_suffix=f"_{base_filename}"
                )
                
                file_size = Path(output_path).stat().st_size if Path(output_path).exists() else 0
                
                return {
                    "format": format_name,
                    "success": True,
                    "output_path": output_path,
                    "file_size": file_size
                }
            except Exception as e:
                logger.error(f"Error exporting to {format_name}: {e}")
                return {
                    "format": format_name,
                    "success": False,
                    "error": str(e)
                }
        
        # Process all formats
        export_tasks = [export_format(fmt) for fmt in output_formats]
        export_results = await asyncio.gather(*export_tasks)
        
        # Process results
        for result in export_results:
            format_name = result["format"]
            results["formats"][format_name] = result
            
            if result.get("success"):
                results["successful"] += 1
            else:
                results["failed"] += 1
        
        return results
    
    async def create_format_package(
        self,
        parsed_content: Dict[str, Any],
        output_formats: List[str],
        package_name: str,
        include_charts: bool = True,
        include_tables: bool = True
    ) -> Optional[str]:
        """
        Create a package with multiple format exports
        
        Args:
            parsed_content: Parsed Markdown content
            output_formats: Formats to include
            package_name: Package name
            include_charts: Include charts
            include_tables: Include tables
            
        Returns:
            Path to package file or None
        """
        try:
            from utils.document_compressor import get_document_compressor
            from config import settings
            import zipfile
            import tempfile
            
            # Export to all formats
            export_results = await self.export_to_multiple_formats(
                parsed_content,
                output_formats,
                package_name,
                include_charts,
                include_tables
            )
            
            # Create ZIP package
            package_path = Path(settings.output_dir) / f"{package_name}_package.zip"
            
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for format_name, result in export_results["formats"].items():
                    if result.get("success"):
                        output_path = result["output_path"]
                        zipf.write(output_path, Path(output_path).name)
            
            return str(package_path)
        except Exception as e:
            logger.error(f"Error creating format package: {e}")
            return None


# Global exporter
_multi_exporter: Optional[MultiFormatExporter] = None


def get_multi_exporter() -> MultiFormatExporter:
    """Get global multi-format exporter"""
    global _multi_exporter
    if _multi_exporter is None:
        _multi_exporter = MultiFormatExporter()
    return _multi_exporter


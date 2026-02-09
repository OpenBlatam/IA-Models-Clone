"""
Export Mixin

Contains export and save functionality.
"""

import logging
import json
from typing import Union, Dict, Any, Optional, List
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class ExportMixin:
    """
    Mixin providing export and save functionality.
    
    This mixin contains:
    - Image export
    - Batch export
    - Report export
    - Statistics export
    - Configuration export
    """
    
    def export_image(
        self,
        image: Image.Image,
        output_path: Union[str, Path],
        format: str = "PNG",
        quality: int = 95,
        optimize: bool = True
    ) -> bool:
        """
        Export image to file.
        
        Args:
            image: Image to export
            output_path: Output file path
            format: Image format (PNG, JPEG, etc.)
            quality: Quality for JPEG (1-100)
            optimize: Optimize file size
            
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_kwargs = {}
            if format.upper() == "JPEG":
                save_kwargs["quality"] = quality
                save_kwargs["optimize"] = optimize
            elif format.upper() == "PNG":
                save_kwargs["optimize"] = optimize
            
            image.save(output_path, format=format, **save_kwargs)
            logger.info(f"Image exported to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export image: {e}")
            return False
    
    def export_batch(
        self,
        images: List[Image.Image],
        output_dir: Union[str, Path],
        base_name: str = "upscaled",
        format: str = "PNG",
        quality: int = 95
    ) -> List[bool]:
        """
        Export multiple images to directory.
        
        Args:
            images: List of images to export
            output_dir: Output directory
            base_name: Base name for files
            format: Image format
            quality: Quality for JPEG
            
        Returns:
            List of success status for each image
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        for i, image in enumerate(images):
            output_path = output_dir / f"{base_name}_{i+1:04d}.{format.lower()}"
            success = self.export_image(image, output_path, format=format, quality=quality)
            results.append(success)
        
        logger.info(f"Exported {sum(results)}/{len(images)} images to {output_dir}")
        return results
    
    def export_report(
        self,
        report: Dict[str, Any],
        output_path: Union[str, Path],
        format: str = "json"
    ) -> bool:
        """
        Export report to file.
        
        Args:
            report: Report dictionary
            output_path: Output file path
            format: Export format (json, txt)
            
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
            elif format.lower() == "txt":
                with open(output_path, 'w') as f:
                    self._write_text_report(f, report)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Report exported to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return False
    
    def export_statistics(
        self,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> bool:
        """
        Export statistics to file.
        
        Args:
            output_path: Output file path
            format: Export format (json, txt)
            
        Returns:
            True if successful
        """
        if not hasattr(self, 'stats'):
            logger.warning("No statistics available")
            return False
        
        stats = self.get_statistics()
        return self.export_report(stats, output_path, format=format)
    
    def export_comparison(
        self,
        comparison: Dict[str, Any],
        output_path: Union[str, Path],
        format: str = "json"
    ) -> bool:
        """
        Export method comparison results.
        
        Args:
            comparison: Comparison dictionary
            output_path: Output file path
            format: Export format (json, txt)
            
        Returns:
            True if successful
        """
        return self.export_report(comparison, output_path, format=format)
    
    def _write_text_report(self, file, report: Dict[str, Any], indent: int = 0):
        """Write report in text format."""
        prefix = "  " * indent
        for key, value in report.items():
            if isinstance(value, dict):
                file.write(f"{prefix}{key}:\n")
                self._write_text_report(file, value, indent + 1)
            elif isinstance(value, list):
                file.write(f"{prefix}{key}:\n")
                for item in value:
                    if isinstance(item, dict):
                        self._write_text_report(file, item, indent + 1)
                    else:
                        file.write(f"{prefix}  - {item}\n")
            else:
                file.write(f"{prefix}{key}: {value}\n")



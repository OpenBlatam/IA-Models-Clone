"""Storage repository implementation."""

from pathlib import Path
from typing import Optional
from PIL import Image

from core.interfaces import IStorageRepository
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class FileStorageRepository(IStorageRepository):
    """File-based storage repository."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or settings.output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_visualization(
        self,
        visualization_id: str,
        image: Image.Image,
        format: str
    ) -> Path:
        """Save visualization image to file system."""
        output_path = self.base_dir / f"{visualization_id}.{format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format=format.upper())
        logger.info(f"Saved visualization to {output_path}")
        return output_path
    
    async def get_visualization(
        self,
        visualization_id: str
    ) -> Optional[Path]:
        """Get visualization image path."""
        image_path = self.base_dir / f"{visualization_id}.{settings.output_format}"
        if image_path.exists() and image_path.is_file():
            return image_path
        return None
    
    async def delete_visualization(
        self,
        visualization_id: str
    ) -> bool:
        """Delete visualization image."""
        image_path = self.base_dir / f"{visualization_id}.{settings.output_format}"
        if image_path.exists():
            image_path.unlink()
            logger.info(f"Deleted visualization {visualization_id}")
            return True
        return False


"""Test helpers and utilities."""

from typing import Any, Dict, Optional
from unittest.mock import Mock, AsyncMock, MagicMock
import asyncio
from pathlib import Path
import tempfile
import shutil

from PIL import Image
import io


class MockImage:
    """Helper for creating mock images in tests."""
    
    @staticmethod
    def create_rgb(width: int = 500, height: int = 500, color: str = "red") -> Image.Image:
        """
        Create a test RGB image.
        
        Args:
            width: Image width
            height: Image height
            color: Color name
            
        Returns:
            PIL Image
        """
        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }
        rgb = color_map.get(color, (255, 0, 0))
        return Image.new('RGB', (width, height), color=rgb)
    
    @staticmethod
    def create_bytes(width: int = 500, height: int = 500, format: str = "PNG") -> bytes:
        """
        Create image as bytes.
        
        Args:
            width: Image width
            height: Image height
            format: Image format
            
        Returns:
            Image bytes
        """
        img = MockImage.create_rgb(width, height)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=format)
        return img_bytes.getvalue()


class TempDirectory:
    """Context manager for temporary directories."""
    
    def __init__(self, prefix: str = "test_"):
        self.prefix = prefix
        self.path: Optional[Path] = None
    
    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix))
        return self.path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and self.path.exists():
            shutil.rmtree(self.path)


class AsyncTestHelper:
    """Helper for async testing."""
    
    @staticmethod
    def run_async(coro):
        """
        Run async function in test.
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Result
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    
    @staticmethod
    def create_async_mock(return_value: Any = None, side_effect: Any = None):
        """
        Create async mock.
        
        Args:
            return_value: Return value
            side_effect: Side effect (exception or function)
            
        Returns:
            AsyncMock
        """
        mock = AsyncMock()
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock


class MockFactory:
    """Factory for creating common mocks."""
    
    @staticmethod
    def create_storage_repository():
        """Create mock storage repository."""
        from core.interfaces import IStorageRepository
        
        mock = MagicMock(spec=IStorageRepository)
        mock.save_visualization = AsyncMock(return_value=Path("/tmp/test.jpg"))
        mock.get_visualization = AsyncMock(return_value=Path("/tmp/test.jpg"))
        mock.delete_visualization = AsyncMock(return_value=True)
        return mock
    
    @staticmethod
    def create_cache_repository():
        """Create mock cache repository."""
        from core.interfaces import ICacheRepository
        
        mock = MagicMock(spec=ICacheRepository)
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock()
        mock.delete = AsyncMock(return_value=True)
        mock.clear = AsyncMock()
        return mock
    
    @staticmethod
    def create_image_processor():
        """Create mock image processor."""
        from core.interfaces import IImageProcessor
        
        mock = MagicMock(spec=IImageProcessor)
        mock.load_from_bytes = AsyncMock(return_value=MockImage.create_rgb())
        mock.load_from_url = AsyncMock(return_value=MockImage.create_rgb())
        mock.save_image = AsyncMock()
        mock.validate_image = Mock(return_value=True)
        return mock
    
    @staticmethod
    def create_ai_processor():
        """Create mock AI processor."""
        from core.interfaces import IAProcessor
        
        mock = MagicMock(spec=IAProcessor)
        mock.process_surgery_visualization = AsyncMock(
            return_value=MockImage.create_rgb()
        )
        return mock
    
    @staticmethod
    def create_metrics_collector():
        """Create mock metrics collector."""
        from core.interfaces import IMetricsCollector
        
        mock = MagicMock(spec=IMetricsCollector)
        mock.increment = Mock()
        mock.record_timing = Mock()
        mock.set_gauge = Mock()
        return mock
    
    @staticmethod
    def create_event_publisher():
        """Create mock event publisher."""
        from core.interfaces import IEventPublisher
        
        mock = MagicMock(spec=IEventPublisher)
        mock.publish = AsyncMock()
        return mock


class AssertHelper:
    """Helper for common assertions."""
    
    @staticmethod
    def assert_image_valid(image: Image.Image) -> None:
        """
        Assert image is valid.
        
        Args:
            image: Image to validate
            
        Raises:
            AssertionError: If image is invalid
        """
        assert image is not None, "Image is None"
        assert isinstance(image, Image.Image), "Not a PIL Image"
        assert image.size[0] > 0 and image.size[1] > 0, "Invalid image size"
    
    @staticmethod
    def assert_path_exists(path: Path) -> None:
        """
        Assert path exists.
        
        Args:
            path: Path to check
            
        Raises:
            AssertionError: If path doesn't exist
        """
        assert path.exists(), f"Path does not exist: {path}"
    
    @staticmethod
    def assert_dict_contains(dict_obj: Dict, keys: list) -> None:
        """
        Assert dictionary contains keys.
        
        Args:
            dict_obj: Dictionary to check
            keys: Keys that must exist
            
        Raises:
            AssertionError: If keys are missing
        """
        for key in keys:
            assert key in dict_obj, f"Key '{key}' not found in dictionary"


class FixtureHelper:
    """Helper for creating test fixtures."""
    
    @staticmethod
    def create_visualization_request(**kwargs) -> Dict[str, Any]:
        """
        Create visualization request dict.
        
        Args:
            **kwargs: Request parameters
            
        Returns:
            Request dictionary
        """
        defaults = {
            "surgery_type": "rhinoplasty",
            "intensity": 0.5,
            "image_data": MockImage.create_bytes(),
        }
        defaults.update(kwargs)
        return defaults
    
    @staticmethod
    def create_test_settings(**kwargs) -> Dict[str, Any]:
        """
        Create test settings dict.
        
        Args:
            **kwargs: Settings overrides
            
        Returns:
            Settings dictionary
        """
        defaults = {
            "upload_dir": "/tmp/test_uploads",
            "output_dir": "/tmp/test_outputs",
            "max_image_size_mb": 10,
            "output_format": "png",
            "supported_formats": ["jpg", "png", "webp"],
        }
        defaults.update(kwargs)
        return defaults


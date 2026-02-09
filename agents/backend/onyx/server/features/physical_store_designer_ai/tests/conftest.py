"""
Pytest configuration and fixtures
"""

import pytest
from pathlib import Path
from typing import Generator
import tempfile
import shutil

from ..core.models import StoreDesign, StoreType, DesignStyle
from ..services.storage_service import StorageService
from ..services.chat_service import ChatService
from ..services.store_designer_service import StoreDesignerService


@pytest.fixture
def temp_storage_path() -> Generator[Path, None, None]:
    """Create temporary storage directory for tests"""
    temp_dir = tempfile.mkdtemp()
    path = Path(temp_dir)
    yield path
    shutil.rmtree(temp_dir)


@pytest.fixture
def storage_service(temp_storage_path: Path) -> StorageService:
    """Create storage service with temporary path"""
    return StorageService(storage_path=str(temp_storage_path))


@pytest.fixture
def chat_service() -> ChatService:
    """Create chat service"""
    return ChatService()


@pytest.fixture
def store_designer_service() -> StoreDesignerService:
    """Create store designer service"""
    return StoreDesignerService()


@pytest.fixture
def sample_store_design() -> StoreDesign:
    """Create sample store design for testing"""
    from ..core.models import StoreDesignRequest
    
    request = StoreDesignRequest(
        store_name="Test Store",
        store_type=StoreType.CAFE,
        style_preference=DesignStyle.MODERN,
        budget_range="medio",
        location="Test Location",
        target_audience="Test Audience",
        dimensions={"width": 10.0, "length": 15.0, "height": 3.0}
    )
    
    service = StoreDesignerService()
    return service.generate_design(request)









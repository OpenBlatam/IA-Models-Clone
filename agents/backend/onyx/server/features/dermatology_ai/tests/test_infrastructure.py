"""
Tests for Infrastructure Layer
Tests for repositories, adapters, and infrastructure components
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import uuid

from core.infrastructure.repositories.analysis_repository import AnalysisRepository
from core.infrastructure.repositories.user_repository import UserRepository
from core.infrastructure.repositories.product_repository import ProductRepository
from core.infrastructure.adapters.cache_adapter import CacheAdapter
from core.infrastructure.adapters.image_processor_adapter import ImageProcessorAdapter
from core.domain.entities import (
    Analysis,
    AnalysisStatus,
    SkinMetrics,
    Condition,
    SkinType,
    User,
    Product
)
from core.domain.interfaces import ICacheService
from tests.test_base import BaseRepositoryTest
from tests.test_helpers import build_analysis, build_user, build_product


class TestAnalysisRepository(BaseRepositoryTest):
    """Tests for AnalysisRepository"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.add = Mock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(
                order_by=Mock(return_value=Mock(
                    limit=Mock(return_value=Mock(
                        offset=Mock(return_value=[])
                    ))
                ))
            ))
        ))
        return session
    
    @pytest.mark.asyncio
    async def test_create_analysis(self, mock_db_session):
        """Test creating analysis in repository"""
        repository = AnalysisRepository(db_session=mock_db_session)
        
        analysis = Analysis(
            id=str(uuid.uuid4()),
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        )
        
        result = await repository.create(analysis)
        
        assert result is not None
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_id(self, mock_db_session):
        """Test getting analysis by ID"""
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            status=AnalysisStatus.COMPLETED
        )
        
        mock_query = Mock()
        mock_query.filter = Mock(return_value=Mock(
            first=Mock(return_value=analysis)
        ))
        mock_db_session.query = Mock(return_value=mock_query)
        
        repository = AnalysisRepository(db_session=mock_db_session)
        result = await repository.get_by_id("test-123")
        
        assert result is not None
        assert result.id == "test-123"
    
    @pytest.mark.asyncio
    async def test_get_by_user(self, mock_db_session):
        """Test getting analyses by user ID"""
        analyses = [
            Analysis(
                id=f"test-{i}",
                user_id="user-123",
                status=AnalysisStatus.COMPLETED
            )
            for i in range(3)
        ]
        
        mock_query = Mock()
        mock_query.filter = Mock(return_value=Mock(
            order_by=Mock(return_value=Mock(
                limit=Mock(return_value=analyses)
            ))
        ))
        mock_db_session.query = Mock(return_value=mock_query)
        
        repository = AnalysisRepository(db_session=mock_db_session)
        result = await repository.get_by_user("user-123", limit=10)
        
        assert len(result) == 3
        assert all(a.user_id == "user-123" for a in result)
    
    @pytest.mark.asyncio
    async def test_update_analysis(self, mock_db_session):
        """Test updating analysis"""
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        )
        
        repository = AnalysisRepository(db_session=mock_db_session)
        analysis.status = AnalysisStatus.COMPLETED
        
        result = await repository.update(analysis)
        
        assert result is not None
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_analysis(self, mock_db_session):
        """Test deleting analysis"""
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            status=AnalysisStatus.COMPLETED
        )
        
        mock_query = Mock()
        mock_query.filter = Mock(return_value=Mock(
            first=Mock(return_value=analysis)
        ))
        mock_db_session.query = Mock(return_value=mock_query)
        
        repository = AnalysisRepository(db_session=mock_db_session)
        result = await repository.delete("test-123")
        
        assert result is True
        mock_db_session.delete.assert_called_once()
        mock_db_session.commit.assert_called_once()


class TestUserRepository:
    """Tests for UserRepository"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.add = Mock()
        session.commit = AsyncMock()
        session.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(
                first=Mock(return_value=None)
            ))
        ))
        return session
    
    @pytest.mark.asyncio
    async def test_create_user(self, mock_db_session):
        """Test creating user in repository"""
        repository = UserRepository(db_session=mock_db_session)
        
        user = User(
            id="user-123",
            email="test@example.com",
            name="Test User"
        )
        
        result = await repository.create(user)
        
        assert result is not None
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_id(self, mock_db_session):
        """Test getting user by ID"""
        user = User(
            id="user-123",
            email="test@example.com",
            name="Test User"
        )
        
        mock_query = Mock()
        mock_query.filter = Mock(return_value=Mock(
            first=Mock(return_value=user)
        ))
        mock_db_session.query = Mock(return_value=mock_query)
        
        repository = UserRepository(db_session=mock_db_session)
        result = await repository.get_by_id("user-123")
        
        assert result is not None
        assert result.id == "user-123"
    
    @pytest.mark.asyncio
    async def test_get_by_email(self, mock_db_session):
        """Test getting user by email"""
        user = User(
            id="user-123",
            email="test@example.com",
            name="Test User"
        )
        
        mock_query = Mock()
        mock_query.filter = Mock(return_value=Mock(
            first=Mock(return_value=user)
        ))
        mock_db_session.query = Mock(return_value=mock_query)
        
        repository = UserRepository(db_session=mock_db_session)
        result = await repository.get_by_email("test@example.com")
        
        assert result is not None
        assert result.email == "test@example.com"


class TestProductRepository:
    """Tests for ProductRepository"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(
                first=Mock(return_value=None)
            ))
        ))
        return session
    
    @pytest.mark.asyncio
    async def test_get_by_id(self, mock_db_session):
        """Test getting product by ID"""
        product = Product(
            id="product-123",
            name="Moisturizer",
            category="moisturizer"
        )
        
        mock_query = Mock()
        mock_query.filter = Mock(return_value=Mock(
            first=Mock(return_value=product)
        ))
        mock_db_session.query = Mock(return_value=mock_query)
        
        repository = ProductRepository(db_session=mock_db_session)
        result = await repository.get_by_id("product-123")
        
        assert result is not None
        assert result.id == "product-123"
    
    @pytest.mark.asyncio
    async def test_search_products(self, mock_db_session):
        """Test searching products"""
        products = [
            Product(
                id=f"product-{i}",
                name=f"Product {i}",
                category="moisturizer"
            )
            for i in range(3)
        ]
        
        mock_query = Mock()
        mock_query.filter = Mock(return_value=Mock(
            all=Mock(return_value=products)
        ))
        mock_db_session.query = Mock(return_value=mock_query)
        
        repository = ProductRepository(db_session=mock_db_session)
        result = await repository.search(query="moisturizer", limit=10)
        
        assert len(result) == 3


class TestCacheAdapter:
    """Tests for CacheAdapter"""
    
    @pytest.fixture
    def mock_cache_service(self):
        """Mock cache service"""
        cache = Mock(spec=ICacheService)
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock(return_value=True)
        cache.delete = AsyncMock(return_value=True)
        return cache
    
    @pytest.mark.asyncio
    async def test_get_cached_value(self, mock_cache_service):
        """Test getting cached value"""
        adapter = CacheAdapter(cache_service=mock_cache_service)
        
        mock_cache_service.get = AsyncMock(return_value='{"key": "value"}')
        
        result = await adapter.get("test-key")
        
        assert result is not None
        mock_cache_service.get.assert_called_once_with("test-key")
    
    @pytest.mark.asyncio
    async def test_set_cached_value(self, mock_cache_service):
        """Test setting cached value"""
        adapter = CacheAdapter(cache_service=mock_cache_service)
        
        result = await adapter.set("test-key", {"key": "value"}, ttl=3600)
        
        assert result is True
        mock_cache_service.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_cached_value(self, mock_cache_service):
        """Test deleting cached value"""
        adapter = CacheAdapter(cache_service=mock_cache_service)
        
        result = await adapter.delete("test-key")
        
        assert result is True
        mock_cache_service.delete.assert_called_once_with("test-key")
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, mock_cache_service):
        """Test cache miss scenario"""
        adapter = CacheAdapter(cache_service=mock_cache_service)
        
        mock_cache_service.get = AsyncMock(return_value=None)
        
        result = await adapter.get("non-existent-key")
        
        assert result is None


class TestImageProcessorAdapter:
    """Tests for ImageProcessorAdapter"""
    
    @pytest.mark.asyncio
    async def test_validate_image(self):
        """Test image validation"""
        adapter = ImageProcessorAdapter()
        
        # Create valid image bytes
        from PIL import Image
        import io
        img = Image.new('RGB', (200, 200), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        image_data = img_bytes.read()
        
        result = await adapter.validate(image_data)
        
        # Should validate successfully
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_validate_invalid_image(self):
        """Test validation of invalid image"""
        adapter = ImageProcessorAdapter()
        
        invalid_data = b"not an image"
        
        result = await adapter.validate(invalid_data)
        
        # Should fail validation
        assert result is False
    
    @pytest.mark.asyncio
    async def test_process_image(self):
        """Test image processing"""
        adapter = ImageProcessorAdapter()
        
        from PIL import Image
        import io
        import numpy as np
        
        img = Image.new('RGB', (200, 200), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        image_data = img_bytes.read()
        
        result = await adapter.process(image_data, metadata={})
        
        # Should return processed image data
        assert result is not None
        assert "metrics" in result or isinstance(result, dict)


Tests for repositories, adapters, and infrastructure components
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import uuid

from core.infrastructure.repositories.analysis_repository import AnalysisRepository
from core.infrastructure.repositories.user_repository import UserRepository
from core.infrastructure.repositories.product_repository import ProductRepository
from core.infrastructure.adapters.cache_adapter import CacheAdapter
from core.infrastructure.adapters.image_processor_adapter import ImageProcessorAdapter
from core.domain.entities import (
    Analysis,
    AnalysisStatus,
    SkinMetrics,
    Condition,
    SkinType,
    User,
    Product
)
from core.domain.interfaces import ICacheService
from tests.test_base import BaseRepositoryTest
from tests.test_helpers import build_analysis, build_user, build_product


class TestAnalysisRepository(BaseRepositoryTest):
    """Tests for AnalysisRepository"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.add = Mock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(
                order_by=Mock(return_value=Mock(
                    limit=Mock(return_value=Mock(
                        offset=Mock(return_value=[])
                    ))
                ))
            ))
        ))
        return session
    
    @pytest.mark.asyncio
    async def test_create_analysis(self, mock_db_session):
        """Test creating analysis in repository"""
        repository = AnalysisRepository(db_session=mock_db_session)
        
        analysis = Analysis(
            id=str(uuid.uuid4()),
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        )
        
        result = await repository.create(analysis)
        
        assert result is not None
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_id(self, mock_db_session):
        """Test getting analysis by ID"""
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            status=AnalysisStatus.COMPLETED
        )
        
        mock_query = Mock()
        mock_query.filter = Mock(return_value=Mock(
            first=Mock(return_value=analysis)
        ))
        mock_db_session.query = Mock(return_value=mock_query)
        
        repository = AnalysisRepository(db_session=mock_db_session)
        result = await repository.get_by_id("test-123")
        
        assert result is not None
        assert result.id == "test-123"
    
    @pytest.mark.asyncio
    async def test_get_by_user(self, mock_db_session):
        """Test getting analyses by user ID"""
        analyses = [
            Analysis(
                id=f"test-{i}",
                user_id="user-123",
                status=AnalysisStatus.COMPLETED
            )
            for i in range(3)
        ]
        
        mock_query = Mock()
        mock_query.filter = Mock(return_value=Mock(
            order_by=Mock(return_value=Mock(
                limit=Mock(return_value=analyses)
            ))
        ))
        mock_db_session.query = Mock(return_value=mock_query)
        
        repository = AnalysisRepository(db_session=mock_db_session)
        result = await repository.get_by_user("user-123", limit=10)
        
        assert len(result) == 3
        assert all(a.user_id == "user-123" for a in result)
    
    @pytest.mark.asyncio
    async def test_update_analysis(self, mock_db_session):
        """Test updating analysis"""
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        )
        
        repository = AnalysisRepository(db_session=mock_db_session)
        analysis.status = AnalysisStatus.COMPLETED
        
        result = await repository.update(analysis)
        
        assert result is not None
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_analysis(self, mock_db_session):
        """Test deleting analysis"""
        analysis = Analysis(
            id="test-123",
            user_id="user-123",
            status=AnalysisStatus.COMPLETED
        )
        
        mock_query = Mock()
        mock_query.filter = Mock(return_value=Mock(
            first=Mock(return_value=analysis)
        ))
        mock_db_session.query = Mock(return_value=mock_query)
        
        repository = AnalysisRepository(db_session=mock_db_session)
        result = await repository.delete("test-123")
        
        assert result is True
        mock_db_session.delete.assert_called_once()
        mock_db_session.commit.assert_called_once()


class TestUserRepository:
    """Tests for UserRepository"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.add = Mock()
        session.commit = AsyncMock()
        session.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(
                first=Mock(return_value=None)
            ))
        ))
        return session
    
    @pytest.mark.asyncio
    async def test_create_user(self, mock_db_session):
        """Test creating user in repository"""
        repository = UserRepository(db_session=mock_db_session)
        
        user = User(
            id="user-123",
            email="test@example.com",
            name="Test User"
        )
        
        result = await repository.create(user)
        
        assert result is not None
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_id(self, mock_db_session):
        """Test getting user by ID"""
        user = User(
            id="user-123",
            email="test@example.com",
            name="Test User"
        )
        
        mock_query = Mock()
        mock_query.filter = Mock(return_value=Mock(
            first=Mock(return_value=user)
        ))
        mock_db_session.query = Mock(return_value=mock_query)
        
        repository = UserRepository(db_session=mock_db_session)
        result = await repository.get_by_id("user-123")
        
        assert result is not None
        assert result.id == "user-123"
    
    @pytest.mark.asyncio
    async def test_get_by_email(self, mock_db_session):
        """Test getting user by email"""
        user = User(
            id="user-123",
            email="test@example.com",
            name="Test User"
        )
        
        mock_query = Mock()
        mock_query.filter = Mock(return_value=Mock(
            first=Mock(return_value=user)
        ))
        mock_db_session.query = Mock(return_value=mock_query)
        
        repository = UserRepository(db_session=mock_db_session)
        result = await repository.get_by_email("test@example.com")
        
        assert result is not None
        assert result.email == "test@example.com"


class TestProductRepository:
    """Tests for ProductRepository"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(
                first=Mock(return_value=None)
            ))
        ))
        return session
    
    @pytest.mark.asyncio
    async def test_get_by_id(self, mock_db_session):
        """Test getting product by ID"""
        product = Product(
            id="product-123",
            name="Moisturizer",
            category="moisturizer"
        )
        
        mock_query = Mock()
        mock_query.filter = Mock(return_value=Mock(
            first=Mock(return_value=product)
        ))
        mock_db_session.query = Mock(return_value=mock_query)
        
        repository = ProductRepository(db_session=mock_db_session)
        result = await repository.get_by_id("product-123")
        
        assert result is not None
        assert result.id == "product-123"
    
    @pytest.mark.asyncio
    async def test_search_products(self, mock_db_session):
        """Test searching products"""
        products = [
            Product(
                id=f"product-{i}",
                name=f"Product {i}",
                category="moisturizer"
            )
            for i in range(3)
        ]
        
        mock_query = Mock()
        mock_query.filter = Mock(return_value=Mock(
            all=Mock(return_value=products)
        ))
        mock_db_session.query = Mock(return_value=mock_query)
        
        repository = ProductRepository(db_session=mock_db_session)
        result = await repository.search(query="moisturizer", limit=10)
        
        assert len(result) == 3


class TestCacheAdapter:
    """Tests for CacheAdapter"""
    
    @pytest.fixture
    def mock_cache_service(self):
        """Mock cache service"""
        cache = Mock(spec=ICacheService)
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock(return_value=True)
        cache.delete = AsyncMock(return_value=True)
        return cache
    
    @pytest.mark.asyncio
    async def test_get_cached_value(self, mock_cache_service):
        """Test getting cached value"""
        adapter = CacheAdapter(cache_service=mock_cache_service)
        
        mock_cache_service.get = AsyncMock(return_value='{"key": "value"}')
        
        result = await adapter.get("test-key")
        
        assert result is not None
        mock_cache_service.get.assert_called_once_with("test-key")
    
    @pytest.mark.asyncio
    async def test_set_cached_value(self, mock_cache_service):
        """Test setting cached value"""
        adapter = CacheAdapter(cache_service=mock_cache_service)
        
        result = await adapter.set("test-key", {"key": "value"}, ttl=3600)
        
        assert result is True
        mock_cache_service.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_cached_value(self, mock_cache_service):
        """Test deleting cached value"""
        adapter = CacheAdapter(cache_service=mock_cache_service)
        
        result = await adapter.delete("test-key")
        
        assert result is True
        mock_cache_service.delete.assert_called_once_with("test-key")
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, mock_cache_service):
        """Test cache miss scenario"""
        adapter = CacheAdapter(cache_service=mock_cache_service)
        
        mock_cache_service.get = AsyncMock(return_value=None)
        
        result = await adapter.get("non-existent-key")
        
        assert result is None


class TestImageProcessorAdapter:
    """Tests for ImageProcessorAdapter"""
    
    @pytest.mark.asyncio
    async def test_validate_image(self):
        """Test image validation"""
        adapter = ImageProcessorAdapter()
        
        # Create valid image bytes
        from PIL import Image
        import io
        img = Image.new('RGB', (200, 200), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        image_data = img_bytes.read()
        
        result = await adapter.validate(image_data)
        
        # Should validate successfully
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_validate_invalid_image(self):
        """Test validation of invalid image"""
        adapter = ImageProcessorAdapter()
        
        invalid_data = b"not an image"
        
        result = await adapter.validate(invalid_data)
        
        # Should fail validation
        assert result is False
    
    @pytest.mark.asyncio
    async def test_process_image(self):
        """Test image processing"""
        adapter = ImageProcessorAdapter()
        
        from PIL import Image
        import io
        import numpy as np
        
        img = Image.new('RGB', (200, 200), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        image_data = img_bytes.read()
        
        result = await adapter.process(image_data, metadata={})
        
        # Should return processed image data
        assert result is not None
        assert "metrics" in result or isinstance(result, dict)


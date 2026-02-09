"""
Domain Interfaces - Contracts for infrastructure layer
Ports in Hexagonal Architecture
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .entities import Analysis, User, Product, Recommendation, SkinMetrics


class IAnalysisRepository(ABC):
    """Repository interface for Analysis entities"""
    
    @abstractmethod
    async def create(self, analysis: Analysis) -> Analysis:
        """Create new analysis"""
        pass
    
    @abstractmethod
    async def get_by_id(self, analysis_id: str) -> Optional[Analysis]:
        """Get analysis by ID"""
        pass
    
    @abstractmethod
    async def get_by_user(self, user_id: str, limit: int = 10) -> List[Analysis]:
        """Get analyses by user"""
        pass
    
    @abstractmethod
    async def update(self, analysis: Analysis) -> Analysis:
        """Update analysis"""
        pass
    
    @abstractmethod
    async def delete(self, analysis_id: str) -> bool:
        """Delete analysis"""
        pass


class IUserRepository(ABC):
    """Repository interface for User entities"""
    
    @abstractmethod
    async def create(self, user: User) -> User:
        """Create new user"""
        pass
    
    @abstractmethod
    async def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        pass
    
    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        pass
    
    @abstractmethod
    async def update(self, user: User) -> User:
        """Update user"""
        pass


class IProductRepository(ABC):
    """Repository interface for Product entities"""
    
    @abstractmethod
    async def get_by_id(self, product_id: str) -> Optional[Product]:
        """Get product by ID"""
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[Product]:
        """Search products"""
        pass
    
    @abstractmethod
    async def get_by_category(self, category: str, limit: int = 10) -> List[Product]:
        """Get products by category"""
        pass


class IImageProcessor(ABC):
    """Interface for image processing"""
    
    @abstractmethod
    async def process(self, image_data: bytes) -> Dict[str, Any]:
        """Process image and extract features"""
        pass
    
    @abstractmethod
    async def validate(self, image_data: bytes) -> bool:
        """Validate image"""
        pass


class IAnalysisService(ABC):
    """Interface for analysis business logic"""
    
    @abstractmethod
    async def analyze_image(
        self,
        user_id: str,
        image_data: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Analysis:
        """Analyze image and create analysis"""
        pass
    
    @abstractmethod
    async def get_analysis(self, analysis_id: str) -> Optional[Analysis]:
        """Get analysis by ID"""
        pass


class IRecommendationService(ABC):
    """Interface for recommendation business logic"""
    
    @abstractmethod
    async def generate_recommendations(
        self,
        analysis: Analysis,
        user: Optional[User] = None
    ) -> List[Recommendation]:
        """Generate product recommendations based on analysis"""
        pass
    
    @abstractmethod
    async def get_recommendations_for_user(
        self,
        user_id: str
    ) -> List[Recommendation]:
        """Get recommendations for user"""
        pass


class ICacheService(ABC):
    """Interface for caching"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass


class IEventPublisher(ABC):
    """Interface for event publishing"""
    
    @abstractmethod
    async def publish(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """Publish event"""
        pass


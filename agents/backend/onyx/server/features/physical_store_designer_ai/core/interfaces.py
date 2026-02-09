"""
Interfaces and protocols for Physical Store Designer AI
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from .models import StoreDesignRequest, StoreDesign


class IStorageService(ABC):
    """Interface for storage services"""
    
    @abstractmethod
    def save_design(self, design: StoreDesign) -> bool:
        """Save a design"""
        pass
    
    @abstractmethod
    def load_design(self, store_id: str) -> Optional[StoreDesign]:
        """Load a design"""
        pass
    
    @abstractmethod
    def delete_design(self, store_id: str) -> bool:
        """Delete a design"""
        pass


class IChatService(ABC):
    """Interface for chat services"""
    
    @abstractmethod
    def create_session(self) -> str:
        """Create a new chat session"""
        pass
    
    @abstractmethod
    async def generate_response(self, session_id: str, user_message: str) -> str:
        """Generate a response"""
        pass
    
    @abstractmethod
    async def extract_store_info(self, session_id: str) -> Dict[str, Any]:
        """Extract store information from chat"""
        pass


class IDesignService(ABC):
    """Interface for design services"""
    
    @abstractmethod
    async def generate_store_design(self, request: StoreDesignRequest) -> StoreDesign:
        """Generate a store design"""
        pass


class IAnalysisService(ABC):
    """Interface for analysis services"""
    
    @abstractmethod
    def analyze(self, store_id: str, **kwargs) -> Dict[str, Any]:
        """Perform analysis"""
        pass


class IExportService(ABC):
    """Interface for export services"""
    
    @abstractmethod
    def export(self, store_id: str, format: str) -> Optional[str]:
        """Export design in specified format"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats"""
        pass


from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from domain.entities import SEOAnalysis
from domain.value_objects import URL
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
SEO Analysis Repository Interface
Abstract repository for SEO analysis persistence
"""



class SEOAnalysisRepository(ABC):
    """Abstract SEO Analysis Repository - Domain interface"""
    
    @abstractmethod
    async def save(self, analysis: SEOAnalysis) -> SEOAnalysis:
        """Save SEO analysis to repository"""
        pass
    
    @abstractmethod
    async def find_by_id(self, analysis_id: str) -> Optional[SEOAnalysis]:
        """Find analysis by ID"""
        pass
    
    @abstractmethod
    async def find_by_url(self, url: URL) -> Optional[SEOAnalysis]:
        """Find analysis by URL"""
        pass
    
    @abstractmethod
    async def find_recent(self, limit: int = 10) -> List[SEOAnalysis]:
        """Find recent analyses"""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: str, limit: int = 100) -> List[SEOAnalysis]:
        """Find analyses by status"""
        pass
    
    @abstractmethod
    async def find_by_score_range(self, min_score: float, max_score: float, limit: int = 100) -> List[SEOAnalysis]:
        """Find analyses by score range"""
        pass
    
    @abstractmethod
    async def find_stale_analyses(self, max_age_hours: int = 24) -> List[SEOAnalysis]:
        """Find stale analyses that need refresh"""
        pass
    
    @abstractmethod
    async def find_failed_analyses(self, limit: int = 100) -> List[SEOAnalysis]:
        """Find failed analyses"""
        pass
    
    @abstractmethod
    async def delete(self, analysis_id: str) -> bool:
        """Delete analysis by ID"""
        pass
    
    @abstractmethod
    async def delete_by_url(self, url: URL) -> bool:
        """Delete analysis by URL"""
        pass
    
    @abstractmethod
    async def delete_old_analyses(self, max_age_days: int = 30) -> int:
        """Delete old analyses and return count of deleted"""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Get total count of analyses"""
        pass
    
    @abstractmethod
    async def count_by_status(self, status: str) -> int:
        """Get count of analyses by status"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics"""
        pass
    
    @abstractmethod
    async def clear_cache(self) -> bool:
        """Clear repository cache"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check repository health"""
        pass 
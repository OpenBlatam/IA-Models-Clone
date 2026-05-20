"""
NLP Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class Sentiment(str, Enum):
    """Sentiment types"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class TokenizedText:
    """Tokenized text"""
    tokens: List[str]
    original_text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Entity:
    """Named entity"""
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float = 1.0


class NLPBase(ABC):
    """Base interface for NLP"""
    
    @abstractmethod
    async def tokenize(self, text: str) -> TokenizedText:
        """Tokenize text"""
        pass
    
    @abstractmethod
    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities"""
        pass
    
    @abstractmethod
    async def analyze_sentiment(self, text: str) -> Sentiment:
        """Analyze sentiment"""
        pass


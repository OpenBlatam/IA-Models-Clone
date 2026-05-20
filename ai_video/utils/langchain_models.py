from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import msgspec
from typing import List, Optional, Any

from typing import Any, List, Dict, Optional
import logging
import asyncio
class LangChainAnalysis(msgspec.Struct, frozen=True, slots=True):
    """
    Resultados de análisis de contenido generados por LangChain.
    """
    content_type: Optional[str] = None
    sentiment: Optional[str] = None
    engagement_score: Optional[float] = None
    viral_potential: Optional[float] = None
    trending_keywords: List[str] = msgspec.field(default_factory=list)
    optimal_duration: Optional[float] = None
    optimal_format: Optional[str] = None
    raw_response: Optional[Any] = None
    embedding: Optional[List[float]] = None
    def is_valid(self) -> bool:
        return self.content_type is not None and self.engagement_score is not None

class ContentOptimization(msgspec.Struct, frozen=True, slots=True):
    """
    Resultados de optimización de contenido generados por LangChain.
    """
    optimal_title: Optional[str] = None
    optimal_tags: List[str] = msgspec.field(default_factory=list)
    optimal_hashtags: List[str] = msgspec.field(default_factory=list)
    engagement_hooks: List[str] = msgspec.field(default_factory=list)
    viral_elements: List[str] = msgspec.field(default_factory=list)
    raw_response: Optional[Any] = None
    embedding: Optional[List[float]] = None
    def is_valid(self) -> bool:
        return self.optimal_title is not None and (self.optimal_tags or self.optimal_hashtags)

class ShortVideoOptimization(msgspec.Struct, frozen=True, slots=True):
    """
    Resultados de optimización para short-form generados por LangChain.
    """
    optimal_clip_length: Optional[float] = None
    hook_duration: Optional[float] = None
    retention_duration: Optional[float] = None
    call_to_action_duration: Optional[float] = None
    hook_type: Optional[str] = None
    engagement_triggers: List[str] = msgspec.field(default_factory=list)
    viral_hooks: List[str] = msgspec.field(default_factory=list)
    emotional_impact: Optional[float] = None
    raw_response: Optional[Any] = None
    embedding: Optional[List[float]] = None
    def is_valid(self) -> bool:
        return self.optimal_clip_length is not None and (self.engagement_triggers or self.viral_hooks) 
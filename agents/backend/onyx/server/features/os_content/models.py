from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Union, Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from typing import Any, List, Dict, Optional
import logging
import asyncio
class NLPRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze")
    lang: str = Field(default="es", pattern="^(es|en)$", description="Language code (es or en)")

class NLPResponse(BaseModel):
    entities: List[Dict[str, Any]]
    tokens: List[str]
    sentiment: Optional[Any] = None
    error: Optional[str] = None

class OSContentUGCVideoRequest(BaseModel):
    id: str
    user_id: str
    title: str
    description: Optional[str] = None
    text_prompt: str
    image_urls: List[str] = []
    video_urls: List[str] = []
    ugc_type: str = "ugc_video_ad"
    created_at: datetime
    metadata: Dict[str, Any] = {}
    language: str = "es"
    estimated_duration: Optional[float] = None
    nlp_analysis: Dict[str, Any] = Field(default_factory=dict, description="Resultados del análisis NLP del texto o guion")

class OSContentUGCVideoResponse(BaseModel):
    request_id: str
    video_url: str = ""
    status: str
    created_at: datetime
    details: Dict[str, Any] = {}
    progress: float = 0.0
    estimated_duration: Optional[float] = None
    nlp_analysis: Dict[str, Any] = Field(default_factory=dict, description="Resultados del análisis NLP del texto o guion")
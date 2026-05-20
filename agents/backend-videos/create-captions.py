from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph

from typing import Dict, List, Optional
from pydantic import BaseModel
from langchain.schema.messages import BaseMessage
from onyx.utils.logger import setup_logger
from onyx.utils.langchain import LangchainField

logger = setup_logger()

class VideoInput(BaseModel):
    """Input model for video processing following Onyx conventions"""
    youtube_url: str = LangchainField(description="URL of the YouTube video to process")
    target_duration: Optional[int] = LangchainField(
        description="Target duration for the short in seconds (default: 60)",
        default=60
    )
    style_preferences: Optional[Dict[str, str]] = LangchainField(
        description="Style preferences for the short (e.g. {'music_style': 'upbeat', 'transition_style': 'smooth'})",
        default={}
    )
    brand_kit: Optional[Dict[str, any]] = LangchainField(
        description="Brand kit information for consistent styling",
        default={}
    )

class VideoOutput(BaseModel):
    """Output model for processed video following Onyx conventions"""
    short_url: str = LangchainField(description="URL of the generated short")
    duration: int = LangchainField(description="Duration of the generated short in seconds")
    segments: List[Dict[str, any]] = LangchainField(
        description="List of video segments with timestamps and metadata"
    )
    captions: str = LangchainField(description="Generated captions for the short")
    metadata: Dict[str, any] = LangchainField(
        description="Additional metadata about the generated short"
    )

class VideoState(BaseModel):
    """State model for video processing workflow following Onyx conventions"""
    messages: List[BaseMessage] = []
    input: Optional[VideoInput] = None
    output: Optional[VideoOutput] = None
    error: Optional[str] = None
    current_step: str = "start"
    processing_status: Dict[str, any] = LangchainField(
        description="Current status of video processing",
        default={}
    )

    class Config:
        arbitrary_types_allowed = True


from typing import List, Optional
from pydantic import BaseModel, Field

class AdditionalContentRequest(BaseModel):
    """Request model for generating additional content."""
    text: str = Field(..., description="The main text content to analyze")
    platform: str = Field(..., description="The social media platform (e.g., 'instagram', 'twitter', 'linkedin')")
    content_type: str = Field(..., description="Type of content (e.g., 'post', 'article', 'tweet')")
    tone: str = Field(..., description="The desired tone of the content")
    max_hashtags: int = Field(default=5, description="Maximum number of hashtags to generate")
    include_cta: bool = Field(default=True, description="Whether to include a call to action")
    target_audience: Optional[str] = Field(None, description="Target audience for the content")

class Hashtag(BaseModel):
    """Model for a hashtag suggestion."""
    tag: str = Field(..., description="The hashtag text")
    relevance_score: float = Field(..., description="Relevance score of the hashtag")
    popularity_score: Optional[float] = Field(None, description="Popularity score of the hashtag")

class CallToAction(BaseModel):
    """Model for a call to action suggestion."""
    text: str = Field(..., description="The call to action text")
    type: str = Field(..., description="Type of CTA (e.g., 'engagement', 'click', 'share')")
    relevance_score: float = Field(..., description="Relevance score of the CTA")

class Link(BaseModel):
    """Model for a link suggestion."""
    text: str = Field(..., description="The link text")
    url: str = Field(..., description="The URL")
    relevance_score: float = Field(..., description="Relevance score of the link")

class AdditionalContentResponse(BaseModel):
    """Response model for generated additional content."""
    hashtags: List[Hashtag] = Field(default_factory=list, description="Generated hashtags")
    call_to_action: Optional[CallToAction] = Field(None, description="Generated call to action")
    suggested_links: List[Link] = Field(default_factory=list, description="Suggested links")
    full_text: str = Field(..., description="The complete text with all additions")
    metadata: dict = Field(default_factory=dict, description="Additional metadata about the generation")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details") 
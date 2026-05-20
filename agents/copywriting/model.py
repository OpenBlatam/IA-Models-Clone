# Very to onny just a propmpt dedicatefrom datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
class CopywritingRequest(BaseModel):
    message: str = Field(description="The original message or content to be transformed into ad copy")
    platform: str = Field(description="The target platform for the ad (e.g., 'facebook', 'youtube')")
    tone: str = Field(description="The desired tone of voice for the copy (e.g., 'professional', 'casual', 'urgent')")
    target_audience: str = Field(description="Brief description of the target audience")
    call_to_action: str = Field(description="The desired call to action for the ad")

class CopywritingResponse(BaseModel):
    ad_copy: str = Field(description="The generated ad copy optimized for the specified platform")
    platform_specific_tips: str = Field(description="Tips for optimizing the copy for the specific platform")
    suggested_hashtags: list[str] = Field(description="Suggested hashtags for social media platforms")
    character_count: int = Field(description="Character count of the generated copy")
    
    @field_validator('character_count')
    def validate_character_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Character count cannot be negative")
        return v

class CopywritingError(BaseModel):
    error: str = Field(description="Error message")
    details: str = Field(description="Detailed error information")




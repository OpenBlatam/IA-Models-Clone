from typing import Dict, Any
from pydantic import BaseModel

class CopywritingConfig(BaseModel):
    """Configuration for the copywriting service."""
    timeout: int = 10
    max_retries: int = 3
    default_tone: str = "professional"
    default_platform: str = "general"
    
    class Config:
        extra = "allow"

def get_copywriting_config() -> Dict[str, Any]:
    """Get the copywriting service configuration."""
    return CopywritingConfig().model_dump()

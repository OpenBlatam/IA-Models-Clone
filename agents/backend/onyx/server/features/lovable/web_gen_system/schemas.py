from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class AccessibilityIssue(BaseModel):
    issue_type: str
    description: str
    element: str
    severity: str = Field(..., description="critical, warning, or info")

class WebGenRequest(BaseModel):
    prompt: str
    style: str = "modern"
    optimize_seo: bool = True
    fix_accessibility: bool = True
    target: str = "html"
    use_agents: bool = False
    
    model_config = ConfigDict(extra="ignore")

class WebGenResponse(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    issues: List[AccessibilityIssue] = Field(default_factory=list)

class AgentContext(BaseModel):
    task_id: str
    shared_memory: Dict[str, Any] = Field(default_factory=dict)
    event_bus: Optional[Any] = None # Placeholder for EventBus
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

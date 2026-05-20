from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# Import the new pipeline
from ..web_gen_system.pipeline import WebGenPipeline

router = APIRouter(prefix="/ai", tags=["AI Features"])

# Initialize pipeline
# In production, this should be a singleton or dependency injected
pipeline = WebGenPipeline()

# --- Request Models ---

class SystemGenerationRequest(BaseModel):
    prompt: str
    target: str = "html" # html, nextjs, expo
    use_agents: bool = True

class AccessibilityRequest(BaseModel):
    html_snippet: str

class ImageContextRequest(BaseModel):
    context: str

class SEORequest(BaseModel):
    content: str

class UIRequest(BaseModel):
    description: str
    style_guide: Optional[str] = "Tailwind CSS"

# --- Endpoints ---

@router.post("/system/generate")
async def generate_system(request: SystemGenerationRequest) -> Dict[str, Any]:
    """
    Generate a full system (Web or Mobile) using the Agentic Pipeline.
    Triggers the Dynamic System Orchestration.
    """
    try:
        result = pipeline.run(
            prompt=request.prompt,
            target=request.target,
            use_agents=request.use_agents
        )
        
        # If result is string (HTML), wrap it
        if isinstance(result, str):
            return {"type": "html", "content": result}
        
        # If result is dict (Project Structure), return as is
        return {"type": "project", "structure": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/accessibility/analyze")
async def analyze_accessibility(request: AccessibilityRequest) -> Dict[str, Any]:
    """Analyze HTML for accessibility issues."""
    # Use pipeline's accessibility component
    return {"issues": pipeline.accessibility.analyze(request.html_snippet)}

@router.post("/seo/meta-tags")
async def generate_meta_tags(request: SEORequest) -> Dict[str, str]:
    """Generate SEO meta tags."""
    # Use pipeline's SEO component
    tags = pipeline.seo.generate_tags(request.content)
    return {"meta_tags": tags}

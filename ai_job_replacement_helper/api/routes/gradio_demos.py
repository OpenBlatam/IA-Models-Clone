"""
Gradio Demos endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.gradio_demos import GradioDemosService, GradioDemoConfig

router = APIRouter()
gradio_service = GradioDemosService()


@router.post("/create-llm-demo")
async def create_llm_demo(
    demo_id: str,
    title: str = "LLM Text Generation",
    description: str = "Generate text using a Large Language Model"
) -> Dict[str, Any]:
    """Crear demo de LLM"""
    try:
        # In production, this would use actual generation function
        def dummy_generate(prompt: str, max_tokens: int, temperature: float) -> str:
            return f"[Generated text for: {prompt[:50]}...]"
        
        config = GradioDemoConfig(title=title, description=description)
        demo = gradio_service.create_llm_demo(demo_id, dummy_generate, config)
        
        return {
            "demo_id": demo_id,
            "status": "created",
            "title": title,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list-demos")
async def list_demos() -> Dict[str, Any]:
    """Listar demos disponibles"""
    try:
        demos = gradio_service.list_demos()
        return {
            "demos": demos,
            "count": len(demos),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





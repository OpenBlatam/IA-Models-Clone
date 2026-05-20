"""
ML API Routes
=============

Endpoints para funcionalidades de Machine Learning.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
from pydantic import BaseModel

router = APIRouter(prefix="/ml", tags=["ml"])


class AnalyzeRequest(BaseModel):
    content: str
    platform: str


class GenerateTextRequest(BaseModel):
    topic: str
    platform: str
    tone: Optional[str] = "professional"
    length: Optional[str] = "medium"


class GenerateImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = 512
    height: Optional[int] = 512


def get_content_analyzer():
    """Dependency para ContentAnalyzer"""
    from ...ml.content_analyzer import ContentAnalyzer
    return ContentAnalyzer()


def get_text_generator():
    """Dependency para AdvancedTextGenerator"""
    from ...ml.text_generator import AdvancedTextGenerator
    return AdvancedTextGenerator()


def get_image_generator():
    """Dependency para ImageGenerator"""
    from ...ml.image_generator import ImageGenerator
    return ImageGenerator()


@router.post("/analyze", response_model=dict)
async def analyze_content(
    request: AnalyzeRequest,
    analyzer = Depends(get_content_analyzer)
):
    """Analizar contenido con ML"""
    try:
        analysis = analyzer.analyze_content_quality(
            request.content,
            request.platform
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-text", response_model=dict)
async def generate_text(
    request: GenerateTextRequest,
    generator = Depends(get_text_generator)
):
    """Generar texto con IA"""
    try:
        generated = generator.generate_post(
            topic=request.topic,
            platform=request.platform,
            tone=request.tone,
            length=request.length
        )
        return {"generated_text": generated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-image", response_model=dict)
async def generate_image(
    request: GenerateImageRequest,
    generator = Depends(get_image_generator)
):
    """Generar imagen con diffusion model"""
    try:
        import base64
        from io import BytesIO
        
        image = generator.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height
        )
        
        if image:
            # Convertir a base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return {
                "image": img_str,
                "format": "PNG",
                "width": request.width,
                "height": request.height
            }
        else:
            raise HTTPException(status_code=500, detail="Error generando imagen")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment", response_model=dict)
async def analyze_sentiment(
    text: str = Query(..., description="Texto a analizar"),
    analyzer = Depends(get_content_analyzer)
):
    """Analizar sentimiento del texto"""
    try:
        sentiment = analyzer.analyze_sentiment(text)
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





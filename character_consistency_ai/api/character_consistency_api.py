"""
Character Consistency API
==========================

FastAPI endpoints for character consistency operations.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import logging

from ..core.character_consistency_service import CharacterConsistencyService
from ..config.character_consistency_config import CharacterConsistencyConfig
from .handlers.embedding_handlers import EmbeddingHandlers
from .handlers.workflow_handlers import WorkflowHandlers
from .handlers.model_handlers import ModelHandlers

logger = logging.getLogger(__name__)

# Initialize service
config = CharacterConsistencyConfig.from_env()
service = CharacterConsistencyService(config=config)

# Initialize handlers
embedding_handlers = EmbeddingHandlers(service)
workflow_handlers = WorkflowHandlers(service)
model_handlers = ModelHandlers(service)

# Create FastAPI app
app = FastAPI(
    title="Character Consistency AI API",
    description="API for generating character consistency safe tensors using Flux2",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create router
router = APIRouter(prefix="/api/v1", tags=["character-consistency"])


@router.post("/generate", response_model=Dict[str, Any])
async def generate_character_embedding(
    images: List[UploadFile] = File(..., description="One or more character images"),
    character_name: Optional[str] = Form(None, description="Character name"),
    save_tensor: bool = Form(True, description="Save as safe tensor"),
    metadata: Optional[str] = Form(None, description="JSON metadata"),
):
    """
    Generate character consistency embedding from uploaded images.
    
    Args:
        images: One or more character images
        character_name: Optional character name
        save_tensor: Whether to save as safe tensor
        metadata: Optional JSON metadata string
        
    Returns:
        Character embedding information
    """
    return await embedding_handlers.generate_embedding(
        images=images,
        character_name=character_name,
        save_tensor=save_tensor,
        metadata=metadata,
    )


@router.post("/workflow", response_model=Dict[str, Any])
async def create_workflow_tensor(
    embedding_path: str = Form(..., description="Path to character embedding safe tensor"),
    prompt_template: str = Form(..., description="Prompt template with {character} placeholder"),
    negative_prompt: Optional[str] = Form(None, description="Negative prompt"),
    num_inference_steps: int = Form(50, description="Number of inference steps"),
    guidance_scale: float = Form(7.5, description="Guidance scale"),
):
    """
    Create workflow-ready safe tensor from existing embedding.
    
    Args:
        embedding_path: Path to character embedding
        prompt_template: Prompt template
        negative_prompt: Negative prompt
        num_inference_steps: Inference steps
        guidance_scale: Guidance scale
        
    Returns:
        Workflow tensor information
    """
    return await workflow_handlers.create_workflow_tensor(
        embedding_path=embedding_path,
        prompt_template=prompt_template,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )


@router.get("/embeddings", response_model=List[Dict[str, Any]])
async def list_embeddings():
    """
    List all generated character embeddings.
    
    Returns:
        List of embedding information
    """
    return await embedding_handlers.list_embeddings()


@router.get("/embedding/{embedding_id}")
async def get_embedding(embedding_id: str):
    """
    Download a specific embedding safe tensor.
    
    Args:
        embedding_id: Embedding filename or ID
        
    Returns:
        Safe tensor file
    """
    return await embedding_handlers.get_embedding(embedding_id)


@router.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """
    Get model information.
    
    Returns:
        Model information
    """
    return await model_handlers.get_model_info()


@router.post("/initialize")
async def initialize_model():
    """
    Initialize the Flux2 model.
    
    Returns:
        Initialization status
    """
    return await model_handlers.initialize_model()


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return await model_handlers.health_check()


# Include router
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Character Consistency AI",
        "version": "1.0.0",
        "status": "running",
    }



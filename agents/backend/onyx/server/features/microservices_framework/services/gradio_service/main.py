"""
Gradio Service - Interactive Model Interfaces
Provides Gradio-based web interfaces for model inference and experimentation.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, List
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import gradio as gr
from gradio_client import Client
import structlog
import httpx

logger = structlog.get_logger()

# Service URLs (configure via environment)
LLM_SERVICE_URL = "http://localhost:8001"
DIFFUSION_SERVICE_URL = "http://localhost:8002"


class GradioInterfaceRequest(BaseModel):
    """Request model for creating Gradio interface."""
    interface_type: str = Field(..., description="Type: text_generation, image_generation, embeddings")
    model_name: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    port: int = Field(default=7860, ge=1024, le=65535)


def create_text_generation_interface(model_name: str = "gpt2"):
    """Create text generation Gradio interface."""
    
    async def generate_text(
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
    ):
        """Generate text using LLM service."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LLM_SERVICE_URL}/generate",
                json={
                    "prompt": prompt,
                    "model_name": model_name,
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                },
                timeout=120.0,
            )
            if response.status_code == 200:
                result = response.json()
                return result["generated_text"]
            else:
                return f"Error: {response.text}"
    
    interface = gr.Interface(
        fn=generate_text,
        inputs=[
            gr.Textbox(
                label="Prompt",
                placeholder="Enter your text prompt here...",
                lines=5,
            ),
            gr.Slider(
                minimum=10,
                maximum=512,
                value=100,
                step=10,
                label="Max Length",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Temperature",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top P",
            ),
            gr.Slider(
                minimum=1,
                maximum=100,
                value=50,
                step=1,
                label="Top K",
            ),
        ],
        outputs=gr.Textbox(label="Generated Text", lines=10),
        title=f"Text Generation - {model_name}",
        description="Generate text using transformer language models",
        examples=[
            ["The future of artificial intelligence"],
            ["Once upon a time"],
            ["In a world where"],
        ],
    )
    
    return interface


def create_image_generation_interface(model_name: str = "runwayml/stable-diffusion-v1-5"):
    """Create image generation Gradio interface."""
    
    async def generate_image(
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
    ):
        """Generate image using Diffusion service."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DIFFUSION_SERVICE_URL}/text-to-image",
                json={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt if negative_prompt else None,
                    "model_name": model_name,
                    "num_inference_steps": num_steps,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                    "seed": seed,
                },
                timeout=300.0,
            )
            if response.status_code == 200:
                # Return image bytes
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(response.content))
                return img
            else:
                return None
    
    interface = gr.Interface(
        fn=generate_image,
        inputs=[
            gr.Textbox(
                label="Prompt",
                placeholder="A beautiful landscape with mountains and lakes",
                lines=3,
            ),
            gr.Textbox(
                label="Negative Prompt (optional)",
                placeholder="blurry, low quality",
                lines=2,
            ),
            gr.Slider(
                minimum=10,
                maximum=100,
                value=50,
                step=5,
                label="Inference Steps",
            ),
            gr.Slider(
                minimum=1.0,
                maximum=20.0,
                value=7.5,
                step=0.5,
                label="Guidance Scale",
            ),
            gr.Slider(
                minimum=256,
                maximum=1024,
                value=512,
                step=64,
                label="Width",
            ),
            gr.Slider(
                minimum=256,
                maximum=1024,
                value=512,
                step=64,
                label="Height",
            ),
            gr.Number(
                label="Seed (optional)",
                value=None,
            ),
        ],
        outputs=gr.Image(label="Generated Image"),
        title=f"Image Generation - {model_name}",
        description="Generate images from text prompts using Stable Diffusion",
        examples=[
            ["A futuristic city at sunset, cyberpunk style"],
            ["A serene Japanese garden with cherry blossoms"],
            ["An astronaut riding a horse on Mars"],
        ],
    )
    
    return interface


def create_embeddings_interface(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Create embeddings Gradio interface."""
    
    async def get_embeddings(texts: str):
        """Get embeddings using LLM service."""
        text_list = [t.strip() for t in texts.split("\n") if t.strip()]
        if not text_list:
            return "Please enter at least one text."
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LLM_SERVICE_URL}/embeddings",
                json={
                    "texts": text_list,
                    "model_name": model_name,
                    "normalize": True,
                },
                timeout=60.0,
            )
            if response.status_code == 200:
                result = response.json()
                return f"Generated {len(result['embeddings'])} embeddings of dimension {result['dimension']}"
            else:
                return f"Error: {response.text}"
    
    interface = gr.Interface(
        fn=get_embeddings,
        inputs=gr.Textbox(
            label="Texts (one per line)",
            placeholder="Enter texts, one per line...",
            lines=10,
        ),
        outputs=gr.Textbox(label="Result", lines=5),
        title=f"Text Embeddings - {model_name}",
        description="Generate embeddings for input texts",
    )
    
    return interface


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager."""
    logger.info("gradio_service_starting")
    yield
    logger.info("gradio_service_shutting_down")


app = FastAPI(
    title="Gradio Service",
    description="Interactive Model Interfaces Service",
    version="1.0.0",
    lifespan=lifespan,
)

# Store running interfaces
_running_interfaces: dict = {}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "gradio_service",
        "active_interfaces": len(_running_interfaces),
    }


@app.post("/interfaces/text-generation")
async def create_text_generation_interface_endpoint(
    model_name: str = "gpt2",
    port: int = 7860,
    share: bool = False,
):
    """Create and launch text generation interface."""
    try:
        interface = create_text_generation_interface(model_name)
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            inbrowser=False,
        )
        
        interface_id = f"text_gen_{model_name}_{port}"
        _running_interfaces[interface_id] = {
            "type": "text_generation",
            "model_name": model_name,
            "port": port,
            "url": f"http://localhost:{port}",
        }
        
        return {
            "interface_id": interface_id,
            "url": f"http://localhost:{port}",
            "status": "running",
        }
    except Exception as e:
        logger.error("interface_creation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create interface: {str(e)}")


@app.post("/interfaces/image-generation")
async def create_image_generation_interface_endpoint(
    model_name: str = "runwayml/stable-diffusion-v1-5",
    port: int = 7861,
    share: bool = False,
):
    """Create and launch image generation interface."""
    try:
        interface = create_image_generation_interface(model_name)
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            inbrowser=False,
        )
        
        interface_id = f"img_gen_{model_name}_{port}"
        _running_interfaces[interface_id] = {
            "type": "image_generation",
            "model_name": model_name,
            "port": port,
            "url": f"http://localhost:{port}",
        }
        
        return {
            "interface_id": interface_id,
            "url": f"http://localhost:{port}",
            "status": "running",
        }
    except Exception as e:
        logger.error("interface_creation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create interface: {str(e)}")


@app.post("/interfaces/embeddings")
async def create_embeddings_interface_endpoint(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    port: int = 7862,
    share: bool = False,
):
    """Create and launch embeddings interface."""
    try:
        interface = create_embeddings_interface(model_name)
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            inbrowser=False,
        )
        
        interface_id = f"embeddings_{model_name}_{port}"
        _running_interfaces[interface_id] = {
            "type": "embeddings",
            "model_name": model_name,
            "port": port,
            "url": f"http://localhost:{port}",
        }
        
        return {
            "interface_id": interface_id,
            "url": f"http://localhost:{port}",
            "status": "running",
        }
    except Exception as e:
        logger.error("interface_creation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create interface: {str(e)}")


@app.get("/interfaces")
async def list_interfaces():
    """List all running interfaces."""
    return {
        "interfaces": list(_running_interfaces.values()),
        "count": len(_running_interfaces),
    }


@app.delete("/interfaces/{interface_id}")
async def close_interface(interface_id: str):
    """Close a running interface."""
    if interface_id not in _running_interfaces:
        raise HTTPException(status_code=404, detail="Interface not found")
    
    # In production, implement proper interface closing
    del _running_interfaces[interface_id]
    return {"status": "closed", "interface_id": interface_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        log_level="info",
    )




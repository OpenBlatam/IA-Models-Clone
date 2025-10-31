"""
Advanced Diffusion Models for Document Generation
===============================================

State-of-the-art diffusion models for document generation, including
text-to-image, image-to-text, and multimodal document creation.

Features:
- Stable Diffusion integration
- Custom diffusion pipelines
- Text-to-image generation
- Image-to-text conversion
- Multimodal document generation
- ControlNet integration
- Inpainting and outpainting
- Style transfer and conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    StableDiffusionPipeline, StableDiffusionXLPipeline,
    ControlNetModel, StableDiffusionControlNetPipeline,
    StableDiffusionInpaintPipeline, StableDiffusionUpscalePipeline,
    DDIMScheduler, DDPMScheduler, PNDMScheduler,
    UNet2DConditionModel, AutoencoderKL, CLIPTextModel, CLIPTokenizer
)
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
import math
import warnings
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision.utils import save_image
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiffusionConfig:
    """Configuration for diffusion models"""
    model_id: str = "runwayml/stable-diffusion-v1-5"
    scheduler_type: str = "ddim"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    enable_memory_efficient_attention: bool = True
    enable_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False

class DocumentDiffusionPipeline:
    """Custom diffusion pipeline for document generation"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load models
        self._load_models()
        
        # Setup scheduler
        self._setup_scheduler()
        
        # Setup memory optimizations
        self._setup_memory_optimizations()
    
    def _load_models(self):
        """Load diffusion models"""
        logger.info(f"Loading models from {self.config.model_id}")
        
        # Load pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.config.dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        # Enable attention slicing for memory efficiency
        if self.config.enable_memory_efficient_attention:
            self.pipeline.enable_attention_slicing()
        
        # Enable CPU offload
        if self.config.enable_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        elif self.config.enable_sequential_cpu_offload:
            self.pipeline.enable_sequential_cpu_offload()
    
    def _setup_scheduler(self):
        """Setup diffusion scheduler"""
        if self.config.scheduler_type == "ddim":
            self.pipeline.scheduler = DDIMScheduler.from_config(
                self.pipeline.scheduler.config
            )
        elif self.config.scheduler_type == "ddpm":
            self.pipeline.scheduler = DDPMScheduler.from_config(
                self.pipeline.scheduler.config
            )
        elif self.config.scheduler_type == "pndm":
            self.pipeline.scheduler = PNDMScheduler.from_config(
                self.pipeline.scheduler.config
            )
    
    def _setup_memory_optimizations(self):
        """Setup memory optimizations"""
        # Enable xformers if available
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
    
    def generate_document_image(self, prompt: str, negative_prompt: str = "",
                              num_images: int = 1, seed: Optional[int] = None) -> List[Image.Image]:
        """Generate document images from text prompt"""
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate images
        with autocast():
            images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                height=self.config.height,
                width=self.config.width
            ).images
        
        return images
    
    def generate_document_layout(self, document_type: str, content: str,
                               style: str = "professional") -> Image.Image:
        """Generate document layout based on type and content"""
        # Create specialized prompts for different document types
        layout_prompts = {
            "contract": f"A professional legal contract document layout with {content[:100]}... text, clean typography, formal design",
            "report": f"A business report document layout with {content[:100]}... content, charts, professional formatting",
            "presentation": f"A presentation slide layout with {content[:100]}... text, modern design, clear hierarchy",
            "letter": f"A formal business letter layout with {content[:100]}... text, letterhead, professional styling",
            "invoice": f"An invoice document layout with {content[:100]}... details, table format, business design"
        }
        
        prompt = layout_prompts.get(document_type, f"A {document_type} document layout with {content[:100]}...")
        
        if style == "modern":
            prompt += ", modern design, clean lines, contemporary styling"
        elif style == "classic":
            prompt += ", classic design, traditional formatting, elegant typography"
        elif style == "minimalist":
            prompt += ", minimalist design, simple layout, clean white space"
        
        negative_prompt = "blurry, low quality, distorted, unprofessional, messy, cluttered"
        
        images = self.generate_document_image(prompt, negative_prompt, num_images=1)
        return images[0] if images else None

class ControlNetDocumentGenerator:
    """ControlNet-based document generator for precise control"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=self.config.dtype
        )
        
        # Load pipeline with ControlNet
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.model_id,
            controlnet=self.controlnet,
            torch_dtype=self.config.dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.pipeline = self.pipeline.to(self.device)
    
    def generate_from_sketch(self, sketch_image: Image.Image, prompt: str,
                           negative_prompt: str = "") -> Image.Image:
        """Generate document from sketch using ControlNet"""
        # Convert sketch to control image
        control_image = self._prepare_control_image(sketch_image)
        
        # Generate image
        with autocast():
            image = self.pipeline(
                prompt=prompt,
                image=control_image,
                negative_prompt=negative_prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                controlnet_conditioning_scale=1.0
            ).images[0]
        
        return image
    
    def _prepare_control_image(self, image: Image.Image) -> Image.Image:
        """Prepare control image for ControlNet"""
        # Convert to grayscale
        image = image.convert("L")
        
        # Apply Canny edge detection
        image_np = np.array(image)
        edges = cv2.Canny(image_np, 100, 200)
        edges_image = Image.fromarray(edges)
        
        return edges_image

class DocumentInpaintingPipeline:
    """Document inpainting for editing and completion"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load inpainting pipeline
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=self.config.dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.pipeline = self.pipeline.to(self.device)
    
    def inpaint_document(self, image: Image.Image, mask: Image.Image,
                        prompt: str, negative_prompt: str = "") -> Image.Image:
        """Inpaint document regions"""
        with autocast():
            result = self.pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                negative_prompt=negative_prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale
            ).images[0]
        
        return result
    
    def remove_text(self, image: Image.Image, text_regions: List[Tuple[int, int, int, int]]) -> Image.Image:
        """Remove text from document regions"""
        # Create mask for text regions
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        
        for x, y, w, h in text_regions:
            mask_draw.rectangle([x, y, x+w, y+h], fill=255)
        
        # Inpaint with background
        prompt = "clean document background, no text, professional document"
        negative_prompt = "text, writing, letters, words, content"
        
        return self.inpaint_document(image, mask, prompt, negative_prompt)
    
    def add_text(self, image: Image.Image, text_regions: List[Tuple[int, int, int, int]],
                text_content: List[str]) -> Image.Image:
        """Add text to document regions"""
        # Create mask for text regions
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        
        for x, y, w, h in text_regions:
            mask_draw.rectangle([x, y, x+w, y+h], fill=255)
        
        # Create prompt with text content
        prompt = f"professional document text: {', '.join(text_content)}"
        negative_prompt = "blurry, low quality, distorted text"
        
        return self.inpaint_document(image, mask, prompt, negative_prompt)

class MultimodalDocumentGenerator:
    """Multimodal document generator combining text and images"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load CLIP models for text-image understanding
        self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load diffusion pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.config.dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.pipeline = self.pipeline.to(self.device)
    
    def generate_document_with_images(self, text_content: str, image_descriptions: List[str],
                                    document_type: str = "report") -> Dict[str, Any]:
        """Generate document with embedded images"""
        # Generate images for descriptions
        generated_images = []
        for description in image_descriptions:
            prompt = f"{description}, professional document image, high quality, clear"
            negative_prompt = "blurry, low quality, distorted, unprofessional"
            
            with autocast():
                image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale
                ).images[0]
            
            generated_images.append(image)
        
        # Generate document layout
        layout_prompt = f"A {document_type} document layout with text content and embedded images, professional design"
        layout_image = self.pipeline(
            prompt=layout_prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale
        ).images[0]
        
        return {
            "layout": layout_image,
            "images": generated_images,
            "text_content": text_content,
            "document_type": document_type
        }
    
    def create_document_mockup(self, content: str, document_type: str,
                             style: str = "modern") -> Image.Image:
        """Create document mockup with content"""
        # Create detailed prompt based on document type and style
        style_prompts = {
            "modern": "modern design, clean lines, contemporary styling, minimalist layout",
            "classic": "classic design, traditional formatting, elegant typography, formal layout",
            "creative": "creative design, artistic layout, unique styling, innovative formatting",
            "technical": "technical document, structured layout, clear hierarchy, professional formatting"
        }
        
        prompt = f"A {document_type} document mockup with {content[:200]}... content, {style_prompts.get(style, 'professional design')}, high quality, clear text"
        negative_prompt = "blurry, low quality, distorted, unreadable text, messy layout"
        
        with autocast():
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                height=768,  # Higher resolution for documents
                width=1024
            ).images[0]
        
        return image

class DocumentStyleTransfer:
    """Document style transfer using diffusion models"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.config.dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.pipeline = self.pipeline.to(self.device)
    
    def transfer_style(self, content: str, source_style: str, target_style: str) -> Image.Image:
        """Transfer document style from source to target"""
        # Define style prompts
        style_prompts = {
            "formal": "formal business document, professional typography, clean layout, corporate design",
            "casual": "casual document, friendly typography, relaxed layout, informal design",
            "academic": "academic document, scholarly typography, structured layout, research paper style",
            "creative": "creative document, artistic typography, unique layout, innovative design",
            "minimalist": "minimalist document, simple typography, clean white space, modern design",
            "vintage": "vintage document, classic typography, traditional layout, retro design"
        }
        
        source_prompt = style_prompts.get(source_style, "professional document")
        target_prompt = style_prompts.get(target_style, "professional document")
        
        # Create transfer prompt
        prompt = f"Transform {source_prompt} to {target_prompt} with content: {content[:200]}..."
        negative_prompt = "blurry, low quality, distorted, inconsistent style"
        
        with autocast():
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale
            ).images[0]
        
        return image

class DocumentUpscaler:
    """Document upscaling using diffusion models"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load upscaling pipeline
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=self.config.dtype
        )
        
        self.pipeline = self.pipeline.to(self.device)
    
    def upscale_document(self, image: Image.Image, prompt: str = "high quality document") -> Image.Image:
        """Upscale document image"""
        with autocast():
            upscaled_image = self.pipeline(
                prompt=prompt,
                image=image,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale
            ).images[0]
        
        return upscaled_image

class DiffusionModelFactory:
    """Factory for creating different diffusion models"""
    
    @staticmethod
    def create_pipeline(pipeline_type: str, config: DiffusionConfig):
        """Create diffusion pipeline based on type"""
        if pipeline_type == "document":
            return DocumentDiffusionPipeline(config)
        elif pipeline_type == "controlnet":
            return ControlNetDocumentGenerator(config)
        elif pipeline_type == "inpainting":
            return DocumentInpaintingPipeline(config)
        elif pipeline_type == "multimodal":
            return MultimodalDocumentGenerator(config)
        elif pipeline_type == "style_transfer":
            return DocumentStyleTransfer(config)
        elif pipeline_type == "upscaler":
            return DocumentUpscaler(config)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available diffusion models"""
        return [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-inpainting",
            "stabilityai/stable-diffusion-x4-upscaler"
        ]

# Example usage
if __name__ == "__main__":
    # Configuration
    config = DiffusionConfig(
        model_id="runwayml/stable-diffusion-v1-5",
        num_inference_steps=20,
        guidance_scale=7.5,
        height=512,
        width=512
    )
    
    # Create document generator
    doc_generator = DiffusionModelFactory.create_pipeline("document", config)
    
    # Generate document image
    prompt = "A professional business contract document layout, clean typography, formal design, high quality"
    negative_prompt = "blurry, low quality, distorted, unprofessional"
    
    images = doc_generator.generate_document_image(prompt, negative_prompt, num_images=1)
    
    if images:
        # Save generated image
        images[0].save("generated_document.png")
        print("Document image generated and saved as 'generated_document.png'")
    
    # Create multimodal generator
    multimodal_gen = DiffusionModelFactory.create_pipeline("multimodal", config)
    
    # Generate document with images
    text_content = "This is a sample business report with key findings and recommendations."
    image_descriptions = ["chart showing sales growth", "pie chart of market share"]
    
    result = multimodal_gen.generate_document_with_images(
        text_content, image_descriptions, "report"
    )
    
    print(f"Generated document with {len(result['images'])} images")

























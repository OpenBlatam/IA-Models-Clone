"""
Content Generator using Diffusion Models and Transformers
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Union
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import (
        GPT2LMHeadModel, GPT2Tokenizer,
        T5ForConditionalGeneration, T5Tokenizer,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class TextGenerator(nn.Module):
    """Text generation using transformers"""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[torch.device] = None,
        use_gpu: bool = True
    ):
        """
        Initialize text generator
        
        Args:
            model_name: Model name (gpt2, gpt2-medium, etc.)
            device: PyTorch device
            use_gpu: Whether to use GPU
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        self.device = device or torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        
        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"TextGenerator initialized with {model_name} on {self.device}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated texts
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_texts
    
    def complete(self, text: str, max_new_tokens: int = 50) -> str:
        """
        Complete text
        
        Args:
            text: Partial text
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Completed text
        """
        generated = self.generate(text, max_length=len(text.split()) + max_new_tokens)
        return generated[0] if generated else text


class T5ContentGenerator(nn.Module):
    """T5-based content generation for better quality"""
    
    def __init__(
        self,
        model_name: str = "t5-small",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        logger.info(f"T5ContentGenerator initialized on {self.device}")
    
    def generate(
        self,
        task: str,
        context: str,
        max_length: int = 100
    ) -> str:
        """
        Generate content based on task and context
        
        Args:
            task: Task prefix (e.g., "summarize:", "expand:", "complete:")
            context: Input context
            max_length: Maximum generation length
            
        Returns:
            Generated content
        """
        input_text = f"{task} {context}"
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated
    
    def summarize(self, text: str) -> str:
        """Summarize text"""
        return self.generate("summarize:", text)
    
    def expand(self, text: str) -> str:
        """Expand text"""
        return self.generate("expand:", text)
    
    def complete_sentence(self, text: str) -> str:
        """Complete sentence"""
        return self.generate("complete:", text)


class DiffusionContentGenerator:
    """Content generation using diffusion models (for images)"""
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[torch.device] = None
    ):
        """
        Initialize diffusion generator
        
        Args:
            model_name: Diffusion model name
            device: PyTorch device
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library is required")
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            self.pipeline = self.pipeline.to(self.device)
            logger.info(f"DiffusionContentGenerator initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load diffusion model: {e}")
            self.pipeline = None
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> Optional[torch.Tensor]:
        """
        Generate image from text prompt
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale
            
        Returns:
            Generated image tensor
        """
        if self.pipeline is None:
            return None
        
        try:
            with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu"):
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
            
            return result.images[0] if result.images else None
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None


def create_text_generator(
    model_name: str = "gpt2",
    device: Optional[torch.device] = None
) -> TextGenerator:
    """Factory function to create text generator"""
    return TextGenerator(model_name=model_name, device=device)


def create_t5_generator(
    model_name: str = "t5-small",
    device: Optional[torch.device] = None
) -> T5ContentGenerator:
    """Factory function to create T5 generator"""
    return T5ContentGenerator(model_name=model_name, device=device)


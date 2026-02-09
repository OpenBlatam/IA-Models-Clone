"""
Text Generation Service using transformers

Uses pre-trained language models for text generation,
embellishment, and completion.
"""

import logging
from typing import Dict, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

from ...config import settings

logger = logging.getLogger(__name__)


class TextGenerationService:
    """Service for text generation using LLMs"""
    
    def __init__(self):
        self.device = settings.device if settings.ai_enabled and torch.cuda.is_available() and settings.use_gpu else "cpu"
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        if settings.text_generation_enabled and settings.ai_enabled:
            self._load_model()
    
    def _load_model(self) -> None:
        """Lazy load the text generation model"""
        try:
            logger.info(f"Loading text generation model: {settings.text_generation_model} on {self.device}")
            
            device_map = "auto" if self.device == "cuda" and torch.cuda.is_available() else None
            device_id = 0 if self.device == "cuda" and torch.cuda.is_available() else -1
            
            self.pipeline = pipeline(
                "text-generation",
                model=settings.text_generation_model,
                device=device_id,
                torch_dtype=torch.float16 if settings.use_mixed_precision and self.device == "cuda" else torch.float32
            )
            
            logger.info("Text generation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading text generation model: {e}", exc_info=True)
            # Fallback to smaller model
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model="gpt2",
                    device=-1
                )
                logger.warning("Using fallback text generation model (GPT-2)")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                self.pipeline = None
    
    def generate_text(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Dict with generated text and metadata
        """
        if not self.pipeline:
            if not settings.text_generation_enabled:
                return {
                    "generated_text": prompt,
                    "enabled": False,
                    "error": "Text generation disabled"
                }
            raise RuntimeError("Text generation model not loaded")
        
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            max_length = max_length or settings.max_generation_length
            
            # Truncate prompt if too long
            max_prompt_length = 512
            prompt_to_use = prompt[:max_prompt_length] if len(prompt) > max_prompt_length else prompt
            
            with torch.no_grad():
                if settings.use_mixed_precision and self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        results = self.pipeline(
                            prompt_to_use,
                            max_length=max_length,
                            num_return_sequences=num_return_sequences,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample,
                            pad_token_id=self.pipeline.tokenizer.eos_token_id if hasattr(self.pipeline, 'tokenizer') else None
                        )
                else:
                    results = self.pipeline(
                        prompt_to_use,
                        max_length=max_length,
                        num_return_sequences=num_return_sequences,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        pad_token_id=self.pipeline.tokenizer.eos_token_id if hasattr(self.pipeline, 'tokenizer') else None
                    )
            
            # Extract generated text
            if isinstance(results, list) and len(results) > 0:
                generated_texts = []
                for result in results:
                    if isinstance(result, dict):
                        text = result.get('generated_text', '')
                    else:
                        text = str(result)
                    # Remove the prompt from generated text
                    if text.startswith(prompt_to_use):
                        text = text[len(prompt_to_use):].strip()
                    generated_texts.append(text)
                
                return {
                    "generated_text": generated_texts[0] if len(generated_texts) == 1 else generated_texts,
                    "all_sequences": generated_texts,
                    "prompt": prompt_to_use,
                    "model": settings.text_generation_model
                }
            else:
                return {
                    "generated_text": prompt_to_use,
                    "error": "No text generated"
                }
                
        except Exception as e:
            logger.error(f"Error generating text: {e}", exc_info=True)
            return {
                "generated_text": prompt,
                "error": str(e)
            }
    
    def enhance_description(self, description: str) -> str:
        """
        Enhance/embellish a chat description
        
        Args:
            description: Original description
            
        Returns:
            Enhanced description
        """
        if not description:
            return description
        
        prompt = f"Improve and enhance the following description while keeping the same meaning:\n\n{description}\n\nEnhanced description:"
        
        result = self.generate_text(
            prompt,
            max_length=len(description) + 100,
            temperature=0.8,
            num_return_sequences=1
        )
        
        enhanced = result.get("generated_text", description)
        return enhanced.strip()
    
    def generate_tags(self, title: str, description: Optional[str] = None) -> list[str]:
        """
        Generate relevant tags for a chat based on title and description
        
        Args:
            title: Chat title
            description: Optional description
            
        Returns:
            List of suggested tags
        """
        text = title
        if description:
            text += f" {description}"
        
        prompt = f"Generate 5-10 relevant tags (comma-separated) for the following content:\n\n{text}\n\nTags:"
        
        result = self.generate_text(
            prompt,
            max_length=100,
            temperature=0.5,
            num_return_sequences=1
        )
        
        generated = result.get("generated_text", "")
        # Extract tags from generated text
        tags = [tag.strip().lower() for tag in generated.split(",") if tag.strip()]
        # Limit to 10 tags
        return tags[:10]
    
    def complete_text(self, partial_text: str, max_additional_length: int = 100) -> str:
        """
        Complete partial text
        
        Args:
            partial_text: Partial text to complete
            max_additional_length: Maximum additional characters to generate
            
        Returns:
            Completed text
        """
        result = self.generate_text(
            partial_text,
            max_length=len(partial_text) + max_additional_length,
            temperature=0.7,
            num_return_sequences=1
        )
        
        return result.get("generated_text", partial_text)
















"""
Text Generator using Transformers
==================================

Text generation using pre-trained transformer models.
"""

import torch
import logging
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    GenerationConfig
)

logger = logging.getLogger(__name__)


class TextGenerator:
    """
    Text generator using transformer models.
    
    Supports:
    - Causal LM (GPT-style)
    - Seq2Seq (T5-style)
    - Custom generation configs
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        model_type: str = "causal",
        device: Optional[str] = None,
        use_pipeline: bool = True
    ):
        """
        Initialize text generator.
        
        Args:
            model_name: HuggingFace model name
            model_type: "causal" or "seq2seq"
            device: Device ("cpu", "cuda", or None for auto)
            use_pipeline: Whether to use pipeline API
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_pipeline = use_pipeline
        
        logger.info(f"Loading model: {model_name} on {self.device}")
        
        if use_pipeline:
            self._load_pipeline()
        else:
            self._load_model()
    
    def _load_pipeline(self):
        """Load model using pipeline API."""
        try:
            if self.model_type == "causal":
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            else:
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1
                )
            self.tokenizer = self.pipeline.tokenizer
            self.model = self.pipeline.model
        except Exception as e:
            logger.error(f"Error loading pipeline: {str(e)}")
            self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer directly."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.model_type == "causal":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            
            self.model.to(self.device)
            self.model.eval()
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated texts
        """
        if self.use_pipeline and hasattr(self, 'pipeline'):
            try:
                results = self.pipeline(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    **kwargs
                )
                
                if self.model_type == "causal":
                    return [r["generated_text"] for r in results]
                else:
                    return [r["generated_text"] for r in results]
            except Exception as e:
                logger.error(f"Pipeline generation error: {str(e)}")
                return self._generate_direct(prompt, max_length, num_return_sequences, temperature, top_p, top_k, do_sample, **kwargs)
        else:
            return self._generate_direct(prompt, max_length, num_return_sequences, temperature, top_p, top_k, do_sample, **kwargs)
    
    def _generate_direct(
        self,
        prompt: str,
        max_length: int,
        num_return_sequences: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        **kwargs
    ) -> List[str]:
        """Generate using model directly."""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generation config
        generation_config = GenerationConfig(
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            **kwargs
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=generation_config
            )
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts
    
    def generate_summary(self, text: str, max_length: int = 50) -> str:
        """
        Generate summary of text.
        
        Args:
            text: Input text
            max_length: Maximum summary length
        
        Returns:
            Summary text
        """
        if self.model_type == "seq2seq":
            prompt = f"summarize: {text}"
        else:
            prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
        
        summaries = self.generate(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.3
        )
        
        return summaries[0] if summaries else ""
    
    def generate_recommendation(self, context: str, max_length: int = 100) -> str:
        """
        Generate recommendation based on context.
        
        Args:
            context: Context information
            max_length: Maximum length
        
        Returns:
            Recommendation text
        """
        prompt = f"Based on the following context, provide a recommendation:\n\n{context}\n\nRecommendation:"
        
        recommendations = self.generate(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7
        )
        
        return recommendations[0] if recommendations else ""





"""
LLM-based Coaching System using Transformers
Enhanced with LoRA support, better generation parameters, and improved tokenization
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import (
        GPT2LMHeadModel, GPT2Tokenizer,
        T5ForConditionalGeneration, T5Tokenizer,
        pipeline, AutoModelForCausalLM,
        AutoTokenizer, AutoModelForCausalLM,
        GenerationConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from .lora_finetuning import LoRAFineTuner
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False


class LLMRecoveryCoach(nn.Module):
    """
    LLM-based recovery coach with enhanced generation and LoRA support
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[torch.device] = None,
        use_gpu: bool = True,
        use_mixed_precision: bool = True,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0
    ):
        """
        Initialize LLM coach
        
        Args:
            model_name: Model name (gpt2, gpt2-medium, etc.)
            device: PyTorch device
            use_gpu: Use GPU
            use_mixed_precision: Use mixed precision (FP16)
            use_lora: Use LoRA fine-tuning
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        self.device = device or torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        self.use_lora = use_lora and LORA_AVAILABLE
        
        # Load model and tokenizer
        torch_dtype = torch.float16 if self.use_mixed_precision else torch.float32
        
        if self.use_lora:
            # Use LoRA fine-tuner
            self.lora_tuner = LoRAFineTuner(
                model_name=model_name,
                rank=lora_rank,
                alpha=lora_alpha,
                device=self.device
            )
            self.model = self.lora_tuner.model
            self.tokenizer = self.lora_tuner.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype
            ).to(self.device)
        
        self.model.eval()
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Enable torch.compile for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        logger.info(
            f"LLMRecoveryCoach initialized with {model_name} on {self.device} "
            f"(mixed_precision={self.use_mixed_precision}, lora={self.use_lora})"
        )
    
    def generate_coaching_message(
        self,
        user_situation: str,
        days_sober: int,
        current_challenge: Optional[str] = None,
        max_length: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        num_beams: int = 1,
        do_sample: bool = True
    ) -> str:
        """
        Generate personalized coaching message with enhanced generation parameters
        
        Args:
            user_situation: User's current situation
            days_sober: Days of sobriety
            current_challenge: Current challenge (optional)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            num_beams: Number of beams for beam search
            do_sample: Use sampling (vs greedy/beam search)
            
        Returns:
            Generated coaching message
        """
        # Create prompt
        prompt = f"User has been sober for {days_sober} days. "
        prompt += f"Situation: {user_situation}. "
        if current_challenge:
            prompt += f"Current challenge: {current_challenge}. "
        prompt += "Provide encouraging and supportive coaching message:"
        
        # Tokenize with proper padding
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generation config
        generation_config = {
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        if not do_sample:
            generation_config["num_beams"] = num_beams
            generation_config.pop("temperature")
            generation_config.pop("top_p")
            generation_config.pop("top_k")
        
        # Generate with mixed precision
        with torch.inference_mode():
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        **generation_config
                    )
            else:
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generation_config
                )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated part
        if prompt in generated:
            generated = generated.split(prompt)[-1].strip()
        
        return generated
    
    def generate_motivational_message(
        self,
        milestone: str,
        achievement: str,
        max_length: int = 100
    ) -> str:
        """Generate motivational message for milestone"""
        prompt = f"Celebrate milestone: {milestone}. Achievement: {achievement}. Motivational message:"
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prompt in generated:
            generated = generated.split(prompt)[-1].strip()
        
        return generated


class T5RecoveryCoach(nn.Module):
    """T5-based recovery coach for better quality"""
    
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
        
        logger.info(f"T5RecoveryCoach initialized on {self.device}")
    
    def generate_coaching(
        self,
        task: str,
        context: str,
        max_length: int = 100
    ) -> str:
        """
        Generate coaching based on task and context
        
        Args:
            task: Task prefix (e.g., "coach:", "motivate:", "advise:")
            context: Context information
            
        Returns:
            Generated coaching text
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
    
    def coach(self, situation: str) -> str:
        """Generate coaching for situation"""
        return self.generate_coaching("coach:", situation)
    
    def motivate(self, achievement: str) -> str:
        """Generate motivation for achievement"""
        return self.generate_coaching("motivate:", achievement)
    
    def advise(self, challenge: str) -> str:
        """Generate advice for challenge"""
        return self.generate_coaching("advise:", challenge)


def create_llm_coach(
    model_name: str = "gpt2",
    device: Optional[torch.device] = None
) -> LLMRecoveryCoach:
    """Factory function for LLM coach"""
    return LLMRecoveryCoach(model_name=model_name, device=device)


def create_t5_coach(
    model_name: str = "t5-small",
    device: Optional[torch.device] = None
) -> T5RecoveryCoach:
    """Factory function for T5 coach"""
    return T5RecoveryCoach(model_name=model_name, device=device)


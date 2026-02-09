from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any, Tuple
import numpy as np
from transformers import (
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import (
import logging
from tqdm import tqdm
import json
import os
from typing import Any, List, Dict, Optional
import asyncio
"""
Advanced LLM Models - Transformers Library Implementation
Featuring LoRA fine-tuning, custom tokenizers, and optimization techniques.
"""

    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
    GenerationConfig,
    BitsAndBytesConfig
)
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig
)

logger = logging.getLogger(__name__)


class AdvancedLLMModel(nn.Module):
    """
    Advanced LLM model with custom features and optimizations.
    """
    
    def __init__(
        self,
        model_name: str,
        use_4bit: bool = True,
        use_8bit: bool = False,
        use_flash_attention: bool = True,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16
    ):
        
    """__init__ function."""
super().__init__()
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.use_flash_attention = use_flash_attention
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # Enable optimizations
        self._enable_optimizations()
    
    def _load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, AutoTokenizer]:
        """Load model and tokenizer with optimizations."""
        # Configure quantization
        quantization_config = None
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def _enable_optimizations(self) -> Any:
        """Enable model optimizations."""
        # Enable flash attention if available
        if self.use_flash_attention and hasattr(self.model, 'enable_flash_attention'):
            self.model.enable_flash_attention()
        
        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Enable xformers memory efficient attention
        if hasattr(self.model, 'enable_xformers_memory_efficient_attention'):
            self.model.enable_xformers_memory_efficient_attention()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> CausalLMOutputWithPast:
        """
        Forward pass of the LLM model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for language modeling
            past_key_values: Past key values for caching
            use_cache: Whether to use cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            
        Returns:
            Model outputs
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate text using the LLM model.
        
        Args:
            prompt: Input prompt(s)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            repetition_penalty: Repetition penalty
            length_penalty: Length penalty
            early_stopping: Whether to use early stopping
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text sequences
        """
        # Tokenize input
        if isinstance(prompt, str):
            prompt = [prompt]
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Configure generation
        generation_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=pad_token_id or self.tokenizer.pad_token_id,
            eos_token_id=eos_token_id or self.tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            **kwargs
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            # Remove input tokens
            generated_tokens = output[inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text to token embeddings.
        
        Args:
            text: Input text
            
        Returns:
            Token embeddings
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]  # Last layer
        
        return embeddings
    
    def get_embeddings(self, text: Union[str, List[str]], pooling: str = "mean") -> torch.Tensor:
        """
        Get text embeddings with pooling.
        
        Args:
            text: Input text
            pooling: Pooling strategy ("mean", "cls", "max")
            
        Returns:
            Text embeddings
        """
        embeddings = self.encode(text)
        
        if pooling == "mean":
            # Mean pooling (excluding padding tokens)
            attention_mask = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )["attention_mask"].to(embeddings.device)
            
            # Apply attention mask
            embeddings = embeddings * attention_mask.unsqueeze(-1)
            pooled_embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        elif pooling == "cls":
            # Use first token (CLS token)
            pooled_embeddings = embeddings[:, 0, :]
        
        elif pooling == "max":
            # Max pooling
            pooled_embeddings = embeddings.max(dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        return pooled_embeddings


class LoRAFineTuner:
    """
    LoRA (Low-Rank Adaptation) fine-tuner for efficient model adaptation.
    """
    
    def __init__(
        self,
        model: AdvancedLLMModel,
        lora_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ):
        
    """__init__ function."""
self.model = model
        self.lora_config = lora_config
        self.training_config = training_config
        
        # Apply LoRA
        self.peft_model = self._apply_lora()
        
        # Setup training components
        self.optimizer = None
        self.scheduler = None
        self._setup_training()
    
    def _apply_lora(self) -> PeftModel:
        """Apply LoRA configuration to model."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_config.get("r", 16),
            lora_alpha=self.lora_config.get("lora_alpha", 32),
            lora_dropout=self.lora_config.get("lora_dropout", 0.1),
            target_modules=self.lora_config.get("target_modules", ["q_proj", "v_proj"]),
            bias=self.lora_config.get("bias", "none")
        )
        
        return get_peft_model(self.model.model, lora_config)
    
    def _setup_training(self) -> Any:
        """Setup training components."""
        # Optimizer
        optimizer_name = self.training_config.get("optimizer", "adamw")
        learning_rate = self.training_config.get("learning_rate", 1e-4)
        weight_decay = self.training_config.get("weight_decay", 0.01)
        
        if optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.peft_model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.peft_model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Scheduler
        scheduler_name = self.training_config.get("scheduler", "cosine")
        num_training_steps = self.training_config.get("num_training_steps", 1000)
        warmup_steps = self.training_config.get("warmup_steps", 100)
        
        if scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=learning_rate * 0.1
            )
        elif scheduler_name == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=num_training_steps
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def train(
        self,
        train_dataset: List[Dict[str, str]],
        val_dataset: Optional[List[Dict[str, str]]] = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        save_steps: int = 500,
        eval_steps: int = 500,
        output_dir: str = "./lora_output"
    ):
        """
        Train the model using LoRA fine-tuning.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm
            save_steps: Save model every N steps
            eval_steps: Evaluate model every N steps
            output_dir: Output directory for saving
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Training
            self.peft_model.train()
            train_loss = 0.0
            
            for i in range(0, len(train_dataset), batch_size):
                batch = train_dataset[i:i + batch_size]
                
                # Prepare batch
                texts = [item["text"] for item in batch]
                labels = [item.get("label", "") for item in batch]
                
                # Tokenize
                inputs = self.model.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move to device
                inputs = {k: v.to(self.model.model.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.peft_model(**inputs)
                loss = outputs.loss
                
                # Backward pass
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                if (i // batch_size + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.peft_model.parameters(),
                        max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                
                train_loss += loss.item()
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    self.save_checkpoint(output_dir, global_step)
                
                # Evaluation
                if val_dataset and global_step % eval_steps == 0:
                    val_loss = self.evaluate(val_dataset, batch_size)
                    logger.info(f"Step {global_step}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint(output_dir, "best")
                else:
                    logger.info(f"Step {global_step}: Train Loss = {train_loss:.4f}")
            
            # End of epoch
            avg_train_loss = train_loss / (len(train_dataset) // batch_size)
            logger.info(f"Epoch {epoch + 1} completed. Average Train Loss: {avg_train_loss:.4f}")
    
    def evaluate(self, val_dataset: List[Dict[str, str]], batch_size: int) -> float:
        """
        Evaluate the model on validation dataset.
        
        Args:
            val_dataset: Validation dataset
            batch_size: Batch size
            
        Returns:
            Average validation loss
        """
        self.peft_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i in range(0, len(val_dataset), batch_size):
                batch = val_dataset[i:i + batch_size]
                
                # Prepare batch
                texts = [item["text"] for item in batch]
                
                # Tokenize
                inputs = self.model.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move to device
                inputs = {k: v.to(self.model.model.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.peft_model(**inputs)
                val_loss += outputs.loss.item()
        
        return val_loss / (len(val_dataset) // batch_size)
    
    def save_checkpoint(self, output_dir: str, step: Union[int, str]):
        """
        Save model checkpoint.
        
        Args:
            output_dir: Output directory
            step: Step number or identifier
        """
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        self.peft_model.save_pretrained(checkpoint_dir)
        
        # Save training config
        config = {
            "lora_config": self.lora_config,
            "training_config": self.training_config
        }
        
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(config, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_dir: Checkpoint directory
        """
        self.peft_model = PeftModel.from_pretrained(
            self.model.model,
            checkpoint_dir
        )
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")


class CustomTokenizer:
    """
    Custom tokenizer with advanced features and optimizations.
    """
    
    def __init__(
        self,
        tokenizer_name: str,
        max_length: int = 512,
        padding_side: str = "right",
        truncation_side: str = "right"
    ):
        
    """__init__ function."""
self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            padding_side=padding_side,
            truncation_side=truncation_side
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            return_tensors: Return tensor type
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            Tokenized inputs
        """
        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length
        )
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return self.tokenizer.tokenize(text)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mappings."""
        return {
            "pad_token": self.tokenizer.pad_token_id,
            "eos_token": self.tokenizer.eos_token_id,
            "bos_token": self.tokenizer.bos_token_id,
            "unk_token": self.tokenizer.unk_token_id
        }


class LLMInferenceEngine:
    """
    Advanced LLM inference engine with optimizations and caching.
    """
    
    def __init__(
        self,
        model: AdvancedLLMModel,
        use_cache: bool = True,
        max_cache_size: int = 1000,
        use_batch_inference: bool = True,
        max_batch_size: int = 8
    ):
        
    """__init__ function."""
self.model = model
        self.use_cache = use_cache
        self.max_cache_size = max_cache_size
        self.use_batch_inference = use_batch_inference
        self.max_batch_size = max_batch_size
        
        # Initialize cache
        if use_cache:
            self.cache = {}
        else:
            self.cache = None
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of prompts
            **kwargs: Generation arguments
            
        Returns:
            List of generated texts
        """
        if not self.use_batch_inference or len(prompts) <= self.max_batch_size:
            return self.model.generate(prompts, **kwargs)
        
        # Process in batches
        results = []
        for i in range(0, len(prompts), self.max_batch_size):
            batch_prompts = prompts[i:i + self.max_batch_size]
            batch_results = self.model.generate(batch_prompts, **kwargs)
            results.extend(batch_results)
        
        return results
    
    def generate_with_cache(
        self,
        prompt: str,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text with caching.
        
        Args:
            prompt: Input prompt
            cache_key: Cache key (defaults to prompt)
            **kwargs: Generation arguments
            
        Returns:
            Generated text
        """
        if not self.use_cache or self.cache is None:
            return self.model.generate(prompt, **kwargs)[0]
        
        if cache_key is None:
            cache_key = prompt
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate
        result = self.model.generate(prompt, **kwargs)[0]
        
        # Store in cache
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        return result
    
    def get_embeddings_batch(
        self,
        texts: List[str],
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Get embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts
            pooling: Pooling strategy
            
        Returns:
            Text embeddings
        """
        return self.model.get_embeddings(texts, pooling)
    
    def similarity_search(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
        pooling: str = "mean"
    ) -> List[Tuple[str, float]]:
        """
        Perform similarity search.
        
        Args:
            query: Query text
            candidates: Candidate texts
            top_k: Number of top results
            pooling: Pooling strategy
            
        Returns:
            List of (text, similarity_score) tuples
        """
        # Get embeddings
        query_embedding = self.model.get_embeddings(query, pooling)
        candidate_embeddings = self.model.get_embeddings(candidates, pooling)
        
        # Compute similarities
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            candidate_embeddings,
            dim=1
        )
        
        # Get top-k results
        top_indices = torch.topk(similarities, top_k).indices
        results = [(candidates[i], similarities[i].item()) for i in top_indices]
        
        return results
    
    def clear_cache(self) -> Any:
        """Clear the inference cache."""
        if self.cache is not None:
            self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache is None:
            return {"cache_enabled": False}
        
        return {
            "cache_enabled": True,
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "cache_hit_rate": 0.0  # Would need to track hits/misses
        } 
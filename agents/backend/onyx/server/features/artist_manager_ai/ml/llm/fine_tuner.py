"""
Fine-tuning with LoRA and P-tuning
===================================

Fine-tuning transformer models using efficient techniques.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("peft not available. Install with: pip install peft")

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    r: int = 8  # Rank
    lora_alpha: int = 16  # LoRA alpha
    target_modules: Optional[list] = None  # Target modules (None = auto)
    lora_dropout: float = 0.1
    bias: str = "none"  # "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"


class FineTuner:
    """
    Fine-tuner for transformer models.
    
    Supports:
    - LoRA (Low-Rank Adaptation)
    - Full fine-tuning
    - Custom training arguments
    """
    
    def __init__(
        self,
        model_name: str,
        use_lora: bool = True,
        lora_config: Optional[LoRAConfig] = None
    ):
        """
        Initialize fine-tuner.
        
        Args:
            model_name: HuggingFace model name
            use_lora: Whether to use LoRA
            lora_config: LoRA configuration
        """
        self.model_name = model_name
        self.use_lora = use_lora and PEFT_AVAILABLE
        self.lora_config = lora_config or LoRAConfig()
        
        self.tokenizer = None
        self.model = None
        self._logger = logger
    
    def load_model(self):
        """Load model and tokenizer."""
        self._logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Apply LoRA if enabled
        if self.use_lora:
            self._apply_lora()
        
        self._logger.info("Model loaded successfully")
    
    def _apply_lora(self):
        """Apply LoRA to model."""
        if not PEFT_AVAILABLE:
            self._logger.warning("PEFT not available. Skipping LoRA.")
            return
        
        # Auto-detect target modules if not specified
        target_modules = self.lora_config.target_modules
        if target_modules is None:
            # Common target modules for different architectures
            if "gpt" in self.model_name.lower() or "llama" in self.model_name.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            elif "bert" in self.model_name.lower():
                target_modules = ["query", "value", "key"]
            else:
                # Try to auto-detect
                target_modules = ["q_proj", "v_proj"]
        
        # Create LoRA config
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        self._logger.info("LoRA applied successfully")
    
    def prepare_dataset(self, texts: list, max_length: int = 512):
        """
        Prepare dataset for training.
        
        Args:
            texts: List of training texts
            max_length: Maximum sequence length
        
        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Tokenize
        tokenized = tokenize_function(texts)
        
        return tokenized
    
    def train(
        self,
        train_dataset,
        output_dir: str = "./results",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        **kwargs
    ):
        """
        Train model.
        
        Args:
            train_dataset: Training dataset
            output_dir: Output directory
            num_train_epochs: Number of epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            logging_steps: Logging frequency
            save_steps: Save frequency
            **kwargs: Additional training arguments
        """
        if self.model is None:
            self.load_model()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            fp16=torch.cuda.is_available(),
            save_total_limit=3,
            load_best_model_at_end=True,
            **kwargs
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )
        
        # Train
        self._logger.info("Starting training...")
        trainer.train()
        
        # Save
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        self._logger.info(f"Training complete. Model saved to {output_dir}")





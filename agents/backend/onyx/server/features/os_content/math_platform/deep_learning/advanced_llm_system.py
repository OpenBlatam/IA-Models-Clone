from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from transformers import (
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import logging as transformers_logging
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import gc
from pathlib import Path
import warnings
from tqdm import tqdm
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced LLM System with Transformers Library and Efficient Fine-tuning
Production-ready LLM system using HuggingFace Transformers with LoRA, P-tuning, and other PEFT methods.
"""

    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    DataCollatorWithPadding, get_linear_schedule_with_warmup,
    BitsAndBytesConfig, pipeline, GenerationConfig,
    PeftModel, PeftConfig, LoraConfig, PrefixTuningConfig,
    PromptTuningConfig, AdaLoraConfig
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM system with PEFT support."""
    # Model configuration
    model_name: str = "gpt2"
    model_type: str = "causal"  # causal, seq2seq, classification, token_classification
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    
    # PEFT configuration
    use_peft: bool = True
    peft_method: str = "lora"  # lora, prefix_tuning, prompt_tuning, p_tuning, adalora
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    num_virtual_tokens: int = 20  # for prefix/prompt tuning
    prompt_encoder_hidden_size: int = 128  # for p-tuning
    
    # Training configuration
    batch_size: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Mixed precision and optimization
    fp16: bool = True
    bf16: bool = False
    use_8bit: bool = False
    use_4bit: bool = False
    gradient_checkpointing: bool = True
    
    # Generation configuration
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    
    # Output configuration
    output_dir: str = "./llm_outputs"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Data configuration
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"


class LLMDataset(Dataset):
    """Custom dataset for LLM training with proper tokenization."""
    
    def __init__(self, texts: List[str], tokenizer, config: LLMConfig, 
                 labels: Optional[List[int]] = None):
        
    """__init__ function."""
self.texts = texts
        self.tokenizer = tokenizer
        self.config = config
        self.labels = labels
        
        # Tokenize all texts
        self.encodings = self._tokenize_texts()
    
    def _tokenize_texts(self) -> Dict[str, List]:
        """Tokenize all texts efficiently."""
        encodings = self.tokenizer(
            self.texts,
            truncation=self.config.truncation,
            padding=self.config.padding,
            max_length=self.config.max_length,
            return_tensors=None  # Return lists instead of tensors
        )
        
        # Add labels if provided
        if self.labels is not None:
            encodings['labels'] = self.labels
        
        return encodings
    
    def __len__(self) -> Any:
        return len(self.texts)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


class AdvancedLLMSystem:
    """Advanced LLM system using HuggingFace Transformers library with PEFT support."""
    
    def __init__(self, config: LLMConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.peft_config = None
        
        # Setup quantization config if needed
        self.quantization_config = self._setup_quantization()
        
        # Initialize components
        self._initialize_tokenizer()
        self._initialize_model()
        
        logger.info(f"LLM system initialized on device: {self.device}")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Model type: {config.model_type}")
        logger.info(f"PEFT method: {config.peft_method if config.use_peft else 'None'}")
    
    def _setup_quantization(self) -> Optional[BitsAndBytesConfig]:
        """Setup quantization configuration for memory efficiency."""
        if self.config.use_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif self.config.use_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return None
    
    def _initialize_tokenizer(self) -> Any:
        """Initialize the tokenizer for the specified model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Tokenizer loaded: {self.config.model_name}")
            logger.info(f"Vocabulary size: {self.tokenizer.vocab_size}")
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def _setup_peft_config(self) -> Any:
        """Setup PEFT configuration based on the specified method."""
        if not self.config.use_peft:
            return None
        
        if self.config.peft_method == "lora":
            return LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
        elif self.config.peft_method == "prefix_tuning":
            return PrefixTuningConfig(
                num_virtual_tokens=self.config.num_virtual_tokens,
                encoder_hidden_size=self.config.prompt_encoder_hidden_size,
                task_type="CAUSAL_LM"
            )
        elif self.config.peft_method == "prompt_tuning":
            return PromptTuningConfig(
                num_virtual_tokens=self.config.num_virtual_tokens,
                task_type="CAUSAL_LM"
            )
        elif self.config.peft_method == "adalora":
            return AdaLoraConfig(
                r=self.config.lora_r,
                target_modules=self.config.target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                task_type="CAUSAL_LM"
            )
        else:
            raise ValueError(f"Unsupported PEFT method: {self.config.peft_method}")
    
    def _initialize_model(self) -> Any:
        """Initialize the model based on the specified type with PEFT support."""
        try:
            model_kwargs = {
                "pretrained_model_name_or_path": self.config.model_name,
                "trust_remote_code": True
            }
            
            # Add quantization config if specified
            if self.quantization_config:
                model_kwargs["quantization_config"] = self.quantization_config
            
            # Load model based on type
            if self.config.model_type == "causal":
                self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            elif self.config.model_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(**model_kwargs)
            elif self.config.model_type == "classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs)
            elif self.config.model_type == "token_classification":
                self.model = AutoModelForTokenClassification.from_pretrained(**model_kwargs)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # Setup PEFT if enabled
            if self.config.use_peft:
                self.peft_config = self._setup_peft_config()
                self.model = PeftModel.from_pretrained(self.model, self.peft_config)
            
            # Move to device
            self.model.to(self.device)
            
            # Enable gradient checkpointing for memory efficiency
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_training_data(self, texts: List[str], 
                            labels: Optional[List[int]] = None) -> LLMDataset:
        """Prepare training data with proper tokenization."""
        return LLMDataset(texts, self.tokenizer, self.config, labels)
    
    def setup_trainer(self, train_dataset: LLMDataset, 
                     eval_dataset: Optional[LLMDataset] = None):
        """Setup the HuggingFace Trainer for training with PEFT support."""
        
        # Setup data collator
        if self.config.model_type == "causal":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # We're doing causal language modeling
            )
        else:
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            logging_strategy="steps",
            report_to=None,  # Disable wandb/tensorboard
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            group_by_length=True,
            length_column_name="length",
            label_names=["labels"] if "labels" in train_dataset[0].keys() else None
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        logger.info("Trainer setup completed")
    
    def train(self, train_texts: List[str], eval_texts: Optional[List[str]] = None,
              train_labels: Optional[List[int]] = None, 
              eval_labels: Optional[List[int]] = None):
        """Train the LLM model with PEFT."""
        
        # Prepare datasets
        train_dataset = self.prepare_training_data(train_texts, train_labels)
        eval_dataset = None
        if eval_texts:
            eval_dataset = self.prepare_training_data(eval_texts, eval_labels)
        
        # Setup trainer
        self.setup_trainer(train_dataset, eval_dataset)
        
        # Train the model
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save the model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Log training results
        logger.info(f"Training completed. Loss: {train_result.training_loss:.4f}")
        
        return train_result
    
    def generate_text(self, prompt: str, max_new_tokens: Optional[int] = None,
                     temperature: Optional[float] = None, 
                     top_p: Optional[float] = None,
                     do_sample: Optional[bool] = None) -> str:
        """Generate text using the trained model."""
        
        # Set generation parameters
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        with torch.no_grad():
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=self.config.top_k,
                do_sample=do_sample,
                repetition_penalty=self.config.repetition_penalty,
                length_penalty=self.config.length_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def create_pipeline(self, task: str = "text-generation") -> pipeline:
        """Create a HuggingFace pipeline for easy inference."""
        return pipeline(
            task,
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32
        )
    
    def save_model(self, path: str):
        """Save the trained model and tokenizer."""
        os.makedirs(path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save configuration
        config_path = os.path.join(path, "llm_config.json")
        with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load a trained model and tokenizer."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Load model
        if self.config.model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(path)
        elif self.config.model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        elif self.config.model_type == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
        elif self.config.model_type == "token_classification":
            self.model = AutoModelForTokenClassification.from_pretrained(path)
        
        # Load PEFT if it was used
        if self.config.use_peft:
            self.model = PeftModel.from_pretrained(self.model, path)
        
        # Move to device
        self.model.to(self.device)
        
        # Load configuration
        config_path = os.path.join(path, "llm_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config_dict = json.load(f)
                self.config = LLMConfig(**config_dict)
        
        logger.info(f"Model loaded from: {path}")
    
    def evaluate_model(self, eval_texts: List[str], 
                      eval_labels: Optional[List[int]] = None) -> Dict[str, float]:
        """Evaluate the model on test data."""
        eval_dataset = self.prepare_training_data(eval_texts, eval_labels)
        
        if self.trainer is None:
            self.setup_trainer(eval_dataset, eval_dataset)
        
        # Run evaluation
        eval_results = self.trainer.evaluate()
        
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "vocabulary_size": self.tokenizer.vocab_size,
            "max_length": self.config.max_length,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "peft_method": self.config.peft_method if self.config.use_peft else "None",
            "quantization": {
                "fp16": self.config.fp16,
                "bf16": self.config.bf16,
                "8bit": self.config.use_8bit,
                "4bit": self.config.use_4bit
            }
        }


def create_llm_system(model_name: str = "gpt2", model_type: str = "causal",
                     use_peft: bool = True, peft_method: str = "lora",
                     use_fp16: bool = True, use_4bit: bool = False) -> AdvancedLLMSystem:
    """Create an LLM system with default configuration and PEFT support."""
    config = LLMConfig(
        model_name=model_name,
        model_type=model_type,
        use_peft=use_peft,
        peft_method=peft_method,
        fp16=use_fp16,
        use_4bit=use_4bit,
        batch_size=2 if use_4bit else 4,
        num_epochs=1
    )
    return AdvancedLLMSystem(config)


# Example usage and testing
if __name__ == "__main__":
    # Create LLM system with LoRA
    llm_system = create_llm_system(
        "gpt2", 
        "causal", 
        use_peft=True, 
        peft_method="lora",
        use_fp16=True
    )
    
    # Sample training data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a powerful programming language for data science.",
        "Transformers have revolutionized natural language processing."
    ]
    
    # Generate text
    prompt = "The future of artificial intelligence"
    generated_text = llm_system.generate_text(prompt, max_new_tokens=50)
    print(f"Generated text: {generated_text}")
    
    # Get model info
    model_info = llm_system.get_model_info()
    print(f"Model info: {model_info}") 
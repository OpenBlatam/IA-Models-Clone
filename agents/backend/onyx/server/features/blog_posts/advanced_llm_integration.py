from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.func import functional_call, vmap, grad
from torch.export import export
from torch._dynamo import optimize
import torch._dynamo as dynamo
from transformers import (
from transformers.models.llama import LlamaTokenizer, LlamaForCausalLM
from transformers.models.mistral import MistralTokenizer, MistralForCausalLM
from transformers.models.gpt2 import GPT2Tokenizer, GPT2LMHeadModel
from transformers.models.t5 import T5Tokenizer, T5ForConditionalGeneration
from transformers.models.bert import BertTokenizer, BertForSequenceClassification
from peft import (
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import structlog
from contextlib import contextmanager
import time
import warnings
import os
import gc
from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Advanced LLM Integration with Modern PyTorch Practices
=====================================================

Production-ready LLM integration with latest transformers, quantization,
optimization, and deployment features.
"""


# Modern PyTorch imports

# Transformers imports
    AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig,
    pipeline, PreTrainedModel, PreTrainedTokenizer, AutoConfig,
    LlamaTokenizer, LlamaForCausalLM, LlamaConfig,
    MistralTokenizer, MistralForCausalLM, MistralConfig,
    GPT2Tokenizer, GPT2LMHeadModel, GPT2Config,
    T5Tokenizer, T5ForConditionalGeneration, T5Config,
    BertTokenizer, BertForSequenceClassification, BertConfig
)

# PEFT imports
    LoraConfig, get_peft_model, TaskType, PeftModel,
    prepare_model_for_kbit_training, PeftConfig
)

# Additional imports

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlog.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
warnings.filterwarnings("ignore")


@dataclass
class LLMConfig:
    """Configuration for LLM models."""
    model_name: str = "microsoft/DialoGPT-medium"
    model_type: str = "causal"  # causal, sequence_classification, conditional_generation
    task: str = "text_generation"
    max_length: int = 512
    batch_size: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 2
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    quantization: str = "4bit"  # none, 4bit, 8bit
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    push_to_hub: bool = False
    report_to: Optional[str] = None


class AdvancedLLMTrainer:
    """Advanced LLM trainer with modern optimizations."""
    
    def __init__(self, config: LLMConfig):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
        # Apply optimizations
        self.model = self._apply_optimizations(self.model)
        
        # Initialize PEFT if enabled
        if self.config.use_peft:
            self.model = self._setup_peft(self.model)
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer based on model type."""
        try:
            if "llama" in self.config.model_name.lower():
                tokenizer = LlamaTokenizer.from_pretrained(self.config.model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            elif "mistral" in self.config.model_name.lower():
                tokenizer = MistralTokenizer.from_pretrained(self.config.model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            elif "gpt2" in self.config.model_name.lower():
                tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            elif "t5" in self.config.model_name.lower():
                tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
            elif "bert" in self.config.model_name.lower():
                tokenizer = BertTokenizer.from_pretrained(self.config.model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            
            self.logger.info(f"Loaded tokenizer: {self.config.model_name}")
            return tokenizer
        
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_model(self) -> PreTrainedModel:
        """Load model with quantization if enabled."""
        try:
            # Setup quantization config
            quantization_config = None
            if self.config.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            elif self.config.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            
            # Load model based on type
            if self.config.model_type == "causal":
                if "llama" in self.config.model_name.lower():
                    model = LlamaForCausalLM.from_pretrained(
                        self.config.model_name,
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                        device_map="auto" if quantization_config else None,
                        use_flash_attention_2=self.config.use_flash_attention
                    )
                elif "mistral" in self.config.model_name.lower():
                    model = MistralForCausalLM.from_pretrained(
                        self.config.model_name,
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                        device_map="auto" if quantization_config else None,
                        use_flash_attention_2=self.config.use_flash_attention
                    )
                elif "gpt2" in self.config.model_name.lower():
                    model = GPT2LMHeadModel.from_pretrained(
                        self.config.model_name,
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16 if self.config.fp16 else torch.float32
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.config.model_name,
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                        device_map="auto" if quantization_config else None
                    )
            
            elif self.config.model_type == "sequence_classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    device_map="auto" if quantization_config else None
                )
            
            elif self.config.model_type == "conditional_generation":
                model = T5ForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    device_map="auto" if quantization_config else None
                )
            
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            self.logger.info(f"Loaded model: {self.config.model_name}")
            return model
        
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _apply_optimizations(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply modern optimizations to the model."""
        try:
            # Gradient checkpointing
            if self.config.use_gradient_checkpointing:
                model.gradient_checkpointing_enable()
                self.logger.info("Enabled gradient checkpointing")
            
            # Flash attention
            if self.config.use_flash_attention and hasattr(model.config, 'use_flash_attention_2'):
                model.config.use_flash_attention_2 = True
                self.logger.info("Enabled flash attention 2")
            
            # Memory efficient attention
            if hasattr(model.config, 'use_memory_efficient_attention'):
                model.config.use_memory_efficient_attention = True
                self.logger.info("Enabled memory efficient attention")
            
            # Compile model if available
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                    self.logger.info("Model compiled with torch.compile")
                except Exception as e:
                    self.logger.warning(f"Model compilation failed: {e}")
            
            return model
        
        except Exception as e:
            self.logger.error(f"Failed to apply optimizations: {e}")
            return model
    
    def _setup_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """Setup PEFT (Parameter-Efficient Fine-Tuning)."""
        try:
            # Prepare model for k-bit training if quantized
            if self.config.quantization != "none":
                model = prepare_model_for_kbit_training(model)
                self.logger.info("Prepared model for k-bit training")
            
            # Setup LoRA config
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self._get_target_modules(),
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM if self.config.model_type == "causal" else TaskType.SEQ_CLS
            )
            
            # Apply PEFT
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            self.logger.info("Applied PEFT configuration")
            return model
        
        except Exception as e:
            self.logger.error(f"Failed to setup PEFT: {e}")
            return model
    
    def _get_target_modules(self) -> List[str]:
        """Get target modules for LoRA based on model type."""
        if "llama" in self.config.model_name.lower():
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "mistral" in self.config.model_name.lower():
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt2" in self.config.model_name.lower():
            return ["c_attn", "c_proj", "c_fc", "c_proj"]
        else:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    def prepare_dataset(self, texts: List[str], labels: Optional[List[int]] = None) -> Dataset:
        """Prepare dataset for training."""
        class LLMDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length, model_type) -> Any:
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.model_type = model_type
            
            def __len__(self) -> Any:
                return len(self.texts)
            
            def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
                text = self.texts[idx]
                
                if self.model_type == "causal":
                    # For causal LM, we need to create input-output pairs
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    
                    # For causal LM, labels are the same as input_ids
                    encoding["labels"] = encoding["input_ids"].clone()
                    
                elif self.model_type == "sequence_classification":
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    
                    if self.labels is not None:
                        encoding["labels"] = torch.tensor(self.labels[idx])
                
                elif self.model_type == "conditional_generation":
                    # For T5, we need source and target texts
                    if isinstance(text, (list, tuple)) and len(text) == 2:
                        source, target = text
                    else:
                        source, target = text, text
                    
                    source_encoding = self.tokenizer(
                        source,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    
                    target_encoding = self.tokenizer(
                        target,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    
                    encoding = {
                        "input_ids": source_encoding["input_ids"],
                        "attention_mask": source_encoding["attention_mask"],
                        "labels": target_encoding["input_ids"]
                    }
                
                # Remove batch dimension
                for key in encoding:
                    if isinstance(encoding[key], torch.Tensor):
                        encoding[key] = encoding[key].squeeze(0)
                
                return encoding
        
        return LLMDataset(texts, labels, self.tokenizer, self.config.max_length, self.config.model_type)
    
    def train(self, train_texts: List[str], train_labels: Optional[List[int]] = None,
              val_texts: Optional[List[str]] = None, val_labels: Optional[List[int]] = None):
        """Train the model using modern best practices."""
        try:
            # Prepare datasets
            train_dataset = self.prepare_dataset(train_texts, train_labels)
            val_dataset = None
            if val_texts and val_labels:
                val_dataset = self.prepare_dataset(val_texts, val_labels)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir="./llm_results",
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                warmup_steps=self.config.warmup_steps,
                weight_decay=self.config.weight_decay,
                logging_dir="./llm_logs",
                logging_steps=self.config.logging_steps,
                evaluation_strategy=self.config.evaluation_strategy,
                save_strategy=self.config.save_strategy,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                load_best_model_at_end=self.config.load_best_model_at_end,
                metric_for_best_model=self.config.metric_for_best_model,
                greater_is_better=self.config.greater_is_better,
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                dataloader_num_workers=self.config.dataloader_num_workers,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                max_grad_norm=self.config.max_grad_norm,
                save_total_limit=self.config.save_total_limit,
                report_to=self.config.report_to,
                remove_unused_columns=self.config.remove_unused_columns,
                push_to_hub=self.config.push_to_hub,
                dataloader_pin_memory=self.config.dataloader_pin_memory
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )
            
            # Train the model
            trainer.train()
            
            # Save the model
            trainer.save_model("./final_llm_model")
            self.tokenizer.save_pretrained("./final_llm_model")
            
            self.logger.info("LLM training completed successfully")
            return trainer
        
        except Exception as e:
            self.logger.error(f"LLM training failed: {e}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7,
                     top_p: float = 0.9, do_sample: bool = True) -> str:
        """Generate text using the trained model."""
        try:
            self.model.eval()
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text
        
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            return ""
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions using the trained model."""
        try:
            self.model.eval()
            predictions = []
            
            for text in texts:
                if self.config.model_type == "causal":
                    # Generate text
                    generated_text = self.generate_text(text)
                    predictions.append({
                        "input_text": text,
                        "generated_text": generated_text
                    })
                
                elif self.config.model_type == "sequence_classification":
                    # Classify text
                    inputs = self.tokenizer(
                        text,
                        truncation=True,
                        padding=True,
                        max_length=self.config.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probs = F.softmax(outputs.logits, dim=-1)
                        pred_class = torch.argmax(probs, dim=-1)
                    
                    predictions.append({
                        "text": text,
                        "predicted_class": pred_class.item(),
                        "probabilities": probs.cpu().numpy().tolist()
                    })
                
                elif self.config.model_type == "conditional_generation":
                    # Generate conditional text
                    inputs = self.tokenizer(
                        text,
                        truncation=True,
                        padding=True,
                        max_length=self.config.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=self.config.max_length,
                            do_sample=True,
                            temperature=0.7
                        )
                    
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    predictions.append({
                        "input_text": text,
                        "generated_text": generated_text
                    })
            
            return predictions
        
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return []
    
    def save_model(self, path: str):
        """Save the trained model."""
        try:
            # Save PEFT model if using PEFT
            if self.config.use_peft:
                self.model.save_pretrained(path)
                self.logger.info(f"Saved PEFT model to {path}")
            else:
                # Save full model
                self.model.save_pretrained(path)
                self.logger.info(f"Saved full model to {path}")
            
            # Save tokenizer
            self.tokenizer.save_pretrained(path)
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, path: str):
        """Load a trained model."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            
            # Load model
            if self.config.use_peft:
                self.model = PeftModel.from_pretrained(self.model, path)
            else:
                if self.config.model_type == "causal":
                    self.model = AutoModelForCausalLM.from_pretrained(path)
                elif self.config.model_type == "sequence_classification":
                    self.model = AutoModelForSequenceClassification.from_pretrained(path)
                elif self.config.model_type == "conditional_generation":
                    self.model = T5ForConditionalGeneration.from_pretrained(path)
            
            self.model = self.model.to(self.device)
            self.logger.info(f"Loaded model from {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise


class LLMPipeline:
    """Production-ready LLM pipeline."""
    
    def __init__(self, model_path: str, config: LLMConfig):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.trainer = AdvancedLLMTrainer(config)
        self.trainer.load_model(model_path)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text with the loaded model."""
        return self.trainer.generate_text(prompt, **kwargs)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    
    def classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify texts if using sequence classification model."""
        return self.trainer.predict(texts)


def main():
    """Example usage of advanced LLM integration."""
    
    # Configuration for different model types
    configs = {
        "gpt2": LLMConfig(
            model_name="gpt2",
            model_type="causal",
            task="text_generation",
            max_length=128,
            batch_size=2,
            learning_rate=5e-5,
            num_epochs=1,
            use_peft=True,
            quantization="4bit"
        ),
        "bert": LLMConfig(
            model_name="bert-base-uncased",
            model_type="sequence_classification",
            task="classification",
            max_length=128,
            batch_size=4,
            learning_rate=2e-5,
            num_epochs=1,
            use_peft=True,
            quantization="4bit"
        ),
        "t5": LLMConfig(
            model_name="t5-small",
            model_type="conditional_generation",
            task="summarization",
            max_length=128,
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,
            use_peft=True,
            quantization="4bit"
        )
    }
    
    # Example training data
    train_texts = [
        "This is a positive example for training.",
        "This is a negative example for training.",
        "I love this product and would recommend it.",
        "I hate this product and would not recommend it."
    ]
    
    train_labels = [1, 0, 1, 0]
    
    # Train different model types
    for model_name, config in configs.items():
        try:
            print(f"\n=== Training {model_name.upper()} ===")
            
            trainer = AdvancedLLMTrainer(config)
            trainer_result = trainer.train(train_texts, train_labels)
            
            # Test generation
            if config.model_type == "causal":
                generated = trainer.generate_text("Hello, how are you?")
                print(f"Generated: {generated}")
            
            # Save model
            trainer.save_model(f"./models/{model_name}")
            
            print(f"✅ {model_name} training completed")
            
        except Exception as e:
            print(f"❌ {model_name} training failed: {e}")
    
    # Test pipeline
    try:
        print("\n=== Testing Pipeline ===")
        pipeline = LLMPipeline("./models/gpt2", configs["gpt2"])
        
        test_prompts = [
            "The future of AI is",
            "Machine learning can",
            "Deep learning models"
        ]
        
        results = pipeline.batch_generate(test_prompts, max_length=50)
        
        for prompt, result in zip(test_prompts, results):
            print(f"Prompt: {prompt}")
            print(f"Result: {result}")
            print()
        
        print("✅ Pipeline testing completed")
        
    except Exception as e:
        print(f"❌ Pipeline testing failed: {e}")


match __name__:
    case "__main__":
    main() 
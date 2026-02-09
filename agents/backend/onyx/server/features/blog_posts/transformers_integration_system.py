from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.func import functional_call, vmap, grad
from torch.export import export
from torch._dynamo import optimize
import torch._dynamo as dynamo
    from transformers import (
    from peft import (
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
import structlog
from contextlib import contextmanager
import time
import warnings
import os
import gc
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Any, List, Dict, Optional
"""
🚀 Advanced Transformers Integration System
==========================================

Comprehensive transformers integration with state-of-the-art models,
optimization techniques, and production-ready features.
Integrates seamlessly with the existing Gradio application.
"""


# Modern PyTorch imports

# Transformers imports
try:
        AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
        AutoModelForTokenClassification, AutoModelForQuestionAnswering, AutoModelForMaskedLM,
        TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig,
        pipeline, PreTrainedModel, PreTrainedTokenizer, AutoConfig,
        LlamaTokenizer, LlamaForCausalLM, LlamaConfig,
        MistralTokenizer, MistralForCausalLM, MistralConfig,
        GPT2Tokenizer, GPT2LMHeadModel, GPT2Config,
        T5Tokenizer, T5ForConditionalGeneration, T5Config,
        BertTokenizer, BertForSequenceClassification, BertConfig,
        RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig,
        DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig,
        DebertaTokenizer, DebertaForSequenceClassification, DebertaConfig,
        get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
        AdamW, get_optimizer_grouped_parameters
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers library not available")

# PEFT imports for efficient fine-tuning
try:
        LoraConfig, get_peft_model, TaskType, PeftModel,
        prepare_model_for_kbit_training, PeftConfig, AdaLoraConfig,
        PrefixTuningConfig, PromptTuningConfig, PromptTuningInit
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT library not available")

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
class TransformersConfig:
    """Comprehensive configuration for transformers models."""
    # Model configuration
    model_name: str = "microsoft/DialoGPT-medium"
    model_type: str = "causal"  # causal, sequence_classification, token_classification, question_answering, masked_lm
    task: str = "text_generation"
    
    # Training configuration
    max_length: int = 512
    batch_size: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Precision and optimization
    fp16: bool = True
    bf16: bool = False
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    use_xformers: bool = True
    
    # PEFT configuration
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    peft_method: str = "lora"  # lora, prefix_tuning, prompt_tuning, adalora
    
    # Quantization
    quantization: str = "none"  # none, 4bit, 8bit
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Training arguments
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    
    # Data configuration
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    
    # Deployment
    push_to_hub: bool = False
    report_to: Optional[str] = None
    
    # Generation parameters
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda, mps
    use_multi_gpu: bool = False
    device_ids: Optional[List[int]] = None


class TransformersDataset(Dataset):
    """Custom dataset for transformers models."""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, 
                 tokenizer: PreTrainedTokenizer, max_length: int = 512,
                 model_type: str = "causal"):
        
    """__init__ function."""
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
            # For causal language modeling
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # For causal LM, input_ids and labels are the same
            encoding["labels"] = encoding["input_ids"].clone()
            
        elif self.model_type == "sequence_classification":
            # For sequence classification
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            if self.labels is not None:
                encoding["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                
        elif self.model_type == "token_classification":
            # For token classification (NER, POS tagging, etc.)
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            if self.labels is not None:
                # Convert labels to token-level labels
                token_labels = self._align_labels(text, self.labels[idx])
                encoding["labels"] = torch.tensor(token_labels, dtype=torch.long)
                
        else:
            # Default to causal
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            encoding["labels"] = encoding["input_ids"].clone()
        
        # Remove batch dimension
        for key in encoding:
            encoding[key] = encoding[key].squeeze(0)
            
        return encoding
    
    def _align_labels(self, text: str, label: int) -> List[int]:
        """Align labels with tokenized text for token classification."""
        # This is a simplified version - in practice, you'd need more sophisticated alignment
        tokens = self.tokenizer.tokenize(text)
        return [label] * len(tokens)


class AdvancedTransformersTrainer:
    """Advanced trainer for transformers models with modern optimizations."""
    
    def __init__(self, config: TransformersConfig):
        
    """__init__ function."""
self.config = config
        self.device = self._setup_device()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        logger.info(f"Initializing AdvancedTransformersTrainer with config: {config}")
        
    def _setup_device(self) -> torch.device:
        """Setup device with automatic detection."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using MPS device")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(self.config.device)
            
        return device
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer with error handling."""
        try:
            logger.info(f"Loading tokenizer: {self.config.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            logger.info(f"Tokenizer loaded successfully: {tokenizer.__class__.__name__}")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_model(self) -> PreTrainedModel:
        """Load model with optimizations and error handling."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Setup quantization config
            quantization_config = None
            if self.config.quantization != "none":
                if not TRANSFORMERS_AVAILABLE:
                    raise ImportError("Transformers library required for quantization")
                    
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self.config.load_in_4bit,
                    load_in_8bit=self.config.load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load model based on type
            if self.config.model_type == "causal":
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    device_map="auto" if quantization_config else None,
                    trust_remote_code=True
                )
            elif self.config.model_type == "sequence_classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    device_map="auto" if quantization_config else None,
                    trust_remote_code=True
                )
            elif self.config.model_type == "token_classification":
                model = AutoModelForTokenClassification.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    device_map="auto" if quantization_config else None,
                    trust_remote_code=True
                )
            elif self.config.model_type == "question_answering":
                model = AutoModelForQuestionAnswering.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    device_map="auto" if quantization_config else None,
                    trust_remote_code=True
                )
            elif self.config.model_type == "masked_lm":
                model = AutoModelForMaskedLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    device_map="auto" if quantization_config else None,
                    trust_remote_code=True
                )
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # Apply optimizations
            model = self._apply_optimizations(model)
            
            # Apply PEFT if enabled
            if self.config.use_peft and PEFT_AVAILABLE:
                model = self._apply_peft(model)
            
            logger.info(f"Model loaded successfully: {model.__class__.__name__}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _apply_optimizations(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply various optimizations to the model."""
        try:
            # Enable gradient checkpointing
            if self.config.use_gradient_checkpointing:
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            
            # Enable flash attention if available
            if self.config.use_flash_attention:
                try:
                    for module in model.modules():
                        if hasattr(module, "config") and hasattr(module.config, "use_flash_attention_2"):
                            module.config.use_flash_attention_2 = True
                    logger.info("Flash attention enabled")
                except Exception as e:
                    logger.warning(f"Failed to enable flash attention: {e}")
            
            # Enable xformers if available
            if self.config.use_xformers:
                try:
                    for module in model.modules():
                        if hasattr(module, "config") and hasattr(module.config, "use_xformers"):
                            module.config.use_xformers = True
                    logger.info("XFormers enabled")
                except Exception as e:
                    logger.warning(f"Failed to enable xformers: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to apply optimizations: {e}")
            return model
    
    def _apply_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply PEFT (Parameter-Efficient Fine-Tuning) to the model."""
        try:
            if self.config.peft_method == "lora":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM if self.config.model_type == "causal" else TaskType.SEQ_CLS,
                    inference_mode=False,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self._get_target_modules()
                )
            elif self.config.peft_method == "prefix_tuning":
                peft_config = PrefixTuningConfig(
                    task_type=TaskType.CAUSAL_LM if self.config.model_type == "causal" else TaskType.SEQ_CLS,
                    inference_mode=False,
                    num_virtual_tokens=20
                )
            elif self.config.peft_method == "prompt_tuning":
                peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM if self.config.model_type == "causal" else TaskType.SEQ_CLS,
                    inference_mode=False,
                    num_virtual_tokens=20,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    prompt_tuning_init_text="Classify if this tweet is a complaint or not:"
                )
            elif self.config.peft_method == "adalora":
                peft_config = AdaLoraConfig(
                    task_type=TaskType.CAUSAL_LM if self.config.model_type == "causal" else TaskType.SEQ_CLS,
                    inference_mode=False,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self._get_target_modules()
                )
            else:
                raise ValueError(f"Unsupported PEFT method: {self.config.peft_method}")
            
            # Prepare model for k-bit training if using quantization
            if self.config.quantization != "none":
                model = prepare_model_for_kbit_training(model)
            
            # Apply PEFT
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            
            logger.info(f"PEFT applied successfully: {self.config.peft_method}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to apply PEFT: {e}")
            return model
    
    def _get_target_modules(self) -> List[str]:
        """Get target modules for PEFT based on model architecture."""
        if "llama" in self.config.model_name.lower():
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "mistral" in self.config.model_name.lower():
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt2" in self.config.model_name.lower():
            return ["c_attn", "c_proj", "c_fc", "c_proj"]
        elif "bert" in self.config.model_name.lower():
            return ["query", "key", "value", "dense"]
        else:
            # Default to common attention modules
            return ["q_proj", "v_proj", "k_proj", "o_proj", "query", "key", "value", "dense"]
    
    def prepare_dataset(self, texts: List[str], labels: Optional[List[int]] = None) -> TransformersDataset:
        """Prepare dataset for training."""
        if self.tokenizer is None:
            self.tokenizer = self._load_tokenizer()
        
        return TransformersDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            model_type=self.config.model_type
        )
    
    def train(self, train_texts: List[str], train_labels: Optional[List[int]] = None,
              val_texts: Optional[List[str]] = None, val_labels: Optional[List[int]] = None) -> Dict[str, Any]:
        """Train the model with comprehensive logging and error handling."""
        try:
            logger.info("Starting model training...")
            
            # Load tokenizer and model
            if self.tokenizer is None:
                self.tokenizer = self._load_tokenizer()
            if self.model is None:
                self.model = self._load_model()
            
            # Prepare datasets
            train_dataset = self.prepare_dataset(train_texts, train_labels)
            val_dataset = None
            if val_texts is not None:
                val_dataset = self.prepare_dataset(val_texts, val_labels)
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir="./transformers_output",
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                warmup_steps=self.config.warmup_steps,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                max_grad_norm=self.config.max_grad_norm,
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                dataloader_num_workers=self.config.dataloader_num_workers,
                dataloader_pin_memory=self.config.dataloader_pin_memory,
                remove_unused_columns=self.config.remove_unused_columns,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                logging_steps=self.config.logging_steps,
                save_total_limit=self.config.save_total_limit,
                load_best_model_at_end=self.config.load_best_model_at_end,
                metric_for_best_model=self.config.metric_for_best_model,
                greater_is_better=self.config.greater_is_better,
                evaluation_strategy=self.config.evaluation_strategy,
                save_strategy=self.config.save_strategy,
                push_to_hub=self.config.push_to_hub,
                report_to=self.config.report_to,
                logging_dir="./logs",
                run_name=f"transformers_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Setup trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer
            )
            
            # Train the model
            logger.info("Training started...")
            train_result = self.trainer.train()
            
            # Save the model
            self.trainer.save_model("./transformers_final_model")
            self.tokenizer.save_pretrained("./transformers_final_model")
            
            logger.info("Training completed successfully!")
            return {
                "success": True,
                "train_loss": train_result.training_loss,
                "global_step": train_result.global_step,
                "model_path": "./transformers_final_model"
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the trained model."""
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model and tokenizer must be loaded before generation")
            
            # Use provided kwargs or default config values
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "do_sample": kwargs.get("do_sample", self.config.do_sample),
                "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return f"Error generating text: {str(e)}"
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions on a list of texts."""
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model and tokenizer must be loaded before prediction")
            
            results = []
            
            for text in texts:
                try:
                    # Tokenize
                    inputs = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.config.max_length,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Predict
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Process outputs based on model type
                    if self.config.model_type == "sequence_classification":
                        probs = F.softmax(outputs.logits, dim=-1)
                        predicted_class = torch.argmax(probs, dim=-1).item()
                        confidence = probs.max().item()
                        
                        results.append({
                            "text": text,
                            "predicted_class": predicted_class,
                            "confidence": confidence,
                            "probabilities": probs[0].tolist()
                        })
                    else:
                        results.append({
                            "text": text,
                            "output": outputs.logits.tolist()
                        })
                        
                except Exception as e:
                    logger.error(f"Prediction failed for text '{text[:50]}...': {e}")
                    results.append({
                        "text": text,
                        "error": str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return [{"error": str(e)}] * len(texts)
    
    def save_model(self, path: str):
        """Save the trained model and tokenizer."""
        try:
            if self.model is not None:
                self.model.save_pretrained(path)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(path)
            logger.info(f"Model saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, path: str):
        """Load a saved model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModel.from_pretrained(path)
            self.model.to(self.device)
            logger.info(f"Model loaded from: {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


class TransformersPipeline:
    """High-level pipeline for transformers models."""
    
    def __init__(self, model_path: str, config: TransformersConfig):
        
    """__init__ function."""
self.config = config
        self.trainer = AdvancedTransformersTrainer(config)
        self.trainer.load_model(model_path)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        return self.trainer.generate_text(prompt, **kwargs)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    
    def classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple texts."""
        return self.trainer.predict(texts)


# Utility functions for Gradio integration
def create_transformers_config(model_name: str = "microsoft/DialoGPT-medium",
                              model_type: str = "causal",
                              task: str = "text_generation",
                              **kwargs) -> TransformersConfig:
    """Create a transformers configuration with default values."""
    return TransformersConfig(
        model_name=model_name,
        model_type=model_type,
        task=task,
        **kwargs
    )


def get_available_models() -> Dict[str, List[str]]:
    """Get available model categories and examples."""
    return {
        "causal_lm": [
            "microsoft/DialoGPT-medium",
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-large"
        ],
        "sequence_classification": [
            "bert-base-uncased",
            "distilbert-base-uncased",
            "roberta-base",
            "microsoft/DistilBERT-base-uncased-finetuned-sst-2-english"
        ],
        "token_classification": [
            "bert-base-uncased",
            "distilbert-base-uncased",
            "dbmdz/bert-large-cased-finetuned-conll03-english"
        ],
        "question_answering": [
            "bert-base-uncased",
            "distilbert-base-uncased",
            "deepset/roberta-base-squad2"
        ],
        "masked_lm": [
            "bert-base-uncased",
            "distilbert-base-uncased",
            "roberta-base"
        ]
    }


def validate_transformers_inputs(text: str, model_name: str, max_length: int) -> Tuple[bool, str]:
    """Validate inputs for transformers operations."""
    if not text or not text.strip():
        return False, "Text cannot be empty"
    
    if len(text) > max_length * 4:  # Rough estimate
        return False, f"Text too long (max {max_length * 4} characters)"
    
    if not model_name or not model_name.strip():
        return False, "Model name cannot be empty"
    
    return True, "Inputs are valid"


# Global variables for Gradio integration
transformers_trainer = None
transformers_pipeline = None


def initialize_transformers_system():
    """Initialize the transformers system for Gradio."""
    global transformers_trainer, transformers_pipeline
    
    try:
        config = TransformersConfig()
        transformers_trainer = AdvancedTransformersTrainer(config)
        logger.info("Transformers system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize transformers system: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    config = TransformersConfig(
        model_name="microsoft/DialoGPT-medium",
        model_type="causal",
        task="text_generation"
    )
    
    trainer = AdvancedTransformersTrainer(config)
    
    # Example training data
    train_texts = [
        "Hello, how are you?",
        "What's the weather like today?",
        "Tell me a joke",
        "How do I make coffee?"
    ]
    
    # Train the model
    result = trainer.train(train_texts)
    print(f"Training result: {result}")
    
    # Generate text
    if result["success"]:
        generated_text = trainer.generate_text("Hello, I want to")
        print(f"Generated text: {generated_text}") 
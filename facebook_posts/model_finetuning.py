from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForMaskedLM, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling, DataCollatorWithPadding,
    PreTrainedTokenizer, PreTrainedModel, pipeline, TextGenerationPipeline
)
from transformers.utils import logging as transformers_logging
import numpy as np
import json
import logging
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import time
from enum import Enum
from torch.utils.data import Dataset, DataLoader
            from peft import LoraConfig, get_peft_model, TaskType
            from peft import PrefixTuningConfig, get_peft_model, TaskType
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Model Fine-tuning with Advanced Features
Comprehensive fine-tuning implementation with advanced features and best practices.
"""

    AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForMaskedLM, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling, DataCollatorWithPadding,
    PreTrainedTokenizer, PreTrainedModel, pipeline, TextGenerationPipeline
)


class FinetuningType(Enum):
    """Types of fine-tuning approaches."""
    FULL_FINETUNING = "full_finetuning"
    LORA_FINETUNING = "lora_finetuning"
    QLORA_FINETUNING = "qlora_finetuning"
    PREFIX_TUNING = "prefix_tuning"
    ADAPTER_TUNING = "adapter_tuning"
    PROMPT_TUNING = "prompt_tuning"


@dataclass
class FinetuningConfig:
    """Configuration for model fine-tuning."""
    # Model configuration
    base_model_name: str = "gpt2"
    model_type: str = "causal_lm"
    
    # Fine-tuning configuration
    finetuning_type: FinetuningType = FinetuningType.FULL_FINETUNING
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # LoRA configuration (if using LoRA)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Advanced features
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    
    # Data configuration
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    
    # Output configuration
    output_dir: str = "./finetuned_model"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True


class CustomDataset(Dataset):
    """Custom dataset for fine-tuning."""
    
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer, 
                 max_length: int = 512, task_type: str = "causal_lm"):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
    
    def __len__(self) -> Any:
        return len(self.texts)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        text = self.texts[idx]
        
        # Tokenize based on task type
        if self.task_type == "causal_lm":
            # For causal language modeling
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Remove batch dimension
            item = {key: val.squeeze(0) for key, val in encoding.items()}
            
            # Labels are the same as input_ids for causal LM
            item['labels'] = item['input_ids'].clone()
            
        elif self.task_type == "sequence_classification":
            # For sequence classification
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            item = {key: val.squeeze(0) for key, val in encoding.items()}
            # Labels will be added separately
            
        else:
            # Default tokenization
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        return item


class AdvancedFinetuningManager:
    """Advanced manager for model fine-tuning."""
    
    def __init__(self, config: FinetuningConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Setup logging
        self._setup_logging()
        
        # Load base model and tokenizer
        self._load_base_components()
        
        # Setup fine-tuning
        self._setup_finetuning()
    
    def _setup_logging(self) -> Any:
        """Setup comprehensive logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('finetuning.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        transformers_logging.set_verbosity_info()
    
    def _load_base_components(self) -> Any:
        """Load base model and tokenizer."""
        self.logger.info(f"Loading base model: {self.config.base_model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            model_kwargs = {
                'torch_dtype': torch.float16 if self.config.use_mixed_precision else torch.float32,
                'device_map': "auto" if self.device.type == "cuda" else None
            }
            
            if self.config.use_flash_attention:
                model_kwargs['use_flash_attention_2'] = True
            
            if self.config.model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name, **model_kwargs
                )
            elif self.config.model_type == "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.base_model_name, **model_kwargs
                )
            elif self.config.model_type == "token_classification":
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.config.base_model_name, **model_kwargs
                )
            elif self.config.model_type == "masked_lm":
                self.model = AutoModelForMaskedLM.from_pretrained(
                    self.config.base_model_name, **model_kwargs
                )
            elif self.config.model_type == "question_answering":
                self.model = AutoModelForQuestionAnswering.from_pretrained(
                    self.config.base_model_name, **model_kwargs
                )
            else:
                self.model = AutoModel.from_pretrained(
                    self.config.base_model_name, **model_kwargs
                )
            
            # Enable gradient checkpointing if requested
            if self.config.use_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            self.logger.info("Base components loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading base components: {e}")
            raise
    
    def _setup_finetuning(self) -> Any:
        """Setup fine-tuning configuration."""
        self.logger.info(f"Setting up {self.config.finetuning_type.value} fine-tuning")
        
        try:
            if self.config.finetuning_type == FinetuningType.LORA_FINETUNING:
                self._setup_lora_finetuning()
            elif self.config.finetuning_type == FinetuningType.QLORA_FINETUNING:
                self._setup_qlora_finetuning()
            elif self.config.finetuning_type == FinetuningType.PREFIX_TUNING:
                self._setup_prefix_tuning()
            else:
                # Full fine-tuning (default)
                self.logger.info("Using full fine-tuning approach")
            
        except Exception as e:
            self.logger.error(f"Error setting up fine-tuning: {e}")
            raise
    
    def _setup_lora_finetuning(self) -> Any:
        """Setup LoRA fine-tuning."""
        try:
            
            # Define LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules or ["q_proj", "v_proj"],
                bias="none",
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            self.logger.info("LoRA fine-tuning setup completed")
            
        except ImportError:
            self.logger.warning("PEFT library not available, falling back to full fine-tuning")
        except Exception as e:
            self.logger.error(f"Error setting up LoRA: {e}")
            raise
    
    def _setup_qlora_finetuning(self) -> Any:
        """Setup QLoRA fine-tuning."""
        try:
            
            # Quantize model to 4-bit
            self.model = self.model.quantize(4)
            
            # Define LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules or ["q_proj", "v_proj"],
                bias="none",
            )
            
            # Apply LoRA to quantized model
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            self.logger.info("QLoRA fine-tuning setup completed")
            
        except ImportError:
            self.logger.warning("PEFT library not available, falling back to full fine-tuning")
        except Exception as e:
            self.logger.error(f"Error setting up QLoRA: {e}")
            raise
    
    def _setup_prefix_tuning(self) -> Any:
        """Setup prefix tuning."""
        try:
            
            # Define prefix tuning configuration
            prefix_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=20,
                encoder_hidden_size=128,
            )
            
            # Apply prefix tuning to model
            self.model = get_peft_model(self.model, prefix_config)
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            self.logger.info("Prefix tuning setup completed")
            
        except ImportError:
            self.logger.warning("PEFT library not available, falling back to full fine-tuning")
        except Exception as e:
            self.logger.error(f"Error setting up prefix tuning: {e}")
            raise
    
    def prepare_dataset(self, texts: List[str], labels: Optional[List[int]] = None) -> Dataset:
        """Prepare dataset for fine-tuning."""
        self.logger.info(f"Preparing dataset with {len(texts)} samples")
        
        dataset = CustomDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            task_type=self.config.model_type
        )
        
        # Add labels if provided
        if labels is not None:
            dataset.labels = labels
        
        return dataset
    
    def setup_training(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Setup training with Trainer."""
        self.logger.info("Setting up training with Trainer")
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_epochs,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False if eval_dataset else None,
            fp16=self.config.use_mixed_precision,
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            report_to=None,  # Disable wandb/tensorboard
            remove_unused_columns=False,
        )
        
        # Create data collator
        if self.config.model_type == "causal_lm":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        else:
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        self.logger.info("Training setup completed")
    
    def train(self) -> Dict[str, Any]:
        """Train the model."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_training() first.")
        
        self.logger.info("Starting fine-tuning")
        
        try:
            # Train the model
            train_result = self.trainer.train()
            
            # Save the model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Get training metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            
            self.logger.info(f"Fine-tuning completed. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {e}")
            raise
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, Any]:
        """Evaluate the fine-tuned model."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_training() first.")
        
        self.logger.info("Starting evaluation")
        
        try:
            # Evaluate the model
            metrics = self.trainer.evaluate(eval_dataset=eval_dataset)
            
            self.trainer.log_metrics("eval", metrics)
            self.trainer.save_metrics("eval", metrics)
            
            self.logger.info(f"Evaluation completed. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise
    
    def generate_text(self, prompt: str, **kwargs) -> List[str]:
        """Generate text using the fine-tuned model."""
        self.logger.info(f"Generating text for prompt: {prompt[:50]}...")
        
        # Update generation parameters
        generation_kwargs = {
            'max_length': self.config.max_length,
            'temperature': 1.0,
            'top_p': 0.9,
            'do_sample': True,
            'num_return_sequences': 1,
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }
        generation_kwargs.update(kwargs)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            
            # Decode outputs
            generated_texts = []
            for output in outputs:
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                generated_texts.append(generated_text)
            
            self.logger.info(f"Generated {len(generated_texts)} sequences")
            return generated_texts
            
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            raise
    
    def save_model(self, path: str = None):
        """Save the fine-tuned model."""
        if path is None:
            path = self.config.output_dir
        
        self.logger.info(f"Saving model to {path}")
        
        try:
            # Save model
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            # Save configuration
            config_dict = {
                'base_model_name': self.config.base_model_name,
                'model_type': self.config.model_type,
                'finetuning_type': self.config.finetuning_type.value,
                'max_length': self.config.max_length,
                'vocab_size': self.tokenizer.vocab_size
            }
            
            with open(f"{path}/finetuning_config.json", 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info("Model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, path: str):
        """Load a fine-tuned model."""
        self.logger.info(f"Loading model from {path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            
            # Load model based on type
            if self.config.model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(path)
            elif self.config.model_type == "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(path)
            else:
                self.model = AutoModel.from_pretrained(path)
            
            self.model.to(self.device)
            
            # Load configuration
            config_path = f"{path}/finetuning_config.json"
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                self.logger.info(f"Loaded configuration: {config_dict}")
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            'base_model_name': self.config.base_model_name,
            'model_type': self.config.model_type,
            'finetuning_type': self.config.finetuning_type.value,
            'device': str(self.device),
            'tokenizer_class': self.tokenizer.__class__.__name__ if self.tokenizer else None,
            'model_class': self.model.__class__.__name__ if self.model else None,
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else None,
            'model_size': sum(p.numel() for p in self.model.parameters()) if self.model else None,
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else None,
        }
        
        return info


def demonstrate_finetuning():
    """Demonstrate model fine-tuning capabilities."""
    print("Model Fine-tuning Demonstration")
    print("=" * 40)
    
    # Sample training data
    training_texts = [
        "The future of artificial intelligence is bright and promising.",
        "Machine learning algorithms are transforming industries worldwide.",
        "Deep learning models can process vast amounts of data efficiently.",
        "Natural language processing enables human-computer interaction.",
        "Computer vision systems can recognize objects in images.",
        "Reinforcement learning agents learn through trial and error.",
        "Neural networks mimic the human brain's structure and function.",
        "Transformers have revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Transfer learning enables models to adapt to new tasks quickly."
    ]
    
    # Test different fine-tuning configurations
    configs = [
        FinetuningConfig(
            base_model_name="gpt2",
            model_type="causal_lm",
            finetuning_type=FinetuningType.FULL_FINETUNING,
            num_epochs=1,
            batch_size=2
        ),
        FinetuningConfig(
            base_model_name="gpt2",
            model_type="causal_lm",
            finetuning_type=FinetuningType.LORA_FINETUNING,
            num_epochs=1,
            batch_size=2
        )
    ]
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\nTesting {config.finetuning_type.value} fine-tuning:")
        
        try:
            # Create manager
            manager = AdvancedFinetuningManager(config)
            
            # Get model info
            model_info = manager.get_model_info()
            print(f"  Base model: {model_info['base_model_name']}")
            print(f"  Model size: {model_info['model_size']:,} parameters")
            print(f"  Trainable parameters: {model_info['trainable_parameters']:,}")
            
            # Prepare dataset
            train_dataset = manager.prepare_dataset(training_texts)
            eval_dataset = manager.prepare_dataset(training_texts[:2])  # Small eval set
            
            # Setup training
            manager.setup_training(train_dataset, eval_dataset)
            
            # Train model
            training_metrics = manager.train()
            print(f"  Training loss: {training_metrics.get('train_loss', 'N/A')}")
            
            # Evaluate model
            eval_metrics = manager.evaluate(eval_dataset)
            print(f"  Evaluation loss: {eval_metrics.get('eval_loss', 'N/A')}")
            
            # Test generation
            test_prompt = "The future of AI"
            generated_texts = manager.generate_text(test_prompt, max_length=50)
            print(f"  Test generation: '{generated_texts[0][:100]}...'")
            
            # Save model
            save_path = f"./finetuned_model_{i}"
            manager.save_model(save_path)
            
            results[f"finetuning_{i}"] = {
                'config': config,
                'model_info': model_info,
                'training_metrics': training_metrics,
                'eval_metrics': eval_metrics,
                'generated_text': generated_texts[0],
                'success': True
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[f"finetuning_{i}"] = {
                'config': config,
                'error': str(e),
                'success': False
            }
    
    return results


if __name__ == "__main__":
    # Demonstrate fine-tuning
    results = demonstrate_finetuning()
    print("\nFine-tuning demonstration completed!") 
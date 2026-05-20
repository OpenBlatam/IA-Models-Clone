"""
Advanced Fine-tuning
LoRA, P-tuning, and efficient fine-tuning techniques
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
        prepare_model_for_kbit_training
    )
    from transformers import (
        AutoModel,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class LoRAFineTuner:
    """
    LoRA (Low-Rank Adaptation) fine-tuning for efficient model adaptation
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: str = "FEATURE_EXTRACTION",
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
    ):
        if not PEFT_AVAILABLE:
            raise ImportError("peft library required for LoRA fine-tuning")
        
        self.model_name = model_name
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure LoRA
        if task_type == "FEATURE_EXTRACTION":
            task = TaskType.FEATURE_EXTRACTION
        elif task_type == "CLASSIFICATION":
            task = TaskType.SEQ_CLS
        else:
            task = TaskType.FEATURE_EXTRACTION
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type=task,
            target_modules=target_modules or ["query", "value"]
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        logger.info(f"Initialized LoRA fine-tuner with r={r}, alpha={lora_alpha}")
    
    def prepare_for_training(self, use_8bit: bool = False):
        """Prepare model for training"""
        if use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        return self.model
    
    def train(
        self,
        train_dataset,
        val_dataset: Optional[Any] = None,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-4,
        output_dir: str = "./lora_output"
    ) -> Dict[str, Any]:
        """Fine-tune model with LoRA"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            fp16=True,  # Mixed precision
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )
        
        train_result = trainer.train()
        
        return {
            "train_loss": train_result.training_loss,
            "global_step": train_result.global_step,
            "model_path": output_dir
        }
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights"""
        self.model.save_pretrained(path)
        logger.info(f"Saved LoRA weights to {path}")
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights"""
        self.model = PeftModel.from_pretrained(self.base_model, path)
        logger.info(f"Loaded LoRA weights from {path}")


class PTuningFineTuner:
    """
    P-tuning for efficient prompt-based fine-tuning
    """
    
    def __init__(
        self,
        model_name: str,
        num_virtual_tokens: int = 20,
        encoder_hidden_size: int = 128
    ):
        if not PEFT_AVAILABLE:
            raise ImportError("peft library required for P-tuning")
        
        from peft import PromptTuningConfig, get_peft_model
        
        self.model_name = model_name
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure P-tuning
        peft_config = PromptTuningConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=encoder_hidden_size
        )
        
        self.model = get_peft_model(self.base_model, peft_config)
        logger.info(f"Initialized P-tuning with {num_virtual_tokens} virtual tokens")
    
    def train(
        self,
        train_dataset,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        output_dir: str = "./ptuning_output"
    ) -> Dict[str, Any]:
        """Fine-tune with P-tuning"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_steps=100
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )
        
        train_result = trainer.train()
        
        return {
            "train_loss": train_result.training_loss,
            "global_step": train_result.global_step,
            "model_path": output_dir
        }


class AdvancedTrainingOptimizer:
    """
    Advanced training optimizations:
    - Gradient accumulation
    - Mixed precision training
    - Learning rate scheduling
    - Gradient clipping
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        accumulation_steps: int = 1,
        use_amp: bool = True,
        max_grad_norm: float = 1.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm
        
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.step_count = 0
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module
    ) -> float:
        """Perform one training step with optimizations"""
        self.model.train()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss = loss / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            loss = loss / self.accumulation_steps
            loss.backward()
        
        self.step_count += 1
        
        # Gradient accumulation
        if self.step_count % self.accumulation_steps == 0:
            if self.use_amp:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
        
        return loss.item() * self.accumulation_steps


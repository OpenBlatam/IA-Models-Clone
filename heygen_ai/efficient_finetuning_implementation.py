from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
from transformers import (
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
    from peft import get_peft_model, LoraConfig, TaskType, PromptTuningConfig, PromptEncoderConfig
    import numpy as np
    from torch.utils.data import Dataset
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Efficient Fine-Tuning Implementation: LoRA and P-tuning
Comprehensive implementation for parameter-efficient fine-tuning of transformers.
"""

    AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorWithPadding
)

try:
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("peft library not found. Please install with 'pip install peft'.")

logger = logging.getLogger(__name__)

@dataclass
class EfficientFinetuneConfig:
    """Configuration for efficient fine-tuning."""
    model_name: str = "gpt2"
    task_type: str = "causal_lm"  # causal_lm, seq_cls, token_cls
    method: str = "lora"  # lora, p_tuning, prompt_tuning
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    prompt_length: int = 20
    prompt_encoder_hidden_size: int = 768
    num_labels: int = 2
    max_length: int = 128
    batch_size: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    output_dir: str = "./efficient_finetune_output"
    device: str = "auto"

class EfficientFinetuner:
    """Efficient fine-tuning manager supporting LoRA and P-tuning."""
    def __init__(self, config: EfficientFinetuneConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self._initialize_model_and_tokenizer()

    def _initialize_model_and_tokenizer(self) -> Any:
        logger.info(f"Loading model: {self.config.model_name}")
        if self.config.task_type == "causal_lm":
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        elif self.config.task_type == "seq_cls":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name, num_labels=self.config.num_labels
            )
        else:
            raise ValueError(f"Unsupported task_type: {self.config.task_type}")
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def apply_lora(self) -> Any:
        if not PEFT_AVAILABLE:
            raise ImportError("peft library is required for LoRA.")
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM if self.config.task_type == "causal_lm" else TaskType.SEQ_CLS
        )
        self.peft_model = get_peft_model(self.model, lora_config)
        logger.info("LoRA applied to model.")
        return self.peft_model

    def apply_prompt_tuning(self) -> Any:
        if not PEFT_AVAILABLE:
            raise ImportError("peft library is required for prompt tuning.")
        prompt_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM if self.config.task_type == "causal_lm" else TaskType.SEQ_CLS,
            num_virtual_tokens=self.config.prompt_length
        )
        self.peft_model = get_peft_model(self.model, prompt_config)
        logger.info("Prompt tuning applied to model.")
        return self.peft_model

    def apply_p_tuning(self) -> Any:
        if not PEFT_AVAILABLE:
            raise ImportError("peft library is required for P-tuning.")
        p_tuning_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM if self.config.task_type == "causal_lm" else TaskType.SEQ_CLS,
            num_virtual_tokens=self.config.prompt_length,
            encoder_hidden_size=self.config.prompt_encoder_hidden_size
        )
        self.peft_model = get_peft_model(self.model, p_tuning_config)
        logger.info("P-tuning applied to model.")
        return self.peft_model

    def prepare_trainer(self, train_dataset, eval_dataset=None) -> Any:
        """Prepare HuggingFace Trainer for fine-tuning."""
        model = self.peft_model if self.peft_model is not None else self.model
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            evaluation_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch",
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            fp16=torch.cuda.is_available(),
            report_to=["none"]
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        return trainer

    def finetune(self, train_dataset, eval_dataset=None) -> Any:
        """Run efficient fine-tuning with the selected method."""
        if self.config.method == "lora":
            self.apply_lora()
        elif self.config.method == "prompt_tuning":
            self.apply_prompt_tuning()
        elif self.config.method == "p_tuning":
            self.apply_p_tuning()
        else:
            raise ValueError(f"Unsupported method: {self.config.method}")
        trainer = self.prepare_trainer(train_dataset, eval_dataset)
        trainer.train()
        logger.info("Fine-tuning completed.")
        return trainer

class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss does not improve.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True):
        
    """__init__ function."""
self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss: float, model: torch.nn.Module):
        
    """__call__ function."""
score = -val_loss
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_state_dict is not None:
                    model.load_state_dict(self.best_state_dict)

    def reset(self) -> Any:
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None


def get_lr_scheduler(optimizer, scheduler_type: str = "plateau", **kwargs):
    """
    Create a learning rate scheduler for the optimizer.
    Supported types: 'plateau', 'cosine', 'step'.
    """
    if scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode='min', factor=kwargs.get('factor', 0.5), patience=kwargs.get('patience', 3), min_lr=kwargs.get('min_lr', 1e-7))
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 10), eta_min=kwargs.get('eta_min', 1e-7))
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=kwargs.get('step_size', 10), gamma=kwargs.get('gamma', 0.1))
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0):
    """
    Clip gradients to prevent exploding gradients.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def has_nan_or_inf(tensor: torch.Tensor) -> bool:
    """
    Check if a tensor contains NaN or Inf values.
    """
    return not torch.isfinite(tensor).all().item()

def safe_backward(loss: torch.Tensor, model: torch.nn.Module, max_grad_norm: float = 1.0) -> bool:
    """
    Perform backward pass with gradient clipping and NaN/Inf checks.
    Returns True if gradients are valid, False otherwise.
    """
    if has_nan_or_inf(loss):
        print("[Warning] Loss is NaN or Inf. Skipping backward.")
        return False
    loss.backward()
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None and has_nan_or_inf(param.grad):
            print(f"[Warning] Gradient for {name} is NaN or Inf. Zeroing gradients.")
            param.grad.zero_()
            return False
    clip_gradients(model, max_grad_norm)
    return True

# Example usage and demonstration
def demonstrate_efficient_finetuning():
    """Demonstrate efficient fine-tuning with LoRA and P-tuning."""

    class DummyDataset(Dataset):
        def __init__(self, tokenizer, size=100, max_length=32) -> Any:
            self.tokenizer = tokenizer
            self.size = size
            self.max_length = max_length
        def __len__(self) -> Any:
            return self.size
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            text = f"Sample input {idx}"
            label = idx % 2
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            item = {k: v.squeeze(0) for k, v in encoding.items()}
            item["labels"] = torch.tensor(label, dtype=torch.long)
            return item

    # Configuration for LoRA fine-tuning
    config = EfficientFinetuneConfig(
        model_name="distilbert-base-uncased",
        task_type="seq_cls",
        method="lora",
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        num_labels=2,
        batch_size=4,
        num_train_epochs=1,
        output_dir="./efficient_finetune_output"
    )
    
    finetuner = EfficientFinetuner(config)
    train_dataset = DummyDataset(finetuner.tokenizer, size=32, max_length=32)
    eval_dataset = DummyDataset(finetuner.tokenizer, size=8, max_length=32)
    
    print("Starting LoRA fine-tuning demo...")
    trainer = finetuner.finetune(train_dataset, eval_dataset)
    print("LoRA fine-tuning completed.")
    
    # Configuration for P-tuning
    config_p = EfficientFinetuneConfig(
        model_name="distilbert-base-uncased",
        task_type="seq_cls",
        method="p_tuning",
        prompt_length=8,
        prompt_encoder_hidden_size=128,
        num_labels=2,
        batch_size=4,
        num_train_epochs=1,
        output_dir="./efficient_finetune_output_p"
    )
    
    finetuner_p = EfficientFinetuner(config_p)
    train_dataset_p = DummyDataset(finetuner_p.tokenizer, size=32, max_length=32)
    eval_dataset_p = DummyDataset(finetuner_p.tokenizer, size=8, max_length=32)
    
    print("Starting P-tuning demo...")
    trainer_p = finetuner_p.finetune(train_dataset_p, eval_dataset_p)
    print("P-tuning completed.")

def demonstrate_early_stopping_and_scheduler():
    """
    Demonstrate early stopping, learning rate scheduling, gradient clipping, NaN/Inf handling, and autograd anomaly detection in a training loop.
    """
    # Dummy model, optimizer, and data
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = get_lr_scheduler(optimizer, scheduler_type="plateau", patience=2)
    early_stopper = EarlyStopping(patience=3, min_delta=0.01)

    # Simulate training loop
    np.random.seed(42)
    val_losses = np.abs(np.random.randn(20))  # Simulated validation losses

    for epoch, val_loss in enumerate(val_losses):
        optimizer.zero_grad()
        dummy_loss = torch.tensor(val_loss, requires_grad=True)
        # Simulate occasional NaN loss
        if epoch == 5:
            dummy_loss = torch.tensor(float('nan'), requires_grad=True)
        # Use autograd anomaly detection for debugging
        try:
            with torch.autograd.detect_anomaly():
                valid_grad = safe_backward(dummy_loss, model, max_grad_norm=0.5)
        except RuntimeError as e:
            print(f"[Autograd Anomaly] RuntimeError at epoch {epoch+1}: {e}")
            valid_grad = False
        if valid_grad:
            optimizer.step()
        else:
            print(f"[Warning] Skipped optimizer step at epoch {epoch+1} due to invalid gradients.")

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1:2d}: val_loss={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print("\n[Info] If you see a traceback above, autograd.detect_anomaly() has caught a backward error or NaN/Inf in the computation graph.")

def demonstrate_gradient_accumulation():
    """
    Demonstrate gradient accumulation for large batch sizes.
    Accumulates gradients over several steps before optimizer.step().
    Integrates with early stopping, scheduler, and anomaly detection.
    """
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = get_lr_scheduler(optimizer, scheduler_type="plateau", patience=2)
    early_stopper = EarlyStopping(patience=3, min_delta=0.01)

    np.random.seed(42)
    val_losses = np.abs(np.random.randn(20))  # Simulated validation losses
    accumulation_steps = 4  # Simulate accumulating gradients over 4 mini-batches
    total_steps = len(val_losses)
    print(f"[Info] Using gradient accumulation: {accumulation_steps} steps per optimizer update.")

    optimizer.zero_grad()
    for epoch, val_loss in enumerate(val_losses):
        dummy_loss = torch.tensor(val_loss, requires_grad=True)
        # Simulate occasional NaN loss
        if epoch == 5:
            dummy_loss = torch.tensor(float('nan'), requires_grad=True)
        try:
            with torch.autograd.detect_anomaly():
                # Scale loss for accumulation
                loss = dummy_loss / accumulation_steps
                valid_grad = safe_backward(loss, model, max_grad_norm=0.5)
        except RuntimeError as e:
            print(f"[Autograd Anomaly] RuntimeError at epoch {epoch+1}: {e}")
            valid_grad = False
        if not valid_grad:
            print(f"[Warning] Skipped optimizer step at epoch {epoch+1} due to invalid gradients.")
            optimizer.zero_grad()
            continue
        # Only step optimizer every accumulation_steps
        if (epoch + 1) % accumulation_steps == 0 or (epoch + 1) == total_steps:
            optimizer.step()
            optimizer.zero_grad()
            print(f"[Info] Optimizer step at epoch {epoch+1}")
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1:2d}: val_loss={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    print("\n[Info] Gradient accumulation demonstration complete.")

def demonstrate_mixed_precision_training():
    """
    Demonstrate mixed precision training with torch.cuda.amp, including gradient accumulation, early stopping, and scheduler.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(10, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = get_lr_scheduler(optimizer, scheduler_type="plateau", patience=2)
    early_stopper = EarlyStopping(patience=3, min_delta=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    np.random.seed(42)
    val_losses = np.abs(np.random.randn(20))  # Simulated validation losses
    accumulation_steps = 4
    total_steps = len(val_losses)
    print(f"[Info] Using mixed precision and gradient accumulation: {accumulation_steps} steps per optimizer update.")

    optimizer.zero_grad()
    for epoch, val_loss in enumerate(val_losses):
        dummy_input = torch.randn(2, 10, device=device)
        dummy_target = torch.randn(2, 1, device=device)
        # Simulate occasional NaN loss
        if epoch == 5:
            dummy_loss = torch.tensor(float('nan'), requires_grad=True, device=device)
        else:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output = model(dummy_input)
                dummy_loss = torch.nn.functional.mse_loss(output, dummy_target) + val_loss * 0.0
        try:
            with torch.autograd.detect_anomaly():
                loss = dummy_loss / accumulation_steps
                if has_nan_or_inf(loss):
                    print(f"[Warning] Loss is NaN or Inf at epoch {epoch+1}. Skipping backward.")
                    optimizer.zero_grad()
                    continue
                scaler.scale(loss).backward()
        except RuntimeError as e:
            print(f"[Autograd Anomaly] RuntimeError at epoch {epoch+1}: {e}")
            optimizer.zero_grad()
            continue
        # Only step optimizer every accumulation_steps
        if (epoch + 1) % accumulation_steps == 0 or (epoch + 1) == total_steps:
            scaler.unscale_(optimizer)
            clip_gradients(model, max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            print(f"[Info] Optimizer step at epoch {epoch+1}")
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1:2d}: val_loss={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    print("\n[Info] Mixed precision training demonstration complete.")

if __name__ == "__main__":
    demonstrate_efficient_finetuning()
    demonstrate_early_stopping_and_scheduler()
    demonstrate_gradient_accumulation()
    demonstrate_mixed_precision_training() 
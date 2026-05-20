"""
Efficient Fine-tuning Techniques System
Comprehensive implementation of LoRA, P-tuning, and other efficient fine-tuning methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizer
)
from peft import (
    LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig,
    PromptTuningConfig, PromptTuningInit, PromptEmbedding,
    PrefixTuningConfig, PrefixEncoder, PromptEncoder
)
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
import time
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import math


@dataclass
class FineTuningConfig:
    """Configuration for efficient fine-tuning techniques."""
    
    # Model settings
    base_model_name: str = "gpt2"
    task_type: str = "CAUSAL_LM"  # CAUSAL_LM, SEQ_CLS, TOKEN_CLASSIFICATION
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None  # ["q_proj", "v_proj"] for GPT-2
    
    # P-tuning settings
    use_p_tuning: bool = False
    num_virtual_tokens: int = 20
    encoder_hidden_size: int = 128
    encoder_num_layers: int = 2
    
    # Prefix tuning settings
    use_prefix_tuning: bool = False
    num_prefix_tokens: int = 10
    prefix_projection: bool = True
    
    # Training settings
    learning_rate: float = 2e-4
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 512
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Evaluation settings
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10


class LoRALayer(nn.Module):
    """Low-Rank Adaptation (LoRA) layer implementation."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        bias: bool = False
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        
        # LoRA low-rank matrices
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Scaling factor
        self.scaling = alpha / r
        
        # Initialize LoRA weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LoRA weights."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer."""
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.original_layer = original_layer
        self.lora_layer = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            r, alpha, dropout
        )
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining original and LoRA weights."""
        original_output = self.original_layer(x)
        lora_output = self.lora_layer(x)
        return original_output + lora_output


class PromptEmbedding(nn.Module):
    """Prompt embedding for P-tuning."""
    
    def __init__(
        self,
        num_virtual_tokens: int,
        hidden_size: int,
        embedding_dim: int = 128
    ):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        
        # Prompt embeddings
        self.prompt_embeddings = nn.Embedding(num_virtual_tokens, embedding_dim)
        
        # MLP to project to hidden size
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize prompt embeddings."""
        nn.init.normal_(self.prompt_embeddings.weight, std=0.02)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """Generate prompt embeddings."""
        prompt_ids = torch.arange(self.num_virtual_tokens, device=self.prompt_embeddings.weight.device)
        prompt_ids = prompt_ids.unsqueeze(0).expand(batch_size, -1)
        
        prompt_embeds = self.prompt_embeddings(prompt_ids)
        prompt_embeds = self.mlp(prompt_embeds)
        
        return prompt_embeds


class PrefixEncoder(nn.Module):
    """Prefix encoder for prefix tuning."""
    
    def __init__(
        self,
        num_prefix_tokens: int,
        hidden_size: int,
        num_layers: int = 2,
        prefix_projection: bool = True
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.hidden_size = hidden_size
        self.prefix_projection = prefix_projection
        
        # Prefix embeddings
        self.prefix_embeddings = nn.Embedding(num_prefix_tokens, hidden_size)
        
        if prefix_projection:
            # MLP for prefix projection
            self.prefix_projection = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size)
            )
        else:
            self.prefix_projection = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize prefix embeddings."""
        nn.init.normal_(self.prefix_embeddings.weight, std=0.02)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """Generate prefix embeddings."""
        prefix_ids = torch.arange(self.num_prefix_tokens, device=self.prefix_embeddings.weight.device)
        prefix_ids = prefix_ids.unsqueeze(0).expand(batch_size, -1)
        
        prefix_embeds = self.prefix_embeddings(prefix_ids)
        
        if self.prefix_projection is not None:
            prefix_embeds = self.prefix_projection(prefix_embeds)
        
        return prefix_embeds


class EfficientFineTuningManager:
    """Manager for efficient fine-tuning techniques."""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        # Load base model and tokenizer
        self._load_base_model()
        
        # Apply fine-tuning technique
        self._apply_fine_tuning_technique()
        
        logging.info(f"Efficient fine-tuning manager initialized with {config.base_model_name}")
    
    def _load_base_model(self):
        """Load base model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            if self.config.task_type == "CAUSAL_LM":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            elif self.config.task_type == "SEQ_CLS":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.base_model_name,
                    num_labels=2,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            logging.info(f"Successfully loaded {self.config.base_model_name}")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def _apply_fine_tuning_technique(self):
        """Apply the selected fine-tuning technique."""
        if self.config.use_lora:
            self._apply_lora()
        elif self.config.use_p_tuning:
            self._apply_p_tuning()
        elif self.config.use_prefix_tuning:
            self._apply_prefix_tuning()
        else:
            logging.warning("No fine-tuning technique selected")
    
    def _apply_lora(self):
        """Apply LoRA fine-tuning."""
        if self.config.lora_target_modules is None:
            # Default target modules for GPT-2
            self.config.lora_target_modules = ["c_attn", "c_proj"]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM if self.config.task_type == "CAUSAL_LM" else TaskType.SEQ_CLS,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            inference_mode=False
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        logging.info("LoRA applied successfully")
    
    def _apply_p_tuning(self):
        """Apply P-tuning fine-tuning."""
        prompt_tuning_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM if self.config.task_type == "CAUSAL_LM" else TaskType.SEQ_CLS,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=self.config.num_virtual_tokens,
            encoder_hidden_size=self.config.encoder_hidden_size,
            encoder_num_layers=self.config.encoder_num_layers
        )
        
        self.peft_model = get_peft_model(self.model, prompt_tuning_config)
        self.peft_model.print_trainable_parameters()
        
        logging.info("P-tuning applied successfully")
    
    def _apply_prefix_tuning(self):
        """Apply prefix tuning fine-tuning."""
        prefix_tuning_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM if self.config.task_type == "CAUSAL_LM" else TaskType.SEQ_CLS,
            num_virtual_tokens=self.config.num_prefix_tokens,
            encoder_hidden_size=self.config.encoder_hidden_size,
            prefix_projection=self.config.prefix_projection
        )
        
        self.peft_model = get_peft_model(self.model, prefix_tuning_config)
        self.peft_model.print_trainable_parameters()
        
        logging.info("Prefix tuning applied successfully")
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get information about trainable parameters."""
        trainable_params = 0
        all_param = 0
        
        for _, param in self.peft_model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            "trainable_params": trainable_params,
            "all_params": all_param,
            "trainable_percentage": 100 * trainable_params / all_param
        }


class FineTuningDataset(Dataset):
    """Dataset for fine-tuning."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        task_type: str = "CAUSAL_LM"
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        if self.task_type == "CAUSAL_LM":
            # For causal language modeling
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": encoding["input_ids"].flatten()
            }
        
        elif self.task_type == "SEQ_CLS":
            # For sequence classification (assuming binary classification)
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # For demonstration, assign random labels
            label = torch.randint(0, 2, (1,)).item()
            
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.long)
            }


class FineTuningTrainer:
    """Trainer for efficient fine-tuning."""
    
    def __init__(self, manager: EfficientFineTuningManager, config: FineTuningConfig):
        self.manager = manager
        self.config = config
        self.device = manager.device
        
        # Setup training arguments
        self.training_args = TrainingArguments(
            output_dir="./fine_tuning_output",
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            evaluation_strategy="steps",
            eval_steps=config.eval_steps,
            save_strategy="steps",
            save_steps=config.save_steps,
            logging_steps=config.logging_steps,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False
        )
    
    def train(
        self,
        train_texts: List[str],
        eval_texts: List[str] = None
    ) -> Dict[str, Any]:
        """Train the model using efficient fine-tuning."""
        # Create datasets
        train_dataset = FineTuningDataset(
            train_texts, self.manager.tokenizer, self.config.max_length, self.config.task_type
        )
        
        eval_dataset = None
        if eval_texts:
            eval_dataset = FineTuningDataset(
                eval_texts, self.manager.tokenizer, self.config.max_length, self.config.task_type
            )
        
        # Create data collator
        if self.config.task_type == "CAUSAL_LM":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.manager.tokenizer,
                mlm=False
            )
        else:
            data_collator = DataCollatorWithPadding(
                tokenizer=self.manager.tokenizer
            )
        
        # Create trainer
        trainer = Trainer(
            model=self.manager.peft_model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.manager.tokenizer
        )
        
        # Train
        logging.info("Starting fine-tuning...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model("./fine_tuned_model")
        self.manager.tokenizer.save_pretrained("./fine_tuned_model")
        
        logging.info("Fine-tuning completed!")
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0)
        }
    
    def evaluate(self, eval_texts: List[str]) -> Dict[str, float]:
        """Evaluate the fine-tuned model."""
        eval_dataset = FineTuningDataset(
            eval_texts, self.manager.tokenizer, self.config.max_length, self.config.task_type
        )
        
        if self.config.task_type == "CAUSAL_LM":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.manager.tokenizer,
                mlm=False
            )
        else:
            data_collator = DataCollatorWithPadding(
                tokenizer=self.manager.tokenizer
            )
        
        trainer = Trainer(
            model=self.manager.peft_model,
            args=self.training_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        eval_result = trainer.evaluate()
        return eval_result


class FineTuningAnalyzer:
    """Analyze fine-tuning performance and efficiency."""
    
    def __init__(self):
        self.results = {}
    
    def compare_techniques(
        self,
        techniques: List[str],
        train_texts: List[str],
        eval_texts: List[str],
        base_model_name: str = "gpt2"
    ) -> Dict[str, Dict[str, Any]]:
        """Compare different fine-tuning techniques."""
        comparison_results = {}
        
        for technique in techniques:
            logging.info(f"Testing {technique}...")
            
            # Configure technique
            config = FineTuningConfig(base_model_name=base_model_name)
            
            if technique == "LoRA":
                config.use_lora = True
                config.use_p_tuning = False
                config.use_prefix_tuning = False
            elif technique == "P-tuning":
                config.use_lora = False
                config.use_p_tuning = True
                config.use_prefix_tuning = False
            elif technique == "Prefix-tuning":
                config.use_lora = False
                config.use_p_tuning = False
                config.use_prefix_tuning = True
            elif technique == "Full-fine-tuning":
                config.use_lora = False
                config.use_p_tuning = False
                config.use_prefix_tuning = False
            
            # Create manager and trainer
            manager = EfficientFineTuningManager(config)
            trainer = FineTuningTrainer(manager, config)
            
            # Get parameter info
            param_info = manager.get_trainable_parameters()
            
            # Train and evaluate
            start_time = time.time()
            train_results = trainer.train(train_texts, eval_texts)
            training_time = time.time() - start_time
            
            eval_results = trainer.evaluate(eval_texts)
            
            comparison_results[technique] = {
                "trainable_params": param_info["trainable_params"],
                "trainable_percentage": param_info["trainable_percentage"],
                "training_time": training_time,
                "train_loss": train_results["train_loss"],
                "eval_loss": eval_results.get("eval_loss", 0),
                "train_samples_per_second": train_results["train_samples_per_second"]
            }
        
        return comparison_results
    
    def visualize_comparison(self, comparison_results: Dict[str, Dict[str, Any]]):
        """Visualize comparison results."""
        techniques = list(comparison_results.keys())
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Trainable parameters comparison
        trainable_params = [comparison_results[t]["trainable_params"] for t in techniques]
        axes[0, 0].bar(techniques, trainable_params)
        axes[0, 0].set_title("Trainable Parameters")
        axes[0, 0].set_ylabel("Number of Parameters")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Training time comparison
        training_times = [comparison_results[t]["training_time"] for t in techniques]
        axes[0, 1].bar(techniques, training_times)
        axes[0, 1].set_title("Training Time")
        axes[0, 1].set_ylabel("Time (seconds)")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Training loss comparison
        train_losses = [comparison_results[t]["train_loss"] for t in techniques]
        axes[1, 0].bar(techniques, train_losses)
        axes[1, 0].set_title("Training Loss")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Training speed comparison
        speeds = [comparison_results[t]["train_samples_per_second"] for t in techniques]
        axes[1, 1].bar(techniques, speeds)
        axes[1, 1].set_title("Training Speed")
        axes[1, 1].set_ylabel("Samples/Second")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def print_comparison_summary(self, comparison_results: Dict[str, Dict[str, Any]]):
        """Print comparison summary."""
        print("\n=== Fine-tuning Techniques Comparison ===")
        print(f"{'Technique':<15} {'Params':<12} {'% Trainable':<12} {'Time(s)':<10} {'Loss':<8} {'Speed':<10}")
        print("-" * 80)
        
        for technique, results in comparison_results.items():
            print(f"{technique:<15} {results['trainable_params']:<12,} "
                  f"{results['trainable_percentage']:<12.2f} "
                  f"{results['training_time']:<10.2f} "
                  f"{results['train_loss']:<8.4f} "
                  f"{results['train_samples_per_second']:<10.2f}")


class ModelGenerator:
    """Generate text using fine-tuned models."""
    
    def __init__(self, manager: EfficientFineTuningManager):
        self.manager = manager
        self.device = manager.device
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate text using the fine-tuned model."""
        # Tokenize input
        inputs = self.manager.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.manager.peft_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.manager.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 0.7
    ) -> List[str]:
        """Generate text for multiple prompts."""
        generated_texts = []
        
        for prompt in prompts:
            generated_text = self.generate_text(prompt, max_length, temperature)
            generated_texts.append(generated_text)
        
        return generated_texts


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Efficient Fine-tuning Techniques Demonstration ===\n")
    
    # Sample training data
    train_texts = [
        "The transformer model is a powerful architecture for natural language processing.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Fine-tuning adapts pre-trained models to specific tasks and domains.",
        "Efficient fine-tuning techniques reduce computational requirements.",
        "LoRA uses low-rank adaptation to reduce trainable parameters.",
        "P-tuning learns continuous prompts for better task adaptation.",
        "Prefix tuning prepends learnable tokens to the input sequence.",
        "Parameter-efficient methods maintain performance while reducing costs."
    ]
    
    eval_texts = [
        "The model demonstrates excellent performance on the evaluation set.",
        "Efficient fine-tuning achieves comparable results to full fine-tuning.",
        "Parameter sharing reduces memory usage during training."
    ]
    
    # 1. Test LoRA fine-tuning
    print("1. Testing LoRA Fine-tuning...")
    
    lora_config = FineTuningConfig(
        base_model_name="gpt2",
        use_lora=True,
        use_p_tuning=False,
        use_prefix_tuning=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1
    )
    
    lora_manager = EfficientFineTuningManager(lora_config)
    lora_trainer = FineTuningTrainer(lora_manager, lora_config)
    
    # Get parameter information
    param_info = lora_manager.get_trainable_parameters()
    print(f"LoRA trainable parameters: {param_info['trainable_params']:,}")
    print(f"LoRA trainable percentage: {param_info['trainable_percentage']:.2f}%")
    
    # Train LoRA model
    lora_results = lora_trainer.train(train_texts, eval_texts)
    print(f"LoRA training completed. Loss: {lora_results['train_loss']:.4f}")
    
    # 2. Test P-tuning
    print("\n2. Testing P-tuning...")
    
    p_tuning_config = FineTuningConfig(
        base_model_name="gpt2",
        use_lora=False,
        use_p_tuning=True,
        use_prefix_tuning=False,
        num_virtual_tokens=20,
        encoder_hidden_size=128
    )
    
    p_tuning_manager = EfficientFineTuningManager(p_tuning_config)
    p_tuning_trainer = FineTuningTrainer(p_tuning_manager, p_tuning_config)
    
    param_info = p_tuning_manager.get_trainable_parameters()
    print(f"P-tuning trainable parameters: {param_info['trainable_params']:,}")
    print(f"P-tuning trainable percentage: {param_info['trainable_percentage']:.2f}%")
    
    # 3. Compare techniques
    print("\n3. Comparing Fine-tuning Techniques...")
    
    analyzer = FineTuningAnalyzer()
    techniques = ["LoRA", "P-tuning", "Prefix-tuning"]
    
    comparison_results = analyzer.compare_techniques(
        techniques, train_texts, eval_texts, "gpt2"
    )
    
    # Visualize comparison
    analyzer.visualize_comparison(comparison_results)
    
    # Print summary
    analyzer.print_comparison_summary(comparison_results)
    
    # 4. Test text generation
    print("\n4. Testing Text Generation...")
    
    generator = ModelGenerator(lora_manager)
    
    test_prompts = [
        "The future of AI is",
        "Efficient fine-tuning",
        "Transformer models"
    ]
    
    for prompt in test_prompts:
        generated_text = generator.generate_text(prompt, max_length=50)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print()
    
    print("=== Demonstration Completed Successfully! ===")






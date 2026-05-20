"""
Custom LLM Trainer Module
=========================

This module provides a custom trainer class for fine-tuning Large Language Models (LLMs)
using the Hugging Face Transformers library. It supports training on JSON datasets
with prompt-response pairs and includes GPU/TPU acceleration support.

Features:
- JSON dataset loading with prompt-response format
- Pre-trained tokenizer support
- Configurable training arguments
- Automatic checkpoint saving
- GPU/TPU detection and utilization
- Comprehensive logging and error handling

Author: BUL System
Date: 2024
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedModel,
    TrainerCallback,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class TrainingProgressCallback(TrainerCallback):
    """Callback to log training progress and metrics."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics."""
        if logs:
            logger.info(f"Step {state.global_step}: {logs}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log epoch end information."""
        logger.info(f"Epoch {state.epoch} completed. Loss: {state.log_history[-1].get('loss', 'N/A')}")


class CustomLLMTrainer(Trainer):
    """
    Custom LLM Trainer class that extends Hugging Face Transformers Trainer.
    
    This class provides a simplified interface for training language models on
    JSON datasets with prompt-response pairs. It handles tokenization, dataset
    preparation, and training configuration automatically.
    
    Attributes:
        model: Pre-trained language model
        tokenizer: Pre-trained tokenizer
        training_args: TrainingArguments configuration
        device: Device to use for training (cuda/cpu/tpu)
        max_length: Maximum sequence length for tokenization
        
    Example:
        >>> trainer = CustomLLMTrainer(
        ...     model_name="gpt2",
        ...     dataset_path="data/training.json",
        ...     output_dir="./checkpoints"
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path] = "./checkpoints",
        learning_rate: float = 3e-5,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        max_length: int = 512,
        tokenizer_name: Optional[str] = None,
        model_type: str = "causal",
        gradient_accumulation_steps: int = 1,
        warmup_steps: Optional[int] = None,
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: Optional[int] = None,
        evaluation_strategy: str = "no",
        load_best_model_at_end: bool = False,
        save_total_limit: int = 3,
        fp16: bool = False,
        bf16: bool = False,
        dataloader_num_workers: int = 4,
        **kwargs
    ):
        """
        Initialize CustomLLMTrainer.
        
        Args:
            model_name: Name or path of the pre-trained model (e.g., "gpt2", "t5-small")
            dataset_path: Path to JSON file with "prompt" and "response" fields
            output_dir: Directory to save checkpoints and logs
            learning_rate: Learning rate for training (default: 3e-5)
            num_train_epochs: Number of training epochs (default: 3)
            batch_size: Training batch size (default: 8)
            max_length: Maximum sequence length for tokenization (default: 512)
            tokenizer_name: Optional tokenizer name (defaults to model_name)
            model_type: Type of model - "causal" or "seq2seq" (default: "causal")
            gradient_accumulation_steps: Number of gradient accumulation steps (default: 1)
            warmup_steps: Number of warmup steps (default: None, auto-calculated)
            weight_decay: Weight decay for regularization (default: 0.01)
            logging_steps: Steps between logging (default: 10)
            save_steps: Steps between checkpoints (default: 500)
            eval_steps: Steps between evaluations (default: None)
            evaluation_strategy: Evaluation strategy - "no", "steps", "epoch" (default: "no")
            load_best_model_at_end: Load best model at end of training (default: False)
            save_total_limit: Maximum number of checkpoints to keep (default: 3)
            fp16: Use mixed precision training with FP16 (default: False)
            bf16: Use mixed precision training with BF16 (default: False)
            dataloader_num_workers: Number of data loader workers (default: 4)
            **kwargs: Additional arguments for TrainingArguments
            
        Raises:
            FileNotFoundError: If dataset_path doesn't exist
            ValueError: If dataset format is invalid
            RuntimeError: If device setup fails
        """
        # Setup device
        self.device = self._setup_device()
        logger.info(f"Using device: {self.device}")
        
        # Load and validate dataset
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        logger.info(f"Loading dataset from: {self.dataset_path}")
        self.raw_data = self._load_dataset()
        self.max_length = max_length
        
        # Load tokenizer
        tokenizer_name = tokenizer_name or model_name
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._setup_tokenizer()
        
        # Load model
        logger.info(f"Loading model: {model_name} (type: {model_type})")
        self.model = self._load_model(model_name, model_type)
        self.model_type = model_type
        
        # Prepare tokenized dataset
        logger.info("Tokenizing dataset...")
        self.tokenized_dataset = self._prepare_dataset()
        
        # Setup training arguments
        self.training_args = self._setup_training_args(
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy=evaluation_strategy,
            load_best_model_at_end=load_best_model_at_end,
            save_total_limit=save_total_limit,
            fp16=fp16,
            bf16=bf16,
            dataloader_num_workers=dataloader_num_workers,
            **kwargs
        )
        
        # Setup data collator
        self.data_collator = self._setup_data_collator()
        
        # Initialize parent Trainer class
        super().__init__(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset["train"] if isinstance(self.tokenized_dataset, DatasetDict) else self.tokenized_dataset,
            eval_dataset=self.tokenized_dataset.get("validation") if isinstance(self.tokenized_dataset, DatasetDict) else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=[TrainingProgressCallback()],
        )
        
        logger.info("CustomLLMTrainer initialized successfully")
    
    def _setup_device(self) -> torch.device:
        """
        Setup and return the appropriate device for training.
        
        Detects and configures GPU, TPU, or CPU based on availability.
        
        Returns:
            torch.device: Configured device
            
        Raises:
            RuntimeError: If device setup fails
        """
        # Check for TPU (XLA)
        try:
            import torch_xla.core.xla_model as xm
            if xm.xla_device():
                logger.info("TPU detected and configured")
                return xm.xla_device()
        except (ImportError, RuntimeError):
            pass
        
        # Check for CUDA/GPU
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {device_count} GPU(s)")
            if device_count > 1:
                logger.info("Multiple GPUs detected - using DataParallel")
            return torch.device("cuda")
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple Silicon (MPS) detected")
            return torch.device("mps")
        
        # Fallback to CPU
        logger.warning("No GPU/TPU detected, using CPU (training will be slow)")
        return torch.device("cpu")
    
    def _load_dataset(self) -> List[Dict[str, str]]:
        """
        Load and validate JSON dataset.
        
        Returns:
            List[Dict[str, str]]: List of dictionaries with "prompt" and "response" keys
            
        Raises:
            ValueError: If dataset format is invalid
        """
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both list and dict formats
            if isinstance(data, dict):
                if "data" in data:
                    data = data["data"]
                elif "examples" in data:
                    data = data["examples"]
                else:
                    raise ValueError("Dataset dict must contain 'data' or 'examples' key")
            
            if not isinstance(data, list):
                raise ValueError("Dataset must be a list of examples")
            
            # Validate format
            required_fields = ["prompt", "response"]
            for i, example in enumerate(data):
                if not isinstance(example, dict):
                    raise ValueError(f"Example {i} must be a dictionary")
                for field in required_fields:
                    if field not in example:
                        raise ValueError(f"Example {i} missing required field: {field}")
                    if not isinstance(example[field], str):
                        raise ValueError(f"Example {i} field '{field}' must be a string")
            
            logger.info(f"Loaded {len(data)} examples from dataset")
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
    
    def _setup_tokenizer(self) -> None:
        """Setup tokenizer with padding and special tokens."""
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Set padding side
        self.tokenizer.padding_side = "left" if self.model_type == "causal" else "right"
        
        logger.info(f"Tokenizer configured (vocab_size: {self.tokenizer.vocab_size})")
    
    def _load_model(self, model_name: str, model_type: str) -> PreTrainedModel:
        """
        Load pre-trained model.
        
        Args:
            model_name: Name or path of the model
            model_type: Type of model ("causal" or "seq2seq")
            
        Returns:
            PreTrainedModel: Loaded model
        """
        try:
            if model_type == "causal":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
            elif model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}. Must be 'causal' or 'seq2seq'")
            
            # Resize token embeddings if tokenizer was modified
            if len(self.tokenizer) > model.config.vocab_size:
                model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Resized token embeddings to {len(self.tokenizer)}")
            
            model.to(self.device)
            logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def _tokenize_function(self, examples: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Tokenize examples for training.
        
        Args:
            examples: Dictionary with "prompt" and "response" lists
            
        Returns:
            Dictionary with tokenized inputs
        """
        prompts = examples["prompt"]
        responses = examples["response"]
        
        if self.model_type == "causal":
            # For causal models, concatenate prompt + response
            texts = [f"{p} {r}" for p, r in zip(prompts, responses)]
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            
            # Labels are same as input_ids for causal LM
            tokenized["labels"] = tokenized["input_ids"].clone()
            
        else:  # seq2seq
            # Tokenize prompts and responses separately
            tokenized_prompts = self.tokenizer(
                prompts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            
            tokenized_responses = self.tokenizer(
                responses,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            
            tokenized = {
                "input_ids": tokenized_prompts["input_ids"],
                "attention_mask": tokenized_prompts["attention_mask"],
                "labels": tokenized_responses["input_ids"],
            }
        
        return tokenized
    
    def _prepare_dataset(self) -> Union[Dataset, DatasetDict]:
        """
        Prepare tokenized dataset from raw data.
        
        Returns:
            Dataset or DatasetDict with tokenized data
        """
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(self.raw_data)
        
        # Split into train/validation if needed (80/20 split)
        if len(dataset) > 100:  # Only split if dataset is large enough
            dataset = dataset.train_test_split(test_size=0.2, seed=42)
            logger.info(f"Dataset split: {len(dataset['train'])} train, {len(dataset['test'])} validation")
            dataset["validation"] = dataset.pop("test")
        else:
            logger.info("Dataset too small for validation split, using all data for training")
            dataset = DatasetDict({"train": dataset})
        
        # Tokenize
        tokenized = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized
    
    def _setup_data_collator(self) -> DataCollatorForLanguageModeling:
        """
        Setup data collator for language modeling.
        
        Returns:
            DataCollatorForLanguageModeling instance
        """
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal/seq2seq LM, not masked LM
        )
    
    def _setup_training_args(
        self,
        output_dir: Union[str, Path],
        learning_rate: float,
        num_train_epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        warmup_steps: Optional[int],
        weight_decay: float,
        logging_steps: int,
        save_steps: int,
        eval_steps: Optional[int],
        evaluation_strategy: str,
        load_best_model_at_end: bool,
        save_total_limit: int,
        fp16: bool,
        bf16: bool,
        dataloader_num_workers: int,
        **kwargs
    ) -> TrainingArguments:
        """
        Setup training arguments.
        
        Returns:
            TrainingArguments instance
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate warmup steps if not provided (10% of total steps)
        if warmup_steps is None:
            total_steps = len(self.tokenized_dataset["train"]) // batch_size * num_train_epochs
            warmup_steps = max(1, int(total_steps * 0.1))
        
        # Auto-detect mixed precision
        if torch.cuda.is_available():
            # Check BF16 support (available on Ampere+ GPUs)
            try:
                bf16_supported = torch.cuda.get_device_capability()[0] >= 8
            except:
                bf16_supported = False
            
            if bf16 and bf16_supported:
                fp16 = False
                logger.info("Using BF16 mixed precision")
            elif fp16:
                logger.info("Using FP16 mixed precision")
            else:
                logger.info("Using FP32 precision")
        
        args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=str(output_dir / "logs"),
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy=evaluation_strategy,
            load_best_model_at_end=load_best_model_at_end,
            save_total_limit=save_total_limit,
            fp16=fp16,
            bf16=bf16,
            dataloader_num_workers=dataloader_num_workers,
            report_to="none",  # Disable wandb/tensorboard by default
            remove_unused_columns=False,
            **kwargs
        )
        
        logger.info(f"Training arguments configured: {args}")
        return args
    
    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None) -> None:
        """
        Train the model and save final checkpoint.
        
        Args:
            resume_from_checkpoint: Path to checkpoint or True to resume from latest
        """
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info(f"Model: {self.model.config.name_or_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training examples: {len(self.tokenized_dataset['train'])}")
        logger.info(f"Epochs: {self.training_args.num_train_epochs}")
        logger.info(f"Batch size: {self.training_args.per_device_train_batch_size}")
        logger.info(f"Learning rate: {self.training_args.learning_rate}")
        logger.info("=" * 80)
        
        try:
            # Train the model
            train_result = super().train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Save final checkpoint
            final_checkpoint_dir = Path(self.training_args.output_dir) / "final_checkpoint"
            logger.info(f"Saving final checkpoint to: {final_checkpoint_dir}")
            self.save_model(str(final_checkpoint_dir))
            self.tokenizer.save_pretrained(str(final_checkpoint_dir))
            
            # Log training metrics
            logger.info("Training completed successfully!")
            logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            if hasattr(train_result, "metrics"):
                logger.info(f"Training metrics: {train_result.metrics}")
            
            # Log system resources
            if torch.cuda.is_available():
                logger.info(f"GPU Memory - Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB, "
                          f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            
            logger.info(f"Final checkpoint saved at: {final_checkpoint_dir}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self, output_dir: Union[str, Path]) -> None:
        """
        Save model and tokenizer to output directory.
        
        Args:
            output_dir: Directory to save model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info(f"Model saved to: {output_dir}")
    
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            eval_dataset: Optional evaluation dataset (uses validation set if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if eval_dataset is None:
            if isinstance(self.tokenized_dataset, DatasetDict) and "validation" in self.tokenized_dataset:
                eval_dataset = self.tokenized_dataset["validation"]
            else:
                raise ValueError("No evaluation dataset available")
        
        logger.info("Evaluating model...")
        eval_results = super().evaluate(eval_dataset=eval_dataset)
        
        logger.info("Evaluation results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return eval_results
    
    def predict(self, prompts: Union[str, List[str]], max_new_tokens: int = 100) -> Union[str, List[str]]:
        """
        Generate predictions from prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response(s)
        """
        self.model.eval()
        
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        generated_texts = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                # Generate
                if self.model_type == "causal":
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                else:  # seq2seq
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                    )
                
                # Decode
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_texts.append(generated)
        
        return generated_texts[0] if is_single else generated_texts


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of CustomLLMTrainer.
    
    This example demonstrates how to use the CustomLLMTrainer class
    to fine-tune a language model on a custom dataset.
    """
    
    # Example dataset format
    example_data = [
        {
            "prompt": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
        },
        {
            "prompt": "Explain neural networks.",
            "response": "Neural networks are computing systems inspired by biological neural networks that process information through interconnected nodes (neurons)."
        }
    ]
    
    # Create example dataset file
    dataset_path = Path("./example_dataset.json")
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(example_data, f, indent=2, ensure_ascii=False)
    
    print("Example dataset created at:", dataset_path)
    print("\nTo use CustomLLMTrainer:")
    print("""
    from custom_llm_trainer import CustomLLMTrainer
    
    trainer = CustomLLMTrainer(
        model_name="gpt2",
        dataset_path="./example_dataset.json",
        output_dir="./checkpoints",
        learning_rate=3e-5,
        num_train_epochs=3,
        batch_size=8
    )
    
    trainer.train()
    """)


"""
Custom LLM Trainer Module
==========================

Main trainer class that integrates all modular components for LLM training.
Provides a simplified interface for fine-tuning language models.

Author: BUL System
Date: 2024
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .device_manager import DeviceManager
from .dataset_loader import DatasetLoader
from .tokenizer_utils import TokenizerUtils
from .model_loader import ModelLoader
from .config import TrainingConfig
from .callbacks import (
    TrainingProgressCallback,
    EarlyStoppingCallback,
    MemoryMonitoringCallback,
    TrainingTimeCallback,
)
from .metrics import compute_metrics
from .data import DatasetValidator, DatasetProcessor
from .models import ModelFactory, ModelConfig
from .plugins import PluginRegistry, CallbackPlugin
from .utils import validate_dataset_path, estimate_training_time
from .training import CheckpointManager, ResumeManager
from .monitoring import ExperimentTracker, PerformanceProfiler
from .distributed import setup_distributed_training, is_distributed

logger = logging.getLogger(__name__)


class CustomLLMTrainer(Trainer):
    """
    Custom LLM Trainer class that extends Hugging Face Transformers Trainer.
    
    This class provides a simplified interface for training language models on
    JSON datasets with prompt-response pairs. It handles tokenization, dataset
    preparation, and training configuration automatically using modular components.
    
    The trainer is composed of:
    - DeviceManager: Handles GPU/TPU/CPU detection
    - DatasetLoader: Loads and validates JSON datasets
    - TokenizerUtils: Manages tokenization
    - ModelLoader: Loads pre-trained models
    - TrainingConfig: Configures training arguments
    
    Attributes:
        model: Pre-trained language model
        tokenizer: Pre-trained tokenizer
        training_args: TrainingArguments configuration
        device_manager: DeviceManager instance
        tokenized_dataset: Tokenized dataset ready for training
        
    Example:
        >>> trainer = CustomLLMTrainer(
        ...     model_name="gpt2",
        ...     dataset_path="data/training.json",
        ...     output_dir="./checkpoints",
        ...     learning_rate=3e-5,
        ...     num_train_epochs=3,
        ...     batch_size=8
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
        early_stopping_patience: Optional[int] = None,
        enable_memory_monitoring: bool = True,
            enable_time_tracking: bool = True,
            compute_metrics_fn: Optional[Callable] = None,
            enable_experiment_tracking: bool = False,
            experiments_dir: Optional[Union[str, Path]] = None,
            enable_profiling: bool = False,
            enable_distributed: bool = False,
        gradient_checkpointing: bool = False,
        optimizer: str = "adamw_torch",
        lr_scheduler_type: str = "linear",
        max_grad_norm: float = 1.0,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize CustomLLMTrainer.
        
        This trainer class provides a complete solution for fine-tuning language models
        with automatic optimizations, validation, and comprehensive error handling.
        
        Args:
            model_name: Name or path of the pre-trained model (e.g., "gpt2", "t5-small")
            dataset_path: Path to JSON file with "prompt" and "response" fields
            output_dir: Directory to save checkpoints and logs (default: "./checkpoints")
            learning_rate: Learning rate for training (default: 3e-5)
            num_train_epochs: Number of training epochs (default: 3)
            batch_size: Training batch size (default: 8, auto-adjusted based on device)
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
            fp16: Use mixed precision training with FP16 (default: False, auto-enabled for compatible GPUs)
            bf16: Use mixed precision training with BF16 (default: False, auto-enabled for Ampere+ GPUs)
            dataloader_num_workers: Number of data loader workers (default: 4, auto-adjusted for CPU)
            early_stopping_patience: Patience for early stopping (default: None, disabled)
            enable_memory_monitoring: Enable GPU memory monitoring (default: True)
            enable_time_tracking: Enable training time tracking (default: True)
            compute_metrics_fn: Custom metrics computation function (default: None)
            gradient_checkpointing: Enable gradient checkpointing to save memory (default: False)
            optimizer: Optimizer type - "adamw_torch", "adam", "sgd" (default: "adamw_torch")
            lr_scheduler_type: Learning rate scheduler - "linear", "cosine", "polynomial" (default: "linear")
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
            seed: Random seed for reproducibility (default: 42)
            **kwargs: Additional arguments for TrainingArguments
            
        Raises:
            FileNotFoundError: If dataset_path doesn't exist
            ValueError: If dataset format is invalid or parameters are incompatible
            RuntimeError: If device setup fails or model loading fails
            
        Example:
            >>> from llm_trainer import CustomLLMTrainer
            >>> 
            >>> trainer = CustomLLMTrainer(
            ...     model_name="gpt2",
            ...     dataset_path="data/training.json",
            ...     output_dir="./checkpoints",
            ...     learning_rate=3e-5,
            ...     num_train_epochs=3,
            ...     batch_size=8
            ... )
            >>> trainer.train()
        """
        # Validate parameters
        self._validate_parameters(
            model_name, model_type, learning_rate, num_train_epochs,
            batch_size, max_length, gradient_accumulation_steps
        )
        
        # Initialize device manager
        self.device_manager = DeviceManager()
        logger.info(f"Using device: {self.device_manager.get_device()}")
        
        # Auto-optimize batch size based on device if not explicitly set
        if batch_size == 8:  # Default value
            recommended_batch = self.device_manager.get_recommended_batch_size(
                model_size_gb=0.5,  # Rough estimate, will be updated after model load
                sequence_length=max_length
            )
            if recommended_batch != batch_size:
                logger.info(f"Auto-adjusting batch size from {batch_size} to {recommended_batch} based on device capabilities")
                batch_size = recommended_batch
        
        # Validate and load dataset using modular components
        validated_path = validate_dataset_path(dataset_path)
        
        # Use DatasetValidator for validation
        dataset_validator = DatasetValidator()
        is_valid, errors, raw_data = dataset_validator.validate_file(validated_path)
        
        if not is_valid:
            error_msg = "Dataset validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)
        
        # Process dataset if needed
        dataset_processor = DatasetProcessor()
        self.raw_data = dataset_processor.clean_dataset(raw_data)
        
        # Validate quality
        quality_report = dataset_validator.validate_quality(self.raw_data)
        if quality_report["warnings"]:
            logger.warning(f"Dataset quality score: {quality_report['quality_score']:.1f}/100")
            for warning in quality_report["warnings"]:
                logger.warning(f"  - {warning}")
            if not quality_report["is_valid"]:
                logger.warning("Dataset quality is low. Training may not be effective.")
        else:
            logger.info(f"Dataset quality score: {quality_report['quality_score']:.1f}/100")
        
        # Load and prepare dataset
        dataset_loader = DatasetLoader(dataset_path)
        # Get and log statistics
        stats = dataset_loader.get_statistics(self.raw_data)
        logger.info(f"Dataset statistics: {stats['total_examples']} examples, "
                   f"avg prompt: {stats['prompt_length']['avg']:.0f} chars, "
                   f"avg response: {stats['response_length']['avg']:.0f} chars")
        
        self.tokenized_dataset = dataset_loader.prepare_dataset(self.raw_data)
        
        # Initialize tokenizer
        tokenizer_name = tokenizer_name or model_name
        self.tokenizer_utils = TokenizerUtils(
            tokenizer_name=tokenizer_name,
            model_type=model_type,
            max_length=max_length
        )
        self.tokenizer = self.tokenizer_utils.get_tokenizer()
        self.model_type = model_type
        self.max_length = max_length
        
        # Load model using ModelFactory for better modularity
        model_factory = ModelFactory(self.device_manager)
        
        if model_type == "causal":
            self.model = model_factory.create_causal_model(
                model_name=model_name,
                tokenizer_vocab_size=len(self.tokenizer)
            )
        else:
            self.model = model_factory.create_seq2seq_model(
                model_name=model_name,
                tokenizer_vocab_size=len(self.tokenizer)
            )
        
        # Tokenize dataset
        logger.info("Tokenizing dataset...")
        self.tokenized_dataset = self._tokenize_dataset()
        
        # Setup training configuration
        training_config = TrainingConfig(
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            device_manager=self.device_manager,
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
            num_train_samples=len(self.tokenized_dataset["train"]),
            optimizer=optimizer,
            lr_scheduler_type=lr_scheduler_type,
            gradient_checkpointing=gradient_checkpointing,
            max_grad_norm=max_grad_norm,
            seed=seed,
            **kwargs
        )
        self.training_args = training_config.get_training_args()
        
        # Setup data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal/seq2seq LM, not masked LM
        )
        
        # Setup callbacks
        callbacks = [TrainingProgressCallback()]
        
        if early_stopping_patience is not None and evaluation_strategy != "no":
            callbacks.append(EarlyStoppingCallback(patience=early_stopping_patience))
            logger.info(f"Early stopping enabled with patience={early_stopping_patience}")
        
        if enable_memory_monitoring and self.device_manager.is_cuda_available():
            callbacks.append(MemoryMonitoringCallback())
            logger.info("Memory monitoring enabled")
        
        if enable_time_tracking:
            callbacks.append(TrainingTimeCallback())
            logger.info("Time tracking enabled")
        
        # Initialize plugin registry for extensibility
        self.plugin_registry = PluginRegistry()
        
        # Initialize checkpoint and resume managers
        self.checkpoint_manager = CheckpointManager(output_dir)
        self.resume_manager = ResumeManager(output_dir)
        
        # Initialize experiment tracker and profiler (optional)
        self.experiment_tracker: Optional[ExperimentTracker] = None
        self.profiler: Optional[PerformanceProfiler] = None
        
        # Setup distributed training if needed
        if enable_distributed:
            dist_config = setup_distributed_training()
            if dist_config.get("distributed"):
                logger.info(f"Distributed training enabled: {dist_config}")
        
        # Setup experiment tracking if enabled
        if enable_experiment_tracking:
            exp_dir = experiments_dir or Path(output_dir) / "experiments"
            self.experiment_tracker = ExperimentTracker(exp_dir)
            self.experiment_tracker.start_experiment(
                experiment_name=f"{model_name}_training",
                description=f"Training {model_name} on {dataset_path}",
                tags=["llm", "training", model_type],
                model_name=model_name,
                model_type=model_type,
                dataset_path=str(dataset_path),
            )
            # Log hyperparameters
            self.experiment_tracker.log_params({
                "learning_rate": learning_rate,
                "num_train_epochs": num_train_epochs,
                "batch_size": batch_size,
                "max_length": max_length,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            })
        
        # Setup profiling if enabled
        if enable_profiling:
            self.profiler = PerformanceProfiler()
            self.profiler.start()
        
        # Setup compute_metrics function
        compute_metrics_fn = compute_metrics_fn or compute_metrics
        
        # Add metric plugins if any are registered
        enabled_plugins = self.plugin_registry.get_enabled()
        for plugin in enabled_plugins:
            if hasattr(plugin, 'compute') and callable(plugin.compute):
                # Combine plugin metrics with default metrics
                original_compute = compute_metrics_fn
                def combined_metrics(eval_pred):
                    base_metrics = original_compute(eval_pred)
                    plugin_metrics = plugin.compute(eval_pred)
                    return {**base_metrics, **plugin_metrics}
                compute_metrics_fn = combined_metrics
        
        # Enable gradient checkpointing if requested
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Initialize parent Trainer class
        super().__init__(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset["train"] if isinstance(self.tokenized_dataset, DatasetDict) else self.tokenized_dataset,
            eval_dataset=self.tokenized_dataset.get("validation") if isinstance(self.tokenized_dataset, DatasetDict) else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=callbacks,
            compute_metrics=compute_metrics_fn if evaluation_strategy != "no" else None,
        )
        
        logger.info("CustomLLMTrainer initialized successfully")
        
        # Log initialization summary
        summary = self.get_training_summary()
        logger.info(f"Training Summary:\n"
                   f"  Model: {summary['model_info']['model_name']} "
                   f"({summary['model_info']['parameters_millions']:.2f}M params)\n"
                   f"  Device: {summary['device_info'].get('type', 'unknown')}\n"
                   f"  Dataset: {summary['dataset_info']['train_size']} train, "
                   f"{summary['dataset_info']['eval_size']} eval\n"
                   f"  Config: lr={summary['training_config']['learning_rate']}, "
                   f"epochs={summary['training_config']['num_epochs']}, "
                   f"batch={summary['training_config']['batch_size']}")
        
        # Log training time estimate
        time_estimate = self.get_estimated_training_time()
        logger.info(f"Estimated training time: {time_estimate['total_minutes']:.1f} minutes "
                   f"({time_estimate['total_hours']:.2f} hours)")
        
        # Log recommendations
        recommendations = self.get_training_recommendations()
        if recommendations:
            logger.info("Training recommendations:")
            for rec in recommendations:
                logger.info(f"  • {rec}")
    
    def _validate_parameters(
        self,
        model_name: str,
        model_type: str,
        learning_rate: float,
        num_train_epochs: int,
        batch_size: int,
        max_length: int,
        gradient_accumulation_steps: int
    ) -> None:
        """
        Validate training parameters.
        
        Args:
            model_name: Model name
            model_type: Model type
            learning_rate: Learning rate
            num_train_epochs: Number of epochs
            batch_size: Batch size
            max_length: Max sequence length
            gradient_accumulation_steps: Gradient accumulation steps
            
        Raises:
            ValueError: If parameters are invalid or incompatible
        """
        # Validate model_name
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
        
        # Validate model_type
        if model_type not in ["causal", "seq2seq"]:
            raise ValueError(f"model_type must be 'causal' or 'seq2seq', got '{model_type}'")
        
        # Validate learning_rate
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if learning_rate > 1.0:
            logger.warning(f"Learning rate {learning_rate} is very high. Typical values are 1e-5 to 1e-3")
        
        # Validate num_train_epochs
        if num_train_epochs <= 0:
            raise ValueError(f"num_train_epochs must be positive, got {num_train_epochs}")
        if num_train_epochs > 100:
            logger.warning(f"Number of epochs {num_train_epochs} is very high")
        
        # Validate batch_size
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > 128:
            logger.warning(f"Batch size {batch_size} is very large. Consider using gradient accumulation instead")
        
        # Validate max_length
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if max_length > 4096:
            logger.warning(f"Max length {max_length} is very large and may cause memory issues")
        
        # Validate gradient_accumulation_steps
        if gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {gradient_accumulation_steps}")
        
        # Check compatibility
        effective_batch = batch_size * gradient_accumulation_steps
        if effective_batch > 1024:
            logger.warning(
                f"Effective batch size ({effective_batch}) is very large. "
                f"This may cause training instability."
            )
    
    def _tokenize_dataset(self) -> Union[Dataset, DatasetDict]:
        """
        Tokenize the dataset using TokenizerUtils.
        
        Returns:
            Tokenized dataset
        """
        tokenized = self.tokenized_dataset.map(
            self.tokenizer_utils.tokenize_examples,
            batched=True,
            remove_columns=self.tokenized_dataset["train"].column_names,
            desc="Tokenizing dataset"
        )
        return tokenized
    
    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None) -> Dict[str, Any]:
        """
        Train the model and save final checkpoint.
        
        This method handles the complete training process including:
        - Training loop execution
        - Checkpoint saving (periodic and final)
        - Error handling with recovery suggestions
        - Resource monitoring
        
        Args:
            resume_from_checkpoint: Path to checkpoint or True to resume from latest.
                                   If str, should be path to checkpoint directory.
                                   If True, resumes from latest checkpoint in output_dir.
            
        Returns:
            Dictionary with training results including:
            - training_loss: Final training loss
            - metrics: Additional training metrics if available
            - checkpoint_path: Path to final checkpoint
            
        Raises:
            RuntimeError: If training fails with specific error messages and suggestions
            KeyboardInterrupt: If interrupted by user (saves checkpoint before raising)
            torch.cuda.OutOfMemoryError: If GPU runs out of memory (with suggestions)
            
        Example:
            >>> trainer = CustomLLMTrainer(...)
            >>> results = trainer.train()
            >>> print(f"Final loss: {results['training_loss']}")
        """
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info(f"Model: {self.model.config.name_or_path}")
        logger.info(f"Device: {self.device_manager.get_device()}")
        logger.info(f"Training examples: {len(self.tokenized_dataset['train'])}")
        logger.info(f"Epochs: {self.training_args.num_train_epochs}")
        logger.info(f"Batch size: {self.training_args.per_device_train_batch_size}")
        logger.info(f"Learning rate: {self.training_args.learning_rate}")
        logger.info("=" * 80)
        
        try:
            # Train the model with profiling if enabled
            if self.profiler:
                with self.profiler.profile("training"):
                    train_result = super().train(resume_from_checkpoint=resume_from_checkpoint)
            else:
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
            memory_info = self.device_manager.get_memory_info()
            if memory_info:
                logger.info(
                    f"GPU Memory - Allocated: {memory_info['allocated']:.2f} GB, "
                    f"Reserved: {memory_info['reserved']:.2f} GB"
                )
            
            logger.info(f"Final checkpoint saved at: {final_checkpoint_dir}")
            
            # Return training results
            results = {
                "training_loss": train_result.training_loss,
                "checkpoint_path": str(final_checkpoint_dir),
            }
            if hasattr(train_result, "metrics"):
                results["metrics"] = train_result.metrics
            
            # Log to experiment tracker
            if self.experiment_tracker:
                self.experiment_tracker.log_metrics(results.get("metrics", {}))
                self.experiment_tracker.log_artifact(final_checkpoint_dir, "final_checkpoint")
                self.experiment_tracker.end_experiment("completed")
            
            # Get profiling report
            if self.profiler:
                profile_report = self.profiler.get_report()
                logger.info(f"Performance report: {profile_report.get('steps_per_second', 0):.2f} steps/sec")
                results["profiling"] = profile_report
            
            return results
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            # Save checkpoint before exiting
            interrupt_checkpoint_dir = Path(self.training_args.output_dir) / "interrupted_checkpoint"
            logger.info(f"Saving interrupted checkpoint to: {interrupt_checkpoint_dir}")
            self.save_model(str(interrupt_checkpoint_dir))
            raise
        except torch.cuda.OutOfMemoryError as e:
            logger.error("=" * 80)
            logger.error("CUDA OUT OF MEMORY ERROR")
            logger.error("=" * 80)
            logger.error("The GPU ran out of memory during training.")
            logger.error("\nSuggested solutions:")
            logger.error("1. Reduce batch_size (current: {})".format(self.training_args.per_device_train_batch_size))
            logger.error("2. Reduce max_length (current: {})".format(self.max_length))
            logger.error("3. Enable gradient_checkpointing=True")
            logger.error("4. Enable fp16=True for mixed precision")
            logger.error("5. Increase gradient_accumulation_steps (current: {})".format(
                self.training_args.gradient_accumulation_steps
            ))
            logger.error("6. Use a smaller model")
            logger.error("=" * 80)
            self.device_manager.clear_cache()
            raise RuntimeError(
                "CUDA out of memory. Suggestions:\n"
                "- Reduce batch_size or max_length\n"
                "- Enable gradient_checkpointing=True\n"
                "- Enable fp16=True\n"
                "- Increase gradient_accumulation_steps"
            ) from e
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
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
    
    def predict(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Union[str, List[str]]:
        """
        Generate predictions from prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling parameter (default: 0.9)
            do_sample: Whether to use sampling (default: True)
            
        Returns:
            Generated response(s)
        """
        self.model.eval()
        
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        generated_texts = []
        device = self.device_manager.get_device()
        
        with torch.no_grad():
            for prompt in prompts:
                try:
                    # Tokenize
                    inputs = self.tokenizer_utils.tokenize_for_inference(prompt)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Generate
                    generate_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": do_sample,
                        "temperature": temperature,
                        "top_p": top_p if do_sample else None,
                    }
                    
                    if self.model_type == "causal":
                        generate_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
                    
                    outputs = self.model.generate(**inputs, **{k: v for k, v in generate_kwargs.items() if v is not None})
                    
                    # Decode
                    generated = self.tokenizer_utils.decode(outputs[0], skip_special_tokens=True)
                    generated_texts.append(generated)
                except Exception as e:
                    logger.error(f"Error generating for prompt '{prompt[:50]}...': {e}")
                    generated_texts.append("")
        
        return generated_texts[0] if is_single else generated_texts
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get device information.
        
        Returns:
            Dictionary with device information
        """
        return self.device_manager.get_device_info()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model information
        """
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Calculate model size in GB (rough estimate)
        model_size_gb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e9
        
        return {
            "model_name": self.model.config.name_or_path,
            "model_type": self.model_type,
            "vocab_size": self.model.config.vocab_size,
            "total_parameters": param_count,
            "trainable_parameters": trainable_params,
            "parameters_millions": param_count / 1e6,
            "parameters_billions": param_count / 1e9,
            "model_size_gb": model_size_gb,
            "device": str(self.device_manager.get_device()),
            "device_summary": self.device_manager.get_device_summary(),
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of training configuration and status.
        
        Returns:
            Dictionary with training summary
        """
        train_size = len(self.tokenized_dataset["train"]) if isinstance(self.tokenized_dataset, DatasetDict) else len(self.tokenized_dataset)
        eval_size = len(self.tokenized_dataset["validation"]) if isinstance(self.tokenized_dataset, DatasetDict) and "validation" in self.tokenized_dataset else 0
        
        return {
            "model_info": self.get_model_info(),
            "device_info": self.device_manager.get_device_info(),
            "dataset_info": {
                "train_size": train_size,
                "eval_size": eval_size,
                "total_size": train_size + eval_size,
            },
            "training_config": {
                "learning_rate": self.training_args.learning_rate,
                "num_epochs": self.training_args.num_train_epochs,
                "batch_size": self.training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": self.training_args.gradient_accumulation_steps,
                "effective_batch_size": self.training_args.per_device_train_batch_size * self.training_args.gradient_accumulation_steps,
                "max_length": self.max_length,
                "fp16": self.training_args.fp16,
                "bf16": self.training_args.bf16,
            },
            "output_dir": str(self.training_args.output_dir),
        }
    
    def register_plugin(self, plugin) -> None:
        """
        Register a plugin with the trainer.
        
        Args:
            plugin: Plugin instance (BasePlugin or subclass)
        """
        self.plugin_registry.register(plugin)
        logger.info(f"Plugin {plugin.name} v{plugin.version} registered")
        
        # If it's a callback plugin, add it to callbacks
        if isinstance(plugin, CallbackPlugin):
            self.add_callback(plugin)
    
    def get_estimated_training_time(self) -> Dict[str, float]:
        """
        Estimate training time based on current configuration.
        
        Returns:
            Dictionary with time estimates in different units
        """
        train_size = len(self.tokenized_dataset["train"]) if isinstance(self.tokenized_dataset, DatasetDict) else len(self.tokenized_dataset)
        batch_size = self.training_args.per_device_train_batch_size
        num_epochs = self.training_args.num_train_epochs
        
        # Estimate steps per second (rough estimate)
        # GPU: ~2-5 steps/s, CPU: ~0.1-0.5 steps/s
        if self.device_manager.is_cuda_available():
            steps_per_second = 3.0  # Conservative estimate
        elif self.device_manager.is_tpu_available():
            steps_per_second = 10.0  # TPUs are fast
        else:
            steps_per_second = 0.3  # CPU is slow
        
        return estimate_training_time(
            num_samples=train_size,
            batch_size=batch_size,
            num_epochs=num_epochs,
            steps_per_second=steps_per_second
        )
    
    def get_training_recommendations(self) -> List[str]:
        """
        Get recommendations for optimizing training based on current configuration.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check dataset size
        train_size = len(self.tokenized_dataset["train"]) if isinstance(self.tokenized_dataset, DatasetDict) else len(self.tokenized_dataset)
        if train_size < 100:
            recommendations.append(f"Dataset is small ({train_size} examples). Consider collecting more data.")
        
        # Check batch size
        batch_size = self.training_args.per_device_train_batch_size
        if batch_size == 1:
            recommendations.append("Batch size is 1. Consider using gradient_accumulation_steps for effective larger batches.")
        
        # Check device
        if not self.device_manager.is_cuda_available() and not self.device_manager.is_tpu_available():
            recommendations.append("Training on CPU will be slow. Consider using GPU if available.")
        
        # Check memory optimizations
        if not self.training_args.gradient_checkpointing and not self.training_args.fp16:
            model_size = self.get_model_info().get("model_size_gb", 0)
            if model_size > 1.0:
                recommendations.append("Large model detected. Consider enabling gradient_checkpointing or fp16 to save memory.")
        
        # Check evaluation
        if self.training_args.evaluation_strategy == "no":
            recommendations.append("No evaluation strategy set. Consider enabling evaluation to monitor overfitting.")
        
        return recommendations
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """
        Get information about available checkpoints.
        
        Returns:
            Dictionary with checkpoint information
        """
        checkpoints = self.checkpoint_manager.list_checkpoints()
        latest = self.checkpoint_manager.get_latest_checkpoint()
        best = self.checkpoint_manager.get_best_checkpoint()
        
        return {
            "total_checkpoints": len(checkpoints),
            "checkpoints": checkpoints,
            "latest_checkpoint": str(latest) if latest else None,
            "best_checkpoint": str(best) if best else None,
        }
    
    def resume_from_latest(self) -> bool:
        """
        Resume training from the latest checkpoint.
        
        Returns:
            True if checkpoint found and resume initiated, False otherwise
        """
        latest = self.resume_manager.find_latest_checkpoint()
        if latest:
            logger.info(f"Resuming training from: {latest}")
            self.train(resume_from_checkpoint=latest)
            return True
        else:
            logger.info("No checkpoint found to resume from")
            return False
    
    def cleanup_checkpoints(self, keep: int = 3) -> List[str]:
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep: Number of checkpoints to keep
            
        Returns:
            List of deleted checkpoint paths
        """
        deleted = self.checkpoint_manager.cleanup_old_checkpoints(keep=keep)
        logger.info(f"Cleaned up {len(deleted)} old checkpoints, kept {keep} most recent")
        return [str(p) for p in deleted]


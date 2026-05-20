import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our custom systems
from gradient_clipping_nan_handling_system import (
    GradientConfig, GradientMonitor, GradientClipper, RobustTrainingSystem
)
from advanced_evaluation_metrics_system import (
    ClassificationMetricsConfig, RegressionMetricsConfig, 
    TextGenerationMetricsConfig, CustomMetricsConfig,
    ComprehensiveEvaluationSystem
)


@dataclass
class ComprehensiveTrainingConfig:
    """Comprehensive training configuration."""
    
    # Model configuration
    model_name: str = "gpt2"
    model_type: str = "causal_lm"  # "causal_lm", "sequence_classification", "custom"
    num_labels: int = 2
    
    # Training configuration
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    
    # Optimization configuration
    use_amp: bool = True
    use_data_parallel: bool = False
    use_distributed: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Gradient configuration
    gradient_clip_val: float = 1.0
    gradient_clip_norm: float = 1.0
    gradient_clip_algorithm: str = "norm"
    
    # NaN/Inf handling
    check_gradients: bool = True
    check_weights: bool = True
    check_loss: bool = True
    zero_nan_gradients: bool = True
    skip_nan_batches: bool = False
    restart_training_on_nan: bool = False
    max_nan_restarts: int = 3
    
    # Evaluation configuration
    evaluation_interval: int = 100
    save_interval: int = 1000
    log_interval: int = 10
    
    # Device configuration
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # Logging and monitoring
    use_tensorboard: bool = True
    tensorboard_dir: str = "runs"
    use_wandb: bool = False
    wandb_project: str = "nlp_training"
    
    # Checkpointing
    save_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    save_best_model: bool = True
    save_last_model: bool = True
    
    # Profiling
    enable_profiling: bool = False
    profile_interval: int = 100


class ComprehensiveDataset(Dataset):
    """Comprehensive dataset for different task types."""
    
    def __init__(
        self, 
        texts: List[str], 
        labels: Optional[List[Union[int, float]]] = None,
        tokenizer=None,
        max_length: int = 512,
        task_type: str = "classification"
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add labels if available
        if self.labels is not None:
            label = self.labels[idx]
            if self.task_type == "classification":
                item["labels"] = torch.tensor(label, dtype=torch.long)
            elif self.task_type == "regression":
                item["labels"] = torch.tensor(label, dtype=torch.float)
            else:  # text generation
                item["labels"] = item["input_ids"].clone()
        
        return item


class ComprehensiveTrainingSystem:
    """Comprehensive training system with robust gradient handling and evaluation."""
    
    def __init__(self, config: ComprehensiveTrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize components
        self._setup_gradient_system()
        self._setup_evaluation_system()
        self._setup_logging()
        
        # Training state
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if self.config.use_amp and torch.cuda.is_available() else None
        
        # Training statistics
        self.training_stats = {
            "epoch": 0,
            "step": 0,
            "total_loss": 0.0,
            "best_metric": float('inf'),
            "patience_counter": 0,
            "early_stopping_patience": 5
        }
        
        # Multi-GPU setup
        self._setup_multi_gpu()
    
    def _setup_device(self) -> torch.device:
        """Setup optimal device configuration."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        if device.type == "cuda":
            # GPU optimization settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.logger.info("Using CPU")
        
        return device
    
    def _setup_gradient_system(self):
        """Setup gradient monitoring and clipping system."""
        gradient_config = GradientConfig(
            gradient_clip_val=self.config.gradient_clip_val,
            gradient_clip_norm=self.config.gradient_clip_norm,
            gradient_clip_algorithm=self.config.gradient_clip_algorithm,
            check_gradients=self.config.check_gradients,
            check_weights=self.config.check_weights,
            check_loss=self.config.check_loss,
            zero_nan_gradients=self.config.zero_nan_gradients,
            skip_nan_batches=self.config.skip_nan_batches,
            restart_training_on_nan=self.config.restart_training_on_nan,
            max_nan_restarts=self.config.max_nan_restarts
        )
        
        self.gradient_monitor = GradientMonitor(gradient_config)
        self.gradient_clipper = GradientClipper(gradient_config)
    
    def _setup_evaluation_system(self):
        """Setup comprehensive evaluation system."""
        # Classification metrics
        classification_config = ClassificationMetricsConfig(
            compute_accuracy=True,
            compute_precision=True,
            compute_recall=True,
            compute_f1=True,
            compute_roc_auc=True,
            compute_pr_auc=True,
            compute_confusion_matrix=True,
            compute_classification_report=True,
            compute_cohen_kappa=True,
            compute_matthews_corrcoef=True
        )
        
        # Regression metrics
        regression_config = RegressionMetricsConfig(
            compute_mse=True,
            compute_mae=True,
            compute_rmse=True,
            compute_r2=True,
            compute_mape=True,
            compute_smape=True,
            compute_huber_loss=True,
            compute_correlation=True
        )
        
        # Text generation metrics
        text_generation_config = TextGenerationMetricsConfig(
            compute_perplexity=True,
            compute_bleu=True,
            compute_rouge=True,
            compute_meteor=True,
            compute_bert_score=True
        )
        
        # Custom metrics
        custom_config = CustomMetricsConfig(
            custom_metrics={
                "custom_f1": lambda y_true, y_pred: F.f1_score(y_true, y_pred, average='weighted', zero_division=0),
                "balanced_accuracy": lambda y_true, y_pred: (y_true == y_pred).mean()
            },
            metric_weights={
                "accuracy": 0.3,
                "f1": 0.3,
                "roc_auc": 0.2,
                "custom_f1": 0.2
            }
        )
        
        self.evaluation_system = ComprehensiveEvaluationSystem(
            classification_config=classification_config,
            regression_config=regression_config,
            text_generation_config=text_generation_config,
            custom_config=custom_config
        )
    
    def _setup_logging(self):
        """Setup logging and monitoring."""
        # Create directories
        os.makedirs(self.config.save_dir, exist_ok=True)
        os.makedirs(self.config.tensorboard_dir, exist_ok=True)
        
        # TensorBoard
        if self.config.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.config.tensorboard_dir)
        else:
            self.writer = None
        
        # Weights & Biases
        if self.config.use_wandb:
            import wandb
            wandb.init(project=self.config.wandb_project, config=vars(self.config))
            self.wandb = wandb
        else:
            self.wandb = None
    
    def _setup_multi_gpu(self):
        """Setup multi-GPU training."""
        if self.config.use_data_parallel and torch.cuda.device_count() > 1:
            self.logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        elif self.config.use_distributed:
            self.logger.info("Using DistributedDataParallel")
    
    def load_model(self, model_path: Optional[str] = None):
        """Load model with proper configuration."""
        try:
            if model_path and os.path.exists(model_path):
                # Load from checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model = checkpoint['model']
                self.optimizer = checkpoint['optimizer']
                self.scheduler = checkpoint['scheduler']
                self.training_stats = checkpoint['training_stats']
                self.logger.info(f"Loaded model from {model_path}")
            else:
                # Load from pretrained
                if self.config.model_type == "causal_lm":
                    self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
                elif self.config.model_type == "sequence_classification":
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.config.model_name, 
                        num_labels=self.config.num_labels
                    )
                else:
                    raise ValueError(f"Unknown model type: {self.config.model_type}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move to device
            self.model.to(self.device)
            
            # Multi-GPU setup
            if self.config.use_data_parallel and torch.cuda.device_count() > 1:
                self.model = DataParallel(self.model)
            elif self.config.use_distributed:
                self.model = DistributedDataParallel(self.model)
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def setup_training(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        """Setup training components."""
        # Create data loaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )
        
        if val_dataset:
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                persistent_workers=self.config.persistent_workers
            )
        else:
            self.val_dataloader = None
        
        # Setup optimizer
        if self.optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # Setup scheduler
        if self.scheduler is None:
            total_steps = len(self.train_dataloader) * self.config.num_epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
        
        self.logger.info("Training setup completed")
    
    def train_epoch(self, task_type: str = "classification") -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Training step
            step_metrics = self._training_step(batch, task_type)
            
            if step_metrics.get("skipped", False):
                continue
            
            epoch_loss += step_metrics["loss"]
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                self._log_training_step(batch_idx, step_metrics)
            
            # Evaluation
            if batch_idx % self.config.evaluation_interval == 0 and self.val_dataloader:
                val_metrics = self.evaluate(task_type)
                self._log_validation_step(batch_idx, val_metrics)
            
            # Checkpointing
            if batch_idx % self.config.save_interval == 0:
                self._save_checkpoint(f"checkpoint_step_{batch_idx}.pt")
            
            # Profiling
            if self.config.enable_profiling and batch_idx % self.config.profile_interval == 0:
                self._profile_training()
        
        # Update training stats
        self.training_stats["epoch"] += 1
        self.training_stats["total_loss"] += epoch_loss
        
        return {
            "epoch_loss": epoch_loss / len(self.train_dataloader),
            "epoch": self.training_stats["epoch"]
        }
    
    def _training_step(self, batch: Dict[str, torch.Tensor], task_type: str) -> Dict[str, float]:
        """Single training step with robust gradient handling."""
        self.optimizer.zero_grad()
        
        try:
            # Forward pass
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            # Check loss validity
            if not self.gradient_monitor.check_loss(loss):
                if self.config.skip_nan_batches:
                    return {"loss": float('inf'), "skipped": True}
                else:
                    raise ValueError("Invalid loss detected")
            
            # Check gradients
            grad_stats = self.gradient_monitor.check_gradients(self.model)
            
            # Check weights
            weight_stats = self.gradient_monitor.check_weights(self.model)
            
            # Clip gradients
            grad_norm = self.gradient_clipper.clip_gradients(self.model, self.optimizer)
            
            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update statistics
            self.training_stats["step"] += 1
            
            return {
                "loss": loss.item() * self.config.gradient_accumulation_steps,
                "grad_norm": grad_norm,
                "has_nan": grad_stats.get("has_nan", False),
                "has_inf": grad_stats.get("has_inf", False),
                "lr": self.optimizer.param_groups[0]["lr"]
            }
            
        except Exception as e:
            self.logger.error(f"Error in training step: {e}")
            return {"loss": float('inf'), "error": str(e)}
    
    def evaluate(self, task_type: str = "classification") -> Dict[str, float]:
        """Evaluate model with comprehensive metrics."""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                if task_type == "classification":
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(logits, dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(batch["labels"].cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                
                elif task_type == "regression":
                    predictions = outputs.logits.squeeze()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(batch["labels"].cpu().numpy())
                
                elif task_type == "text_generation":
                    # For text generation, we need to generate text
                    generated_texts = self._generate_text(batch)
                    all_predictions.extend(generated_texts)
                    all_targets.extend([batch["labels"].cpu().numpy()])
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities) if all_probabilities else None
        
        # Compute metrics
        metrics = {"loss": total_loss / len(self.val_dataloader)}
        
        if task_type == "classification":
            task_metrics = self.evaluation_system.evaluate_classification(
                all_targets, all_predictions, all_probabilities
            )
            metrics.update(task_metrics)
        
        elif task_type == "regression":
            task_metrics = self.evaluation_system.evaluate_regression(
                all_targets, all_predictions
            )
            metrics.update(task_metrics)
        
        elif task_type == "text_generation":
            task_metrics = self.evaluation_system.evaluate_text_generation(
                all_targets, all_predictions
            )
            metrics.update(task_metrics)
        
        return metrics
    
    def _generate_text(self, batch: Dict[str, torch.Tensor]) -> List[str]:
        """Generate text for evaluation."""
        # This is a placeholder implementation
        # In practice, you would use the model's generate method
        return ["Generated text"] * batch["input_ids"].size(0)
    
    def _log_training_step(self, batch_idx: int, step_metrics: Dict[str, float]):
        """Log training step metrics."""
        if self.writer:
            self.writer.add_scalar("Loss/Train", step_metrics["loss"], batch_idx)
            self.writer.add_scalar("Gradient/Norm", step_metrics["grad_norm"], batch_idx)
            self.writer.add_scalar("Learning_Rate", step_metrics["lr"], batch_idx)
        
        if self.wandb:
            self.wandb.log({
                "train_loss": step_metrics["loss"],
                "grad_norm": step_metrics["grad_norm"],
                "learning_rate": step_metrics["lr"],
                "step": batch_idx
            })
        
        self.logger.info(
            f"Step {batch_idx}: Loss={step_metrics['loss']:.4f}, "
            f"GradNorm={step_metrics['grad_norm']:.4f}, LR={step_metrics['lr']:.6f}"
        )
    
    def _log_validation_step(self, batch_idx: int, val_metrics: Dict[str, float]):
        """Log validation step metrics."""
        if self.writer:
            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, batch_idx)
        
        if self.wandb:
            self.wandb.log({f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})
        
        # Check for best model
        if "weighted_score" in val_metrics:
            if val_metrics["weighted_score"] < self.training_stats["best_metric"]:
                self.training_stats["best_metric"] = val_metrics["weighted_score"]
                self.training_stats["patience_counter"] = 0
                if self.config.save_best_model:
                    self._save_checkpoint("best_model.pt")
            else:
                self.training_stats["patience_counter"] += 1
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'training_stats': self.training_stats,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.config.save_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _profile_training(self):
        """Profile training performance."""
        if not self.config.enable_profiling:
            return
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            # Run a few training steps for profiling
            for i, batch in enumerate(self.train_dataloader):
                if i >= 10:  # Profile 10 steps
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self._training_step(batch, "classification")
        
        # Save profiling results
        prof.export_chrome_trace(os.path.join(self.config.save_dir, "training_profile.json"))
        self.logger.info("Training profile saved")
    
    def train(self, task_type: str = "classification") -> Dict[str, Any]:
        """Complete training loop."""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        training_history = []
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Train epoch
            epoch_metrics = self.train_epoch(task_type)
            
            # Evaluate
            if self.val_dataloader:
                val_metrics = self.evaluate(task_type)
                epoch_metrics.update(val_metrics)
            
            # Log epoch
            epoch_time = time.time() - epoch_start_time
            self._log_epoch(epoch, epoch_metrics, epoch_time)
            
            training_history.append(epoch_metrics)
            
            # Early stopping
            if self.training_stats["patience_counter"] >= self.training_stats["early_stopping_patience"]:
                self.logger.info("Early stopping triggered")
                break
        
        # Save final model
        if self.config.save_last_model:
            self._save_checkpoint("final_model.pt")
        
        # Close logging
        if self.writer:
            self.writer.close()
        if self.wandb:
            self.wandb.finish()
        
        return {
            "training_history": training_history,
            "final_metrics": training_history[-1] if training_history else {},
            "best_metric": self.training_stats["best_metric"]
        }
    
    def _log_epoch(self, epoch: int, epoch_metrics: Dict[str, float], epoch_time: float):
        """Log epoch metrics."""
        if self.writer:
            for metric_name, metric_value in epoch_metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.writer.add_scalar(f"Epoch/{metric_name}", metric_value, epoch)
        
        if self.wandb:
            self.wandb.log({
                f"epoch_{k}": v for k, v in epoch_metrics.items() if isinstance(v, (int, float))
            })
        
        self.logger.info(
            f"Epoch {epoch}: "
            f"TrainLoss={epoch_metrics.get('epoch_loss', 0):.4f}, "
            f"ValLoss={epoch_metrics.get('loss', 0):.4f}, "
            f"Time={epoch_time:.2f}s"
        )


# Example usage
def create_comprehensive_training_example():
    """Example of using the comprehensive training system."""
    
    # Configuration
    config = ComprehensiveTrainingConfig(
        model_name="gpt2",
        model_type="causal_lm",
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=3,
        gradient_clip_val=1.0,
        gradient_clip_norm=1.0,
        check_gradients=True,
        check_weights=True,
        check_loss=True,
        zero_nan_gradients=True,
        evaluation_interval=50,
        save_interval=200,
        log_interval=10,
        use_tensorboard=True,
        save_best_model=True,
        save_last_model=True
    )
    
    # Create training system
    training_system = ComprehensiveTrainingSystem(config)
    
    # Load model
    training_system.load_model()
    
    # Create sample datasets
    sample_texts = [
        "This is a positive example.",
        "This is a negative example.",
        "Another positive case here.",
        "Yet another negative instance."
    ] * 100  # Repeat to create more data
    
    sample_labels = [1, 0, 1, 0] * 100
    
    train_dataset = ComprehensiveDataset(
        texts=sample_texts[:300],
        labels=sample_labels[:300],
        task_type="classification"
    )
    
    val_dataset = ComprehensiveDataset(
        texts=sample_texts[300:],
        labels=sample_labels[300:],
        task_type="classification"
    )
    
    # Setup training
    training_system.setup_training(train_dataset, val_dataset)
    
    return training_system


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create example
    training_system = create_comprehensive_training_example()
    print("Comprehensive training system created successfully!")
    
    # Start training (uncomment to run)
    # results = training_system.train(task_type="classification")
    # print(f"Training completed: {results}")





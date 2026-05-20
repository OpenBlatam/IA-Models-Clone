"""
Complete Training Pipeline

Provides end-to-end training pipeline with all best practices:
- Data loading
- Model initialization
- Training loop
- Evaluation
- Checkpointing
- Logging
"""

import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..training import (
    LoRAFineTuner,
    Trainer,
    EarlyStopping,
    ModelCheckpoint
)
from ..evaluation import ModelEvaluator
from ..tracking import ExperimentTracker
from ..utils import (
    NaNInfDetector,
    MemoryProfiler,
    detect_anomaly
)
from ..data import TextDataset, TextPreprocessor

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline following all best practices
    """
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        device: Optional[torch.device] = None,
        use_lora: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize training pipeline
        
        Args:
            model_name: Name of base model
            num_labels: Number of classification labels
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            test_dataset: Test dataset (optional)
            device: Device to use
            use_lora: Whether to use LoRA fine-tuning
            config: Configuration dictionary
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lora = use_lora
        self.config = config or {}
        
        self.model = None
        self.trainer = None
        self.history = None
    
    def setup_model(self) -> None:
        """Setup model (LoRA or full)"""
        if self.use_lora:
            from ..training import LoRAFineTuner
            
            self.fine_tuner = LoRAFineTuner(
                model_name=self.model_name,
                r=self.config.get("lora_r", 16),
                alpha=self.config.get("lora_alpha", 32),
                dropout=self.config.get("lora_dropout", 0.1)
            )
            self.fine_tuner.load_model(num_labels=self.num_labels)
            self.model = self.fine_tuner.peft_model
        else:
            from ..training import FullFineTuner
            
            self.fine_tuner = FullFineTuner(
                model_name=self.model_name,
                num_labels=self.num_labels
            )
            self.fine_tuner.load_model()
            self.model = self.fine_tuner.model
        
        logger.info(f"Model setup complete: {self.model_name}")
    
    def setup_data_loaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Dict[str, DataLoader]:
        """Setup data loaders"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda"
        )
        
        loaders = {"train": train_loader}
        
        if self.val_dataset:
            loaders["val"] = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=self.device.type == "cuda"
            )
        
        if self.test_dataset:
            loaders["test"] = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=self.device.type == "cuda"
            )
        
        return loaders
    
    def train(
        self,
        num_epochs: int = 10,
        learning_rate: float = 2e-4,
        batch_size: int = 32,
        checkpoint_dir: Optional[str] = None,
        tracker: Optional[ExperimentTracker] = None,
        enable_debug: bool = False
    ) -> Dict[str, List[float]]:
        """
        Run complete training pipeline
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            checkpoint_dir: Directory for checkpoints
            tracker: Experiment tracker
            enable_debug: Enable debug mode
            
        Returns:
            Training history
        """
        # Setup
        self.setup_model()
        loaders = self.setup_data_loaders(batch_size=batch_size)
        
        # Check for issues
        if enable_debug:
            NaNInfDetector.check_model(self.model)
            MemoryProfiler.log_memory_stats(self.device, "Before training")
        
        # Setup callbacks
        early_stopping = None
        checkpoint = None
        
        if self.val_dataset:
            early_stopping = EarlyStopping(
                patience=self.config.get("early_stopping_patience", 5),
                mode="min"
            )
        
        if checkpoint_dir:
            checkpoint = ModelCheckpoint(
                save_dir=checkpoint_dir,
                save_best=True,
                monitor="val_loss" if self.val_dataset else "train_loss"
            )
        
        # Train
        if self.use_lora:
            # Use LoRA fine-tuner
            with detect_anomaly() if enable_debug else torch.no_grad():
                self.history = self.fine_tuner.train(
                    train_dataloader=loaders["train"],
                    val_dataloader=loaders.get("val"),
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    early_stopping=early_stopping,
                    tracker=tracker
                )
        else:
            # Use full trainer
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
            
            optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=self.config.get("weight_decay", 0.01)
            )
            
            criterion = nn.CrossEntropyLoss()
            
            total_steps = len(loaders["train"]) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.get("warmup_steps", 500),
                num_training_steps=total_steps
            )
            
            self.trainer = Trainer(
                model=self.model,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                scheduler=scheduler,
                early_stopping=early_stopping,
                checkpoint=checkpoint,
                tracker=tracker
            )
            
            self.history = self.trainer.fit(
                train_loader=loaders["train"],
                val_loader=loaders.get("val"),
                num_epochs=num_epochs
            )
        
        logger.info("Training completed successfully")
        return self.history
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Evaluation results
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        evaluator = ModelEvaluator(self.model, self.device)
        results = evaluator.evaluate(dataloader, compute_probs=True)
        
        return results
    
    def save_model(self, save_path: str) -> None:
        """Save trained model"""
        if self.fine_tuner:
            self.fine_tuner.save_model(save_path)
        elif self.model:
            torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
















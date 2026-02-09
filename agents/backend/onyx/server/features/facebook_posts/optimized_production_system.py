import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import pickle
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer, AutoModel
import gc

# Import our custom modules
from custom_nn_modules import (
    FacebookContentAnalysisTransformer,
    MultiModalFacebookAnalyzer,
    TemporalEngagementPredictor,
    AdaptiveContentOptimizer,
    FacebookDiffusionUNet
)

from forward_reverse_diffusion import (
    DiffusionConfig,
    BetaSchedule,
    ForwardDiffusionProcess,
    ReverseDiffusionProcess,
    DiffusionTraining,
    DiffusionVisualizer
)


@dataclass
class OptimizationConfig:
    """Configuration for the optimized production system"""
    # Model configurations
    transformer_config: Dict[str, Any] = None
    multimodal_config: Dict[str, Any] = None
    temporal_config: Dict[str, Any] = None
    adaptive_config: Dict[str, Any] = None
    diffusion_config: DiffusionConfig = None
    
    # Training configurations
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimization configurations
    use_mixed_precision: bool = True
    use_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 4
    use_early_stopping: bool = True
    patience: int = 10
    
    # Hardware configurations
    device: str = "auto"  # "auto", "cuda", "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging and monitoring
    use_wandb: bool = True
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500
    
    # Model saving
    save_dir: str = "models"
    model_name: str = "facebook_optimizer"
    
    def __post_init__(self):
        if self.transformer_config is None:
            self.transformer_config = {
                "vocab_size": 30522,
                "d_model": 768,
                "nhead": 12,
                "num_layers": 12,
                "dropout": 0.1
            }
        
        if self.multimodal_config is None:
            self.multimodal_config = {
                "text_model_name": "facebook/bart-base",
                "fusion_dim": 1024,
                "dropout": 0.1
            }
        
        if self.temporal_config is None:
            self.temporal_config = {
                "input_dim": 768,
                "hidden_dim": 256,
                "num_layers": 3,
                "sequence_length": 24
            }
        
        if self.adaptive_config is None:
            self.adaptive_config = {
                "feature_dim": 768,
                "hidden_dim": 512,
                "num_optimization_steps": 5
            }
        
        if self.diffusion_config is None:
            self.diffusion_config = DiffusionConfig(
                num_timesteps=1000,
                beta_schedule=BetaSchedule.COSINE,
                prediction_type="epsilon"
            )


class FacebookContentDataset(Dataset):
    """Dataset for Facebook content optimization"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "facebook/bart-base",
        max_length: int = 512,
        split: str = "train"
    ):
        self.data_path = data_path
        self.max_length = max_length
        self.split = split
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load and preprocess data"""
        # This is a placeholder - replace with actual data loading
        # For now, we'll create synthetic data
        synthetic_data = []
        for i in range(1000):
            synthetic_data.append({
                "text": f"This is a sample Facebook post {i} with some engaging content.",
                "engagement_score": np.random.uniform(0, 1),
                "viral_potential": np.random.uniform(0, 1),
                "content_quality": np.random.uniform(0, 1),
                "content_type": np.random.choice(["post", "story", "reel", "video", "live"]),
                "timestamp": time.time() + np.random.uniform(-86400, 86400),  # ±24 hours
                "performance_history": np.random.uniform(0, 1, 10).tolist()
            })
        return synthetic_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create content type embedding
        content_types = ["post", "story", "reel", "video", "live"]
        content_type_id = content_types.index(item["content_type"])
        
        # Create temporal features (hour of day, day of week)
        timestamp = item["timestamp"]
        hour = (timestamp // 3600) % 24
        day_of_week = (timestamp // 86400) % 7
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "content_type_id": torch.tensor(content_type_id, dtype=torch.long),
            "engagement_score": torch.tensor(item["engagement_score"], dtype=torch.float),
            "viral_potential": torch.tensor(item["viral_potential"], dtype=torch.float),
            "content_quality": torch.tensor(item["content_quality"], dtype=torch.float),
            "hour": torch.tensor(hour, dtype=torch.float),
            "day_of_week": torch.tensor(day_of_week, dtype=torch.float),
            "performance_history": torch.tensor(item["performance_history"], dtype=torch.float)
        }


class OptimizedFacebookProductionSystem:
    """
    Optimized production system for Facebook content optimization
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Initialize models
        self.models = self._initialize_models()
        
        # Initialize optimizers and schedulers
        self.optimizers = self._initialize_optimizers()
        self.schedulers = self._initialize_schedulers()
        
        # Initialize training components
        self.scaler = GradScaler() if config.use_mixed_precision else None
        self.criterion = self._initialize_loss_functions()
        
        # Initialize logging
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {}
        
    def _setup_device(self) -> torch.device:
        """Setup device for training"""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        logging.info(f"Using device: {device}")
        return device
    
    def _initialize_models(self) -> Dict[str, nn.Module]:
        """Initialize all models"""
        models = {}
        
        # Content Analysis Transformer
        models["transformer"] = FacebookContentAnalysisTransformer(
            **self.config.transformer_config
        ).to(self.device)
        
        # Multi-modal Analyzer
        models["multimodal"] = MultiModalFacebookAnalyzer(
            **self.config.multimodal_config
        ).to(self.device)
        
        # Temporal Engagement Predictor
        models["temporal"] = TemporalEngagementPredictor(
            **self.config.temporal_config
        ).to(self.device)
        
        # Adaptive Content Optimizer
        models["adaptive"] = AdaptiveContentOptimizer(
            **self.config.adaptive_config
        ).to(self.device)
        
        # Diffusion UNet
        models["diffusion"] = FacebookDiffusionUNet().to(self.device)
        
        # Log model parameters
        total_params = sum(p.numel() for model in models.values() for p in model.parameters())
        trainable_params = sum(p.numel() for model in models.values() for p in model.parameters() if p.requires_grad)
        
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        
        return models
    
    def _initialize_optimizers(self) -> Dict[str, optim.Optimizer]:
        """Initialize optimizers for all models"""
        optimizers = {}
        
        for name, model in self.models.items():
            optimizers[name] = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        return optimizers
    
    def _initialize_schedulers(self) -> Dict[str, optim.lr_scheduler._LRScheduler]:
        """Initialize learning rate schedulers"""
        schedulers = {}
        
        for name, optimizer in self.optimizers.items():
            schedulers[name] = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
        
        return schedulers
    
    def _initialize_loss_functions(self) -> Dict[str, nn.Module]:
        """Initialize loss functions"""
        return {
            "mse": nn.MSELoss(),
            "bce": nn.BCELoss(),
            "cross_entropy": nn.CrossEntropyLoss(),
            "l1": nn.L1Loss(),
            "huber": nn.HuberLoss()
        }
    
    def _setup_logging(self):
        """Setup logging and monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.model_name}_training.log"),
                logging.StreamHandler()
            ]
        )
        
        if self.config.use_wandb:
            wandb.init(
                project="facebook-content-optimizer",
                name=self.config.model_name,
                config=vars(self.config)
            )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self._set_models_mode(train=True)
        
        epoch_losses = {name: [] for name in self.models.keys()}
        epoch_metrics = {}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                with autocast():
                    losses, metrics = self._forward_pass(batch)
            else:
                losses, metrics = self._forward_pass(batch)
            
            # Backward pass
            total_loss = sum(losses.values())
            
            if self.config.use_mixed_precision:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(list(self.optimizers.values())[0])
                
                for optimizer in self.optimizers.values():
                    torch.nn.utils.clip_grad_norm_(
                        optimizer.param_groups[0]['params'],
                        self.config.max_grad_norm
                    )
                
                # Optimizer step
                for optimizer in self.optimizers.values():
                    if self.config.use_mixed_precision:
                        self.scaler.step(optimizer)
                    else:
                        optimizer.step()
                
                # Scheduler step
                for scheduler in self.schedulers.values():
                    scheduler.step()
                
                # Zero gradients
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad()
                
                if self.config.use_mixed_precision:
                    self.scaler.update()
                
                self.global_step += 1
            
            # Log losses
            for name, loss in losses.items():
                epoch_losses[name].append(loss.item())
            
            # Update progress bar
            avg_losses = {name: np.mean(losses) for name, losses in epoch_losses.items()}
            progress_bar.set_postfix(avg_losses)
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                self._log_training_step(avg_losses, metrics)
            
            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint()
        
        # Calculate epoch metrics
        for name, losses in epoch_losses.items():
            epoch_metrics[f"{name}_loss"] = np.mean(losses)
        
        return epoch_metrics
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Forward pass through all models"""
        losses = {}
        metrics = {}
        
        # 1. Content Analysis Transformer
        transformer_outputs = self.models["transformer"](
            batch["input_ids"],
            batch["attention_mask"],
            batch["content_type_id"].unsqueeze(1).expand(-1, batch["input_ids"].shape[1])
        )
        
        transformer_loss = (
            self.criterion["mse"](transformer_outputs["engagement_score"], batch["engagement_score"].unsqueeze(1)) +
            self.criterion["bce"](transformer_outputs["viral_potential"], batch["viral_potential"].unsqueeze(1)) +
            self.criterion["bce"](transformer_outputs["content_quality"], batch["content_quality"].unsqueeze(1))
        )
        losses["transformer"] = transformer_loss
        
        # 2. Multi-modal Analyzer
        text_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"]
        }
        multimodal_outputs = self.models["multimodal"](text_inputs)
        
        multimodal_loss = (
            self.criterion["mse"](multimodal_outputs["engagement_score"], batch["engagement_score"].unsqueeze(1)) +
            self.criterion["bce"](multimodal_outputs["viral_potential"], batch["viral_potential"].unsqueeze(1)) +
            self.criterion["bce"](multimodal_outputs["content_quality"], batch["content_quality"].unsqueeze(1))
        )
        losses["multimodal"] = multimodal_loss
        
        # 3. Temporal Engagement Predictor
        content_features = transformer_outputs["pooled_output"]
        temporal_outputs = self.models["temporal"](content_features)
        
        temporal_loss = (
            self.criterion["mse"](temporal_outputs["engagement_score"], batch["engagement_score"].unsqueeze(1)) +
            self.criterion["mse"](temporal_outputs["peak_time_probability"], batch["engagement_score"].unsqueeze(1))
        )
        losses["temporal"] = temporal_loss
        
        # 4. Adaptive Content Optimizer
        adaptive_outputs = self.models["adaptive"](content_features, batch["performance_history"])
        
        adaptive_loss = (
            self.criterion["mse"](adaptive_outputs["predicted_performance"], batch["engagement_score"].unsqueeze(1)) +
            self.criterion["cross_entropy"](adaptive_outputs["content_type_probabilities"], batch["content_type_id"])
        )
        losses["adaptive"] = adaptive_loss
        
        # 5. Diffusion Model (if we have image data)
        # For now, we'll skip diffusion training as we don't have image data
        # In a real implementation, you would train the diffusion model separately
        
        # Calculate metrics
        for name, outputs in [("transformer", transformer_outputs), ("multimodal", multimodal_outputs)]:
            if "engagement_score" in outputs:
                pred = outputs["engagement_score"].squeeze()
                true = batch["engagement_score"]
                metrics[f"{name}_engagement_mae"] = F.l1_loss(pred, true).item()
                metrics[f"{name}_engagement_mse"] = F.mse_loss(pred, true).item()
        
        return losses, metrics
    
    def _set_models_mode(self, train: bool = True):
        """Set all models to train or eval mode"""
        for model in self.models.values():
            model.train() if train else model.eval()
    
    def _log_training_step(self, losses: Dict[str, float], metrics: Dict[str, float]):
        """Log training step"""
        log_dict = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            **losses,
            **metrics
        }
        
        # Add learning rates
        for name, scheduler in self.schedulers.items():
            log_dict[f"{name}_lr"] = scheduler.get_last_lr()[0]
        
        if self.config.use_wandb:
            wandb.log(log_dict)
        
        logging.info(f"Step {self.global_step}: {log_dict}")
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_metrics": self.best_metrics,
            "config": self.config,
            "models": {name: model.state_dict() for name, model in self.models.items()},
            "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            "schedulers": {name: scheduler.state_dict() for name, scheduler in self.schedulers.items()}
        }
        
        checkpoint_path = save_dir / f"{self.config.model_name}_step_{self.global_step}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only the latest checkpoint
        checkpoints = list(save_dir.glob(f"{self.config.model_name}_step_*.pth"))
        if len(checkpoints) > 3:  # Keep only 3 most recent checkpoints
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            for checkpoint_file in checkpoints[:-3]:
                checkpoint_file.unlink()
        
        logging.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_metrics = checkpoint["best_metrics"]
        
        # Load model states
        for name, state_dict in checkpoint["models"].items():
            self.models[name].load_state_dict(state_dict)
        
        # Load optimizer states
        for name, state_dict in checkpoint["optimizers"].items():
            self.optimizers[name].load_state_dict(state_dict)
        
        # Load scheduler states
        for name, state_dict in checkpoint["schedulers"].items():
            self.schedulers[name].load_state_dict(state_dict)
        
        logging.info(f"Loaded checkpoint: {checkpoint_path}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop"""
        logging.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                
                # Early stopping
                if self.config.use_early_stopping:
                    if self._should_stop_early(val_metrics):
                        logging.info("Early stopping triggered")
                        break
            
            # Log epoch metrics
            epoch_log = {
                "epoch": epoch,
                **train_metrics
            }
            
            if val_loader is not None:
                epoch_log.update({f"val_{k}": v for k, v in val_metrics.items()})
            
            if self.config.use_wandb:
                wandb.log(epoch_log)
            
            logging.info(f"Epoch {epoch} completed: {epoch_log}")
        
        logging.info("Training completed!")
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate models on validation set"""
        self._set_models_mode(train=False)
        
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                losses, metrics = self._forward_pass(batch)
                
                # Accumulate metrics
                for name, value in metrics.items():
                    if name not in total_metrics:
                        total_metrics[name] = 0
                    total_metrics[name] += value
                
                num_batches += 1
        
        # Average metrics
        avg_metrics = {name: value / num_batches for name, value in total_metrics.items()}
        
        return avg_metrics
    
    def _should_stop_early(self, val_metrics: Dict[str, float]) -> bool:
        """Check if training should stop early"""
        # Simple early stopping based on validation loss
        val_loss = val_metrics.get("transformer_loss", float('inf'))
        
        if "best_val_loss" not in self.best_metrics:
            self.best_metrics["best_val_loss"] = val_loss
            self.best_metrics["patience_counter"] = 0
            return False
        
        if val_loss < self.best_metrics["best_val_loss"]:
            self.best_metrics["best_val_loss"] = val_loss
            self.best_metrics["patience_counter"] = 0
        else:
            self.best_metrics["patience_counter"] += 1
        
        return self.best_metrics["patience_counter"] >= self.config.patience
    
    def optimize_content(self, text: str, content_type: str = "post") -> Dict[str, Any]:
        """Optimize content using trained models"""
        self._set_models_mode(train=False)
        
        # Tokenize text
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        encoding = tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Create content type ID
        content_types = ["post", "story", "reel", "video", "live"]
        content_type_id = torch.tensor([content_types.index(content_type)], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # 1. Content Analysis
            transformer_outputs = self.models["transformer"](
                input_ids,
                attention_mask,
                content_type_id.unsqueeze(1).expand(-1, input_ids.shape[1])
            )
            
            # 2. Multi-modal Analysis
            text_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            multimodal_outputs = self.models["multimodal"](text_inputs)
            
            # 3. Temporal Analysis
            content_features = transformer_outputs["pooled_output"]
            temporal_outputs = self.models["temporal"](content_features)
            
            # 4. Adaptive Optimization
            # Create dummy performance history
            performance_history = torch.zeros(1, 10).to(self.device)
            adaptive_outputs = self.models["adaptive"](content_features, performance_history)
        
        # Compile results
        optimization_results = {
            "original_text": text,
            "content_type": content_type,
            "engagement_score": transformer_outputs["engagement_score"].item(),
            "viral_potential": transformer_outputs["viral_potential"].item(),
            "content_quality": transformer_outputs["content_quality"].item(),
            "peak_time_probability": temporal_outputs["peak_time_probability"].item(),
            "predicted_performance": adaptive_outputs["predicted_performance"].item(),
            "content_type_probabilities": adaptive_outputs["content_type_probabilities"].cpu().numpy(),
            "optimization_suggestions": adaptive_outputs["optimization_suggestions"].cpu().numpy(),
            "recommended_content_type": content_types[adaptive_outputs["content_type_probabilities"].argmax().item()]
        }
        
        return optimization_results


def main():
    """Main training script"""
    # Configuration
    config = OptimizationConfig(
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=50,
        use_mixed_precision=True,
        use_wandb=True
    )
    
    # Initialize system
    system = OptimizedFacebookProductionSystem(config)
    
    # Create datasets
    train_dataset = FacebookContentDataset("train_data.json", split="train")
    val_dataset = FacebookContentDataset("val_data.json", split="val")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Train
    system.train(train_loader, val_loader)
    
    # Test optimization
    test_text = "Check out our amazing new product! 🚀"
    results = system.optimize_content(test_text, "post")
    print("Optimization Results:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()



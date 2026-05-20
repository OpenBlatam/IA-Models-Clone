"""
Model Training and Evaluation System
Implements efficient data loading, splits, cross-validation, early stopping, LR scheduling, metrics, and optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import logging
import json
import os
from pathlib import Path
from tqdm import tqdm
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class EarlyStopping:
    """Early stopping mechanism to prevent overfitting"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class LearningRateScheduler:
    """Advanced learning rate scheduling with multiple strategies"""
    
    def __init__(self, optimizer: optim.Optimizer, scheduler_type: str = "cosine", **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.scheduler = self._create_scheduler(**kwargs)
        
    def _create_scheduler(self, **kwargs):
        if self.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=kwargs.get("T_max", 100),
                eta_min=kwargs.get("eta_min", 1e-6)
            )
        elif self.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get("step_size", 30),
                gamma=kwargs.get("gamma", 0.1)
            )
        elif self.scheduler_type == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=kwargs.get("gamma", 0.95)
            )
        elif self.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=kwargs.get("mode", "min"),
                factor=kwargs.get("factor", 0.1),
                patience=kwargs.get("patience", 10),
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def step(self, metrics: Optional[float] = None):
        """Step the scheduler"""
        if self.scheduler_type == "plateau" and metrics is not None:
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rate"""
        return self.scheduler.get_last_lr()

class MetricsTracker:
    """Comprehensive metrics tracking and visualization"""
    
    def __init__(self, save_dir: str = "./metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.metrics_history = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
            "learning_rate": [], "gradient_norm": [], "epoch_time": []
        }
        
    def update(self, metrics: Dict[str, float], epoch: int):
        """Update metrics for current epoch"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
                
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.metrics_history["train_loss"], label="Train Loss")
        axes[0, 0].plot(self.metrics_history["val_loss"], label="Val Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.metrics_history["train_acc"], label="Train Acc")
        axes[0, 1].plot(self.metrics_history["val_acc"], label="Val Acc")
        axes[0, 1].set_title("Training and Validation Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.metrics_history["learning_rate"])
        axes[1, 0].set_title("Learning Rate")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("LR")
        axes[1, 0].grid(True)
        
        # Gradient norm
        axes[1, 1].plot(self.metrics_history["gradient_norm"])
        axes[1, 1].set_title("Gradient Norm")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Norm")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_metrics(self, filename: str = "training_metrics.json"):
        """Save metrics to JSON file"""
        metrics_file = self.save_dir / filename
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

class DataSplitter:
    """Advanced data splitting with cross-validation support"""
    
    @staticmethod
    def train_val_test_split(dataset: Dataset, 
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15,
                            random_seed: int = 42) -> Tuple[Subset, Subset, Subset]:
        """Split dataset into train/val/test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def k_fold_split(dataset: Dataset, n_splits: int = 5, random_seed: int = 42) -> List[Tuple[Subset, Subset]]:
        """Create k-fold cross-validation splits"""
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        splits = []
        
        for train_idx, val_idx in kfold.split(dataset):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            splits.append((train_subset, val_subset))
            
        return splits

class GradientClipper:
    """Gradient clipping utilities"""
    
    @staticmethod
    def clip_gradients(model: nn.Module, max_norm: float = 1.0):
        """Clip gradients to prevent exploding gradients"""
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    @staticmethod
    def get_gradient_norm(model: nn.Module) -> float:
        """Calculate gradient norm for monitoring"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

class NaNInfHandler:
    """Handle NaN and Inf values during training"""
    
    @staticmethod
    def check_tensor(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Check if tensor contains NaN or Inf values"""
        if torch.isnan(tensor).any():
            warnings.warn(f"NaN detected in {name}")
            return False
        if torch.isinf(tensor).any():
            warnings.warn(f"Inf detected in {name}")
            return False
        return True
    
    @staticmethod
    def check_model_parameters(model: nn.Module) -> bool:
        """Check all model parameters for NaN/Inf"""
        for name, param in model.named_parameters():
            if not NaNInfHandler.check_tensor(param.data, f"parameter {name}"):
                return False
        return True
    
    @staticmethod
    def check_gradients(model: nn.Module) -> bool:
        """Check gradients for NaN/Inf"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not NaNInfHandler.check_tensor(param.grad, f"gradient {name}"):
                    return False
        return True

class ModelTrainer:
    """Comprehensive model training with all optimizations"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 scheduler_type: str = "cosine",
                 early_stopping_patience: int = 7,
                 gradient_clip_norm: float = 1.0,
                 mixed_precision: bool = True,
                 log_dir: str = "./logs"):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip_norm = gradient_clip_norm
        
        # Setup components
        self.scheduler = LearningRateScheduler(optimizer, scheduler_type)
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.metrics_tracker = MetricsTracker()
        self.writer = SummaryWriter(log_dir)
        
        # Mixed precision setup
        self.mixed_precision = mixed_precision
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                # Check for NaN/Inf in gradients
                if not NaNInfHandler.check_gradients(self.model):
                    self.logger.warning("NaN/Inf detected in gradients, skipping batch")
                    continue
                
                # Clip gradients
                self.scaler.unscale_(self.optimizer)
                GradientClipper.clip_gradients(self.model, self.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # Check for NaN/Inf in gradients
                if not NaNInfHandler.check_gradients(self.model):
                    self.logger.warning("NaN/Inf detected in gradients, skipping batch")
                    continue
                
                # Clip gradients
                GradientClipper.clip_gradients(self.model, self.gradient_clip_norm)
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {"train_loss": avg_loss, "train_acc": accuracy}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return {"val_loss": avg_loss, "val_acc": accuracy}
    
    def train(self, num_epochs: int, save_path: str = "./checkpoints") -> Dict[str, List[float]]:
        """Complete training loop"""
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            if isinstance(self.scheduler.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["val_loss"])
            else:
                self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Get gradient norm
            gradient_norm = GradientClipper.get_gradient_norm(self.model)
            
            # Combine metrics
            epoch_metrics = {
                **train_metrics, **val_metrics,
                "learning_rate": current_lr,
                "gradient_norm": gradient_norm
            }
            
            # Update metrics tracker
            self.metrics_tracker.update(epoch_metrics, epoch)
            
            # Log to tensorboard
            for key, value in epoch_metrics.items():
                self.writer.add_scalar(key, value, epoch)
            
            # Log to console
            self.logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                           f"Train Acc: {train_metrics['train_acc']:.2f}%, "
                           f"Val Loss: {val_metrics['val_loss']:.4f}, "
                           f"Val Acc: {val_metrics['val_acc']:.2f}%, "
                           f"LR: {current_lr:.6f}")
            
            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'metrics': epoch_metrics
                }, save_path / "best_model.pth")
                self.logger.info("Saved best model")
            
            # Check early stopping
            if self.early_stopping(val_metrics["val_loss"], self.model):
                self.logger.info("Early stopping triggered")
                break
            
            # Check for NaN/Inf in model parameters
            if not NaNInfHandler.check_model_parameters(self.model):
                self.logger.error("NaN/Inf detected in model parameters, stopping training")
                break
        
        # Save final metrics
        self.metrics_tracker.save_metrics()
        self.metrics_tracker.plot_metrics()
        
        # Cleanup
        self.writer.close()
        
        return self.metrics_tracker.metrics_history

class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics"""
    
    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set"""
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        metrics = {
            "test_loss": avg_loss,
            "test_accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, save_path="./confusion_matrix.png")
        
        return metrics
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Example usage of the training and evaluation system"""
    
    # This is a placeholder - in practice you would use your actual dataset
    class DummyDataset(Dataset):
        def __init__(self, size=1000, input_dim=784, num_classes=10):
            self.data = torch.randn(size, input_dim)
            self.targets = torch.randint(0, num_classes, (size,))
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    # Create dummy dataset
    dataset = DummyDataset()
    
    # Split data
    train_dataset, val_dataset, test_dataset = DataSplitter.train_val_test_split(dataset)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model (simple MLP for demonstration)
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10)
    )
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler_type="cosine",
        early_stopping_patience=5,
        gradient_clip_norm=1.0,
        mixed_precision=True
    )
    
    # Train model
    print("Starting training...")
    metrics_history = trainer.train(num_epochs=20)
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = ModelEvaluator(model)
    test_metrics = evaluator.evaluate(test_loader)
    
    print("Test Metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()



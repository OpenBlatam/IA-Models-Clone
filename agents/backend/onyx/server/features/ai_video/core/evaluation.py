from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .models import BaseVideoModel, ModelConfig
from .data_loader import DataConfig
    from .models import ModelConfig, create_model
    from .data_loader import DataConfig, create_train_val_test_loaders
from typing import Any, List, Dict, Optional
import asyncio
"""
AI Video Evaluation Module
==========================

This module provides a modular structure for evaluating AI video generation models,
including metrics calculation, result analysis, and performance assessment.

Features:
- Multiple evaluation metrics (PSNR, SSIM, FID, etc.)
- Comprehensive result analysis
- Visualization and reporting
- Model comparison utilities
- Performance benchmarking
"""



# Import local modules

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Evaluation parameters
    batch_size: int = 8
    num_samples: Optional[int] = None  # None for all samples
    device: str = "cuda"
    
    # Metrics to compute
    compute_psnr: bool = True
    compute_ssim: bool = True
    compute_lpips: bool = True
    compute_fid: bool = False  # Requires additional setup
    
    # Output parameters
    save_results: bool = True
    save_videos: bool = False
    output_dir: str = "evaluation_results"
    
    # Visualization parameters
    create_plots: bool = True
    plot_samples: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'batch_size': self.batch_size,
            'num_samples': self.num_samples,
            'device': self.device,
            'compute_psnr': self.compute_psnr,
            'compute_ssim': self.compute_ssim,
            'compute_lpips': self.compute_lpips,
            'compute_fid': self.compute_fid,
            'save_results': self.save_results,
            'save_videos': self.save_videos,
            'output_dir': self.output_dir,
            'create_plots': self.create_plots,
            'plot_samples': self.plot_samples
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EvaluationConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'EvaluationConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class BaseMetric(ABC):
    """Base class for evaluation metrics."""
    
    def __init__(self, config: EvaluationConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device(config.device)
    
    @abstractmethod
    def compute(self, predictions: Tensor, targets: Tensor) -> float:
        """Compute metric value. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get metric name. Must be implemented by subclasses."""
        pass


class PSNRMetric(BaseMetric):
    """Peak Signal-to-Noise Ratio metric."""
    
    def get_name(self) -> str:
        return "PSNR"
    
    def compute(self, predictions: Tensor, targets: Tensor) -> float:
        """Compute PSNR between predictions and targets."""
        # Ensure values are in [0, 1] range
        predictions = torch.clamp(predictions, 0, 1)
        targets = torch.clamp(targets, 0, 1)
        
        # Convert to [0, 255] range for PSNR calculation
        predictions = predictions * 255
        targets = targets * 255
        
        # Compute MSE
        mse = F.mse_loss(predictions, targets)
        
        # Avoid division by zero
        if mse == 0:
            return float('inf')
        
        # Compute PSNR
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
        
        return psnr.item()


class SSIMMetric(BaseMetric):
    """Structural Similarity Index metric."""
    
    def get_name(self) -> str:
        return "SSIM"
    
    def compute(self, predictions: Tensor, targets: Tensor) -> float:
        """Compute SSIM between predictions and targets."""
        # Ensure values are in [0, 1] range
        predictions = torch.clamp(predictions, 0, 1)
        targets = torch.clamp(targets, 0, 1)
        
        # Compute SSIM for each frame and average
        ssim_values = []
        
        for i in range(predictions.shape[2]):  # Iterate over frames
            pred_frame = predictions[:, :, i, :, :]
            target_frame = targets[:, :, i, :, :]
            
            ssim = self._compute_ssim_frame(pred_frame, target_frame)
            ssim_values.append(ssim)
        
        return np.mean(ssim_values)
    
    def _compute_ssim_frame(self, pred: Tensor, target: Tensor) -> float:
        """Compute SSIM for a single frame."""
        # Simplified SSIM implementation
        # In practice, you might want to use a more sophisticated implementation
        
        # Convert to numpy for easier computation
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Compute means
        mu_pred = np.mean(pred_np)
        mu_target = np.mean(target_np)
        
        # Compute variances
        var_pred = np.var(pred_np)
        var_target = np.var(target_np)
        
        # Compute covariance
        cov = np.mean((pred_np - mu_pred) * (target_np - mu_target))
        
        # SSIM parameters
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # Compute SSIM
        numerator = (2 * mu_pred * mu_target + c1) * (2 * cov + c2)
        denominator = (mu_pred ** 2 + mu_target ** 2 + c1) * (var_pred + var_target + c2)
        
        ssim = numerator / denominator
        
        return ssim


class LPIPSMetric(BaseMetric):
    """Learned Perceptual Image Patch Similarity metric."""
    
    def __init__(self, config: EvaluationConfig):
        
    """__init__ function."""
super().__init__(config)
        # Load pre-trained LPIPS model (simplified version)
        self.lpips_model = self._load_lpips_model()
    
    def get_name(self) -> str:
        return "LPIPS"
    
    def _load_lpips_model(self) -> nn.Module:
        """Load LPIPS model (simplified implementation)."""
        # Simplified LPIPS-like model
        # In practice, you would use the actual LPIPS implementation
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)
    
    def compute(self, predictions: Tensor, targets: Tensor) -> float:
        """Compute LPIPS between predictions and targets."""
        # Ensure values are in [0, 1] range
        predictions = torch.clamp(predictions, 0, 1)
        targets = torch.clamp(targets, 0, 1)
        
        # Compute LPIPS for each frame and average
        lpips_values = []
        
        for i in range(predictions.shape[2]):  # Iterate over frames
            pred_frame = predictions[:, :, i, :, :]
            target_frame = targets[:, :, i, :, :]
            
            with torch.no_grad():
                lpips = self.lpips_model(pred_frame) - self.lpips_model(target_frame)
                lpips = torch.mean(torch.abs(lpips))
                lpips_values.append(lpips.item())
        
        return np.mean(lpips_values)


class MetricFactory:
    """Factory class for creating evaluation metrics."""
    
    _metrics = {
        'psnr': PSNRMetric,
        'ssim': SSIMMetric,
        'lpips': LPIPSMetric
    }
    
    @classmethod
    def create_metric(cls, metric_type: str, config: EvaluationConfig) -> BaseMetric:
        """Create a metric instance."""
        if metric_type not in cls._metrics:
            raise ValueError(f"Unknown metric type: {metric_type}. Available: {list(cls._metrics.keys())}")
        
        metric_class = cls._metrics[metric_type]
        return metric_class(config)
    
    @classmethod
    def create_all_metrics(cls, config: EvaluationConfig) -> List[BaseMetric]:
        """Create all enabled metrics."""
        metrics = []
        
        if config.compute_psnr:
            metrics.append(cls.create_metric('psnr', config))
        if config.compute_ssim:
            metrics.append(cls.create_metric('ssim', config))
        if config.compute_lpips:
            metrics.append(cls.create_metric('lpips', config))
        
        return metrics
    
    @classmethod
    def get_available_metrics(cls) -> List[str]:
        """Get list of available metric types."""
        return list(cls._metrics.keys())


class VideoEvaluator:
    """Main evaluation class for AI video models."""
    
    def __init__(self, 
                 model: BaseVideoModel,
                 test_loader: DataLoader,
                 config: EvaluationConfig = None):
        
        
    """__init__ function."""
self.model = model
        self.test_loader = test_loader
        self.config = config or EvaluationConfig()
        
        # Initialize metrics
        self.metrics = MetricFactory.create_all_metrics(self.config)
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {
            'metrics': {},
            'samples': [],
            'metadata': {}
        }
        
        logger.info(f"Initialized evaluator with {len(self.metrics)} metrics")
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model and return results."""
        logger.info("Starting model evaluation")
        
        # Set model to evaluation mode
        self.model.eval_mode()
        
        # Initialize metric accumulators
        metric_values = {metric.get_name(): [] for metric in self.metrics}
        samples = []
        
        # Evaluate on test set
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                # Limit number of samples if specified
                if self.config.num_samples and batch_idx * self.config.batch_size >= self.config.num_samples:
                    break
                
                # Get batch data
                videos = batch['video'].to(self.model.device)
                
                # Generate predictions
                predictions = self.model(videos)
                
                # Compute metrics
                for metric in self.metrics:
                    metric_value = metric.compute(predictions, videos)
                    metric_values[metric.get_name()].append(metric_value)
                
                # Store sample results
                if batch_idx < self.config.plot_samples:
                    sample = {
                        'predictions': predictions.cpu(),
                        'targets': videos.cpu(),
                        'batch_idx': batch_idx
                    }
                    samples.append(sample)
                
                # Save videos if requested
                if self.config.save_videos:
                    self._save_videos(predictions, videos, batch_idx)
        
        # Compute average metrics
        avg_metrics = {}
        for metric_name, values in metric_values.items():
            avg_metrics[metric_name] = np.mean(values)
            avg_metrics[f"{metric_name}_std"] = np.std(values)
        
        # Store results
        self.results['metrics'] = avg_metrics
        self.results['samples'] = samples
        self.results['metadata'] = {
            'evaluation_date': datetime.now().isoformat(),
            'num_samples': len(metric_values[list(metric_values.keys())[0]]),
            'model_info': self.model.get_model_info(),
            'config': self.config.to_dict()
        }
        
        # Save results
        if self.config.save_results:
            self._save_results()
        
        # Create visualizations
        if self.config.create_plots:
            self._create_plots()
        
        logger.info(f"Evaluation completed. Results: {avg_metrics}")
        return self.results
    
    def _save_videos(self, predictions: Tensor, targets: Tensor, batch_idx: int) -> None:
        """Save generated and target videos."""
        videos_dir = self.output_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        # Save first video from batch
        pred_video = predictions[0].cpu().numpy()
        target_video = targets[0].cpu().numpy()
        
        # Convert to uint8
        pred_video = (pred_video * 255).astype(np.uint8)
        target_video = (target_video * 255).astype(np.uint8)
        
        # Save as video files
        pred_path = videos_dir / f"pred_batch_{batch_idx}.mp4"
        target_path = videos_dir / f"target_batch_{batch_idx}.mp4"
        
        self._save_video_file(pred_video, pred_path)
        self._save_video_file(target_video, target_path)
    
    def _save_video_file(self, video: np.ndarray, filepath: Path) -> None:
        """Save video array to file."""
        # video shape: (frames, channels, height, width)
        frames, channels, height, width = video.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(filepath), fourcc, 30.0, (width, height))
        
        try:
            for frame in video:
                # Convert to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        finally:
            out.release()
    
    def _save_results(self) -> None:
        """Save evaluation results to files."""
        # Save metrics as JSON
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.results, f, indent=2, default=str)
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame([self.results['metrics']])
        metrics_file = self.output_dir / "metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _create_plots(self) -> None:
        """Create visualization plots."""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Create metric comparison plot
        self._plot_metrics(plots_dir)
        
        # Create sample comparison plots
        self._plot_samples(plots_dir)
    
    def _plot_metrics(self, plots_dir: Path) -> None:
        """Create metrics visualization."""
        metrics = self.results['metrics']
        
        # Filter out std metrics for plotting
        plot_metrics = {k: v for k, v in metrics.items() if not k.endswith('_std')}
        
        plt.figure(figsize=(10, 6))
        plt.bar(plot_metrics.keys(), plot_metrics.values())
        plt.title('Model Evaluation Metrics')
        plt.ylabel('Metric Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_path = plots_dir / "metrics_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_samples(self, plots_dir: Path) -> None:
        """Create sample comparison plots."""
        samples = self.results['samples']
        
        for i, sample in enumerate(samples):
            predictions = sample['predictions']
            targets = sample['targets']
            
            # Create comparison plot for first frame
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot prediction
            pred_frame = predictions[0, :, 0, :, :].permute(1, 2, 0)
            pred_frame = torch.clamp(pred_frame, 0, 1)
            axes[0].imshow(pred_frame)
            axes[0].set_title('Prediction')
            axes[0].axis('off')
            
            # Plot target
            target_frame = targets[0, :, 0, :, :].permute(1, 2, 0)
            target_frame = torch.clamp(target_frame, 0, 1)
            axes[1].imshow(target_frame)
            axes[1].set_title('Target')
            axes[1].axis('off')
            
            plt.tight_layout()
            
            plot_path = plots_dir / f"sample_comparison_{i}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def compare_models(self, other_results: Dict[str, Any], model_names: List[str]) -> Dict[str, Any]:
        """Compare results between multiple models."""
        comparison = {
            'model_names': model_names,
            'metrics_comparison': {},
            'metadata': {
                'comparison_date': datetime.now().isoformat(),
                'num_models': len(model_names)
            }
        }
        
        # Compare metrics
        all_metrics = set(self.results['metrics'].keys())
        for metric in all_metrics:
            if not metric.endswith('_std'):
                comparison['metrics_comparison'][metric] = {
                    model_names[0]: self.results['metrics'][metric],
                    model_names[1]: other_results['metrics'][metric]
                }
        
        # Save comparison
        comparison_file = self.output_dir / "model_comparison.json"
        with open(comparison_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(comparison, f, indent=2, default=str)
        
        # Create comparison plot
        self._plot_model_comparison(comparison)
        
        return comparison
    
    def _plot_model_comparison(self, comparison: Dict[str, Any]) -> None:
        """Create model comparison plot."""
        metrics = comparison['metrics_comparison']
        model_names = comparison['model_names']
        
        # Prepare data for plotting
        metric_names = list(metrics.keys())
        model1_values = [metrics[m][model_names[0]] for m in metric_names]
        model2_values = [metrics[m][model_names[1]] for m in metric_names]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, model1_values, width, label=model_names[0])
        plt.bar(x + width/2, model2_values, width, label=model_names[1])
        
        plt.xlabel('Metrics')
        plt.ylabel('Metric Value')
        plt.title('Model Comparison')
        plt.xticks(x, metric_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        plot_path = self.output_dir / "model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


# Convenience functions
def create_evaluator(model: BaseVideoModel,
                    test_loader: DataLoader,
                    config: EvaluationConfig = None) -> VideoEvaluator:
    """Create an evaluator instance."""
    return VideoEvaluator(model, test_loader, config)


def evaluate_model(model: BaseVideoModel,
                  test_loader: DataLoader,
                  config: EvaluationConfig = None) -> Dict[str, Any]:
    """Evaluate a model and return results."""
    evaluator = create_evaluator(model, test_loader, config)
    return evaluator.evaluate()


def compare_models(model1: BaseVideoModel,
                  model2: BaseVideoModel,
                  test_loader: DataLoader,
                  config: EvaluationConfig = None,
                  model_names: List[str] = None) -> Dict[str, Any]:
    """Compare two models."""
    if model_names is None:
        model_names = ["Model 1", "Model 2"]
    
    # Evaluate both models
    evaluator1 = create_evaluator(model1, test_loader, config)
    results1 = evaluator1.evaluate()
    
    evaluator2 = create_evaluator(model2, test_loader, config)
    results2 = evaluator2.evaluate()
    
    # Compare results
    comparison = evaluator1.compare_models(results2, model_names)
    
    return comparison


if __name__ == "__main__":
    # Example usage
    
    # Create model
    model_config = ModelConfig(
        model_type="diffusion",
        model_name="test_model",
        frame_size=(64, 64),
        num_frames=8
    )
    model = create_model("diffusion", model_config)
    
    # Create data loaders
    data_config = DataConfig(
        data_dir="data/videos",
        frame_size=(64, 64),
        num_frames=8,
        batch_size=4
    )
    loaders = create_train_val_test_loaders("video_file", data_config)
    
    # Create evaluation config
    eval_config = EvaluationConfig(
        batch_size=4,
        num_samples=20,
        compute_psnr=True,
        compute_ssim=True,
        compute_lpips=True
    )
    
    # Evaluate model
    results = evaluate_model(model, loaders['test'], eval_config)
    print(f"Evaluation completed. Results: {results['metrics']}") 
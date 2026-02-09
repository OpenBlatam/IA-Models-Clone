"""
Optimized Evaluation System for Video-OpusClip

Comprehensive evaluation metrics for video processing tasks including
caption generation, viral prediction, video quality assessment, and performance analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import structlog
from dataclasses import dataclass, field
import time
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import cv2
from PIL import Image
import warnings
from collections import defaultdict

logger = structlog.get_logger()

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # General evaluation settings
    batch_size: int = 32
    device: str = "cuda"
    num_workers: int = 4
    
    # Task-specific settings
    task_type: str = "caption_generation"  # 'caption', 'viral_prediction', 'video_quality'
    
    # Caption evaluation
    bleu_weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)  # BLEU-4 weights
    use_meteor: bool = True
    use_rouge: bool = True
    rouge_metrics: List[str] = field(default_factory=lambda: ['rouge1', 'rouge2', 'rougeL'])
    
    # Viral prediction evaluation
    viral_threshold: float = 0.5
    use_roc_auc: bool = True
    use_pr_curve: bool = True
    
    # Video quality evaluation
    quality_metrics: List[str] = field(default_factory=lambda: ['psnr', 'ssim', 'lpips'])
    reference_videos: Optional[str] = None
    
    # Performance evaluation
    measure_inference_time: bool = True
    measure_memory_usage: bool = True
    measure_throughput: bool = True

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # General metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Caption metrics
    bleu_score: float = 0.0
    meteor_score: float = 0.0
    rouge_scores: Dict[str, float] = field(default_factory=dict)
    
    # Viral prediction metrics
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    viral_accuracy: float = 0.0
    
    # Video quality metrics
    psnr: float = 0.0
    ssim: float = 0.0
    lpips: float = 0.0
    
    # Performance metrics
    inference_time: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0
    
    # Additional metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metrics: EvaluationMetrics
    predictions: List[Any]
    targets: List[Any]
    metadata: Dict[str, Any]
    config: EvaluationConfig

# =============================================================================
# CAPTION EVALUATION METRICS
# =============================================================================

class CaptionEvaluator:
    """Evaluator for caption generation tasks."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.smoothing = SmoothingFunction().method1
        
        if config.use_rouge:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                config.rouge_metrics, 
                use_stemmer=True
            )
    
    def evaluate_captions(
        self,
        predicted_captions: List[str],
        reference_captions: List[List[str]],
        **kwargs
    ) -> EvaluationMetrics:
        """Evaluate caption generation quality."""
        metrics = EvaluationMetrics()
        
        # BLEU Score
        bleu_scores = []
        for pred, refs in zip(predicted_captions, reference_captions):
            # Tokenize captions
            pred_tokens = pred.lower().split()
            ref_tokens = [ref.lower().split() for ref in refs]
            
            # Calculate BLEU
            bleu = sentence_bleu(
                ref_tokens, 
                pred_tokens, 
                weights=self.config.bleu_weights,
                smoothing_function=self.smoothing
            )
            bleu_scores.append(bleu)
        
        metrics.bleu_score = np.mean(bleu_scores)
        
        # METEOR Score
        if self.config.use_meteor:
            try:
                meteor_scores = []
                for pred, refs in zip(predicted_captions, reference_captions):
                    score = meteor_score([refs], pred)
                    meteor_scores.append(score)
                metrics.meteor_score = np.mean(meteor_scores)
            except Exception as e:
                logger.warning(f"METEOR score calculation failed: {e}")
                metrics.meteor_score = 0.0
        
        # ROUGE Scores
        if self.config.use_rouge:
            rouge_scores = defaultdict(list)
            for pred, refs in zip(predicted_captions, reference_captions):
                # Use first reference for ROUGE
                scores = self.rouge_scorer.score(refs[0], pred)
                for metric, score in scores.items():
                    rouge_scores[metric].append(score.fmeasure)
            
            metrics.rouge_scores = {
                metric: np.mean(scores) 
                for metric, scores in rouge_scores.items()
            }
        
        # Custom metrics
        metrics.custom_metrics = {
            'avg_caption_length': np.mean([len(cap.split()) for cap in predicted_captions]),
            'caption_diversity': self._calculate_diversity(predicted_captions),
            'vocabulary_size': self._calculate_vocabulary_size(predicted_captions)
        }
        
        return metrics
    
    def _calculate_diversity(self, captions: List[str]) -> float:
        """Calculate caption diversity using unique n-grams."""
        all_ngrams = set()
        total_ngrams = 0
        
        for caption in captions:
            tokens = caption.lower().split()
            # Collect bigrams and trigrams
            for n in [2, 3]:
                for i in range(len(tokens) - n + 1):
                    ngram = ' '.join(tokens[i:i+n])
                    all_ngrams.add(ngram)
                    total_ngrams += 1
        
        return len(all_ngrams) / total_ngrams if total_ngrams > 0 else 0.0
    
    def _calculate_vocabulary_size(self, captions: List[str]) -> int:
        """Calculate vocabulary size."""
        vocabulary = set()
        for caption in captions:
            tokens = caption.lower().split()
            vocabulary.update(tokens)
        return len(vocabulary)

# =============================================================================
# VIRAL PREDICTION EVALUATION
# =============================================================================

class ViralPredictionEvaluator:
    """Evaluator for viral prediction tasks."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate_viral_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        **kwargs
    ) -> EvaluationMetrics:
        """Evaluate viral prediction performance."""
        metrics = EvaluationMetrics()
        
        # Convert to binary predictions
        binary_predictions = (predictions > self.config.viral_threshold).astype(int)
        
        # Basic classification metrics
        metrics.accuracy = accuracy_score(targets, binary_predictions)
        metrics.precision = precision_score(targets, binary_predictions, average='weighted')
        metrics.recall = recall_score(targets, binary_predictions, average='weighted')
        metrics.f1_score = f1_score(targets, binary_predictions, average='weighted')
        
        # ROC AUC
        if self.config.use_roc_auc:
            try:
                metrics.roc_auc = roc_auc_score(targets, predictions)
            except Exception as e:
                logger.warning(f"ROC AUC calculation failed: {e}")
                metrics.roc_auc = 0.0
        
        # PR AUC
        if self.config.use_pr_curve:
            try:
                from sklearn.metrics import average_precision_score
                metrics.pr_auc = average_precision_score(targets, predictions)
            except Exception as e:
                logger.warning(f"PR AUC calculation failed: {e}")
                metrics.pr_auc = 0.0
        
        # Viral-specific metrics
        metrics.viral_accuracy = self._calculate_viral_accuracy(predictions, targets)
        
        # Custom metrics
        metrics.custom_metrics = {
            'prediction_confidence': np.mean(predictions),
            'prediction_std': np.std(predictions),
            'viral_rate': np.mean(targets),
            'predicted_viral_rate': np.mean(binary_predictions)
        }
        
        return metrics
    
    def _calculate_viral_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate accuracy specifically for viral predictions."""
        viral_mask = targets == 1
        if np.sum(viral_mask) == 0:
            return 0.0
        
        viral_predictions = predictions[viral_mask]
        viral_targets = targets[viral_mask]
        viral_binary_predictions = (viral_predictions > self.config.viral_threshold).astype(int)
        
        return accuracy_score(viral_targets, viral_binary_predictions)

# =============================================================================
# VIDEO QUALITY EVALUATION
# =============================================================================

class VideoQualityEvaluator:
    """Evaluator for video quality assessment."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.lpips_model = None
        
        if 'lpips' in config.quality_metrics:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net='alex')
            except ImportError:
                logger.warning("LPIPS not available, skipping LPIPS metric")
    
    def evaluate_video_quality(
        self,
        predicted_videos: List[np.ndarray],
        reference_videos: List[np.ndarray],
        **kwargs
    ) -> EvaluationMetrics:
        """Evaluate video quality metrics."""
        metrics = EvaluationMetrics()
        
        # PSNR
        if 'psnr' in self.config.quality_metrics:
            psnr_scores = []
            for pred, ref in zip(predicted_videos, reference_videos):
                psnr = self._calculate_psnr(pred, ref)
                psnr_scores.append(psnr)
            metrics.psnr = np.mean(psnr_scores)
        
        # SSIM
        if 'ssim' in self.config.quality_metrics:
            ssim_scores = []
            for pred, ref in zip(predicted_videos, reference_videos):
                ssim = self._calculate_ssim(pred, ref)
                ssim_scores.append(ssim)
            metrics.ssim = np.mean(ssim_scores)
        
        # LPIPS
        if 'lpips' in self.config.quality_metrics and self.lpips_model is not None:
            lpips_scores = []
            for pred, ref in zip(predicted_videos, reference_videos):
                lpips = self._calculate_lpips(pred, ref)
                lpips_scores.append(lpips)
            metrics.lpips = np.mean(lpips_scores)
        
        # Custom metrics
        metrics.custom_metrics = {
            'avg_video_length': np.mean([len(video) for video in predicted_videos]),
            'video_resolution': self._get_video_resolution(predicted_videos[0]),
            'frame_rate': self._estimate_frame_rate(predicted_videos[0])
        }
        
        return metrics
    
    def _calculate_psnr(self, pred: np.ndarray, ref: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        mse = np.mean((pred.astype(float) - ref.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))
    
    def _calculate_ssim(self, pred: np.ndarray, ref: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        try:
            from skimage.metrics import structural_similarity as ssim
            return ssim(pred, ref, multichannel=True, data_range=255)
        except ImportError:
            logger.warning("scikit-image not available, skipping SSIM")
            return 0.0
    
    def _calculate_lpips(self, pred: np.ndarray, ref: np.ndarray) -> float:
        """Calculate LPIPS perceptual similarity."""
        if self.lpips_model is None:
            return 0.0
        
        # Convert to torch tensors
        pred_tensor = torch.from_numpy(pred).float().unsqueeze(0) / 255.0
        ref_tensor = torch.from_numpy(ref).float().unsqueeze(0) / 255.0
        
        # Calculate LPIPS
        with torch.no_grad():
            lpips_score = self.lpips_model(pred_tensor, ref_tensor).item()
        
        return lpips_score
    
    def _get_video_resolution(self, video: np.ndarray) -> Tuple[int, int]:
        """Get video resolution."""
        if len(video.shape) >= 3:
            return video.shape[-2], video.shape[-1]
        return 0, 0
    
    def _estimate_frame_rate(self, video: np.ndarray) -> float:
        """Estimate frame rate from video."""
        # This is a simplified estimation
        return 30.0  # Default assumption

# =============================================================================
# PERFORMANCE EVALUATION
# =============================================================================

class PerformanceEvaluator:
    """Evaluator for model performance metrics."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.timings = []
        self.memory_usage = []
    
    def evaluate_performance(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        **kwargs
    ) -> EvaluationMetrics:
        """Evaluate model performance."""
        metrics = EvaluationMetrics()
        
        model.eval()
        device = next(model.parameters()).device
        
        # Measure inference time
        if self.config.measure_inference_time:
            inference_times = []
            
            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(device)
                    else:
                        inputs = batch.video_frames.to(device)
                    
                    # Warmup
                    if len(inference_times) == 0:
                        for _ in range(10):
                            _ = model(inputs)
                    
                    # Measure inference time
                    start_time = time.time()
                    _ = model(inputs)
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    end_time = time.time()
                    
                    inference_times.append(end_time - start_time)
            
            metrics.inference_time = np.mean(inference_times)
            metrics.throughput = len(dataloader.dataset) / (metrics.inference_time * len(dataloader))
        
        # Measure memory usage
        if self.config.measure_memory_usage and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            metrics.memory_usage = memory_allocated
            metrics.custom_metrics['memory_reserved_gb'] = memory_reserved
        
        # GPU utilization
        if torch.cuda.is_available():
            metrics.custom_metrics['gpu_utilization'] = torch.cuda.utilization()
        
        return metrics

# =============================================================================
# OPTIMIZED EVALUATOR
# =============================================================================

class OptimizedEvaluator:
    """Main evaluator that combines all evaluation types."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        # Initialize task-specific evaluators
        self.caption_evaluator = CaptionEvaluator(config)
        self.viral_evaluator = ViralPredictionEvaluator(config)
        self.quality_evaluator = VideoQualityEvaluator(config)
        self.performance_evaluator = PerformanceEvaluator(config)
        
        logger.info(f"Initialized evaluator for task: {config.task_type}")
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        targets: Optional[List[Any]] = None,
        **kwargs
    ) -> EvaluationResult:
        """Comprehensive evaluation."""
        logger.info("Starting evaluation...")
        
        # Get predictions
        predictions = self._get_predictions(model, dataloader)
        
        # Initialize metrics
        metrics = EvaluationMetrics()
        
        # Task-specific evaluation
        if self.config.task_type == 'caption_generation':
            if targets is None:
                raise ValueError("Targets required for caption evaluation")
            metrics = self.caption_evaluator.evaluate_captions(
                predictions, targets, **kwargs
            )
        
        elif self.config.task_type == 'viral_prediction':
            if targets is None:
                raise ValueError("Targets required for viral prediction evaluation")
            metrics = self.viral_evaluator.evaluate_viral_predictions(
                predictions, targets, **kwargs
            )
        
        elif self.config.task_type == 'video_quality':
            if targets is None:
                raise ValueError("Reference videos required for quality evaluation")
            metrics = self.quality_evaluator.evaluate_video_quality(
                predictions, targets, **kwargs
            )
        
        # Performance evaluation
        if self.config.measure_inference_time or self.config.measure_memory_usage:
            perf_metrics = self.performance_evaluator.evaluate_performance(
                model, dataloader, **kwargs
            )
            
            # Merge performance metrics
            metrics.inference_time = perf_metrics.inference_time
            metrics.memory_usage = perf_metrics.memory_usage
            metrics.throughput = perf_metrics.throughput
            metrics.custom_metrics.update(perf_metrics.custom_metrics)
        
        # Create result
        result = EvaluationResult(
            metrics=metrics,
            predictions=predictions,
            targets=targets or [],
            metadata={
                'task_type': self.config.task_type,
                'num_samples': len(predictions),
                'evaluation_time': time.time()
            },
            config=self.config
        )
        
        logger.info("Evaluation completed")
        return result
    
    def _get_predictions(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> List[Any]:
        """Get model predictions."""
        model.eval()
        device = next(model.parameters()).device
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.video_frames.to(device)
                
                outputs = model(inputs)
                
                # Process outputs based on task type
                if self.config.task_type == 'caption_generation':
                    # Decode captions
                    batch_predictions = self._decode_captions(outputs)
                elif self.config.task_type == 'viral_prediction':
                    # Get viral scores
                    batch_predictions = torch.sigmoid(outputs).cpu().numpy()
                elif self.config.task_type == 'video_quality':
                    # Get generated videos
                    batch_predictions = outputs.cpu().numpy()
                else:
                    batch_predictions = outputs.cpu().numpy()
                
                predictions.extend(batch_predictions)
        
        return predictions
    
    def _decode_captions(self, outputs: torch.Tensor) -> List[str]:
        """Decode model outputs to captions."""
        # This is a placeholder - implement based on your tokenizer
        if outputs.dim() > 1:
            # Assume outputs are logits
            predicted_tokens = outputs.argmax(dim=-1)
            # Convert tokens to captions using your tokenizer
            captions = []
            for tokens in predicted_tokens:
                caption = " ".join([str(token.item()) for token in tokens])
                captions.append(caption)
            return captions
        else:
            return [str(output.item()) for output in outputs]

# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def create_evaluator(
    task_type: str,
    **kwargs
) -> OptimizedEvaluator:
    """Create evaluator with default settings."""
    config = EvaluationConfig(task_type=task_type, **kwargs)
    return OptimizedEvaluator(config)

def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    task_type: str,
    targets: Optional[List[Any]] = None,
    **kwargs
) -> EvaluationResult:
    """Quick evaluation function."""
    evaluator = create_evaluator(task_type, **kwargs)
    return evaluator.evaluate(model, dataloader, targets, **kwargs)

def save_evaluation_results(
    result: EvaluationResult,
    save_path: str
):
    """Save evaluation results to file."""
    # Convert to serializable format
    serializable_result = {
        'metrics': {
            'accuracy': result.metrics.accuracy,
            'precision': result.metrics.precision,
            'recall': result.metrics.recall,
            'f1_score': result.metrics.f1_score,
            'bleu_score': result.metrics.bleu_score,
            'meteor_score': result.metrics.meteor_score,
            'rouge_scores': result.metrics.rouge_scores,
            'roc_auc': result.metrics.roc_auc,
            'pr_auc': result.metrics.pr_auc,
            'psnr': result.metrics.psnr,
            'ssim': result.metrics.ssim,
            'lpips': result.metrics.lpips,
            'inference_time': result.metrics.inference_time,
            'memory_usage': result.metrics.memory_usage,
            'throughput': result.metrics.throughput,
            'custom_metrics': result.metrics.custom_metrics
        },
        'metadata': result.metadata,
        'config': {
            'task_type': result.config.task_type,
            'batch_size': result.config.batch_size,
            'device': result.config.device
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(serializable_result, f, indent=2)
    
    logger.info(f"Evaluation results saved to {save_path}")

def plot_evaluation_results(
    result: EvaluationResult,
    save_path: Optional[str] = None
):
    """Plot evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Task-specific plots
    if result.config.task_type == 'caption_generation':
        # Caption metrics
        caption_metrics = ['bleu_score', 'meteor_score']
        caption_values = [getattr(result.metrics, metric) for metric in caption_metrics]
        
        axes[0, 0].bar(caption_metrics, caption_values)
        axes[0, 0].set_title('Caption Generation Metrics')
        axes[0, 0].set_ylabel('Score')
        
        # ROUGE scores
        if result.metrics.rouge_scores:
            rouge_metrics = list(result.metrics.rouge_scores.keys())
            rouge_values = list(result.metrics.rouge_scores.values())
            
            axes[0, 1].bar(rouge_metrics, rouge_values)
            axes[0, 1].set_title('ROUGE Scores')
            axes[0, 1].set_ylabel('Score')
    
    elif result.config.task_type == 'viral_prediction':
        # Classification metrics
        class_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        class_values = [getattr(result.metrics, metric) for metric in class_metrics]
        
        axes[0, 0].bar(class_metrics, class_values)
        axes[0, 0].set_title('Classification Metrics')
        axes[0, 0].set_ylabel('Score')
        
        # AUC metrics
        auc_metrics = ['roc_auc', 'pr_auc']
        auc_values = [getattr(result.metrics, metric) for metric in auc_metrics]
        
        axes[0, 1].bar(auc_metrics, auc_values)
        axes[0, 1].set_title('AUC Metrics')
        axes[0, 1].set_ylabel('Score')
    
    elif result.config.task_type == 'video_quality':
        # Quality metrics
        quality_metrics = ['psnr', 'ssim']
        quality_values = [getattr(result.metrics, metric) for metric in quality_metrics]
        
        axes[0, 0].bar(quality_metrics, quality_values)
        axes[0, 0].set_title('Video Quality Metrics')
        axes[0, 0].set_ylabel('Score')
    
    # Performance metrics
    perf_metrics = ['inference_time', 'memory_usage', 'throughput']
    perf_values = [getattr(result.metrics, metric) for metric in perf_metrics]
    
    axes[1, 0].bar(perf_metrics, perf_values)
    axes[1, 0].set_title('Performance Metrics')
    axes[1, 0].set_ylabel('Value')
    
    # Custom metrics
    if result.metrics.custom_metrics:
        custom_metrics = list(result.metrics.custom_metrics.keys())[:5]  # Top 5
        custom_values = list(result.metrics.custom_metrics.values())[:5]
        
        axes[1, 1].bar(custom_metrics, custom_values)
        axes[1, 1].set_title('Custom Metrics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Evaluation plots saved to {save_path}")
    
    plt.show()

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_evaluator_factory(task_type: str, **kwargs):
    """Get evaluator factory with default settings."""
    
    def create_evaluator(**override_kwargs):
        """Create evaluator with overridden parameters."""
        params = {
            'task_type': task_type,
            **kwargs,
            **override_kwargs
        }
        
        return create_evaluator(**params)
    
    return create_evaluator

# Global factory instances
evaluator_factories = {}

def get_global_evaluator_factory(task_type: str, **kwargs):
    """Get global evaluator factory."""
    if task_type not in evaluator_factories:
        evaluator_factories[task_type] = get_evaluator_factory(task_type, **kwargs)
    return evaluator_factories[task_type] 
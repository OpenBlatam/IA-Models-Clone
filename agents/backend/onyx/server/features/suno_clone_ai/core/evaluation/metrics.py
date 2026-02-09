"""
Evaluation Metrics Module

Implements:
- Audio quality metrics
- Perceptual metrics
- Training metrics
- Comparison utilities
"""

import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class AudioMetrics:
    """Audio-specific evaluation metrics."""
    
    @staticmethod
    def signal_to_noise_ratio(
        signal: np.ndarray,
        noise: np.ndarray
    ) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR).
        
        Args:
            signal: Signal array
            noise: Noise array
            
        Returns:
            SNR in dB
        """
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    @staticmethod
    def spectral_convergence(
        reference: np.ndarray,
        generated: np.ndarray,
        sample_rate: int = 32000
    ) -> float:
        """
        Calculate spectral convergence.
        
        Args:
            reference: Reference audio
            generated: Generated audio
            sample_rate: Sample rate
            
        Returns:
            Spectral convergence value
        """
        try:
            import librosa
            
            # Compute STFT
            ref_stft = np.abs(librosa.stft(reference, sr=sample_rate))
            gen_stft = np.abs(librosa.stft(generated, sr=sample_rate))
            
            # Ensure same shape
            min_freq = min(ref_stft.shape[0], gen_stft.shape[0])
            min_time = min(ref_stft.shape[1], gen_stft.shape[1])
            ref_stft = ref_stft[:min_freq, :min_time]
            gen_stft = gen_stft[:min_freq, :min_time]
            
            # Spectral convergence
            numerator = np.linalg.norm(ref_stft - gen_stft, ord='fro')
            denominator = np.linalg.norm(ref_stft, ord='fro')
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except ImportError:
            logger.warning("librosa not available for spectral metrics")
            return 0.0
    
    @staticmethod
    def compute_all_audio_metrics(
        reference: np.ndarray,
        generated: np.ndarray,
        sample_rate: int = 32000
    ) -> Dict[str, float]:
        """
        Compute all audio metrics.
        
        Args:
            reference: Reference audio
            generated: Generated audio
            
        Returns:
            Dictionary of metrics
        """
        # Ensure same length
        min_len = min(len(reference), len(generated))
        reference = reference[:min_len]
        generated = generated[:min_len]
        
        # Compute error
        error = reference - generated
        
        metrics = {
            'snr': AudioMetrics.signal_to_noise_ratio(reference, error),
            'spectral_convergence': AudioMetrics.spectral_convergence(
                reference, generated, sample_rate
            ),
            'mse': float(np.mean((reference - generated) ** 2)),
            'mae': float(np.mean(np.abs(reference - generated))),
            'rmse': float(np.sqrt(np.mean((reference - generated) ** 2)))
        }
        
        return metrics


class TrainingMetrics:
    """Training-specific metrics."""
    
    @staticmethod
    def compute_loss_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean"
    ) -> Dict[str, float]:
        """
        Compute various loss metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            reduction: Reduction method
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Mean Squared Error
        mse = F.mse_loss(predictions, targets, reduction=reduction)
        metrics['mse'] = mse.item() if isinstance(mse, torch.Tensor) else mse
        
        # Mean Absolute Error
        mae = F.l1_loss(predictions, targets, reduction=reduction)
        metrics['mae'] = mae.item() if isinstance(mae, torch.Tensor) else mae
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        return metrics
    
    @staticmethod
    def compute_accuracy(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        Compute accuracy (for classification tasks).
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            threshold: Classification threshold
            
        Returns:
            Accuracy value
        """
        pred_binary = (predictions > threshold).float()
        correct = (pred_binary == targets).float()
        accuracy = correct.mean().item()
        return accuracy


class PerceptualMetrics:
    """Perceptual audio quality metrics."""
    
    @staticmethod
    def compute_perceptual_loss(
        reference: torch.Tensor,
        generated: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss using feature extraction.
        
        Args:
            reference: Reference audio
            generated: Generated audio
            
        Returns:
            Perceptual loss value
        """
        # This is a placeholder - implement with actual perceptual model
        # (e.g., VGG features, or learned perceptual features)
        return F.mse_loss(reference, generated)


def compute_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reference_audio: Optional[np.ndarray] = None,
    generated_audio: Optional[np.ndarray] = None,
    sample_rate: int = 32000
) -> Dict[str, float]:
    """
    Compute all available metrics.
    
    Args:
        predictions: Model predictions (tensor)
        targets: Ground truth targets (tensor)
        reference_audio: Reference audio (numpy array)
        generated_audio: Generated audio (numpy array)
        sample_rate: Sample rate for audio metrics
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Training metrics
    training_metrics = TrainingMetrics.compute_loss_metrics(predictions, targets)
    metrics.update(training_metrics)
    
    # Audio metrics (if audio provided)
    if reference_audio is not None and generated_audio is not None:
        audio_metrics = AudioMetrics.compute_all_audio_metrics(
            reference_audio,
            generated_audio,
            sample_rate
        )
        metrics.update(audio_metrics)
    
    return metrics




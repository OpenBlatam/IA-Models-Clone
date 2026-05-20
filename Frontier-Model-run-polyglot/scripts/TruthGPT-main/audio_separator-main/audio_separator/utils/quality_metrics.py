"""
Quality metrics for separated audio.
"""

from typing import Dict, Optional
import numpy as np

from ..logger import logger


def calculate_separation_quality(
    separated: Dict[str, np.ndarray],
    mixture: np.ndarray,
    reference: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate quality metrics for separated audio.
    
    Args:
        separated: Dictionary of separated sources
        mixture: Original mixture
        reference: Optional reference sources for comparison
        
    Returns:
        Dictionary of quality metrics per source
    """
    from ..eval.metrics import (
        calculate_sdr,
        calculate_sar,
        calculate_isdr
    )
    
    metrics = {}
    
    for source_name, source_audio in separated.items():
        source_metrics = {}
        
        # Calculate ISDR (improvement over mixture)
        isdr = calculate_isdr(source_audio, mixture, mixture)
        source_metrics["ISDR"] = isdr
        
        # If reference is available, calculate SDR and SAR
        if reference and source_name in reference:
            ref_audio = reference[source_name]
            sdr = calculate_sdr(source_audio, ref_audio)
            sar = calculate_sar(source_audio, ref_audio)
            source_metrics["SDR"] = sdr
            source_metrics["SAR"] = sar
        
        # Calculate energy ratio
        source_energy = np.sum(source_audio ** 2)
        mixture_energy = np.sum(mixture ** 2)
        if mixture_energy > 0:
            energy_ratio = source_energy / mixture_energy
            source_metrics["energy_ratio"] = float(energy_ratio)
        
        # Calculate spectral similarity (if librosa available)
        try:
            import librosa
            
            source_stft = librosa.stft(source_audio)
            mixture_stft = librosa.stft(mixture)
            
            source_mag = np.abs(source_stft)
            mixture_mag = np.abs(mixture_stft)
            
            # Cosine similarity
            source_flat = source_mag.flatten()
            mixture_flat = mixture_mag.flatten()
            
            dot_product = np.dot(source_flat, mixture_flat)
            norm_product = np.linalg.norm(source_flat) * np.linalg.norm(mixture_flat)
            
            if norm_product > 0:
                spectral_similarity = dot_product / norm_product
                source_metrics["spectral_similarity"] = float(spectral_similarity)
                
        except ImportError:
            pass
        
        metrics[source_name] = source_metrics
    
    return metrics


def assess_audio_quality(
    audio: np.ndarray,
    sample_rate: int = 44100
) -> Dict[str, float]:
    """
    Assess overall audio quality.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        
    Returns:
        Dictionary of quality metrics
    """
    quality = {}
    
    # Signal-to-noise ratio approximation
    signal_power = np.mean(audio ** 2)
    noise_estimate = np.var(audio - np.mean(audio))
    if noise_estimate > 0:
        snr = 10 * np.log10(signal_power / noise_estimate)
        quality["snr_db"] = float(snr)
    
    # Dynamic range
    peak = np.abs(audio).max()
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        dynamic_range = 20 * np.log10(peak / rms)
        quality["dynamic_range_db"] = float(dynamic_range)
    
    # Clipping detection
    clipping_ratio = np.sum(np.abs(audio) >= 0.99) / len(audio)
    quality["clipping_ratio"] = float(clipping_ratio)
    
    # Frequency content analysis
    try:
        import librosa
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        quality["brightness"] = float(np.mean(spectral_centroid))
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        quality["spectral_rolloff"] = float(np.mean(spectral_rolloff))
        
    except ImportError:
        pass
    
    return quality


def compare_separations(
    separation1: Dict[str, np.ndarray],
    separation2: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compare two separation results.
    
    Args:
        separation1: First separation result
        separation2: Second separation result
        
    Returns:
        Dictionary of comparison metrics
    """
    from ..eval.metrics import calculate_sdr
    
    comparison = {}
    
    # Compare common sources
    common_sources = set(separation1.keys()) & set(separation2.keys())
    
    for source_name in common_sources:
        source1 = separation1[source_name]
        source2 = separation2[source_name]
        
        # Ensure same length
        min_len = min(len(source1), len(source2))
        source1 = source1[:min_len]
        source2 = source2[:min_len]
        
        # Calculate similarity
        sdr = calculate_sdr(source1, source2)
        comparison[f"{source_name}_sdr"] = sdr
        
        # Calculate correlation
        correlation = np.corrcoef(source1, source2)[0, 1]
        comparison[f"{source_name}_correlation"] = float(correlation)
    
    return comparison


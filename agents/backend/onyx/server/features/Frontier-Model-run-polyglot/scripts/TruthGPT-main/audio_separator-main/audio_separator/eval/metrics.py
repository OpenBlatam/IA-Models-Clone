"""
Evaluation metrics for audio separation.
"""

from typing import Dict
import numpy as np


def calculate_sdr(estimated: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate Signal-to-Distortion Ratio (SDR).
    
    Args:
        estimated: Estimated/separated audio
        target: Target/reference audio
        
    Returns:
        SDR value in dB
    """
    # Ensure same length
    min_len = min(len(estimated), len(target))
    estimated = estimated[:min_len]
    target = target[:min_len]
    
    # Calculate SDR
    numerator = np.sum(target ** 2)
    denominator = np.sum((estimated - target) ** 2)
    
    if denominator == 0:
        return float('inf') if numerator > 0 else 0.0
    
    sdr = 10 * np.log10(numerator / denominator)
    return float(sdr)


def calculate_sir(estimated: np.ndarray, target: np.ndarray, 
                  interference: np.ndarray) -> float:
    """
    Calculate Signal-to-Interference Ratio (SIR).
    
    Args:
        estimated: Estimated audio
        target: Target source
        interference: Interference from other sources
        
    Returns:
        SIR value in dB
    """
    min_len = min(len(estimated), len(target), len(interference))
    estimated = estimated[:min_len]
    target = target[:min_len]
    interference = interference[:min_len]
    
    # Project estimated onto target
    target_norm = np.sum(target ** 2)
    if target_norm == 0:
        return 0.0
    
    projection = np.sum(estimated * target) / target_norm
    target_component = projection * target
    interference_component = estimated - target_component
    
    numerator = np.sum(target_component ** 2)
    denominator = np.sum(interference_component ** 2)
    
    if denominator == 0:
        return float('inf') if numerator > 0 else 0.0
    
    sir = 10 * np.log10(numerator / denominator)
    return float(sir)


def calculate_sar(estimated: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate Signal-to-Artifacts Ratio (SAR).
    
    Args:
        estimated: Estimated audio
        target: Target source
        
    Returns:
        SAR value in dB
    """
    min_len = min(len(estimated), len(target))
    estimated = estimated[:min_len]
    target = target[:min_len]
    
    # Project estimated onto target
    target_norm = np.sum(target ** 2)
    if target_norm == 0:
        return 0.0
    
    projection = np.sum(estimated * target) / target_norm
    target_component = projection * target
    artifacts = estimated - target_component
    
    numerator = np.sum(target_component ** 2)
    denominator = np.sum(artifacts ** 2)
    
    if denominator == 0:
        return float('inf') if numerator > 0 else 0.0
    
    sar = 10 * np.log10(numerator / denominator)
    return float(sar)


def calculate_isdr(estimated: np.ndarray, target: np.ndarray,
                  mixture: np.ndarray) -> float:
    """
    Calculate Improvement in Signal-to-Distortion Ratio (ISDR).
    
    Args:
        estimated: Estimated/separated audio
        target: Target source
        mixture: Original mixture
        
    Returns:
        ISDR value in dB
    """
    sdr_separated = calculate_sdr(estimated, target)
    sdr_mixture = calculate_sdr(mixture, target)
    
    isdr = sdr_separated - sdr_mixture
    return float(isdr)


def evaluate_separation(
    separated: Dict[str, np.ndarray],
    references: Dict[str, np.ndarray],
    mixture: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate separation quality using multiple metrics.
    
    Args:
        separated: Dictionary of separated sources
        references: Dictionary of reference sources
        mixture: Original mixture
        
    Returns:
        Dictionary mapping source names to metric dictionaries
    """
    results = {}
    
    for source_name in separated.keys():
        if source_name not in references:
            continue
        
        estimated = separated[source_name]
        target = references[source_name]
        
        # Calculate all metrics
        metrics = {
            "SDR": calculate_sdr(estimated, target),
            "SAR": calculate_sar(estimated, target),
            "ISDR": calculate_isdr(estimated, target, mixture)
        }
        
        # Calculate SIR if we have other sources
        other_sources = [s for s in separated.keys() if s != source_name]
        if other_sources:
            interference = sum(separated[s] for s in other_sources)
            metrics["SIR"] = calculate_sir(estimated, target, interference)
        
        results[source_name] = metrics
    
    return results


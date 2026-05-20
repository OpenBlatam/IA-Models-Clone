"""
Statistics Analyzer
Statistical analysis utilities
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class StatisticsAnalyzer:
    """
    Statistical analysis utilities
    """
    
    @staticmethod
    def compute_statistics(data: np.ndarray) -> Dict[str, float]:
        """
        Compute basic statistics
        
        Args:
            data: Data array
            
        Returns:
            Dictionary with statistics
        """
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
        }
    
    @staticmethod
    def compute_correlation(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compute correlation between two arrays
        
        Args:
            x: First array
            y: Second array
            
        Returns:
            Dictionary with correlation metrics
        """
        if len(x) != len(y):
            raise ValueError("Arrays must have same length")
        
        correlation = np.corrcoef(x, y)[0, 1]
        
        return {
            'correlation': float(correlation),
            'pearson': float(correlation),
        }
    
    @staticmethod
    def analyze_distribution(data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze data distribution
        
        Args:
            data: Data array
            
        Returns:
            Dictionary with distribution analysis
        """
        stats = StatisticsAnalyzer.compute_statistics(data)
        
        # Skewness and kurtosis
        from scipy import stats as scipy_stats
        try:
            skewness = float(scipy_stats.skew(data))
            kurtosis = float(scipy_stats.kurtosis(data))
        except ImportError:
            logger.warning("scipy not available for skewness/kurtosis")
            skewness = None
            kurtosis = None
        
        return {
            **stats,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_normal': abs(stats['mean'] - stats['median']) < stats['std'] if skewness is None else abs(skewness) < 0.5,
        }
    
    @staticmethod
    def compare_distributions(
        data1: np.ndarray,
        data2: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compare two distributions
        
        Args:
            data1: First data array
            data2: Second data array
            
        Returns:
            Dictionary with comparison metrics
        """
        stats1 = StatisticsAnalyzer.compute_statistics(data1)
        stats2 = StatisticsAnalyzer.compute_statistics(data2)
        
        # T-test if scipy available
        try:
            from scipy import stats as scipy_stats
            t_stat, p_value = scipy_stats.ttest_ind(data1, data2)
        except ImportError:
            logger.warning("scipy not available for t-test")
            t_stat = None
            p_value = None
        
        return {
            'data1_stats': stats1,
            'data2_stats': stats2,
            'mean_diff': stats1['mean'] - stats2['mean'],
            'std_diff': stats1['std'] - stats2['std'],
            't_statistic': float(t_stat) if t_stat is not None else None,
            'p_value': float(p_value) if p_value is not None else None,
        }




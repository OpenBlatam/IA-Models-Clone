"""
Advanced Data Validation and Sanitization
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
import re

logger = logging.getLogger(__name__)


class DataValidator:
    """Advanced data validator"""
    
    def __init__(self):
        """Initialize validator"""
        self.validation_rules = {}
        logger.info("DataValidator initialized")
    
    def validate_features(
        self,
        features: Dict[str, float],
        rules: Optional[Dict[str, Dict]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate feature dictionary
        
        Args:
            features: Feature dictionary
            rules: Optional validation rules
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if rules is None:
            rules = {
                "days_sober": {"min": 0, "max": 3650, "type": float},
                "cravings_level": {"min": 0, "max": 10, "type": float},
                "stress_level": {"min": 0, "max": 10, "type": float},
                "support_level": {"min": 0, "max": 10, "type": float},
                "mood_score": {"min": 0, "max": 10, "type": float}
            }
        
        for key, value in features.items():
            if key not in rules:
                continue
            
            rule = rules[key]
            
            # Type check
            if not isinstance(value, rule.get("type", float)):
                return False, f"{key} must be {rule['type'].__name__}"
            
            # Range check
            if "min" in rule and value < rule["min"]:
                return False, f"{key} must be >= {rule['min']}"
            
            if "max" in rule and value > rule["max"]:
                return False, f"{key} must be <= {rule['max']}"
        
        return True, None
    
    def validate_tensor(
        self,
        tensor: torch.Tensor,
        shape: Optional[tuple] = None,
        dtype: Optional[torch.dtype] = None,
        range: Optional[tuple] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate tensor
        
        Args:
            tensor: Input tensor
            shape: Expected shape
            dtype: Expected dtype
            range: Expected value range (min, max)
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Shape check
        if shape is not None and tensor.shape != shape:
            return False, f"Shape mismatch: expected {shape}, got {tensor.shape}"
        
        # Dtype check
        if dtype is not None and tensor.dtype != dtype:
            return False, f"Dtype mismatch: expected {dtype}, got {tensor.dtype}"
        
        # Range check
        if range is not None:
            min_val, max_val = range
            if torch.any(tensor < min_val) or torch.any(tensor > max_val):
                return False, f"Values out of range [{min_val}, {max_val}]"
        
        # NaN/Inf check
        if torch.isnan(tensor).any():
            return False, "Tensor contains NaN values"
        
        if torch.isinf(tensor).any():
            return False, "Tensor contains Inf values"
        
        return True, None
    
    def sanitize_text(self, text: str, max_length: int = 1000) -> str:
        """
        Sanitize text input
        
        Args:
            text: Input text
            max_length: Maximum length
        
        Returns:
            Sanitized text
        """
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Truncate
        if len(text) > max_length:
            text = text[:max_length]
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def normalize_features(
        self,
        features: Dict[str, float],
        normalization: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, float]:
        """
        Normalize features
        
        Args:
            features: Feature dictionary
            normalization: Normalization rules
        
        Returns:
            Normalized features
        """
        if normalization is None:
            normalization = {
                "days_sober": {"method": "divide", "value": 365.0},
                "cravings_level": {"method": "divide", "value": 10.0},
                "stress_level": {"method": "divide", "value": 10.0},
                "support_level": {"method": "divide", "value": 10.0},
                "mood_score": {"method": "divide", "value": 10.0}
            }
        
        normalized = {}
        for key, value in features.items():
            if key in normalization:
                rule = normalization[key]
                if rule["method"] == "divide":
                    normalized[key] = value / rule["value"]
                elif rule["method"] == "minmax":
                    min_val = rule.get("min", 0)
                    max_val = rule.get("max", 10)
                    normalized[key] = (value - min_val) / (max_val - min_val)
                else:
                    normalized[key] = value
            else:
                normalized[key] = value
        
        return normalized


class AnomalyDetector:
    """Detect anomalies in input data"""
    
    def __init__(self, threshold: float = 3.0):
        """
        Initialize anomaly detector
        
        Args:
            threshold: Z-score threshold
        """
        self.threshold = threshold
        self.stats = {}
    
    def fit(self, data: torch.Tensor):
        """Fit detector on data"""
        self.stats = {
            "mean": data.mean(dim=0),
            "std": data.std(dim=0) + 1e-8  # Avoid division by zero
        }
    
    def detect(self, data: torch.Tensor) -> tuple[bool, torch.Tensor]:
        """
        Detect anomalies
        
        Args:
            data: Input data
        
        Returns:
            Tuple of (has_anomalies, anomaly_scores)
        """
        if not self.stats:
            return False, torch.zeros(data.shape[0])
        
        # Z-score
        z_scores = (data - self.stats["mean"]) / self.stats["std"]
        z_scores = z_scores.abs()
        
        # Anomaly scores
        anomaly_scores = z_scores.max(dim=1)[0]
        
        # Detect anomalies
        has_anomalies = (anomaly_scores > self.threshold).any()
        
        return has_anomalies, anomaly_scores
    
    def filter_anomalies(
        self,
        data: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Filter out anomalies
        
        Args:
            data: Input data
            labels: Optional labels
        
        Returns:
            Filtered data and labels
        """
        has_anomalies, scores = self.detect(data)
        
        if not has_anomalies:
            return data, labels
        
        # Filter
        mask = scores <= self.threshold
        filtered_data = data[mask]
        
        if labels is not None:
            filtered_labels = labels[mask]
        else:
            filtered_labels = None
        
        return filtered_data, filtered_labels


class DataQualityChecker:
    """Check data quality"""
    
    def __init__(self):
        """Initialize quality checker"""
        self.checks = []
    
    def check_completeness(self, data: torch.Tensor) -> Dict[str, Any]:
        """Check data completeness"""
        total = data.numel()
        nan_count = torch.isnan(data).sum().item()
        inf_count = torch.isinf(data).sum().item()
        
        return {
            "total": total,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "completeness": (total - nan_count - inf_count) / total if total > 0 else 0.0
        }
    
    def check_distribution(
        self,
        data: torch.Tensor,
        expected_mean: Optional[float] = None,
        expected_std: Optional[float] = None
    ) -> Dict[str, Any]:
        """Check data distribution"""
        actual_mean = data.mean().item()
        actual_std = data.std().item()
        
        result = {
            "mean": actual_mean,
            "std": actual_std,
            "min": data.min().item(),
            "max": data.max().item()
        }
        
        if expected_mean is not None:
            result["mean_diff"] = abs(actual_mean - expected_mean)
        
        if expected_std is not None:
            result["std_diff"] = abs(actual_std - expected_std)
        
        return result
    
    def check_correlation(
        self,
        data: torch.Tensor,
        threshold: float = 0.95
    ) -> Dict[str, Any]:
        """Check for high correlations (multicollinearity)"""
        if data.shape[1] < 2:
            return {"high_correlations": []}
        
        # Compute correlation matrix
        data_np = data.numpy()
        corr_matrix = np.corrcoef(data_np.T)
        
        # Find high correlations
        high_corrs = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > threshold:
                    high_corrs.append((i, j, corr_matrix[i, j]))
        
        return {
            "high_correlations": high_corrs,
            "max_correlation": corr_matrix.max() if len(corr_matrix) > 0 else 0.0
        }


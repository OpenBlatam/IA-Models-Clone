"""
Formatting Utilities
Format output and data for display
"""

from typing import Any, Dict, List
import torch
import numpy as np


class Formatter:
    """
    Format data for display
    """
    
    @staticmethod
    def format_number(num: float, precision: int = 4) -> str:
        """
        Format number with precision
        
        Args:
            num: Number to format
            precision: Decimal precision
            
        Returns:
            Formatted string
        """
        if isinstance(num, (int, float)):
            return f"{num:.{precision}f}".rstrip('0').rstrip('.')
        return str(num)
    
    @staticmethod
    def format_tensor_shape(shape: tuple) -> str:
        """
        Format tensor shape
        
        Args:
            shape: Tensor shape
            
        Returns:
            Formatted string
        """
        return "x".join(map(str, shape))
    
    @staticmethod
    def format_bytes(bytes_size: int) -> str:
        """
        Format bytes to human-readable format
        
        Args:
            bytes_size: Size in bytes
            
        Returns:
            Formatted string
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} PB"
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """
        Format time in seconds to human-readable format
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted string
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"
    
    @staticmethod
    def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
        """
        Format metrics dictionary
        
        Args:
            metrics: Dictionary of metrics
            precision: Decimal precision
            
        Returns:
            Formatted string
        """
        formatted = []
        for key, value in metrics.items():
            formatted.append(f"{key}: {Formatter.format_number(value, precision)}")
        return ", ".join(formatted)
    
    @staticmethod
    def format_model_summary(model: torch.nn.Module) -> str:
        """
        Format model summary
        
        Args:
            model: Model to summarize
            
        Returns:
            Formatted string
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return (
            f"Model: {model.__class__.__name__}\n"
            f"Total Parameters: {total_params:,}\n"
            f"Trainable Parameters: {trainable_params:,}\n"
            f"Non-trainable Parameters: {total_params - trainable_params:,}"
        )




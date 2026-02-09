"""
Tracing Utilities
Model and computation tracing
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class TraceManager:
    """
    Manage tracing for debugging
    """
    
    def __init__(self):
        """Initialize trace manager"""
        self.traces = []
        self.enabled = False
    
    def enable(self) -> None:
        """Enable tracing"""
        self.enabled = True
        torch.autograd.set_detect_anomaly(True)
        logger.info("Tracing enabled")
    
    def disable(self) -> None:
        """Disable tracing"""
        self.enabled = False
        torch.autograd.set_detect_anomaly(False)
        logger.info("Tracing disabled")
    
    def trace_forward(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Trace forward pass
        
        Args:
            model: Model to trace
            inputs: Input tensor
            
        Returns:
            Dictionary with trace information
        """
        if not self.enabled:
            self.enable()
        
        try:
            with torch.no_grad():
                outputs = model(inputs)
            
            trace = {
                'type': 'forward',
                'input_shape': list(inputs.shape),
                'output_shape': list(outputs.shape),
                'success': True,
            }
            self.traces.append(trace)
            return trace
        except Exception as e:
            trace = {
                'type': 'forward',
                'input_shape': list(inputs.shape),
                'error': str(e),
                'success': False,
            }
            self.traces.append(trace)
            logger.error(f"Forward trace failed: {e}")
            return trace
    
    def get_traces(self) -> List[Dict[str, Any]]:
        """
        Get all traces
        
        Returns:
            List of trace dictionaries
        """
        return self.traces.copy()
    
    def clear_traces(self) -> None:
        """Clear all traces"""
        self.traces.clear()
        logger.info("Traces cleared")




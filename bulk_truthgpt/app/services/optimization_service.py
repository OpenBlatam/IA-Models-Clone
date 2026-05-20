"""
Optimization service for Ultimate Enhanced Supreme Production system
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from app.models.optimization import OptimizationResult, OptimizationMetrics
from app.core.optimization_core import OptimizationCore

logger = logging.getLogger(__name__)

class OptimizationService:
    """Optimization service."""
    
    def __init__(self):
        """Initialize service."""
        self.core = OptimizationCore()
        self.logger = logger
    
    def process_optimization(self, request_data: Dict[str, Any]) -> OptimizationResult:
        """Process optimization request."""
        try:
            start_time = time.perf_counter()
            
            # Process optimization with core
            result = asyncio.run(self.core.process_optimization(request_data))
            
            processing_time = time.perf_counter() - start_time
            
            # Create optimization result
            optimization_result = OptimizationResult(
                speed_improvement=result.get('speed_improvement', 0.0),
                memory_reduction=result.get('memory_reduction', 0.0),
                accuracy_preservation=result.get('accuracy_preservation', 0.0),
                energy_efficiency=result.get('energy_efficiency', 0.0),
                optimization_time=processing_time,
                level=result.get('level', 'unknown'),
                techniques_applied=result.get('techniques_applied', []),
                performance_metrics=result.get('performance_metrics', {})
            )
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"❌ Error processing optimization: {e}")
            return OptimizationResult(
                speed_improvement=0.0,
                memory_reduction=0.0,
                accuracy_preservation=0.0,
                energy_efficiency=0.0,
                optimization_time=0.0,
                level='error',
                techniques_applied=[],
                performance_metrics={}
            )
    
    def process_batch_optimization(self, request_data: Dict[str, Any]) -> List[OptimizationResult]:
        """Process batch optimization request."""
        try:
            start_time = time.perf_counter()
            
            # Process batch optimization with core
            results = asyncio.run(self.core.process_batch_optimization(request_data))
            
            processing_time = time.perf_counter() - start_time
            
            # Create optimization results
            optimization_results = []
            for result in results:
                optimization_result = OptimizationResult(
                    speed_improvement=result.get('speed_improvement', 0.0),
                    memory_reduction=result.get('memory_reduction', 0.0),
                    accuracy_preservation=result.get('accuracy_preservation', 0.0),
                    energy_efficiency=result.get('energy_efficiency', 0.0),
                    optimization_time=result.get('optimization_time', 0.0),
                    level=result.get('level', 'unknown'),
                    techniques_applied=result.get('techniques_applied', []),
                    performance_metrics=result.get('performance_metrics', {})
                )
                optimization_results.append(optimization_result)
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Error processing batch optimization: {e}")
            return []
    
    def get_optimization_metrics(self) -> OptimizationMetrics:
        """Get optimization metrics."""
        try:
            metrics = self.core.get_optimization_metrics()
            return metrics
        except Exception as e:
            self.logger.error(f"❌ Error getting optimization metrics: {e}")
            return OptimizationMetrics()
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get optimization status."""
        try:
            status = self.core.get_optimization_status()
            return status
        except Exception as e:
            self.logger.error(f"❌ Error getting optimization status: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': time.time()
            }










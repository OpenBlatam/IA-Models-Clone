"""
Pipeline Optimizer
Optimize entire inference pipeline for maximum speed
"""

import torch
import torch.nn as nn
from typing import Optional, List, Callable, Any
import logging
from queue import Queue
from threading import Thread
import time

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Optimized inference pipeline with preprocessing and postprocessing
    """
    
    def __init__(
        self,
        model: nn.Module,
        preprocess: Optional[Callable] = None,
        postprocess: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 64
    ):
        """
        Initialize inference pipeline
        
        Args:
            model: Model
            preprocess: Preprocessing function
            postprocess: Postprocessing function
            device: Device
            batch_size: Batch size
        """
        self.model = model.to(device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()
        self.preprocess = preprocess or (lambda x: x)
        self.postprocess = postprocess or (lambda x: x)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Compile model
        if hasattr(torch, 'compile') and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="max-autotune", fullgraph=True)
            except:
                pass
    
    @torch.inference_mode()
    def process(self, inputs: List[Any]) -> List[Any]:
        """
        Process inputs through pipeline
        
        Args:
            inputs: List of inputs
            
        Returns:
            List of outputs
        """
        results = []
        
        # Process in batches
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            
            # Preprocess
            preprocessed = [self.preprocess(item) for item in batch]
            
            # Stack if tensors
            if isinstance(preprocessed[0], torch.Tensor):
                batch_tensor = torch.stack(preprocessed).to(self.device, non_blocking=True)
            else:
                batch_tensor = torch.tensor(preprocessed).to(self.device, non_blocking=True)
            
            # Inference
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_tensor)
            else:
                outputs = self.model(batch_tensor)
            
            # Postprocess
            outputs_cpu = outputs.cpu()
            batch_results = [self.postprocess(outputs_cpu[j]) for j in range(len(batch))]
            results.extend(batch_results)
        
        return results


class StreamingInference:
    """
    Streaming inference for real-time processing
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        buffer_size: int = 10
    ):
        """
        Initialize streaming inference
        
        Args:
            model: Model
            device: Device
            buffer_size: Buffer size
        """
        self.model = model.to(device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_queue = Queue(maxsize=buffer_size)
        self.output_queue = Queue(maxsize=buffer_size)
        self.running = False
        self.worker_thread = None
    
    def start(self):
        """Start streaming inference"""
        self.running = True
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def stop(self):
        """Stop streaming inference"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
    
    def _worker(self):
        """Worker thread for processing"""
        batch = []
        
        while self.running:
            try:
                # Get input with timeout
                item = self.input_queue.get(timeout=0.1)
                batch.append(item)
                
                # Process batch when full or timeout
                if len(batch) >= 32:
                    self._process_batch(batch)
                    batch = []
            except:
                # Process remaining batch
                if batch:
                    self._process_batch(batch)
                    batch = []
                time.sleep(0.01)
    
    def _process_batch(self, batch: List[torch.Tensor]):
        """Process batch"""
        try:
            batch_tensor = torch.stack(batch).to(self.device, non_blocking=True)
            
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_tensor)
                else:
                    outputs = self.model(batch_tensor)
            
            # Put results in output queue
            for output in outputs.cpu().split(1):
                self.output_queue.put(output.squeeze(0))
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
    
    def put(self, item: torch.Tensor):
        """Put item in input queue"""
        self.input_queue.put(item)
    
    def get(self, timeout: float = 1.0) -> Optional[torch.Tensor]:
        """Get result from output queue"""
        try:
            return self.output_queue.get(timeout=timeout)
        except:
            return None


def create_inference_pipeline(
    model: nn.Module,
    preprocess: Optional[Callable] = None,
    postprocess: Optional[Callable] = None,
    device: Optional[torch.device] = None
) -> InferencePipeline:
    """Factory for inference pipeline"""
    return InferencePipeline(model, preprocess, postprocess, device)


def create_streaming_inference(
    model: nn.Module,
    device: Optional[torch.device] = None
) -> StreamingInference:
    """Factory for streaming inference"""
    return StreamingInference(model, device)














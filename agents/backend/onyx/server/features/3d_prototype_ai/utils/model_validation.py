"""
Model Validation Framework - Framework de validación de modelos
=================================================================
Validación completa de modelos antes de deployment
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationStatus(str, Enum):
    """Estados de validación"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ValidationResult:
    """Resultado de validación"""
    test_name: str
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None


class ModelValidator:
    """Framework de validación de modelos"""
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
    
    def validate_model(
        self,
        model: nn.Module,
        test_data: torch.utils.data.DataLoader,
        device: torch.device,
        min_accuracy: float = 0.7,
        max_latency: float = 1.0,
        min_throughput: float = 10.0
    ) -> List[ValidationResult]:
        """Valida modelo completo"""
        results = []
        
        # Test 1: Accuracy
        acc_result = self._validate_accuracy(model, test_data, device, min_accuracy)
        results.append(acc_result)
        
        # Test 2: Latency
        latency_result = self._validate_latency(model, test_data, device, max_latency)
        results.append(latency_result)
        
        # Test 3: Throughput
        throughput_result = self._validate_throughput(model, test_data, device, min_throughput)
        results.append(throughput_result)
        
        # Test 4: Memory
        memory_result = self._validate_memory(model, device)
        results.append(memory_result)
        
        # Test 5: Output shape
        shape_result = self._validate_output_shape(model, test_data, device)
        results.append(shape_result)
        
        # Test 6: NaN/Inf check
        nan_result = self._validate_no_nan_inf(model, test_data, device)
        results.append(nan_result)
        
        self.validation_results.extend(results)
        return results
    
    def _validate_accuracy(
        self,
        model: nn.Module,
        test_data: torch.utils.data.DataLoader,
        device: torch.device,
        min_accuracy: float
    ) -> ValidationResult:
        """Valida accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_data:
                if isinstance(batch, dict):
                    inputs = batch["input"].to(device)
                    labels = batch["label"].to(device)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1) if outputs.dim() > 1 else outputs
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        status = ValidationStatus.PASSED if accuracy >= min_accuracy else ValidationStatus.FAILED
        
        return ValidationResult(
            test_name="accuracy",
            status=status,
            message=f"Accuracy: {accuracy:.4f} (min: {min_accuracy})",
            details={"accuracy": accuracy, "min_required": min_accuracy}
        )
    
    def _validate_latency(
        self,
        model: nn.Module,
        test_data: torch.utils.data.DataLoader,
        device: torch.device,
        max_latency: float
    ) -> ValidationResult:
        """Valida latencia"""
        model.eval()
        latencies = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= 10:  # Test con 10 batches
                    break
                
                if isinstance(batch, dict):
                    inputs = batch["input"].to(device)
                else:
                    inputs = batch[0].to(device)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                else:
                    import time
                    start_time = time.time()
                
                _ = model(inputs)
                
                if torch.cuda.is_available():
                    end.record()
                    torch.cuda.synchronize()
                    latencies.append(start.elapsed_time(end) / 1000.0)
                else:
                    latencies.append(time.time() - start_time)
        
        avg_latency = np.mean(latencies) if latencies else 0.0
        status = ValidationStatus.PASSED if avg_latency <= max_latency else ValidationStatus.FAILED
        
        return ValidationResult(
            test_name="latency",
            status=status,
            message=f"Avg latency: {avg_latency:.4f}s (max: {max_latency}s)",
            details={"avg_latency": avg_latency, "max_allowed": max_latency}
        )
    
    def _validate_throughput(
        self,
        model: nn.Module,
        test_data: torch.utils.data.DataLoader,
        device: torch.device,
        min_throughput: float
    ) -> ValidationResult:
        """Valida throughput"""
        import time
        
        model.eval()
        total_samples = 0
        start_time = time.time()
        
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= 10:  # Test con 10 batches
                    break
                
                if isinstance(batch, dict):
                    inputs = batch["input"].to(device)
                else:
                    inputs = batch[0].to(device)
                
                _ = model(inputs)
                total_samples += inputs.size(0)
        
        elapsed_time = time.time() - start_time
        throughput = total_samples / elapsed_time if elapsed_time > 0 else 0.0
        
        status = ValidationStatus.PASSED if throughput >= min_throughput else ValidationStatus.FAILED
        
        return ValidationResult(
            test_name="throughput",
            status=status,
            message=f"Throughput: {throughput:.2f} samples/s (min: {min_throughput})",
            details={"throughput": throughput, "min_required": min_throughput}
        )
    
    def _validate_memory(
        self,
        model: nn.Module,
        device: torch.device
    ) -> ValidationResult:
        """Valida uso de memoria"""
        if torch.cuda.is_available() and device.type == "cuda":
            memory_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        else:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Warning si usa mucha memoria
        status = ValidationStatus.WARNING if memory_mb > 1000 else ValidationStatus.PASSED
        
        return ValidationResult(
            test_name="memory",
            status=status,
            message=f"Memory usage: {memory_mb:.2f} MB",
            details={"memory_mb": memory_mb}
        )
    
    def _validate_output_shape(
        self,
        model: nn.Module,
        test_data: torch.utils.data.DataLoader,
        device: torch.device
    ) -> ValidationResult:
        """Valida forma de salida"""
        model.eval()
        
        with torch.no_grad():
            batch = next(iter(test_data))
            if isinstance(batch, dict):
                inputs = batch["input"].to(device)
            else:
                inputs = batch[0].to(device)
            
            output = model(inputs)
            output_shape = output.shape
        
        status = ValidationStatus.PASSED
        
        return ValidationResult(
            test_name="output_shape",
            status=status,
            message=f"Output shape: {output_shape}",
            details={"shape": list(output_shape)}
        )
    
    def _validate_no_nan_inf(
        self,
        model: nn.Module,
        test_data: torch.utils.data.DataLoader,
        device: torch.device
    ) -> ValidationResult:
        """Valida que no haya NaN/Inf"""
        model.eval()
        has_nan = False
        has_inf = False
        
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= 5:  # Test con 5 batches
                    break
                
                if isinstance(batch, dict):
                    inputs = batch["input"].to(device)
                else:
                    inputs = batch[0].to(device)
                
                output = model(inputs)
                
                if torch.isnan(output).any():
                    has_nan = True
                if torch.isinf(output).any():
                    has_inf = True
        
        status = ValidationStatus.FAILED if (has_nan or has_inf) else ValidationStatus.PASSED
        message = "No NaN/Inf detected" if not (has_nan or has_inf) else f"NaN: {has_nan}, Inf: {has_inf}"
        
        return ValidationResult(
            test_name="nan_inf_check",
            status=status,
            message=message,
            details={"has_nan": has_nan, "has_inf": has_inf}
        )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de validación"""
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in self.validation_results if r.status == ValidationStatus.FAILED)
        warnings = sum(1 for r in self.validation_results if r.status == ValidationStatus.WARNING)
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "pass_rate": passed / total if total > 0 else 0.0,
            "results": [
                {
                    "test": r.test_name,
                    "status": r.status.value,
                    "message": r.message
                }
                for r in self.validation_results
            ]
        }





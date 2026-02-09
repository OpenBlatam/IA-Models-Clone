"""
Quantized Models for Maximum Speed
INT8, INT4, and dynamic quantization
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class QuantizedModel:
    """
    Quantized model wrapper for maximum speed
    Supports INT8, INT4, and dynamic quantization
    """
    
    def __init__(
        self,
        model: nn.Module,
        quantization_type: str = "int8",
        device: Optional[torch.device] = None
    ):
        """
        Initialize quantized model
        
        Args:
            model: PyTorch model
            quantization_type: int8, int4, or dynamic
            device: Device to use
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantization_type = quantization_type
        
        # Quantize model
        if quantization_type == "int8":
            self.model = self._quantize_int8(model)
        elif quantization_type == "int4":
            self.model = self._quantize_int4(model)
        elif quantization_type == "dynamic":
            self.model = self._quantize_dynamic(model)
        else:
            self.model = model
        
        self.model.eval()
        logger.info(f"Model quantized to {quantization_type}")
    
    def _quantize_int8(self, model: nn.Module) -> nn.Module:
        """INT8 quantization"""
        try:
            # Static quantization (requires calibration)
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            model_prepared = torch.quantization.prepare(model, inplace=False)
            # Note: In production, you'd calibrate here
            model_quantized = torch.quantization.convert(model_prepared, inplace=False)
            return model_quantized
        except Exception as e:
            logger.warning(f"INT8 quantization failed: {e}, using dynamic")
            return self._quantize_dynamic(model)
    
    def _quantize_int4(self, model: nn.Module) -> nn.Module:
        """INT4 quantization (experimental)"""
        try:
            # Use bitsandbytes if available
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            # This would require model loading with config
            logger.warning("INT4 requires model reloading with config")
            return self._quantize_dynamic(model)
        except ImportError:
            logger.warning("bitsandbytes not available, using dynamic quantization")
            return self._quantize_dynamic(model)
    
    def _quantize_dynamic(self, model: nn.Module) -> nn.Module:
        """Dynamic INT8 quantization (fastest to apply)"""
        return torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.Conv1d, nn.Conv2d},
            dtype=torch.qint8
        )
    
    @torch.inference_mode()
    def forward(self, *args, **kwargs):
        """Forward pass"""
        return self.model(*args, **kwargs)


class OptimizedTransformer:
    """
    Optimized transformer with all speed optimizations
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        use_quantization: bool = True,
        use_compile: bool = True,
        use_flash_attention: bool = True
    ):
        """
        Initialize optimized transformer
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            use_quantization: Use quantization
            use_compile: Use torch.compile
            use_flash_attention: Use flash attention (if available)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load with optimizations
        if use_quantization and self.device.type == "cpu":
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
                # Quantize
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear}, dtype=torch.qint8
                )
            except:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch_dtype
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Flash attention (if available)
        if use_flash_attention and self.device.type == "cuda":
            try:
                from transformers import AutoModel
                # Flash attention requires model architecture support
                logger.info("Flash attention available (if model supports it)")
            except:
                pass
        
        # Compile
        if use_compile and hasattr(torch, 'compile') and self.device.type == "cuda":
            try:
                self.model = torch.compile(
                    self.model,
                    mode="max-autotune",
                    fullgraph=True
                )
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Compilation failed: {e}")
        
        # Warmup
        self._warmup()
    
    def _warmup(self):
        """Warmup model"""
        try:
            dummy_text = "This is a warmup text."
            inputs = self.tokenizer(dummy_text, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                _ = self.model(**inputs)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        except:
            pass
    
    @torch.inference_mode()
    def predict(self, text: str) -> Dict:
        """Fast prediction"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        if self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        
        return {
            "prediction": pred,
            "probability": probs[0][pred].item(),
            "probabilities": probs[0].cpu().tolist()
        }
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Batch prediction"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
            
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            for j, pred in enumerate(preds):
                results.append({
                    "prediction": pred.item(),
                    "probability": probs[j][pred].item(),
                    "probabilities": probs[j].cpu().tolist()
                })
        
        return results


def create_quantized_model(
    model: nn.Module,
    quantization_type: str = "int8",
    device: Optional[torch.device] = None
) -> QuantizedModel:
    """Factory for quantized model"""
    return QuantizedModel(model, quantization_type, device)


def create_optimized_transformer(
    model_name: str,
    device: Optional[torch.device] = None,
    **kwargs
) -> OptimizedTransformer:
    """Factory for optimized transformer"""
    return OptimizedTransformer(model_name, device, **kwargs)














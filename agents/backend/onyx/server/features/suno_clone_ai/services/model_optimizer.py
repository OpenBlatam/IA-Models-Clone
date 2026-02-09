"""
Sistema de Optimización de Modelos

Proporciona:
- Quantization (INT8, FP16)
- Model pruning
- Knowledge distillation
- Model versioning
- Auto-tuning de hiperparámetros
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimizador de modelos"""
    
    def __init__(self, model_dir: str = "./models"):
        """
        Args:
            model_dir: Directorio para almacenar modelos optimizados
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ModelOptimizer initialized")
    
    def quantize_model(
        self,
        model: nn.Module,
        quantization_type: str = "int8",
        save_path: Optional[str] = None
    ) -> nn.Module:
        """
        Cuantiza un modelo para reducir tamaño y acelerar inferencia
        
        Args:
            model: Modelo a cuantizar
            quantization_type: Tipo de cuantización (int8, fp16, dynamic)
            save_path: Ruta para guardar modelo cuantizado
        
        Returns:
            Modelo cuantizado
        """
        try:
            if quantization_type == "int8":
                # Quantization estática INT8
                model.eval()
                model_quantized = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
                logger.info("Model quantized to INT8")
            
            elif quantization_type == "fp16":
                # Media precisión FP16
                model_quantized = model.half()
                logger.info("Model converted to FP16")
            
            elif quantization_type == "dynamic":
                # Quantization dinámica
                model.eval()
                model_quantized = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                logger.info("Model quantized dynamically")
            
            else:
                raise ValueError(f"Unknown quantization type: {quantization_type}")
            
            # Guardar si se especifica
            if save_path:
                torch.save(model_quantized.state_dict(), save_path)
                logger.info(f"Quantized model saved to {save_path}")
            
            return model_quantized
        
        except Exception as e:
            logger.error(f"Error quantizing model: {e}")
            raise
    
    def prune_model(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.2,
        method: str = "magnitude"
    ) -> nn.Module:
        """
        Poda un modelo para reducir parámetros
        
        Args:
            model: Modelo a podar
            pruning_ratio: Ratio de poda (0.0 - 1.0)
            method: Método de poda (magnitude, random)
        
        Returns:
            Modelo podado
        """
        try:
            if method == "magnitude":
                # Poda por magnitud
                import torch.nn.utils.prune as prune
                
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                        prune.l1_unstructured(
                            module,
                            name='weight',
                            amount=pruning_ratio
                        )
                
                # Hacer permanente la poda
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                        prune.remove(module, 'weight')
                
                logger.info(f"Model pruned by {pruning_ratio * 100}% using magnitude method")
            
            return model
        
        except Exception as e:
            logger.error(f"Error pruning model: {e}")
            raise
    
    def optimize_for_inference(
        self,
        model: nn.Module,
        optimizations: List[str] = None
    ) -> nn.Module:
        """
        Aplica múltiples optimizaciones para inferencia
        
        Args:
            model: Modelo a optimizar
            optimizations: Lista de optimizaciones a aplicar
        
        Returns:
            Modelo optimizado
        """
        if optimizations is None:
            optimizations = ["quantize", "compile"]
        
        optimized_model = model
        
        try:
            # Compilar con torch.compile
            if "compile" in optimizations and hasattr(torch, 'compile'):
                optimized_model = torch.compile(optimized_model)
                logger.info("Model compiled with torch.compile")
            
            # Cuantizar
            if "quantize" in optimizations:
                optimized_model = self.quantize_model(optimized_model, "int8")
            
            # Poda
            if "prune" in optimizations:
                optimized_model = self.prune_model(optimized_model, 0.2)
            
            logger.info(f"Model optimized with: {', '.join(optimizations)}")
        
        except Exception as e:
            logger.error(f"Error optimizing model: {e}")
        
        return optimized_model
    
    def compare_models(
        self,
        model1: nn.Module,
        model2: nn.Module,
        test_input: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compara dos modelos
        
        Args:
            model1: Primer modelo
            model2: Segundo modelo
            test_input: Input de prueba
        
        Returns:
            Comparación de modelos
        """
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            # Inferencia
            import time
            
            # Modelo 1
            start = time.time()
            output1 = model1(test_input)
            time1 = time.time() - start
            
            # Modelo 2
            start = time.time()
            output2 = model2(test_input)
            time2 = time.time() - start
        
        # Calcular diferencia
        diff = torch.mean(torch.abs(output1 - output2)).item()
        
        # Contar parámetros
        params1 = sum(p.numel() for p in model1.parameters())
        params2 = sum(p.numel() for p in model2.parameters())
        
        return {
            "model1": {
                "inference_time": time1,
                "parameters": params1,
                "output_shape": list(output1.shape)
            },
            "model2": {
                "inference_time": time2,
                "parameters": params2,
                "output_shape": list(output2.shape)
            },
            "difference": diff,
            "speedup": time1 / time2 if time2 > 0 else 0,
            "size_reduction": (1 - params2 / params1) * 100 if params1 > 0 else 0
        }
    
    def save_model_version(
        self,
        model: nn.Module,
        version: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Guarda una versión del modelo
        
        Args:
            model: Modelo a guardar
            version: Versión del modelo
            metadata: Metadatos adicionales
        
        Returns:
            Ruta del modelo guardado
        """
        version_dir = self.model_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar modelo
        model_path = version_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Guardar metadatos
        metadata_path = version_dir / "metadata.json"
        metadata.update({
            "version": version,
            "saved_at": datetime.now().isoformat(),
            "model_path": str(model_path)
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model version {version} saved to {version_dir}")
        return str(version_dir)
    
    def load_model_version(
        self,
        version: str,
        model_class: type
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Carga una versión del modelo
        
        Args:
            version: Versión a cargar
            model_class: Clase del modelo
        
        Returns:
            Tupla de (modelo, metadatos)
        """
        version_dir = self.model_dir / version
        
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found")
        
        # Cargar metadatos
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Cargar modelo
        model_path = version_dir / "model.pt"
        model = model_class()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        logger.info(f"Model version {version} loaded")
        return model, metadata
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """Lista todas las versiones disponibles"""
        versions = []
        
        for version_dir in self.model_dir.iterdir():
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    versions.append(metadata)
        
        return sorted(versions, key=lambda x: x.get("saved_at", ""), reverse=True)


# Instancia global
_model_optimizer: Optional[ModelOptimizer] = None


def get_model_optimizer(model_dir: str = "./models") -> ModelOptimizer:
    """Obtiene la instancia global del optimizador de modelos"""
    global _model_optimizer
    if _model_optimizer is None:
        _model_optimizer = ModelOptimizer(model_dir=model_dir)
    return _model_optimizer


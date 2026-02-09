"""
Ahead-of-Time Compilation
=========================

Compilación ahead-of-time para inferencia instantánea.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import logging
import os
import pickle

logger = logging.getLogger(__name__)


class AOTCompiler:
    """
    Compilador ahead-of-time.
    """
    
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...] = (1, 20)):
        """
        Inicializar compilador AOT.
        
        Args:
            model: Modelo
            input_shape: Forma de entrada
        """
        self.model = model
        self.input_shape = input_shape
        self.compiled_model = None
        self.compiled_path = None
    
    def compile(self, output_path: str = "compiled_model.pt") -> str:
        """
        Compilar modelo ahead-of-time.
        
        Args:
            output_path: Ruta donde guardar
            
        Returns:
            Ruta del modelo compilado
        """
        self.model.eval()
        
        # Crear input de ejemplo
        example_input = torch.randn(*self.input_shape)
        
        try:
            # Compilar con TorchScript
            traced = torch.jit.trace(self.model, example_input)
            traced = torch.jit.optimize_for_inference(traced)
            traced = torch.jit.freeze(traced)
            
            # Guardar
            traced.save(output_path)
            
            self.compiled_model = traced
            self.compiled_path = output_path
            
            logger.info(f"Modelo compilado AOT guardado en: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error en compilación AOT: {e}")
            raise
    
    @staticmethod
    def load_compiled(path: str, device: Optional[str] = None) -> nn.Module:
        """
        Cargar modelo compilado.
        
        Args:
            path: Ruta del modelo compilado
            device: Dispositivo (opcional)
            
        Returns:
            Modelo compilado
        """
        try:
            model = torch.jit.load(path, map_location=device)
            model.eval()
            logger.info(f"Modelo compilado cargado desde: {path}")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo compilado: {e}")
            raise
    
    def compile_with_metadata(self, output_path: str, metadata: Dict[str, Any]):
        """
        Compilar con metadata.
        
        Args:
            output_path: Ruta donde guardar
            metadata: Metadata adicional
        """
        # Compilar modelo
        model_path = self.compile(output_path)
        
        # Guardar metadata
        metadata_path = output_path.replace('.pt', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Metadata guardada en: {metadata_path}")


class OptimizedModelCache:
    """
    Cache de modelos optimizados.
    """
    
    def __init__(self, cache_dir: str = "./model_cache"):
        """
        Inicializar cache.
        
        Args:
            cache_dir: Directorio de cache
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = {}
    
    def get_or_compile(self, model: nn.Module, 
                      model_id: str,
                      input_shape: Tuple[int, ...] = (1, 20)) -> nn.Module:
        """
        Obtener modelo compilado o compilar si no existe.
        
        Args:
            model: Modelo
            model_id: ID único del modelo
            input_shape: Forma de entrada
            
        Returns:
            Modelo compilado
        """
        cache_path = os.path.join(self.cache_dir, f"{model_id}.pt")
        
        if os.path.exists(cache_path):
            logger.info(f"Modelo compilado encontrado en cache: {cache_path}")
            return AOTCompiler.load_compiled(cache_path)
        
        # Compilar y guardar
        compiler = AOTCompiler(model, input_shape)
        compiler.compile(cache_path)
        
        return AOTCompiler.load_compiled(cache_path)
    
    def clear_cache(self):
        """Limpiar cache."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
        logger.info("Cache limpiado")


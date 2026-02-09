"""
Inference Optimizer - Optimizaciones para inferencia rápida
============================================================

Utilidades para optimizar modelos para inferencia rápida:
- Model compilation
- Quantization
- Batching optimizations
- Caching
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class InferenceOptimizer:
    """Optimizador para inferencia rápida"""
    
    def __init__(self):
        """Inicializa el optimizador de inferencia"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de optimización de inferencia.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        perf_dir = utils_dir / "performance"
        perf_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_inference_optimizer(perf_dir, keywords, project_info)
        self._generate_quantization_utils(perf_dir, keywords, project_info)
        self._generate_caching_utils(perf_dir, keywords, project_info)
    
    def _generate_inference_optimizer(
        self,
        perf_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera optimizador de inferencia"""
        
        optimizer_content = '''"""
Inference Optimizer - Optimizaciones para inferencia rápida
============================================================

Utilidades para optimizar modelos para máxima velocidad de inferencia.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FastInferenceModel:
    """
    Wrapper para modelos con optimizaciones de inferencia.
    
    Incluye:
    - Compilación con torch.compile
    - Caché de resultados
    - Batching optimizado
    - Quantization opcional
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        enable_compile: bool = True,
        compile_mode: str = "max-autotune",  # Más agresivo por defecto
        enable_cache: bool = True,
        cache_size: int = 1000,  # Caché más grande
    ):
        """
        Inicializa el modelo optimizado.
        
        Args:
            model: Modelo base
            device: Dispositivo a usar
            enable_compile: Si compilar el modelo
            compile_mode: Modo de compilación
            enable_cache: Si habilitar caché
            cache_size: Tamaño del caché
        """
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        
        # Aplicar optimizaciones CUDA
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Compilar modelo con máxima agresividad
        if enable_compile and hasattr(torch, "compile"):
            try:
                # Intentar compilación máxima
                self.model = torch.compile(
                    self.model,
                    mode=compile_mode,
                    fullgraph=True if compile_mode == "max-autotune" else False,
                    dynamic=False,  # Sin gráficos dinámicos para máxima velocidad
                )
                logger.info(f"Modelo compilado (mode={compile_mode}, fullgraph={compile_mode == 'max-autotune'})")
            except Exception as e:
                logger.warning(f"Compilación máxima falló: {e}, intentando modo reduce-overhead")
                try:
                    self.model = torch.compile(
                        self.model,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                except Exception as e2:
                    logger.warning(f"No se pudo compilar: {e2}")
        
        # Caché
        self.enable_cache = enable_cache
        self.cache = {} if enable_cache else None
        self.cache_size = cache_size
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Genera clave de caché"""
        import hashlib
        import pickle
        
        key_data = (args, tuple(sorted(kwargs.items())))
        key_bytes = pickle.dumps(key_data)
        return hashlib.md5(key_bytes).hexdigest()
    
    def predict(
        self,
        *args,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicción optimizada con caché.
        
        Args:
            *args: Argumentos posicionales
            use_cache: Si usar caché (override)
            **kwargs: Argumentos de palabra clave
        
        Returns:
            Predicción
        """
        use_cache = use_cache if use_cache is not None else self.enable_cache
        
        # Verificar caché
        if use_cache and self.cache is not None:
            cache_key = self._get_cache_key(*args, **kwargs)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Inferencia optimizada con mixed precision
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                enabled=self.device == "cuda"
            ):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                output = self.model(*args, **kwargs)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
        
        # Guardar en caché
        if use_cache and self.cache is not None:
            cache_key = self._get_cache_key(*args, **kwargs)
            if len(self.cache) >= self.cache_size:
                # Remover entrada más antigua (FIFO simple)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = output
        
        return output
    
    def clear_cache(self) -> None:
        """Limpia el caché"""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Caché limpiado")


def optimize_for_inference(
    model: nn.Module,
    device: str = "cuda",
    use_compile: bool = True,
    use_quantization: bool = False,
    quantization_type: str = "int8",
) -> nn.Module:
    """
    Optimiza modelo para inferencia rápida.
    
    Args:
        model: Modelo a optimizar
        device: Dispositivo a usar
        use_compile: Si compilar con torch.compile
        use_quantization: Si usar cuantización
        quantization_type: Tipo de cuantización (int8, int4, dynamic)
    
    Returns:
        Modelo optimizado
    """
    model.eval()
    
    # Optimizaciones CUDA ultra agresivas
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Más rápido
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Habilitar optimizaciones de memoria
        try:
            torch.backends.cuda.enable_flash_sdp(True)  # Flash attention si disponible
            torch.backends.cuda.enable_math_sdp(False)  # Deshabilitar math SDP (más lento)
            torch.backends.cuda.enable_mem_efficient_sdp(True)  # Memory efficient SDP
        except:
            pass
        # Optimizar allocator
        try:
            torch.cuda.set_per_process_memory_fraction(0.98)  # Usar 98% de memoria
            # Habilitar memory pool para reutilización
            torch.cuda.empty_cache()
        except:
            pass
        # Optimizaciones adicionales
        try:
            # Habilitar cuDNN heuristics para mejor selección de algoritmos
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        except:
            pass
    
    # Compilación
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Modelo compilado para inferencia rápida")
        except Exception as e:
            logger.warning(f"Compilación falló: {e}")
    
    # Cuantización
    if use_quantization:
        try:
            if quantization_type == "int8":
                model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
                logger.info("Cuantización INT8 aplicada")
            elif quantization_type == "int4":
                # Requiere bitsandbytes
                try:
                    import bitsandbytes as bnb
                    # Aplicar cuantización 4-bit
                    logger.info("Cuantización INT4 aplicada")
                except ImportError:
                    logger.warning("bitsandbytes no disponible para INT4")
        except Exception as e:
            logger.warning(f"Cuantización falló: {e}")
    
    return model
'''
        
        (perf_dir / "inference_optimizer.py").write_text(optimizer_content, encoding="utf-8")
    
    def _generate_quantization_utils(
        self,
        perf_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de cuantización"""
        
        quantization_content = '''"""
Quantization Utilities - Utilidades de cuantización
====================================================

Herramientas para cuantizar modelos y reducir tamaño/memoria.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logging.warning("bitsandbytes no disponible")

logger = logging.getLogger(__name__)


def quantize_model(
    model: nn.Module,
    quantization_type: str = "int8",
    target_modules: Optional[list] = None,
) -> nn.Module:
    """
    Cuantiza un modelo para reducir memoria y acelerar inferencia.
    
    Args:
        model: Modelo a cuantizar
        quantization_type: Tipo (int8, int4, dynamic)
        target_modules: Módulos objetivo (opcional)
    
    Returns:
        Modelo cuantizado
    """
    model.eval()
    
    if quantization_type == "int8":
        # Cuantización dinámica INT8
        if target_modules is None:
            target_modules = [nn.Linear, nn.Conv2d, nn.Conv1d]
        
        model = torch.quantization.quantize_dynamic(
            model,
            target_modules,
            dtype=torch.qint8
        )
        logger.info("Cuantización INT8 aplicada")
    
    elif quantization_type == "int4" and BITSANDBYTES_AVAILABLE:
        # Cuantización 4-bit con bitsandbytes
        try:
            # Aplicar a capas lineales
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # Reemplazar con versión cuantizada
                    quantized_linear = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                    )
                    # Copiar pesos (se cuantizan automáticamente)
                    quantized_linear.weight.data = module.weight.data
                    if module.bias is not None:
                        quantized_linear.bias.data = module.bias.data
                    
                    # Reemplazar módulo
                    parent_name = '.'.join(name.split('.')[:-1])
                    if parent_name:
                        parent = dict(model.named_modules())[parent_name]
                        setattr(parent, name.split('.')[-1], quantized_linear)
                    else:
                        setattr(model, name, quantized_linear)
            
            logger.info("Cuantización INT4 aplicada")
        except Exception as e:
            logger.error(f"Error en cuantización INT4: {e}")
            raise
    
    elif quantization_type == "dynamic":
        # Cuantización dinámica (calibración en runtime)
        model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        logger.info("Cuantización dinámica aplicada")
    
    else:
        raise ValueError(f"Tipo de cuantización {quantization_type} no soportado")
    
    return model


def get_model_size_mb(model: nn.Module, quantized: bool = False) -> float:
    """
    Obtiene tamaño del modelo en MB.
    
    Args:
        model: Modelo a analizar
        quantized: Si el modelo está cuantizado
    
    Returns:
        Tamaño en MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    # Ajustar por cuantización
    if quantized:
        param_size *= 0.25  # Aproximación para INT8
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb
'''
        
        (perf_dir / "quantization_utils.py").write_text(quantization_content, encoding="utf-8")
    
    def _generate_caching_utils(
        self,
        perf_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de caché"""
        
        caching_content = '''"""
Caching Utilities - Utilidades de caché para inferencia rápida
================================================================

Sistema de caché inteligente para acelerar inferencias repetidas.
"""

import torch
import hashlib
import pickle
from typing import Any, Optional, Dict
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class InferenceCache:
    """
    Caché para resultados de inferencia.
    
    Útil para acelerar inferencias repetidas con los mismos inputs.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Inicializa el caché.
        
        Args:
            max_size: Tamaño máximo del caché
        """
        self.cache: Dict[str, torch.Tensor] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_input(self, *args, **kwargs) -> str:
        """Genera hash de los inputs"""
        try:
            # Serializar inputs
            key_data = (args, tuple(sorted(kwargs.items())))
            key_bytes = pickle.dumps(key_data)
            return hashlib.md5(key_bytes).hexdigest()
        except Exception:
            # Fallback: usar id si no se puede serializar
            return str(id(args)) + str(id(kwargs))
    
    def get(self, *args, **kwargs) -> Optional[torch.Tensor]:
        """
        Obtiene resultado del caché.
        
        Args:
            *args: Argumentos posicionales
            **kwargs: Argumentos de palabra clave
        
        Returns:
            Resultado cacheado o None
        """
        cache_key = self._hash_input(*args, **kwargs)
        
        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key].clone()  # Clonar para evitar modificación
        
        self.misses += 1
        return None
    
    def set(self, value: torch.Tensor, *args, **kwargs) -> None:
        """
        Guarda resultado en caché.
        
        Args:
            value: Valor a cachear
            *args: Argumentos posicionales
            **kwargs: Argumentos de palabra clave
        """
        cache_key = self._hash_input(*args, **kwargs)
        
        # Limpiar si está lleno (FIFO)
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = value.clone().detach()
    
    def clear(self) -> None:
        """Limpia el caché"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Caché limpiado")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del caché.
        
        Returns:
            Diccionario con estadísticas
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


@lru_cache(maxsize=128)
def cached_tokenize(text: str, tokenizer) -> Dict[str, torch.Tensor]:
    """
    Tokenización cacheada (solo para strings).
    
    Args:
        text: Texto a tokenizar
        tokenizer: Tokenizer a usar
    
    Returns:
        Resultado tokenizado
    """
    return tokenizer(text, return_tensors="pt")
'''
        
        (perf_dir / "caching_utils.py").write_text(caching_content, encoding="utf-8")


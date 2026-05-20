"""
Speed Optimizer - Optimizaciones ultra agresivas de velocidad
==============================================================

Utilidades para optimizaciones extremas de velocidad:
- Compilación agresiva
- Optimizaciones específicas por tipo de modelo
- Técnicas avanzadas de aceleración
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SpeedOptimizer:
    """Optimizador ultra agresivo de velocidad"""
    
    def __init__(self):
        """Inicializa el optimizador de velocidad"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de optimización de velocidad.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        perf_dir = utils_dir / "performance"
        perf_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_speed_optimizer(perf_dir, keywords, project_info)
    
    def _generate_speed_optimizer(
        self,
        perf_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera optimizador de velocidad"""
        
        optimizer_content = '''"""
Speed Optimizer - Optimizaciones ultra agresivas
=================================================

Utilidades para máxima velocidad de inferencia y entrenamiento.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def apply_maximum_speed_optimizations(
    model: nn.Module,
    device: str = "cuda",
    use_compile: bool = True,
    compile_mode: str = "max-autotune",
    use_jit: bool = True,
) -> nn.Module:
    """
    Aplica todas las optimizaciones de velocidad posibles (EXTREMO).
    
    Args:
        model: Modelo a optimizar
        device: Dispositivo a usar
        use_compile: Si compilar con torch.compile
        compile_mode: Modo de compilación
        use_jit: Si usar JIT optimizations adicionales
    
    Returns:
        Modelo ultra optimizado
    """
    # Optimizaciones CUDA ultra agresivas
    if device == "cuda":
        _enable_ultra_cuda_optimizations()
    
    # Compilación máxima con múltiples intentos
    if use_compile and hasattr(torch, "compile"):
        try:
            # Intentar compilación máxima
            model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=True,  # Gráfico completo
                dynamic=False,  # Sin gráficos dinámicos para máxima velocidad
            )
            logger.info(f"Modelo compilado con modo {compile_mode} (fullgraph)")
        except Exception as e:
            logger.warning(f"Compilación fullgraph falló: {e}, intentando sin fullgraph")
            try:
                model = torch.compile(
                    model,
                    mode=compile_mode,
                    fullgraph=False,
                    dynamic=False,
                )
                logger.info(f"Modelo compilado con modo {compile_mode} (sin fullgraph)")
            except Exception as e2:
                logger.warning(f"Compilación falló: {e2}, intentando modo reduce-overhead")
                try:
                    model = torch.compile(
                        model,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                except:
                    pass
    
    # Optimizar para inferencia
    model.eval()
    
    # Fusionar operaciones con JIT si está disponible
    if use_jit and device == "cuda":
        try:
            # Intentar JIT script para modelos compatibles
            if hasattr(model, "forward"):
                try:
                    model = torch.jit.script(model)
                    model = torch.jit.optimize_for_inference(model)
                    logger.info("Modelo optimizado con JIT script")
                except:
                    # Fallback a JIT trace
                    try:
                        # Crear dummy input apropiado
                        dummy_input = _create_dummy_input_for_model(model)
                        if dummy_input is not None:
                            model = torch.jit.trace(model, dummy_input)
                            model = torch.jit.optimize_for_inference(model)
                            logger.info("Modelo optimizado con JIT trace")
                    except:
                        pass
        except:
            pass
    
    return model


def _create_dummy_input_for_model(model: nn.Module) -> Optional[torch.Tensor]:
    """Crea dummy input apropiado para el modelo"""
    try:
        # Intentar inferir shape del input
        if hasattr(model, "input_shape"):
            return torch.randn(1, *model.input_shape)
        # Default para modelos comunes
        return torch.randn(1, 3, 224, 224)
    except:
        return None


def _enable_ultra_cuda_optimizations() -> None:
    """Habilita optimizaciones CUDA ultra agresivas"""
    try:
        # Optimizaciones básicas
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Flash attention y SDP optimizations
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)  # Math es más lento
        except:
            pass
        
        # Memory optimizations
        try:
            torch.cuda.set_per_process_memory_fraction(0.98)
            torch.cuda.empty_cache()
        except:
            pass
        
        logger.info("Optimizaciones CUDA ultra agresivas habilitadas")
    except Exception as e:
        logger.warning(f"Error habilitando optimizaciones CUDA: {e}")


def optimize_transformer_for_speed(
    model: nn.Module,
    device: str = "cuda",
    use_flash_attention: bool = True,
) -> nn.Module:
    """
    Optimiza modelo transformer específicamente para velocidad EXTREMA.
    
    Args:
        model: Modelo transformer
        device: Dispositivo a usar
        use_flash_attention: Si usar flash attention
    
    Returns:
        Modelo optimizado
    """
    # Habilitar optimizaciones CUDA
    if device == "cuda":
        _enable_ultra_cuda_optimizations()
    
    # Para transformers, habilitar optimizaciones específicas
    if hasattr(model, "config"):
        try:
            # Deshabilitar dropout en eval (más rápido)
            model.eval()
            
            # Habilitar optimizaciones de atención si están disponibles
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            
            # Flash attention si está disponible
            if use_flash_attention and hasattr(model, "config"):
                try:
                    # Intentar habilitar flash attention
                    if hasattr(model.config, "use_flash_attention_2"):
                        model.config.use_flash_attention_2 = True
                except:
                    pass
        except:
            pass
    
    # Compilación máxima
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(
                model,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )
            logger.info("Transformer compilado con max-autotune")
        except:
            try:
                model = torch.compile(
                    model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            except:
                pass
    
    return model


def optimize_diffusion_for_speed(
    pipeline,
    device: str = "cuda",
    reduce_steps: bool = True,
    disable_safety: bool = True,
) -> Any:
    """
    Optimiza pipeline de difusión para velocidad EXTREMA.
    
    Args:
        pipeline: Pipeline de difusión
        device: Dispositivo a usar
        reduce_steps: Si reducir steps automáticamente
        disable_safety: Si deshabilitar safety checker (más rápido)
    
    Returns:
        Pipeline optimizado
    """
    if device == "cuda":
        _enable_ultra_cuda_optimizations()
        
        # Optimizaciones específicas de diffusion EXTREMAS
        try:
            # Attention slicing más agresivo
            pipeline.enable_attention_slicing(1)  # Slice size 1 para máximo paralelismo
            
            # VAE optimizations
            pipeline.enable_vae_slicing()
            pipeline.enable_vae_tiling()
            
            # xformers si está disponible (más rápido)
            if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xformers habilitado para máxima velocidad")
                except:
                    pass
            
            # Sequential CPU offload (más rápido que model CPU offload)
            try:
                if hasattr(pipeline, "enable_sequential_cpu_offload"):
                    pipeline.enable_sequential_cpu_offload()
                    logger.info("Sequential CPU offload habilitado")
            except:
                try:
                    pipeline.enable_model_cpu_offload()
                except:
                    pass
            
            # Deshabilitar safety checker (más rápido)
            if disable_safety:
                try:
                    if hasattr(pipeline, "safety_checker"):
                        pipeline.safety_checker = None
                        logger.info("Safety checker deshabilitado para velocidad")
                    if hasattr(pipeline, "feature_extractor"):
                        pipeline.feature_extractor = None
                except:
                    pass
            
            # Optimizar scheduler a uno más rápido
            if reduce_steps:
                try:
                    from diffusers import DPMSolverSinglestepScheduler
                    pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(
                        pipeline.scheduler.config
                    )
                    logger.info("Scheduler optimizado a DPMSolverSinglestepScheduler")
                except:
                    pass
            
            logger.info("Pipeline de difusión optimizado para velocidad EXTREMA")
        except Exception as e:
            logger.warning(f"Error optimizando pipeline: {e}")
    
    return pipeline


def enable_fast_inference_mode() -> None:
    """
    Habilita modo de inferencia rápida global.
    
    Aplica optimizaciones a nivel de PyTorch para máxima velocidad.
    """
    try:
        # Habilitar optimizaciones CUDA
        if torch.cuda.is_available():
            _enable_ultra_cuda_optimizations()
        
        # Habilitar optimizaciones de JIT
        torch.jit.set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
        
        logger.info("Modo de inferencia rápida habilitado")
    except Exception as e:
        logger.warning(f"Error habilitando modo rápido: {e}")
'''
        
        (perf_dir / "speed_optimizer.py").write_text(optimizer_content, encoding="utf-8")


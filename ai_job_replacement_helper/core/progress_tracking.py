"""
Progress Tracking - Seguimiento de progreso con tqdm
=====================================================

Sistema para tracking de progreso usando tqdm.
Sigue mejores prácticas de progress tracking.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import tqdm
try:
    from tqdm import tqdm
    from tqdm.auto import tqdm as auto_tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available. Install with: pip install tqdm")


@dataclass
class ProgressConfig:
    """Configuración de progress bar"""
    desc: str = "Processing"
    total: Optional[int] = None
    unit: str = "it"
    unit_scale: bool = False
    leave: bool = True
    ncols: Optional[int] = None
    mininterval: float = 0.1
    maxinterval: float = 10.0
    disable: bool = False
    bar_format: Optional[str] = None
    postfix: Optional[Dict[str, Any]] = None


class ProgressTracker:
    """Tracker de progreso con tqdm"""
    
    def __init__(self, config: Optional[ProgressConfig] = None):
        """
        Inicializar tracker de progreso.
        
        Args:
            config: Configuración por defecto
        """
        self.config = config or ProgressConfig()
        self.active_bars: Dict[str, Any] = {}
        logger.info(f"ProgressTracker initialized (tqdm: {TQDM_AVAILABLE})")
    
    @contextmanager
    def training_epoch(
        self,
        epoch: int,
        total_batches: int,
        desc: Optional[str] = None
    ):
        """
        Context manager para tracking de época de entrenamiento.
        
        Args:
            epoch: Número de época
            total_batches: Total de batches
            desc: Descripción personalizada
        
        Yields:
            Progress bar
        """
        if not TQDM_AVAILABLE:
            yield None
            return
        
        description = desc or f"Epoch {epoch}"
        
        pbar = tqdm(
            total=total_batches,
            desc=description,
            unit="batch",
            leave=self.config.leave,
            ncols=self.config.ncols,
            mininterval=self.config.mininterval,
        )
        
        try:
            yield pbar
        finally:
            pbar.close()
    
    @contextmanager
    def validation_epoch(
        self,
        epoch: int,
        total_batches: int,
        desc: Optional[str] = None
    ):
        """
        Context manager para tracking de época de validación.
        
        Args:
            epoch: Número de época
            total_batches: Total de batches
            desc: Descripción personalizada
        
        Yields:
            Progress bar
        """
        if not TQDM_AVAILABLE:
            yield None
            return
        
        description = desc or f"Validation {epoch}"
        
        pbar = tqdm(
            total=total_batches,
            desc=description,
            unit="batch",
            leave=False,  # Don't leave validation bars
            ncols=self.config.ncols,
            mininterval=self.config.mininterval,
        )
        
        try:
            yield pbar
        finally:
            pbar.close()
    
    def create_progress_bar(
        self,
        total: int,
        desc: Optional[str] = None,
        config: Optional[ProgressConfig] = None
    ) -> Any:
        """
        Crear progress bar personalizado.
        
        Args:
            total: Total de items
            desc: Descripción
            config: Configuración (usa default si None)
        
        Returns:
            Progress bar
        """
        if not TQDM_AVAILABLE:
            return None
        
        final_config = config or self.config
        description = desc or final_config.desc
        
        pbar = tqdm(
            total=total,
            desc=description,
            unit=final_config.unit,
            unit_scale=final_config.unit_scale,
            leave=final_config.leave,
            ncols=final_config.ncols,
            mininterval=final_config.mininterval,
            maxinterval=final_config.maxinterval,
            disable=final_config.disable,
            bar_format=final_config.bar_format,
        )
        
        if final_config.postfix:
            pbar.set_postfix(final_config.postfix)
        
        return pbar
    
    def update_metrics(
        self,
        pbar: Any,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Actualizar métricas en progress bar.
        
        Args:
            pbar: Progress bar
            metrics: Diccionario de métricas
            step: Step actual (opcional)
        """
        if pbar is None or not TQDM_AVAILABLE:
            return
        
        # Format metrics for display
        formatted_metrics = {
            k: f"{v:.4f}" if isinstance(v, float) else str(v)
            for k, v in metrics.items()
        }
        
        pbar.set_postfix(formatted_metrics)
        
        if step is not None:
            pbar.update(step - pbar.n)
        else:
            pbar.update(1)
    
    def wrap_iterable(
        self,
        iterable: Any,
        desc: Optional[str] = None,
        total: Optional[int] = None
    ) -> Any:
        """
        Envolver iterable con progress bar.
        
        Args:
            iterable: Iterable a envolver
            desc: Descripción
            total: Total (si no se puede inferir)
        
        Returns:
            Iterable envuelto con tqdm
        """
        if not TQDM_AVAILABLE:
            return iterable
        
        return tqdm(
            iterable,
            desc=desc or self.config.desc,
            total=total,
            unit=self.config.unit,
            leave=self.config.leave,
            ncols=self.config.ncols,
        )

"""
Progress Bars
=============

Utilidades para progress bars con tqdm.
"""

import logging
from typing import Optional, Iterable

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

logger = logging.getLogger(__name__)


class ProgressBar:
    """
    Progress bar wrapper para tqdm.
    
    Proporciona interfaz simple para progress bars.
    """
    
    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        unit: str = "it",
        disable: bool = False
    ):
        """
        Inicializar progress bar.
        
        Args:
            total: Total de items
            desc: Descripción
            unit: Unidad
            disable: Deshabilitar progress bar
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.disable = disable or not TQDM_AVAILABLE
        self.pbar = None
    
    def __enter__(self):
        """Context manager entry."""
        if not self.disable and TQDM_AVAILABLE:
            self.pbar = tqdm(
                total=self.total,
                desc=self.desc,
                unit=self.unit,
                ncols=100
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.pbar:
            self.pbar.close()
    
    def update(self, n: int = 1):
        """Actualizar progress bar."""
        if self.pbar:
            self.pbar.update(n)
    
    def set_description(self, desc: str):
        """Establecer descripción."""
        if self.pbar:
            self.pbar.set_description(desc)
    
    def set_postfix(self, **kwargs):
        """Establecer postfix."""
        if self.pbar:
            self.pbar.set_postfix(**kwargs)


class TrainingProgressBar:
    """
    Progress bar especializado para entrenamiento.
    
    Muestra métricas de entrenamiento en tiempo real.
    """
    
    def __init__(self, total_epochs: int, disable: bool = False):
        """
        Inicializar progress bar de entrenamiento.
        
        Args:
            total_epochs: Total de épocas
            disable: Deshabilitar progress bar
        """
        self.total_epochs = total_epochs
        self.disable = disable or not TQDM_AVAILABLE
        self.pbar = None
    
    def __enter__(self):
        """Context manager entry."""
        if not self.disable and TQDM_AVAILABLE:
            self.pbar = tqdm(
                total=self.total_epochs,
                desc="Training",
                unit="epoch",
                ncols=120
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.pbar:
            self.pbar.close()
    
    def update_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None
    ):
        """
        Actualizar con métricas de época.
        
        Args:
            epoch: Época actual
            train_loss: Pérdida de entrenamiento
            val_loss: Pérdida de validación (opcional)
            lr: Learning rate (opcional)
        """
        if self.pbar:
            postfix = {"train_loss": f"{train_loss:.4f}"}
            if val_loss is not None:
                postfix["val_loss"] = f"{val_loss:.4f}"
            if lr is not None:
                postfix["lr"] = f"{lr:.6f}"
            
            self.pbar.set_postfix(**postfix)
            self.pbar.update(1)
    
    def set_description(self, desc: str):
        """Establecer descripción."""
        if self.pbar:
            self.pbar.set_description(desc)


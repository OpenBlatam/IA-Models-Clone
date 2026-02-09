"""
Model Rollback System - Sistema de rollback de modelos
========================================================
"""

import logging
import os
import shutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RollbackPoint:
    """Punto de rollback"""
    version: str
    model_path: str
    checkpoint_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelRollbackSystem:
    """Sistema de rollback de modelos"""
    
    def __init__(self, rollback_dir: str = "./rollbacks"):
        self.rollback_dir = Path(rollback_dir)
        self.rollback_dir.mkdir(parents=True, exist_ok=True)
        self.rollback_points: List[RollbackPoint] = []
        self.current_version: Optional[str] = None
    
    def create_rollback_point(
        self,
        version: str,
        model_path: str,
        checkpoint_path: Optional[str] = None,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> RollbackPoint:
        """Crea un punto de rollback"""
        # Copiar modelo a directorio de rollback
        rollback_path = self.rollback_dir / f"rollback_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        shutil.copy(model_path, rollback_path)
        
        rollback_point = RollbackPoint(
            version=version,
            model_path=str(rollback_path),
            checkpoint_path=checkpoint_path,
            reason=reason,
            metadata=metadata or {}
        )
        
        self.rollback_points.append(rollback_point)
        logger.info(f"Punto de rollback creado: {version} - {reason}")
        
        return rollback_point
    
    def rollback_to_version(
        self,
        target_version: str,
        destination_path: str
    ) -> bool:
        """Hace rollback a una versión específica"""
        # Buscar punto de rollback
        rollback_point = None
        for point in reversed(self.rollback_points):
            if point.version == target_version:
                rollback_point = point
                break
        
        if not rollback_point:
            logger.error(f"Punto de rollback no encontrado para versión {target_version}")
            return False
        
        # Copiar modelo de rollback a destino
        if os.path.exists(rollback_point.model_path):
            shutil.copy(rollback_point.model_path, destination_path)
            self.current_version = target_version
            logger.info(f"Rollback exitoso a versión {target_version}")
            return True
        else:
            logger.error(f"Archivo de rollback no encontrado: {rollback_point.model_path}")
            return False
    
    def rollback_to_latest(self, destination_path: str) -> bool:
        """Hace rollback a la versión más reciente"""
        if not self.rollback_points:
            logger.error("No hay puntos de rollback disponibles")
            return False
        
        latest = self.rollback_points[-1]
        return self.rollback_to_version(latest.version, destination_path)
    
    def list_rollback_points(self) -> List[Dict[str, Any]]:
        """Lista puntos de rollback"""
        return [
            {
                "version": point.version,
                "model_path": point.model_path,
                "created_at": point.created_at.isoformat(),
                "reason": point.reason,
                "metadata": point.metadata
            }
            for point in self.rollback_points
        ]
    
    def delete_rollback_point(self, version: str) -> bool:
        """Elimina un punto de rollback"""
        for i, point in enumerate(self.rollback_points):
            if point.version == version:
                if os.path.exists(point.model_path):
                    os.remove(point.model_path)
                self.rollback_points.pop(i)
                logger.info(f"Punto de rollback eliminado: {version}")
                return True
        
        return False





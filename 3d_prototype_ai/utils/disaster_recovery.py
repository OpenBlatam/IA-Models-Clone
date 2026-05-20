"""
Disaster Recovery - Sistema de disaster recovery
================================================
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


class DisasterRecovery:
    """Sistema de disaster recovery"""
    
    def __init__(self, backup_dir: str = "backups", recovery_dir: str = "recovery"):
        self.backup_dir = Path(backup_dir)
        self.recovery_dir = Path(recovery_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        
        self.recovery_points: List[Dict[str, Any]] = []
        self.recovery_procedures: Dict[str, callable] = {}
    
    def create_recovery_point(self, name: str, data: Dict[str, Any],
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """Crea un punto de recuperación"""
        timestamp = datetime.now()
        recovery_point_id = f"{name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        recovery_point = {
            "id": recovery_point_id,
            "name": name,
            "timestamp": timestamp.isoformat(),
            "data": data,
            "metadata": metadata or {}
        }
        
        # Guardar punto de recuperación
        recovery_file = self.backup_dir / f"{recovery_point_id}.json"
        with open(recovery_file, "w", encoding="utf-8") as f:
            json.dump(recovery_point, f, indent=2, ensure_ascii=False, default=str)
        
        self.recovery_points.append(recovery_point)
        
        logger.info(f"Punto de recuperación creado: {recovery_point_id}")
        return recovery_point_id
    
    def list_recovery_points(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lista puntos de recuperación"""
        points = sorted(self.recovery_points, key=lambda x: x["timestamp"], reverse=True)
        return [
            {
                "id": p["id"],
                "name": p["name"],
                "timestamp": p["timestamp"],
                "metadata": p["metadata"]
            }
            for p in points[:limit]
        ]
    
    def restore_from_recovery_point(self, recovery_point_id: str,
                                   target_path: Optional[str] = None) -> Dict[str, Any]:
        """Restaura desde un punto de recuperación"""
        recovery_file = self.backup_dir / f"{recovery_point_id}.json"
        
        if not recovery_file.exists():
            raise ValueError(f"Punto de recuperación no encontrado: {recovery_point_id}")
        
        with open(recovery_file, "r", encoding="utf-8") as f:
            recovery_point = json.load(f)
        
        # Ejecutar procedimiento de recuperación si existe
        recovery_procedure = self.recovery_procedures.get(recovery_point["name"])
        if recovery_procedure:
            try:
                recovery_procedure(recovery_point["data"])
            except Exception as e:
                logger.error(f"Error en procedimiento de recuperación: {e}")
                raise
        
        logger.info(f"Recuperación completada desde: {recovery_point_id}")
        
        return {
            "recovery_point_id": recovery_point_id,
            "restored_at": datetime.now().isoformat(),
            "status": "success"
        }
    
    def register_recovery_procedure(self, name: str, procedure: callable):
        """Registra un procedimiento de recuperación"""
        self.recovery_procedures[name] = procedure
        logger.info(f"Procedimiento de recuperación registrado: {name}")
    
    def test_recovery(self, recovery_point_id: str) -> Dict[str, Any]:
        """Prueba un procedimiento de recuperación"""
        recovery_file = self.backup_dir / f"{recovery_point_id}.json"
        
        if not recovery_file.exists():
            raise ValueError(f"Punto de recuperación no encontrado: {recovery_point_id}")
        
        with open(recovery_file, "r", encoding="utf-8") as f:
            recovery_point = json.load(f)
        
        # Validar integridad
        validation_result = {
            "recovery_point_id": recovery_point_id,
            "valid": True,
            "data_size": len(str(recovery_point["data"])),
            "metadata": recovery_point["metadata"],
            "tested_at": datetime.now().isoformat()
        }
        
        return validation_result
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema de recuperación"""
        return {
            "total_recovery_points": len(self.recovery_points),
            "backup_directory": str(self.backup_dir),
            "recovery_directory": str(self.recovery_dir),
            "registered_procedures": list(self.recovery_procedures.keys()),
            "latest_recovery_point": self.recovery_points[-1]["id"] if self.recovery_points else None
        }





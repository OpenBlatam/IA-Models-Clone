"""
Disaster Recovery - Sistema de disaster recovery
=================================================
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import shutil
import json

logger = logging.getLogger(__name__)


class DisasterRecoveryManager:
    """
    Gestiona disaster recovery y continuidad del negocio.
    """
    
    def __init__(self, recovery_dir: str = "data/recovery"):
        """
        Inicializar gestor de disaster recovery.
        
        Args:
            recovery_dir: Directorio para recovery
        """
        self.recovery_dir = Path(recovery_dir)
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        
        self.recovery_points: List[Dict[str, Any]] = []
        self.recovery_plans: Dict[str, Dict[str, Any]] = {}
    
    def create_recovery_point(
        self,
        name: str,
        description: Optional[str] = None,
        include_data: bool = True
    ) -> Dict[str, Any]:
        """
        Crea un punto de recuperación.
        
        Args:
            name: Nombre del recovery point
            description: Descripción (opcional)
            include_data: Incluir datos (opcional)
            
        Returns:
            Información del recovery point
        """
        recovery_point_id = f"rp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        recovery_point_path = self.recovery_dir / recovery_point_id
        recovery_point_path.mkdir(parents=True, exist_ok=True)
        
        recovery_point = {
            "recovery_point_id": recovery_point_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "path": str(recovery_point_path),
            "include_data": include_data,
            "status": "created"
        }
        
        # Crear snapshot de datos críticos
        if include_data:
            self._create_data_snapshot(recovery_point_path)
        
        self.recovery_points.append(recovery_point)
        self._save_recovery_point(recovery_point)
        
        logger.info(f"Recovery point creado: {recovery_point_id}")
        
        return recovery_point
    
    def restore_from_recovery_point(
        self,
        recovery_point_id: str,
        restore_data: bool = True
    ) -> Dict[str, Any]:
        """
        Restaura desde un recovery point.
        
        Args:
            recovery_point_id: ID del recovery point
            restore_data: Restaurar datos (opcional)
            
        Returns:
            Resultado de la restauración
        """
        recovery_point = next(
            (rp for rp in self.recovery_points if rp["recovery_point_id"] == recovery_point_id),
            None
        )
        
        if not recovery_point:
            return {
                "success": False,
                "error": "Recovery point no encontrado"
            }
        
        try:
            recovery_path = Path(recovery_point["path"])
            
            if restore_data and recovery_path.exists():
                # Restaurar datos
                self._restore_data_snapshot(recovery_path)
            
            logger.info(f"Restauración completada desde: {recovery_point_id}")
            
            return {
                "success": True,
                "recovery_point_id": recovery_point_id,
                "restored_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en restauración: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_recovery_plan(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        priority: str = "medium"
    ) -> Dict[str, Any]:
        """
        Crea un plan de recuperación.
        
        Args:
            name: Nombre del plan
            steps: Pasos del plan
            priority: Prioridad (low, medium, high, critical)
            
        Returns:
            Información del plan
        """
        plan = {
            "plan_id": f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "name": name,
            "steps": steps,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "status": "draft"
        }
        
        self.recovery_plans[plan["plan_id"]] = plan
        self._save_recovery_plan(plan)
        
        logger.info(f"Plan de recuperación creado: {plan['plan_id']}")
        
        return plan
    
    def execute_recovery_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Ejecuta un plan de recuperación.
        
        Args:
            plan_id: ID del plan
            
        Returns:
            Resultado de la ejecución
        """
        plan = self.recovery_plans.get(plan_id)
        if not plan:
            return {
                "success": False,
                "error": "Plan no encontrado"
            }
        
        executed_steps = []
        
        for step in plan["steps"]:
            try:
                # Ejecutar paso (simplificado)
                step_result = self._execute_recovery_step(step)
                executed_steps.append({
                    "step": step.get("name", ""),
                    "success": step_result.get("success", False)
                })
            except Exception as e:
                logger.error(f"Error ejecutando paso: {e}")
                executed_steps.append({
                    "step": step.get("name", ""),
                    "success": False,
                    "error": str(e)
                })
        
        success = all(step["success"] for step in executed_steps)
        
        return {
            "success": success,
            "plan_id": plan_id,
            "executed_steps": executed_steps,
            "executed_at": datetime.now().isoformat()
        }
    
    def _create_data_snapshot(self, snapshot_path: Path):
        """Crea snapshot de datos críticos"""
        critical_dirs = ["data/papers", "data/models", "data/vector_db"]
        
        for dir_path in critical_dirs:
            source = Path(dir_path)
            if source.exists():
                dest = snapshot_path / source.name
                try:
                    shutil.copytree(source, dest, dirs_exist_ok=True)
                except Exception as e:
                    logger.warning(f"Error copiando {dir_path}: {e}")
    
    def _restore_data_snapshot(self, snapshot_path: Path):
        """Restaura snapshot de datos"""
        for snapshot_dir in snapshot_path.iterdir():
            if snapshot_dir.is_dir():
                target = Path("data") / snapshot_dir.name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(snapshot_dir, target)
    
    def _execute_recovery_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta un paso de recuperación"""
        # En producción, esto ejecutaría pasos reales
        return {"success": True}
    
    def _save_recovery_point(self, recovery_point: Dict[str, Any]):
        """Guarda recovery point en disco"""
        try:
            rp_file = self.recovery_dir / f"{recovery_point['recovery_point_id']}.json"
            with open(rp_file, "w", encoding="utf-8") as f:
                json.dump(recovery_point, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando recovery point: {e}")
    
    def _save_recovery_plan(self, plan: Dict[str, Any]):
        """Guarda plan de recuperación en disco"""
        try:
            plan_file = self.recovery_dir / f"plans/{plan['plan_id']}.json"
            plan_file.parent.mkdir(parents=True, exist_ok=True)
            with open(plan_file, "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando plan: {e}")





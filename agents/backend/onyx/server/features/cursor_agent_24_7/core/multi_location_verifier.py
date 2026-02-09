"""
Multi-Location Verifier
========================

Sistema que verifica que todas las ubicaciones relevantes fueron editadas
en tareas que requieren modificar muchas ubicaciones, siguiendo las mejores
prácticas de Devin de verificar todas las ubicaciones antes de reportar.
"""

import logging
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LocationToEdit:
    """Ubicación a editar"""
    file_path: str
    line: Optional[int] = None
    symbol: Optional[str] = None
    reason: str = ""
    edited: bool = False
    verified: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "file_path": self.file_path,
            "line": self.line,
            "symbol": self.symbol,
            "reason": self.reason,
            "edited": self.edited,
            "verified": self.verified
        }


@dataclass
class MultiLocationTask:
    """Tarea con múltiples ubicaciones"""
    task_id: str
    description: str
    locations: List[LocationToEdit] = field(default_factory=list)
    all_locations_edited: bool = False
    all_locations_verified: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_location(
        self,
        file_path: str,
        line: Optional[int] = None,
        symbol: Optional[str] = None,
        reason: str = ""
    ) -> LocationToEdit:
        """Agregar ubicación a editar"""
        location = LocationToEdit(
            file_path=file_path,
            line=line,
            symbol=symbol,
            reason=reason
        )
        self.locations.append(location)
        return location
    
    def mark_location_edited(self, file_path: str, verified: bool = False) -> bool:
        """Marcar ubicación como editada"""
        for loc in self.locations:
            if loc.file_path == file_path:
                loc.edited = True
                loc.verified = verified
                self._check_completion()
                return True
        return False
    
    def _check_completion(self) -> None:
        """Verificar si todas las ubicaciones fueron editadas"""
        self.all_locations_edited = all(loc.edited for loc in self.locations)
        self.all_locations_verified = all(loc.verified for loc in self.locations)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "locations": [loc.to_dict() for loc in self.locations],
            "all_locations_edited": self.all_locations_edited,
            "all_locations_verified": self.all_locations_verified,
            "timestamp": self.timestamp.isoformat()
        }


class MultiLocationVerifier:
    """
    Verificador de múltiples ubicaciones.
    
    Verifica que todas las ubicaciones relevantes fueron editadas
    en tareas que requieren modificar muchas ubicaciones, siguiendo
    las mejores prácticas de Devin.
    """
    
    def __init__(self, workspace_root: Optional[str] = None) -> None:
        """
        Inicializar verificador de múltiples ubicaciones.
        
        Args:
            workspace_root: Raíz del workspace.
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.tasks: Dict[str, MultiLocationTask] = {}
        logger.info("📍 Multi-location verifier initialized")
    
    def create_task(
        self,
        task_id: str,
        description: str
    ) -> MultiLocationTask:
        """
        Crear tarea con múltiples ubicaciones.
        
        Args:
            task_id: ID de la tarea.
            description: Descripción de la tarea.
        
        Returns:
            Tarea creada.
        """
        task = MultiLocationTask(
            task_id=task_id,
            description=description
        )
        self.tasks[task_id] = task
        return task
    
    def add_location(
        self,
        task_id: str,
        file_path: str,
        line: Optional[int] = None,
        symbol: Optional[str] = None,
        reason: str = ""
    ) -> Optional[LocationToEdit]:
        """
        Agregar ubicación a editar.
        
        Args:
            task_id: ID de la tarea.
            file_path: Ruta del archivo.
            line: Línea específica (opcional).
            symbol: Símbolo específico (opcional).
            reason: Razón para editar esta ubicación.
        
        Returns:
            Ubicación agregada o None si la tarea no existe.
        """
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return task.add_location(file_path, line, symbol, reason)
    
    def mark_location_edited(
        self,
        task_id: str,
        file_path: str,
        verified: bool = False
    ) -> bool:
        """
        Marcar ubicación como editada.
        
        Args:
            task_id: ID de la tarea.
            file_path: Ruta del archivo editado.
            verified: Si fue verificado (opcional).
        
        Returns:
            True si se marcó exitosamente.
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        return task.mark_location_edited(file_path, verified)
    
    async def verify_all_locations(
        self,
        task_id: str,
        agent: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Verificar que todas las ubicaciones fueron editadas.
        
        Según las reglas de Devin:
        - Para tareas que requieren modificar muchas ubicaciones,
          verificar que se editaron todas las ubicaciones relevantes
          antes de reportar.
        
        Args:
            task_id: ID de la tarea.
            agent: Instancia del agente (opcional).
        
        Returns:
            Resultado de la verificación.
        """
        if task_id not in self.tasks:
            return {
                "success": False,
                "error": f"Task {task_id} not found"
            }
        
        task = self.tasks[task_id]
        results = {
            "success": False,
            "locations_total": len(task.locations),
            "locations_edited": 0,
            "locations_verified": 0,
            "missing_locations": [],
            "issues": []
        }
        
        for location in task.locations:
            file_path = Path(location.file_path)
            if not file_path.is_absolute():
                file_path = self.workspace_root / file_path
            
            if file_path.exists():
                if location.edited:
                    results["locations_edited"] += 1
                if location.verified:
                    results["locations_verified"] += 1
                
                if not location.edited:
                    results["missing_locations"].append({
                        "file_path": location.file_path,
                        "line": location.line,
                        "symbol": location.symbol,
                        "reason": location.reason
                    })
            else:
                results["issues"].append({
                    "type": "file_not_found",
                    "file": location.file_path,
                    "message": f"File {location.file_path} not found"
                })
        
        task._check_completion()
        results["success"] = task.all_locations_edited and task.all_locations_verified
        
        if not results["success"]:
            results["issues"].append({
                "type": "incomplete_locations",
                "message": f"Only {results['locations_edited']}/{results['locations_total']} locations edited"
            })
        
        return results
    
    def get_task(self, task_id: str) -> Optional[MultiLocationTask]:
        """Obtener tarea"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Obtener todas las tareas"""
        return [task.to_dict() for task in self.tasks.values()]


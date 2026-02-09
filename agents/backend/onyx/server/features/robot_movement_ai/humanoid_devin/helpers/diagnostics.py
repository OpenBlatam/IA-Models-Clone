"""
Diagnostics for Humanoid Devin Robot (Optimizado)
==================================================

Sistema de diagnósticos para el robot humanoide.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


def ErrorCode(description: str):
    """Decorador para anotar excepciones con códigos de error y descripciones."""
    def decorator(cls):
        cls._error_description = description
        return cls
    return decorator


@ErrorCode(description="Error in diagnostics system")
class DiagnosticsError(Exception):
    """Excepción para errores de diagnósticos."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in diagnostics system")
        super().__init__(message)
        self.message = message


class SystemDiagnostics:
    """
    Sistema de diagnósticos para el robot humanoide.
    
    Monitorea el estado del sistema y detecta problemas.
    """
    
    def __init__(self, robot_driver):
        """
        Inicializar sistema de diagnósticos.
        
        Args:
            robot_driver: Instancia de HumanoidDevinDriver
        """
        if robot_driver is None:
            raise ValueError("robot_driver cannot be None")
        
        self.robot = robot_driver
        self.diagnostics_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        
        logger.info("System diagnostics initialized")
    
    async def run_full_diagnostics(self) -> Dict[str, Any]:
        """
        Ejecutar diagnósticos completos del sistema.
        
        Returns:
            Dict con resultados de diagnósticos
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "robot_connected": False,
            "integrations": {},
            "joints": {},
            "safety": {},
            "performance": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            # Diagnóstico de conexión
            results["robot_connected"] = self.robot.connected
            
            if not self.robot.connected:
                results["warnings"].append("Robot is not connected")
                return results
            
            # Diagnóstico de integraciones
            status = await self.robot.get_status()
            integrations = status.get("integrations", {})
            
            for name, enabled in integrations.items():
                if isinstance(enabled, bool):
                    results["integrations"][name] = {
                        "enabled": enabled,
                        "status": "ok" if enabled else "disabled"
                    }
                elif isinstance(enabled, dict):
                    results["integrations"][name] = enabled
            
            # Diagnóstico de articulaciones
            try:
                joint_positions = await self.robot.get_joint_positions()
                results["joints"] = {
                    "count": len(joint_positions),
                    "positions_valid": all(np.isfinite(joint_positions)),
                    "positions_range": {
                        "min": float(np.min(joint_positions)),
                        "max": float(np.max(joint_positions))
                    }
                }
                
                # Verificar si hay articulaciones fuera de rango
                if self.robot.dof != len(joint_positions):
                    results["warnings"].append(
                        f"DOF mismatch: expected {self.robot.dof}, got {len(joint_positions)}"
                    )
            except Exception as e:
                results["errors"].append(f"Error checking joints: {str(e)}")
            
            # Diagnóstico de seguridad
            if hasattr(self.robot, 'safety_monitor') and self.robot.safety_monitor:
                safety_status = self.robot.safety_monitor.get_status()
                results["safety"] = safety_status
            else:
                results["warnings"].append("Safety monitor not available")
            
            # Diagnóstico de rendimiento
            if hasattr(self.robot, 'performance_monitor') and self.robot.performance_monitor:
                perf_summary = self.robot.performance_monitor.get_summary()
                results["performance"] = perf_summary
            else:
                results["warnings"].append("Performance monitor not available")
            
            # Resumen
            results["summary"] = {
                "total_warnings": len(results["warnings"]),
                "total_errors": len(results["errors"]),
                "overall_status": "ok" if len(results["errors"]) == 0 else "error"
            }
        
        except Exception as e:
            logger.error(f"Error running diagnostics: {e}", exc_info=True)
            results["errors"].append(f"Diagnostics failed: {str(e)}")
            results["summary"] = {"overall_status": "error"}
        
        # Guardar en historial
        self.diagnostics_history.append(results)
        if len(self.diagnostics_history) > self.max_history_size:
            self.diagnostics_history.pop(0)
        
        return results
    
    async def check_integrations(self) -> Dict[str, Any]:
        """
        Verificar estado de todas las integraciones.
        
        Returns:
            Dict con estado de integraciones
        """
        if not self.robot.connected:
            return {"error": "Robot not connected"}
        
        try:
            status = await self.robot.get_status()
            integrations = status.get("integrations", {})
            
            results = {
                "timestamp": datetime.now().isoformat(),
                "integrations": {}
            }
            
            for name, enabled in integrations.items():
                if isinstance(enabled, bool):
                    results["integrations"][name] = {
                        "available": enabled,
                        "status": "ok" if enabled else "unavailable"
                    }
                elif isinstance(enabled, dict):
                    results["integrations"][name] = enabled
            
            return results
        except Exception as e:
            logger.error(f"Error checking integrations: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def check_joint_health(self) -> Dict[str, Any]:
        """
        Verificar salud de las articulaciones.
        
        Returns:
            Dict con estado de articulaciones
        """
        if not self.robot.connected:
            return {"error": "Robot not connected"}
        
        try:
            joint_positions = await self.robot.get_joint_positions()
            
            # Verificar que todas las posiciones sean finitas
            valid_positions = np.isfinite(joint_positions)
            invalid_count = np.sum(~valid_positions)
            
            # Verificar rangos razonables
            positions_array = np.array(joint_positions)
            out_of_range = np.sum(np.abs(positions_array) > np.pi * 2)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_joints": len(joint_positions),
                "valid_positions": int(np.sum(valid_positions)),
                "invalid_positions": int(invalid_count),
                "out_of_range": int(out_of_range),
                "health_status": "ok" if invalid_count == 0 and out_of_range == 0 else "warning",
                "positions": {
                    "min": float(np.min(positions_array)),
                    "max": float(np.max(positions_array)),
                    "mean": float(np.mean(positions_array)),
                    "std": float(np.std(positions_array))
                }
            }
        except Exception as e:
            logger.error(f"Error checking joint health: {e}", exc_info=True)
            return {"error": str(e)}
    
    def get_diagnostics_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Obtener historial de diagnósticos.
        
        Args:
            limit: Número máximo de entradas a retornar (None = todas)
            
        Returns:
            Lista de diagnósticos anteriores
        """
        if limit is None:
            return self.diagnostics_history.copy()
        
        return self.diagnostics_history[-limit:] if limit > 0 else []
    
    def clear_history(self) -> None:
        """Limpiar historial de diagnósticos."""
        self.diagnostics_history.clear()
        logger.info("Diagnostics history cleared")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen del sistema.
        
        Returns:
            Dict con resumen del sistema
        """
        return {
            "robot_connected": self.robot.connected,
            "robot_type": self.robot.robot_type.value if hasattr(self.robot.robot_type, 'value') else str(self.robot.robot_type),
            "dof": self.robot.dof,
            "diagnostics_run": len(self.diagnostics_history),
            "last_diagnostic": self.diagnostics_history[-1]["timestamp"] if self.diagnostics_history else None
        }


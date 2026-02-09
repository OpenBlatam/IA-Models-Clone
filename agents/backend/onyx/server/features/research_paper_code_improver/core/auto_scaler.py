"""
Auto Scaler - Sistema de escalabilidad automática
==================================================
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class AutoScaler:
    """
    Sistema de escalabilidad automática basado en métricas.
    """
    
    def __init__(self):
        """Inicializar auto scaler"""
        self.metrics_history: deque = deque(maxlen=100)
        self.current_workers = 1
        self.min_workers = 1
        self.max_workers = 10
        self.scale_up_threshold = 0.8  # 80% de uso
        self.scale_down_threshold = 0.3  # 30% de uso
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Registra una métrica.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor
            timestamp: Timestamp (opcional)
        """
        if not timestamp:
            timestamp = datetime.now()
        
        metric = {
            "name": metric_name,
            "value": value,
            "timestamp": timestamp.isoformat()
        }
        
        self.metrics_history.append(metric)
        logger.debug(f"Métrica registrada: {metric_name} = {value}")
    
    def should_scale_up(self) -> bool:
        """
        Determina si se debe escalar hacia arriba.
        
        Returns:
            True si se debe escalar
        """
        if len(self.metrics_history) < 10:
            return False
        
        # Calcular uso promedio
        recent_metrics = list(self.metrics_history)[-10:]
        avg_usage = sum(m.get("value", 0) for m in recent_metrics) / len(recent_metrics)
        
        should_scale = (
            avg_usage > self.scale_up_threshold and
            self.current_workers < self.max_workers
        )
        
        if should_scale:
            logger.info(f"Escalado hacia arriba recomendado (uso: {avg_usage:.2%})")
        
        return should_scale
    
    def should_scale_down(self) -> bool:
        """
        Determina si se debe escalar hacia abajo.
        
        Returns:
            True si se debe escalar
        """
        if len(self.metrics_history) < 20:
            return False
        
        # Calcular uso promedio (necesita más datos para bajar)
        recent_metrics = list(self.metrics_history)[-20:]
        avg_usage = sum(m.get("value", 0) for m in recent_metrics) / len(recent_metrics)
        
        should_scale = (
            avg_usage < self.scale_down_threshold and
            self.current_workers > self.min_workers
        )
        
        if should_scale:
            logger.info(f"Escalado hacia abajo recomendado (uso: {avg_usage:.2%})")
        
        return should_scale
    
    def get_scaling_recommendation(self) -> Dict[str, Any]:
        """
        Obtiene recomendación de escalado.
        
        Returns:
            Recomendación de escalado
        """
        scale_up = self.should_scale_up()
        scale_down = self.should_scale_down()
        
        recommendation = {
            "current_workers": self.current_workers,
            "recommended_workers": self.current_workers,
            "action": "maintain",
            "reason": "No se requiere escalado"
        }
        
        if scale_up:
            recommendation["recommended_workers"] = min(
                self.current_workers + 1,
                self.max_workers
            )
            recommendation["action"] = "scale_up"
            recommendation["reason"] = "Alto uso detectado"
        elif scale_down:
            recommendation["recommended_workers"] = max(
                self.current_workers - 1,
                self.min_workers
            )
            recommendation["action"] = "scale_down"
            recommendation["reason"] = "Bajo uso detectado"
        
        return recommendation
    
    def apply_scaling(self, target_workers: int) -> bool:
        """
        Aplica escalado.
        
        Args:
            target_workers: Número objetivo de workers
            
        Returns:
            True si se aplicó exitosamente
        """
        if target_workers < self.min_workers or target_workers > self.max_workers:
            logger.warning(f"Número de workers fuera de rango: {target_workers}")
            return False
        
        old_workers = self.current_workers
        self.current_workers = target_workers
        
        logger.info(f"Escalado aplicado: {old_workers} -> {target_workers} workers")
        
        return True





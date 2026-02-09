"""
Continuous Burnout Processing Service
======================================
Servicio que procesa evaluaciones de burnout de forma continua.
"""

from collections import deque
from typing import Optional, Dict, Any
from ..core.types import JSONDict

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .continuous_processor import ContinuousProcessor
from .burnout_service import BurnoutPreventionService
from ..schemas import BurnoutAssessmentRequest
from ..core.validators import validate_assessment_data
from ..core.constants import MAX_PENDING_ASSESSMENTS
from ..core.exceptions import ProcessingError

# Singleton instance
_continuous_service_instance: Optional['ContinuousBurnoutService'] = None


class ContinuousBurnoutService:
    """
    Servicio de procesamiento continuo para burnout prevention.
    
    Ejecuta evaluaciones automáticas de forma continua hasta que se detiene.
    """
    
    def __init__(
        self,
        burnout_service: BurnoutPreventionService,
        interval_seconds: float = 10.0
    ):
        """
        Inicializar servicio continuo.
        
        Args:
            burnout_service: Servicio de burnout prevention
            interval_seconds: Intervalo entre evaluaciones automáticas
        """
        self.burnout_service = burnout_service
        self.interval_seconds = interval_seconds
        
        # Procesador continuo
        self.processor = ContinuousProcessor(
            process_function=self._process_cycle,
            interval_seconds=interval_seconds,
            name="ContinuousBurnoutProcessor"
        )
        
        # Datos de procesamiento (deque para O(1) popleft)
        self._pending_assessments: deque = deque()
        self._processed_count = 0
    
    async def _process_cycle(self) -> None:
        """
        Ciclo de procesamiento continuo (optimized).
        
        Processes up to BATCH_SIZE assessments per cycle for better throughput
        when there are many pending assessments.
        """
        if not self._pending_assessments:
            return
        
        # Process multiple assessments per cycle if queue is large (optimized)
        from ..core.constants import BATCH_SIZE
        batch_size = min(BATCH_SIZE, len(self._pending_assessments))
        processed_this_cycle = 0
        errors_this_cycle = 0
        
        for _ in range(batch_size):
            if not self._pending_assessments:
                break
            
            # Use popleft for O(1) instead of pop(0) which is O(n)
            assessment_data = self._pending_assessments.popleft()
            try:
                request = BurnoutAssessmentRequest(**assessment_data)
                result = await self.burnout_service.assess_burnout(request)
                self._processed_count += 1
                processed_this_cycle += 1
                
                # Log only first item in batch to reduce log noise
                if processed_this_cycle == 1:
                    logger.info(
                        "Assessment processed",
                        count=self._processed_count,
                        risk_level=result.burnout_risk_level,
                        score=result.burnout_score,
                        pending=len(self._pending_assessments),
                        batch_size=batch_size
                    )
            except ValueError as e:
                # Validation errors - log but don't increment error count
                from ..core.logging_helpers import log_warning, truncate_error_message
                error_msg = truncate_error_message(e)
                errors_this_cycle += 1
                log_warning(
                    "Invalid assessment data",
                    context={
                        "error": error_msg,
                        "pending": len(self._pending_assessments),
                        "processed": self._processed_count
                    }
                )
            except Exception as e:
                from ..core.logging_helpers import log_error
                errors_this_cycle += 1
                log_error(
                    "Error processing assessment",
                    e,
                    context={"pending": len(self._pending_assessments), "processed": self._processed_count}
                )
        
        # Log batch summary if multiple items processed
        if batch_size > 1:
            logger.debug(
                "Batch processing completed",
                processed=processed_this_cycle,
                errors=errors_this_cycle,
                pending=len(self._pending_assessments)
            )
    
    def add_assessment_request(self, assessment_data: JSONDict) -> None:
        """
        Agregar evaluación a la cola de procesamiento.
        
        Args:
            assessment_data: Datos de evaluación a procesar
            
        Raises:
            ProcessingError si la cola está llena
        """
        from ..core.validators import sanitize_assessment_data
        
        # Check queue limit
        if len(self._pending_assessments) >= MAX_PENDING_ASSESSMENTS:
            raise ProcessingError(
                f"Queue is full. Maximum {MAX_PENDING_ASSESSMENTS} pending assessments allowed."
            )
        
        # Sanitize before validation
        sanitized_data = sanitize_assessment_data(assessment_data)
        validate_assessment_data(sanitized_data)
        self._pending_assessments.append(sanitized_data)
        logger.info("Assessment added to queue", pending=len(self._pending_assessments))
    
    async def start(self, interval_seconds: Optional[float] = None) -> None:
        """Iniciar procesamiento continuo."""
        if interval_seconds:
            self.processor.update_interval(interval_seconds)
        await self.processor.start()
    
    async def stop(self) -> None:
        """Detener procesamiento continuo."""
        await self.processor.stop()
    
    async def restart(self, interval_seconds: Optional[float] = None) -> None:
        """
        Reiniciar procesamiento continuo.
        
        Args:
            interval_seconds: Optional new interval to set before restarting
        """
        await self.processor.restart(new_interval=interval_seconds)
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado del procesador."""
        stats = self.processor.get_stats()
        stats.update({
            "pending_assessments": len(self._pending_assessments),
            "processed_count": self._processed_count,
        })
        return stats
    
    @property
    def is_active(self) -> bool:
        """Verificar si está activo."""
        return self.processor.is_active


def get_continuous_service() -> ContinuousBurnoutService:
    """
    Obtener instancia singleton del servicio continuo.
    
    Similar al patrón de gestión de estado en continuous-agent.
    """
    global _continuous_service_instance
    
    if _continuous_service_instance is None:
        # Crear instancia con servicio de burnout
        from .burnout_service import BurnoutPreventionService
        from ..infrastructure.openrouter import OpenRouterClient
        
        openrouter_client = OpenRouterClient()
        burnout_service = BurnoutPreventionService(openrouter_client)
        
        _continuous_service_instance = ContinuousBurnoutService(
            burnout_service=burnout_service,
            interval_seconds=10.0
        )
        logger.info("ContinuousBurnoutService instance created")
    
    return _continuous_service_instance


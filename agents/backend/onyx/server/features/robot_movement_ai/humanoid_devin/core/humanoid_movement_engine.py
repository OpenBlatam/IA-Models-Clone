"""
Humanoid Movement Engine - Professional Deep Learning Integration
==================================================================

Motor profesional de movimiento para robot humanoide con integración avanzada
de Deep Learning, Transformers y modelos de difusión para control suave y
natural de movimientos complejos.

Sigue las mejores prácticas de PyTorch, Transformers y desarrollo profesional.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
import numpy as np
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Intentar importar PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from ..drivers.humanoid_devin_driver import (
    HumanoidDevinDriver,
    HumanoidPose,
    MovementType,
    HumanoidMotionConfig,
    HumanoidState
)
from ...core.dl_models.transformer_trajectory import TransformerTrajectoryPredictor
from ...core.dl_models.diffusion_trajectory import DiffusionTrajectoryGenerator
from ...native.wrapper import (
    NativeTrajectoryOptimizerWrapper,
    validate_array,
    performance_timer
)

logger = logging.getLogger(__name__)


class MovementPriority(Enum):
    """Prioridad de movimientos."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EMERGENCY = "emergency"


@dataclass
class MovementTask:
    """Tarea de movimiento humanoide."""
    task_id: str
    movement_type: MovementType
    config: HumanoidMotionConfig
    priority: MovementPriority = MovementPriority.NORMAL
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HumanoidMovementEngine:
    """
    Motor profesional de movimiento para robot humanoide.
    
    Características:
    - Gestión de tareas de movimiento
    - Integración con modelos de Deep Learning
    - Optimización de trayectorias
    - Control de prioridades
    - Manejo robusto de errores
    - Logging estructurado
    """
    
    def __init__(
        self,
        driver: HumanoidDevinDriver,
        use_ml: bool = True,
        use_diffusion: bool = False
    ):
        """
        Inicializar motor de movimiento humanoide profesional.
        
        Args:
            driver: Driver del robot humanoide
            use_ml: Si usar modelos de ML
            use_diffusion: Si usar modelos de difusión
        """
        if driver is None:
            raise ValueError("driver cannot be None")
        
        self.driver = driver
        self.use_ml = use_ml
        self.use_diffusion = use_diffusion
        
        # Estado del motor
        self.is_moving = False
        self.current_action: Optional[MovementTask] = None
        self.movement_history: deque = deque(maxlen=1000)
        self.task_queue: deque = deque()
        
        # Modelos de ML (si están disponibles)
        self.trajectory_predictor = None
        self.diffusion_generator = None
        self.trajectory_optimizer = None
        
        if use_ml and TORCH_AVAILABLE:
            self._initialize_ml_components()
        
        # Estadísticas
        self.total_movements = 0
        self.successful_movements = 0
        self.failed_movements = 0
        
        logger.info(
            f"✅ Humanoid Movement Engine initialized: "
            f"ML={use_ml}, Diffusion={use_diffusion}"
        )
    
    def _initialize_ml_components(self):
        """Inicializar componentes de ML."""
        try:
            # Obtener modelos del driver si están disponibles
            if hasattr(self.driver, 'trajectory_predictor') and self.driver.trajectory_predictor:
                self.trajectory_predictor = self.driver.trajectory_predictor
                logger.info("✅ Using driver's trajectory predictor")
            
            if hasattr(self.driver, 'diffusion_generator') and self.driver.diffusion_generator:
                self.diffusion_generator = self.driver.diffusion_generator
                logger.info("✅ Using driver's diffusion generator")
            
            # Optimizador de trayectorias
            self.trajectory_optimizer = NativeTrajectoryOptimizerWrapper(
                energy_weight=0.3,
                time_weight=0.3,
                smoothness_weight=0.2
            )
            logger.info("✅ Trajectory optimizer initialized")
            
        except Exception as e:
            logger.warning(f"⚠️  Error initializing ML components: {e}")
            self.trajectory_predictor = None
            self.diffusion_generator = None
            self.trajectory_optimizer = None
    
    async def initialize(self):
        """
        Inicializar motor de movimiento.
        
        Raises:
            RuntimeError: Si no se puede conectar al robot
        """
        if not self.driver.connected:
            logger.info("🔌 Connecting to humanoid robot...")
            success = await self.driver.connect()
            if not success:
                raise RuntimeError("Failed to connect to humanoid robot")
        
        logger.info("✅ Humanoid Movement Engine initialized and ready")
    
    async def shutdown(self):
        """Apagar motor de movimiento de forma segura."""
        logger.info("🛑 Shutting down Humanoid Movement Engine...")
        
        try:
            # Detener movimiento actual
            if self.is_moving:
                await self.driver.stop()
            
            # Limpiar cola de tareas
            self.task_queue.clear()
            
            # Desconectar driver
            if self.driver.connected:
                await self.driver.disconnect()
            
            logger.info("✅ Humanoid Movement Engine shut down successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}", exc_info=True)
            raise
    
    async def execute_movement(
        self,
        movement_type: MovementType,
        config: Optional[HumanoidMotionConfig] = None,
        priority: MovementPriority = MovementPriority.NORMAL
    ) -> str:
        """
        Ejecutar movimiento humanoide con gestión de tareas.
        
        Args:
            movement_type: Tipo de movimiento
            config: Configuración del movimiento
            priority: Prioridad del movimiento
            
        Returns:
            ID de la tarea creada
        """
        import uuid
        
        # Crear configuración si no se proporciona
        if config is None:
            config = HumanoidMotionConfig(
                movement_type=movement_type,
                use_ml=self.use_ml,
                use_diffusion=self.use_diffusion
            )
        
        # Crear tarea
        task = MovementTask(
            task_id=str(uuid.uuid4()),
            movement_type=movement_type,
            config=config,
            priority=priority
        )
        
        # Agregar a cola según prioridad
        if priority == MovementPriority.EMERGENCY:
            self.task_queue.appendleft(task)
        elif priority == MovementPriority.HIGH:
            # Insertar después de emergencias pero antes de normales
            insert_idx = sum(1 for t in self.task_queue if t.priority == MovementPriority.EMERGENCY)
            if insert_idx == 0:
                self.task_queue.appendleft(task)
            else:
                temp_list = list(self.task_queue)
                temp_list.insert(insert_idx, task)
                self.task_queue = deque(temp_list)
        else:
            self.task_queue.append(task)
        
        logger.info(
            f"📋 Movement task created: {movement_type.value} "
            f"(priority={priority.value}, task_id={task.task_id[:8]})"
        )
        
        # Ejecutar si no hay movimiento en curso
        if not self.is_moving:
            asyncio.create_task(self._process_task_queue())
        
        return task.task_id
    
    async def _process_task_queue(self):
        """Procesar cola de tareas de movimiento."""
        while self.task_queue:
            task = self.task_queue.popleft()
            
            try:
                await self._execute_task(task)
            except Exception as e:
                logger.error(f"❌ Error executing task {task.task_id}: {e}", exc_info=True)
                task.status = "failed"
                task.error = str(e)
                self.failed_movements += 1
            
            self.movement_history.append(task)
    
    async def _execute_task(self, task: MovementTask):
        """Ejecutar tarea de movimiento."""
        task.status = "running"
        task.started_at = datetime.now().isoformat()
        self.is_moving = True
        self.current_action = task
        
        logger.info(f"🚀 Executing movement: {task.movement_type.value}")
        
        try:
            with performance_timer(f"Movement {task.movement_type.value}"):
                # Ejecutar movimiento según tipo
                if task.movement_type == MovementType.WALK:
                    await self._execute_walk_task(task)
                elif task.movement_type == MovementType.RUN:
                    await self._execute_run_task(task)
                elif task.movement_type == MovementType.JUMP:
                    await self._execute_jump_task(task)
                elif task.movement_type == MovementType.GRASP:
                    await self._execute_grasp_task(task)
                elif task.movement_type == MovementType.RELEASE:
                    await self._execute_release_task(task)
                elif task.movement_type == MovementType.WAVE:
                    await self._execute_wave_task(task)
                else:
                    raise ValueError(f"Unknown movement type: {task.movement_type}")
            
            task.status = "completed"
            task.completed_at = datetime.now().isoformat()
            self.successful_movements += 1
            self.total_movements += 1
            
            logger.info(f"✅ Movement completed: {task.movement_type.value}")
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now().isoformat()
            self.failed_movements += 1
            self.total_movements += 1
            raise
        
        finally:
            self.is_moving = False
            self.current_action = None
    
    async def _execute_walk_task(self, task: MovementTask):
        """Ejecutar tarea de caminar."""
        config = task.config
        direction = config.metadata.get("direction", "forward")
        distance = config.metadata.get("distance", 1.0)
        speed = config.speed
        
        await self.driver.walk(
            direction=direction,
            distance=distance,
            speed=speed,
            use_diffusion=config.use_diffusion
        )
    
    async def _execute_run_task(self, task: MovementTask):
        """Ejecutar tarea de correr."""
        config = task.config
        distance = config.metadata.get("distance", 1.0)
        
        await self.driver.walk(
            direction="forward",
            distance=distance,
            speed=0.8,  # Velocidad alta para correr
            use_diffusion=config.use_diffusion
        )
    
    async def _execute_jump_task(self, task: MovementTask):
        """Ejecutar tarea de salto."""
        # Secuencia: crouch -> extend -> land
        await self.driver.crouch()
        await asyncio.sleep(0.3)
        await self.driver.stand()
        await asyncio.sleep(0.2)
    
    async def _execute_grasp_task(self, task: MovementTask):
        """Ejecutar tarea de agarre."""
        config = task.config
        hand = config.metadata.get("hand", "right")
        object_position = config.metadata.get("object_position")
        
        await self.driver.grasp(
            hand=hand,
            object_position=object_position,
            use_ml=config.use_ml
        )
    
    async def _execute_release_task(self, task: MovementTask):
        """Ejecutar tarea de soltar."""
        config = task.config
        hand = config.metadata.get("hand", "right")
        
        await self.driver.release(hand=hand)
    
    async def _execute_wave_task(self, task: MovementTask):
        """Ejecutar tarea de saludo."""
        config = task.config
        hand = config.metadata.get("hand", "right")
        
        await self.driver.wave(
            hand=hand,
            use_diffusion=config.use_diffusion
        )
    
    async def stop(self):
        """Detener movimiento actual."""
        logger.info("🛑 Stopping current movement")
        
        if self.is_moving and self.current_action:
            self.current_action.status = "cancelled"
            self.current_action.completed_at = datetime.now().isoformat()
        
        await self.driver.stop()
        self.is_moving = False
        self.current_action = None
        
        # Limpiar cola (excepto emergencias)
        emergency_tasks = deque(t for t in self.task_queue if t.priority == MovementPriority.EMERGENCY)
        self.task_queue = emergency_tasks
    
    async def emergency_stop(self):
        """Parada de emergencia."""
        logger.warning("🚨 EMERGENCY STOP")
        
        await self.driver.emergency_stop()
        self.is_moving = False
        self.current_action = None
        self.task_queue.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado del motor de movimiento.
        
        Returns:
            Dict con estado completo
        """
        status = {
            "is_moving": self.is_moving,
            "current_action": {
                "task_id": self.current_action.task_id if self.current_action else None,
                "movement_type": self.current_action.movement_type.value if self.current_action else None,
                "status": self.current_action.status if self.current_action else None
            },
            "queue_length": len(self.task_queue),
            "queue_priorities": {
                priority.value: sum(1 for t in self.task_queue if t.priority == priority)
                for priority in MovementPriority
            } if self.task_queue else {},
            "statistics": {
                "total_movements": self.total_movements,
                "successful_movements": self.successful_movements,
                "failed_movements": self.failed_movements,
                "success_rate": (
                    self.successful_movements / self.total_movements
                    if self.total_movements > 0
                    else 0.0
                )
            },
            "ml_enabled": self.use_ml,
            "diffusion_enabled": self.use_diffusion,
            "models_available": {
                "trajectory_predictor": self.trajectory_predictor is not None,
                "diffusion_generator": self.diffusion_generator is not None,
                "trajectory_optimizer": self.trajectory_optimizer is not None
            }
        }
        
        # Agregar estado del robot si está disponible (debe ser obtenido de forma async)
        status["robot_status"] = None
        
        return status
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del motor.
        
        Returns:
            Dict con estadísticas detalladas
        """
        return {
            "total_movements": self.total_movements,
            "successful_movements": self.successful_movements,
            "failed_movements": self.failed_movements,
            "success_rate": (
                self.successful_movements / self.total_movements
                if self.total_movements > 0
                else 0.0
            ),
            "is_moving": self.is_moving,
            "current_action": self.current_action.movement_type.value if self.current_action else None,
            "queue_length": len(self.task_queue),
            "history_length": len(self.movement_history),
            "movement_types": {
                mt.value: sum(
                    1 for t in self.movement_history
                    if t.movement_type == mt and t.status == "completed"
                )
                for mt in MovementType
            }
        }
    
    def get_recent_movements(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener movimientos recientes.
        
        Args:
            limit: Número máximo de movimientos a retornar
            
        Returns:
            Lista de movimientos recientes
        """
        recent = self.movement_history[-limit:]
        
        return [
            {
                "task_id": task.task_id,
                "movement_type": task.movement_type.value,
                "status": task.status,
                "priority": task.priority.value,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "error": task.error
            }
            for task in recent
        ]

"""
Robot Movement Engine
=====================

Motor principal que orquesta todos los componentes para controlar el robot.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
import numpy as np

from ...config.robot_config import RobotConfig
from ..optimization.trajectory_optimizer import TrajectoryOptimizer
from ..types.types import TrajectoryPoint
from .inverse_kinematics import InverseKinematicsSolver, EndEffectorPose, JointState
from .visual_processor import VisualProcessor, SceneAnalysis
from .real_time_feedback import RealTimeFeedbackSystem, FeedbackData
try:
    from .event_system import get_event_emitter, EventType
except ImportError:
    # Fallback if event_system doesn't exist in robot module
    try:
        from ..optimization.event_system import get_event_emitter, EventType
    except ImportError:
        def get_event_emitter():
            return None
        class EventType:
            pass

try:
    from ..native.native_integration import (
        get_native_ik_wrapper,
        get_native_collision_detector,
        get_native_motion_planner
    )
    USE_NATIVE = True
except ImportError:
    USE_NATIVE = False

try:
    from .performance import euclidean_distance_fast
    USE_PERFORMANCE_UTILS = True
except ImportError:
    USE_PERFORMANCE_UTILS = False

logger = logging.getLogger(__name__)


class RobotMovementEngine:
    """
    Motor principal de movimiento robótico.
    
    Integra:
    - Optimización de trayectorias (RL)
    - Cinemática inversa
    - Procesamiento visual
    - Feedback en tiempo real
    """
    
    def __init__(self, config: RobotConfig):
        """
        Inicializar motor de movimiento.
        
        Args:
            config: Configuración del sistema
        """
        self.config = config
        
        # Inicializar componentes
        self.trajectory_optimizer = TrajectoryOptimizer(
            model_path=config.rl_model_path
        )
        
        self.ik_solver = InverseKinematicsSolver(
            robot_type=config.robot_brand.value,
            model_path=config.ik_model_path
        )
        
        # Inicializar wrappers nativos si están disponibles
        if USE_NATIVE:
            try:
                self.native_ik = get_native_ik_wrapper()
                self.native_collision = get_native_collision_detector()
                self.native_planner = get_native_motion_planner()
                logger.info("Native libraries integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize native libraries: {e}")
                self.native_ik = None
                self.native_collision = None
                self.native_planner = None
        else:
            self.native_ik = None
            self.native_collision = None
            self.native_planner = None
        
        self.visual_processor = VisualProcessor(
            model_path=config.cnn_model_path,
            camera_resolution=config.camera_resolution
        ) if config.camera_enabled else None
        
        self.feedback_system = RealTimeFeedbackSystem(
            frequency=config.feedback_frequency,
            callback=self._on_feedback_received
        )
        
        # Estado del robot
        self.current_joint_state: Optional[JointState] = None
        self.current_trajectory: List[TrajectoryPoint] = []
        self.is_moving = False
        self.current_task: Optional[asyncio.Task] = None
        
        # Obstáculos conocidos (actualizados dinámicamente)
        self.known_obstacles: List[np.ndarray] = []
        
        # Historial de movimientos
        self.movement_history: List[Dict[str, Any]] = []
        self.max_history = 100
        
        # Estadísticas
        self.total_movements = 0
        self.successful_movements = 0
        self.failed_movements = 0
        self.total_distance_traveled = 0.0
        
        # Replanificación
        self.replanification_enabled = True
        self.replanification_threshold = 0.1  # 10cm de desviación
        
        logger.info("Robot Movement Engine initialized with advanced features")
    
    async def initialize(self):
        """Inicializar y conectar con el robot."""
        logger.info("Initializing robot connection...")
        
        # Iniciar sistema de feedback
        await self.feedback_system.start()
        
        # Obtener estado inicial
        initial_feedback = self.feedback_system.get_latest_feedback()
        if initial_feedback:
            self.current_joint_state = JointState(
                angles=initial_feedback.joint_positions,
                velocities=initial_feedback.joint_velocities
            )
        
        logger.info("Robot initialized")
    
    async def shutdown(self):
        """Apagar y desconectar el robot."""
        logger.info("Shutting down robot...")
        
        # Detener movimiento si está activo
        if self.is_moving:
            await self.stop_movement()
        
        # Detener feedback
        await self.feedback_system.stop()
        
        logger.info("Robot shut down")
    
    async def move_to_pose(
        self,
        target_pose: EndEffectorPose,
        obstacles: Optional[List[np.ndarray]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mover el robot a una pose objetivo.
        
        Args:
            target_pose: Pose objetivo
            obstacles: Lista de obstáculos
            constraints: Restricciones adicionales
            
        Returns:
            True si el movimiento fue exitoso
        
        Raises:
            RobotMovementInProgressError: Si ya hay un movimiento en progreso
            TrajectoryError: Si no se puede generar la trayectoria
            CollisionDetectedError: Si se detecta colisión
        """
        logger.info(f"Moving to pose: {target_pose.position}")
        
        # Validar que no haya movimiento en progreso
        if self.is_moving:
            from ..exceptions import RobotMovementInProgressError
            raise RobotMovementInProgressError(
                "Robot is already moving",
                error_code="MOVEMENT_IN_PROGRESS",
                details={"current_task": str(self.current_task)}
            )
        
        try:
            # 1. Resolver cinemática inversa
            ik_solutions = self.ik_solver.solve(
                target_pose,
                self.current_joint_state
            )
            
            if not ik_solutions:
                from ..exceptions import IKError
                raise IKError(
                    "No valid IK solution found",
                    error_code="NO_IK_SOLUTION",
                    details={"target_pose": target_pose.position.tolist()}
                )
            
            target_joint_state = ik_solutions[0]
            
            # 2. Obtener pose inicial
            if self.current_joint_state:
                start_pose = self.ik_solver.forward_kinematics(self.current_joint_state)
            else:
                # Usar feedback actual
                feedback = self.feedback_system.get_latest_feedback()
                if feedback:
                    start_pose = EndEffectorPose(
                        position=feedback.end_effector_position,
                        orientation=feedback.end_effector_orientation
                    )
                else:
                    logger.error("No initial pose available")
                    return False
            
            # 3. Optimizar trayectoria
            start_point = TrajectoryPoint(
                position=start_pose.position,
                orientation=start_pose.orientation,
                timestamp=0.0
            )
            
            goal_point = TrajectoryPoint(
                position=target_pose.position,
                orientation=target_pose.orientation,
                timestamp=0.0
            )
            
            trajectory = self.trajectory_optimizer.optimize_trajectory(
                start_point,
                goal_point,
                obstacles,
                constraints
            )
            
            # 4. Convertir trayectoria a comandos de articulaciones
            joint_trajectory = self._trajectory_to_joint_commands(trajectory)
            
            # 5. Ejecutar movimiento
            self.current_trajectory = trajectory
            self.is_moving = True
            
            # Emitir evento de inicio
            emitter = get_event_emitter()
            emitter.emit(
                EventType.MOVEMENT_STARTED,
                data={
                    "target_position": target_pose.position.tolist(),
                    "trajectory_length": len(trajectory)
                },
                source="movement_engine"
            )
            
            self.current_task = asyncio.create_task(
                self._execute_joint_trajectory(joint_trajectory)
            )
            
            try:
                await self.current_task
                self.is_moving = False
                
                # Emitir evento de éxito
                emitter.emit(
                    EventType.MOVEMENT_COMPLETED,
                    data={
                        "target_position": target_pose.position.tolist(),
                        "duration": self.movement_history[-1]["duration"] if self.movement_history else 0.0
                    },
                    source="movement_engine"
                )
                
                logger.info("Movement completed successfully")
                return True
            except Exception as e:
                self.is_moving = False
                
                # Emitir evento de fallo
                emitter.emit(
                    EventType.MOVEMENT_FAILED,
                    data={
                        "target_position": target_pose.position.tolist(),
                        "error": str(e)
                    },
                    source="movement_engine"
                )
                
                raise
            
        except Exception as e:
            logger.error(f"Error during movement: {e}", exc_info=True)
            self.is_moving = False
            return False
    
    def _trajectory_to_joint_commands(
        self,
        trajectory: List[TrajectoryPoint]
    ) -> List[JointState]:
        """Convertir trayectoria de poses a comandos de articulaciones."""
        if not trajectory:
            return []
        
        traj_len = len(trajectory)
        joint_commands = [None] * traj_len
        initial_angles = np.array(self.current_joint_state.angles, dtype=np.float64) if self.current_joint_state else None
        last_valid = None
        current_state = self.current_joint_state
        
        for i, point in enumerate(trajectory):
            if self.native_ik:
                try:
                    solution = self.native_ik.solve(
                        point.position,
                        point.orientation,
                        initial_angles
                    )
                    if solution is not None:
                        joint_commands[i] = JointState(angles=solution.tolist())
                        last_valid = joint_commands[i]
                        initial_angles = solution
                        current_state = joint_commands[i]
                        continue
                except Exception:
                    pass
            
            pose = EndEffectorPose(
                position=point.position,
                orientation=point.orientation
            )
            solutions = self.ik_solver.solve(pose, current_state, use_cache=True)
            if solutions:
                sol = solutions[0]
                joint_commands[i] = sol
                last_valid = sol
                current_state = sol
            elif last_valid:
                joint_commands[i] = last_valid
        
        self.current_joint_state = current_state
        return [jc for jc in joint_commands if jc is not None]
    
    async def _execute_joint_trajectory(
        self,
        joint_trajectory: List[JointState]
    ):
        """Ejecutar trayectoria de articulaciones con replanificación."""
        logger.debug(f"Executing joint trajectory with {len(joint_trajectory)} points")
        
        trajectory_index = 0
        start_time = asyncio.get_event_loop().time()
        
        trajectory_len = len(joint_trajectory)
        check_deviation_interval = max(1, trajectory_len // 10)
        
        while trajectory_index < trajectory_len and self.is_moving:
            joint_state = joint_trajectory[trajectory_index]
            
            if self.replanification_enabled and trajectory_index > 0 and trajectory_index % check_deviation_interval == 0:
                deviation = await self._check_trajectory_deviation(joint_state)
                if deviation > self.replanification_threshold:
                    if logger.isEnabledFor(logging.WARNING):
                        logger.warning(f"Trajectory deviation detected: {deviation}m, replanning...")
                    await self._replan_trajectory(trajectory_index, joint_trajectory)
                    continue
            
            await self._send_joint_command(joint_state)
            self.current_joint_state = joint_state
            
            if trajectory_index > 0 and trajectory_index % 5 == 0:
                prev_pose = self.ik_solver.forward_kinematics(joint_trajectory[trajectory_index - 1], use_cache=True)
                curr_pose = self.ik_solver.forward_kinematics(joint_state, use_cache=True)
                if USE_PERFORMANCE_UTILS:
                    self.total_distance_traveled += euclidean_distance_fast(curr_pose.position, prev_pose.position)
                else:
                    self.total_distance_traveled += np.linalg.norm(curr_pose.position - prev_pose.position)
            
            trajectory_index += 1
            
            if trajectory_index % 10 == 0:
                await asyncio.sleep(0.01)
            
            if not self.is_moving:
                break
        
        # Registrar movimiento en historial
        duration = asyncio.get_event_loop().time() - start_time
        self._record_movement(duration, trajectory_index == len(joint_trajectory))
    
    async def _check_trajectory_deviation(self, expected_state: JointState) -> float:
        """Verificar desviación de la trayectoria esperada."""
        feedback = self.feedback_system.get_latest_feedback()
        if not feedback:
            return 0.0
        
        # Calcular pose actual desde feedback
        current_state = JointState(
            angles=feedback.joint_positions,
            velocities=feedback.joint_velocities
        )
        current_pose = self.ik_solver.forward_kinematics(current_state)
        expected_pose = self.ik_solver.forward_kinematics(expected_state)
        
        # Calcular desviación
        deviation = np.linalg.norm(current_pose.position - expected_pose.position)
        return deviation
    
    async def _replan_trajectory(
        self,
        current_index: int,
        original_trajectory: List[JointState]
    ):
        """Replanificar trayectoria desde punto actual."""
        if current_index >= len(original_trajectory) - 1:
            return
        
        # Obtener punto objetivo original
        target_state = original_trajectory[-1]
        target_pose = self.ik_solver.forward_kinematics(target_state)
        
        # Obtener pose actual
        feedback = self.feedback_system.get_latest_feedback()
        if not feedback:
            return
        
        current_pose = EndEffectorPose(
            position=feedback.end_effector_position,
            orientation=feedback.end_effector_orientation
        )
        
        # Optimizar nueva trayectoria
        start_point = TrajectoryPoint(
            position=current_pose.position,
            orientation=current_pose.orientation,
            timestamp=0.0
        )
        
        goal_point = TrajectoryPoint(
            position=target_pose.position,
            orientation=target_pose.orientation,
            timestamp=0.0
        )
        
        # Usar obstáculos conocidos
        obstacles = self.known_obstacles if self.known_obstacles else None
        
        # Intentar usar planificador nativo primero
        if self.native_planner and obstacles:
            try:
                obstacles_list = [obs.tolist() for obs in obstacles]
                native_path = self.native_planner.plan(
                    start_point.position,
                    goal_point.position,
                    obstacles_list
                )
                if native_path:
                    new_trajectory = [
                        TrajectoryPoint(
                            position=np.array(p),
                            orientation=start_point.orientation,
                            timestamp=i * 0.01
                        )
                        for i, p in enumerate(native_path)
                    ]
                    logger.debug("Used native motion planner for replanning")
                else:
                    new_trajectory = self.trajectory_optimizer.optimize_trajectory(
                        start_point, goal_point, obstacles
                    )
            except Exception as e:
                logger.debug(f"Native planner failed: {e}, using Python optimizer")
                new_trajectory = self.trajectory_optimizer.optimize_trajectory(
                    start_point, goal_point, obstacles
                )
        else:
            new_trajectory = self.trajectory_optimizer.optimize_trajectory(
                start_point, goal_point, obstacles
            )
        
        # Convertir a comandos de articulaciones
        new_joint_trajectory = self._trajectory_to_joint_commands(new_trajectory)
        
        # Actualizar trayectoria actual
        self.current_trajectory = new_trajectory
        
        # Continuar ejecución con nueva trayectoria
        await self._execute_joint_trajectory(new_joint_trajectory)
    
    def _record_movement(self, duration: float, success: bool):
        """Registrar movimiento en historial."""
        self.total_movements += 1
        if success:
            self.successful_movements += 1
        else:
            self.failed_movements += 1
        
        movement_record = {
            "timestamp": asyncio.get_event_loop().time(),
            "duration": duration,
            "success": success,
            "trajectory_length": len(self.current_trajectory),
            "distance": self.total_distance_traveled
        }
        
        self.movement_history.append(movement_record)
        
        # Limitar historial
        if len(self.movement_history) > self.max_history:
            self.movement_history = self.movement_history[-self.max_history:]
    
    def update_obstacles(self, obstacles: List[np.ndarray]):
        """Actualizar lista de obstáculos conocidos."""
        self.known_obstacles = obstacles
        if self.native_collision:
            logger.debug("Obstacles updated, native collision detector ready")
        logger.info(f"Updated obstacles: {len(obstacles)} obstacles known")
    
    async def _send_joint_command(self, joint_state: JointState):
        """Enviar comando de articulaciones al robot."""
        # En producción, esto enviaría comandos al hardware real
        # o a través de ROS/drivers
        logger.debug(f"Sending joint command: {joint_state.angles}")
        
        # Simulación: solo loggear
        # En producción: robot_driver.set_joint_positions(joint_state.angles)
    
    async def stop_movement(self):
        """Detener movimiento actual."""
        logger.info("Stopping movement")
        self.is_moving = False
        
        if self.current_task:
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass
    
    def _on_feedback_received(self, feedback: FeedbackData):
        """Callback cuando se recibe feedback."""
        # Detectar anomalías
        anomalies = self.feedback_system.detect_anomalies(feedback)
        if anomalies:
            logger.warning(f"Anomalies detected: {anomalies}")
        
        # Actualizar estado actual
        self.current_joint_state = JointState(
            angles=feedback.joint_positions,
            velocities=feedback.joint_velocities
        )
        
        # Verificar límites de seguridad
        if self.config.emergency_stop_enabled:
            self._check_safety_limits(feedback)
    
    def _check_safety_limits(self, feedback: FeedbackData):
        """Verificar límites de seguridad."""
        # Verificar velocidades
        for i, vel in enumerate(feedback.joint_velocities):
            if abs(vel) > self.config.max_joint_velocity:
                logger.error(f"Joint {i} velocity exceeds limit: {vel}")
                asyncio.create_task(self.stop_movement())
        
        # Verificar posición del efector final
        ee_velocity = np.linalg.norm(feedback.end_effector_position)
        if ee_velocity > self.config.max_velocity:
            logger.error(f"End effector velocity exceeds limit: {ee_velocity}")
            asyncio.create_task(self.stop_movement())
    
    async def process_visual_feedback(
        self,
        frame: np.ndarray,
        depth_frame: Optional[np.ndarray] = None
    ) -> SceneAnalysis:
        """
        Procesar feedback visual y actualizar modelo del entorno.
        
        Args:
            frame: Frame RGB
            depth_frame: Frame de profundidad opcional
            
        Returns:
            Análisis de la escena
        """
        if not self.visual_processor:
            logger.warning("Visual processing not enabled")
            return None
        
        analysis = self.visual_processor.process_frame(frame, depth_frame)
        
        # Usar análisis para actualizar obstáculos conocidos
        # y ajustar trayectorias si es necesario
        
        return analysis
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado actual del robot."""
        feedback = self.feedback_system.get_latest_feedback()
        
        status = {
            "is_moving": self.is_moving,
            "current_joint_state": (
                self.current_joint_state.angles if self.current_joint_state else None
            ),
            "trajectory_length": len(self.current_trajectory),
            "feedback_stats": self.feedback_system.get_statistics(),
            "known_obstacles_count": len(self.known_obstacles),
            "replanification_enabled": self.replanification_enabled,
        }
        
        # Agregar información de pose actual si está disponible
        if feedback:
            status["current_position"] = feedback.end_effector_position.tolist()
            status["current_orientation"] = feedback.end_effector_orientation.tolist()
        
        return status
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del motor."""
        success_rate = (
            self.successful_movements / self.total_movements
            if self.total_movements > 0
            else 0.0
        )
        
        avg_duration = (
            np.mean([m["duration"] for m in self.movement_history])
            if self.movement_history
            else 0.0
        )
        
        return {
            "total_movements": self.total_movements,
            "successful_movements": self.successful_movements,
            "failed_movements": self.failed_movements,
            "success_rate": success_rate,
            "total_distance_traveled": self.total_distance_traveled,
            "average_movement_duration": avg_duration,
            "movement_history_size": len(self.movement_history),
            "trajectory_optimizer_stats": self.trajectory_optimizer.get_statistics(),
        }
    
    async def move_along_path(
        self,
        waypoints: List[EndEffectorPose],
        obstacles: Optional[List[np.ndarray]] = None
    ) -> bool:
        """
        Mover a lo largo de una ruta con múltiples waypoints.
        
        Args:
            waypoints: Lista de poses objetivo
            obstacles: Obstáculos conocidos
            
        Returns:
            True si todos los movimientos fueron exitosos
        """
        logger.info(f"Moving along path with {len(waypoints)} waypoints")
        
        all_success = True
        for i, waypoint in enumerate(waypoints):
            logger.debug(f"Moving to waypoint {i+1}/{len(waypoints)}")
            success = await self.move_to_pose(waypoint, obstacles)
            if not success:
                logger.error(f"Failed to reach waypoint {i+1}")
                all_success = False
                break
        
        return all_success
    
    async def follow_trajectory(
        self,
        trajectory: List[TrajectoryPoint],
        obstacles: Optional[List[np.ndarray]] = None
    ) -> bool:
        """
        Seguir una trayectoria pre-calculada.
        
        Args:
            trajectory: Trayectoria a seguir
            obstacles: Obstáculos conocidos
            
        Returns:
            True si la trayectoria se siguió exitosamente
        """
        logger.info(f"Following trajectory with {len(trajectory)} points")
        
        # Validar trayectoria
        self.trajectory_optimizer._validate_trajectory(trajectory, obstacles)
        
        # Convertir a comandos de articulaciones
        joint_trajectory = self._trajectory_to_joint_commands(trajectory)
        
        # Ejecutar
        self.current_trajectory = trajectory
        self.is_moving = True
        
        self.current_task = asyncio.create_task(
            self._execute_joint_trajectory(joint_trajectory)
        )
        
        try:
            await self.current_task
            self.is_moving = False
            logger.info("Trajectory followed successfully")
            return True
        except Exception as e:
            logger.error(f"Error following trajectory: {e}", exc_info=True)
            self.is_moving = False
            return False


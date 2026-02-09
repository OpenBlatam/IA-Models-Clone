"""
Inverse Kinematics Solver
==========================

Modelos predictivos para resolución de cinemática inversa.
Convierte posiciones y orientaciones del efector final a ángulos de articulaciones.
Optimizado con numba para máximo rendimiento.
"""

import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from functools import lru_cache

try:
    from .performance import (
        euclidean_distance_fast,
        normalize_vector_fast,
        quaternion_multiply_fast
    )
    USE_PERFORMANCE_UTILS = True
except ImportError:
    USE_PERFORMANCE_UTILS = False

try:
    from .performance import fast_dh_transform, fast_matrix_multiply
    USE_FAST_MATH = True
except ImportError:
    USE_FAST_MATH = False

logger = logging.getLogger(__name__)


@dataclass
class JointState:
    """Estado de las articulaciones del robot."""
    angles: List[float]  # Ángulos de las articulaciones en radianes
    velocities: Optional[List[float]] = None
    accelerations: Optional[List[float]] = None


@dataclass
class EndEffectorPose:
    """Pose del efector final."""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [qx, qy, qz, qw] (quaternion) o [roll, pitch, yaw]


class InverseKinematicsSolver:
    """
    Solucionador de cinemática inversa usando modelos predictivos.
    
    Características:
    - Resolución rápida usando modelos ML
    - Múltiples soluciones posibles
    - Validación de límites de articulaciones
    - Optimización de configuración
    """
    
    def __init__(
        self,
        robot_type: str = "generic",
        num_joints: int = 6,
        model_path: Optional[str] = None
    ) -> None:
        """
        Inicializar solucionador de cinemática inversa.
        
        Args:
            robot_type: Tipo de robot (kuka, abb, fanuc, etc.)
            num_joints: Número de articulaciones
            model_path: Ruta al modelo ML pre-entrenado
        
        Raises:
            ConfigurationError: Si los parámetros son inválidos
        """
        from ..exceptions import ConfigurationError
        from ..error_handling import validate_range
        
        # Validar parámetros
        if num_joints <= 0 or num_joints > 20:
            raise ConfigurationError(
                f"Number of joints must be between 1 and 20, got {num_joints}",
                error_code="INVALID_NUM_JOINTS",
                details={"num_joints": num_joints}
            )
        self.robot_type = robot_type
        self.num_joints = num_joints
        self.model_path = model_path
        self.model = None
        
        # Parámetros del robot (DH parameters o configuración)
        self.dh_params = self._get_dh_parameters(robot_type)
        
        # Límites de articulaciones
        self.joint_limits = self._get_joint_limits(robot_type)
        
        # Cargar modelo si existe
        if model_path:
            self._load_model()
        
        # Cache para soluciones IK frecuentes
        self._ik_cache = {}
        self._cache_max_size = 1000
        self._fk_cache = {}
        self._fk_cache_max_size = 500
        
        logger.info(f"Inverse Kinematics Solver initialized for {robot_type} robot")
    
    def _get_dh_parameters(self, robot_type: str) -> List[Dict[str, float]]:
        """Obtener parámetros Denavit-Hartenberg según tipo de robot."""
        # Parámetros DH genéricos para robot de 6 DOF
        # En producción, estos vendrían de archivos de configuración específicos
        if robot_type == "kuka":
            return [
                {"a": 0.35, "d": 0.33, "alpha": -np.pi/2, "theta": 0},
                {"a": 0.31, "d": 0, "alpha": 0, "theta": 0},
                {"a": 0.04, "d": 0, "alpha": np.pi/2, "theta": 0},
                {"a": 0, "d": 0.29, "alpha": -np.pi/2, "theta": 0},
                {"a": 0, "d": 0, "alpha": np.pi/2, "theta": 0},
                {"a": 0, "d": 0.08, "alpha": 0, "theta": 0},
            ]
        else:
            # Parámetros genéricos
            return [
                {"a": 0.3, "d": 0.3, "alpha": -np.pi/2, "theta": 0},
                {"a": 0.3, "d": 0, "alpha": 0, "theta": 0},
                {"a": 0.05, "d": 0, "alpha": np.pi/2, "theta": 0},
                {"a": 0, "d": 0.3, "alpha": -np.pi/2, "theta": 0},
                {"a": 0, "d": 0, "alpha": np.pi/2, "theta": 0},
                {"a": 0, "d": 0.1, "alpha": 0, "theta": 0},
            ]
    
    def _get_joint_limits(self, robot_type: str) -> List[Tuple[float, float]]:
        """Obtener límites de articulaciones según tipo de robot."""
        # Límites típicos en radianes: (min, max)
        if robot_type in ["kuka", "abb", "fanuc"]:
            return [
                (-np.pi, np.pi),      # Joint 1
                (-np.pi/2, np.pi/2),  # Joint 2
                (-np.pi, np.pi),      # Joint 3
                (-np.pi, np.pi),      # Joint 4
                (-np.pi/2, np.pi/2),  # Joint 5
                (-np.pi, np.pi),      # Joint 6
            ]
        else:
            # Límites genéricos
            return [(-np.pi, np.pi)] * self.num_joints
    
    def _load_model(self):
        """Cargar modelo ML pre-entrenado."""
        try:
            logger.info(f"Loading IK model from {self.model_path}")
            # Aquí se cargaría el modelo (neural network, etc.)
            # self.model = load_ik_model(self.model_path)
            logger.info("IK model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load IK model: {e}. Using analytical IK.")
    
    def solve(
        self,
        target_pose: EndEffectorPose,
        initial_joint_state: Optional[JointState] = None,
        preferred_config: Optional[str] = None,
        use_cache: bool = True
    ) -> List[JointState]:
        """
        Resolver cinemática inversa para una pose objetivo.
        
        Args:
            target_pose: Pose objetivo del efector final
            initial_joint_state: Estado inicial de articulaciones (para optimización)
            preferred_config: Configuración preferida (lefty, righty, elbow_up, etc.)
            
        Returns:
            Lista de soluciones posibles (puede haber múltiples)
        """
        # Verificar cache primero (más rápido)
        if use_cache:
            cache_key = self._get_cache_key(target_pose, initial_joint_state)
            if cache_key in self._ik_cache:
                return self._ik_cache[cache_key]
        
        # Intentar usar modelo ML primero
        if self.model:
            solutions = self._solve_with_model(target_pose, initial_joint_state)
        else:
            # Usar método analítico o numérico
            solutions = self._solve_analytical(target_pose, initial_joint_state)
        
        # Filtrar soluciones válidas
        valid_solutions = []
        for sol in solutions:
            if self._validate_solution(sol):
                valid_solutions.append(sol)
        
        if valid_solutions:
            best_solution = self._select_best_solution(
                valid_solutions,
                initial_joint_state,
                preferred_config
            )
            result = [best_solution]
            
            # Guardar en cache
            if use_cache:
                cache_key = self._get_cache_key(target_pose, initial_joint_state)
                if len(self._ik_cache) >= self._cache_max_size:
                    # Eliminar entrada más antigua (FIFO)
                    self._ik_cache.pop(next(iter(self._ik_cache)))
                self._ik_cache[cache_key] = result
            
            return result
        
        logger.warning("No valid IK solution found")
        return []
    
    def _solve_with_model(
        self,
        target_pose: EndEffectorPose,
        initial_joint_state: Optional[JointState]
    ) -> List[JointState]:
        """Resolver usando modelo ML."""
        # Placeholder para solución con modelo
        # En producción, esto usaría el modelo cargado
        logger.debug("Using ML model for IK")
        return self._solve_analytical(target_pose, initial_joint_state)
    
    def _solve_analytical(
        self,
        target_pose: EndEffectorPose,
        initial_joint_state: Optional[JointState]
    ) -> List[JointState]:
        """
        Resolver usando método analítico (para robots de 6 DOF).
        
        Implementación simplificada - en producción usaría métodos
        específicos según tipo de robot.
        """
        logger.debug("Using analytical IK")
        
        # Convertir quaternion a matriz de rotación si es necesario
        if len(target_pose.orientation) == 4:
            R = self._quaternion_to_rotation_matrix(target_pose.orientation)
        else:
            R = self._euler_to_rotation_matrix(target_pose.orientation)
        
        # Extraer posición
        p = target_pose.position
        
        # Para robots de 6 DOF, usar método geométrico simplificado
        # Esto es una implementación placeholder
        solutions = []
        
        # Calcular primeros 3 joints (posición)
        # Joint 1: rotación alrededor de Z
        theta1 = np.arctan2(p[1], p[0])
        
        # Joint 2 y 3: resolver usando geometría del brazo
        # Simplificación: asumir configuración estándar
        if USE_PERFORMANCE_UTILS:
            r = np.sqrt(p[0]**2 + p[1]**2)
        else:
            r = np.linalg.norm(p[:2])
        z = p[2] - self.dh_params[0]["d"]
        
        # Calcular ángulos usando ley de cosenos
        # (simplificado - implementación real sería más compleja)
        L1 = self.dh_params[0]["a"]
        L2 = self.dh_params[1]["a"]
        
        # Calcular theta2 y theta3
        D = (r**2 + z**2 - L1**2 - L2**2) / (2 * L1 * L2)
        D = np.clip(D, -1.0, 1.0)
        
        theta3 = np.arccos(D)
        theta2 = np.arctan2(z, r) - np.arctan2(L2 * np.sin(theta3), L1 + L2 * np.cos(theta3))
        
        # Calcular últimos 3 joints (orientación)
        # Simplificado: usar orientación objetivo directamente
        # En producción, esto calcularía theta4, theta5, theta6 desde la matriz R
        
        # Solución 1
        solution1 = JointState(
            angles=[theta1, theta2, theta3, 0.0, 0.0, 0.0][:self.num_joints]
        )
        solutions.append(solution1)
        
        # Solución 2 (configuración alternativa)
        solution2 = JointState(
            angles=[theta1 + np.pi, -theta2, -theta3, 0.0, 0.0, 0.0][:self.num_joints]
        )
        solutions.append(solution2)
        
        return solutions
    
    def _quaternion_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convertir quaternion a matriz de rotación."""
        qx, qy, qz, qw = quat
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        return R
    
    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """Convertir ángulos de Euler a matriz de rotación."""
        roll, pitch, yaw = euler
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        return R_z @ R_y @ R_x
    
    def _validate_solution(self, solution: JointState) -> bool:
        """Validar que una solución está dentro de los límites."""
        if len(solution.angles) != self.num_joints:
            return False
        
        for i, angle in enumerate(solution.angles):
            min_angle, max_angle = self.joint_limits[i]
            if angle < min_angle or angle > max_angle:
                return False
        
        return True
    
    def _get_cache_key(
        self,
        target_pose: EndEffectorPose,
        initial_joint_state: Optional[JointState]
    ) -> str:
        """Generar clave de cache para solución IK."""
        import hashlib
        pos_str = f"{target_pose.position[0]:.4f},{target_pose.position[1]:.4f},{target_pose.position[2]:.4f}"
        orient_str = ",".join(f"{o:.4f}" for o in target_pose.orientation)
        init_str = ",".join(f"{a:.4f}" for a in initial_joint_state.angles) if initial_joint_state else ""
        key_str = f"{pos_str}|{orient_str}|{init_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _select_best_solution(
        self,
        solutions: List[JointState],
        initial_joint_state: Optional[JointState],
        preferred_config: Optional[str]
    ) -> JointState:
        """Seleccionar la mejor solución según criterios."""
        if len(solutions) == 1:
            return solutions[0]
        
        # Si hay estado inicial, preferir solución más cercana
        if initial_joint_state:
            if USE_PERFORMANCE_UTILS:
                initial_angles = np.array(initial_joint_state.angles)
                best_solution = min(
                    solutions,
                    key=lambda s: np.sum((np.array(s.angles) - initial_angles)**2)
                )
            else:
                best_solution = min(
                    solutions,
                    key=lambda s: sum((a1 - a2)**2 for a1, a2 in zip(s.angles, initial_joint_state.angles))
                )
            return best_solution
        
        # Si no, seleccionar primera solución válida
        return solutions[0]
    
    def forward_kinematics(self, joint_state: JointState, use_cache: bool = True) -> EndEffectorPose:
        """
        Calcular cinemática directa (para validación).
        
        Args:
            joint_state: Estado de las articulaciones
            use_cache: Usar cache si está disponible
            
        Returns:
            Pose del efector final
        """
        if use_cache:
            angles_tuple = tuple(joint_state.angles)
            if angles_tuple in self._fk_cache:
                return self._fk_cache[angles_tuple]
        
        # Inicializar transformación
        T = np.eye(4)
        
        # Aplicar transformaciones DH para cada articulación
        if USE_FAST_MATH:
            for dh, theta in zip(self.dh_params, joint_state.angles):
                a, d, alpha, base_theta = dh["a"], dh["d"], dh["alpha"], dh["theta"]
                theta_total = base_theta + theta
                T_i = fast_dh_transform(a, d, alpha, theta_total)
                T = fast_matrix_multiply(T, T_i)
        else:
            for i, (dh, theta) in enumerate(zip(self.dh_params, joint_state.angles)):
                a, d, alpha, base_theta = dh["a"], dh["d"], dh["alpha"], dh["theta"]
                theta_total = base_theta + theta
                cos_theta, sin_theta = np.cos(theta_total), np.sin(theta_total)
                cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
                T_i = np.array([
                    [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
                    [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
                    [0, sin_alpha, cos_alpha, d],
                    [0, 0, 0, 1]
                ])
                T = T @ T_i
        
        # Extraer posición y orientación
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        
        # Convertir matriz de rotación a quaternion
        orientation = self._rotation_matrix_to_quaternion(rotation_matrix)
        
        result = EndEffectorPose(position=position, orientation=orientation)
        
        if use_cache:
            angles_tuple = tuple(joint_state.angles)
            if len(self._fk_cache) >= self._fk_cache_max_size:
                self._fk_cache.pop(next(iter(self._fk_cache)))
            self._fk_cache[angles_tuple] = result
        
        return result
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convertir matriz de rotación a quaternion."""
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
        
        return np.array([qx, qy, qz, qw])







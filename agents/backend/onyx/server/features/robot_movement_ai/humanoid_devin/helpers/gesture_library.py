"""
Gesture Library for Humanoid Devin Robot (Optimizado)
======================================================

Biblioteca de gestos predefinidos para el robot humanoide.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class GestureLibrary:
    """
    Biblioteca de gestos predefinidos para el robot humanoide.
    
    Contiene secuencias de movimientos para gestos comunes.
    """
    
    def __init__(self, robot_driver):
        """
        Inicializar biblioteca de gestos.
        
        Args:
            robot_driver: Instancia de HumanoidDevinDriver
        """
        if robot_driver is None:
            raise ValueError("robot_driver cannot be None")
        
        self.robot = robot_driver
        logger.info("Gesture library initialized")
    
    def get_wave_gesture(
        self,
        hand: str = "right",
        repetitions: int = 3,
        amplitude: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Obtener secuencia de gesto de saludo.
        
        Args:
            hand: "left" o "right"
            repetitions: Número de repeticiones
            amplitude: Amplitud del movimiento (radianes)
            
        Returns:
            Lista de pasos de movimiento
        """
        steps = []
        
        # Posición inicial (brazo extendido)
        initial_positions = [0.0] * self.robot.dof
        
        # Índices aproximados para articulaciones del brazo
        if hand == "right":
            shoulder_pitch_idx = 9  # right_shoulder_pitch
            elbow_pitch_idx = 12    # right_elbow_pitch
        else:
            shoulder_pitch_idx = 2  # left_shoulder_pitch
            elbow_pitch_idx = 5     # left_elbow_pitch
        
        # Levantar brazo
        up_positions = initial_positions.copy()
        up_positions[shoulder_pitch_idx] = -amplitude
        up_positions[elbow_pitch_idx] = amplitude / 2
        
        steps.append({
            "type": "joint_positions",
            "positions": up_positions,
            "duration": 0.5
        })
        
        # Movimiento de saludo (repetir)
        for _ in range(repetitions):
            # Mover hacia adelante
            forward_positions = up_positions.copy()
            forward_positions[shoulder_pitch_idx] = -amplitude * 0.5
            
            steps.append({
                "type": "joint_positions",
                "positions": forward_positions,
                "duration": 0.3
            })
            
            # Mover hacia atrás
            steps.append({
                "type": "joint_positions",
                "positions": up_positions,
                "duration": 0.3
            })
        
        # Volver a posición inicial
        steps.append({
            "type": "joint_positions",
            "positions": initial_positions,
            "duration": 0.5
        })
        
        return steps
    
    def get_pointing_gesture(
        self,
        hand: str = "right",
        direction: str = "forward"
    ) -> List[Dict[str, Any]]:
        """
        Obtener secuencia de gesto de señalar.
        
        Args:
            hand: "left" o "right"
            direction: "forward", "left", "right", "up", "down"
            
        Returns:
            Lista de pasos de movimiento
        """
        steps = []
        
        # Calcular posición objetivo según dirección
        base_position = np.array([0.3, 0.0, 1.0])
        
        direction_offsets = {
            "forward": np.array([0.5, 0.0, 1.0]),
            "left": np.array([0.3, 0.3, 1.0]),
            "right": np.array([0.3, -0.3, 1.0]),
            "up": np.array([0.3, 0.0, 1.3]),
            "down": np.array([0.3, 0.0, 0.7])
        }
        
        target_position = direction_offsets.get(direction, base_position)
        orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Mover a posición de señalar
        steps.append({
            "type": "pose",
            "position": target_position.tolist(),
            "orientation": orientation.tolist(),
            "hand": hand,
            "duration": 1.0
        })
        
        # Mantener posición
        steps.append({
            "type": "wait",
            "duration": 1.0
        })
        
        return steps
    
    def get_clapping_gesture(
        self,
        repetitions: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Obtener secuencia de gesto de aplaudir.
        
        Args:
            repetitions: Número de repeticiones
            
        Returns:
            Lista de pasos de movimiento
        """
        steps = []
        
        # Posición inicial (brazos extendidos al frente)
        initial_positions = [0.0] * self.robot.dof
        left_shoulder = 2
        right_shoulder = 9
        left_elbow = 5
        right_elbow = 12
        
        # Levantar brazos
        up_positions = initial_positions.copy()
        up_positions[left_shoulder] = -0.5
        up_positions[right_shoulder] = -0.5
        up_positions[left_elbow] = 0.5
        up_positions[right_elbow] = 0.5
        
        steps.append({
            "type": "joint_positions",
            "positions": up_positions,
            "duration": 0.5
        })
        
        # Aplaudir (repetir)
        for _ in range(repetitions):
            # Juntar manos
            clap_positions = up_positions.copy()
            clap_positions[left_shoulder] = -0.3
            clap_positions[right_shoulder] = -0.3
            
            steps.append({
                "type": "joint_positions",
                "positions": clap_positions,
                "duration": 0.2
            })
            
            # Separar manos
            steps.append({
                "type": "joint_positions",
                "positions": up_positions,
                "duration": 0.2
            })
        
        # Volver a posición inicial
        steps.append({
            "type": "joint_positions",
            "positions": initial_positions,
            "duration": 0.5
        })
        
        return steps
    
    def get_bowing_gesture(
        self,
        depth: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Obtener secuencia de gesto de inclinación.
        
        Args:
            depth: Profundidad de la inclinación (radianes)
            
        Returns:
            Lista de pasos de movimiento
        """
        steps = []
        
        # Posición inicial
        initial_positions = [0.0] * self.robot.dof
        torso_pitch_idx = 17  # torso_pitch
        
        # Inclinarse hacia adelante
        bow_positions = initial_positions.copy()
        bow_positions[torso_pitch_idx] = depth
        
        steps.append({
            "type": "joint_positions",
            "positions": bow_positions,
            "duration": 1.0
        })
        
        # Mantener inclinación
        steps.append({
            "type": "wait",
            "duration": 1.0
        })
        
        # Volver a posición inicial
        steps.append({
            "type": "joint_positions",
            "positions": initial_positions,
            "duration": 1.0
        })
        
        return steps
    
    def get_thumbs_up_gesture(
        self,
        hand: str = "right"
    ) -> List[Dict[str, Any]]:
        """
        Obtener secuencia de gesto de pulgar arriba.
        
        Args:
            hand: "left" o "right"
            
        Returns:
            Lista de pasos de movimiento
        """
        steps = []
        
        # Posición del brazo (extendido hacia arriba)
        target_position = np.array([0.2, 0.0 if hand == "right" else 0.0, 1.3])
        orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Mover brazo a posición
        steps.append({
            "type": "pose",
            "position": target_position.tolist(),
            "orientation": orientation.tolist(),
            "hand": hand,
            "duration": 1.0
        })
        
        # Mantener posición
        steps.append({
            "type": "wait",
            "duration": 2.0
        })
        
        return steps
    
    def get_all_gestures(self) -> Dict[str, Callable]:
        """
        Obtener diccionario de todos los gestos disponibles.
        
        Returns:
            Dict con nombres de gestos y sus funciones
        """
        return {
            "wave": self.get_wave_gesture,
            "point": self.get_pointing_gesture,
            "clap": self.get_clapping_gesture,
            "bow": self.get_bowing_gesture,
            "thumbs_up": self.get_thumbs_up_gesture
        }


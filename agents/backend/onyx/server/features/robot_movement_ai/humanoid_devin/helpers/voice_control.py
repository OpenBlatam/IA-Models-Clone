"""
Voice Control System for Humanoid Devin Robot (Optimizado)
==========================================================

Sistema de control por voz para el robot humanoide.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


def ErrorCode(description: str):
    """
    Decorador para anotar excepciones con códigos de error y descripciones.
    
    Args:
        description: Descripción del error que se usará en el constructor.
    
    Usage:
        @ErrorCode(description="Invalid input provided")
        class MyException(Exception):
            def __init__(self):
                super().__init__(description)
    """
    def decorator(cls):
        # Almacenar la descripción en la clase
        cls._error_description = description
        return cls
    return decorator


class VoiceCommandType(str, Enum):
    """Tipos de comandos de voz."""
    MOVEMENT = "movement"
    POSTURE = "posture"
    GESTURE = "gesture"
    MANIPULATION = "manipulation"
    CONTROL = "control"
    QUERY = "query"


@ErrorCode(description="Error in voice control system")
class VoiceControlError(Exception):
    """Excepción para errores de control por voz."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in voice control system")
        super().__init__(message)
        self.message = message


class VoiceControlSystem:
    """
    Sistema de control por voz para el robot humanoide.
    
    Procesa comandos de voz y los convierte en acciones del robot.
    """
    
    def __init__(
        self,
        language: str = "es",
        enable_nlp: bool = True,
        confidence_threshold: float = 0.7
    ):
        """
        Inicializar sistema de control por voz.
        
        Args:
            language: Idioma (es, en, etc.)
            enable_nlp: Habilitar procesamiento de lenguaje natural
            confidence_threshold: Umbral de confianza (0-1)
        """
        if not isinstance(language, str) or not language:
            raise ValueError("language must be a non-empty string")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        self.language = language
        self.enable_nlp = enable_nlp
        self.confidence_threshold = confidence_threshold
        
        # Patrones de comandos
        self.command_patterns: Dict[str, List[str]] = {
            "walk_forward": ["caminar adelante", "avanzar", "ir adelante", "walk forward"],
            "walk_backward": ["caminar atrás", "retroceder", "ir atrás", "walk backward"],
            "turn_left": ["girar izquierda", "izquierda", "turn left"],
            "turn_right": ["girar derecha", "derecha", "turn right"],
            "stand": ["pararse", "levantarse", "stand up", "stand"],
            "sit": ["sentarse", "sit down", "sit"],
            "wave": ["saludar", "ondear", "wave"],
            "grasp": ["agarrar", "tomar", "grasp", "pick up"],
            "release": ["soltar", "liberar", "release", "drop"],
            "stop": ["detener", "parar", "stop", "halt"]
        }
        
        # Callbacks para comandos
        self.command_handlers: Dict[str, Callable] = {}
        
        # Historial de comandos
        self.command_history: List[Dict[str, Any]] = []
        
        # Estadísticas
        self.total_commands = 0
        self.successful_commands = 0
        
        logger.info(
            f"Voice control system initialized: "
            f"language={language}, nlp={enable_nlp}, threshold={confidence_threshold}"
        )
    
    def register_command_handler(
        self,
        command_type: str,
        handler: Callable
    ) -> None:
        """
        Registrar manejador de comando.
        
        Args:
            command_type: Tipo de comando
            handler: Función manejadora
        """
        if not command_type or not isinstance(command_type, str):
            raise ValueError("command_type must be a non-empty string")
        if not callable(handler):
            raise ValueError("handler must be callable")
        
        self.command_handlers[command_type] = handler
        logger.info(f"Command handler registered for: {command_type}")
    
    def process_voice_command(
        self,
        audio_text: str,
        confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Procesar comando de voz.
        
        Args:
            audio_text: Texto transcrito del audio
            confidence: Confianza de la transcripción (opcional)
            
        Returns:
            Dict con comando procesado
        """
        if not audio_text or not isinstance(audio_text, str):
            raise ValueError("audio_text must be a non-empty string")
        
        # Verificar confianza
        if confidence is not None and confidence < self.confidence_threshold:
            return {
                "success": False,
                "error": "low_confidence",
                "confidence": confidence,
                "threshold": self.confidence_threshold
            }
        
        # Normalizar texto
        normalized_text = audio_text.lower().strip()
        
        # Buscar patrón de comando
        matched_command = None
        matched_pattern = None
        
        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                if pattern.lower() in normalized_text:
                    matched_command = command_type
                    matched_pattern = pattern
                    break
            if matched_command:
                break
        
        if not matched_command:
            return {
                "success": False,
                "error": "command_not_recognized",
                "text": audio_text
            }
        
        # Extraer parámetros si es posible
        parameters = self._extract_parameters(normalized_text, matched_command)
        
        command = {
            "success": True,
            "command_type": matched_command,
            "pattern": matched_pattern,
            "original_text": audio_text,
            "normalized_text": normalized_text,
            "parameters": parameters,
            "confidence": confidence or 1.0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Registrar en historial
        self.command_history.append(command)
        self.total_commands += 1
        
        return command
    
    def _extract_parameters(
        self,
        text: str,
        command_type: str
    ) -> Dict[str, Any]:
        """
        Extraer parámetros del texto.
        
        Args:
            text: Texto normalizado
            command_type: Tipo de comando
            
        Returns:
            Dict con parámetros extraídos
        """
        parameters = {}
        
        # Extraer distancia
        import re
        distance_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:metro|meter|m)', text)
        if distance_match:
            parameters["distance"] = float(distance_match.group(1))
        
        # Extraer ángulo
        angle_match = re.search(r'(\d+)\s*(?:grado|degree|deg)', text)
        if angle_match:
            parameters["angle"] = float(angle_match.group(1))
        
        # Extraer dirección
        if "izquierda" in text or "left" in text:
            parameters["direction"] = "left"
        elif "derecha" in text or "right" in text:
            parameters["direction"] = "right"
        elif "adelante" in text or "forward" in text:
            parameters["direction"] = "forward"
        elif "atrás" in text or "backward" in text:
            parameters["direction"] = "backward"
        
        # Extraer mano
        if "mano derecha" in text or "right hand" in text:
            parameters["hand"] = "right"
        elif "mano izquierda" in text or "left hand" in text:
            parameters["hand"] = "left"
        
        return parameters
    
    async def execute_command(
        self,
        command: Dict[str, Any],
        robot_driver: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Ejecutar comando procesado.
        
        Args:
            command: Comando procesado
            robot_driver: Driver del robot (opcional)
            
        Returns:
            Dict con resultado de ejecución
        """
        if not command.get("success"):
            return {
                "success": False,
                "error": command.get("error", "unknown_error")
            }
        
        command_type = command["command_type"]
        parameters = command.get("parameters", {})
        
        # Verificar si hay manejador personalizado
        if command_type in self.command_handlers:
            try:
                result = await self.command_handlers[command_type](parameters, robot_driver)
                self.successful_commands += 1
                return {
                    "success": True,
                    "command_type": command_type,
                    "result": result
                }
            except Exception as e:
                logger.error(f"Error executing command: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }
        
        # Ejecutar comando por defecto si hay robot driver
        if robot_driver:
            try:
                result = await self._execute_default_command(
                    command_type, parameters, robot_driver
                )
                self.successful_commands += 1
                return {
                    "success": True,
                    "command_type": command_type,
                    "result": result
                }
            except Exception as e:
                logger.error(f"Error executing default command: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "success": False,
            "error": "no_handler_available"
        }
    
    async def _execute_default_command(
        self,
        command_type: str,
        parameters: Dict[str, Any],
        robot_driver: Any
    ) -> Dict[str, Any]:
        """Ejecutar comando por defecto."""
        if command_type == "walk_forward":
            distance = parameters.get("distance", 1.0)
            await robot_driver.walk(direction="forward", distance=distance)
            return {"action": "walk", "direction": "forward", "distance": distance}
        
        elif command_type == "walk_backward":
            distance = parameters.get("distance", 1.0)
            await robot_driver.walk(direction="backward", distance=distance)
            return {"action": "walk", "direction": "backward", "distance": distance}
        
        elif command_type == "turn_left":
            angle = parameters.get("angle", 90.0)
            await robot_driver.turn(angle=angle, direction="left")
            return {"action": "turn", "direction": "left", "angle": angle}
        
        elif command_type == "turn_right":
            angle = parameters.get("angle", 90.0)
            await robot_driver.turn(angle=angle, direction="right")
            return {"action": "turn", "direction": "right", "angle": angle}
        
        elif command_type == "stand":
            await robot_driver.stand()
            return {"action": "stand"}
        
        elif command_type == "sit":
            await robot_driver.sit()
            return {"action": "sit"}
        
        elif command_type == "wave":
            hand = parameters.get("hand", "right")
            await robot_driver.wave(hand=hand)
            return {"action": "wave", "hand": hand}
        
        elif command_type == "grasp":
            hand = parameters.get("hand", "right")
            await robot_driver.grasp(hand=hand)
            return {"action": "grasp", "hand": hand}
        
        elif command_type == "release":
            hand = parameters.get("hand", "right")
            await robot_driver.release(hand=hand)
            return {"action": "release", "hand": hand}
        
        elif command_type == "stop":
            await robot_driver.stop()
            return {"action": "stop"}
        
        else:
            raise VoiceControlError(f"Unknown command type: {command_type}")
    
    def add_command_pattern(
        self,
        command_type: str,
        patterns: List[str]
    ) -> None:
        """
        Agregar patrón de comando.
        
        Args:
            command_type: Tipo de comando
            patterns: Lista de patrones de texto
        """
        if not command_type or not isinstance(command_type, str):
            raise ValueError("command_type must be a non-empty string")
        if not isinstance(patterns, list):
            raise ValueError("patterns must be a list")
        
        if command_type not in self.command_patterns:
            self.command_patterns[command_type] = []
        
        self.command_patterns[command_type].extend(patterns)
        logger.info(f"Command patterns added for: {command_type}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del sistema de voz.
        
        Returns:
            Dict con estadísticas
        """
        success_rate = (
            self.successful_commands / self.total_commands
            if self.total_commands > 0 else 0.0
        )
        
        return {
            "total_commands": self.total_commands,
            "successful_commands": self.successful_commands,
            "success_rate": success_rate,
            "language": self.language,
            "nlp_enabled": self.enable_nlp,
            "confidence_threshold": self.confidence_threshold,
            "command_types": list(self.command_patterns.keys()),
            "recent_commands": len([
                c for c in self.command_history
                if (datetime.now() - datetime.fromisoformat(c["timestamp"])).total_seconds() < 3600
            ])
        }
    
    def clear_history(self) -> None:
        """Limpiar historial de comandos."""
        self.command_history.clear()
        logger.info("Command history cleared")


# Importar datetime aquí para evitar error circular
from datetime import datetime


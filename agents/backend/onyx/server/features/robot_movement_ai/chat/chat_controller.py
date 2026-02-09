"""
Chat Robot Controller
=====================

Control de robots mediante chat tipo Tesla Prime.
Permite controlar robots usando lenguaje natural.
"""

import logging
import re
import asyncio
import json
from typing import Optional, Dict, Any, List
from collections import deque
import numpy as np

from ..core.robot.movement_engine import RobotMovementEngine
from ..core.error_handling import safe_execute_async, handle_errors_async
from ..core.async_utils import run_with_timeout, retry_async
try:
    from ..core.robot.inverse_kinematics import EndEffectorPose
except ImportError:
    from ..core.inverse_kinematics import EndEffectorPose
from ..config.robot_config import RobotConfig

logger = logging.getLogger(__name__)

# Intentar importar clientes LLM
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ChatRobotController:
    """
    Controlador de robots mediante chat.
    
    Interpreta comandos en lenguaje natural y los convierte
    en acciones del robot.
    """
    
    def __init__(
        self,
        movement_engine: RobotMovementEngine,
        config: RobotConfig,
        llm_provider: Optional[str] = None
    ):
        """
        Inicializar controlador de chat.
        
        Args:
            movement_engine: Motor de movimiento
            config: Configuración
            llm_provider: Proveedor de LLM (opcional)
        """
        self.movement_engine = movement_engine
        self.config = config
        self.llm_provider = llm_provider or config.llm_provider
        
        # Patrones para reconocimiento de comandos
        self.command_patterns = self._initialize_patterns()
        
        # Historial de conversación para contexto (usando deque para mejor rendimiento)
        self.conversation_history: deque = deque(maxlen=20)
        self.max_history = 10
        
        # Caché para comandos comunes
        self.command_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_max_size = 100
        
        # Inicializar cliente LLM si está disponible
        self.llm_client = None
        if config.llm_api_key:
            self._initialize_llm_client()
        
        # Estadísticas
        self.command_count = 0
        self.successful_commands = 0
        self.llm_usage_count = 0
        self.cache_hits = 0
        
        logger.info("Chat Robot Controller initialized with enhanced LLM support")
    
    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Inicializar patrones de reconocimiento de comandos."""
        return {
            "move_to": re.compile(
                r"move\s+to\s+\(?\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)?",
                re.IGNORECASE
            ),
            "move_relative": re.compile(
                r"move\s+(?:relative|by)\s+\(?\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)?",
                re.IGNORECASE
            ),
            "pick": re.compile(r"pick\s+(?:up|object)", re.IGNORECASE),
            "place": re.compile(r"place\s+(?:down|object)", re.IGNORECASE),
            "stop": re.compile(r"stop|halt|emergency\s+stop", re.IGNORECASE),
            "home": re.compile(r"go\s+home|home\s+position", re.IGNORECASE),
            "status": re.compile(r"status|state|where\s+are\s+you", re.IGNORECASE),
        }
    
    async def process_command(self, command: str) -> Dict[str, Any]:
        """
        Procesar comando de chat.
        
        Args:
            command: Comando en lenguaje natural
            
        Returns:
            Respuesta con resultado del comando
        """
        logger.info(f"Processing command: {command}")
        
        command_lower = command.lower().strip()
        command_key = command_lower[:100]
        
        # Verificar caché primero
        if command_key in self.command_cache:
            self.cache_hits += 1
            cached_result = self.command_cache[command_key].copy()
            cached_result["cached"] = True
            return cached_result
        
        # Intentar reconocer comando directo
        result = await self._parse_direct_command(command_lower)
        if result:
            self._cache_command(command_key, result)
            return result
        
        # Si no se reconoce, usar LLM para interpretar
        if self.config.llm_api_key:
            result = await self._parse_with_llm(command)
            if result:
                self._cache_command(command_key, result)
                return result
        
        # Comando no reconocido
        error_response = {
            "success": False,
            "message": "Command not recognized. Try: 'move to (x, y, z)', 'stop', 'status'",
            "suggestions": [
                "move to (0.5, 0.3, 0.2)",
                "stop",
                "status",
                "go home"
            ]
        }
        return error_response
    
    async def _parse_direct_command(self, command: str) -> Optional[Dict[str, Any]]:
        """Intentar parsear comando directo usando patrones."""
        # Move to absolute position
        match = self.command_patterns["move_to"].search(command)
        if match:
            x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
            return await self._execute_move_to(x, y, z)
        
        # Move relative
        match = self.command_patterns["move_relative"].search(command)
        if match:
            dx, dy, dz = float(match.group(1)), float(match.group(2)), float(match.group(3))
            return await self._execute_move_relative(dx, dy, dz)
        
        # Stop
        if self.command_patterns["stop"].search(command):
            await self.movement_engine.stop_movement()
            return {
                "success": True,
                "message": "Robot stopped",
                "action": "stop"
            }
        
        # Home position
        if self.command_patterns["home"].search(command):
            return await self._execute_move_to(0.0, 0.0, 0.5)  # Posición home genérica
        
        # Status
        if self.command_patterns["status"].search(command):
            status = self.movement_engine.get_status()
            return {
                "success": True,
                "message": "Robot status retrieved",
                "status": status
            }
        
        return None
    
    def _initialize_llm_client(self):
        """Inicializar cliente LLM según proveedor."""
        try:
            if self.llm_provider == "openai" and OPENAI_AVAILABLE:
                self.llm_client = openai.AsyncOpenAI(api_key=self.config.llm_api_key)
                logger.info("OpenAI client initialized")
            elif self.llm_provider == "anthropic" and ANTHROPIC_AVAILABLE:
                self.llm_client = anthropic.AsyncAnthropic(api_key=self.config.llm_api_key)
                logger.info("Anthropic client initialized")
            else:
                logger.warning(f"LLM provider {self.llm_provider} not available or not configured")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
    
    async def _parse_with_llm(self, command: str) -> Optional[Dict[str, Any]]:
        """Usar LLM para interpretar comando complejo."""
        if not self.llm_client:
            return None
        
        self.llm_usage_count += 1
        logger.debug(f"Using LLM to parse command: {command}")
        
        try:
            # Construir prompt con contexto
            system_prompt = """You are a robot control assistant. Your job is to interpret natural language commands and convert them to robot actions.

Available commands:
- move to (x, y, z): Move to absolute position
- move relative (dx, dy, dz): Move relative to current position
- stop: Stop current movement
- go home: Move to home position
- status: Get robot status

Respond in JSON format:
{
    "action": "move_to|move_relative|stop|home|status",
    "parameters": {"x": float, "y": float, "z": float} or null,
    "confidence": float (0-1)
}"""

            user_prompt = f"Command: {command}\n\nContext: {self._get_context_summary()}"
            
            # Llamar a LLM
            if self.llm_provider == "openai":
                response = await self._call_openai(system_prompt, user_prompt)
            elif self.llm_provider == "anthropic":
                response = await self._call_anthropic(system_prompt, user_prompt)
            else:
                return None
            
            # Parsear respuesta
            if response:
                return await self._execute_llm_command(response, command)
            
        except Exception as e:
            logger.error(f"Error in LLM parsing: {e}", exc_info=True)
        
        return None
    
    async def _call_openai(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Llamar a OpenAI API."""
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
    
    async def _call_anthropic(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Llamar a Anthropic API."""
        try:
            message = await self.llm_client.messages.create(
                model=self.config.llm_model,
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            content = message.content[0].text
            return json.loads(content)
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None
    
    def _get_context_summary(self) -> str:
        """Obtener resumen del contexto actual."""
        try:
            status = self.movement_engine.get_status()
            feedback = self.movement_engine.feedback_system.get_latest_feedback()
            
            context_parts = []
            
            if feedback:
                pos = feedback.end_effector_position
                context_parts.append(f"Current position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                context_parts.append(f"Robot is {'moving' if status.get('is_moving') else 'idle'}")
            
            if self.conversation_history:
                recent = list(self.conversation_history)[-3:]
                recent_msgs = [c.get('message', '')[:50] for c in recent if isinstance(c, dict)]
                if recent_msgs:
                    context_parts.append(f"Recent commands: {recent_msgs}")
            
            return ". ".join(context_parts) if context_parts else "No context available"
        except Exception as e:
            logger.warning(f"Error getting context summary: {e}")
            return "Context unavailable"
    
    async def _execute_llm_command(self, llm_response: Dict[str, Any], original_command: str) -> Dict[str, Any]:
        """Ejecutar comando interpretado por LLM."""
        action = llm_response.get("action")
        parameters = llm_response.get("parameters", {})
        confidence = llm_response.get("confidence", 0.5)
        
        # Verificar confianza mínima
        if confidence < 0.5:
            return {
                "success": False,
                "message": f"Low confidence ({confidence:.2f}) in command interpretation. Please rephrase.",
                "original_command": original_command
            }
        
        # Ejecutar acción según tipo
        if action == "move_to":
            x = parameters.get("x", 0)
            y = parameters.get("y", 0)
            z = parameters.get("z", 0)
            return await self._execute_move_to(x, y, z)
        
        elif action == "move_relative":
            dx = parameters.get("x", 0)
            dy = parameters.get("y", 0)
            dz = parameters.get("z", 0)
            return await self._execute_move_relative(dx, dy, dz)
        
        elif action == "stop":
            await self.movement_engine.stop_movement()
            return {
                "success": True,
                "message": "Robot stopped",
                "action": "stop"
            }
        
        elif action == "home":
            return await self._execute_move_to(0.0, 0.0, 0.5)
        
        elif action == "status":
            status = self.movement_engine.get_status()
            return {
                "success": True,
                "message": "Robot status retrieved",
                "status": status
            }
        
        else:
            return {
                "success": False,
                "message": f"Unknown action: {action}",
                "original_command": original_command
            }
    
    async def _execute_move_to(
        self,
        x: float,
        y: float,
        z: float,
        orientation: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Ejecutar movimiento a posición absoluta."""
        # Validar parámetros
        if not all(isinstance(coord, (int, float)) for coord in [x, y, z]):
            return {
                "success": False,
                "message": "Invalid coordinates. Must be numbers.",
                "action": "move_to"
            }
        
        if orientation is None:
            orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        
        try:
            target_pose = EndEffectorPose(
                position=np.array([float(x), float(y), float(z)], dtype=np.float64),
                orientation=orientation
            )
            
            success = await self.movement_engine.move_to_pose(target_pose)
            
            return {
                "success": success,
                "message": f"Moving to ({x:.3f}, {y:.3f}, {z:.3f})",
                "action": "move_to",
                "target": {"x": float(x), "y": float(y), "z": float(z)}
            }
        except Exception as e:
            logger.error(f"Error executing move_to: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to move: {str(e)}",
                "action": "move_to"
            }
    
    async def _execute_move_relative(
        self,
        dx: float,
        dy: float,
        dz: float
    ) -> Dict[str, Any]:
        """Ejecutar movimiento relativo."""
        # Validar parámetros
        if not all(isinstance(delta, (int, float)) for delta in [dx, dy, dz]):
            return {
                "success": False,
                "message": "Invalid deltas. Must be numbers.",
                "action": "move_relative"
            }
        
        try:
            feedback = self.movement_engine.feedback_system.get_latest_feedback()
            
            if feedback and feedback.end_effector_position is not None:
                current_pos = feedback.end_effector_position
                target_pos = current_pos + np.array([float(dx), float(dy), float(dz)], dtype=np.float64)
                
                target_pose = EndEffectorPose(
                    position=target_pos,
                    orientation=feedback.end_effector_orientation
                )
                
                success = await self.movement_engine.move_to_pose(target_pose)
                
                return {
                    "success": success,
                    "message": f"Moving relative by ({dx:.3f}, {dy:.3f}, {dz:.3f})",
                    "action": "move_relative",
                    "delta": {"x": float(dx), "y": float(dy), "z": float(dz)}
                }
            else:
                return {
                    "success": False,
                    "message": "Cannot get current position for relative movement",
                    "action": "move_relative"
                }
        except Exception as e:
            logger.error(f"Error executing move_relative: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to move relative: {str(e)}",
                "action": "move_relative"
            }
    
    async def process_chat_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Procesar mensaje de chat con contexto.
        
        Args:
            message: Mensaje del usuario
            context: Contexto adicional de la conversación
            
        Returns:
            Respuesta del sistema
        """
        # Agregar a historial (deque maneja automáticamente el límite)
        self.conversation_history.append({
            "role": "user",
            "message": message
        })
        
        # Intentar como comando directo primero
        result = await self.process_command(message)
        
        # Si es un comando reconocido, retornar resultado
        if result.get("success") or result.get("action"):
            self.command_count += 1
            if result.get("success"):
                self.successful_commands += 1
            
            # Agregar respuesta al historial
            self.conversation_history.append({
                "role": "assistant",
                "message": result.get("message", "")
            })
            
            return result
        
        # Si no, usar LLM para respuesta conversacional
        if self.llm_client:
            conversational_response = await self._generate_conversational_response(message)
            if conversational_response:
                self.conversation_history.append({
                    "role": "assistant",
                    "message": conversational_response.get("message", "")
                })
                return conversational_response
        
        # Fallback
        return {
            "success": True,
            "message": "I understand you want to control the robot. Please use specific commands like 'move to (x, y, z)' or 'stop'.",
            "type": "conversation",
            "suggestions": [
                "move to (0.5, 0.3, 0.2)",
                "stop",
                "status"
            ]
        }
    
    async def _generate_conversational_response(self, message: str) -> Optional[Dict[str, Any]]:
        """Generar respuesta conversacional usando LLM."""
        try:
            system_prompt = """You are a helpful robot control assistant. Answer questions about the robot's capabilities and help users control it. Be friendly and concise."""
            
            user_prompt = f"User: {message}\n\nContext: {self._get_context_summary()}\n\nProvide a helpful response."
            
            if self.llm_provider == "openai":
                response = await self.llm_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )
                content = response.choices[0].message.content
            elif self.llm_provider == "anthropic":
                response = await self.llm_client.messages.create(
                    model=self.config.llm_model,
                    max_tokens=200,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                content = response.content[0].text
            else:
                return None
            
            return {
                "success": True,
                "message": content,
                "type": "conversation"
            }
        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            return None
    
    def _cache_command(self, command_key: str, result: Dict[str, Any]) -> None:
        """Agregar comando al caché."""
        if len(self.command_cache) >= self.cache_max_size:
            # Eliminar el más antiguo (FIFO)
            oldest_key = next(iter(self.command_cache))
            del self.command_cache[oldest_key]
        
        # Guardar sin el flag cached
        result_copy = {k: v for k, v in result.items() if k != "cached"}
        self.command_cache[command_key] = result_copy
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del controlador."""
        success_rate = (
            self.successful_commands / self.command_count
            if self.command_count > 0
            else 0.0
        )
        
        cache_hit_rate = (
            self.cache_hits / self.command_count
            if self.command_count > 0
            else 0.0
        )
        
        return {
            "total_commands": self.command_count,
            "successful_commands": self.successful_commands,
            "success_rate": success_rate,
            "llm_usage_count": self.llm_usage_count,
            "llm_usage_rate": (
                self.llm_usage_count / self.command_count
                if self.command_count > 0
                else 0.0
            ),
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.command_cache),
            "conversation_history_length": len(self.conversation_history)
        }


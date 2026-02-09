"""
Humanoid Chat Controller - Professional Deep Learning Integration
=================================================================

Control profesional de robot humanoide mediante chat natural
con integración avanzada de Transformers y LLMs para interpretación inteligente
de comandos complejos.

Sigue las mejores prácticas de PyTorch, Transformers y desarrollo profesional.
"""

import logging
import re
import asyncio
import json
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from functools import lru_cache
from datetime import datetime
import numpy as np

# Intentar importar Transformers para modelos locales
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        pipeline
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoModelForSequenceClassification = None
    pipeline = None

# Intentar importar sistema de LLM interno
try:
    from ...core.llm_processor import get_llm_processor, LLMTask, LLMConfig
    INTERNAL_LLM_AVAILABLE = True
except ImportError:
    INTERNAL_LLM_AVAILABLE = False

from ...chat.chat_controller import ChatRobotController
from ..drivers.humanoid_devin_driver import HumanoidDevinDriver, HumanoidPose, MovementType

logger = logging.getLogger(__name__)

# Intentar importar clientes LLM externos
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


class HumanoidChatController:
    """
    Controlador profesional de chat para robot humanoide.
    
    Características:
    - Interpretación de comandos naturales usando Transformers
    - Modelos locales para parsing rápido
    - LLMs externos para comandos complejos
    - Validación robusta de comandos
    - Caché inteligente
    - Logging estructurado
    """
    
    def __init__(
        self,
        driver: HumanoidDevinDriver,
        llm_provider: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4",
        use_local_llm: bool = True
    ):
        """
        Inicializar controlador de chat humanoide profesional.
        
        Args:
            driver: Driver del robot humanoide
            llm_provider: Proveedor de LLM ("openai" o "anthropic")
            llm_api_key: API key para LLM
            llm_model: Modelo de LLM a usar
            use_local_llm: Si usar modelos Transformers locales
        """
        # Validación de entrada
        if driver is None:
            raise ValueError("driver cannot be None")
        
        self.driver = driver
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.use_local_llm = use_local_llm
        
        # Inicializar componentes
        self.command_patterns = self._initialize_patterns()
        self.conversation_history: deque = deque(maxlen=20)
        self.command_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_max_size = 100
        
        # Estadísticas
        self.command_count = 0
        self.successful_commands = 0
        self.llm_usage_count = 0
        self.local_llm_usage_count = 0
        self.cache_hits = 0
        
        # Inicializar clientes LLM
        self.llm_client = None
        self.internal_llm = None
        self.local_model_id = None
        self.intent_classifier = None
        
        if llm_api_key:
            self._initialize_llm_client()
        
        if use_local_llm and INTERNAL_LLM_AVAILABLE:
            self._initialize_local_model()
        
        if TRANSFORMERS_AVAILABLE:
            self._initialize_intent_classifier()
        
        logger.info(
            f"✅ Humanoid Chat Controller initialized: "
            f"LLM={llm_provider}, Local={use_local_llm}, "
            f"Transformers={TRANSFORMERS_AVAILABLE}"
        )
    
    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Inicializar patrones de reconocimiento de comandos humanoides."""
        return {
            "walk_forward": re.compile(
                r"walk\s+(?:forward|ahead|straight)\s*(?:for\s+)?([\d.]+)?\s*(?:meters?|m)?",
                re.IGNORECASE
            ),
            "walk_backward": re.compile(
                r"walk\s+(?:backward|back)\s*(?:for\s+)?([\d.]+)?\s*(?:meters?|m)?",
                re.IGNORECASE
            ),
            "walk_left": re.compile(
                r"walk\s+(?:left|to\s+the\s+left)\s*(?:for\s+)?([\d.]+)?\s*(?:meters?|m)?",
                re.IGNORECASE
            ),
            "walk_right": re.compile(
                r"walk\s+(?:right|to\s+the\s+right)\s*(?:for\s+)?([\d.]+)?\s*(?:meters?|m)?",
                re.IGNORECASE
            ),
            "run": re.compile(
                r"run\s+(?:forward|ahead)\s*(?:for\s+)?([\d.]+)?\s*(?:meters?|m)?",
                re.IGNORECASE
            ),
            "turn_left": re.compile(
                r"turn\s+left\s*(?:by\s+)?([\d.]+)?\s*(?:degrees?|deg)?",
                re.IGNORECASE
            ),
            "turn_right": re.compile(
                r"turn\s+right\s*(?:by\s+)?([\d.]+)?\s*(?:degrees?|deg)?",
                re.IGNORECASE
            ),
            "stand": re.compile(r"stand\s+(?:up|straight)", re.IGNORECASE),
            "sit": re.compile(r"sit\s+(?:down)?", re.IGNORECASE),
            "crouch": re.compile(r"crouch\s+(?:down)?", re.IGNORECASE),
            "jump": re.compile(r"jump\s*(?:up|high)?", re.IGNORECASE),
            "grasp": re.compile(
                r"(?:grasp|grab|pick\s+up|take)\s+(?:with\s+)?(left|right)?\s*(?:hand)?",
                re.IGNORECASE
            ),
            "release": re.compile(
                r"(?:release|drop|let\s+go)\s+(?:with\s+)?(left|right)?\s*(?:hand)?",
                re.IGNORECASE
            ),
            "wave": re.compile(
                r"wave\s+(?:with\s+)?(left|right)?\s*(?:hand)?",
                re.IGNORECASE
            ),
            "point": re.compile(
                r"point\s+(?:with\s+)?(left|right)?\s*(?:hand|finger)?",
                re.IGNORECASE
            ),
            "move_hand": re.compile(
                r"move\s+(left|right)\s+hand\s+to\s+\(?\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)?",
                re.IGNORECASE
            ),
            "stop": re.compile(r"stop|halt|emergency\s+stop", re.IGNORECASE),
            "status": re.compile(r"status|state|how\s+are\s+you", re.IGNORECASE),
        }
    
    def _initialize_llm_client(self):
        """Inicializar cliente LLM externo."""
        try:
            if self.llm_provider == "openai" and OPENAI_AVAILABLE:
                self.llm_client = openai.AsyncOpenAI(api_key=self.llm_api_key)
                logger.info("✅ OpenAI client initialized for humanoid")
            elif self.llm_provider == "anthropic" and ANTHROPIC_AVAILABLE:
                self.llm_client = anthropic.AsyncAnthropic(api_key=self.llm_api_key)
                logger.info("✅ Anthropic client initialized for humanoid")
            else:
                logger.warning(f"⚠️  LLM provider {self.llm_provider} not available")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM client: {e}", exc_info=True)
            self.llm_client = None
    
    def _initialize_local_model(self):
        """Inicializar modelo local de Transformers."""
        if not INTERNAL_LLM_AVAILABLE:
            return
        
        try:
            self.internal_llm = get_llm_processor()
            model_name = "distilgpt2"  # Modelo pequeño y eficiente
            
            self.local_model_id = self.internal_llm.load_model(
                model_name=model_name,
                task=LLMTask.COMMAND_PARSING,
                config=LLMConfig(
                    model_id="",
                    model_name=model_name,
                    task=LLMTask.COMMAND_PARSING,
                    max_length=128,
                    temperature=0.3,
                    device="auto"
                )
            )
            logger.info(f"✅ Local LLM model {model_name} loaded for humanoid commands")
        except Exception as e:
            logger.warning(f"⚠️  Could not load local LLM model: {e}")
            self.local_model_id = None
    
    def _initialize_intent_classifier(self):
        """Inicializar clasificador de intenciones."""
        if not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.intent_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.intent_model.eval()
            
            if torch and torch.cuda.is_available():
                self.intent_model = self.intent_model.cuda()
            
            logger.info("✅ Intent classifier initialized for humanoid")
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize intent classifier: {e}")
            self.intent_classifier = None
    
    async def process_command(self, command: str) -> Dict[str, Any]:
        """
        Procesar comando de chat para robot humanoide con validación.
        
        Args:
            command: Comando en lenguaje natural
            
        Returns:
            Respuesta con resultado del comando
        """
        if not command or not isinstance(command, str):
            return {
                "success": False,
                "message": "Invalid command: must be a non-empty string",
                "error": "invalid_input"
            }
        
        logger.info(f"📝 Processing humanoid command: {command[:100]}")
        
        command_lower = command.lower().strip()
        command_key = command_lower[:100]
        
        # Verificar caché
        if command_key in self.command_cache:
            self.cache_hits += 1
            cached_result = self.command_cache[command_key].copy()
            cached_result["cached"] = True
            logger.debug(f"✅ Cache hit for command: {command_key[:50]}")
            return cached_result
        
        # Intentar parsear comando directo
        result = await self._parse_direct_command(command_lower)
        if result:
            self._cache_command(command_key, result)
            return result
        
        # Intentar con modelo local (rápido y económico)
        if self.local_model_id and self.internal_llm:
            result = await self._parse_with_local_llm(command)
            if result:
                self.local_llm_usage_count += 1
                self._cache_command(command_key, result)
                return result
        
        # Usar LLM externo para comandos complejos
        if self.llm_client:
            result = await self._parse_with_llm(command)
            if result:
                self.llm_usage_count += 1
                self._cache_command(command_key, result)
                return result
        
        # Comando no reconocido
        return {
            "success": False,
            "message": "Command not recognized. Try: 'walk forward 2 meters', 'stand up', 'grasp with right hand', 'wave'",
            "suggestions": [
                "walk forward 2 meters",
                "stand up",
                "grasp with right hand",
                "wave with left hand",
                "sit down",
                "status"
            ],
            "error": "command_not_recognized"
        }
    
    async def _parse_direct_command(self, command: str) -> Optional[Dict[str, Any]]:
        """Parsear comando directo usando patrones regex."""
        # Walk commands
        for direction in ["forward", "backward", "left", "right"]:
            pattern_key = f"walk_{direction}"
            match = self.command_patterns[pattern_key].search(command)
            if match:
                distance = float(match.group(1)) if match.group(1) else 1.0
                return await self._execute_walk(direction, distance)
        
        # Run command
        match = self.command_patterns["run"].search(command)
        if match:
            distance = float(match.group(1)) if match.group(1) else 1.0
            return await self._execute_run(distance)
        
        # Turn commands
        for direction in ["left", "right"]:
            pattern_key = f"turn_{direction}"
            match = self.command_patterns[pattern_key].search(command)
            if match:
                angle = float(match.group(1)) if match.group(1) else 90.0
                return await self._execute_turn(direction, angle)
        
        # Pose commands
        if self.command_patterns["stand"].search(command):
            success = await self.driver.stand()
            return {
                "success": success,
                "message": "Standing up",
                "action": "stand"
            }
        
        if self.command_patterns["sit"].search(command):
            success = await self.driver.sit()
            return {
                "success": success,
                "message": "Sitting down",
                "action": "sit"
            }
        
        if self.command_patterns["crouch"].search(command):
            success = await self.driver.crouch()
            return {
                "success": success,
                "message": "Crouching",
                "action": "crouch"
            }
        
        if self.command_patterns["jump"].search(command):
            return await self._execute_jump()
        
        # Manipulation commands
        match = self.command_patterns["grasp"].search(command)
        if match:
            hand = match.group(1).lower() if match.group(1) else "right"
            success = await self.driver.grasp(hand=hand, use_ml=True)
            return {
                "success": success,
                "message": f"Grasping with {hand} hand",
                "action": "grasp",
                "hand": hand
            }
        
        match = self.command_patterns["release"].search(command)
        if match:
            hand = match.group(1).lower() if match.group(1) else "right"
            success = await self.driver.release(hand=hand)
            return {
                "success": success,
                "message": f"Releasing {hand} hand",
                "action": "release",
                "hand": hand
            }
        
        match = self.command_patterns["wave"].search(command)
        if match:
            hand = match.group(1).lower() if match.group(1) else "right"
            success = await self.driver.wave(hand=hand, use_diffusion=True)
            return {
                "success": success,
                "message": f"Waving with {hand} hand",
                "action": "wave",
                "hand": hand
            }
        
        match = self.command_patterns["point"].search(command)
        if match:
            hand = match.group(1).lower() if match.group(1) else "right"
            return await self._execute_point(hand)
        
        # Move hand command
        match = self.command_patterns["move_hand"].search(command)
        if match:
            hand = match.group(1).lower()
            x, y, z = float(match.group(2)), float(match.group(3)), float(match.group(4))
            return await self._execute_move_hand(hand, x, y, z)
        
        # Control commands
        if self.command_patterns["stop"].search(command):
            await self.driver.stop()
            return {
                "success": True,
                "message": "Humanoid robot stopped",
                "action": "stop"
            }
        
        if self.command_patterns["status"].search(command):
            status = await self.driver.get_status()
            ml_info = self.driver.get_ml_model_info()
            return {
                "success": True,
                "message": "Humanoid robot status retrieved",
                "status": status,
                "ml_models": ml_info
            }
        
        return None
    
    async def _parse_with_local_llm(self, command: str) -> Optional[Dict[str, Any]]:
        """Usar modelo local de Transformers para interpretar comando."""
        if not self.local_model_id or not self.internal_llm:
            return None
        
        try:
            prompt = f"""Parse this humanoid robot command and extract intent and parameters in JSON:
Command: {command}

Respond with JSON:
{{
    "action": "walk|stand|sit|crouch|grasp|release|wave|move_hand|stop|status",
    "parameters": {{"direction": str, "distance": float, "hand": str, "x": float, "y": float, "z": float}},
    "confidence": float
}}"""
            
            response = self.internal_llm.generate_text(
                self.local_model_id,
                prompt,
                max_length=150,
                temperature=0.2
            )
            
            # Intentar parsear JSON
            response_text = response.text.strip()
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return await self._execute_llm_command(parsed, command)
                except json.JSONDecodeError:
                    pass
            
            # Fallback heurístico
            return await self._extract_command_heuristic(response_text, command)
            
        except Exception as e:
            logger.error(f"❌ Error in local LLM parsing: {e}", exc_info=True)
            return None
    
    async def _extract_command_heuristic(
        self,
        text: str,
        original_command: str
    ) -> Optional[Dict[str, Any]]:
        """Extraer comando de forma heurística."""
        text_lower = text.lower()
        
        # Detectar acciones comunes
        if "walk" in text_lower:
            coords = re.findall(r'-?\d+\.?\d*', original_command)
            distance = float(coords[0]) if coords else 1.0
            direction = "forward" if "forward" in text_lower else "backward"
            return await self._execute_walk(direction, distance)
        
        if "grasp" in text_lower or "grab" in text_lower:
            hand = "left" if "left" in text_lower else "right"
            success = await self.driver.grasp(hand=hand, use_ml=True)
            return {
                "success": success,
                "message": f"Grasping with {hand} hand",
                "action": "grasp"
            }
        
        return None
    
    async def _parse_with_llm(self, command: str) -> Optional[Dict[str, Any]]:
        """Usar LLM externo para interpretar comando complejo."""
        if not self.llm_client:
            return None
        
        self.llm_usage_count += 1
        logger.debug(f"🤖 Using LLM to parse humanoid command: {command}")
        
        try:
            system_prompt = """You are a humanoid robot control assistant. Interpret natural language commands for humanoid robot movements.

Available commands:
- walk forward/backward/left/right [distance]: Walk in direction
- run forward [distance]: Run forward
- turn left/right [angle]: Turn in place
- stand up: Stand from sitting/crouching
- sit down: Sit down
- crouch: Crouch down
- jump: Jump up
- grasp/release with [left/right] hand: Grasp or release object
- wave with [left/right] hand: Wave hand
- point with [left/right] hand: Point finger
- move [left/right] hand to (x, y, z): Move hand to position
- stop: Stop all movement
- status: Get robot status

Respond in JSON:
{
    "action": "walk|run|turn|stand|sit|crouch|jump|grasp|release|wave|point|move_hand|stop|status",
    "parameters": {...},
    "confidence": float (0-1)
}"""

            user_prompt = f"Command: {command}"
            
            if self.llm_provider == "openai":
                response = await self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                llm_response = json.loads(content)
            elif self.llm_provider == "anthropic":
                response = await self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                content = response.content[0].text
                llm_response = json.loads(content)
            else:
                return None
            
            if llm_response.get("confidence", 0) < 0.5:
                logger.warning(f"⚠️  Low confidence ({llm_response.get('confidence')}) in LLM response")
                return None
            
            return await self._execute_llm_command(llm_response, command)
            
        except Exception as e:
            logger.error(f"❌ Error in LLM parsing: {e}", exc_info=True)
            return None
    
    async def _execute_llm_command(
        self,
        llm_response: Dict[str, Any],
        original_command: str
    ) -> Dict[str, Any]:
        """Ejecutar comando interpretado por LLM."""
        action = llm_response.get("action")
        parameters = llm_response.get("parameters", {})
        confidence = llm_response.get("confidence", 0.5)
        
        if confidence < 0.5:
            return {
                "success": False,
                "message": f"Low confidence ({confidence:.2f}) in command interpretation",
                "original_command": original_command
            }
        
        try:
            if action == "walk":
                direction = parameters.get("direction", "forward")
                distance = parameters.get("distance", 1.0)
                return await self._execute_walk(direction, distance)
            
            elif action == "run":
                distance = parameters.get("distance", 1.0)
                return await self._execute_run(distance)
            
            elif action == "turn":
                direction = parameters.get("direction", "left")
                angle = parameters.get("angle", 90.0)
                return await self._execute_turn(direction, angle)
            
            elif action == "stand":
                success = await self.driver.stand()
                return {"success": success, "message": "Standing up", "action": "stand"}
            
            elif action == "sit":
                success = await self.driver.sit()
                return {"success": success, "message": "Sitting down", "action": "sit"}
            
            elif action == "crouch":
                success = await self.driver.crouch()
                return {"success": success, "message": "Crouching", "action": "crouch"}
            
            elif action == "jump":
                return await self._execute_jump()
            
            elif action == "grasp":
                hand = parameters.get("hand", "right")
                success = await self.driver.grasp(hand=hand, use_ml=True)
                return {"success": success, "message": f"Grasping with {hand} hand", "action": "grasp"}
            
            elif action == "release":
                hand = parameters.get("hand", "right")
                success = await self.driver.release(hand=hand)
                return {"success": success, "message": f"Releasing {hand} hand", "action": "release"}
            
            elif action == "wave":
                hand = parameters.get("hand", "right")
                success = await self.driver.wave(hand=hand, use_diffusion=True)
                return {"success": success, "message": f"Waving with {hand} hand", "action": "wave"}
            
            elif action == "point":
                hand = parameters.get("hand", "right")
                return await self._execute_point(hand)
            
            elif action == "move_hand":
                hand = parameters.get("hand", "right")
                x = parameters.get("x", 0)
                y = parameters.get("y", 0)
                z = parameters.get("z", 0)
                return await self._execute_move_hand(hand, x, y, z)
            
            elif action == "stop":
                await self.driver.stop()
                return {"success": True, "message": "Stopped", "action": "stop"}
            
            elif action == "status":
                status = await self.driver.get_status()
                ml_info = self.driver.get_ml_model_info()
                return {
                    "success": True,
                    "message": "Status retrieved",
                    "status": status,
                    "ml_models": ml_info
                }
            
            else:
                return {
                    "success": False,
                    "message": f"Unknown action: {action}",
                    "original_command": original_command
                }
                
        except Exception as e:
            logger.error(f"❌ Error executing LLM command: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to execute command: {str(e)}",
                "action": action,
                "original_command": original_command
            }
    
    async def _execute_walk(
        self,
        direction: str,
        distance: float
    ) -> Dict[str, Any]:
        """Ejecutar comando de caminar con validación."""
        try:
            # Validar entrada
            if direction not in ["forward", "backward", "left", "right"]:
                raise ValueError(f"Invalid direction: {direction}")
            if distance <= 0:
                raise ValueError(f"Distance must be positive, got {distance}")
            
            success = await self.driver.walk(
                direction=direction,
                distance=distance,
                use_diffusion=True  # Usar difusión para movimiento suave
            )
            
            return {
                "success": success,
                "message": f"Walking {direction} for {distance}m",
                "action": "walk",
                "direction": direction,
                "distance": distance
            }
        except Exception as e:
            logger.error(f"❌ Error executing walk: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to walk: {str(e)}",
                "action": "walk"
            }
    
    async def _execute_run(self, distance: float) -> Dict[str, Any]:
        """Ejecutar comando de correr."""
        try:
            if distance <= 0:
                raise ValueError(f"Distance must be positive, got {distance}")
            
            # Correr es similar a caminar pero más rápido
            success = await self.driver.walk(
                direction="forward",
                distance=distance,
                speed=0.8,  # Velocidad más alta
                use_diffusion=True
            )
            
            return {
                "success": success,
                "message": f"Running forward for {distance}m",
                "action": "run",
                "distance": distance
            }
        except Exception as e:
            logger.error(f"❌ Error executing run: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to run: {str(e)}",
                "action": "run"
            }
    
    async def _execute_turn(self, direction: str, angle: float) -> Dict[str, Any]:
        """Ejecutar comando de giro."""
        try:
            if direction not in ["left", "right"]:
                raise ValueError(f"Invalid turn direction: {direction}")
            if angle <= 0:
                raise ValueError(f"Angle must be positive, got {angle}")
            
            # Convertir ángulo a distancia de giro (aproximado)
            turn_distance = angle / 90.0  # Normalizar
            
            success = await self.driver.walk(
                direction=f"turn_{direction}",
                distance=turn_distance,
                use_diffusion=True
            )
            
            return {
                "success": success,
                "message": f"Turning {direction} by {angle} degrees",
                "action": "turn",
                "direction": direction,
                "angle": angle
            }
        except Exception as e:
            logger.error(f"❌ Error executing turn: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to turn: {str(e)}",
                "action": "turn"
            }
    
    async def _execute_jump(self) -> Dict[str, Any]:
        """Ejecutar comando de salto."""
        try:
            # Salto es una secuencia: crouch -> extend -> land
            await self.driver.crouch()
            await asyncio.sleep(0.5)
            
            # Extender (simulado)
            await self.driver.stand()
            
            return {
                "success": True,
                "message": "Jumping",
                "action": "jump"
            }
        except Exception as e:
            logger.error(f"❌ Error executing jump: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to jump: {str(e)}",
                "action": "jump"
            }
    
    async def _execute_point(self, hand: str) -> Dict[str, Any]:
        """Ejecutar comando de apuntar."""
        try:
            if hand not in ["left", "right"]:
                raise ValueError(f"hand must be 'left' or 'right', got '{hand}'")
            
            # Apuntar es similar a mover mano pero con gesto específico
            # (simplificado - en implementación real usaría gesto de apuntar)
            success = await self.driver.wave(hand=hand, use_diffusion=True)
            
            return {
                "success": success,
                "message": f"Pointing with {hand} hand",
                "action": "point",
                "hand": hand
            }
        except Exception as e:
            logger.error(f"❌ Error executing point: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to point: {str(e)}",
                "action": "point"
            }
    
    async def _execute_move_hand(
        self,
        hand: str,
        x: float,
        y: float,
        z: float
    ) -> Dict[str, Any]:
        """Ejecutar movimiento de mano con validación."""
        try:
            if hand not in ["left", "right"]:
                raise ValueError(f"hand must be 'left' or 'right', got '{hand}'")
            
            position = np.array([x, y, z])
            orientation = np.array([0.0, 0.0, 0.0, 1.0])
            
            success = await self.driver.move_to_pose(
                position,
                orientation,
                hand=hand,
                use_ml=True  # Usar ML para movimiento suave
            )
            
            return {
                "success": success,
                "message": f"Moving {hand} hand to ({x:.3f}, {y:.3f}, {z:.3f})",
                "action": "move_hand",
                "hand": hand,
                "target": {"x": x, "y": y, "z": z}
            }
        except Exception as e:
            logger.error(f"❌ Error executing move_hand: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to move hand: {str(e)}",
                "action": "move_hand"
            }
    
    def _cache_command(self, command_key: str, result: Dict[str, Any]) -> None:
        """Agregar comando al caché con límite de tamaño."""
        if len(self.command_cache) >= self.cache_max_size:
            oldest_key = next(iter(self.command_cache))
            del self.command_cache[oldest_key]
        
        result_copy = {k: v for k, v in result.items() if k != "cached"}
        self.command_cache[command_key] = result_copy
    
    async def process_chat_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Procesar mensaje de chat con contexto.
        
        Args:
            message: Mensaje del usuario
            context: Contexto adicional
            
        Returns:
            Respuesta del sistema
        """
        # Validar entrada
        if not message or not isinstance(message, str):
            return {
                "success": False,
                "message": "Invalid message: must be a non-empty string"
            }
        
        # Agregar a historial
        self.conversation_history.append({
            "role": "user",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Procesar comando
        result = await self.process_command(message)
        
        # Actualizar estadísticas
        self.command_count += 1
        if result.get("success"):
            self.successful_commands += 1
        
        # Agregar respuesta al historial
        self.conversation_history.append({
            "role": "assistant",
            "message": result.get("message", ""),
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del controlador.
        
        Returns:
            Dict con estadísticas completas
        """
        success_rate = (
            self.successful_commands / self.command_count
            if self.command_count > 0
            else 0.0
        )
        
        return {
            "total_commands": self.command_count,
            "successful_commands": self.successful_commands,
            "success_rate": success_rate,
            "llm_usage_count": self.llm_usage_count,
            "local_llm_usage_count": self.local_llm_usage_count,
            "llm_usage_rate": (
                self.llm_usage_count / self.command_count
                if self.command_count > 0
                else 0.0
            ),
            "local_llm_usage_rate": (
                self.local_llm_usage_count / self.command_count
                if self.command_count > 0
                else 0.0
            ),
            "cache_hits": self.cache_hits,
            "cache_hit_rate": (
                self.cache_hits / self.command_count
                if self.command_count > 0
                else 0.0
            ),
            "conversation_history_length": len(self.conversation_history),
            "local_model_available": self.local_model_id is not None,
            "intent_classifier_available": self.intent_classifier is not None,
            "external_llm_available": self.llm_client is not None
        }

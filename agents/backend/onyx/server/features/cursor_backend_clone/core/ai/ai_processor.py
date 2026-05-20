"""
AI Processor - Procesamiento inteligente con LLMs
===================================================

Procesa comandos usando LLMs para entender intención y generar código.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CommandIntent(Enum):
    """Intenciones de comandos"""
    EXECUTE = "execute"
    QUERY = "query"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ANALYZE = "analyze"
    SEARCH = "search"
    UNKNOWN = "unknown"


@dataclass
class ProcessedCommand:
    """Comando procesado por IA"""
    original: str
    intent: CommandIntent
    confidence: float
    extracted_code: Optional[str] = None
    parameters: Dict[str, Any] = None
    suggested_actions: List[str] = None
    error: Optional[str] = None


class AIProcessor:
    """Procesador de comandos con IA"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_openai: bool = False,
        use_local: bool = True,
        api_key: Optional[str] = None
    ):
        self.model_name = model_name or "gpt-3.5-turbo"
        self.use_openai = use_openai
        self.use_local = use_local
        self.api_key = api_key
        
        self._transformer_model = None
        self._embedding_model = None
        self._llm_pipeline = None
        self._initialized = False
        
    async def initialize(self):
        """Inicializar modelos de IA"""
        if self._initialized:
            return
        
        try:
            if self.use_local:
                await self._load_local_models()
            elif self.use_openai:
                await self._setup_openai()
            
            self._initialized = True
            logger.info("✅ AI Processor initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize AI models: {e}")
            logger.info("Falling back to rule-based processing")
    
    async def _load_local_models(self):
        """Cargar modelos locales"""
        try:
            # Intentar cargar LLM Pipeline
            try:
                from .llm_pipeline import LLMPipeline, LLMConfig
                
                config = LLMConfig(
                    model_name="gpt2",  # Modelo pequeño por defecto
                    device="auto"
                )
                self._llm_pipeline = LLMPipeline(config)
                await self._llm_pipeline.initialize()
                logger.info("✅ LLM Pipeline loaded")
            except Exception as e:
                logger.debug(f"Could not load LLM Pipeline: {e}")
            
            # Intentar cargar transformers para embeddings
            try:
                from transformers import pipeline, AutoTokenizer, AutoModel
                import torch
                
                # Modelo ligero para embeddings
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                logger.info(f"Loading embedding model: {model_name}")
                
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._embedding_model = AutoModel.from_pretrained(model_name)
                self._embedding_model.eval()
                
                logger.info("✅ Local embedding model loaded")
            except ImportError:
                logger.warning("transformers not available, skipping local models")
            except Exception as e:
                logger.warning(f"Could not load local models: {e}")
        except Exception as e:
            logger.warning(f"Error loading local models: {e}")
    
    async def _setup_openai(self):
        """Configurar OpenAI"""
        try:
            import openai
            if self.api_key:
                openai.api_key = self.api_key
            logger.info("✅ OpenAI configured")
        except ImportError:
            logger.warning("openai package not available")
        except Exception as e:
            logger.warning(f"Error setting up OpenAI: {e}")
    
    async def process_command(self, command: str) -> ProcessedCommand:
        """Procesar comando con IA"""
        if not self._initialized:
            await self.initialize()
        
        # Análisis básico primero
        intent = self._detect_intent(command)
        confidence = self._calculate_confidence(command, intent)
        
        # Extraer código si es posible
        extracted_code = self._extract_code(command)
        
        # Generar sugerencias
        suggested_actions = self._generate_suggestions(command, intent)
        
        # Si tenemos modelos de IA, usar procesamiento avanzado
        if self._embedding_model:
            try:
                enhanced = await self._enhance_with_embeddings(command, intent)
                if enhanced:
                    intent = enhanced.get("intent", intent)
                    confidence = enhanced.get("confidence", confidence)
            except Exception as e:
                logger.debug(f"Could not enhance with embeddings: {e}")
        
        return ProcessedCommand(
            original=command,
            intent=intent,
            confidence=confidence,
            extracted_code=extracted_code,
            parameters=self._extract_parameters(command),
            suggested_actions=suggested_actions
        )
    
    def _detect_intent(self, command: str) -> CommandIntent:
        """Detectar intención del comando"""
        command_lower = command.lower()
        
        # Palabras clave para diferentes intenciones
        if any(word in command_lower for word in ["create", "make", "new", "generate"]):
            return CommandIntent.CREATE
        elif any(word in command_lower for word in ["update", "modify", "change", "edit"]):
            return CommandIntent.UPDATE
        elif any(word in command_lower for word in ["delete", "remove", "drop"]):
            return CommandIntent.DELETE
        elif any(word in command_lower for word in ["analyze", "examine", "inspect"]):
            return CommandIntent.ANALYZE
        elif any(word in command_lower for word in ["search", "find", "look", "query"]):
            return CommandIntent.SEARCH
        elif any(word in command_lower for word in ["what", "how", "why", "tell", "show"]):
            return CommandIntent.QUERY
        else:
            return CommandIntent.EXECUTE
    
    def _calculate_confidence(self, command: str, intent: CommandIntent) -> float:
        """Calcular confianza en la detección"""
        # Confianza base basada en palabras clave
        keywords = {
            CommandIntent.CREATE: ["create", "make", "new", "generate"],
            CommandIntent.UPDATE: ["update", "modify", "change", "edit"],
            CommandIntent.DELETE: ["delete", "remove", "drop"],
            CommandIntent.ANALYZE: ["analyze", "examine", "inspect"],
            CommandIntent.SEARCH: ["search", "find", "look", "query"],
            CommandIntent.QUERY: ["what", "how", "why", "tell", "show"],
        }
        
        command_lower = command.lower()
        matches = sum(1 for word in keywords.get(intent, []) if word in command_lower)
        
        if matches > 0:
            return min(0.9, 0.5 + (matches * 0.1))
        return 0.5
    
    def _extract_code(self, command: str) -> Optional[str]:
        """Extraer código del comando"""
        # Buscar bloques de código
        if "```" in command:
            parts = command.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Partes impares son código
                    return part.strip()
        
        # Buscar código Python
        if "python" in command.lower() or "def " in command or "import " in command:
            lines = command.split("\n")
            code_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
            if code_lines:
                return "\n".join(code_lines)
        
        return None
    
    def _extract_parameters(self, command: str) -> Dict[str, Any]:
        """Extraer parámetros del comando"""
        params = {}
        
        # Buscar parámetros comunes
        if "timeout" in command.lower():
            import re
            timeout_match = re.search(r'timeout[=:]\s*(\d+)', command, re.IGNORECASE)
            if timeout_match:
                params["timeout"] = int(timeout_match.group(1))
        
        if "async" in command.lower() or "await" in command:
            params["async"] = True
        
        return params
    
    def _generate_suggestions(self, command: str, intent: CommandIntent) -> List[str]:
        """Generar sugerencias basadas en el comando"""
        suggestions = []
        
        if intent == CommandIntent.QUERY:
            suggestions.append("Consider using a search or analysis command")
        
        if "error" in command.lower() or "fail" in command.lower():
            suggestions.append("Check logs for more details")
            suggestions.append("Verify command syntax")
        
        return suggestions
    
    async def _enhance_with_embeddings(self, command: str, intent: CommandIntent) -> Optional[Dict[str, Any]]:
        """Mejorar detección usando embeddings"""
        if not self._embedding_model:
            return None
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Tokenizar y obtener embeddings
            inputs = self._tokenizer(command, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self._embedding_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Aquí podrías comparar con embeddings de ejemplos conocidos
            # Por ahora, solo retornamos None para usar el método básico
            
            return None
        except Exception as e:
            logger.debug(f"Error in embedding enhancement: {e}")
            return None
    
    async def generate_code(self, description: str, language: str = "python") -> Optional[str]:
        """Generar código a partir de una descripción"""
        if not self._initialized:
            await self.initialize()
        
        # Si tenemos LLM Pipeline, usarlo
        if self._llm_pipeline:
            try:
                prompt = f"Write {language} code for: {description}\n\nCode:\n```{language}\n"
                code = self._llm_pipeline.generate(prompt, max_new_tokens=200, temperature=0.3)
                # Extraer código del bloque de código si está presente
                if "```" in code:
                    parts = code.split("```")
                    for i, part in enumerate(parts):
                        if language in part.lower() or i % 2 == 1:
                            return part.strip()
                return code.strip()
            except Exception as e:
                logger.debug(f"LLM Pipeline generation failed: {e}")
        
        # Si tenemos OpenAI, usarlo
        if self.use_openai:
            try:
                return await self._generate_with_openai(description, language)
            except Exception as e:
                logger.warning(f"OpenAI generation failed: {e}")
        
        # Fallback: generación basada en templates
        return self._generate_with_templates(description, language)
    
    async def _generate_with_openai(self, description: str, language: str) -> Optional[str]:
        """Generar código con OpenAI"""
        try:
            import openai
            
            prompt = f"Generate {language} code for: {description}\n\nCode:"
            
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You are a helpful {language} programming assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return None
    
    def _generate_with_templates(self, description: str, language: str) -> Optional[str]:
        """Generar código usando templates básicos"""
        # Templates simples para casos comunes
        templates = {
            "python": {
                "function": lambda desc: f"def {desc.lower().replace(' ', '_')}():\n    # {desc}\n    pass",
                "class": lambda desc: f"class {desc.title().replace(' ', '')}:\n    def __init__(self):\n        pass",
            }
        }
        
        # Detectar tipo de código necesario
        if "function" in description.lower() or "def" in description.lower():
            return templates.get(language, {}).get("function", lambda x: None)(description)
        elif "class" in description.lower():
            return templates.get(language, {}).get("class", lambda x: None)(description)
        
        return None
    
    async def summarize_result(self, result: str, max_length: int = 200) -> str:
        """Resumir resultado de ejecución"""
        if len(result) <= max_length:
            return result
        
        # Si tenemos LLM Pipeline, usarlo para resumir
        if self._llm_pipeline:
            try:
                prompt = f"Summarize the following in {max_length} words or less:\n\n{result}\n\nSummary:"
                summary = self._llm_pipeline.generate(prompt, max_new_tokens=max_length, temperature=0.5)
                return summary.strip()
            except Exception as e:
                logger.debug(f"LLM summarization failed: {e}")
        
        # Resumen simple: primeras y últimas líneas
        lines = result.split("\n")
        if len(lines) <= 10:
            return result
        
        summary_lines = lines[:5] + ["...", f"[{len(lines) - 10} lines omitted]", "..."] + lines[-5:]
        return "\n".join(summary_lines)
    
    @property
    def llm_pipeline(self):
        """Acceso al pipeline LLM"""
        return self._llm_pipeline


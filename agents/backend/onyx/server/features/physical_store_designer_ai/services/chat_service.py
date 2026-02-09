"""
Chat Service - Servicio de chat interactivo para recopilar información
"""

import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from ..core.models import ChatMessage, ChatSession, StoreDesignRequest, StoreType, DesignStyle
from ..core.logging_config import get_logger
from ..core.exceptions import NotFoundError
from .llm_service import LLMService

logger = get_logger(__name__)


class ChatService:
    """Servicio para manejar conversaciones con el cliente"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.sessions: Dict[str, ChatSession] = {}
        self.llm_service = llm_service or LLMService()
    
    def create_session(self) -> str:
        """Crear nueva sesión de chat"""
        session_id = str(uuid.uuid4())
        session = ChatSession(
            session_id=session_id,
            messages=[],
            store_info={}
        )
        self.sessions[session_id] = session
        
        # Mensaje inicial del asistente
        welcome_message = ChatMessage(
            role="assistant",
            content=(
                "¡Hola! 👋 Soy tu asistente de diseño de locales físicos. "
                "Te ayudaré a crear el diseño perfecto para tu nueva tienda.\n\n"
                "Para comenzar, cuéntame:\n"
                "1. ¿Qué tipo de tienda quieres abrir?\n"
                "2. ¿Cómo te gustaría que se vea tu local?\n"
                "3. ¿Tienes alguna preferencia de estilo o decoración?\n\n"
                "¡Estoy aquí para ayudarte en cada paso!"
            )
        )
        session.messages.append(welcome_message)
        
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str) -> ChatMessage:
        """Agregar mensaje a la sesión"""
        if session_id not in self.sessions:
            raise NotFoundError("Chat session", session_id)
        
        session = self.sessions[session_id]
        message = ChatMessage(role=role, content=content)
        session.messages.append(message)
        session.updated_at = datetime.now()
        
        logger.debug(
            f"Mensaje agregado a sesión: {session_id}",
            extra={"session_id": session_id, "role": role, "message_length": len(content)}
        )
        
        return message
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Obtener sesión"""
        return self.sessions.get(session_id)
    
    async def extract_store_info(self, session_id: str) -> Dict[str, Any]:
        """Extraer información de la tienda de los mensajes usando LLM"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        # Primero intentar con extracción básica
        store_info = self._extract_basic_info(session)
        
        # Luego mejorar con LLM si está disponible
        if self.llm_service.client:
            try:
                llm_info = await self._extract_with_llm(session)
                store_info.update(llm_info)
            except Exception as e:
                logger.warning(f"Error extrayendo info con LLM: {e}")
        
        session.store_info.update(store_info)
        return session.store_info
    
    def _extract_basic_info(self, session: ChatSession) -> Dict[str, Any]:
        """Extracción básica usando keywords"""
        store_info = {}
        
        for message in session.messages:
            if message.role == "user":
                content_lower = message.content.lower()
                
                # Detectar tipo de tienda
                store_types = {
                    "restaurante": StoreType.RESTAURANT,
                    "café": StoreType.CAFE,
                    "cafe": StoreType.CAFE,
                    "boutique": StoreType.BOUTIQUE,
                    "supermercado": StoreType.SUPERMARKET,
                    "farmacia": StoreType.PHARMACY,
                    "electrónica": StoreType.ELECTRONICS,
                    "ropa": StoreType.CLOTHING,
                    "muebles": StoreType.FURNITURE,
                }
                
                for keyword, store_type in store_types.items():
                    if keyword in content_lower:
                        store_info["store_type"] = store_type.value
                        break
                
                # Detectar estilo
                styles = {
                    "moderno": DesignStyle.MODERN,
                    "clásico": DesignStyle.CLASSIC,
                    "minimalista": DesignStyle.MINIMALIST,
                    "industrial": DesignStyle.INDUSTRIAL,
                    "rústico": DesignStyle.RUSTIC,
                    "lujo": DesignStyle.LUXURY,
                    "ecológico": DesignStyle.ECO_FRIENDLY,
                    "vintage": DesignStyle.VINTAGE,
                }
                
                for keyword, style in styles.items():
                    if keyword in content_lower:
                        store_info["style_preference"] = style.value
                        break
                
                # Detectar nombre de tienda
                if "se llama" in content_lower or "nombre" in content_lower:
                    # Intentar extraer nombre
                    parts = message.content.split("llama")
                    if len(parts) > 1:
                        name = parts[1].strip().split()[0]
                        if name and len(name) > 2:
                            store_info["store_name"] = name
                
                # Detectar presupuesto
                budget_keywords = ["presupuesto", "budget", "costo", "precio"]
                if any(kw in content_lower for kw in budget_keywords):
                    if "bajo" in content_lower or "low" in content_lower:
                        store_info["budget_range"] = "bajo"
                    elif "alto" in content_lower or "high" in content_lower:
                        store_info["budget_range"] = "alto"
                    else:
                        store_info["budget_range"] = "medio"
        
        return store_info
    
    async def _extract_with_llm(self, session: ChatSession) -> Dict[str, Any]:
        """Extraer información usando LLM"""
        # Construir contexto de la conversación
        conversation = "\n".join([
            f"{msg.role}: {msg.content}" for msg in session.messages
        ])
        
        system_prompt = """Eres un asistente experto en extraer información sobre negocios.
        Extrae información relevante de la conversación y devuélvela en formato JSON.
        Campos posibles: store_name, store_type, style_preference, budget_range, location, target_audience, dimensions, additional_info.
        Si no encuentras información para un campo, no lo incluyas."""
        
        prompt = f"""De la siguiente conversación, extrae información sobre la tienda que el usuario quiere abrir:

{conversation}

Extrae solo la información que esté explícitamente mencionada o claramente inferida."""
        
        result = await self.llm_service.generate_structured(
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        return result
    
    async def generate_response(self, session_id: str, user_message: str) -> str:
        """Generar respuesta del asistente basada en el contexto"""
        if session_id not in self.sessions:
            raise NotFoundError("Chat session", session_id)
        
        self.add_message(session_id, "user", user_message)
        store_info = await self.extract_store_info(session_id)
        session = self.get_session(session_id)
        
        if not session:
            raise NotFoundError("Chat session", session_id)
        
        # Generar respuesta usando LLM si está disponible
        if self.llm_service.client:
            try:
                response = await self._generate_llm_response(session, store_info)
                logger.debug(
                    f"Respuesta generada con LLM para sesión: {session_id}",
                    extra={"session_id": session_id, "response_length": len(response)}
                )
            except Exception as e:
                logger.warning(
                    f"Error generando respuesta con LLM: {e}",
                    extra={"session_id": session_id},
                    exc_info=True
                )
                response = self._generate_contextual_response(user_message, store_info, session)
        else:
            response = self._generate_contextual_response(user_message, store_info, session)
        
        self.add_message(session_id, "assistant", response)
        return response
    
    async def _generate_llm_response(
        self,
        session: ChatSession,
        store_info: Dict[str, Any]
    ) -> str:
        """Generar respuesta usando LLM"""
        system_prompt = """Eres un asistente experto en diseño de locales físicos.
        Ayudas a los clientes a diseñar sus tiendas proporcionando información útil y haciendo preguntas relevantes.
        Sé amigable, profesional y proactivo. Haz preguntas específicas para entender mejor las necesidades del cliente."""
        
        # Construir contexto
        context = "Información recopilada hasta ahora:\n"
        if store_info:
            for key, value in store_info.items():
                context += f"- {key}: {value}\n"
        else:
            context += "- Aún no hay información recopilada\n"
        
        # Construir mensajes para el chat
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Agregar mensajes de la conversación (últimos 10 para no exceder tokens)
        recent_messages = session.messages[-10:]
        for msg in recent_messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Agregar contexto al último mensaje
        if messages:
            messages[-1]["content"] = f"{context}\n\n{messages[-1]['content']}"
        
        response = await self.llm_service.chat(messages, temperature=0.7)
        return response
    
    def _generate_contextual_response(
        self, 
        user_message: str, 
        store_info: Dict[str, Any],
        session: ChatSession
    ) -> str:
        """Generar respuesta contextual basada en la información recopilada"""
        message_lower = user_message.lower()
        user_message_count = len([m for m in session.messages if m.role == "user"])
        
        # Si es el primer mensaje del usuario
        if user_message_count == 1:
            return (
                "¡Perfecto! Veo que quieres abrir una tienda. 🏪\n\n"
                "Para darte el mejor diseño, necesito conocer más detalles:\n\n"
                "• ¿Cuál es el nombre de tu tienda?\n"
                "• ¿Qué tipo de productos o servicios ofrecerás?\n"
                "• ¿Tienes un presupuesto aproximado? (bajo, medio, alto)\n"
                "• ¿Ya tienes un local o estás buscando uno?\n"
                "• ¿Tienes alguna preferencia de estilo? (moderno, clásico, minimalista, etc.)"
            )
        
        # Detectar si el usuario está listo para generar el diseño
        ready_keywords = ["listo", "generar", "crear", "diseñar", "empezar", "ya tengo", "proceder"]
        if any(keyword in message_lower for keyword in ready_keywords):
            if not store_info.get("store_type"):
                return (
                    "Antes de generar el diseño, necesito saber:\n"
                    "• ¿Qué tipo de tienda quieres abrir? (restaurante, café, boutique, etc.)\n\n"
                    "Una vez que me digas esto, podré crear un diseño personalizado para ti."
                )
            
            # Resumir información recopilada
            summary = "Información recopilada:\n"
            if store_info.get("store_name"):
                summary += f"• Nombre: {store_info['store_name']}\n"
            if store_info.get("store_type"):
                summary += f"• Tipo: {store_info['store_type']}\n"
            if store_info.get("style_preference"):
                summary += f"• Estilo: {store_info['style_preference']}\n"
            
            return (
                f"¡Excelente! {summary}\n\n"
                "Voy a generar para ti:\n"
                "✨ Diseño visual de tu local (exterior, interior, layout)\n"
                "📊 Plan completo de marketing y ventas\n"
                "🎨 Plan detallado de decoración\n\n"
                "¿Procedo con la generación del diseño completo?"
            )
        
        # Detectar preguntas específicas
        if "presupuesto" in message_lower or "costo" in message_lower:
            return (
                "El presupuesto es importante para el diseño. ¿Cuál es tu rango aproximado?\n\n"
                "• Bajo: $5,000 - $15,000\n"
                "• Medio: $15,000 - $40,000\n"
                "• Alto: $40,000+\n\n"
                "Esto me ayudará a recomendarte muebles y decoración apropiados."
            )
        
        if "estilo" in message_lower or "decoración" in message_lower:
            return (
                "Los estilos disponibles son:\n\n"
                "• Moderno: Líneas limpias y minimalistas\n"
                "• Clásico: Elegante y tradicional\n"
                "• Minimalista: Espacios abiertos\n"
                "• Industrial: Materiales rústicos y metálicos\n"
                "• Rústico: Ambiente acogedor y natural\n"
                "• Lujo: Materiales premium\n"
                "• Ecológico: Sostenible y verde\n"
                "• Vintage: Estilo retro\n\n"
                "¿Cuál te gusta más?"
            )
        
        # Respuesta genérica que anima a seguir la conversación
        missing_info = []
        if not store_info.get("store_name"):
            missing_info.append("el nombre de tu tienda")
        if not store_info.get("store_type"):
            missing_info.append("el tipo de tienda")
        if not store_info.get("style_preference"):
            missing_info.append("tu preferencia de estilo")
        
        if missing_info:
            return (
                f"Para crear el mejor diseño, aún necesito saber sobre {', '.join(missing_info)}.\n\n"
                "¿Puedes compartir esta información?"
            )
        
        return (
            "Entiendo. ¿Hay algo más específico que te gustaría incluir en el diseño?\n\n"
            "Por ejemplo:\n"
            "• Preferencias de colores\n"
            "• Elementos decorativos especiales\n"
            "• Requisitos de accesibilidad\n"
            "• Dimensiones del local\n"
            "• Ubicación o características del espacio"
        )


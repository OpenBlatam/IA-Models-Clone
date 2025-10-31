"""
Conversation AI - Sistema de IA conversacional avanzado
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

from ..models import Language, SentimentType
from .embeddings import EmbeddingManager
from .ai_integration import AIIntegrationManager

logger = logging.getLogger(__name__)


class ConversationType(Enum):
    """Tipos de conversación."""
    CUSTOMER_SERVICE = "customer_service"
    SALES = "sales"
    TECHNICAL_SUPPORT = "technical_support"
    EDUCATIONAL = "educational"
    CASUAL = "casual"
    PROFESSIONAL = "professional"


class MessageRole(Enum):
    """Roles en la conversación."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    CONTEXT = "context"


@dataclass
class ConversationMessage:
    """Mensaje en una conversación."""
    message_id: str
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sentiment: Optional[SentimentType] = None
    confidence: float = 0.0


@dataclass
class ConversationContext:
    """Contexto de una conversación."""
    conversation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_type: ConversationType = ConversationType.CASUAL
    language: Language = Language.ENGLISH
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    context_data: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationResponse:
    """Respuesta de conversación."""
    message_id: str
    content: str
    confidence: float
    response_type: str
    suggested_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationAI:
    """
    Sistema de IA conversacional avanzado.
    """
    
    def __init__(self, embedding_manager: EmbeddingManager, ai_integration: AIIntegrationManager):
        """Inicializar sistema de conversación."""
        self.embedding_manager = embedding_manager
        self.ai_integration = ai_integration
        
        # Almacenamiento de conversaciones
        self.conversations: Dict[str, List[ConversationMessage]] = {}
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        
        # Configuración
        self.max_conversation_length = 50
        self.context_window = 10
        self.response_timeout = 30
        
        # Templates de respuesta
        self.response_templates = {
            ConversationType.CUSTOMER_SERVICE: {
                "greeting": "¡Hola! Soy tu asistente de atención al cliente. ¿En qué puedo ayudarte hoy?",
                "acknowledgment": "Entiendo tu consulta. Déjame ayudarte con eso.",
                "clarification": "Para poder ayudarte mejor, ¿podrías proporcionar más detalles?",
                "resolution": "Espero haber resuelto tu consulta. ¿Hay algo más en lo que pueda ayudarte?",
                "farewell": "¡Gracias por contactarnos! Que tengas un excelente día."
            },
            ConversationType.SALES: {
                "greeting": "¡Hola! Soy tu asesor de ventas. ¿Te interesa conocer nuestros productos?",
                "product_info": "Te puedo proporcionar información detallada sobre nuestros productos.",
                "pricing": "Te puedo ayudar con información sobre precios y ofertas especiales.",
                "objection_handling": "Entiendo tu preocupación. Déjame explicarte cómo podemos resolver eso.",
                "closing": "¿Te gustaría proceder con la compra o tienes alguna pregunta adicional?"
            },
            ConversationType.TECHNICAL_SUPPORT: {
                "greeting": "¡Hola! Soy tu técnico de soporte. ¿Qué problema técnico estás experimentando?",
                "troubleshooting": "Vamos a diagnosticar el problema paso a paso.",
                "solution": "Aquí tienes la solución para tu problema técnico.",
                "escalation": "Tu consulta requiere atención especializada. Te conectaré con un experto.",
                "follow_up": "¿El problema se ha resuelto? ¿Necesitas ayuda adicional?"
            }
        }
        
        # Patrones de intención
        self.intent_patterns = {
            "greeting": ["hola", "buenos días", "buenas tardes", "buenas noches", "hi", "hello"],
            "farewell": ["adiós", "hasta luego", "nos vemos", "bye", "goodbye", "see you"],
            "question": ["qué", "cómo", "cuándo", "dónde", "por qué", "what", "how", "when", "where", "why"],
            "complaint": ["problema", "error", "malo", "terrible", "problem", "issue", "bad"],
            "compliment": ["excelente", "bueno", "genial", "perfecto", "excellent", "good", "great", "perfect"],
            "request": ["necesito", "quiero", "puedes", "ayuda", "need", "want", "can you", "help"],
            "confirmation": ["sí", "correcto", "exacto", "yes", "correct", "exactly"],
            "denial": ["no", "incorrecto", "mal", "wrong", "incorrect"]
        }
        
        logger.info("ConversationAI inicializado")
    
    async def initialize(self):
        """Inicializar el sistema de conversación."""
        try:
            logger.info("ConversationAI inicializado exitosamente")
        except Exception as e:
            logger.error(f"Error al inicializar ConversationAI: {e}")
            raise
    
    async def start_conversation(
        self,
        user_id: Optional[str] = None,
        conversation_type: ConversationType = ConversationType.CASUAL,
        language: Language = Language.ENGLISH,
        initial_context: Dict[str, Any] = None
    ) -> str:
        """
        Iniciar una nueva conversación.
        
        Args:
            user_id: ID del usuario
            conversation_type: Tipo de conversación
            language: Idioma de la conversación
            initial_context: Contexto inicial
            
        Returns:
            ID de la conversación
        """
        try:
            conversation_id = str(uuid.uuid4())
            
            # Crear contexto de conversación
            context = ConversationContext(
                conversation_id=conversation_id,
                user_id=user_id,
                conversation_type=conversation_type,
                language=language,
                context_data=initial_context or {}
            )
            
            # Almacenar contexto
            self.conversation_contexts[conversation_id] = context
            self.conversations[conversation_id] = []
            
            # Generar mensaje de saludo
            greeting_message = await self._generate_greeting(conversation_type, language)
            
            # Crear mensaje de saludo
            greeting = ConversationMessage(
                message_id=str(uuid.uuid4()),
                role=MessageRole.ASSISTANT,
                content=greeting_message,
                metadata={"type": "greeting", "conversation_type": conversation_type.value}
            )
            
            # Agregar mensaje inicial
            self.conversations[conversation_id].append(greeting)
            
            logger.info(f"Conversación {conversation_id} iniciada")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error al iniciar conversación: {e}")
            raise
    
    async def send_message(
        self,
        conversation_id: str,
        message: str,
        user_id: Optional[str] = None
    ) -> ConversationResponse:
        """
        Enviar mensaje a una conversación.
        
        Args:
            conversation_id: ID de la conversación
            message: Mensaje del usuario
            user_id: ID del usuario
            
        Returns:
            Respuesta de la conversación
        """
        try:
            if conversation_id not in self.conversations:
                raise ValueError(f"Conversación {conversation_id} no encontrada")
            
            # Crear mensaje del usuario
            user_message = ConversationMessage(
                message_id=str(uuid.uuid4()),
                role=MessageRole.USER,
                content=message,
                metadata={"user_id": user_id}
            )
            
            # Agregar mensaje del usuario
            self.conversations[conversation_id].append(user_message)
            
            # Actualizar contexto
            context = self.conversation_contexts[conversation_id]
            context.last_activity = datetime.now()
            
            # Procesar mensaje y generar respuesta
            response = await self._process_message(conversation_id, user_message)
            
            # Crear mensaje de respuesta
            assistant_message = ConversationMessage(
                message_id=response.message_id,
                role=MessageRole.ASSISTANT,
                content=response.content,
                metadata=response.metadata,
                confidence=response.confidence
            )
            
            # Agregar respuesta
            self.conversations[conversation_id].append(assistant_message)
            
            # Limpiar conversación si es muy larga
            await self._cleanup_conversation(conversation_id)
            
            logger.info(f"Mensaje procesado en conversación {conversation_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error al procesar mensaje: {e}")
            raise
    
    async def _process_message(self, conversation_id: str, user_message: ConversationMessage) -> ConversationResponse:
        """Procesar mensaje del usuario y generar respuesta."""
        try:
            context = self.conversation_contexts[conversation_id]
            conversation_history = self.conversations[conversation_id]
            
            # Analizar intención del mensaje
            intent = await self._analyze_intent(user_message.content)
            
            # Analizar sentimiento
            sentiment = await self._analyze_sentiment(user_message.content)
            
            # Generar respuesta basada en el tipo de conversación
            if context.conversation_type == ConversationType.CUSTOMER_SERVICE:
                response = await self._generate_customer_service_response(
                    user_message, intent, sentiment, conversation_history
                )
            elif context.conversation_type == ConversationType.SALES:
                response = await self._generate_sales_response(
                    user_message, intent, sentiment, conversation_history
                )
            elif context.conversation_type == ConversationType.TECHNICAL_SUPPORT:
                response = await self._generate_technical_support_response(
                    user_message, intent, sentiment, conversation_history
                )
            else:
                response = await self._generate_general_response(
                    user_message, intent, sentiment, conversation_history
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error al procesar mensaje: {e}")
            # Respuesta de fallback
            return ConversationResponse(
                message_id=str(uuid.uuid4()),
                content="Lo siento, no pude procesar tu mensaje. ¿Podrías intentar de nuevo?",
                confidence=0.0,
                response_type="error"
            )
    
    async def _analyze_intent(self, message: str) -> str:
        """Analizar intención del mensaje."""
        try:
            message_lower = message.lower()
            
            # Buscar patrones de intención
            for intent, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    if pattern in message_lower:
                        return intent
            
            # Análisis más sofisticado usando embeddings
            if len(message.split()) > 3:
                # Usar similitud semántica para intenciones más complejas
                intent_embeddings = await self._get_intent_embeddings()
                message_embedding = await self.embedding_manager.get_embedding(message)
                
                # Calcular similitud con intenciones conocidas
                similarities = {}
                for intent, embedding in intent_embeddings.items():
                    similarity = await self.embedding_manager.calculate_similarity(
                        message, f"intent: {intent}"
                    )
                    similarities[intent] = similarity
                
                # Retornar intención con mayor similitud
                if similarities:
                    return max(similarities, key=similarities.get)
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error al analizar intención: {e}")
            return "unknown"
    
    async def _analyze_sentiment(self, message: str) -> SentimentType:
        """Analizar sentimiento del mensaje."""
        try:
            # Análisis básico de sentimiento
            positive_words = ["bueno", "excelente", "genial", "perfecto", "gracias", "good", "excellent", "great", "perfect", "thanks"]
            negative_words = ["malo", "terrible", "horrible", "problema", "error", "bad", "terrible", "awful", "problem", "issue"]
            
            message_lower = message.lower()
            
            positive_count = sum(1 for word in positive_words if word in message_lower)
            negative_count = sum(1 for word in negative_words if word in message_lower)
            
            if positive_count > negative_count:
                return SentimentType.POSITIVE
            elif negative_count > positive_count:
                return SentimentType.NEGATIVE
            else:
                return SentimentType.NEUTRAL
                
        except Exception as e:
            logger.error(f"Error al analizar sentimiento: {e}")
            return SentimentType.NEUTRAL
    
    async def _generate_customer_service_response(
        self,
        user_message: ConversationMessage,
        intent: str,
        sentiment: SentimentType,
        conversation_history: List[ConversationMessage]
    ) -> ConversationResponse:
        """Generar respuesta de atención al cliente."""
        try:
            templates = self.response_templates[ConversationType.CUSTOMER_SERVICE]
            
            if intent == "greeting":
                content = templates["greeting"]
                response_type = "greeting"
            elif intent == "complaint" or sentiment == SentimentType.NEGATIVE:
                content = "Entiendo tu preocupación. Déjame ayudarte a resolver este problema. ¿Podrías proporcionarme más detalles?"
                response_type = "complaint_handling"
            elif intent == "question":
                content = templates["acknowledgment"] + " " + templates["clarification"]
                response_type = "question_handling"
            elif intent == "farewell":
                content = templates["farewell"]
                response_type = "farewell"
            else:
                content = templates["acknowledgment"]
                response_type = "general"
            
            # Agregar sugerencias de acciones
            suggested_actions = await self._generate_customer_service_actions(intent, sentiment)
            
            return ConversationResponse(
                message_id=str(uuid.uuid4()),
                content=content,
                confidence=0.8,
                response_type=response_type,
                suggested_actions=suggested_actions,
                metadata={
                    "intent": intent,
                    "sentiment": sentiment.value,
                    "conversation_type": "customer_service"
                }
            )
            
        except Exception as e:
            logger.error(f"Error al generar respuesta de atención al cliente: {e}")
            raise
    
    async def _generate_sales_response(
        self,
        user_message: ConversationMessage,
        intent: str,
        sentiment: SentimentType,
        conversation_history: List[ConversationMessage]
    ) -> ConversationResponse:
        """Generar respuesta de ventas."""
        try:
            templates = self.response_templates[ConversationType.SALES]
            
            if intent == "greeting":
                content = templates["greeting"]
                response_type = "greeting"
            elif "producto" in user_message.content.lower() or "product" in user_message.content.lower():
                content = templates["product_info"]
                response_type = "product_information"
            elif "precio" in user_message.content.lower() or "price" in user_message.content.lower():
                content = templates["pricing"]
                response_type = "pricing_information"
            elif intent == "complaint" or sentiment == SentimentType.NEGATIVE:
                content = templates["objection_handling"]
                response_type = "objection_handling"
            else:
                content = templates["greeting"]
                response_type = "general"
            
            # Agregar sugerencias de acciones
            suggested_actions = await self._generate_sales_actions(intent, sentiment)
            
            return ConversationResponse(
                message_id=str(uuid.uuid4()),
                content=content,
                confidence=0.8,
                response_type=response_type,
                suggested_actions=suggested_actions,
                metadata={
                    "intent": intent,
                    "sentiment": sentiment.value,
                    "conversation_type": "sales"
                }
            )
            
        except Exception as e:
            logger.error(f"Error al generar respuesta de ventas: {e}")
            raise
    
    async def _generate_technical_support_response(
        self,
        user_message: ConversationMessage,
        intent: str,
        sentiment: SentimentType,
        conversation_history: List[ConversationMessage]
    ) -> ConversationResponse:
        """Generar respuesta de soporte técnico."""
        try:
            templates = self.response_templates[ConversationType.TECHNICAL_SUPPORT]
            
            if intent == "greeting":
                content = templates["greeting"]
                response_type = "greeting"
            elif "problema" in user_message.content.lower() or "problem" in user_message.content.lower():
                content = templates["troubleshooting"]
                response_type = "troubleshooting"
            elif intent == "question":
                content = templates["solution"]
                response_type = "solution_provided"
            else:
                content = templates["greeting"]
                response_type = "general"
            
            # Agregar sugerencias de acciones
            suggested_actions = await self._generate_technical_support_actions(intent, sentiment)
            
            return ConversationResponse(
                message_id=str(uuid.uuid4()),
                content=content,
                confidence=0.8,
                response_type=response_type,
                suggested_actions=suggested_actions,
                metadata={
                    "intent": intent,
                    "sentiment": sentiment.value,
                    "conversation_type": "technical_support"
                }
            )
            
        except Exception as e:
            logger.error(f"Error al generar respuesta de soporte técnico: {e}")
            raise
    
    async def _generate_general_response(
        self,
        user_message: ConversationMessage,
        intent: str,
        sentiment: SentimentType,
        conversation_history: List[ConversationMessage]
    ) -> ConversationResponse:
        """Generar respuesta general."""
        try:
            if intent == "greeting":
                content = "¡Hola! ¿En qué puedo ayudarte hoy?"
                response_type = "greeting"
            elif intent == "farewell":
                content = "¡Hasta luego! Que tengas un excelente día."
                response_type = "farewell"
            elif intent == "question":
                content = "Esa es una excelente pregunta. Déjame ayudarte con eso."
                response_type = "question_handling"
            else:
                content = "Entiendo. ¿Hay algo específico en lo que pueda ayudarte?"
                response_type = "general"
            
            return ConversationResponse(
                message_id=str(uuid.uuid4()),
                content=content,
                confidence=0.7,
                response_type=response_type,
                metadata={
                    "intent": intent,
                    "sentiment": sentiment.value,
                    "conversation_type": "general"
                }
            )
            
        except Exception as e:
            logger.error(f"Error al generar respuesta general: {e}")
            raise
    
    async def _generate_greeting(self, conversation_type: ConversationType, language: Language) -> str:
        """Generar mensaje de saludo."""
        templates = self.response_templates.get(conversation_type, {})
        return templates.get("greeting", "¡Hola! ¿En qué puedo ayudarte?")
    
    async def _generate_customer_service_actions(self, intent: str, sentiment: SentimentType) -> List[str]:
        """Generar acciones sugeridas para atención al cliente."""
        actions = []
        
        if intent == "complaint" or sentiment == SentimentType.NEGATIVE:
            actions.extend(["Escalar a supervisor", "Ofrecer compensación", "Programar seguimiento"])
        elif intent == "question":
            actions.extend(["Proporcionar información", "Conectar con especialista", "Crear ticket"])
        else:
            actions.extend(["Proporcionar ayuda", "Obtener más información", "Conectar con agente"])
        
        return actions
    
    async def _generate_sales_actions(self, intent: str, sentiment: SentimentType) -> List[str]:
        """Generar acciones sugeridas para ventas."""
        actions = []
        
        if "producto" in intent or "product" in intent:
            actions.extend(["Mostrar catálogo", "Programar demo", "Enviar información"])
        elif "precio" in intent or "price" in intent:
            actions.extend(["Mostrar precios", "Calcular cotización", "Aplicar descuentos"])
        else:
            actions.extend(["Identificar necesidades", "Mostrar productos", "Programar reunión"])
        
        return actions
    
    async def _generate_technical_support_actions(self, intent: str, sentiment: SentimentType) -> List[str]:
        """Generar acciones sugeridas para soporte técnico."""
        actions = []
        
        if "problema" in intent or "problem" in intent:
            actions.extend(["Diagnosticar problema", "Proporcionar solución", "Escalar a técnico"])
        elif intent == "question":
            actions.extend(["Proporcionar documentación", "Programar llamada", "Crear ticket"])
        else:
            actions.extend(["Obtener más información", "Proporcionar ayuda", "Conectar con experto"])
        
        return actions
    
    async def _get_intent_embeddings(self) -> Dict[str, Any]:
        """Obtener embeddings de intenciones."""
        # Implementación básica - en producción usar embeddings pre-entrenados
        return {
            "greeting": await self.embedding_manager.get_embedding("greeting hello hi"),
            "question": await self.embedding_manager.get_embedding("question what how when where why"),
            "complaint": await self.embedding_manager.get_embedding("complaint problem issue bad"),
            "request": await self.embedding_manager.get_embedding("request need want help")
        }
    
    async def _cleanup_conversation(self, conversation_id: str):
        """Limpiar conversación si es muy larga."""
        try:
            if len(self.conversations[conversation_id]) > self.max_conversation_length:
                # Mantener solo los últimos mensajes
                self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_conversation_length:]
                logger.info(f"Conversación {conversation_id} limpiada")
        except Exception as e:
            logger.error(f"Error al limpiar conversación: {e}")
    
    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Obtener historial de conversación."""
        try:
            if conversation_id not in self.conversations:
                return []
            
            return [
                {
                    "message_id": msg.message_id,
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata,
                    "sentiment": msg.sentiment.value if msg.sentiment else None,
                    "confidence": msg.confidence
                }
                for msg in self.conversations[conversation_id]
            ]
        except Exception as e:
            logger.error(f"Error al obtener historial de conversación: {e}")
            return []
    
    async def get_conversation_context(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Obtener contexto de conversación."""
        try:
            if conversation_id not in self.conversation_contexts:
                return None
            
            context = self.conversation_contexts[conversation_id]
            return {
                "conversation_id": context.conversation_id,
                "user_id": context.user_id,
                "conversation_type": context.conversation_type.value,
                "language": context.language.value,
                "created_at": context.created_at.isoformat(),
                "last_activity": context.last_activity.isoformat(),
                "context_data": context.context_data,
                "user_preferences": context.user_preferences
            }
        except Exception as e:
            logger.error(f"Error al obtener contexto de conversación: {e}")
            return None
    
    async def update_conversation_context(
        self,
        conversation_id: str,
        context_data: Dict[str, Any]
    ) -> bool:
        """Actualizar contexto de conversación."""
        try:
            if conversation_id not in self.conversation_contexts:
                return False
            
            context = self.conversation_contexts[conversation_id]
            context.context_data.update(context_data)
            context.last_activity = datetime.now()
            
            return True
        except Exception as e:
            logger.error(f"Error al actualizar contexto de conversación: {e}")
            return False
    
    async def end_conversation(self, conversation_id: str) -> bool:
        """Finalizar conversación."""
        try:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
            
            if conversation_id in self.conversation_contexts:
                del self.conversation_contexts[conversation_id]
            
            logger.info(f"Conversación {conversation_id} finalizada")
            return True
        except Exception as e:
            logger.error(f"Error al finalizar conversación: {e}")
            return False
    
    async def get_conversation_analytics(self, conversation_id: str) -> Dict[str, Any]:
        """Obtener analíticas de conversación."""
        try:
            if conversation_id not in self.conversations:
                return {}
            
            messages = self.conversations[conversation_id]
            context = self.conversation_contexts.get(conversation_id)
            
            # Calcular métricas
            total_messages = len(messages)
            user_messages = [m for m in messages if m.role == MessageRole.USER]
            assistant_messages = [m for m in messages if m.role == MessageRole.ASSISTANT]
            
            # Análisis de sentimiento
            sentiments = [m.sentiment for m in messages if m.sentiment]
            sentiment_distribution = {}
            if sentiments:
                sentiment_counts = Counter([s.value for s in sentiments])
                total_sentiments = len(sentiments)
                sentiment_distribution = {
                    sentiment: count / total_sentiments
                    for sentiment, count in sentiment_counts.items()
                }
            
            # Tiempo promedio de respuesta
            response_times = []
            for i in range(1, len(messages)):
                if (messages[i].role == MessageRole.ASSISTANT and 
                    messages[i-1].role == MessageRole.USER):
                    response_time = (messages[i].timestamp - messages[i-1].timestamp).total_seconds()
                    response_times.append(response_time)
            
            avg_response_time = statistics.mean(response_times) if response_times else 0
            
            return {
                "conversation_id": conversation_id,
                "total_messages": total_messages,
                "user_messages": len(user_messages),
                "assistant_messages": len(assistant_messages),
                "conversation_type": context.conversation_type.value if context else "unknown",
                "language": context.language.value if context else "unknown",
                "duration_minutes": (
                    (messages[-1].timestamp - messages[0].timestamp).total_seconds() / 60
                    if len(messages) > 1 else 0
                ),
                "sentiment_distribution": sentiment_distribution,
                "average_response_time": avg_response_time,
                "created_at": context.created_at.isoformat() if context else None,
                "last_activity": context.last_activity.isoformat() if context else None
            }
        except Exception as e:
            logger.error(f"Error al obtener analíticas de conversación: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del sistema de conversación."""
        try:
            return {
                "status": "healthy",
                "active_conversations": len(self.conversations),
                "conversation_contexts": len(self.conversation_contexts),
                "embedding_manager_status": await self.embedding_manager.health_check(),
                "ai_integration_status": await self.ai_integration.health_check(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error en health check de ConversationAI: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }





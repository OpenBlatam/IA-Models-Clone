"""
Devin Persona System
====================

Sistema que implementa el comportamiento y personalidad de Devin,
un ingeniero de software experto que usa herramientas reales para
completar tareas de manera eficiente y profesional.
"""

import asyncio
import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    """Modos de operación del agente"""
    PLANNING = "planning"
    STANDARD = "standard"


class CommunicationLevel(Enum):
    """Niveles de comunicación con el usuario"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    DEBUG = "debug"


@dataclass
class UserMessage:
    """Mensaje para el usuario"""
    content: str
    level: CommunicationLevel = CommunicationLevel.INFO
    attachments: Optional[List[str]] = None
    request_auth: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "content": self.content,
            "level": self.level.value,
            "attachments": self.attachments or [],
            "request_auth": self.request_auth,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EnvironmentIssue:
    """Problema de entorno reportado"""
    issue_type: str
    description: str
    suggestion: str
    severity: str = "medium"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "issue_type": self.issue_type,
            "description": self.description,
            "suggestion": self.suggestion,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ReasoningContext:
    """Contexto para razonamiento interno"""
    observations: List[str] = field(default_factory=list)
    considerations: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_observation(self, observation: str) -> None:
        """Agregar observación"""
        self.observations.append(f"[{datetime.now().isoformat()}] {observation}")
    
    def add_consideration(self, consideration: str) -> None:
        """Agregar consideración"""
        self.considerations.append(f"[{datetime.now().isoformat()}] {consideration}")
    
    def add_conclusion(self, conclusion: str) -> None:
        """Agregar conclusión"""
        self.conclusions.append(f"[{datetime.now().isoformat()}] {conclusion}")
    
    def add_next_step(self, step: str) -> None:
        """Agregar siguiente paso"""
        self.next_steps.append(f"[{datetime.now().isoformat()}] {step}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "observations": self.observations,
            "considerations": self.considerations,
            "conclusions": self.conclusions,
            "next_steps": self.next_steps,
            "timestamp": self.timestamp.isoformat()
        }


class DevinPersona:
    """
    Sistema de personalidad Devin.
    
    Implementa el comportamiento descrito en el prompt de Devin:
    - Comunicación estratégica con el usuario
    - Reporte de problemas de entorno
    - Razonamiento interno
    - Modos de operación (planning/standard)
    - Mejores prácticas de código
    """
    
    def __init__(self, agent: Optional[Any] = None) -> None:
        """
        Inicializar personalidad Devin.
        
        Args:
            agent: Instancia del agente (opcional, para integración con CI).
        """
        self.mode: AgentMode = AgentMode.STANDARD
        self.language: str = "es"
        self.user_messages: List[UserMessage] = []
        self.environment_issues: List[EnvironmentIssue] = []
        self.reasoning_contexts: List[ReasoningContext] = []
        self.message_callbacks: List[Callable[[UserMessage], None]] = []
        self.issue_callbacks: List[Callable[[EnvironmentIssue], None]] = []
        self._agent: Optional[Any] = agent
        
        logger.info("🤖 Devin Persona initialized")
    
    def set_mode(self, mode: AgentMode) -> None:
        """
        Establecer modo de operación.
        
        Args:
            mode: Modo a establecer (PLANNING o STANDARD).
        """
        self.mode = mode
        logger.info(f"🔄 Mode changed to: {mode.value}")
    
    def set_language(self, language: str) -> None:
        """
        Establecer idioma de comunicación.
        
        Args:
            language: Código de idioma (ej: "es", "en").
        """
        self.language = language
        logger.debug(f"Language set to: {language}")
    
    def should_communicate(
        self,
        reason: str,
        critical: bool = False
    ) -> bool:
        """
        Determinar si debe comunicarse con el usuario.
        
        Según las reglas de Devin:
        - Cuando encuentra problemas de entorno
        - Para compartir entregables
        - Cuando información crítica no puede ser accedida
        - Cuando solicita permisos o claves
        - Usar el mismo idioma que el usuario
        
        Args:
            reason: Razón para comunicarse.
            critical: Si es crítico (siempre comunica).
        
        Returns:
            True si debe comunicarse.
        """
        if critical:
            return True
        
        reasons_to_communicate = [
            "environment_issue",
            "deliverable",
            "critical_info_unavailable",
            "permission_needed",
            "key_needed"
        ]
        
        return reason in reasons_to_communicate
    
    async def message_user(
        self,
        content: str,
        level: CommunicationLevel = CommunicationLevel.INFO,
        attachments: Optional[List[str]] = None,
        request_auth: bool = False
    ) -> None:
        """
        Enviar mensaje al usuario.
        
        Args:
            content: Contenido del mensaje.
            level: Nivel de comunicación.
            attachments: Archivos adjuntos (rutas).
            request_auth: Si solicita autenticación.
        """
        message = UserMessage(
            content=content,
            level=level,
            attachments=attachments,
            request_auth=request_auth
        )
        
        self.user_messages.append(message)
        
        logger.info(f"💬 Message to user ({level.value}): {content[:100]}...")
        
        for callback in self.message_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                logger.error(f"Error in message callback: {e}", exc_info=True)
    
    async def report_environment_issue(
        self,
        issue_type: str,
        description: str,
        suggestion: str,
        severity: str = "medium"
    ) -> None:
        """
        Reportar problema de entorno.
        
        Args:
            issue_type: Tipo de problema.
            description: Descripción del problema.
            suggestion: Sugerencia de solución.
            severity: Severidad (low, medium, high, critical).
        """
        issue = EnvironmentIssue(
            issue_type=issue_type,
            description=description,
            suggestion=suggestion,
            severity=severity
        )
        
        self.environment_issues.append(issue)
        
        logger.warning(
            f"⚠️ Environment issue ({severity}): {issue_type} - {description}"
        )
        
        message_content = (
            f"**Environment Issue Detected**\n\n"
            f"**Type:** {issue_type}\n"
            f"**Description:** {description}\n"
            f"**Suggestion:** {suggestion}\n"
            f"**Severity:** {severity}"
        )
        
        level_map = {
            "low": CommunicationLevel.INFO,
            "medium": CommunicationLevel.WARNING,
            "high": CommunicationLevel.ERROR,
            "critical": CommunicationLevel.ERROR
        }
        
        await self.message_user(
            message_content,
            level=level_map.get(severity, CommunicationLevel.WARNING)
        )
        
        if hasattr(self, '_agent') and self._agent:
            if hasattr(self._agent, 'ci_integration') and self._agent.ci_integration:
                if self._agent.ci_integration.should_use_ci(has_environment_issue=True):
                    await self.message_user(
                        "💡 Sugerencia: Usando CI para testing en lugar del entorno local debido al problema detectado.",
                        level=CommunicationLevel.INFO
                    )
        
        for callback in self.issue_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(issue)
                else:
                    callback(issue)
            except Exception as e:
                logger.error(f"Error in issue callback: {e}", exc_info=True)
    
    def start_reasoning(self) -> ReasoningContext:
        """
        Iniciar sesión de razonamiento.
        
        Returns:
            Contexto de razonamiento nuevo.
        """
        context = ReasoningContext()
        self.reasoning_contexts.append(context)
        logger.debug("🧠 Started reasoning session")
        return context
    
    def get_recent_reasoning(self, limit: int = 5) -> List[ReasoningContext]:
        """
        Obtener contextos de razonamiento recientes.
        
        Args:
            limit: Número máximo de contextos a retornar.
        
        Returns:
            Lista de contextos de razonamiento.
        """
        return self.reasoning_contexts[-limit:]
    
    def on_message(self, callback: Callable[[UserMessage], None]) -> None:
        """
        Registrar callback para mensajes al usuario.
        
        Args:
            callback: Función o coroutine a llamar.
        """
        self.message_callbacks.append(callback)
    
    def on_environment_issue(
        self,
        callback: Callable[[EnvironmentIssue], None]
    ) -> None:
        """
        Registrar callback para problemas de entorno.
        
        Args:
            callback: Función o coroutine a llamar.
        """
        self.issue_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado de la personalidad.
        
        Returns:
            Diccionario con estado actual.
        """
        return {
            "mode": self.mode.value,
            "language": self.language,
            "messages_count": len(self.user_messages),
            "issues_count": len(self.environment_issues),
            "reasoning_sessions": len(self.reasoning_contexts),
            "recent_messages": [
                msg.to_dict() for msg in self.user_messages[-5:]
            ],
            "recent_issues": [
                issue.to_dict() for issue in self.environment_issues[-5:]
            ]
        }
    
    def clear_history(self) -> None:
        """Limpiar historial de mensajes e issues"""
        self.user_messages.clear()
        self.environment_issues.clear()
        self.reasoning_contexts.clear()
        logger.info("🧹 Cleared Devin persona history")
    
    async def suggest_plan(self, plan_details: Dict[str, Any]) -> None:
        """
        Sugerir un plan al usuario (similar a <suggest_plan/>).
        
        Verifica que se tiene toda la información necesaria antes de sugerir,
        siguiendo las mejores prácticas de Devin.
        
        Args:
            plan_details: Detalles del plan a sugerir.
        """
        plan_id = plan_details.get('id', 'default')
        
        # Verificar antes de sugerir si hay agente disponible
        if hasattr(self, '_agent') and self._agent:
            if hasattr(self._agent, 'planning_verifier') and self._agent.planning_verifier:
                locations = plan_details.get('locations_to_edit', [])
                references = plan_details.get('references_to_update', [])
                information = plan_details.get('information_gathered', {})
                
                verification_result = await self._agent.planning_verifier.verify_before_suggesting_plan(
                    plan_id=plan_id,
                    locations_to_edit=locations if locations else None,
                    references_to_update=references if references else None,
                    information_gathered=information if information else None,
                    agent=self._agent
                )
                
                if not verification_result.get('can_suggest_plan', False):
                    issues = verification_result.get('issues', [])
                    issues_text = '\n'.join(f"- {issue.get('description', 'Unknown')}" for issue in issues)
                    await self.message_user(
                        f"⚠️ No se puede sugerir el plan aún. Faltan verificaciones:\n{issues_text}",
                        level=CommunicationLevel.WARNING
                    )
                    return
        
        plan_summary = f"**Plan Sugerido**\n\n"
        plan_summary += f"**Título:** {plan_details.get('title', 'Sin título')}\n"
        plan_summary += f"**Descripción:** {plan_details.get('description', 'Sin descripción')}\n"
        
        if 'steps' in plan_details:
            plan_summary += f"\n**Pasos ({len(plan_details['steps'])}):**\n"
            for i, step in enumerate(plan_details['steps'], 1):
                plan_summary += f"{i}. {step.get('description', 'Sin descripción')}\n"
        
        if 'locations_to_edit' in plan_details and plan_details['locations_to_edit']:
            plan_summary += f"\n**Ubicaciones a Editar ({len(plan_details['locations_to_edit'])}):**\n"
            for loc in plan_details['locations_to_edit']:
                plan_summary += f"- {loc.get('file_path', 'Unknown')}\n"
        
        if 'references_to_update' in plan_details and plan_details['references_to_update']:
            plan_summary += f"\n**Referencias a Actualizar ({len(plan_details['references_to_update'])}):**\n"
            for ref in plan_details['references_to_update']:
                plan_summary += f"- {ref.get('file_path', 'Unknown')}\n"
        
        await self.message_user(
            plan_summary,
            level=CommunicationLevel.INFO
        )


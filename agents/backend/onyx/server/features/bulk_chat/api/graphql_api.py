"""
GraphQL API - API GraphQL
=========================

API GraphQL para consultas flexibles y eficientes.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
try:
    from strawberry import Schema, Query, Mutation, Field, type as strawberry_type
    from strawberry.fastapi import GraphQLRouter
    import strawberry
    GRAPHQL_AVAILABLE = True
except ImportError:
    GRAPHQL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Strawberry GraphQL not available. Install with: pip install strawberry-graphql")

from ..core.chat_engine import ContinuousChatEngine
from ..core.chat_session import ChatSession, ChatState

logger = logging.getLogger(__name__)

# Solo definir tipos si GraphQL está disponible
if GRAPHQL_AVAILABLE:
    @strawberry.type
    class Message:
        """Tipo GraphQL para mensaje."""
        id: str
        role: str
        content: str
        timestamp: str

    @strawberry.type
    class Session:
        """Tipo GraphQL para sesión."""
        session_id: str
        user_id: Optional[str]
        state: str
        is_paused: bool
        message_count: int
        auto_continue: bool
        messages: List[Message]
        
        @strawberry.field
        def recent_messages(self, limit: int = 10) -> List[Message]:
            """Obtener mensajes recientes."""
            return self.messages[-limit:] if limit else self.messages

    @strawberry.type
    class Metrics:
        """Tipo GraphQL para métricas."""
        total_sessions: int
        active_sessions: int
        total_messages: int
        average_response_time: float

    @strawberry.type
    class Query:
        """Query GraphQL."""
        
        @strawberry.field
        def session(self, session_id: str, info) -> Optional[Session]:
            """Obtener sesión por ID."""
            chat_engine = info.context["chat_engine"]
            session = chat_engine.get_session(session_id)
            if not session:
                return None
            
            messages = [
                Message(
                    id=msg.id,
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp.isoformat(),
                )
                for msg in session.messages
            ]
            
            return Session(
                session_id=session.session_id,
                user_id=session.user_id,
                state=session.state.value,
                is_paused=session.is_paused,
                message_count=len(session.messages),
                auto_continue=session.auto_continue,
                messages=messages,
            )
        
        @strawberry.field
        def sessions(self, user_id: Optional[str] = None, info=None) -> List[Session]:
            """Listar sesiones."""
            chat_engine = info.context["chat_engine"]
            sessions = list(chat_engine.sessions.values())
            
            if user_id:
                sessions = [s for s in sessions if s.user_id == user_id]
            
            return [
                Session(
                    session_id=s.session_id,
                    user_id=s.user_id,
                    state=s.state.value,
                    is_paused=s.is_paused,
                    message_count=len(s.messages),
                    auto_continue=s.auto_continue,
                    messages=[
                        Message(
                            id=msg.id,
                            role=msg.role,
                            content=msg.content,
                            timestamp=msg.timestamp.isoformat(),
                        )
                        for msg in s.messages
                    ],
                )
                for s in sessions
            ]
        
        @strawberry.field
        def metrics(self, info) -> Optional[Metrics]:
            """Obtener métricas globales."""
            chat_engine = info.context["chat_engine"]
            if not chat_engine.metrics:
                return None
            
            global_metrics = chat_engine.metrics.get_global_metrics()
            
            return Metrics(
                total_sessions=global_metrics.get("total_sessions", 0),
                active_sessions=global_metrics.get("active_sessions", 0),
                total_messages=global_metrics.get("total_messages", 0),
                average_response_time=global_metrics.get("average_response_time", 0.0),
            )

    @strawberry.type
    class Mutation:
        """Mutation GraphQL."""
        
        @strawberry.mutation
        async def create_session(
            self,
            user_id: Optional[str] = None,
            initial_message: Optional[str] = None,
            auto_continue: bool = True,
            info=None,
        ) -> Session:
            """Crear nueva sesión."""
            chat_engine = info.context["chat_engine"]
            session = await chat_engine.create_session(
                user_id=user_id,
                initial_message=initial_message,
                auto_continue=auto_continue,
            )
            
            if initial_message:
                await chat_engine.start_continuous_chat(session.session_id)
            
            return Session(
                session_id=session.session_id,
                user_id=session.user_id,
                state=session.state.value,
                is_paused=session.is_paused,
                message_count=len(session.messages),
                auto_continue=session.auto_continue,
                messages=[],
            )
        
        @strawberry.mutation
        async def pause_session(self, session_id: str, info=None) -> bool:
            """Pausar sesión."""
            chat_engine = info.context["chat_engine"]
            try:
                await chat_engine.pause_session(session_id)
                return True
            except Exception:
                return False
        
        @strawberry.mutation
        async def resume_session(self, session_id: str, info=None) -> bool:
            """Reanudar sesión."""
            chat_engine = info.context["chat_engine"]
            try:
                await chat_engine.resume_session(session_id)
                return True
            except Exception:
                return False


def create_graphql_router(chat_engine: ContinuousChatEngine):
    """Crear router GraphQL."""
    if not GRAPHQL_AVAILABLE:
        raise ImportError("Strawberry GraphQL not available. Install with: pip install strawberry-graphql")
    
    # Crear schema con contexto
    schema = Schema(
        query=Query,
        mutation=Mutation,
    )
    
    # Inyectar chat_engine en el contexto
    def get_context():
        return {"chat_engine": chat_engine}
    
    router = GraphQLRouter(
        schema,
        context_getter=get_context,
    )
    
    return router

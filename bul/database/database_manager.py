"""
Database Manager
================

Gestor de base de datos para el sistema BUL.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import json

from .models import Base, Document, DocumentRequest, Agent, AgentUsageLog, DocumentFeedback, SystemStats, UserSession, CacheEntry, APILog, Template, Configuration

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Gestor de base de datos
    
    Maneja las conexiones y operaciones de base de datos para el sistema BUL.
    """
    
    def __init__(self, database_url: str, echo: bool = False):
        self.database_url = database_url
        self.echo = echo
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        self.is_initialized = False
        
        logger.info("Database Manager initialized")
    
    async def initialize(self) -> bool:
        """Inicializar el gestor de base de datos"""
        try:
            # Crear motor síncrono
            self.engine = create_engine(
                self.database_url,
                echo=self.echo,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Crear motor asíncrono si es posible
            if self.database_url.startswith(("postgresql://", "postgresql+asyncpg://")):
                async_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://")
                self.async_engine = create_async_engine(
                    async_url,
                    echo=self.echo,
                    pool_pre_ping=True,
                    pool_recycle=3600
                )
                self.async_session_factory = async_sessionmaker(
                    self.async_engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )
            
            # Crear factory de sesiones síncronas
            self.session_factory = sessionmaker(bind=self.engine)
            
            # Crear tablas
            await self.create_tables()
            
            # Inicializar datos por defecto
            await self.initialize_default_data()
            
            self.is_initialized = True
            logger.info("Database Manager fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Database Manager: {e}")
            return False
    
    async def create_tables(self):
        """Crear tablas de la base de datos"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    async def initialize_default_data(self):
        """Inicializar datos por defecto"""
        try:
            with self.get_session() as session:
                # Verificar si ya hay datos
                if session.query(Agent).count() > 0:
                    logger.info("Default data already exists")
                    return
                
                # Crear agentes por defecto
                default_agents = [
                    {
                        "name": "María González - Especialista en Marketing",
                        "agent_type": "marketing_specialist",
                        "experience_years": 8,
                        "success_rate": 0.92,
                        "specializations": ["Marketing Digital", "Redes Sociales", "SEO", "Content Marketing"],
                        "capabilities": ["marketing_strategy", "content_strategy"]
                    },
                    {
                        "name": "Carlos Rodríguez - Experto en Ventas",
                        "agent_type": "sales_expert",
                        "experience_years": 10,
                        "success_rate": 0.89,
                        "specializations": ["Ventas B2B", "CRM", "Negociación", "Funnel de Ventas"],
                        "capabilities": ["sales_proposal", "business_plan"]
                    },
                    {
                        "name": "Ana Martínez - Gerente de Operaciones",
                        "agent_type": "operations_manager",
                        "experience_years": 12,
                        "success_rate": 0.94,
                        "specializations": ["Procesos", "Logística", "Calidad", "Optimización"],
                        "capabilities": ["operational_manual", "business_plan"]
                    },
                    {
                        "name": "Luis Fernández - Consultor de RRHH",
                        "agent_type": "hr_consultant",
                        "experience_years": 7,
                        "success_rate": 0.87,
                        "specializations": ["Reclutamiento", "Capacitación", "Políticas", "Cultura Organizacional"],
                        "capabilities": ["hr_policy", "operational_manual"]
                    },
                    {
                        "name": "Patricia López - Asesora Financiera",
                        "agent_type": "financial_advisor",
                        "experience_years": 15,
                        "success_rate": 0.96,
                        "specializations": ["Finanzas Corporativas", "Presupuestos", "Análisis Financiero", "Inversiones"],
                        "capabilities": ["financial_report", "business_plan"]
                    }
                ]
                
                for agent_data in default_agents:
                    agent = Agent(**agent_data)
                    session.add(agent)
                
                # Crear configuraciones por defecto
                default_configs = [
                    {
                        "key": "max_documents_per_batch",
                        "value": "10",
                        "value_type": "int",
                        "description": "Máximo número de documentos por lote",
                        "category": "limits"
                    },
                    {
                        "key": "default_language",
                        "value": "es",
                        "value_type": "string",
                        "description": "Idioma por defecto",
                        "category": "localization"
                    },
                    {
                        "key": "cache_ttl",
                        "value": "3600",
                        "value_type": "int",
                        "description": "TTL por defecto del cache en segundos",
                        "category": "cache"
                    }
                ]
                
                for config_data in default_configs:
                    config = Configuration(**config_data)
                    session.add(config)
                
                session.commit()
                logger.info("Default data initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing default data: {e}")
            raise
    
    @asynccontextmanager
    async def get_async_session(self):
        """Obtener sesión asíncrona"""
        if not self.async_session_factory:
            raise RuntimeError("Async engine not available")
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    def get_session(self):
        """Obtener sesión síncrona"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    async def save_document(self, document_data: Dict[str, Any]) -> str:
        """Guardar documento en la base de datos"""
        try:
            with self.get_session() as session:
                document = Document(**document_data)
                session.add(document)
                session.commit()
                return str(document.id)
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            raise
    
    async def save_document_request(self, request_data: Dict[str, Any]) -> str:
        """Guardar solicitud de documento"""
        try:
            with self.get_session() as session:
                request = DocumentRequest(**request_data)
                session.add(request)
                session.commit()
                return str(request.id)
        except Exception as e:
            logger.error(f"Error saving document request: {e}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Obtener documento por ID"""
        try:
            with self.get_session() as session:
                document = session.query(Document).filter(Document.id == document_id).first()
                if document:
                    return {
                        "id": str(document.id),
                        "title": document.title,
                        "content": document.content,
                        "summary": document.summary,
                        "business_area": document.business_area,
                        "document_type": document.document_type,
                        "company_name": document.company_name,
                        "industry": document.industry,
                        "word_count": document.word_count,
                        "processing_time": document.processing_time,
                        "confidence_score": document.confidence_score,
                        "agent_used": document.agent_used,
                        "created_at": document.created_at.isoformat(),
                        "metadata": document.metadata
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None
    
    async def get_documents_by_company(self, company_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener documentos por empresa"""
        try:
            with self.get_session() as session:
                documents = session.query(Document).filter(
                    Document.company_name.ilike(f"%{company_name}%")
                ).order_by(Document.created_at.desc()).limit(limit).all()
                
                return [
                    {
                        "id": str(doc.id),
                        "title": doc.title,
                        "business_area": doc.business_area,
                        "document_type": doc.document_type,
                        "created_at": doc.created_at.isoformat(),
                        "word_count": doc.word_count
                    }
                    for doc in documents
                ]
        except Exception as e:
            logger.error(f"Error getting documents by company: {e}")
            return []
    
    async def get_agent_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de agentes"""
        try:
            with self.get_session() as session:
                agents = session.query(Agent).all()
                
                stats = {
                    "total_agents": len(agents),
                    "active_agents": len([a for a in agents if a.is_active]),
                    "total_documents": sum(a.total_documents_generated for a in agents),
                    "average_success_rate": sum(a.success_rate for a in agents) / len(agents) if agents else 0,
                    "agents": [
                        {
                            "id": str(agent.id),
                            "name": agent.name,
                            "agent_type": agent.agent_type,
                            "experience_years": agent.experience_years,
                            "success_rate": agent.success_rate,
                            "total_documents_generated": agent.total_documents_generated,
                            "average_rating": agent.average_rating,
                            "is_active": agent.is_active
                        }
                        for agent in agents
                    ]
                }
                
                return stats
        except Exception as e:
            logger.error(f"Error getting agent stats: {e}")
            return {}
    
    async def log_agent_usage(self, agent_id: str, document_request_id: str, 
                            processing_time: float, confidence_score: float, 
                            success: bool = True, error_message: str = None):
        """Registrar uso de agente"""
        try:
            with self.get_session() as session:
                usage_log = AgentUsageLog(
                    agent_id=agent_id,
                    document_request_id=document_request_id,
                    processing_time=processing_time,
                    confidence_score=confidence_score,
                    success=success,
                    error_message=error_message
                )
                session.add(usage_log)
                
                # Actualizar estadísticas del agente
                agent = session.query(Agent).filter(Agent.id == agent_id).first()
                if agent:
                    agent.total_documents_generated += 1
                    agent.last_used = datetime.utcnow()
                
                session.commit()
        except Exception as e:
            logger.error(f"Error logging agent usage: {e}")
    
    async def save_document_feedback(self, document_id: str, user_id: str, 
                                   rating: int, feedback_text: str = None,
                                   quality_score: float = None, relevance_score: float = None,
                                   completeness_score: float = None):
        """Guardar feedback de documento"""
        try:
            with self.get_session() as session:
                feedback = DocumentFeedback(
                    document_id=document_id,
                    user_id=user_id,
                    rating=rating,
                    feedback_text=feedback_text,
                    quality_score=quality_score,
                    relevance_score=relevance_score,
                    completeness_score=completeness_score
                )
                session.add(feedback)
                session.commit()
        except Exception as e:
            logger.error(f"Error saving document feedback: {e}")
    
    async def get_system_stats(self, days: int = 30) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        try:
            with self.get_session() as session:
                # Estadísticas de documentos
                total_documents = session.query(Document).count()
                recent_documents = session.query(Document).filter(
                    Document.created_at >= datetime.utcnow() - timedelta(days=days)
                ).count()
                
                # Estadísticas de agentes
                total_agents = session.query(Agent).count()
                active_agents = session.query(Agent).filter(Agent.is_active == True).count()
                
                # Estadísticas de feedback
                total_feedback = session.query(DocumentFeedback).count()
                avg_rating = session.query(DocumentFeedback.rating).filter(
                    DocumentFeedback.rating.isnot(None)
                ).all()
                avg_rating = sum(r[0] for r in avg_rating) / len(avg_rating) if avg_rating else 0
                
                # Estadísticas por área de negocio
                business_areas = session.query(
                    Document.business_area,
                    session.query(Document).filter(Document.business_area == Document.business_area).count().label('count')
                ).group_by(Document.business_area).all()
                
                business_areas_dict = {area: count for area, count in business_areas}
                
                return {
                    "total_documents": total_documents,
                    "recent_documents": recent_documents,
                    "total_agents": total_agents,
                    "active_agents": active_agents,
                    "total_feedback": total_feedback,
                    "average_rating": round(avg_rating, 2),
                    "business_areas_usage": business_areas_dict,
                    "period_days": days
                }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    async def cleanup_old_data(self, days: int = 90):
        """Limpiar datos antiguos"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            with self.get_session() as session:
                # Limpiar logs de API antiguos
                old_api_logs = session.query(APILog).filter(
                    APILog.created_at < cutoff_date
                ).delete()
                
                # Limpiar sesiones inactivas
                old_sessions = session.query(UserSession).filter(
                    UserSession.last_activity < cutoff_date
                ).delete()
                
                # Limpiar entradas de cache expiradas
                old_cache = session.query(CacheEntry).filter(
                    CacheEntry.expires_at < datetime.utcnow()
                ).delete()
                
                session.commit()
                
                logger.info(f"Cleaned up {old_api_logs} API logs, {old_sessions} sessions, {old_cache} cache entries")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    async def close(self):
        """Cerrar conexiones de base de datos"""
        try:
            if self.engine:
                self.engine.dispose()
            if self.async_engine:
                await self.async_engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

async def get_global_db_manager() -> DatabaseManager:
    """Obtener la instancia global del gestor de base de datos"""
    global _db_manager
    if _db_manager is None:
        from ..config import get_config
        config = get_config()
        
        _db_manager = DatabaseManager(
            database_url=config.database.url,
            echo=config.database.echo
        )
        await _db_manager.initialize()
    
    return _db_manager

























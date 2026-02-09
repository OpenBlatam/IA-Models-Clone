"""
Sistema de A/B Testing para contenido
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from sqlalchemy import Column, String, Text, DateTime, JSON, Boolean, Integer, Float
from sqlalchemy.orm import Session

from ..db.base import Base, get_db_session

logger = logging.getLogger(__name__)


class ABTestModel(Base):
    """Modelo de A/B test en BD"""
    __tablename__ = "ab_tests"
    
    id = Column(String(64), primary_key=True, index=True)
    identity_profile_id = Column(String(64), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    variants = Column(JSON, nullable=False)  # Lista de variantes
    traffic_split = Column(JSON, nullable=False)  # Distribución de tráfico
    status = Column(String(20), default="draft", nullable=False, index=True)  # draft, running, completed
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class ABTestResultModel(Base):
    """Modelo de resultados de A/B test"""
    __tablename__ = "ab_test_results"
    
    id = Column(String(64), primary_key=True, index=True)
    test_id = Column(String(64), nullable=False, index=True)
    variant = Column(String(50), nullable=False, index=True)
    content_id = Column(String(64), nullable=False, index=True)
    views = Column(Integer, default=0, nullable=False)
    likes = Column(Integer, default=0, nullable=False)
    comments = Column(Integer, default=0, nullable=False)
    shares = Column(Integer, default=0, nullable=False)
    engagement_rate = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


@dataclass
class ABTestVariant:
    """Variante de A/B test"""
    variant_id: str
    name: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTest:
    """A/B Test"""
    test_id: str
    identity_profile_id: str
    name: str
    description: Optional[str]
    variants: List[ABTestVariant]
    traffic_split: Dict[str, float]  # variant_id -> percentage
    status: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]


class ABTestService:
    """Servicio de A/B testing"""
    
    def __init__(self):
        self._init_table()
    
    def _init_table(self):
        """Inicializa tablas de A/B testing"""
        from ..db.base import init_db
        init_db()
    
    def create_test(
        self,
        identity_profile_id: str,
        name: str,
        variants: List[Dict[str, Any]],
        traffic_split: Optional[Dict[str, float]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Crea un A/B test
        
        Args:
            identity_profile_id: ID de la identidad
            name: Nombre del test
            variants: Lista de variantes
            traffic_split: Distribución de tráfico (default: 50/50)
            description: Descripción
            
        Returns:
            ID del test
        """
        test_id = str(uuid.uuid4())
        
        # Validar traffic_split
        if not traffic_split:
            # Distribución equitativa
            num_variants = len(variants)
            traffic_split = {
                variant["variant_id"]: 100.0 / num_variants
                for variant in variants
            }
        
        # Validar que suma 100%
        total = sum(traffic_split.values())
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Traffic split debe sumar 100%, actual: {total}%")
        
        with get_db_session() as db:
            test = ABTestModel(
                id=test_id,
                identity_profile_id=identity_profile_id,
                name=name,
                description=description,
                variants=variants,
                traffic_split=traffic_split,
                status="draft"
            )
            db.add(test)
            db.commit()
        
        logger.info(f"A/B test creado: {test_id} ({name})")
        return test_id
    
    def start_test(self, test_id: str) -> bool:
        """Inicia un A/B test"""
        with get_db_session() as db:
            test = db.query(ABTestModel).filter_by(id=test_id).first()
            if not test:
                return False
            
            test.status = "running"
            test.start_date = datetime.utcnow()
            db.commit()
        
        logger.info(f"A/B test iniciado: {test_id}")
        return True
    
    def stop_test(self, test_id: str) -> bool:
        """Detiene un A/B test"""
        with get_db_session() as db:
            test = db.query(ABTestModel).filter_by(id=test_id).first()
            if not test:
                return False
            
            test.status = "completed"
            test.end_date = datetime.utcnow()
            db.commit()
        
        logger.info(f"A/B test detenido: {test_id}")
        return True
    
    def get_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene un test"""
        with get_db_session() as db:
            test = db.query(ABTestModel).filter_by(id=test_id).first()
            if not test:
                return None
            
            # Obtener resultados
            results = db.query(ABTestResultModel).filter_by(test_id=test_id).all()
            
            return {
                "test_id": test.id,
                "identity_profile_id": test.identity_profile_id,
                "name": test.name,
                "description": test.description,
                "variants": test.variants,
                "traffic_split": test.traffic_split,
                "status": test.status,
                "start_date": test.start_date.isoformat() if test.start_date else None,
                "end_date": test.end_date.isoformat() if test.end_date else None,
                "results": [
                    {
                        "variant": r.variant,
                        "views": r.views,
                        "likes": r.likes,
                        "comments": r.comments,
                        "shares": r.shares,
                        "engagement_rate": r.engagement_rate
                    }
                    for r in results
                ]
            }
    
    def record_result(
        self,
        test_id: str,
        variant: str,
        content_id: str,
        views: int = 0,
        likes: int = 0,
        comments: int = 0,
        shares: int = 0
    ):
        """Registra resultado de una variante"""
        engagement_rate = (likes + comments + shares) / max(views, 1) * 100
        
        with get_db_session() as db:
            result = ABTestResultModel(
                id=str(uuid.uuid4()),
                test_id=test_id,
                variant=variant,
                content_id=content_id,
                views=views,
                likes=likes,
                comments=comments,
                shares=shares,
                engagement_rate=engagement_rate
            )
            db.add(result)
            db.commit()
    
    def get_winner(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene la variante ganadora"""
        with get_db_session() as db:
            results = db.query(ABTestResultModel).filter_by(test_id=test_id).all()
            
            if not results:
                return None
            
            # Agregar resultados por variante
            variant_stats = {}
            for result in results:
                if result.variant not in variant_stats:
                    variant_stats[result.variant] = {
                        "views": 0,
                        "likes": 0,
                        "comments": 0,
                        "shares": 0,
                        "engagement_rate": 0.0,
                        "count": 0
                    }
                
                stats = variant_stats[result.variant]
                stats["views"] += result.views
                stats["likes"] += result.likes
                stats["comments"] += result.comments
                stats["shares"] += result.shares
                stats["engagement_rate"] += result.engagement_rate or 0
                stats["count"] += 1
            
            # Calcular promedio de engagement rate
            for variant, stats in variant_stats.items():
                if stats["count"] > 0:
                    stats["engagement_rate"] = stats["engagement_rate"] / stats["count"]
            
            # Encontrar ganador (mayor engagement rate)
            winner = max(
                variant_stats.items(),
                key=lambda x: x[1]["engagement_rate"]
            )
            
            return {
                "variant": winner[0],
                "stats": winner[1],
                "all_stats": variant_stats
            }


# Singleton global
_ab_test_service: Optional[ABTestService] = None


def get_ab_test_service() -> ABTestService:
    """Obtiene instancia singleton del servicio de A/B testing"""
    global _ab_test_service
    if _ab_test_service is None:
        _ab_test_service = ABTestService()
    return _ab_test_service





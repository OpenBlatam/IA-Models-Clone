"""
Modelos de base de datos para el sistema de recuperación
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Optional

Base = declarative_base()


class User(Base):
    """Modelo de usuario"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=True)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    profiles = relationship("UserProfile", back_populates="user", cascade="all, delete-orphan")
    entries = relationship("DailyEntry", back_populates="user", cascade="all, delete-orphan")
    plans = relationship("RecoveryPlan", back_populates="user", cascade="all, delete-orphan")


class UserProfile(Base):
    """Perfil de usuario con información de adicción"""
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    addiction_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    frequency = Column(String, nullable=False)
    duration_years = Column(Float, nullable=True)
    daily_cost = Column(Float, nullable=True)
    triggers = Column(JSON, default=list)
    motivations = Column(JSON, default=list)
    previous_attempts = Column(Integer, default=0)
    support_system = Column(Boolean, default=False)
    medical_conditions = Column(JSON, default=list)
    additional_info = Column(Text, nullable=True)
    start_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    user = relationship("User", back_populates="profiles")


class RecoveryPlan(Base):
    """Plan de recuperación del usuario"""
    __tablename__ = "recovery_plans"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    addiction_type = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=False)
    approach = Column(String, nullable=False)  # abstinencia_total, reduccion_gradual
    goals = Column(JSON, default=list)
    milestones = Column(JSON, default=list)
    strategies = Column(JSON, default=list)
    daily_tasks = Column(JSON, default=list)
    weekly_tasks = Column(JSON, default=list)
    support_resources = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    user = relationship("User", back_populates="plans")


class DailyEntry(Base):
    """Entrada diaria de seguimiento"""
    __tablename__ = "daily_entries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    mood = Column(String, nullable=False)
    cravings_level = Column(Integer, nullable=False)  # 1-10
    triggers_encountered = Column(JSON, default=list)
    consumed = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    user = relationship("User", back_populates="entries")


class Milestone(Base):
    """Hitos alcanzados por el usuario"""
    __tablename__ = "milestones"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    milestone_days = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    achieved_at = Column(DateTime, default=datetime.utcnow)
    celebrated = Column(Boolean, default=False)


class RelapseRiskCheck(Base):
    """Registro de evaluaciones de riesgo de recaída"""
    __tablename__ = "relapse_risk_checks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    warning_signs = Column(JSON, default=list)
    recommendations = Column(JSON, default=list)
    checked_at = Column(DateTime, default=datetime.utcnow)


class CoachingSession(Base):
    """Sesiones de coaching"""
    __tablename__ = "coaching_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    topic = Column(String, nullable=False)
    current_situation = Column(Text, nullable=False)
    guidance = Column(Text, nullable=True)
    action_items = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Gestor de base de datos"""
    
    def __init__(self, database_url: str = "sqlite:///./recovery.db"):
        """
        Inicializa el gestor de base de datos
        
        Args:
            database_url: URL de la base de datos
        """
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Obtiene una sesión de base de datos"""
        return self.SessionLocal()
    
    def create_user(self, user_id: str, email: Optional[str] = None, name: Optional[str] = None):
        """Crea un nuevo usuario"""
        session = self.get_session()
        try:
            user = User(id=user_id, email=email, name=name)
            session.add(user)
            session.commit()
            return user
        finally:
            session.close()
    
    def get_user(self, user_id: str):
        """Obtiene un usuario por ID"""
        session = self.get_session()
        try:
            return session.query(User).filter(User.id == user_id).first()
        finally:
            session.close()


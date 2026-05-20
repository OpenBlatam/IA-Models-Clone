"""
Motor de Personalización AI
===========================

Motor para personalización inteligente, recomendaciones adaptativas y experiencia de usuario personalizada.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from pathlib import Path
import hashlib
import numpy as np
from collections import defaultdict, deque
import random
import math
import sqlite3
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import redis
import elasticsearch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
import torch
import time
import networkx as nx
from scipy import stats
import base64
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class PersonalizationType(str, Enum):
    """Tipos de personalización"""
    CONTENT_RECOMMENDATION = "content_recommendation"
    INTERFACE_CUSTOMIZATION = "interface_customization"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    NOTIFICATION_PREFERENCES = "notification_preferences"
    DOCUMENT_TEMPLATES = "document_templates"
    LANGUAGE_PREFERENCES = "language_preferences"
    ACCESSIBILITY_FEATURES = "accessibility_features"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"

class UserBehaviorType(str, Enum):
    """Tipos de comportamiento de usuario"""
    DOCUMENT_PROCESSING = "document_processing"
    FEATURE_USAGE = "feature_usage"
    NAVIGATION_PATTERNS = "navigation_patterns"
    SEARCH_QUERIES = "search_queries"
    COLLABORATION_ACTIVITY = "collaboration_activity"
    LEARNING_PREFERENCES = "learning_preferences"
    INNOVATION_ENGAGEMENT = "innovation_engagement"
    AUTOMATION_USAGE = "automation_usage"

class RecommendationType(str, Enum):
    """Tipos de recomendaciones"""
    DOCUMENT_TEMPLATES = "document_templates"
    FEATURES = "features"
    WORKFLOWS = "workflows"
    AUTOMATIONS = "automations"
    COLLABORATORS = "collaborators"
    INTEGRATIONS = "integrations"
    LEARNING_RESOURCES = "learning_resources"
    INNOVATION_OPPORTUNITIES = "innovation_opportunities"

@dataclass
class UserProfile:
    """Perfil de usuario personalizado"""
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    behavior_patterns: Dict[str, Any] = field(default_factory=dict)
    skill_level: str = "beginner"  # beginner, intermediate, advanced, expert
    interests: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    learning_style: str = "visual"  # visual, auditory, kinesthetic, reading
    work_style: str = "collaborative"  # collaborative, independent, mixed
    time_preferences: Dict[str, Any] = field(default_factory=dict)
    device_preferences: Dict[str, Any] = field(default_factory=dict)
    accessibility_needs: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class UserBehavior:
    """Comportamiento de usuario"""
    user_id: str
    behavior_type: UserBehaviorType
    action: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0
    success: bool = True
    feedback: Optional[float] = None

@dataclass
class Recommendation:
    """Recomendación personalizada"""
    id: str
    user_id: str
    recommendation_type: RecommendationType
    title: str
    description: str
    content: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    relevance_score: float = 0.0
    priority: str = "medium"  # low, medium, high, critical
    category: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_dismissed: bool = False
    is_acted_upon: bool = False

@dataclass
class PersonalizationRule:
    """Regla de personalización"""
    id: str
    name: str
    description: str
    condition: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class PersonalizationInsight:
    """Insight de personalización"""
    id: str
    user_id: str
    insight_type: str
    title: str
    description: str
    confidence: float = 0.0
    actionable_items: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

class AIPersonalizationEngine:
    """Motor de personalización AI optimizado"""
    
    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.user_behaviors: List[UserBehavior] = []
        self.recommendations: Dict[str, List[Recommendation]] = defaultdict(list)
        self.personalization_rules: Dict[str, PersonalizationRule] = {}
        self.personalization_insights: List[PersonalizationInsight] = []
        
        # Configuración de personalización optimizada
        self.behavior_retention_days = 90
        self.recommendation_retention_days = 30
        self.insight_generation_interval = 3600  # 1 hora
        self.profile_update_interval = 1800  # 30 minutos
        
        # Workers de personalización
        self.personalization_workers: Dict[str, asyncio.Task] = {}
        self.personalization_active = False
        
        # Algoritmos de recomendación avanzados
        self.recommendation_algorithms = {
            "collaborative_filtering": self._collaborative_filtering,
            "content_based": self._content_based_filtering,
            "hybrid": self._hybrid_recommendation,
            "deep_learning": self._deep_learning_recommendation,
            "neural_collaborative": self._neural_collaborative_filtering,
            "matrix_factorization": self._matrix_factorization,
            "deep_fm": self._deep_fm_recommendation,
            "transformer_based": self._transformer_based_recommendation
        }
        
        # Componentes de optimización
        self.redis_client: Optional[redis.Redis] = None
        self.elasticsearch_client: Optional[elasticsearch.Elasticsearch] = None
        self.db_connection: Optional[sqlite3.Connection] = None
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.ml_models: Dict[str, Any] = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.user_embeddings: Dict[str, np.ndarray] = {}
        self.item_embeddings: Dict[str, np.ndarray] = {}
        
        # Cache de personalización
        self.personalization_cache: Dict[str, Any] = {}
        self.cache_lock = threading.Lock()
        
        # Métricas de rendimiento
        self.performance_metrics = {
            "recommendation_accuracy": 0.0,
            "response_time": 0.0,
            "cache_hit_rate": 0.0,
            "user_satisfaction": 0.0
        }
        
    async def initialize(self):
        """Inicializa el motor de personalización optimizado"""
        logger.info("Inicializando motor de personalización AI optimizado...")
        
        # Inicializar componentes de optimización
        await self._initialize_optimization_components()
        
        # Cargar datos existentes
        await self._load_personalization_data()
        
        # Cargar reglas de personalización
        await self._load_personalization_rules()
        
        # Inicializar modelos ML
        await self._initialize_ml_models()
        
        # Iniciar workers de personalización
        await self._start_personalization_workers()
        
        # Inicializar cache distribuido
        await self._initialize_distributed_cache()
        
        logger.info("Motor de personalización AI optimizado inicializado")
    
    async def _initialize_optimization_components(self):
        """Inicializa componentes de optimización"""
        try:
            # Inicializar Redis para cache distribuido
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("Redis conectado para cache distribuido")
            except Exception as e:
                logger.warning(f"No se pudo conectar a Redis: {e}")
            
            # Inicializar Elasticsearch para búsqueda avanzada
            try:
                self.elasticsearch_client = elasticsearch.Elasticsearch(
                    ['localhost:9200'],
                    timeout=30
                )
                if self.elasticsearch_client.ping():
                    logger.info("Elasticsearch conectado para búsqueda avanzada")
            except Exception as e:
                logger.warning(f"No se pudo conectar a Elasticsearch: {e}")
            
            # Inicializar base de datos SQLite
            self.db_connection = sqlite3.connect(
                'data/personalization.db',
                check_same_thread=False
            )
            await self._create_database_schema()
            logger.info("Base de datos SQLite inicializada")
            
        except Exception as e:
            logger.error(f"Error inicializando componentes de optimización: {e}")
    
    async def _create_database_schema(self):
        """Crea esquema de base de datos"""
        try:
            cursor = self.db_connection.cursor()
            
            # Tabla de perfiles de usuario
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT,
                    behavior_patterns TEXT,
                    skill_level TEXT,
                    interests TEXT,
                    goals TEXT,
                    constraints TEXT,
                    learning_style TEXT,
                    work_style TEXT,
                    time_preferences TEXT,
                    device_preferences TEXT,
                    accessibility_needs TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            # Tabla de comportamientos
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_behaviors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    behavior_type TEXT,
                    action TEXT,
                    context TEXT,
                    timestamp TEXT,
                    duration REAL,
                    success BOOLEAN,
                    feedback REAL
                )
            ''')
            
            # Tabla de recomendaciones
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommendations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    recommendation_type TEXT,
                    title TEXT,
                    description TEXT,
                    content TEXT,
                    confidence_score REAL,
                    relevance_score REAL,
                    priority TEXT,
                    category TEXT,
                    tags TEXT,
                    created_at TEXT,
                    expires_at TEXT,
                    is_dismissed BOOLEAN,
                    is_acted_upon BOOLEAN
                )
            ''')
            
            # Tabla de insights
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS personalization_insights (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    insight_type TEXT,
                    title TEXT,
                    description TEXT,
                    confidence REAL,
                    actionable_items TEXT,
                    created_at TEXT
                )
            ''')
            
            # Índices para optimización
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_behaviors_user_timestamp ON user_behaviors(user_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_recommendations_user ON recommendations(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_insights_user ON personalization_insights(user_id)')
            
            self.db_connection.commit()
            logger.info("Esquema de base de datos creado")
            
        except Exception as e:
            logger.error(f"Error creando esquema de base de datos: {e}")
    
    async def _initialize_ml_models(self):
        """Inicializa modelos de machine learning"""
        try:
            # Inicializar modelo de clustering para segmentación de usuarios
            self.ml_models['user_clustering'] = KMeans(n_clusters=5, random_state=42)
            
            # Inicializar modelo de recomendación con TensorFlow
            self.ml_models['neural_cf'] = self._create_neural_collaborative_filtering_model()
            
            # Inicializar modelo de análisis de sentimientos
            try:
                self.ml_models['sentiment_analyzer'] = AutoTokenizer.from_pretrained(
                    'nlptown/bert-base-multilingual-uncased-sentiment'
                )
                self.ml_models['sentiment_model'] = AutoModel.from_pretrained(
                    'nlptown/bert-base-multilingual-uncased-sentiment'
                )
                logger.info("Modelo de análisis de sentimientos cargado")
            except Exception as e:
                logger.warning(f"No se pudo cargar modelo de sentimientos: {e}")
            
            # Inicializar modelo de embeddings
            self.ml_models['embedding_model'] = self._create_embedding_model()
            
            logger.info("Modelos ML inicializados")
            
        except Exception as e:
            logger.error(f"Error inicializando modelos ML: {e}")
    
    def _create_neural_collaborative_filtering_model(self):
        """Crea modelo de filtrado colaborativo neural"""
        try:
            # Modelo simple de recomendación neural
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error creando modelo neural: {e}")
            return None
    
    def _create_embedding_model(self):
        """Crea modelo de embeddings"""
        try:
            # Modelo simple de embeddings
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(10000, 128, input_length=100),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu')
            ])
            
            return model
            
        except Exception as e:
            logger.error(f"Error creando modelo de embeddings: {e}")
            return None
    
    async def _initialize_distributed_cache(self):
        """Inicializa cache distribuido"""
        try:
            if self.redis_client:
                # Configurar cache de personalización
                await self.redis_client.set(
                    'personalization:cache:initialized',
                    'true',
                    ex=3600
                )
                logger.info("Cache distribuido inicializado")
            
        except Exception as e:
            logger.error(f"Error inicializando cache distribuido: {e}")
    
    async def _load_personalization_data(self):
        """Carga datos de personalización"""
        try:
            # Cargar perfiles de usuario
            profiles_file = Path("data/personalization_profiles.json")
            if profiles_file.exists():
                with open(profiles_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                
                for profile_data in profiles_data:
                    profile = UserProfile(
                        user_id=profile_data["user_id"],
                        preferences=profile_data["preferences"],
                        behavior_patterns=profile_data["behavior_patterns"],
                        skill_level=profile_data["skill_level"],
                        interests=profile_data["interests"],
                        goals=profile_data["goals"],
                        constraints=profile_data["constraints"],
                        learning_style=profile_data["learning_style"],
                        work_style=profile_data["work_style"],
                        time_preferences=profile_data["time_preferences"],
                        device_preferences=profile_data["device_preferences"],
                        accessibility_needs=profile_data["accessibility_needs"],
                        created_at=datetime.fromisoformat(profile_data["created_at"]),
                        updated_at=datetime.fromisoformat(profile_data["updated_at"])
                    )
                    self.user_profiles[profile.user_id] = profile
                
                logger.info(f"Cargados {len(self.user_profiles)} perfiles de usuario")
            
            # Cargar recomendaciones
            recommendations_file = Path("data/personalization_recommendations.json")
            if recommendations_file.exists():
                with open(recommendations_file, 'r', encoding='utf-8') as f:
                    recommendations_data = json.load(f)
                
                for rec_data in recommendations_data:
                    recommendation = Recommendation(
                        id=rec_data["id"],
                        user_id=rec_data["user_id"],
                        recommendation_type=RecommendationType(rec_data["recommendation_type"]),
                        title=rec_data["title"],
                        description=rec_data["description"],
                        content=rec_data["content"],
                        confidence_score=rec_data["confidence_score"],
                        relevance_score=rec_data["relevance_score"],
                        priority=rec_data["priority"],
                        category=rec_data["category"],
                        tags=rec_data["tags"],
                        created_at=datetime.fromisoformat(rec_data["created_at"]),
                        expires_at=datetime.fromisoformat(rec_data["expires_at"]) if rec_data.get("expires_at") else None,
                        is_dismissed=rec_data["is_dismissed"],
                        is_acted_upon=rec_data["is_acted_upon"]
                    )
                    self.recommendations[recommendation.user_id].append(recommendation)
                
                logger.info(f"Cargadas {sum(len(recs) for recs in self.recommendations.values())} recomendaciones")
            
        except Exception as e:
            logger.error(f"Error cargando datos de personalización: {e}")
    
    async def _load_personalization_rules(self):
        """Carga reglas de personalización"""
        try:
            # Reglas de personalización predefinidas
            default_rules = [
                {
                    "id": "beginner_guidance",
                    "name": "Guía para Principiantes",
                    "description": "Proporcionar guías y tutoriales para usuarios principiantes",
                    "condition": {"skill_level": "beginner", "feature_usage_count": {"<": 10}},
                    "action": {"recommend": "learning_resources", "priority": "high"},
                    "priority": 10
                },
                {
                    "id": "power_user_optimization",
                    "name": "Optimización para Usuarios Avanzados",
                    "description": "Sugerir características avanzadas para usuarios expertos",
                    "condition": {"skill_level": "expert", "feature_usage_count": {">": 100}},
                    "action": {"recommend": "advanced_features", "priority": "medium"},
                    "priority": 8
                },
                {
                    "id": "collaboration_enhancement",
                    "name": "Mejora de Colaboración",
                    "description": "Sugerir herramientas de colaboración para usuarios colaborativos",
                    "condition": {"work_style": "collaborative", "collaboration_frequency": {">": 5}},
                    "action": {"recommend": "collaboration_tools", "priority": "medium"},
                    "priority": 7
                },
                {
                    "id": "accessibility_support",
                    "name": "Soporte de Accesibilidad",
                    "description": "Proporcionar características de accesibilidad según necesidades",
                    "condition": {"accessibility_needs": {"!=": []}},
                    "action": {"recommend": "accessibility_features", "priority": "high"},
                    "priority": 9
                },
                {
                    "id": "time_optimization",
                    "name": "Optimización de Tiempo",
                    "description": "Sugerir automatizaciones para usuarios con poco tiempo",
                    "condition": {"time_preferences.available_hours": {"<": 4}},
                    "action": {"recommend": "automation_tools", "priority": "high"},
                    "priority": 8
                }
            ]
            
            for rule_data in default_rules:
                rule = PersonalizationRule(
                    id=rule_data["id"],
                    name=rule_data["name"],
                    description=rule_data["description"],
                    condition=rule_data["condition"],
                    action=rule_data["action"],
                    priority=rule_data["priority"]
                )
                self.personalization_rules[rule.id] = rule
            
            logger.info(f"Cargadas {len(self.personalization_rules)} reglas de personalización")
            
        except Exception as e:
            logger.error(f"Error cargando reglas de personalización: {e}")
    
    async def _start_personalization_workers(self):
        """Inicia workers de personalización"""
        try:
            self.personalization_active = True
            
            # Worker de análisis de comportamiento
            asyncio.create_task(self._behavior_analysis_worker())
            
            # Worker de generación de recomendaciones
            asyncio.create_task(self._recommendation_generation_worker())
            
            # Worker de actualización de perfiles
            asyncio.create_task(self._profile_update_worker())
            
            # Worker de generación de insights
            asyncio.create_task(self._insight_generation_worker())
            
            logger.info("Workers de personalización iniciados")
            
        except Exception as e:
            logger.error(f"Error iniciando workers de personalización: {e}")
    
    async def _behavior_analysis_worker(self):
        """Worker de análisis de comportamiento"""
        while self.personalization_active:
            try:
                await asyncio.sleep(300)  # Cada 5 minutos
                
                # Analizar comportamientos recientes
                await self._analyze_recent_behaviors()
                
                # Actualizar patrones de comportamiento
                await self._update_behavior_patterns()
                
            except Exception as e:
                logger.error(f"Error en worker de análisis de comportamiento: {e}")
                await asyncio.sleep(60)
    
    async def _recommendation_generation_worker(self):
        """Worker de generación de recomendaciones"""
        while self.personalization_active:
            try:
                await asyncio.sleep(1800)  # Cada 30 minutos
                
                # Generar recomendaciones para usuarios activos
                await self._generate_recommendations_for_active_users()
                
                # Limpiar recomendaciones expiradas
                await self._cleanup_expired_recommendations()
                
            except Exception as e:
                logger.error(f"Error en worker de generación de recomendaciones: {e}")
                await asyncio.sleep(300)
    
    async def _profile_update_worker(self):
        """Worker de actualización de perfiles"""
        while self.personalization_active:
            try:
                await asyncio.sleep(self.profile_update_interval)
                
                # Actualizar perfiles basados en comportamiento
                await self._update_user_profiles()
                
            except Exception as e:
                logger.error(f"Error en worker de actualización de perfiles: {e}")
                await asyncio.sleep(300)
    
    async def _insight_generation_worker(self):
        """Worker de generación de insights"""
        while self.personalization_active:
            try:
                await asyncio.sleep(self.insight_generation_interval)
                
                # Generar insights de personalización
                await self._generate_personalization_insights()
                
            except Exception as e:
                logger.error(f"Error en worker de generación de insights: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_recent_behaviors(self):
        """Analiza comportamientos recientes"""
        try:
            # Obtener comportamientos de la última hora
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_behaviors = [
                behavior for behavior in self.user_behaviors
                if behavior.timestamp > cutoff_time
            ]
            
            # Agrupar por usuario
            user_behaviors = defaultdict(list)
            for behavior in recent_behaviors:
                user_behaviors[behavior.user_id].append(behavior)
            
            # Analizar patrones por usuario
            for user_id, behaviors in user_behaviors.items():
                await self._analyze_user_behavior_patterns(user_id, behaviors)
            
        except Exception as e:
            logger.error(f"Error analizando comportamientos recientes: {e}")
    
    async def _analyze_user_behavior_patterns(self, user_id: str, behaviors: List[UserBehavior]):
        """Analiza patrones de comportamiento de usuario"""
        try:
            if user_id not in self.user_profiles:
                return
            
            profile = self.user_profiles[user_id]
            
            # Analizar tipos de comportamiento
            behavior_types = defaultdict(int)
            for behavior in behaviors:
                behavior_types[behavior.behavior_type.value] += 1
            
            # Actualizar patrones de comportamiento
            profile.behavior_patterns["recent_activity"] = dict(behavior_types)
            profile.behavior_patterns["last_activity"] = max(behavior.timestamp for behavior in behaviors).isoformat()
            
            # Analizar eficiencia
            successful_actions = sum(1 for b in behaviors if b.success)
            total_actions = len(behaviors)
            efficiency = successful_actions / total_actions if total_actions > 0 else 0
            
            profile.behavior_patterns["efficiency"] = efficiency
            
            # Analizar duración promedio
            avg_duration = np.mean([b.duration for b in behaviors if b.duration > 0])
            profile.behavior_patterns["avg_duration"] = avg_duration
            
        except Exception as e:
            logger.error(f"Error analizando patrones de comportamiento: {e}")
    
    async def _update_behavior_patterns(self):
        """Actualiza patrones de comportamiento"""
        try:
            for user_id, profile in self.user_profiles.items():
                # Obtener comportamientos del usuario
                user_behaviors = [
                    behavior for behavior in self.user_behaviors
                    if behavior.user_id == user_id
                ]
                
                if not user_behaviors:
                    continue
                
                # Analizar patrones a largo plazo
                await self._analyze_long_term_patterns(user_id, user_behaviors)
                
        except Exception as e:
            logger.error(f"Error actualizando patrones de comportamiento: {e}")
    
    async def _analyze_long_term_patterns(self, user_id: str, behaviors: List[UserBehavior]):
        """Analiza patrones a largo plazo"""
        try:
            profile = self.user_profiles[user_id]
            
            # Analizar frecuencia de uso
            usage_frequency = defaultdict(int)
            for behavior in behaviors:
                usage_frequency[behavior.action] += 1
            
            profile.behavior_patterns["usage_frequency"] = dict(usage_frequency)
            
            # Analizar horarios de uso
            hour_usage = defaultdict(int)
            for behavior in behaviors:
                hour = behavior.timestamp.hour
                hour_usage[hour] += 1
            
            profile.behavior_patterns["hour_usage"] = dict(hour_usage)
            
            # Determinar horario preferido
            if hour_usage:
                preferred_hour = max(hour_usage, key=hour_usage.get)
                profile.time_preferences["preferred_hour"] = preferred_hour
            
            # Analizar días de la semana
            weekday_usage = defaultdict(int)
            for behavior in behaviors:
                weekday = behavior.timestamp.weekday()
                weekday_usage[weekday] += 1
            
            profile.behavior_patterns["weekday_usage"] = dict(weekday_usage)
            
        except Exception as e:
            logger.error(f"Error analizando patrones a largo plazo: {e}")
    
    async def _generate_recommendations_for_active_users(self):
        """Genera recomendaciones para usuarios activos"""
        try:
            # Obtener usuarios activos (últimas 24 horas)
            cutoff_time = datetime.now() - timedelta(hours=24)
            active_users = set()
            
            for behavior in self.user_behaviors:
                if behavior.timestamp > cutoff_time:
                    active_users.add(behavior.user_id)
            
            # Generar recomendaciones para cada usuario activo
            for user_id in active_users:
                await self._generate_user_recommendations(user_id)
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones para usuarios activos: {e}")
    
    async def _generate_user_recommendations(self, user_id: str):
        """Genera recomendaciones para un usuario específico"""
        try:
            if user_id not in self.user_profiles:
                return
            
            profile = self.user_profiles[user_id]
            
            # Aplicar reglas de personalización
            for rule in self.personalization_rules.values():
                if not rule.is_active:
                    continue
                
                if await self._evaluate_rule_condition(rule.condition, profile):
                    await self._execute_rule_action(rule.action, user_id, profile)
            
            # Generar recomendaciones basadas en algoritmos
            await self._generate_algorithmic_recommendations(user_id, profile)
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones para usuario: {e}")
    
    async def _evaluate_rule_condition(self, condition: Dict[str, Any], profile: UserProfile) -> bool:
        """Evalúa condición de regla"""
        try:
            for key, value in condition.items():
                if key == "skill_level":
                    if profile.skill_level != value:
                        return False
                elif key == "work_style":
                    if profile.work_style != value:
                        return False
                elif key == "accessibility_needs":
                    if value == [] and profile.accessibility_needs:
                        return False
                    elif value != [] and not profile.accessibility_needs:
                        return False
                elif key == "feature_usage_count":
                    # Simular evaluación de uso de características
                    usage_count = len(profile.behavior_patterns.get("usage_frequency", {}))
                    if "<" in value:
                        if usage_count >= value["<"]:
                            return False
                    elif ">" in value:
                        if usage_count <= value[">"]:
                            return False
                elif key == "collaboration_frequency":
                    # Simular evaluación de frecuencia de colaboración
                    collab_count = profile.behavior_patterns.get("usage_frequency", {}).get("collaboration", 0)
                    if ">" in value:
                        if collab_count <= value[">"]:
                            return False
                elif key == "time_preferences.available_hours":
                    # Simular evaluación de tiempo disponible
                    available_hours = profile.time_preferences.get("available_hours", 8)
                    if "<" in value:
                        if available_hours >= value["<"]:
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluando condición de regla: {e}")
            return False
    
    async def _execute_rule_action(self, action: Dict[str, Any], user_id: str, profile: UserProfile):
        """Ejecuta acción de regla"""
        try:
            if "recommend" in action:
                recommendation_type = action["recommend"]
                priority = action.get("priority", "medium")
                
                # Generar recomendación basada en tipo
                await self._create_rule_based_recommendation(user_id, recommendation_type, priority, profile)
            
        except Exception as e:
            logger.error(f"Error ejecutando acción de regla: {e}")
    
    async def _create_rule_based_recommendation(self, user_id: str, recommendation_type: str, priority: str, profile: UserProfile):
        """Crea recomendación basada en regla"""
        try:
            recommendation_id = f"rec_{uuid.uuid4().hex[:8]}"
            
            # Generar contenido de recomendación basado en tipo
            if recommendation_type == "learning_resources":
                title = "Recursos de Aprendizaje Recomendados"
                description = "Basado en tu nivel de experiencia, te recomendamos estos recursos para mejorar tus habilidades."
                content = {
                    "type": "learning_resources",
                    "resources": [
                        "Tutorial de procesamiento de documentos",
                        "Guía de mejores prácticas",
                        "Casos de uso avanzados"
                    ]
                }
            elif recommendation_type == "advanced_features":
                title = "Características Avanzadas"
                description = "Como usuario experto, podrías beneficiarte de estas características avanzadas."
                content = {
                    "type": "advanced_features",
                    "features": [
                        "Automatización personalizada",
                        "Integración con APIs externas",
                        "Análisis avanzado de datos"
                    ]
                }
            elif recommendation_type == "collaboration_tools":
                title = "Herramientas de Colaboración"
                description = "Mejora tu experiencia colaborativa con estas herramientas."
                content = {
                    "type": "collaboration_tools",
                    "tools": [
                        "Espacios de trabajo compartidos",
                        "Comentarios en tiempo real",
                        "Gestión de tareas colaborativas"
                    ]
                }
            elif recommendation_type == "accessibility_features":
                title = "Características de Accesibilidad"
                description = "Personaliza la interfaz según tus necesidades de accesibilidad."
                content = {
                    "type": "accessibility_features",
                    "features": [
                        "Alto contraste",
                        "Texto más grande",
                        "Navegación por teclado"
                    ]
                }
            elif recommendation_type == "automation_tools":
                title = "Herramientas de Automatización"
                description = "Ahorra tiempo con estas herramientas de automatización."
                content = {
                    "type": "automation_tools",
                    "tools": [
                        "Procesamiento automático de documentos",
                        "Workflows personalizados",
                        "Notificaciones automáticas"
                    ]
                }
            else:
                return
            
            recommendation = Recommendation(
                id=recommendation_id,
                user_id=user_id,
                recommendation_type=RecommendationType.CONTENT_RECOMMENDATION,
                title=title,
                description=description,
                content=content,
                confidence_score=0.8,
                relevance_score=0.7,
                priority=priority,
                category=recommendation_type,
                expires_at=datetime.now() + timedelta(days=7)
            )
            
            self.recommendations[user_id].append(recommendation)
            
        except Exception as e:
            logger.error(f"Error creando recomendación basada en regla: {e}")
    
    async def _generate_algorithmic_recommendations(self, user_id: str, profile: UserProfile):
        """Genera recomendaciones algorítmicas"""
        try:
            # Usar algoritmo híbrido por defecto
            recommendations = await self._hybrid_recommendation(user_id, profile)
            
            # Agregar recomendaciones generadas
            for rec_data in recommendations:
                recommendation = Recommendation(
                    id=f"rec_{uuid.uuid4().hex[:8]}",
                    user_id=user_id,
                    recommendation_type=RecommendationType(rec_data["type"]),
                    title=rec_data["title"],
                    description=rec_data["description"],
                    content=rec_data["content"],
                    confidence_score=rec_data["confidence"],
                    relevance_score=rec_data["relevance"],
                    priority=rec_data["priority"],
                    category=rec_data["category"],
                    expires_at=datetime.now() + timedelta(days=14)
                )
                
                self.recommendations[user_id].append(recommendation)
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones algorítmicas: {e}")
    
    async def _hybrid_recommendation(self, user_id: str, profile: UserProfile) -> List[Dict[str, Any]]:
        """Algoritmo de recomendación híbrido"""
        try:
            recommendations = []
            
            # Combinar filtrado colaborativo y basado en contenido
            collaborative_recs = await self._collaborative_filtering(user_id, profile)
            content_recs = await self._content_based_filtering(user_id, profile)
            
            # Combinar y puntuar recomendaciones
            all_recs = collaborative_recs + content_recs
            
            # Eliminar duplicados y puntuar
            unique_recs = {}
            for rec in all_recs:
                key = f"{rec['type']}_{rec['title']}"
                if key not in unique_recs:
                    unique_recs[key] = rec
                else:
                    # Promediar puntuaciones
                    unique_recs[key]["confidence"] = (unique_recs[key]["confidence"] + rec["confidence"]) / 2
                    unique_recs[key]["relevance"] = (unique_recs[key]["relevance"] + rec["relevance"]) / 2
            
            # Ordenar por puntuación combinada
            sorted_recs = sorted(
                unique_recs.values(),
                key=lambda x: (x["confidence"] + x["relevance"]) / 2,
                reverse=True
            )
            
            return sorted_recs[:5]  # Top 5 recomendaciones
            
        except Exception as e:
            logger.error(f"Error en recomendación híbrida: {e}")
            return []
    
    async def _collaborative_filtering(self, user_id: str, profile: UserProfile) -> List[Dict[str, Any]]:
        """Filtrado colaborativo"""
        try:
            recommendations = []
            
            # Simular filtrado colaborativo
            # En implementación real, encontrar usuarios similares
            similar_users = await self._find_similar_users(user_id, profile)
            
            for similar_user in similar_users:
                # Obtener características usadas por usuarios similares
                user_profile = self.user_profiles.get(similar_user)
                if user_profile:
                    # Simular recomendaciones basadas en usuarios similares
                    recommendations.append({
                        "type": "features",
                        "title": f"Característica popular entre usuarios similares",
                        "description": "Esta característica es popular entre usuarios con perfil similar al tuyo.",
                        "content": {"feature": "advanced_analytics"},
                        "confidence": 0.7,
                        "relevance": 0.6,
                        "priority": "medium",
                        "category": "collaborative"
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error en filtrado colaborativo: {e}")
            return []
    
    async def _content_based_filtering(self, user_id: str, profile: UserProfile) -> List[Dict[str, Any]]:
        """Filtrado basado en contenido"""
        try:
            recommendations = []
            
            # Recomendaciones basadas en intereses del usuario
            for interest in profile.interests:
                recommendations.append({
                    "type": "document_templates",
                    "title": f"Plantilla para {interest}",
                    "description": f"Plantilla personalizada para documentos relacionados con {interest}.",
                    "content": {"template": f"{interest}_template", "category": interest},
                    "confidence": 0.8,
                    "relevance": 0.9,
                    "priority": "high",
                    "category": "content_based"
                })
            
            # Recomendaciones basadas en nivel de habilidad
            if profile.skill_level == "beginner":
                recommendations.append({
                    "type": "learning_resources",
                    "title": "Tutoriales para Principiantes",
                    "description": "Recursos de aprendizaje diseñados para usuarios principiantes.",
                    "content": {"resources": ["tutorial_basics", "getting_started_guide"]},
                    "confidence": 0.9,
                    "relevance": 0.8,
                    "priority": "high",
                    "category": "skill_based"
                })
            elif profile.skill_level == "expert":
                recommendations.append({
                    "type": "advanced_features",
                    "title": "Características Avanzadas",
                    "description": "Herramientas avanzadas para usuarios expertos.",
                    "content": {"features": ["api_integration", "custom_workflows"]},
                    "confidence": 0.8,
                    "relevance": 0.9,
                    "priority": "medium",
                    "category": "skill_based"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error en filtrado basado en contenido: {e}")
            return []
    
    async def _deep_learning_recommendation(self, user_id: str, profile: UserProfile) -> List[Dict[str, Any]]:
        """Recomendación con deep learning"""
        try:
            # Simular recomendaciones con deep learning
            # En implementación real, usar modelo entrenado
            recommendations = []
            
            # Simular análisis de patrones complejos
            if profile.behavior_patterns.get("efficiency", 0) < 0.7:
                recommendations.append({
                    "type": "workflow_optimization",
                    "title": "Optimización de Workflow",
                    "description": "Mejora tu eficiencia con estos workflows optimizados.",
                    "content": {"workflows": ["efficient_processing", "batch_optimization"]},
                    "confidence": 0.85,
                    "relevance": 0.9,
                    "priority": "high",
                    "category": "deep_learning"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error en recomendación con deep learning: {e}")
            return []
    
    async def _find_similar_users(self, user_id: str, profile: UserProfile) -> List[str]:
        """Encuentra usuarios similares"""
        try:
            similar_users = []
            
            for other_user_id, other_profile in self.user_profiles.items():
                if other_user_id == user_id:
                    continue
                
                # Calcular similitud
                similarity = await self._calculate_user_similarity(profile, other_profile)
                
                if similarity > 0.7:  # Umbral de similitud
                    similar_users.append(other_user_id)
            
            return similar_users[:5]  # Top 5 usuarios similares
            
        except Exception as e:
            logger.error(f"Error encontrando usuarios similares: {e}")
            return []
    
    async def _calculate_user_similarity(self, profile1: UserProfile, profile2: UserProfile) -> float:
        """Calcula similitud entre usuarios"""
        try:
            similarity = 0.0
            
            # Similitud en nivel de habilidad
            if profile1.skill_level == profile2.skill_level:
                similarity += 0.3
            
            # Similitud en estilo de trabajo
            if profile1.work_style == profile2.work_style:
                similarity += 0.2
            
            # Similitud en intereses
            common_interests = set(profile1.interests) & set(profile2.interests)
            if profile1.interests or profile2.interests:
                interest_similarity = len(common_interests) / max(len(profile1.interests), len(profile2.interests))
                similarity += interest_similarity * 0.3
            
            # Similitud en patrones de comportamiento
            if profile1.behavior_patterns and profile2.behavior_patterns:
                behavior_similarity = await self._calculate_behavior_similarity(
                    profile1.behavior_patterns, profile2.behavior_patterns
                )
                similarity += behavior_similarity * 0.2
            
            return min(1.0, similarity)
            
        except Exception as e:
            logger.error(f"Error calculando similitud de usuarios: {e}")
            return 0.0
    
    async def _calculate_behavior_similarity(self, patterns1: Dict[str, Any], patterns2: Dict[str, Any]) -> float:
        """Calcula similitud de patrones de comportamiento"""
        try:
            # Similitud en frecuencia de uso
            freq1 = patterns1.get("usage_frequency", {})
            freq2 = patterns2.get("usage_frequency", {})
            
            all_actions = set(freq1.keys()) | set(freq2.keys())
            if not all_actions:
                return 0.0
            
            similarity = 0.0
            for action in all_actions:
                count1 = freq1.get(action, 0)
                count2 = freq2.get(action, 0)
                max_count = max(count1, count2)
                if max_count > 0:
                    similarity += min(count1, count2) / max_count
            
            return similarity / len(all_actions) if all_actions else 0.0
            
        except Exception as e:
            logger.error(f"Error calculando similitud de comportamiento: {e}")
            return 0.0
    
    async def _cleanup_expired_recommendations(self):
        """Limpia recomendaciones expiradas"""
        try:
            current_time = datetime.now()
            
            for user_id in list(self.recommendations.keys()):
                user_recommendations = self.recommendations[user_id]
                
                # Filtrar recomendaciones no expiradas
                valid_recommendations = [
                    rec for rec in user_recommendations
                    if not rec.expires_at or rec.expires_at > current_time
                ]
                
                self.recommendations[user_id] = valid_recommendations
                
                # Eliminar usuario si no tiene recomendaciones
                if not valid_recommendations:
                    del self.recommendations[user_id]
            
        except Exception as e:
            logger.error(f"Error limpiando recomendaciones expiradas: {e}")
    
    async def _update_user_profiles(self):
        """Actualiza perfiles de usuario"""
        try:
            for user_id, profile in self.user_profiles.items():
                # Actualizar nivel de habilidad basado en comportamiento
                await self._update_skill_level(user_id, profile)
                
                # Actualizar intereses basados en uso
                await self._update_interests(user_id, profile)
                
                # Actualizar timestamp
                profile.updated_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Error actualizando perfiles de usuario: {e}")
    
    async def _update_skill_level(self, user_id: str, profile: UserProfile):
        """Actualiza nivel de habilidad"""
        try:
            # Obtener comportamientos del usuario
            user_behaviors = [
                behavior for behavior in self.user_behaviors
                if behavior.user_id == user_id
            ]
            
            if not user_behaviors:
                return
            
            # Calcular métricas de habilidad
            total_actions = len(user_behaviors)
            successful_actions = sum(1 for b in user_behaviors if b.success)
            success_rate = successful_actions / total_actions if total_actions > 0 else 0
            
            avg_duration = np.mean([b.duration for b in user_behaviors if b.duration > 0])
            
            # Actualizar nivel de habilidad
            if success_rate > 0.9 and avg_duration < 30 and total_actions > 50:
                profile.skill_level = "expert"
            elif success_rate > 0.8 and avg_duration < 60 and total_actions > 20:
                profile.skill_level = "advanced"
            elif success_rate > 0.6 and total_actions > 10:
                profile.skill_level = "intermediate"
            else:
                profile.skill_level = "beginner"
            
        except Exception as e:
            logger.error(f"Error actualizando nivel de habilidad: {e}")
    
    async def _update_interests(self, user_id: str, profile: UserProfile):
        """Actualiza intereses del usuario"""
        try:
            # Obtener comportamientos del usuario
            user_behaviors = [
                behavior for behavior in self.user_behaviors
                if behavior.user_id == user_id
            ]
            
            if not user_behaviors:
                return
            
            # Analizar acciones para determinar intereses
            action_categories = defaultdict(int)
            for behavior in user_behaviors:
                action = behavior.action
                if "document" in action.lower():
                    action_categories["document_processing"] += 1
                elif "collaboration" in action.lower():
                    action_categories["collaboration"] += 1
                elif "automation" in action.lower():
                    action_categories["automation"] += 1
                elif "analysis" in action.lower():
                    action_categories["data_analysis"] += 1
                elif "translation" in action.lower():
                    action_categories["translation"] += 1
            
            # Actualizar intereses basados en categorías más usadas
            top_categories = sorted(action_categories.items(), key=lambda x: x[1], reverse=True)[:3]
            new_interests = [category for category, count in top_categories if count > 0]
            
            # Combinar con intereses existentes
            all_interests = list(set(profile.interests + new_interests))
            profile.interests = all_interests[:10]  # Limitar a 10 intereses
            
        except Exception as e:
            logger.error(f"Error actualizando intereses: {e}")
    
    async def _generate_personalization_insights(self):
        """Genera insights de personalización"""
        try:
            # Analizar patrones globales
            await self._analyze_global_patterns()
            
            # Generar insights por usuario
            for user_id, profile in self.user_profiles.items():
                await self._generate_user_insights(user_id, profile)
            
        except Exception as e:
            logger.error(f"Error generando insights de personalización: {e}")
    
    async def _analyze_global_patterns(self):
        """Analiza patrones globales"""
        try:
            # Analizar patrones de uso más comunes
            all_behaviors = self.user_behaviors
            if not all_behaviors:
                return
            
            # Patrones de uso por hora
            hour_patterns = defaultdict(int)
            for behavior in all_behaviors:
                hour_patterns[behavior.timestamp.hour] += 1
            
            # Patrones de uso por día de la semana
            weekday_patterns = defaultdict(int)
            for behavior in all_behaviors:
                weekday_patterns[behavior.timestamp.weekday()] += 1
            
            # Acciones más populares
            action_popularity = defaultdict(int)
            for behavior in all_behaviors:
                action_popularity[behavior.action] += 1
            
            # Crear insight global
            insight_id = f"insight_{uuid.uuid4().hex[:8]}"
            insight = PersonalizationInsight(
                id=insight_id,
                user_id="global",
                insight_type="global_patterns",
                title="Patrones de Uso Globales",
                description="Análisis de patrones de uso de todos los usuarios del sistema.",
                confidence=0.8,
                actionable_items=[
                    "Optimizar rendimiento en horas pico",
                    "Mejorar características más utilizadas",
                    "Personalizar interfaz según patrones de uso"
                ]
            )
            
            self.personalization_insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error analizando patrones globales: {e}")
    
    async def _generate_user_insights(self, user_id: str, profile: UserProfile):
        """Genera insights para usuario específico"""
        try:
            # Obtener comportamientos del usuario
            user_behaviors = [
                behavior for behavior in self.user_behaviors
                if behavior.user_id == user_id
            ]
            
            if len(user_behaviors) < 10:  # Mínimo de comportamientos para insights
                return
            
            # Analizar eficiencia
            efficiency = profile.behavior_patterns.get("efficiency", 0)
            if efficiency < 0.7:
                insight_id = f"insight_{uuid.uuid4().hex[:8]}"
                insight = PersonalizationInsight(
                    id=insight_id,
                    user_id=user_id,
                    insight_type="efficiency_improvement",
                    title="Oportunidad de Mejora de Eficiencia",
                    description=f"Tu tasa de éxito actual es del {efficiency*100:.1f}%. Te recomendamos revisar los tutoriales disponibles.",
                    confidence=0.9,
                    actionable_items=[
                        "Revisar tutoriales de uso",
                        "Explorar características de automatización",
                        "Contactar soporte para asistencia personalizada"
                    ]
                )
                self.personalization_insights.append(insight)
            
            # Analizar patrones de tiempo
            hour_usage = profile.behavior_patterns.get("hour_usage", {})
            if hour_usage:
                peak_hour = max(hour_usage, key=hour_usage.get)
                insight_id = f"insight_{uuid.uuid4().hex[:8]}"
                insight = PersonalizationInsight(
                    id=insight_id,
                    user_id=user_id,
                    insight_type="usage_pattern",
                    title="Patrón de Uso Identificado",
                    description=f"Tu hora de mayor actividad es las {peak_hour}:00. Considera programar tareas importantes en este horario.",
                    confidence=0.8,
                    actionable_items=[
                        f"Programar tareas importantes a las {peak_hour}:00",
                        "Configurar notificaciones para tu horario preferido",
                        "Optimizar workflows para tu patrón de uso"
                    ]
                )
                self.personalization_insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generando insights de usuario: {e}")
    
    async def track_user_behavior(
        self,
        user_id: str,
        behavior_type: UserBehaviorType,
        action: str,
        context: Dict[str, Any] = None,
        duration: float = 0.0,
        success: bool = True,
        feedback: float = None
    ):
        """Registra comportamiento de usuario"""
        try:
            behavior = UserBehavior(
                user_id=user_id,
                behavior_type=behavior_type,
                action=action,
                context=context or {},
                duration=duration,
                success=success,
                feedback=feedback
            )
            
            self.user_behaviors.append(behavior)
            
            # Limpiar comportamientos antiguos
            cutoff_date = datetime.now() - timedelta(days=self.behavior_retention_days)
            self.user_behaviors = [
                b for b in self.user_behaviors
                if b.timestamp > cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Error registrando comportamiento de usuario: {e}")
    
    async def get_user_recommendations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtiene recomendaciones para usuario"""
        try:
            user_recommendations = self.recommendations.get(user_id, [])
            
            # Filtrar recomendaciones no descartadas
            active_recommendations = [
                rec for rec in user_recommendations
                if not rec.is_dismissed
            ]
            
            # Ordenar por relevancia y confianza
            sorted_recommendations = sorted(
                active_recommendations,
                key=lambda x: (x.relevance_score + x.confidence_score) / 2,
                reverse=True
            )
            
            # Convertir a formato serializable
            recommendations_data = []
            for rec in sorted_recommendations[:limit]:
                recommendations_data.append({
                    "id": rec.id,
                    "title": rec.title,
                    "description": rec.description,
                    "type": rec.recommendation_type.value,
                    "content": rec.content,
                    "confidence_score": rec.confidence_score,
                    "relevance_score": rec.relevance_score,
                    "priority": rec.priority,
                    "category": rec.category,
                    "tags": rec.tags,
                    "created_at": rec.created_at.isoformat(),
                    "expires_at": rec.expires_at.isoformat() if rec.expires_at else None
                })
            
            return recommendations_data
            
        except Exception as e:
            logger.error(f"Error obteniendo recomendaciones de usuario: {e}")
            return []
    
    async def dismiss_recommendation(self, user_id: str, recommendation_id: str) -> bool:
        """Descarta recomendación"""
        try:
            user_recommendations = self.recommendations.get(user_id, [])
            
            for rec in user_recommendations:
                if rec.id == recommendation_id:
                    rec.is_dismissed = True
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error descartando recomendación: {e}")
            return False
    
    async def get_personalization_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos para dashboard de personalización"""
        try:
            # Estadísticas generales
            total_users = len(self.user_profiles)
            active_users = len(set(behavior.user_id for behavior in self.user_behaviors))
            total_recommendations = sum(len(recs) for recs in self.recommendations.values())
            total_insights = len(self.personalization_insights)
            
            # Distribución por nivel de habilidad
            skill_distribution = defaultdict(int)
            for profile in self.user_profiles.values():
                skill_distribution[profile.skill_level] += 1
            
            # Distribución por estilo de trabajo
            work_style_distribution = defaultdict(int)
            for profile in self.user_profiles.values():
                work_style_distribution[profile.work_style] += 1
            
            # Recomendaciones más populares
            recommendation_types = defaultdict(int)
            for user_recs in self.recommendations.values():
                for rec in user_recs:
                    recommendation_types[rec.recommendation_type.value] += 1
            
            # Insights recientes
            recent_insights = sorted(
                self.personalization_insights,
                key=lambda x: x.created_at,
                reverse=True
            )[:5]
            
            return {
                "total_users": total_users,
                "active_users": active_users,
                "total_recommendations": total_recommendations,
                "total_insights": total_insights,
                "skill_distribution": dict(skill_distribution),
                "work_style_distribution": dict(work_style_distribution),
                "recommendation_types": dict(recommendation_types),
                "recent_insights": [
                    {
                        "id": insight.id,
                        "user_id": insight.user_id,
                        "type": insight.insight_type,
                        "title": insight.title,
                        "confidence": insight.confidence,
                        "created_at": insight.created_at.isoformat()
                    }
                    for insight in recent_insights
                ],
                "personalization_active": self.personalization_active,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos del dashboard de personalización: {e}")
            return {"error": str(e)}
    
    async def _neural_collaborative_filtering(self, user_id: str, profile: UserProfile) -> List[Dict[str, Any]]:
        """Filtrado colaborativo neural avanzado"""
        try:
            recommendations = []
            
            # Obtener embeddings del usuario
            user_embedding = await self._get_user_embedding(user_id, profile)
            if user_embedding is None:
                return []
            
            # Encontrar usuarios similares usando embeddings
            similar_users = await self._find_similar_users_by_embedding(user_embedding)
            
            # Generar recomendaciones basadas en usuarios similares
            for similar_user_id in similar_users[:3]:
                user_recommendations = self.recommendations.get(similar_user_id, [])
                
                for rec in user_recommendations:
                    if not rec.is_dismissed and rec.confidence_score > 0.7:
                        recommendations.append({
                            "type": rec.recommendation_type.value,
                            "title": f"Recomendado por usuarios similares: {rec.title}",
                            "description": rec.description,
                            "content": rec.content,
                            "confidence": rec.confidence_score * 0.8,
                            "relevance": rec.relevance_score * 0.9,
                            "priority": "medium",
                            "category": "neural_collaborative"
                        })
            
            return recommendations[:5]
            
        except Exception as e:
            logger.error(f"Error en filtrado colaborativo neural: {e}")
            return []
    
    async def _matrix_factorization(self, user_id: str, profile: UserProfile) -> List[Dict[str, Any]]:
        """Factorización de matrices para recomendaciones"""
        try:
            recommendations = []
            
            # Crear matriz usuario-item
            user_item_matrix = await self._create_user_item_matrix()
            
            # Aplicar factorización de matrices (SVD)
            if user_item_matrix is not None and user_item_matrix.shape[0] > 1:
                from sklearn.decomposition import TruncatedSVD
                
                svd = TruncatedSVD(n_components=min(10, user_item_matrix.shape[1]-1))
                user_factors = svd.fit_transform(user_item_matrix)
                item_factors = svd.components_
                
                # Obtener factor del usuario
                user_index = await self._get_user_index(user_id)
                if user_index is not None and user_index < len(user_factors):
                    user_factor = user_factors[user_index]
                    
                    # Calcular puntuaciones para todos los items
                    item_scores = np.dot(user_factor, item_factors)
                    
                    # Obtener items con mayor puntuación
                    top_items = np.argsort(item_scores)[::-1][:5]
                    
                    for item_idx in top_items:
                        if item_scores[item_idx] > 0.5:
                            recommendations.append({
                                "type": "content_recommendation",
                                "title": f"Item recomendado #{item_idx + 1}",
                                "description": f"Recomendado por factorización de matrices con puntuación {item_scores[item_idx]:.2f}",
                                "content": {"item_id": item_idx, "score": float(item_scores[item_idx])},
                                "confidence": float(item_scores[item_idx]),
                                "relevance": float(item_scores[item_idx]) * 0.9,
                                "priority": "medium",
                                "category": "matrix_factorization"
                            })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error en factorización de matrices: {e}")
            return []
    
    async def _deep_fm_recommendation(self, user_id: str, profile: UserProfile) -> List[Dict[str, Any]]:
        """Recomendación con DeepFM"""
        try:
            recommendations = []
            
            # Simular modelo DeepFM
            # En implementación real, usar modelo entrenado
            user_features = await self._extract_user_features(user_id, profile)
            item_features = await self._get_item_features()
            
            # Calcular puntuaciones con DeepFM
            for item_id, item_feature in item_features.items():
                # Simular cálculo de puntuación
                score = await self._calculate_deepfm_score(user_features, item_feature)
                
                if score > 0.6:
                    recommendations.append({
                        "type": "content_recommendation",
                        "title": f"Recomendación DeepFM #{item_id}",
                        "description": f"Recomendado por modelo DeepFM con puntuación {score:.2f}",
                        "content": {"item_id": item_id, "score": score},
                        "confidence": score,
                        "relevance": score * 0.85,
                        "priority": "high",
                        "category": "deep_fm"
                    })
            
            return sorted(recommendations, key=lambda x: x["confidence"], reverse=True)[:5]
            
        except Exception as e:
            logger.error(f"Error en recomendación DeepFM: {e}")
            return []
    
    async def _transformer_based_recommendation(self, user_id: str, profile: UserProfile) -> List[Dict[str, Any]]:
        """Recomendación basada en Transformers"""
        try:
            recommendations = []
            
            # Usar modelo Transformer para análisis de secuencias de comportamiento
            behavior_sequence = await self._get_user_behavior_sequence(user_id)
            
            if not behavior_sequence:
                return []
            
            # Simular análisis con Transformer
            # En implementación real, usar modelo BERT o similar
            next_actions = await self._predict_next_actions(behavior_sequence)
            
            for action, probability in next_actions.items():
                if probability > 0.7:
                    recommendations.append({
                        "type": "workflow_optimization",
                        "title": f"Acción predicha: {action}",
                        "description": f"Basado en tu patrón de comportamiento, probablemente quieras realizar: {action}",
                        "content": {"predicted_action": action, "probability": probability},
                        "confidence": probability,
                        "relevance": probability * 0.9,
                        "priority": "medium",
                        "category": "transformer_based"
                    })
            
            return recommendations[:3]
            
        except Exception as e:
            logger.error(f"Error en recomendación basada en Transformers: {e}")
            return []
    
    async def _get_user_embedding(self, user_id: str, profile: UserProfile) -> Optional[np.ndarray]:
        """Obtiene embedding del usuario"""
        try:
            if user_id in self.user_embeddings:
                return self.user_embeddings[user_id]
            
            # Crear embedding basado en perfil
            embedding_features = []
            
            # Características categóricas
            skill_level_map = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
            work_style_map = {"collaborative": 0, "independent": 1, "mixed": 2}
            learning_style_map = {"visual": 0, "auditory": 1, "kinesthetic": 2, "reading": 3}
            
            embedding_features.extend([
                skill_level_map.get(profile.skill_level, 0),
                work_style_map.get(profile.work_style, 0),
                learning_style_map.get(profile.learning_style, 0)
            ])
            
            # Características numéricas
            embedding_features.extend([
                len(profile.interests),
                len(profile.goals),
                len(profile.constraints),
                profile.behavior_patterns.get("efficiency", 0.5)
            ])
            
            # Padding o truncamiento a tamaño fijo
            target_size = 32
            if len(embedding_features) < target_size:
                embedding_features.extend([0] * (target_size - len(embedding_features)))
            else:
                embedding_features = embedding_features[:target_size]
            
            embedding = np.array(embedding_features, dtype=np.float32)
            self.user_embeddings[user_id] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error obteniendo embedding de usuario: {e}")
            return None
    
    async def _find_similar_users_by_embedding(self, user_embedding: np.ndarray) -> List[str]:
        """Encuentra usuarios similares por embedding"""
        try:
            similarities = []
            
            for other_user_id, other_embedding in self.user_embeddings.items():
                # Calcular similitud coseno
                similarity = np.dot(user_embedding, other_embedding) / (
                    np.linalg.norm(user_embedding) * np.linalg.norm(other_embedding)
                )
                similarities.append((other_user_id, similarity))
            
            # Ordenar por similitud
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return [user_id for user_id, sim in similarities[:5] if sim > 0.7]
            
        except Exception as e:
            logger.error(f"Error encontrando usuarios similares por embedding: {e}")
            return []
    
    async def _create_user_item_matrix(self) -> Optional[np.ndarray]:
        """Crea matriz usuario-item"""
        try:
            # Obtener todos los usuarios y items únicos
            users = list(self.user_profiles.keys())
            items = set()
            
            for user_recs in self.recommendations.values():
                for rec in user_recs:
                    if rec.content and "item_id" in rec.content:
                        items.add(rec.content["item_id"])
            
            if not users or not items:
                return None
            
            items = list(items)
            matrix = np.zeros((len(users), len(items)))
            
            # Llenar matriz con interacciones
            for i, user_id in enumerate(users):
                user_recs = self.recommendations.get(user_id, [])
                for rec in user_recs:
                    if rec.content and "item_id" in rec.content:
                        item_id = rec.content["item_id"]
                        if item_id in items:
                            j = items.index(item_id)
                            matrix[i, j] = rec.confidence_score
            
            return matrix
            
        except Exception as e:
            logger.error(f"Error creando matriz usuario-item: {e}")
            return None
    
    async def _get_user_index(self, user_id: str) -> Optional[int]:
        """Obtiene índice del usuario en la matriz"""
        try:
            users = list(self.user_profiles.keys())
            if user_id in users:
                return users.index(user_id)
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo índice de usuario: {e}")
            return None
    
    async def _extract_user_features(self, user_id: str, profile: UserProfile) -> Dict[str, Any]:
        """Extrae características del usuario"""
        try:
            features = {
                "skill_level": profile.skill_level,
                "work_style": profile.work_style,
                "learning_style": profile.learning_style,
                "interests_count": len(profile.interests),
                "goals_count": len(profile.goals),
                "efficiency": profile.behavior_patterns.get("efficiency", 0.5),
                "activity_level": len(profile.behavior_patterns.get("usage_frequency", {})),
                "collaboration_frequency": profile.behavior_patterns.get("usage_frequency", {}).get("collaboration", 0)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extrayendo características de usuario: {e}")
            return {}
    
    async def _get_item_features(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene características de items"""
        try:
            # Simular características de items
            items = {
                "doc_processing": {"type": "feature", "complexity": "medium", "category": "productivity"},
                "collaboration": {"type": "feature", "complexity": "low", "category": "social"},
                "automation": {"type": "feature", "complexity": "high", "category": "efficiency"},
                "analytics": {"type": "feature", "complexity": "high", "category": "insights"},
                "translation": {"type": "feature", "complexity": "medium", "category": "language"}
            }
            
            return items
            
        except Exception as e:
            logger.error(f"Error obteniendo características de items: {e}")
            return {}
    
    async def _calculate_deepfm_score(self, user_features: Dict[str, Any], item_features: Dict[str, Any]) -> float:
        """Calcula puntuación con DeepFM"""
        try:
            # Simular cálculo de puntuación DeepFM
            # En implementación real, usar modelo entrenado
            
            score = 0.0
            
            # Factorización de características
            for user_key, user_val in user_features.items():
                for item_key, item_val in item_features.items():
                    if isinstance(user_val, (int, float)) and isinstance(item_val, (int, float)):
                        score += user_val * item_val * 0.1
                    elif user_val == item_val:
                        score += 0.2
            
            # Componente deep
            feature_vector = list(user_features.values()) + list(item_features.values())
            deep_score = sum(f for f in feature_vector if isinstance(f, (int, float))) * 0.01
            
            total_score = min(1.0, score + deep_score)
            return total_score
            
        except Exception as e:
            logger.error(f"Error calculando puntuación DeepFM: {e}")
            return 0.0
    
    async def _get_user_behavior_sequence(self, user_id: str) -> List[str]:
        """Obtiene secuencia de comportamiento del usuario"""
        try:
            user_behaviors = [
                behavior for behavior in self.user_behaviors
                if behavior.user_id == user_id
            ]
            
            # Ordenar por timestamp
            user_behaviors.sort(key=lambda x: x.timestamp)
            
            # Extraer secuencia de acciones
            sequence = [behavior.action for behavior in user_behaviors[-10:]]  # Últimas 10 acciones
            
            return sequence
            
        except Exception as e:
            logger.error(f"Error obteniendo secuencia de comportamiento: {e}")
            return []
    
    async def _predict_next_actions(self, behavior_sequence: List[str]) -> Dict[str, float]:
        """Predice próximas acciones usando Transformer"""
        try:
            # Simular predicción con Transformer
            # En implementación real, usar modelo BERT o similar
            
            predictions = {}
            
            # Análisis de patrones en la secuencia
            if "document_processing" in behavior_sequence:
                predictions["collaboration"] = 0.8
                predictions["automation"] = 0.7
            
            if "collaboration" in behavior_sequence:
                predictions["analytics"] = 0.6
                predictions["sharing"] = 0.8
            
            if "automation" in behavior_sequence:
                predictions["optimization"] = 0.9
                predictions["monitoring"] = 0.7
            
            # Agregar predicciones basadas en frecuencia
            action_counts = {}
            for action in behavior_sequence:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            for action, count in action_counts.items():
                if count > 1:
                    predictions[f"repeat_{action}"] = min(0.9, count * 0.2)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error prediciendo próximas acciones: {e}")
            return {}
    
    async def get_advanced_analytics(self) -> Dict[str, Any]:
        """Obtiene análisis avanzados de personalización"""
        try:
            analytics = {
                "user_segmentation": await self._analyze_user_segmentation(),
                "behavior_patterns": await self._analyze_behavior_patterns(),
                "recommendation_performance": await self._analyze_recommendation_performance(),
                "engagement_metrics": await self._analyze_engagement_metrics(),
                "predictive_insights": await self._generate_predictive_insights()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error obteniendo análisis avanzados: {e}")
            return {"error": str(e)}
    
    async def _analyze_user_segmentation(self) -> Dict[str, Any]:
        """Analiza segmentación de usuarios"""
        try:
            if not self.user_profiles:
                return {"segments": [], "total_users": 0}
            
            # Preparar datos para clustering
            user_features = []
            user_ids = []
            
            for user_id, profile in self.user_profiles.items():
                features = [
                    len(profile.interests),
                    len(profile.goals),
                    profile.behavior_patterns.get("efficiency", 0.5),
                    len(profile.behavior_patterns.get("usage_frequency", {}))
                ]
                user_features.append(features)
                user_ids.append(user_id)
            
            if len(user_features) < 2:
                return {"segments": [], "total_users": len(user_ids)}
            
            # Aplicar clustering
            clustering_model = self.ml_models.get('user_clustering')
            if clustering_model:
                clusters = clustering_model.fit_predict(user_features)
                
                # Analizar segmentos
                segments = {}
                for i, cluster in enumerate(clusters):
                    if cluster not in segments:
                        segments[cluster] = []
                    segments[cluster].append(user_ids[i])
                
                segment_analysis = []
                for cluster_id, users in segments.items():
                    segment_analysis.append({
                        "cluster_id": int(cluster_id),
                        "user_count": len(users),
                        "percentage": len(users) / len(user_ids) * 100,
                        "characteristics": await self._analyze_cluster_characteristics(users)
                    })
                
                return {
                    "segments": segment_analysis,
                    "total_users": len(user_ids),
                    "clustering_algorithm": "KMeans"
                }
            
            return {"segments": [], "total_users": len(user_ids)}
            
        except Exception as e:
            logger.error(f"Error analizando segmentación de usuarios: {e}")
            return {"error": str(e)}
    
    async def _analyze_cluster_characteristics(self, user_ids: List[str]) -> Dict[str, Any]:
        """Analiza características de un cluster"""
        try:
            if not user_ids:
                return {}
            
            # Analizar características del cluster
            skill_levels = defaultdict(int)
            work_styles = defaultdict(int)
            interests = defaultdict(int)
            
            for user_id in user_ids:
                profile = self.user_profiles.get(user_id)
                if profile:
                    skill_levels[profile.skill_level] += 1
                    work_styles[profile.work_style] += 1
                    for interest in profile.interests:
                        interests[interest] += 1
            
            return {
                "dominant_skill_level": max(skill_levels, key=skill_levels.get) if skill_levels else "unknown",
                "dominant_work_style": max(work_styles, key=work_styles.get) if work_styles else "unknown",
                "top_interests": dict(sorted(interests.items(), key=lambda x: x[1], reverse=True)[:5]),
                "average_efficiency": np.mean([
                    self.user_profiles[uid].behavior_patterns.get("efficiency", 0.5)
                    for uid in user_ids if uid in self.user_profiles
                ]) if user_ids else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analizando características de cluster: {e}")
            return {}
    
    async def _analyze_behavior_patterns(self) -> Dict[str, Any]:
        """Analiza patrones de comportamiento"""
        try:
            if not self.user_behaviors:
                return {"patterns": [], "insights": []}
            
            # Analizar patrones temporales
            hour_patterns = defaultdict(int)
            weekday_patterns = defaultdict(int)
            action_patterns = defaultdict(int)
            
            for behavior in self.user_behaviors:
                hour_patterns[behavior.timestamp.hour] += 1
                weekday_patterns[behavior.timestamp.weekday()] += 1
                action_patterns[behavior.action] += 1
            
            # Generar insights
            insights = []
            
            # Hora pico
            peak_hour = max(hour_patterns, key=hour_patterns.get) if hour_patterns else 0
            insights.append(f"Hora de mayor actividad: {peak_hour}:00")
            
            # Día más activo
            weekday_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            peak_weekday = max(weekday_patterns, key=weekday_patterns.get) if weekday_patterns else 0
            insights.append(f"Día más activo: {weekday_names[peak_weekday]}")
            
            # Acción más popular
            popular_action = max(action_patterns, key=action_patterns.get) if action_patterns else "N/A"
            insights.append(f"Acción más popular: {popular_action}")
            
            return {
                "patterns": {
                    "hour_distribution": dict(hour_patterns),
                    "weekday_distribution": dict(weekday_patterns),
                    "action_frequency": dict(action_patterns)
                },
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"Error analizando patrones de comportamiento: {e}")
            return {"error": str(e)}
    
    async def _analyze_recommendation_performance(self) -> Dict[str, Any]:
        """Analiza rendimiento de recomendaciones"""
        try:
            total_recommendations = sum(len(recs) for recs in self.recommendations.values())
            dismissed_recommendations = sum(
                len([rec for rec in recs if rec.is_dismissed])
                for recs in self.recommendations.values()
            )
            acted_upon_recommendations = sum(
                len([rec for rec in recs if rec.is_acted_upon])
                for recs in self.recommendations.values()
            )
            
            if total_recommendations == 0:
                return {"metrics": {}, "insights": []}
            
            # Calcular métricas
            dismissal_rate = dismissed_recommendations / total_recommendations
            action_rate = acted_upon_recommendations / total_recommendations
            
            # Análisis por tipo de recomendación
            type_performance = defaultdict(lambda: {"total": 0, "dismissed": 0, "acted": 0})
            
            for user_recs in self.recommendations.values():
                for rec in user_recs:
                    type_performance[rec.recommendation_type.value]["total"] += 1
                    if rec.is_dismissed:
                        type_performance[rec.recommendation_type.value]["dismissed"] += 1
                    if rec.is_acted_upon:
                        type_performance[rec.recommendation_type.value]["acted"] += 1
            
            # Generar insights
            insights = []
            if dismissal_rate > 0.5:
                insights.append("Tasa de descarte alta - revisar relevancia de recomendaciones")
            if action_rate < 0.2:
                insights.append("Tasa de acción baja - mejorar calidad de recomendaciones")
            
            return {
                "metrics": {
                    "total_recommendations": total_recommendations,
                    "dismissal_rate": dismissal_rate,
                    "action_rate": action_rate,
                    "type_performance": dict(type_performance)
                },
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"Error analizando rendimiento de recomendaciones: {e}")
            return {"error": str(e)}
    
    async def _analyze_engagement_metrics(self) -> Dict[str, Any]:
        """Analiza métricas de engagement"""
        try:
            if not self.user_behaviors:
                return {"metrics": {}, "insights": []}
            
            # Calcular métricas de engagement
            total_behaviors = len(self.user_behaviors)
            successful_behaviors = sum(1 for b in self.user_behaviors if b.success)
            avg_duration = np.mean([b.duration for b in self.user_behaviors if b.duration > 0])
            
            # Engagement por usuario
            user_engagement = defaultdict(lambda: {"behaviors": 0, "success_rate": 0, "avg_duration": 0})
            
            for behavior in self.user_behaviors:
                user_engagement[behavior.user_id]["behaviors"] += 1
            
            for user_id, data in user_engagement.items():
                user_behaviors = [b for b in self.user_behaviors if b.user_id == user_id]
                successful = sum(1 for b in user_behaviors if b.success)
                durations = [b.duration for b in user_behaviors if b.duration > 0]
                
                data["success_rate"] = successful / len(user_behaviors) if user_behaviors else 0
                data["avg_duration"] = np.mean(durations) if durations else 0
            
            # Generar insights
            insights = []
            if successful_behaviors / total_behaviors < 0.8:
                insights.append("Tasa de éxito baja - revisar usabilidad")
            if avg_duration > 300:  # 5 minutos
                insights.append("Duración promedio alta - optimizar workflows")
            
            return {
                "metrics": {
                    "total_behaviors": total_behaviors,
                    "success_rate": successful_behaviors / total_behaviors if total_behaviors > 0 else 0,
                    "avg_duration": avg_duration,
                    "active_users": len(user_engagement),
                    "user_engagement": dict(user_engagement)
                },
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"Error analizando métricas de engagement: {e}")
            return {"error": str(e)}
    
    async def _generate_predictive_insights(self) -> Dict[str, Any]:
        """Genera insights predictivos"""
        try:
            insights = []
            
            # Predicción de abandono
            inactive_users = await self._predict_user_churn()
            if inactive_users:
                insights.append(f"Predicción: {len(inactive_users)} usuarios en riesgo de abandono")
            
            # Predicción de crecimiento
            growth_prediction = await self._predict_user_growth()
            insights.append(f"Predicción: Crecimiento esperado de {growth_prediction:.1f}% en próximos 30 días")
            
            # Predicción de características populares
            popular_features = await self._predict_popular_features()
            insights.append(f"Predicción: Características más demandadas: {', '.join(popular_features[:3])}")
            
            return {
                "insights": insights,
                "churn_risk_users": inactive_users,
                "growth_prediction": growth_prediction,
                "popular_features": popular_features
            }
            
        except Exception as e:
            logger.error(f"Error generando insights predictivos: {e}")
            return {"error": str(e)}
    
    async def _predict_user_churn(self) -> List[str]:
        """Predice usuarios en riesgo de abandono"""
        try:
            churn_risk_users = []
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for user_id, profile in self.user_profiles.items():
                # Obtener última actividad
                user_behaviors = [
                    b for b in self.user_behaviors
                    if b.user_id == user_id
                ]
                
                if not user_behaviors:
                    churn_risk_users.append(user_id)
                    continue
                
                last_activity = max(b.timestamp for b in user_behaviors)
                
                # Criterios de riesgo de abandono
                if last_activity < cutoff_date:
                    churn_risk_users.append(user_id)
                elif profile.behavior_patterns.get("efficiency", 0.5) < 0.3:
                    churn_risk_users.append(user_id)
            
            return churn_risk_users
            
        except Exception as e:
            logger.error(f"Error prediciendo abandono de usuarios: {e}")
            return []
    
    async def _predict_user_growth(self) -> float:
        """Predice crecimiento de usuarios"""
        try:
            # Análisis simple de tendencia
            if len(self.user_behaviors) < 10:
                return 0.0
            
            # Agrupar comportamientos por día
            daily_behaviors = defaultdict(int)
            for behavior in self.user_behaviors:
                date = behavior.timestamp.date()
                daily_behaviors[date] += 1
            
            if len(daily_behaviors) < 7:
                return 0.0
            
            # Calcular tendencia
            dates = sorted(daily_behaviors.keys())
            counts = [daily_behaviors[date] for date in dates]
            
            # Regresión lineal simple
            x = np.arange(len(counts))
            slope, _, _, _, _ = stats.linregress(x, counts)
            
            # Predecir crecimiento
            growth_rate = slope / np.mean(counts) * 100 if np.mean(counts) > 0 else 0
            
            return max(0, min(50, growth_rate))  # Limitar entre 0% y 50%
            
        except Exception as e:
            logger.error(f"Error prediciendo crecimiento de usuarios: {e}")
            return 0.0
    
    async def _predict_popular_features(self) -> List[str]:
        """Predice características populares"""
        try:
            # Analizar tendencias en uso de características
            feature_usage = defaultdict(int)
            
            for behavior in self.user_behaviors:
                action = behavior.action.lower()
                if "document" in action:
                    feature_usage["document_processing"] += 1
                elif "collaboration" in action:
                    feature_usage["collaboration"] += 1
                elif "automation" in action:
                    feature_usage["automation"] += 1
                elif "analytics" in action:
                    feature_usage["analytics"] += 1
                elif "translation" in action:
                    feature_usage["translation"] += 1
            
            # Ordenar por popularidad
            popular_features = sorted(
                feature_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [feature for feature, count in popular_features]
            
        except Exception as e:
            logger.error(f"Error prediciendo características populares: {e}")
            return []
    
    async def create_personalization_dashboard(self) -> str:
        """Crea dashboard de personalización con visualizaciones"""
        try:
            # Obtener datos de análisis
            analytics = await self.get_advanced_analytics()
            
            # Crear visualizaciones
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Segmentación de Usuarios', 'Patrones de Comportamiento', 
                              'Rendimiento de Recomendaciones', 'Métricas de Engagement'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Gráfico de segmentación
            if analytics.get("user_segmentation", {}).get("segments"):
                segments = analytics["user_segmentation"]["segments"]
                labels = [f"Segmento {s['cluster_id']}" for s in segments]
                values = [s["user_count"] for s in segments]
                
                fig.add_trace(
                    go.Pie(labels=labels, values=values, name="Segmentación"),
                    row=1, col=1
                )
            
            # Gráfico de patrones de comportamiento
            if analytics.get("behavior_patterns", {}).get("patterns", {}).get("hour_distribution"):
                hour_data = analytics["behavior_patterns"]["patterns"]["hour_distribution"]
                hours = list(hour_data.keys())
                counts = list(hour_data.values())
                
                fig.add_trace(
                    go.Bar(x=hours, y=counts, name="Actividad por Hora"),
                    row=1, col=2
                )
            
            # Gráfico de rendimiento de recomendaciones
            if analytics.get("recommendation_performance", {}).get("metrics", {}).get("type_performance"):
                type_perf = analytics["recommendation_performance"]["metrics"]["type_performance"]
                types = list(type_perf.keys())
                action_rates = [
                    type_perf[t]["acted"] / type_perf[t]["total"] 
                    if type_perf[t]["total"] > 0 else 0
                    for t in types
                ]
                
                fig.add_trace(
                    go.Bar(x=types, y=action_rates, name="Tasa de Acción"),
                    row=2, col=1
                )
            
            # Gráfico de engagement
            if analytics.get("engagement_metrics", {}).get("metrics", {}).get("user_engagement"):
                user_eng = analytics["engagement_metrics"]["metrics"]["user_engagement"]
                users = list(user_eng.keys())[:10]  # Top 10 usuarios
                success_rates = [user_eng[u]["success_rate"] for u in users]
                behaviors = [user_eng[u]["behaviors"] for u in users]
                
                fig.add_trace(
                    go.Scatter(x=behaviors, y=success_rates, mode='markers', 
                             text=users, name="Engagement"),
                    row=2, col=2
                )
            
            # Configurar layout
            fig.update_layout(
                title="Dashboard de Personalización AI",
                showlegend=False,
                height=800
            )
            
            # Convertir a HTML
            dashboard_html = fig.to_html(include_plotlyjs='cdn')
            
            return dashboard_html
            
        except Exception as e:
            logger.error(f"Error creando dashboard de personalización: {e}")
            return f"<html><body><h1>Error creando dashboard: {str(e)}</h1></body></html>"


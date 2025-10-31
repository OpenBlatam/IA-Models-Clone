"""
Motor de Optimización de Rendimiento AI
========================================

Motor para optimización automática de rendimiento, auto-tuning de modelos y mejora continua del sistema.
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
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import pickle
import base64
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
import ray
from ray import tune
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import wandb
import joblib
import dask
from dask.distributed import Client
import redis
import elasticsearch
from scipy import stats
import networkx as nx

logger = logging.getLogger(__name__)

class OptimizationType(str, Enum):
    """Tipos de optimización"""
    MODEL_PERFORMANCE = "model_performance"
    SYSTEM_RESOURCES = "system_resources"
    API_RESPONSE_TIME = "api_response_time"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    CACHE_EFFICIENCY = "cache_efficiency"
    DATABASE_QUERIES = "database_queries"
    NETWORK_LATENCY = "network_latency"
    BATCH_PROCESSING = "batch_processing"
    CONCURRENT_REQUESTS = "concurrent_requests"

class OptimizationStrategy(str, Enum):
    """Estrategias de optimización"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY_STRATEGY = "evolutionary_strategy"
    MULTI_OBJECTIVE = "multi_objective"

class PerformanceMetric(str, Enum):
    """Métricas de rendimiento"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    LATENCY = "latency"
    BANDWIDTH = "bandwidth"

@dataclass
class PerformanceBaseline:
    """Línea base de rendimiento"""
    metric_name: str
    current_value: float
    target_value: float
    threshold_min: float
    threshold_max: float
    unit: str
    measurement_time: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    """Resultado de optimización"""
    id: str
    optimization_type: OptimizationType
    strategy: OptimizationStrategy
    baseline_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]
    improvement_percentage: float
    parameters: Dict[str, Any]
    execution_time: float
    status: str  # success, failed, partial
    created_at: datetime = field(default_factory=datetime.now)
    notes: str = ""

@dataclass
class OptimizationConfig:
    """Configuración de optimización"""
    id: str
    name: str
    description: str
    optimization_type: OptimizationType
    strategy: OptimizationStrategy
    target_metrics: List[PerformanceMetric]
    constraints: Dict[str, Any]
    max_iterations: int = 100
    timeout_seconds: int = 3600
    parallel_workers: int = 4
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class SystemMetrics:
    """Métricas del sistema"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    active_connections: int = 0
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0

class AIPerformanceOptimizer:
    """Motor de optimización de rendimiento AI"""
    
    def __init__(self):
        self.performance_baselines: Dict[str, PerformanceBaseline] = {}
        self.optimization_results: List[OptimizationResult] = []
        self.optimization_configs: Dict[str, OptimizationConfig] = {}
        self.system_metrics: List[SystemMetrics] = []
        
        # Configuración de optimización
        self.metrics_collection_interval = 60  # 1 minuto
        self.optimization_interval = 3600  # 1 hora
        self.baseline_update_interval = 86400  # 24 horas
        
        # Workers de optimización
        self.optimization_workers: Dict[str, asyncio.Task] = {}
        self.optimization_active = False
        
        # Componentes de optimización
        self.redis_client: Optional[redis.Redis] = None
        self.elasticsearch_client: Optional[elasticsearch.Elasticsearch] = None
        self.db_connection: Optional[sqlite3.Connection] = None
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.ray_client: Optional[ray.Client] = None
        self.dask_client: Optional[Client] = None
        
        # Modelos de optimización
        self.optimization_models: Dict[str, Any] = {}
        self.performance_predictors: Dict[str, Any] = {}
        
        # Cache de optimización
        self.optimization_cache: Dict[str, Any] = {}
        self.cache_lock = threading.Lock()
        
        # Métricas de rendimiento
        self.performance_metrics = {
            "optimization_success_rate": 0.0,
            "average_improvement": 0.0,
            "optimization_time": 0.0,
            "system_efficiency": 0.0
        }
        
        # Configuración de MLflow
        self.mlflow_tracking_uri = "http://localhost:5000"
        self.mlflow_experiment_name = "ai_performance_optimization"
        
    async def initialize(self):
        """Inicializa el motor de optimización de rendimiento"""
        logger.info("Inicializando motor de optimización de rendimiento AI...")
        
        # Inicializar componentes de optimización
        await self._initialize_optimization_components()
        
        # Cargar configuraciones existentes
        await self._load_optimization_configs()
        
        # Establecer líneas base de rendimiento
        await self._establish_performance_baselines()
        
        # Inicializar modelos de optimización
        await self._initialize_optimization_models()
        
        # Iniciar workers de optimización
        await self._start_optimization_workers()
        
        # Configurar MLflow
        await self._setup_mlflow()
        
        logger.info("Motor de optimización de rendimiento AI inicializado")
    
    async def _initialize_optimization_components(self):
        """Inicializa componentes de optimización"""
        try:
            # Inicializar Redis para cache distribuido
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=1,
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("Redis conectado para optimización")
            except Exception as e:
                logger.warning(f"No se pudo conectar a Redis: {e}")
            
            # Inicializar Elasticsearch para métricas
            try:
                self.elasticsearch_client = elasticsearch.Elasticsearch(
                    ['localhost:9200'],
                    timeout=30
                )
                if self.elasticsearch_client.ping():
                    logger.info("Elasticsearch conectado para métricas")
            except Exception as e:
                logger.warning(f"No se pudo conectar a Elasticsearch: {e}")
            
            # Inicializar base de datos SQLite
            self.db_connection = sqlite3.connect(
                'data/performance_optimization.db',
                check_same_thread=False
            )
            await self._create_database_schema()
            logger.info("Base de datos SQLite inicializada")
            
            # Inicializar Ray para computación distribuida
            try:
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                self.ray_client = ray
                logger.info("Ray inicializado para computación distribuida")
            except Exception as e:
                logger.warning(f"No se pudo inicializar Ray: {e}")
            
            # Inicializar Dask para procesamiento paralelo
            try:
                self.dask_client = Client()
                logger.info("Dask inicializado para procesamiento paralelo")
            except Exception as e:
                logger.warning(f"No se pudo inicializar Dask: {e}")
            
        except Exception as e:
            logger.error(f"Error inicializando componentes de optimización: {e}")
    
    async def _create_database_schema(self):
        """Crea esquema de base de datos"""
        try:
            cursor = self.db_connection.cursor()
            
            # Tabla de líneas base de rendimiento
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_baselines (
                    id TEXT PRIMARY KEY,
                    metric_name TEXT,
                    current_value REAL,
                    target_value REAL,
                    threshold_min REAL,
                    threshold_max REAL,
                    unit TEXT,
                    measurement_time TEXT,
                    context TEXT
                )
            ''')
            
            # Tabla de resultados de optimización
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id TEXT PRIMARY KEY,
                    optimization_type TEXT,
                    strategy TEXT,
                    baseline_metrics TEXT,
                    optimized_metrics TEXT,
                    improvement_percentage REAL,
                    parameters TEXT,
                    execution_time REAL,
                    status TEXT,
                    created_at TEXT,
                    notes TEXT
                )
            ''')
            
            # Tabla de configuraciones de optimización
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_configs (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    optimization_type TEXT,
                    strategy TEXT,
                    target_metrics TEXT,
                    constraints TEXT,
                    max_iterations INTEGER,
                    timeout_seconds INTEGER,
                    parallel_workers INTEGER,
                    is_active BOOLEAN,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            # Tabla de métricas del sistema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    network_io TEXT,
                    active_connections INTEGER,
                    response_time REAL,
                    throughput REAL,
                    error_rate REAL,
                    cache_hit_rate REAL
                )
            ''')
            
            # Índices para optimización
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_baselines_metric ON performance_baselines(metric_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_type ON optimization_results(optimization_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp)')
            
            self.db_connection.commit()
            logger.info("Esquema de base de datos creado")
            
        except Exception as e:
            logger.error(f"Error creando esquema de base de datos: {e}")
    
    async def _load_optimization_configs(self):
        """Carga configuraciones de optimización"""
        try:
            # Configuraciones predefinidas
            default_configs = [
                {
                    "id": "model_performance_opt",
                    "name": "Optimización de Rendimiento de Modelos",
                    "description": "Optimiza parámetros de modelos ML para mejorar precisión y velocidad",
                    "optimization_type": OptimizationType.MODEL_PERFORMANCE,
                    "strategy": OptimizationStrategy.BAYESIAN_OPTIMIZATION,
                    "target_metrics": [PerformanceMetric.ACCURACY, PerformanceMetric.RESPONSE_TIME],
                    "constraints": {"max_memory": "8GB", "max_cpu": "80%"},
                    "max_iterations": 50,
                    "timeout_seconds": 1800
                },
                {
                    "id": "system_resources_opt",
                    "name": "Optimización de Recursos del Sistema",
                    "description": "Optimiza uso de CPU, memoria y recursos del sistema",
                    "optimization_type": OptimizationType.SYSTEM_RESOURCES,
                    "strategy": OptimizationStrategy.GRID_SEARCH,
                    "target_metrics": [PerformanceMetric.CPU_USAGE, PerformanceMetric.MEMORY_USAGE],
                    "constraints": {"min_performance": 0.8},
                    "max_iterations": 30,
                    "timeout_seconds": 1200
                },
                {
                    "id": "api_response_opt",
                    "name": "Optimización de Tiempo de Respuesta API",
                    "description": "Optimiza tiempo de respuesta de APIs y endpoints",
                    "optimization_type": OptimizationType.API_RESPONSE_TIME,
                    "strategy": OptimizationStrategy.REINFORCEMENT_LEARNING,
                    "target_metrics": [PerformanceMetric.RESPONSE_TIME, PerformanceMetric.THROUGHPUT],
                    "constraints": {"max_latency": 1000},
                    "max_iterations": 100,
                    "timeout_seconds": 2400
                },
                {
                    "id": "cache_efficiency_opt",
                    "name": "Optimización de Eficiencia de Cache",
                    "description": "Optimiza estrategias de cache para mejorar hit rate",
                    "optimization_type": OptimizationType.CACHE_EFFICIENCY,
                    "strategy": OptimizationStrategy.MULTI_OBJECTIVE,
                    "target_metrics": [PerformanceMetric.CACHE_HIT_RATE, PerformanceMetric.MEMORY_USAGE],
                    "constraints": {"min_hit_rate": 0.7},
                    "max_iterations": 40,
                    "timeout_seconds": 900
                }
            ]
            
            for config_data in default_configs:
                config = OptimizationConfig(
                    id=config_data["id"],
                    name=config_data["name"],
                    description=config_data["description"],
                    optimization_type=config_data["optimization_type"],
                    strategy=config_data["strategy"],
                    target_metrics=config_data["target_metrics"],
                    constraints=config_data["constraints"],
                    max_iterations=config_data["max_iterations"],
                    timeout_seconds=config_data["timeout_seconds"]
                )
                self.optimization_configs[config.id] = config
            
            logger.info(f"Cargadas {len(self.optimization_configs)} configuraciones de optimización")
            
        except Exception as e:
            logger.error(f"Error cargando configuraciones de optimización: {e}")
    
    async def _establish_performance_baselines(self):
        """Establece líneas base de rendimiento"""
        try:
            # Obtener métricas actuales del sistema
            current_metrics = await self._collect_current_metrics()
            
            # Establecer líneas base
            baselines = [
                PerformanceBaseline(
                    metric_name="response_time",
                    current_value=current_metrics.get("response_time", 100.0),
                    target_value=50.0,
                    threshold_min=20.0,
                    threshold_max=200.0,
                    unit="ms"
                ),
                PerformanceBaseline(
                    metric_name="cpu_usage",
                    current_value=current_metrics.get("cpu_usage", 30.0),
                    target_value=20.0,
                    threshold_min=5.0,
                    threshold_max=80.0,
                    unit="%"
                ),
                PerformanceBaseline(
                    metric_name="memory_usage",
                    current_value=current_metrics.get("memory_usage", 40.0),
                    target_value=30.0,
                    threshold_min=10.0,
                    threshold_max=90.0,
                    unit="%"
                ),
                PerformanceBaseline(
                    metric_name="throughput",
                    current_value=current_metrics.get("throughput", 100.0),
                    target_value=200.0,
                    threshold_min=50.0,
                    threshold_max=500.0,
                    unit="requests/sec"
                ),
                PerformanceBaseline(
                    metric_name="error_rate",
                    current_value=current_metrics.get("error_rate", 1.0),
                    target_value=0.1,
                    threshold_min=0.0,
                    threshold_max=5.0,
                    unit="%"
                )
            ]
            
            for baseline in baselines:
                self.performance_baselines[baseline.metric_name] = baseline
            
            logger.info(f"Establecidas {len(self.performance_baselines)} líneas base de rendimiento")
            
        except Exception as e:
            logger.error(f"Error estableciendo líneas base de rendimiento: {e}")
    
    async def _collect_current_metrics(self) -> Dict[str, float]:
        """Recolecta métricas actuales del sistema"""
        try:
            # Métricas del sistema
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Métricas de red
            network = psutil.net_io_counters()
            
            # Métricas simuladas de la aplicación
            response_time = np.random.normal(100, 20)  # Simular tiempo de respuesta
            throughput = np.random.normal(150, 30)     # Simular throughput
            error_rate = np.random.exponential(0.5)   # Simular tasa de error
            
            return {
                "cpu_usage": cpu_usage,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "network_sent": network.bytes_sent,
                "network_recv": network.bytes_recv,
                "response_time": response_time,
                "throughput": throughput,
                "error_rate": error_rate,
                "active_connections": np.random.randint(10, 100)
            }
            
        except Exception as e:
            logger.error(f"Error recolectando métricas actuales: {e}")
            return {}
    
    async def _initialize_optimization_models(self):
        """Inicializa modelos de optimización"""
        try:
            # Modelo de predicción de rendimiento
            self.performance_predictors['response_time'] = self._create_performance_predictor()
            self.performance_predictors['throughput'] = self._create_performance_predictor()
            self.performance_predictors['resource_usage'] = self._create_performance_predictor()
            
            # Modelos de optimización específicos
            self.optimization_models['bayesian_optimizer'] = self._create_bayesian_optimizer()
            self.optimization_models['genetic_optimizer'] = self._create_genetic_optimizer()
            self.optimization_models['reinforcement_optimizer'] = self._create_reinforcement_optimizer()
            
            logger.info("Modelos de optimización inicializados")
            
        except Exception as e:
            logger.error(f"Error inicializando modelos de optimización: {e}")
    
    def _create_performance_predictor(self):
        """Crea predictor de rendimiento"""
        try:
            # Modelo simple de predicción
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='linear')
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error creando predictor de rendimiento: {e}")
            return None
    
    def _create_bayesian_optimizer(self):
        """Crea optimizador bayesiano"""
        try:
            # Usar Optuna para optimización bayesiana
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler()
            )
            return study
            
        except Exception as e:
            logger.error(f"Error creando optimizador bayesiano: {e}")
            return None
    
    def _create_genetic_optimizer(self):
        """Crea optimizador genético"""
        try:
            # Implementación simple de algoritmo genético
            class GeneticOptimizer:
                def __init__(self, population_size=50, generations=100):
                    self.population_size = population_size
                    self.generations = generations
                    self.population = []
                
                def optimize(self, objective_function, bounds):
                    # Implementación simplificada
                    best_solution = None
                    best_fitness = float('-inf')
                    
                    for generation in range(self.generations):
                        # Evaluar población
                        fitness_scores = []
                        for individual in self.population:
                            fitness = objective_function(individual)
                            fitness_scores.append(fitness)
                            
                            if fitness > best_fitness:
                                best_fitness = fitness
                                best_solution = individual
                        
                        # Selección, cruce y mutación
                        self.population = self._evolve_population(fitness_scores)
                    
                    return best_solution, best_fitness
                
                def _evolve_population(self, fitness_scores):
                    # Implementación simplificada de evolución
                    return self.population  # Placeholder
            
            return GeneticOptimizer()
            
        except Exception as e:
            logger.error(f"Error creando optimizador genético: {e}")
            return None
    
    def _create_reinforcement_optimizer(self):
        """Crea optimizador de aprendizaje por refuerzo"""
        try:
            # Implementación simple de RL para optimización
            class RLOptimizer:
                def __init__(self, state_size=10, action_size=5):
                    self.state_size = state_size
                    self.action_size = action_size
                    self.q_table = np.zeros((state_size, action_size))
                    self.learning_rate = 0.1
                    self.epsilon = 0.1
                
                def optimize(self, environment):
                    # Implementación simplificada de Q-learning
                    for episode in range(1000):
                        state = environment.reset()
                        
                        for step in range(100):
                            action = self._select_action(state)
                            next_state, reward, done = environment.step(action)
                            
                            # Actualizar Q-table
                            self.q_table[state, action] += self.learning_rate * (
                                reward + 0.9 * np.max(self.q_table[next_state]) - self.q_table[state, action]
                            )
                            
                            state = next_state
                            if done:
                                break
                    
                    return self.q_table
                
                def _select_action(self, state):
                    if np.random.random() < self.epsilon:
                        return np.random.choice(self.action_size)
                    return np.argmax(self.q_table[state])
            
            return RLOptimizer()
            
        except Exception as e:
            logger.error(f"Error creando optimizador de RL: {e}")
            return None
    
    async def _start_optimization_workers(self):
        """Inicia workers de optimización"""
        try:
            self.optimization_active = True
            
            # Worker de recolección de métricas
            asyncio.create_task(self._metrics_collection_worker())
            
            # Worker de optimización automática
            asyncio.create_task(self._automatic_optimization_worker())
            
            # Worker de actualización de líneas base
            asyncio.create_task(self._baseline_update_worker())
            
            # Worker de análisis de rendimiento
            asyncio.create_task(self._performance_analysis_worker())
            
            logger.info("Workers de optimización iniciados")
            
        except Exception as e:
            logger.error(f"Error iniciando workers de optimización: {e}")
    
    async def _metrics_collection_worker(self):
        """Worker de recolección de métricas"""
        while self.optimization_active:
            try:
                await asyncio.sleep(self.metrics_collection_interval)
                
                # Recolectar métricas del sistema
                metrics = await self._collect_current_metrics()
                
                # Crear objeto de métricas
                system_metrics = SystemMetrics(
                    cpu_usage=metrics.get("cpu_usage", 0.0),
                    memory_usage=metrics.get("memory_usage", 0.0),
                    disk_usage=metrics.get("disk_usage", 0.0),
                    network_io={
                        "sent": metrics.get("network_sent", 0),
                        "received": metrics.get("network_recv", 0)
                    },
                    active_connections=metrics.get("active_connections", 0),
                    response_time=metrics.get("response_time", 0.0),
                    throughput=metrics.get("throughput", 0.0),
                    error_rate=metrics.get("error_rate", 0.0),
                    cache_hit_rate=metrics.get("cache_hit_rate", 0.0)
                )
                
                self.system_metrics.append(system_metrics)
                
                # Limpiar métricas antiguas (mantener últimas 24 horas)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.system_metrics = [
                    m for m in self.system_metrics
                    if m.timestamp > cutoff_time
                ]
                
            except Exception as e:
                logger.error(f"Error en worker de recolección de métricas: {e}")
                await asyncio.sleep(60)
    
    async def _automatic_optimization_worker(self):
        """Worker de optimización automática"""
        while self.optimization_active:
            try:
                await asyncio.sleep(self.optimization_interval)
                
                # Verificar si se necesita optimización
                needs_optimization = await self._check_optimization_needs()
                
                if needs_optimization:
                    # Ejecutar optimizaciones automáticas
                    await self._execute_automatic_optimizations()
                
            except Exception as e:
                logger.error(f"Error en worker de optimización automática: {e}")
                await asyncio.sleep(300)
    
    async def _baseline_update_worker(self):
        """Worker de actualización de líneas base"""
        while self.optimization_active:
            try:
                await asyncio.sleep(self.baseline_update_interval)
                
                # Actualizar líneas base basadas en métricas recientes
                await self._update_performance_baselines()
                
            except Exception as e:
                logger.error(f"Error en worker de actualización de líneas base: {e}")
                await asyncio.sleep(3600)
    
    async def _performance_analysis_worker(self):
        """Worker de análisis de rendimiento"""
        while self.optimization_active:
            try:
                await asyncio.sleep(1800)  # Cada 30 minutos
                
                # Analizar tendencias de rendimiento
                await self._analyze_performance_trends()
                
                # Generar reportes de rendimiento
                await self._generate_performance_reports()
                
            except Exception as e:
                logger.error(f"Error en worker de análisis de rendimiento: {e}")
                await asyncio.sleep(300)
    
    async def _check_optimization_needs(self) -> bool:
        """Verifica si se necesita optimización"""
        try:
            if not self.system_metrics:
                return False
            
            # Obtener métricas recientes
            recent_metrics = self.system_metrics[-10:]  # Últimas 10 mediciones
            
            # Verificar si alguna métrica está por debajo del umbral
            for baseline in self.performance_baselines.values():
                metric_name = baseline.metric_name
                
                # Obtener valor promedio reciente
                recent_values = []
                for metric in recent_metrics:
                    if hasattr(metric, metric_name):
                        recent_values.append(getattr(metric, metric_name))
                
                if recent_values:
                    avg_value = np.mean(recent_values)
                    
                    # Verificar si está por debajo del umbral
                    if avg_value > baseline.threshold_max or avg_value < baseline.threshold_min:
                        logger.info(f"Métrica {metric_name} fuera de umbral: {avg_value}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error verificando necesidades de optimización: {e}")
            return False
    
    async def _execute_automatic_optimizations(self):
        """Ejecuta optimizaciones automáticas"""
        try:
            # Ejecutar optimizaciones para cada configuración activa
            for config in self.optimization_configs.values():
                if not config.is_active:
                    continue
                
                # Ejecutar optimización
                result = await self._run_optimization(config)
                
                if result:
                    self.optimization_results.append(result)
                    
                    # Log del resultado
                    logger.info(f"Optimización completada: {config.name} - Mejora: {result.improvement_percentage:.2f}%")
            
        except Exception as e:
            logger.error(f"Error ejecutando optimizaciones automáticas: {e}")
    
    async def _run_optimization(self, config: OptimizationConfig) -> Optional[OptimizationResult]:
        """Ejecuta una optimización específica"""
        try:
            start_time = time.time()
            
            # Obtener métricas de línea base
            baseline_metrics = await self._get_baseline_metrics(config.target_metrics)
            
            # Ejecutar optimización según estrategia
            if config.strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
                optimized_params = await self._run_bayesian_optimization(config)
            elif config.strategy == OptimizationStrategy.GRID_SEARCH:
                optimized_params = await self._run_grid_search_optimization(config)
            elif config.strategy == OptimizationStrategy.RANDOM_SEARCH:
                optimized_params = await self._run_random_search_optimization(config)
            elif config.strategy == OptimizationStrategy.GENETIC_ALGORITHM:
                optimized_params = await self._run_genetic_optimization(config)
            elif config.strategy == OptimizationStrategy.REINFORCEMENT_LEARNING:
                optimized_params = await self._run_reinforcement_optimization(config)
            else:
                optimized_params = await self._run_default_optimization(config)
            
            # Medir métricas optimizadas
            optimized_metrics = await self._measure_optimized_metrics(config.target_metrics, optimized_params)
            
            # Calcular mejora
            improvement = await self._calculate_improvement(baseline_metrics, optimized_metrics)
            
            execution_time = time.time() - start_time
            
            # Crear resultado
            result = OptimizationResult(
                id=f"opt_{uuid.uuid4().hex[:8]}",
                optimization_type=config.optimization_type,
                strategy=config.strategy,
                baseline_metrics=baseline_metrics,
                optimized_metrics=optimized_metrics,
                improvement_percentage=improvement,
                parameters=optimized_params,
                execution_time=execution_time,
                status="success" if improvement > 0 else "failed"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error ejecutando optimización: {e}")
            return None
    
    async def _get_baseline_metrics(self, target_metrics: List[PerformanceMetric]) -> Dict[str, float]:
        """Obtiene métricas de línea base"""
        try:
            baseline_metrics = {}
            
            for metric in target_metrics:
                metric_name = metric.value
                
                if metric_name in self.performance_baselines:
                    baseline_metrics[metric_name] = self.performance_baselines[metric_name].current_value
                else:
                    # Obtener de métricas recientes
                    if self.system_metrics:
                        recent_metrics = self.system_metrics[-5:]
                        values = []
                        for m in recent_metrics:
                            if hasattr(m, metric_name):
                                values.append(getattr(m, metric_name))
                        
                        if values:
                            baseline_metrics[metric_name] = np.mean(values)
                        else:
                            baseline_metrics[metric_name] = 0.0
                    else:
                        baseline_metrics[metric_name] = 0.0
            
            return baseline_metrics
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas de línea base: {e}")
            return {}
    
    async def _run_bayesian_optimization(self, config: OptimizationConfig) -> Dict[str, Any]:
        """Ejecuta optimización bayesiana"""
        try:
            # Usar Optuna para optimización bayesiana
            study = optuna.create_study(direction='maximize')
            
            def objective(trial):
                # Definir espacio de búsqueda
                params = {}
                
                if config.optimization_type == OptimizationType.MODEL_PERFORMANCE:
                    params['learning_rate'] = trial.suggest_float('learning_rate', 0.001, 0.1)
                    params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
                    params['hidden_units'] = trial.suggest_int('hidden_units', 32, 256)
                elif config.optimization_type == OptimizationType.SYSTEM_RESOURCES:
                    params['worker_threads'] = trial.suggest_int('worker_threads', 1, 16)
                    params['cache_size'] = trial.suggest_int('cache_size', 100, 10000)
                    params['connection_pool'] = trial.suggest_int('connection_pool', 5, 50)
                elif config.optimization_type == OptimizationType.API_RESPONSE_TIME:
                    params['timeout'] = trial.suggest_int('timeout', 100, 5000)
                    params['retry_attempts'] = trial.suggest_int('retry_attempts', 1, 5)
                    params['concurrent_requests'] = trial.suggest_int('concurrent_requests', 10, 100)
                
                # Evaluar objetivo
                score = await self._evaluate_optimization_objective(config, params)
                return score
            
            # Ejecutar optimización
            study.optimize(objective, n_trials=config.max_iterations)
            
            return study.best_params
            
        except Exception as e:
            logger.error(f"Error en optimización bayesiana: {e}")
            return {}
    
    async def _run_grid_search_optimization(self, config: OptimizationConfig) -> Dict[str, Any]:
        """Ejecuta búsqueda en cuadrícula"""
        try:
            # Definir cuadrícula de parámetros
            param_grid = {}
            
            if config.optimization_type == OptimizationType.MODEL_PERFORMANCE:
                param_grid = {
                    'learning_rate': [0.001, 0.01, 0.1],
                    'batch_size': [16, 32, 64],
                    'hidden_units': [64, 128, 256]
                }
            elif config.optimization_type == OptimizationType.SYSTEM_RESOURCES:
                param_grid = {
                    'worker_threads': [2, 4, 8, 16],
                    'cache_size': [1000, 5000, 10000],
                    'connection_pool': [10, 20, 50]
                }
            
            best_params = {}
            best_score = float('-inf')
            
            # Generar todas las combinaciones
            from itertools import product
            
            keys = list(param_grid.keys())
            values = list(param_grid.values())
            
            for combination in product(*values):
                params = dict(zip(keys, combination))
                score = await self._evaluate_optimization_objective(config, params)
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error en búsqueda en cuadrícula: {e}")
            return {}
    
    async def _run_random_search_optimization(self, config: OptimizationConfig) -> Dict[str, Any]:
        """Ejecuta búsqueda aleatoria"""
        try:
            best_params = {}
            best_score = float('-inf')
            
            for _ in range(config.max_iterations):
                # Generar parámetros aleatorios
                params = {}
                
                if config.optimization_type == OptimizationType.MODEL_PERFORMANCE:
                    params['learning_rate'] = np.random.uniform(0.001, 0.1)
                    params['batch_size'] = np.random.choice([16, 32, 64, 128])
                    params['hidden_units'] = np.random.randint(32, 257)
                elif config.optimization_type == OptimizationType.SYSTEM_RESOURCES:
                    params['worker_threads'] = np.random.randint(1, 17)
                    params['cache_size'] = np.random.randint(100, 10001)
                    params['connection_pool'] = np.random.randint(5, 51)
                
                # Evaluar objetivo
                score = await self._evaluate_optimization_objective(config, params)
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error en búsqueda aleatoria: {e}")
            return {}
    
    async def _run_genetic_optimization(self, config: OptimizationConfig) -> Dict[str, Any]:
        """Ejecuta optimización genética"""
        try:
            genetic_optimizer = self.optimization_models.get('genetic_optimizer')
            if not genetic_optimizer:
                return {}
            
            def objective_function(individual):
                # Convertir individuo a parámetros
                params = self._individual_to_params(individual, config)
                return asyncio.run(self._evaluate_optimization_objective(config, params))
            
            # Definir límites
            bounds = self._get_optimization_bounds(config)
            
            # Ejecutar optimización genética
            best_solution, best_fitness = genetic_optimizer.optimize(objective_function, bounds)
            
            return self._individual_to_params(best_solution, config)
            
        except Exception as e:
            logger.error(f"Error en optimización genética: {e}")
            return {}
    
    async def _run_reinforcement_optimization(self, config: OptimizationConfig) -> Dict[str, Any]:
        """Ejecuta optimización por aprendizaje por refuerzo"""
        try:
            rl_optimizer = self.optimization_models.get('reinforcement_optimizer')
            if not rl_optimizer:
                return {}
            
            # Crear entorno de optimización
            environment = self._create_optimization_environment(config)
            
            # Ejecutar optimización RL
            q_table = rl_optimizer.optimize(environment)
            
            # Extraer mejores parámetros del Q-table
            best_params = self._extract_best_params_from_qtable(q_table, config)
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error en optimización por RL: {e}")
            return {}
    
    async def _run_default_optimization(self, config: OptimizationConfig) -> Dict[str, Any]:
        """Ejecuta optimización por defecto"""
        try:
            # Optimización simple basada en reglas
            params = {}
            
            if config.optimization_type == OptimizationType.MODEL_PERFORMANCE:
                params = {
                    'learning_rate': 0.01,
                    'batch_size': 32,
                    'hidden_units': 128
                }
            elif config.optimization_type == OptimizationType.SYSTEM_RESOURCES:
                params = {
                    'worker_threads': 8,
                    'cache_size': 5000,
                    'connection_pool': 20
                }
            elif config.optimization_type == OptimizationType.API_RESPONSE_TIME:
                params = {
                    'timeout': 1000,
                    'retry_attempts': 3,
                    'concurrent_requests': 50
                }
            
            return params
            
        except Exception as e:
            logger.error(f"Error en optimización por defecto: {e}")
            return {}
    
    async def _evaluate_optimization_objective(self, config: OptimizationConfig, params: Dict[str, Any]) -> float:
        """Evalúa objetivo de optimización"""
        try:
            # Simular evaluación de objetivo
            # En implementación real, aplicar parámetros y medir rendimiento
            
            score = 0.0
            
            # Simular mejora basada en parámetros
            if config.optimization_type == OptimizationType.MODEL_PERFORMANCE:
                # Simular mejora en precisión
                score += params.get('learning_rate', 0.01) * 100
                score += params.get('batch_size', 32) * 0.1
                score += params.get('hidden_units', 128) * 0.01
            elif config.optimization_type == OptimizationType.SYSTEM_RESOURCES:
                # Simular mejora en eficiencia de recursos
                score += (16 - params.get('worker_threads', 8)) * 0.5
                score += params.get('cache_size', 5000) * 0.001
                score += (50 - params.get('connection_pool', 20)) * 0.2
            elif config.optimization_type == OptimizationType.API_RESPONSE_TIME:
                # Simular mejora en tiempo de respuesta
                score += (5000 - params.get('timeout', 1000)) * 0.01
                score += (5 - params.get('retry_attempts', 3)) * 0.5
                score += params.get('concurrent_requests', 50) * 0.1
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluando objetivo de optimización: {e}")
            return 0.0
    
    async def _measure_optimized_metrics(self, target_metrics: List[PerformanceMetric], params: Dict[str, Any]) -> Dict[str, float]:
        """Mide métricas optimizadas"""
        try:
            # Simular medición de métricas optimizadas
            optimized_metrics = {}
            
            for metric in target_metrics:
                metric_name = metric.value
                
                # Simular mejora basada en parámetros
                if metric_name == "response_time":
                    # Simular reducción en tiempo de respuesta
                    improvement = params.get('timeout', 1000) * 0.1
                    optimized_metrics[metric_name] = max(10.0, 100.0 - improvement)
                elif metric_name == "throughput":
                    # Simular aumento en throughput
                    improvement = params.get('concurrent_requests', 50) * 2
                    optimized_metrics[metric_name] = 150.0 + improvement
                elif metric_name == "cpu_usage":
                    # Simular reducción en uso de CPU
                    improvement = params.get('worker_threads', 8) * 0.5
                    optimized_metrics[metric_name] = max(5.0, 30.0 - improvement)
                elif metric_name == "memory_usage":
                    # Simular reducción en uso de memoria
                    improvement = params.get('cache_size', 5000) * 0.001
                    optimized_metrics[metric_name] = max(10.0, 40.0 - improvement)
                else:
                    # Métrica por defecto
                    optimized_metrics[metric_name] = 50.0
            
            return optimized_metrics
            
        except Exception as e:
            logger.error(f"Error midiendo métricas optimizadas: {e}")
            return {}
    
    async def _calculate_improvement(self, baseline_metrics: Dict[str, float], optimized_metrics: Dict[str, float]) -> float:
        """Calcula mejora porcentual"""
        try:
            improvements = []
            
            for metric_name in baseline_metrics:
                if metric_name in optimized_metrics:
                    baseline = baseline_metrics[metric_name]
                    optimized = optimized_metrics[metric_name]
                    
                    if baseline > 0:
                        # Para métricas donde menor es mejor (tiempo, uso de recursos)
                        if metric_name in ["response_time", "cpu_usage", "memory_usage", "error_rate"]:
                            improvement = (baseline - optimized) / baseline * 100
                        else:
                            # Para métricas donde mayor es mejor (throughput, precisión)
                            improvement = (optimized - baseline) / baseline * 100
                        
                        improvements.append(improvement)
            
            return np.mean(improvements) if improvements else 0.0
            
        except Exception as e:
            logger.error(f"Error calculando mejora: {e}")
            return 0.0
    
    def _individual_to_params(self, individual, config: OptimizationConfig) -> Dict[str, Any]:
        """Convierte individuo genético a parámetros"""
        try:
            params = {}
            
            if config.optimization_type == OptimizationType.MODEL_PERFORMANCE:
                params['learning_rate'] = individual[0] if len(individual) > 0 else 0.01
                params['batch_size'] = int(individual[1]) if len(individual) > 1 else 32
                params['hidden_units'] = int(individual[2]) if len(individual) > 2 else 128
            elif config.optimization_type == OptimizationType.SYSTEM_RESOURCES:
                params['worker_threads'] = int(individual[0]) if len(individual) > 0 else 8
                params['cache_size'] = int(individual[1]) if len(individual) > 1 else 5000
                params['connection_pool'] = int(individual[2]) if len(individual) > 2 else 20
            
            return params
            
        except Exception as e:
            logger.error(f"Error convirtiendo individuo a parámetros: {e}")
            return {}
    
    def _get_optimization_bounds(self, config: OptimizationConfig) -> List[Tuple[float, float]]:
        """Obtiene límites de optimización"""
        try:
            bounds = []
            
            if config.optimization_type == OptimizationType.MODEL_PERFORMANCE:
                bounds = [
                    (0.001, 0.1),    # learning_rate
                    (16, 128),       # batch_size
                    (32, 256)        # hidden_units
                ]
            elif config.optimization_type == OptimizationType.SYSTEM_RESOURCES:
                bounds = [
                    (1, 16),         # worker_threads
                    (100, 10000),    # cache_size
                    (5, 50)          # connection_pool
                ]
            
            return bounds
            
        except Exception as e:
            logger.error(f"Error obteniendo límites de optimización: {e}")
            return []
    
    def _create_optimization_environment(self, config: OptimizationConfig):
        """Crea entorno de optimización"""
        try:
            # Implementación simple de entorno
            class OptimizationEnvironment:
                def __init__(self, config):
                    self.config = config
                    self.state = 0
                    self.max_steps = 100
                
                def reset(self):
                    self.state = 0
                    return self.state
                
                def step(self, action):
                    # Simular paso del entorno
                    self.state = (self.state + 1) % 10
                    reward = np.random.normal(0, 1)  # Recompensa simulada
                    done = self.state >= self.max_steps
                    
                    return self.state, reward, done
            
            return OptimizationEnvironment(config)
            
        except Exception as e:
            logger.error(f"Error creando entorno de optimización: {e}")
            return None
    
    def _extract_best_params_from_qtable(self, q_table: np.ndarray, config: OptimizationConfig) -> Dict[str, Any]:
        """Extrae mejores parámetros del Q-table"""
        try:
            # Encontrar mejor acción para cada estado
            best_actions = np.argmax(q_table, axis=1)
            
            # Convertir a parámetros
            params = {}
            
            if config.optimization_type == OptimizationType.API_RESPONSE_TIME:
                params['timeout'] = 1000 + best_actions[0] * 100
                params['retry_attempts'] = 1 + best_actions[1] % 5
                params['concurrent_requests'] = 10 + best_actions[2] * 10
            
            return params
            
        except Exception as e:
            logger.error(f"Error extrayendo parámetros del Q-table: {e}")
            return {}
    
    async def _update_performance_baselines(self):
        """Actualiza líneas base de rendimiento"""
        try:
            if not self.system_metrics:
                return
            
            # Calcular promedios de las últimas 24 horas
            recent_metrics = self.system_metrics[-1440:]  # 24 horas de métricas (1 por minuto)
            
            if not recent_metrics:
                return
            
            # Actualizar líneas base
            for baseline in self.performance_baselines.values():
                metric_name = baseline.metric_name
                
                # Obtener valores recientes
                values = []
                for metric in recent_metrics:
                    if hasattr(metric, metric_name):
                        values.append(getattr(metric, metric_name))
                
                if values:
                    # Actualizar valor actual
                    baseline.current_value = np.mean(values)
                    baseline.measurement_time = datetime.now()
                    
                    # Ajustar objetivos si es necesario
                    if baseline.current_value > baseline.threshold_max:
                        baseline.target_value = baseline.threshold_max * 0.8
                    elif baseline.current_value < baseline.threshold_min:
                        baseline.target_value = baseline.threshold_min * 1.2
            
            logger.info("Líneas base de rendimiento actualizadas")
            
        except Exception as e:
            logger.error(f"Error actualizando líneas base de rendimiento: {e}")
    
    async def _analyze_performance_trends(self):
        """Analiza tendencias de rendimiento"""
        try:
            if len(self.system_metrics) < 10:
                return
            
            # Analizar tendencias por métrica
            trends = {}
            
            for baseline in self.performance_baselines.values():
                metric_name = baseline.metric_name
                
                # Obtener valores recientes
                values = []
                timestamps = []
                
                for metric in self.system_metrics[-100:]:  # Últimas 100 mediciones
                    if hasattr(metric, metric_name):
                        values.append(getattr(metric, metric_name))
                        timestamps.append(metric.timestamp.timestamp())
                
                if len(values) > 5:
                    # Calcular tendencia
                    x = np.array(timestamps)
                    y = np.array(values)
                    
                    # Regresión lineal
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    trends[metric_name] = {
                        "slope": slope,
                        "r_squared": r_value ** 2,
                        "p_value": p_value,
                        "trend": "improving" if slope < 0 and metric_name in ["response_time", "cpu_usage", "memory_usage"] else "degrading" if slope > 0 else "stable"
                    }
            
            # Almacenar tendencias
            self.performance_metrics["trends"] = trends
            
        except Exception as e:
            logger.error(f"Error analizando tendencias de rendimiento: {e}")
    
    async def _generate_performance_reports(self):
        """Genera reportes de rendimiento"""
        try:
            # Crear reporte de rendimiento
            report = {
                "timestamp": datetime.now().isoformat(),
                "baselines": {
                    name: {
                        "current_value": baseline.current_value,
                        "target_value": baseline.target_value,
                        "threshold_min": baseline.threshold_min,
                        "threshold_max": baseline.threshold_max,
                        "unit": baseline.unit
                    }
                    for name, baseline in self.performance_baselines.items()
                },
                "recent_optimizations": [
                    {
                        "id": result.id,
                        "type": result.optimization_type.value,
                        "strategy": result.strategy.value,
                        "improvement": result.improvement_percentage,
                        "status": result.status,
                        "created_at": result.created_at.isoformat()
                    }
                    for result in self.optimization_results[-10:]  # Últimos 10 resultados
                ],
                "trends": self.performance_metrics.get("trends", {}),
                "system_health": await self._calculate_system_health()
            }
            
            # Guardar reporte
            report_file = Path(f"data/performance_reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Reporte de rendimiento generado: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generando reportes de rendimiento: {e}")
    
    async def _calculate_system_health(self) -> Dict[str, Any]:
        """Calcula salud del sistema"""
        try:
            if not self.system_metrics:
                return {"status": "unknown", "score": 0.0}
            
            # Obtener métricas recientes
            recent_metrics = self.system_metrics[-10:]
            
            # Calcular puntuación de salud
            health_score = 100.0
            
            # Penalizar por métricas fuera de umbral
            for baseline in self.performance_baselines.values():
                metric_name = baseline.metric_name
                
                values = []
                for metric in recent_metrics:
                    if hasattr(metric, metric_name):
                        values.append(getattr(metric, metric_name))
                
                if values:
                    avg_value = np.mean(values)
                    
                    if avg_value > baseline.threshold_max:
                        penalty = (avg_value - baseline.threshold_max) / baseline.threshold_max * 20
                        health_score -= penalty
                    elif avg_value < baseline.threshold_min:
                        penalty = (baseline.threshold_min - avg_value) / baseline.threshold_min * 20
                        health_score -= penalty
            
            # Determinar estado
            if health_score >= 90:
                status = "excellent"
            elif health_score >= 75:
                status = "good"
            elif health_score >= 50:
                status = "fair"
            elif health_score >= 25:
                status = "poor"
            else:
                status = "critical"
            
            return {
                "status": status,
                "score": max(0.0, min(100.0, health_score)),
                "recommendations": await self._generate_health_recommendations(health_score)
            }
            
        except Exception as e:
            logger.error(f"Error calculando salud del sistema: {e}")
            return {"status": "error", "score": 0.0}
    
    async def _generate_health_recommendations(self, health_score: float) -> List[str]:
        """Genera recomendaciones de salud"""
        try:
            recommendations = []
            
            if health_score < 50:
                recommendations.append("Sistema requiere optimización inmediata")
                recommendations.append("Revisar configuración de recursos")
                recommendations.append("Considerar escalado horizontal")
            elif health_score < 75:
                recommendations.append("Monitorear métricas de rendimiento")
                recommendations.append("Optimizar configuraciones actuales")
            else:
                recommendations.append("Sistema funcionando correctamente")
                recommendations.append("Continuar monitoreo preventivo")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones de salud: {e}")
            return []
    
    async def _setup_mlflow(self):
        """Configura MLflow para tracking de experimentos"""
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(self.mlflow_experiment_name)
            logger.info("MLflow configurado para tracking de experimentos")
            
        except Exception as e:
            logger.error(f"Error configurando MLflow: {e}")
    
    async def get_optimization_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos para dashboard de optimización"""
        try:
            # Estadísticas generales
            total_optimizations = len(self.optimization_results)
            successful_optimizations = len([r for r in self.optimization_results if r.status == "success"])
            avg_improvement = np.mean([r.improvement_percentage for r in self.optimization_results]) if self.optimization_results else 0.0
            
            # Métricas actuales
            current_metrics = await self._collect_current_metrics()
            
            # Tendencias de rendimiento
            trends = self.performance_metrics.get("trends", {})
            
            # Salud del sistema
            system_health = await self._calculate_system_health()
            
            # Optimizaciones recientes
            recent_optimizations = [
                {
                    "id": result.id,
                    "type": result.optimization_type.value,
                    "strategy": result.strategy.value,
                    "improvement": result.improvement_percentage,
                    "status": result.status,
                    "execution_time": result.execution_time,
                    "created_at": result.created_at.isoformat()
                }
                for result in sorted(self.optimization_results, key=lambda x: x.created_at, reverse=True)[:10]
            ]
            
            return {
                "total_optimizations": total_optimizations,
                "successful_optimizations": successful_optimizations,
                "success_rate": successful_optimizations / total_optimizations if total_optimizations > 0 else 0.0,
                "average_improvement": avg_improvement,
                "current_metrics": current_metrics,
                "performance_baselines": {
                    name: {
                        "current_value": baseline.current_value,
                        "target_value": baseline.target_value,
                        "unit": baseline.unit
                    }
                    for name, baseline in self.performance_baselines.items()
                },
                "trends": trends,
                "system_health": system_health,
                "recent_optimizations": recent_optimizations,
                "optimization_active": self.optimization_active,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos del dashboard de optimización: {e}")
            return {"error": str(e)}
    
    async def create_optimization_dashboard(self) -> str:
        """Crea dashboard de optimización con visualizaciones"""
        try:
            # Obtener datos del dashboard
            dashboard_data = await self.get_optimization_dashboard_data()
            
            # Crear visualizaciones
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Métricas de Rendimiento', 'Tendencias de Optimización', 
                              'Salud del Sistema', 'Optimizaciones Recientes'),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "indicator"}, {"type": "bar"}]]
            )
            
            # Gráfico de métricas de rendimiento
            if dashboard_data.get("performance_baselines"):
                baselines = dashboard_data["performance_baselines"]
                metrics = list(baselines.keys())
                current_values = [baselines[m]["current_value"] for m in metrics]
                target_values = [baselines[m]["target_value"] for m in metrics]
                
                fig.add_trace(
                    go.Bar(name="Valor Actual", x=metrics, y=current_values),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(name="Valor Objetivo", x=metrics, y=target_values),
                    row=1, col=1
                )
            
            # Gráfico de tendencias
            if dashboard_data.get("recent_optimizations"):
                optimizations = dashboard_data["recent_optimizations"]
                improvements = [opt["improvement"] for opt in optimizations]
                timestamps = [opt["created_at"] for opt in optimizations]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=improvements, mode='lines+markers', name="Mejora %"),
                    row=1, col=2
                )
            
            # Indicador de salud del sistema
            system_health = dashboard_data.get("system_health", {})
            health_score = system_health.get("score", 0.0)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=health_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Salud del Sistema"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 25], 'color': "lightgray"},
                               {'range': [25, 50], 'color': "yellow"},
                               {'range': [50, 75], 'color': "orange"},
                               {'range': [75, 100], 'color': "green"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 90}}
                ),
                row=2, col=1
            )
            
            # Gráfico de optimizaciones por tipo
            if dashboard_data.get("recent_optimizations"):
                opt_types = defaultdict(int)
                for opt in optimizations:
                    opt_types[opt["type"]] += 1
                
                fig.add_trace(
                    go.Bar(x=list(opt_types.keys()), y=list(opt_types.values()), name="Optimizaciones"),
                    row=2, col=2
                )
            
            # Configurar layout
            fig.update_layout(
                title="Dashboard de Optimización de Rendimiento AI",
                showlegend=True,
                height=800
            )
            
            # Convertir a HTML
            dashboard_html = fig.to_html(include_plotlyjs='cdn')
            
            return dashboard_html
            
        except Exception as e:
            logger.error(f"Error creando dashboard de optimización: {e}")
            return f"<html><body><h1>Error creando dashboard: {str(e)}</h1></body></html>"


















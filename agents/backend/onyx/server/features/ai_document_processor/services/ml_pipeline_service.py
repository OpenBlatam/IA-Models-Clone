"""
Servicio de Pipeline de Machine Learning
========================================

Servicio para crear y gestionar pipelines de machine learning automatizados.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import hashlib

logger = logging.getLogger(__name__)

class PipelineStage(str, Enum):
    """Etapas del pipeline"""
    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_SELECTION = "feature_selection"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_VALIDATION = "model_validation"
    PREDICTION = "prediction"
    POSTPROCESSING = "postprocessing"

class PipelineStatus(str, Enum):
    """Estados del pipeline"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PipelineStep:
    """Paso del pipeline"""
    id: str
    name: str
    stage: PipelineStage
    component_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    output_schema: Optional[Dict[str, Any]] = None
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0

@dataclass
class MLPipeline:
    """Pipeline de Machine Learning"""
    id: str
    name: str
    description: str
    version: str
    steps: List[PipelineStep]
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class PipelineExecution:
    """Ejecución del pipeline"""
    id: str
    pipeline_id: str
    status: PipelineStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    input_data: Any = None
    output_data: Any = None
    step_results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

class MLPipelineService:
    """Servicio de Pipeline de Machine Learning"""
    
    def __init__(self):
        self.pipelines: Dict[str, MLPipeline] = {}
        self.executions: Dict[str, PipelineExecution] = {}
        self.running_executions: Dict[str, asyncio.Task] = {}
        self.components: Dict[str, Any] = {}
        self.data_cache: Dict[str, Any] = {}
        
    async def initialize(self):
        """Inicializa el servicio de pipeline"""
        logger.info("Inicializando servicio de pipeline ML...")
        
        # Registrar componentes por defecto
        await self._register_default_components()
        
        # Cargar pipelines predefinidos
        await self._load_predefined_pipelines()
        
        logger.info("Servicio de pipeline ML inicializado")
    
    async def _register_default_components(self):
        """Registra componentes por defecto"""
        try:
            # Componentes de preprocesamiento
            self.components = {
                # Data Loading
                "csv_loader": self._csv_loader,
                "json_loader": self._json_loader,
                "text_loader": self._text_loader,
                
                # Preprocessing
                "text_cleaner": self._text_cleaner,
                "text_tokenizer": self._text_tokenizer,
                "text_normalizer": self._text_normalizer,
                "missing_value_handler": self._missing_value_handler,
                "outlier_detector": self._outlier_detector,
                
                # Feature Engineering
                "tfidf_vectorizer": self._tfidf_vectorizer,
                "word2vec_embedder": self._word2vec_embedder,
                "sentiment_analyzer": self._sentiment_analyzer,
                "entity_extractor": self._entity_extractor,
                "topic_modeler": self._topic_modeler,
                
                # Feature Selection
                "chi2_selector": self._chi2_selector,
                "mutual_info_selector": self._mutual_info_selector,
                "rfe_selector": self._rfe_selector,
                
                # Model Training
                "random_forest_trainer": self._random_forest_trainer,
                "svm_trainer": self._svm_trainer,
                "logistic_regression_trainer": self._logistic_regression_trainer,
                "naive_bayes_trainer": self._naive_bayes_trainer,
                "neural_network_trainer": self._neural_network_trainer,
                
                # Model Evaluation
                "cross_validator": self._cross_validator,
                "performance_evaluator": self._performance_evaluator,
                "confusion_matrix_generator": self._confusion_matrix_generator,
                
                # Prediction
                "predictor": self._predictor,
                "probability_predictor": self._probability_predictor,
                
                # Postprocessing
                "result_formatter": self._result_formatter,
                "confidence_calculator": self._confidence_calculator
            }
            
            logger.info(f"Registrados {len(self.components)} componentes de pipeline")
            
        except Exception as e:
            logger.error(f"Error registrando componentes: {e}")
    
    async def _load_predefined_pipelines(self):
        """Carga pipelines predefinidos"""
        try:
            # Pipeline de clasificación de texto
            text_classification_pipeline = MLPipeline(
                id="text_classification",
                name="Clasificación de Texto",
                description="Pipeline completo para clasificación de documentos de texto",
                version="1.0.0",
                steps=[
                    PipelineStep(
                        id="load_data",
                        name="Cargar Datos",
                        stage=PipelineStage.DATA_LOADING,
                        component_type="csv_loader",
                        parameters={"file_path": "data/training_data.csv"}
                    ),
                    PipelineStep(
                        id="clean_text",
                        name="Limpiar Texto",
                        stage=PipelineStage.PREPROCESSING,
                        component_type="text_cleaner",
                        dependencies=["load_data"],
                        parameters={"remove_html": True, "remove_special_chars": True}
                    ),
                    PipelineStep(
                        id="tokenize_text",
                        name="Tokenizar Texto",
                        stage=PipelineStage.PREPROCESSING,
                        component_type="text_tokenizer",
                        dependencies=["clean_text"],
                        parameters={"language": "spanish"}
                    ),
                    PipelineStep(
                        id="vectorize_text",
                        name="Vectorizar Texto",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        component_type="tfidf_vectorizer",
                        dependencies=["tokenize_text"],
                        parameters={"max_features": 10000, "ngram_range": [1, 2]}
                    ),
                    PipelineStep(
                        id="select_features",
                        name="Seleccionar Características",
                        stage=PipelineStage.FEATURE_SELECTION,
                        component_type="chi2_selector",
                        dependencies=["vectorize_text"],
                        parameters={"k": 5000}
                    ),
                    PipelineStep(
                        id="train_model",
                        name="Entrenar Modelo",
                        stage=PipelineStage.MODEL_TRAINING,
                        component_type="random_forest_trainer",
                        dependencies=["select_features"],
                        parameters={"n_estimators": 100, "max_depth": 10}
                    ),
                    PipelineStep(
                        id="evaluate_model",
                        name="Evaluar Modelo",
                        stage=PipelineStage.MODEL_EVALUATION,
                        component_type="cross_validator",
                        dependencies=["train_model"],
                        parameters={"cv_folds": 5}
                    )
                ]
            )
            
            # Pipeline de análisis de sentimientos
            sentiment_analysis_pipeline = MLPipeline(
                id="sentiment_analysis",
                name="Análisis de Sentimientos",
                description="Pipeline para análisis de sentimientos en texto",
                version="1.0.0",
                steps=[
                    PipelineStep(
                        id="load_text",
                        name="Cargar Texto",
                        stage=PipelineStage.DATA_LOADING,
                        component_type="text_loader",
                        parameters={"encoding": "utf-8"}
                    ),
                    PipelineStep(
                        id="preprocess_text",
                        name="Preprocesar Texto",
                        stage=PipelineStage.PREPROCESSING,
                        component_type="text_cleaner",
                        dependencies=["load_text"],
                        parameters={"lowercase": True, "remove_stopwords": True}
                    ),
                    PipelineStep(
                        id="analyze_sentiment",
                        name="Analizar Sentimiento",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        component_type="sentiment_analyzer",
                        dependencies=["preprocess_text"],
                        parameters={"method": "vader"}
                    ),
                    PipelineStep(
                        id="format_results",
                        name="Formatear Resultados",
                        stage=PipelineStage.POSTPROCESSING,
                        component_type="result_formatter",
                        dependencies=["analyze_sentiment"],
                        parameters={"output_format": "json"}
                    )
                ]
            )
            
            # Pipeline de extracción de entidades
            entity_extraction_pipeline = MLPipeline(
                id="entity_extraction",
                name="Extracción de Entidades",
                description="Pipeline para extracción de entidades nombradas",
                version="1.0.0",
                steps=[
                    PipelineStep(
                        id="load_documents",
                        name="Cargar Documentos",
                        stage=PipelineStage.DATA_LOADING,
                        component_type="json_loader",
                        parameters={"key_field": "text"}
                    ),
                    PipelineStep(
                        id="normalize_text",
                        name="Normalizar Texto",
                        stage=PipelineStage.PREPROCESSING,
                        component_type="text_normalizer",
                        dependencies=["load_documents"],
                        parameters={"language": "spanish"}
                    ),
                    PipelineStep(
                        id="extract_entities",
                        name="Extraer Entidades",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        component_type="entity_extractor",
                        dependencies=["normalize_text"],
                        parameters={"entity_types": ["PERSON", "ORG", "LOC"]}
                    ),
                    PipelineStep(
                        id="calculate_confidence",
                        name="Calcular Confianza",
                        stage=PipelineStage.POSTPROCESSING,
                        component_type="confidence_calculator",
                        dependencies=["extract_entities"],
                        parameters={"threshold": 0.7}
                    )
                ]
            )
            
            # Guardar pipelines
            self.pipelines["text_classification"] = text_classification_pipeline
            self.pipelines["sentiment_analysis"] = sentiment_analysis_pipeline
            self.pipelines["entity_extraction"] = entity_extraction_pipeline
            
            logger.info(f"Cargados {len(self.pipelines)} pipelines predefinidos")
            
        except Exception as e:
            logger.error(f"Error cargando pipelines predefinidos: {e}")
    
    async def create_pipeline(self, pipeline: MLPipeline) -> bool:
        """Crea un nuevo pipeline"""
        try:
            self.pipelines[pipeline.id] = pipeline
            logger.info(f"Pipeline creado: {pipeline.id}")
            return True
        except Exception as e:
            logger.error(f"Error creando pipeline: {e}")
            return False
    
    async def execute_pipeline(
        self,
        pipeline_id: str,
        input_data: Any = None,
        parameters: Dict[str, Any] = None
    ) -> str:
        """Ejecuta un pipeline"""
        try:
            if pipeline_id not in self.pipelines:
                raise ValueError(f"Pipeline no encontrado: {pipeline_id}")
            
            pipeline = self.pipelines[pipeline_id]
            
            # Crear ejecución
            execution_id = f"{pipeline_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            execution = PipelineExecution(
                id=execution_id,
                pipeline_id=pipeline_id,
                status=PipelineStatus.PENDING,
                started_at=datetime.now(),
                input_data=input_data
            )
            
            self.executions[execution_id] = execution
            
            # Ejecutar pipeline
            task = asyncio.create_task(self._execute_pipeline(execution, parameters))
            self.running_executions[execution_id] = task
            
            logger.info(f"Pipeline ejecutado: {pipeline_id} (execution: {execution_id})")
            return execution_id
            
        except Exception as e:
            logger.error(f"Error ejecutando pipeline: {e}")
            raise
    
    async def _execute_pipeline(self, execution: PipelineExecution, parameters: Dict[str, Any] = None):
        """Ejecuta un pipeline específico"""
        try:
            execution.status = PipelineStatus.RUNNING
            pipeline = self.pipelines[execution.pipeline_id]
            
            # Ejecutar pasos en orden de dependencias
            completed_steps = set()
            current_data = execution.input_data
            
            while len(completed_steps) < len(pipeline.steps):
                # Encontrar pasos listos para ejecutar
                ready_steps = []
                for step in pipeline.steps:
                    if (step.id not in completed_steps and 
                        step.status == "pending" and
                        all(dep in completed_steps for dep in step.dependencies)):
                        ready_steps.append(step)
                
                if not ready_steps:
                    # No hay pasos listos, verificar errores
                    failed_steps = [s for s in pipeline.steps if s.status == "failed"]
                    if failed_steps:
                        execution.status = PipelineStatus.FAILED
                        execution.error = f"Pasos fallidos: {[s.id for s in failed_steps]}"
                        break
                    else:
                        # Esperar un poco y reintentar
                        await asyncio.sleep(1)
                        continue
                
                # Ejecutar pasos listos en paralelo si no tienen dependencias entre sí
                for step in ready_steps:
                    try:
                        step.status = "running"
                        step_start = datetime.now()
                        
                        # Obtener componente
                        if step.component_type not in self.components:
                            raise ValueError(f"Componente no encontrado: {step.component_type}")
                        
                        component = self.components[step.component_type]
                        
                        # Preparar parámetros
                        step_params = step.parameters.copy()
                        if parameters:
                            step_params.update(parameters)
                        
                        # Ejecutar componente
                        result = await component(current_data, step_params, execution.step_results)
                        
                        # Guardar resultado
                        step.result = result
                        step.status = "completed"
                        step.execution_time = (datetime.now() - step_start).total_seconds()
                        execution.step_results[step.id] = result
                        
                        # Actualizar datos para siguiente paso
                        if step.stage in [PipelineStage.FEATURE_ENGINEERING, PipelineStage.MODEL_TRAINING]:
                            current_data = result
                        
                        completed_steps.add(step.id)
                        
                    except Exception as e:
                        step.status = "failed"
                        step.error = str(e)
                        step.execution_time = (datetime.now() - step_start).total_seconds()
                        logger.error(f"Error en paso {step.id}: {e}")
            
            # Finalizar ejecución
            if execution.status == PipelineStatus.RUNNING:
                execution.status = PipelineStatus.COMPLETED
                execution.completed_at = datetime.now()
                execution.output_data = current_data
            
            # Limpiar ejecución después de un tiempo
            asyncio.create_task(self._cleanup_execution(execution.id))
            
        except Exception as e:
            logger.error(f"Error ejecutando pipeline {execution.id}: {e}")
            execution.status = PipelineStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()
    
    # Componentes de Data Loading
    async def _csv_loader(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> pd.DataFrame:
        """Carga datos desde CSV"""
        try:
            file_path = params.get("file_path")
            if not file_path:
                raise ValueError("file_path no especificado")
            
            df = pd.read_csv(file_path, encoding=params.get("encoding", "utf-8"))
            logger.info(f"Datos cargados desde CSV: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error cargando CSV: {e}")
            raise
    
    async def _json_loader(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Carga datos desde JSON"""
        try:
            file_path = params.get("file_path")
            if not file_path:
                raise ValueError("file_path no especificado")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            logger.info(f"Datos cargados desde JSON: {len(json_data)} elementos")
            return json_data
            
        except Exception as e:
            logger.error(f"Error cargando JSON: {e}")
            raise
    
    async def _text_loader(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> str:
        """Carga texto desde archivo"""
        try:
            file_path = params.get("file_path")
            if not file_path:
                raise ValueError("file_path no especificado")
            
            with open(file_path, 'r', encoding=params.get("encoding", "utf-8")) as f:
                text = f.read()
            
            logger.info(f"Texto cargado: {len(text)} caracteres")
            return text
            
        except Exception as e:
            logger.error(f"Error cargando texto: {e}")
            raise
    
    # Componentes de Preprocessing
    async def _text_cleaner(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Limpia texto"""
        try:
            import re
            
            if isinstance(data, str):
                text = data
            elif isinstance(data, pd.DataFrame):
                text_column = params.get("text_column", "text")
                text = data[text_column].str.cat(sep=" ")
            else:
                text = str(data)
            
            # Aplicar limpieza según parámetros
            if params.get("remove_html", False):
                text = re.sub(r'<[^>]+>', '', text)
            
            if params.get("remove_special_chars", False):
                text = re.sub(r'[^\w\s]', '', text)
            
            if params.get("lowercase", False):
                text = text.lower()
            
            if params.get("remove_stopwords", False):
                # Lista básica de stopwords en español
                stopwords = {"el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te", "lo", "le", "da", "su", "por", "son", "con", "para", "al", "del", "los", "las", "una", "pero", "sus", "las", "más", "muy", "ya", "todo", "esta", "está", "están", "estas", "estos", "como", "más", "pero", "sus", "le", "ha", "me", "si", "sin", "sobre", "este", "ya", "entre", "cuando", "todo", "esta", "ser", "son", "dos", "también", "fue", "había", "era", "eran", "sido", "estado", "estados", "estaba", "fueron", "será", "serán", "habrá", "habrán", "había", "habían", "hubo", "hubieron"}
                words = text.split()
                text = " ".join([word for word in words if word.lower() not in stopwords])
            
            logger.info(f"Texto limpiado: {len(text)} caracteres")
            return text
            
        except Exception as e:
            logger.error(f"Error limpiando texto: {e}")
            raise
    
    async def _text_tokenizer(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> List[str]:
        """Tokeniza texto"""
        try:
            if isinstance(data, str):
                text = data
            else:
                text = str(data)
            
            # Tokenización simple
            tokens = text.split()
            
            # Filtrar tokens por longitud mínima
            min_length = params.get("min_length", 2)
            tokens = [token for token in tokens if len(token) >= min_length]
            
            logger.info(f"Texto tokenizado: {len(tokens)} tokens")
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizando texto: {e}")
            raise
    
    async def _text_normalizer(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> str:
        """Normaliza texto"""
        try:
            if isinstance(data, str):
                text = data
            else:
                text = str(data)
            
            # Normalización básica
            import unicodedata
            
            # Normalizar unicode
            text = unicodedata.normalize('NFKD', text)
            
            # Remover acentos (opcional)
            if params.get("remove_accents", False):
                text = ''.join(c for c in text if not unicodedata.combining(c))
            
            # Normalizar espacios
            text = ' '.join(text.split())
            
            logger.info(f"Texto normalizado: {len(text)} caracteres")
            return text
            
        except Exception as e:
            logger.error(f"Error normalizando texto: {e}")
            raise
    
    async def _missing_value_handler(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Maneja valores faltantes"""
        try:
            if isinstance(data, pd.DataFrame):
                strategy = params.get("strategy", "drop")
                
                if strategy == "drop":
                    data = data.dropna()
                elif strategy == "fill":
                    fill_value = params.get("fill_value", "")
                    data = data.fillna(fill_value)
                elif strategy == "mean":
                    data = data.fillna(data.mean())
                
                logger.info(f"Valores faltantes manejados: {data.shape}")
                return data
            else:
                return data
                
        except Exception as e:
            logger.error(f"Error manejando valores faltantes: {e}")
            raise
    
    async def _outlier_detector(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta outliers"""
        try:
            if isinstance(data, pd.DataFrame):
                # Detección simple de outliers usando IQR
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                outliers = {}
                
                for column in numeric_columns:
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
                    outliers[column] = outlier_mask.sum()
                
                logger.info(f"Outliers detectados: {outliers}")
                return {"outliers": outliers, "data": data}
            else:
                return {"outliers": {}, "data": data}
                
        except Exception as e:
            logger.error(f"Error detectando outliers: {e}")
            raise
    
    # Componentes de Feature Engineering
    async def _tfidf_vectorizer(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Vectoriza texto usando TF-IDF"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            if isinstance(data, list):
                texts = data
            elif isinstance(data, str):
                texts = [data]
            else:
                texts = [str(data)]
            
            vectorizer = TfidfVectorizer(
                max_features=params.get("max_features", 10000),
                ngram_range=tuple(params.get("ngram_range", [1, 2])),
                stop_words=params.get("stop_words", None)
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            logger.info(f"TF-IDF vectorizado: {tfidf_matrix.shape}")
            return {
                "matrix": tfidf_matrix,
                "vectorizer": vectorizer,
                "feature_names": vectorizer.get_feature_names_out()
            }
            
        except Exception as e:
            logger.error(f"Error vectorizando TF-IDF: {e}")
            raise
    
    async def _word2vec_embedder(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Genera embeddings Word2Vec"""
        try:
            # Implementación simplificada
            if isinstance(data, list):
                tokens = data
            else:
                tokens = str(data).split()
            
            # Simular embeddings (en implementación real usar gensim)
            embedding_dim = params.get("embedding_dim", 100)
            embeddings = np.random.rand(len(tokens), embedding_dim)
            
            logger.info(f"Embeddings generados: {embeddings.shape}")
            return {
                "embeddings": embeddings,
                "tokens": tokens,
                "dimension": embedding_dim
            }
            
        except Exception as e:
            logger.error(f"Error generando embeddings: {e}")
            raise
    
    async def _sentiment_analyzer(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza sentimientos"""
        try:
            if isinstance(data, str):
                text = data
            else:
                text = str(data)
            
            # Análisis simple de sentimientos
            positive_words = ["bueno", "excelente", "fantástico", "genial", "perfecto", "maravilloso"]
            negative_words = ["malo", "terrible", "horrible", "pésimo", "defectuoso", "inútil"]
            
            words = text.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            total_sentiment_words = positive_count + negative_count
            if total_sentiment_words > 0:
                sentiment_score = (positive_count - negative_count) / total_sentiment_words
            else:
                sentiment_score = 0.0
            
            if sentiment_score > 0.1:
                sentiment = "positive"
            elif sentiment_score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            result = {
                "sentiment": sentiment,
                "score": sentiment_score,
                "positive_count": positive_count,
                "negative_count": negative_count
            }
            
            logger.info(f"Sentimiento analizado: {sentiment} (score: {sentiment_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error analizando sentimiento: {e}")
            raise
    
    async def _entity_extractor(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae entidades nombradas"""
        try:
            if isinstance(data, str):
                text = data
            else:
                text = str(data)
            
            # Extracción simple de entidades (en implementación real usar spaCy)
            import re
            
            entities = {
                "PERSON": [],
                "ORG": [],
                "LOC": []
            }
            
            # Patrones simples para detectar entidades
            person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
            org_pattern = r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|S\.A\.|S\.L\.)\b'
            loc_pattern = r'\b(?:en|de|del|la|el) [A-Z][a-z]+\b'
            
            entities["PERSON"] = re.findall(person_pattern, text)
            entities["ORG"] = re.findall(org_pattern, text)
            entities["LOC"] = re.findall(loc_pattern, text)
            
            result = {
                "entities": entities,
                "total_entities": sum(len(ents) for ents in entities.values())
            }
            
            logger.info(f"Entidades extraídas: {result['total_entities']}")
            return result
            
        except Exception as e:
            logger.error(f"Error extrayendo entidades: {e}")
            raise
    
    async def _topic_modeler(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Modela temas"""
        try:
            # Implementación simplificada de modelado de temas
            if isinstance(data, list):
                texts = data
            else:
                texts = [str(data)]
            
            num_topics = params.get("num_topics", 5)
            
            # Simular modelado de temas
            topics = []
            for i in range(num_topics):
                topics.append({
                    "topic_id": i,
                    "words": [f"word_{j}" for j in range(10)],
                    "weight": np.random.random()
                })
            
            result = {
                "topics": topics,
                "num_topics": num_topics
            }
            
            logger.info(f"Temas modelados: {num_topics}")
            return result
            
        except Exception as e:
            logger.error(f"Error modelando temas: {e}")
            raise
    
    # Componentes de Feature Selection
    async def _chi2_selector(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Selecciona características usando Chi2"""
        try:
            from sklearn.feature_selection import SelectKBest, chi2
            
            if isinstance(data, dict) and "matrix" in data:
                X = data["matrix"]
                feature_names = data["feature_names"]
            else:
                raise ValueError("Datos no compatibles para selección de características")
            
            k = params.get("k", 1000)
            selector = SelectKBest(score_func=chi2, k=k)
            
            # Necesitamos etiquetas para Chi2 (simuladas)
            y = np.random.randint(0, 2, X.shape[0])
            
            X_selected = selector.fit_transform(X, y)
            selected_features = feature_names[selector.get_support()]
            
            result = {
                "matrix": X_selected,
                "feature_names": selected_features,
                "selector": selector,
                "scores": selector.scores_
            }
            
            logger.info(f"Características seleccionadas: {X_selected.shape[1]}")
            return result
            
        except Exception as e:
            logger.error(f"Error seleccionando características: {e}")
            raise
    
    async def _mutual_info_selector(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Selecciona características usando información mutua"""
        try:
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            
            if isinstance(data, dict) and "matrix" in data:
                X = data["matrix"]
                feature_names = data["feature_names"]
            else:
                raise ValueError("Datos no compatibles para selección de características")
            
            k = params.get("k", 1000)
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            
            # Etiquetas simuladas
            y = np.random.randint(0, 2, X.shape[0])
            
            X_selected = selector.fit_transform(X, y)
            selected_features = feature_names[selector.get_support()]
            
            result = {
                "matrix": X_selected,
                "feature_names": selected_features,
                "selector": selector,
                "scores": selector.scores_
            }
            
            logger.info(f"Características seleccionadas (MI): {X_selected.shape[1]}")
            return result
            
        except Exception as e:
            logger.error(f"Error seleccionando características con MI: {e}")
            raise
    
    async def _rfe_selector(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Selecciona características usando RFE"""
        try:
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestClassifier
            
            if isinstance(data, dict) and "matrix" in data:
                X = data["matrix"]
                feature_names = data["feature_names"]
            else:
                raise ValueError("Datos no compatibles para selección de características")
            
            n_features = params.get("n_features", 1000)
            estimator = RandomForestClassifier(n_estimators=10, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features)
            
            # Etiquetas simuladas
            y = np.random.randint(0, 2, X.shape[0])
            
            X_selected = selector.fit_transform(X, y)
            selected_features = feature_names[selector.get_support()]
            
            result = {
                "matrix": X_selected,
                "feature_names": selected_features,
                "selector": selector
            }
            
            logger.info(f"Características seleccionadas (RFE): {X_selected.shape[1]}")
            return result
            
        except Exception as e:
            logger.error(f"Error seleccionando características con RFE: {e}")
            raise
    
    # Componentes de Model Training
    async def _random_forest_trainer(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Entrena modelo Random Forest"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            if isinstance(data, dict) and "matrix" in data:
                X = data["matrix"]
            else:
                raise ValueError("Datos no compatibles para entrenamiento")
            
            # Generar etiquetas simuladas
            y = np.random.randint(0, 2, X.shape[0])
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar modelo
            model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluar modelo
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            result = {
                "model": model,
                "train_score": train_score,
                "test_score": test_score,
                "feature_importance": model.feature_importances_.tolist()
            }
            
            logger.info(f"Random Forest entrenado - Train: {train_score:.3f}, Test: {test_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error entrenando Random Forest: {e}")
            raise
    
    async def _svm_trainer(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Entrena modelo SVM"""
        try:
            from sklearn.svm import SVC
            from sklearn.model_selection import train_test_split
            
            if isinstance(data, dict) and "matrix" in data:
                X = data["matrix"]
            else:
                raise ValueError("Datos no compatibles para entrenamiento")
            
            # Generar etiquetas simuladas
            y = np.random.randint(0, 2, X.shape[0])
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar modelo
            model = SVC(
                C=params.get("C", 1.0),
                kernel=params.get("kernel", "rbf"),
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluar modelo
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            result = {
                "model": model,
                "train_score": train_score,
                "test_score": test_score
            }
            
            logger.info(f"SVM entrenado - Train: {train_score:.3f}, Test: {test_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error entrenando SVM: {e}")
            raise
    
    async def _logistic_regression_trainer(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Entrena modelo Logistic Regression"""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            
            if isinstance(data, dict) and "matrix" in data:
                X = data["matrix"]
            else:
                raise ValueError("Datos no compatibles para entrenamiento")
            
            # Generar etiquetas simuladas
            y = np.random.randint(0, 2, X.shape[0])
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar modelo
            model = LogisticRegression(
                C=params.get("C", 1.0),
                penalty=params.get("penalty", "l2"),
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluar modelo
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            result = {
                "model": model,
                "train_score": train_score,
                "test_score": test_score,
                "coefficients": model.coef_.tolist()
            }
            
            logger.info(f"Logistic Regression entrenado - Train: {train_score:.3f}, Test: {test_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error entrenando Logistic Regression: {e}")
            raise
    
    async def _naive_bayes_trainer(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Entrena modelo Naive Bayes"""
        try:
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.model_selection import train_test_split
            
            if isinstance(data, dict) and "matrix" in data:
                X = data["matrix"]
            else:
                raise ValueError("Datos no compatibles para entrenamiento")
            
            # Generar etiquetas simuladas
            y = np.random.randint(0, 2, X.shape[0])
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar modelo
            model = MultinomialNB(alpha=params.get("alpha", 1.0))
            model.fit(X_train, y_train)
            
            # Evaluar modelo
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            result = {
                "model": model,
                "train_score": train_score,
                "test_score": test_score
            }
            
            logger.info(f"Naive Bayes entrenado - Train: {train_score:.3f}, Test: {test_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error entrenando Naive Bayes: {e}")
            raise
    
    async def _neural_network_trainer(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Entrena red neuronal"""
        try:
            from sklearn.neural_network import MLPClassifier
            from sklearn.model_selection import train_test_split
            
            if isinstance(data, dict) and "matrix" in data:
                X = data["matrix"]
            else:
                raise ValueError("Datos no compatibles para entrenamiento")
            
            # Generar etiquetas simuladas
            y = np.random.randint(0, 2, X.shape[0])
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar modelo
            model = MLPClassifier(
                hidden_layer_sizes=params.get("hidden_layer_sizes", (100,)),
                max_iter=params.get("max_iter", 1000),
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluar modelo
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            result = {
                "model": model,
                "train_score": train_score,
                "test_score": test_score
            }
            
            logger.info(f"Red neuronal entrenada - Train: {train_score:.3f}, Test: {test_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error entrenando red neuronal: {e}")
            raise
    
    # Componentes de Model Evaluation
    async def _cross_validator(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza validación cruzada"""
        try:
            from sklearn.model_selection import cross_val_score
            
            if isinstance(data, dict) and "model" in data:
                model = data["model"]
            else:
                raise ValueError("Modelo no encontrado en datos")
            
            # Obtener datos de entrenamiento (simulados)
            X = np.random.rand(100, 10)
            y = np.random.randint(0, 2, 100)
            
            cv_folds = params.get("cv_folds", 5)
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='f1_macro')
            
            result = {
                "cv_scores": scores.tolist(),
                "mean_score": scores.mean(),
                "std_score": scores.std(),
                "cv_folds": cv_folds
            }
            
            logger.info(f"Validación cruzada completada - Mean: {scores.mean():.3f} ± {scores.std():.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error en validación cruzada: {e}")
            raise
    
    async def _performance_evaluator(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa rendimiento del modelo"""
        try:
            from sklearn.metrics import classification_report, confusion_matrix
            
            if isinstance(data, dict) and "model" in data:
                model = data["model"]
            else:
                raise ValueError("Modelo no encontrado en datos")
            
            # Datos simulados para evaluación
            X_test = np.random.rand(50, 10)
            y_test = np.random.randint(0, 2, 50)
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            result = {
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "accuracy": report["accuracy"]
            }
            
            logger.info(f"Evaluación de rendimiento completada - Accuracy: {report['accuracy']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluando rendimiento: {e}")
            raise
    
    async def _confusion_matrix_generator(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Genera matriz de confusión"""
        try:
            from sklearn.metrics import confusion_matrix
            
            # Datos simulados
            y_true = np.random.randint(0, 2, 100)
            y_pred = np.random.randint(0, 2, 100)
            
            cm = confusion_matrix(y_true, y_pred)
            
            result = {
                "confusion_matrix": cm.tolist(),
                "true_negatives": int(cm[0, 0]),
                "false_positives": int(cm[0, 1]),
                "false_negatives": int(cm[1, 0]),
                "true_positives": int(cm[1, 1])
            }
            
            logger.info("Matriz de confusión generada")
            return result
            
        except Exception as e:
            logger.error(f"Error generando matriz de confusión: {e}")
            raise
    
    # Componentes de Prediction
    async def _predictor(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Realiza predicciones"""
        try:
            if isinstance(data, dict) and "model" in data:
                model = data["model"]
            else:
                raise ValueError("Modelo no encontrado en datos")
            
            # Datos de entrada para predicción
            X_new = np.random.rand(10, 10)
            predictions = model.predict(X_new)
            
            result = {
                "predictions": predictions.tolist(),
                "num_predictions": len(predictions)
            }
            
            logger.info(f"Predicciones realizadas: {len(predictions)}")
            return result
            
        except Exception as e:
            logger.error(f"Error realizando predicciones: {e}")
            raise
    
    async def _probability_predictor(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Realiza predicciones con probabilidades"""
        try:
            if isinstance(data, dict) and "model" in data:
                model = data["model"]
            else:
                raise ValueError("Modelo no encontrado en datos")
            
            # Datos de entrada para predicción
            X_new = np.random.rand(10, 10)
            
            # Verificar si el modelo soporta predict_proba
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_new)
                predictions = model.predict(X_new)
            else:
                predictions = model.predict(X_new)
                probabilities = np.ones((len(predictions), 2)) * 0.5
            
            result = {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist(),
                "num_predictions": len(predictions)
            }
            
            logger.info(f"Predicciones con probabilidades realizadas: {len(predictions)}")
            return result
            
        except Exception as e:
            logger.error(f"Error realizando predicciones con probabilidades: {e}")
            raise
    
    # Componentes de Postprocessing
    async def _result_formatter(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Any:
        """Formatea resultados"""
        try:
            output_format = params.get("output_format", "json")
            
            if output_format == "json":
                result = {
                    "formatted_data": data,
                    "format": "json",
                    "timestamp": datetime.now().isoformat()
                }
            elif output_format == "csv":
                if isinstance(data, dict):
                    result = {"formatted_data": str(data), "format": "csv"}
                else:
                    result = {"formatted_data": str(data), "format": "csv"}
            else:
                result = {"formatted_data": data, "format": output_format}
            
            logger.info(f"Resultados formateados en formato: {output_format}")
            return result
            
        except Exception as e:
            logger.error(f"Error formateando resultados: {e}")
            raise
    
    async def _confidence_calculator(self, data: Any, params: Dict[str, Any], step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula confianza de resultados"""
        try:
            threshold = params.get("threshold", 0.7)
            
            if isinstance(data, dict):
                if "probabilities" in data:
                    probabilities = np.array(data["probabilities"])
                    max_probs = np.max(probabilities, axis=1)
                    confident_predictions = np.sum(max_probs >= threshold)
                    
                    result = {
                        "total_predictions": len(max_probs),
                        "confident_predictions": int(confident_predictions),
                        "confidence_rate": float(confident_predictions / len(max_probs)),
                        "threshold": threshold,
                        "confidence_scores": max_probs.tolist()
                    }
                else:
                    result = {
                        "total_predictions": 1,
                        "confident_predictions": 1,
                        "confidence_rate": 1.0,
                        "threshold": threshold,
                        "confidence_scores": [1.0]
                    }
            else:
                result = {
                    "total_predictions": 1,
                    "confident_predictions": 1,
                    "confidence_rate": 1.0,
                    "threshold": threshold,
                    "confidence_scores": [1.0]
                }
            
            logger.info(f"Confianza calculada: {result['confidence_rate']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculando confianza: {e}")
            raise
    
    async def _cleanup_execution(self, execution_id: str):
        """Limpia ejecución después de un tiempo"""
        try:
            await asyncio.sleep(3600)  # Esperar 1 hora
            
            if execution_id in self.running_executions:
                del self.running_executions[execution_id]
            
            if execution_id in self.executions:
                del self.executions[execution_id]
            
            logger.info(f"Ejecución limpiada: {execution_id}")
            
        except Exception as e:
            logger.error(f"Error limpiando ejecución: {e}")
    
    async def get_pipeline_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de un pipeline"""
        try:
            if execution_id not in self.executions:
                return None
            
            execution = self.executions[execution_id]
            pipeline = self.pipelines[execution.pipeline_id]
            
            return {
                "execution_id": execution.id,
                "pipeline_id": execution.pipeline_id,
                "status": execution.status.value,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "steps": [
                    {
                        "id": step.id,
                        "name": step.name,
                        "stage": step.stage.value,
                        "status": step.status,
                        "execution_time": step.execution_time,
                        "error": step.error
                    }
                    for step in pipeline.steps
                ],
                "metrics": execution.metrics,
                "error": execution.error
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado de pipeline: {e}")
            return None
    
    async def get_pipeline_definitions(self) -> List[Dict[str, Any]]:
        """Obtiene definiciones de pipelines"""
        try:
            return [
                {
                    "id": pipeline.id,
                    "name": pipeline.name,
                    "description": pipeline.description,
                    "version": pipeline.version,
                    "step_count": len(pipeline.steps),
                    "created_at": pipeline.created_at.isoformat(),
                    "updated_at": pipeline.updated_at.isoformat()
                }
                for pipeline in self.pipelines.values()
            ]
        except Exception as e:
            logger.error(f"Error obteniendo definiciones de pipelines: {e}")
            return []
    
    async def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene historial de ejecuciones"""
        try:
            executions = list(self.executions.values())
            executions.sort(key=lambda x: x.started_at, reverse=True)
            
            return [
                {
                    "execution_id": execution.id,
                    "pipeline_id": execution.pipeline_id,
                    "status": execution.status.value,
                    "started_at": execution.started_at.isoformat(),
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "duration_seconds": (
                        (execution.completed_at - execution.started_at).total_seconds()
                        if execution.completed_at else None
                    )
                }
                for execution in executions[:limit]
            ]
        except Exception as e:
            logger.error(f"Error obteniendo historial de ejecuciones: {e}")
            return []



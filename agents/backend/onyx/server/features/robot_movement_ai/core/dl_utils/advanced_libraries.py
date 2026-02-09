"""
Advanced Deep Learning Libraries Integration
============================================

Utilidades para integrar y usar librerías avanzadas de deep learning,
transformers, diffusion models, y LLMs.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# PEFT (Parameter-Efficient Fine-Tuning)
# ============================================================================
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
        PeftType
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available. Install with: pip install peft")

# ============================================================================
# Sentence Transformers
# ============================================================================
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# ============================================================================
# FAISS (Similarity Search)
# ============================================================================
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

# ============================================================================
# Timm (PyTorch Image Models)
# ============================================================================
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    timm = None

# ============================================================================
# Einops (Tensor Operations)
# ============================================================================
try:
    from einops import rearrange, reduce, repeat
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False

# ============================================================================
# TorchMetrics
# ============================================================================
try:
    import torchmetrics
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    torchmetrics = None

# ============================================================================
# Datasets & Evaluate
# ============================================================================
try:
    from datasets import load_dataset, Dataset
    from evaluate import load as load_metric
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    load_dataset = None
    Dataset = None
    load_metric = None


class LoRAConfig:
    """
    Configuración para LoRA (Low-Rank Adaptation).
    
    Permite fine-tuning eficiente de modelos grandes.
    """
    
    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.1,
        bias: str = "none",
        task_type: str = "FEATURE_EXTRACTION"
    ):
        """
        Inicializar configuración LoRA.
        
        Args:
            r: Rango de la descomposición de bajo rango
            lora_alpha: Factor de escalado
            target_modules: Módulos objetivo (None = auto-detect)
            lora_dropout: Dropout para LoRA
            bias: Tipo de bias ("none", "all", "lora_only")
            task_type: Tipo de tarea
        """
        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.task_type = task_type
    
    def to_peft_config(self):
        """Convertir a configuración PEFT."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required for LoRA")
        
        task_type_map = {
            "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
            "SEQ_CLS": TaskType.SEQ_CLS,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS
        }
        
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=task_type_map.get(self.task_type, TaskType.FEATURE_EXTRACTION)
        )


def apply_lora_to_model(
    model,
    config: LoRAConfig,
    model_name: Optional[str] = None
):
    """
    Aplicar LoRA a un modelo.
    
    Args:
        model: Modelo PyTorch o Transformers
        config: Configuración LoRA
        model_name: Nombre del modelo (para auto-detectar módulos)
        
    Returns:
        Modelo con LoRA aplicado
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT is required. Install with: pip install peft")
    
    peft_config = config.to_peft_config()
    
    # Auto-detectar módulos objetivo si no se especifican
    if peft_config.target_modules is None and model_name:
        if "bert" in model_name.lower():
            peft_config.target_modules = ["query", "value"]
        elif "gpt" in model_name.lower() or "llama" in model_name.lower():
            peft_config.target_modules = ["q_proj", "v_proj"]
        elif "t5" in model_name.lower():
            peft_config.target_modules = ["q", "v"]
    
    peft_model = get_peft_model(model, peft_config)
    
    logger.info(f"Applied LoRA to model. Trainable parameters: {peft_model.get_nb_trainable_parameters()}")
    
    return peft_model


class EmbeddingManager:
    """
    Gestor de embeddings usando Sentence Transformers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializar gestor de embeddings.
        
        Args:
            model_name: Nombre del modelo de Sentence Transformers
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("Sentence Transformers required. Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Codificar textos a embeddings.
        
        Args:
            texts: Texto o lista de textos
            batch_size: Tamaño del batch
            show_progress: Mostrar progreso
            
        Returns:
            Array de embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray]
    ) -> float:
        """
        Calcular similitud entre dos textos o embeddings.
        
        Args:
            text1: Primer texto o embedding
            text2: Segundo texto o embedding
            
        Returns:
            Similitud coseno (0-1)
        """
        if isinstance(text1, str):
            emb1 = self.encode(text1)
        else:
            emb1 = text1
        
        if isinstance(text2, str):
            emb2 = self.encode(text2)
        else:
            emb2 = text2
        
        # Similitud coseno
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(similarity)


class FAISSSearch:
    """
    Búsqueda de similitud usando FAISS.
    """
    
    def __init__(self, dimension: int, use_gpu: bool = False):
        """
        Inicializar índice FAISS.
        
        Args:
            dimension: Dimensión de los embeddings
            use_gpu: Usar GPU si está disponible
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS required. Install with: pip install faiss-cpu or faiss-gpu")
        
        self.dimension = dimension
        self.use_gpu = use_gpu
        
        # Crear índice L2 (Euclidean distance)
        self.index = faiss.IndexFlatL2(dimension)
        
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            logger.info("Using GPU for FAISS")
        else:
            logger.info("Using CPU for FAISS")
    
    def add(self, embeddings: np.ndarray):
        """
        Agregar embeddings al índice.
        
        Args:
            embeddings: Array de embeddings [n, dimension]
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}")
        
        # Normalizar para búsqueda coseno
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> tuple:
        """
        Buscar k vecinos más cercanos.
        
        Args:
            query: Query embedding [dimension] o [n, dimension]
            k: Número de vecinos
        
        Returns:
            (distancias, índices)
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Normalizar
        faiss.normalize_L2(query)
        
        distances, indices = self.index.search(query.astype('float32'), k)
        
        return distances, indices
    
    def reset(self):
        """Resetear índice."""
        self.index.reset()


class TimmModelLoader:
    """
    Cargador de modelos usando timm (PyTorch Image Models).
    """
    
    @staticmethod
    def list_models(pattern: Optional[str] = None) -> List[str]:
        """
        Listar modelos disponibles.
        
        Args:
            pattern: Patrón de búsqueda (opcional)
            
        Returns:
            Lista de nombres de modelos
        """
        if not TIMM_AVAILABLE:
            raise ImportError("timm required. Install with: pip install timm")
        
        if pattern:
            return timm.list_models(pattern)
        return timm.list_models()
    
    @staticmethod
    def create_model(
        model_name: str,
        pretrained: bool = True,
        num_classes: int = 1000,
        **kwargs
    ):
        """
        Crear modelo de timm.
        
        Args:
            model_name: Nombre del modelo
            pretrained: Usar pesos pre-entrenados
            num_classes: Número de clases
            **kwargs: Argumentos adicionales
            
        Returns:
            Modelo PyTorch
        """
        if not TIMM_AVAILABLE:
            raise ImportError("timm required. Install with: pip install timm")
        
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs
        )
        
        logger.info(f"Created timm model: {model_name}")
        
        return model


class MetricsManager:
    """
    Gestor de métricas usando torchmetrics.
    """
    
    def __init__(self):
        """Inicializar gestor de métricas."""
        if not TORCHMETRICS_AVAILABLE:
            raise ImportError("torchmetrics required. Install with: pip install torchmetrics")
        
        self.metrics: Dict[str, Any] = {}
    
    def add_metric(self, name: str, metric):
        """
        Agregar métrica.
        
        Args:
            name: Nombre de la métrica
            metric: Instancia de métrica torchmetrics
        """
        self.metrics[name] = metric
    
    def update(self, name: str, preds, target):
        """
        Actualizar métrica.
        
        Args:
            name: Nombre de la métrica
            preds: Predicciones
            target: Objetivos
        """
        if name not in self.metrics:
            raise ValueError(f"Metric {name} not found")
        
        self.metrics[name].update(preds, target)
    
    def compute(self, name: Optional[str] = None) -> Dict[str, float]:
        """
        Calcular métrica(s).
        
        Args:
            name: Nombre de métrica (None = todas)
            
        Returns:
            Diccionario de métricas
        """
        if name:
            if name not in self.metrics:
                raise ValueError(f"Metric {name} not found")
            return {name: self.metrics[name].compute().item()}
        
        return {
            name: metric.compute().item()
            for name, metric in self.metrics.items()
        }
    
    def reset(self, name: Optional[str] = None):
        """
        Resetear métrica(s).
        
        Args:
            name: Nombre de métrica (None = todas)
        """
        if name:
            if name not in self.metrics:
                raise ValueError(f"Metric {name} not found")
            self.metrics[name].reset()
        else:
            for metric in self.metrics.values():
                metric.reset()


def get_available_libraries() -> Dict[str, bool]:
    """
    Obtener estado de librerías disponibles.
    
    Returns:
        Diccionario con estado de cada librería
    """
    return {
        "peft": PEFT_AVAILABLE,
        "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
        "faiss": FAISS_AVAILABLE,
        "timm": TIMM_AVAILABLE,
        "einops": EINOPS_AVAILABLE,
        "torchmetrics": TORCHMETRICS_AVAILABLE,
        "datasets": DATASETS_AVAILABLE
    }


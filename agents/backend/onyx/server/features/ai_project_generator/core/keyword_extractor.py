"""
Keyword Extractor - Extractor de palabras clave
================================================

Módulo especializado para extraer keywords y características del proyecto
basándose en la descripción del usuario.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

from typing import Dict, Any, List, Set

_DL_PATTERNS: Set[str] = {
    "deep learning", "pytorch", "torch", "neural network", "neural",
    "cnn", "rnn", "lstm", "gru", "gan", "vae"
}

_TRANSFORMER_PATTERNS: Set[str] = {
    "transformer", "bert", "gpt", "attention", "self-attention",
    "encoder", "decoder", "transformer model"
}

_DIFFUSION_PATTERNS: Set[str] = {
    "diffusion", "stable diffusion", "ddpm", "ddim",
    "latent diffusion", "image generation", "text-to-image"
}

_LLM_PATTERNS: Set[str] = {
    "llm", "large language model", "language model", "llama",
    "mistral", "falcon", "chatgpt", "gpt", "claude", "gemini"
}

_GRADIO_PATTERNS: Set[str] = {
    "gradio", "demo", "interactive", "interactivo", "ui", "interface", "interfaz"
}

_AI_PATTERNS: Dict[str, Set[str]] = {
    "chat": {"chat", "conversación", "conversational", "assistant", "asistente", "bot", "chatbot"},
    "vision": {"image", "imagen", "vision", "computer vision", "cv", "detection", "detección", "recognition", "reconocimiento", "ocr"},
    "audio": {"audio", "música", "music", "sound", "voice", "voz", "speech", "habla", "transcription", "transcripción"},
    "nlp": {"text", "texto", "nlp", "language", "idioma", "translation", "traducción", "sentiment", "sentimiento", "analysis", "análisis"},
    "video": {"video", "vídeo", "streaming", "live"},
    "recommendation": {"recommendation", "recomendación", "recommend", "suggest", "sugerencia"},
    "analytics": {"analysis", "análisis", "analytics", "data", "datos", "report", "reporte"},
    "generation": {"generate", "generar", "generation", "generación", "create", "crear", "content", "contenido"},
    "classification": {"classify", "clasificar", "classification", "clasificación", "categorize", "categorizar"},
    "summarization": {"summarize", "resumir", "summary", "resumen"},
    "qa": {"question", "pregunta", "answer", "respuesta", "qa", "q&a"},
}

_COMPLEXITY_INDICATORS: Dict[str, Set[str]] = {
    "simple": {"simple", "básico", "basic", "easy", "fácil"},
    "medium": {"medium", "medio", "standard", "estándar"},
    "complex": {"complex", "complejo", "advanced", "avanzado", "enterprise", "empresarial"},
}

_FEATURE_PATTERNS: Dict[str, Set[str]] = {
    "requires_auth": {"auth", "autenticación", "login", "user", "usuario", "sign in", "sign up"},
    "requires_database": {"database", "base de datos", "db", "storage", "almacenamiento", "postgres", "mysql", "mongodb"},
    "requires_streaming": {"stream", "streaming", "tiempo real", "real-time", "real time", "live"},
    "requires_websocket": {"websocket", "websockets", "socket", "ws", "real-time", "tiempo real"},
    "requires_file_upload": {"upload", "subir", "file", "archivo", "image", "imagen", "document", "documento"},
    "requires_cache": {"cache", "caching", "redis", "memcached"},
    "requires_queue": {"queue", "cola", "task", "tarea", "background", "fondo", "async", "asíncrono"},
}

_FEATURE_DETECTORS: Dict[str, Set[str]] = {
    "dashboard": {"dashboard", "panel"},
    "rest_api": {"api", "rest"},
    "graphql": {"graphql"},
    "admin_panel": {"admin"},
    "monitoring": {"monitoring", "monitoreo"},
    "logging": {"logging", "logs"},
    "testing": {"testing", "tests"},
    "docker": {"docker", "container"},
}

_PROVIDER_PATTERNS: Dict[str, Set[str]] = {
    "openai": {"openai", "gpt", "chatgpt"},
    "anthropic": {"anthropic", "claude"},
    "google": {"google", "gemini", "bard"},
    "huggingface": {"huggingface", "hugging face", "transformers"},
    "local": {"local", "modelo local", "llama", "mistral"},
}


def _get_default_keywords() -> Dict[str, Any]:
    """
    Retorna keywords por defecto (función pura).
    
    Returns:
        Diccionario con keywords por defecto
    """
    return {
        "ai_type": "general",
        "features": [],
        "requires_auth": False,
        "requires_database": False,
        "requires_api": True,
        "requires_ml": False,
        "requires_streaming": False,
        "requires_websocket": False,
        "requires_file_upload": False,
        "requires_cache": False,
        "requires_queue": False,
        "model_providers": [],
        "complexity": "medium",
        "is_deep_learning": False,
        "is_transformer": False,
        "is_diffusion": False,
        "is_llm": False,
        "requires_pytorch": False,
        "requires_gradio": False,
        "requires_training": False,
        "requires_fine_tuning": False,
        "model_architecture": None,
    }


def _contains_any_pattern(text: str, patterns: Set[str]) -> bool:
    """
    Verifica si el texto contiene algún patrón (función pura).
    
    Args:
        text: Texto a verificar
        patterns: Set de patrones
        
    Returns:
        True si contiene algún patrón, False en caso contrario
    """
    return any(pattern in text for pattern in patterns)


def _detect_deep_learning(keywords: Dict[str, Any], description_lower: str) -> None:
    """
    Detecta características de Deep Learning.
    
    Args:
        keywords: Diccionario de keywords a actualizar
        description_lower: Descripción en minúsculas
    """
    if _contains_any_pattern(description_lower, _DL_PATTERNS):
        keywords["is_deep_learning"] = True
        keywords["requires_pytorch"] = True
        keywords["requires_ml"] = True
        keywords["requires_training"] = True
    
    if _contains_any_pattern(description_lower, _TRANSFORMER_PATTERNS):
        keywords["is_transformer"] = True
        keywords["is_deep_learning"] = True
        keywords["requires_pytorch"] = True
        keywords["requires_ml"] = True
        keywords["model_architecture"] = "transformer"
    
    if _contains_any_pattern(description_lower, _DIFFUSION_PATTERNS):
        keywords["is_diffusion"] = True
        keywords["is_deep_learning"] = True
        keywords["requires_pytorch"] = True
        keywords["requires_ml"] = True
        keywords["model_architecture"] = "diffusion"
        keywords["requires_file_upload"] = True
    
    if _contains_any_pattern(description_lower, _LLM_PATTERNS):
        keywords["is_llm"] = True
        keywords["is_transformer"] = True
        keywords["is_deep_learning"] = True
        keywords["requires_pytorch"] = True
        keywords["requires_ml"] = True
        keywords["model_architecture"] = "llm"
        keywords["requires_fine_tuning"] = True
    
    if _contains_any_pattern(description_lower, _GRADIO_PATTERNS):
        keywords["requires_gradio"] = True


def _detect_ai_type(keywords: Dict[str, Any], description_lower: str) -> None:
    """
    Detecta el tipo de IA.
    
    Args:
        keywords: Diccionario de keywords a actualizar
        description_lower: Descripción en minúsculas
    """
    for ai_type, patterns in _AI_PATTERNS.items():
        if _contains_any_pattern(description_lower, patterns):
            keywords["ai_type"] = ai_type
            if ai_type in {"chat", "video"}:
                keywords["requires_websocket"] = True
                keywords["requires_streaming"] = True
            if ai_type in {"vision", "audio", "nlp", "video", "generation", "classification"}:
                keywords["requires_ml"] = True
                if not keywords["is_deep_learning"] and ai_type in {"vision", "audio", "nlp", "generation"}:
                    keywords["requires_ml"] = True
            break


def _detect_features(keywords: Dict[str, Any], description_lower: str) -> None:
    """
    Detecta características avanzadas.
    
    Args:
        keywords: Diccionario de keywords a actualizar
        description_lower: Descripción en minúsculas
    """
    for feature, patterns in _FEATURE_PATTERNS.items():
        if _contains_any_pattern(description_lower, patterns):
            keywords[feature] = True
    
    features = []
    for feature_name, patterns in _FEATURE_DETECTORS.items():
        if _contains_any_pattern(description_lower, patterns):
            features.append(feature_name)
    
    keywords["features"] = features


def _detect_providers(keywords: Dict[str, Any], description_lower: str) -> None:
    """
    Detecta proveedores de modelos.
    
    Args:
        keywords: Diccionario de keywords a actualizar
        description_lower: Descripción en minúsculas
    """
    for provider, patterns in _PROVIDER_PATTERNS.items():
        if _contains_any_pattern(description_lower, patterns):
            keywords["model_providers"].append(provider)


def _detect_complexity(keywords: Dict[str, Any], description_lower: str) -> None:
    """
    Detecta la complejidad del proyecto.
    
    Args:
        keywords: Diccionario de keywords a actualizar
        description_lower: Descripción en minúsculas
    """
    for complexity, indicators in _COMPLEXITY_INDICATORS.items():
        if _contains_any_pattern(description_lower, indicators):
            keywords["complexity"] = complexity
            break


class KeywordExtractor:
    """Extractor de keywords de proyectos de IA."""
    
    def extract(self, description: str) -> Dict[str, Any]:
        """
        Extrae keywords y características del proyecto de la descripción.
        
        Args:
            description: Descripción del proyecto
            
        Returns:
            Diccionario con keywords extraídas
        """
        if not description:
            return _get_default_keywords()
        
        description_lower = description.lower()
        keywords = _get_default_keywords()
        
        _detect_deep_learning(keywords, description_lower)
        _detect_ai_type(keywords, description_lower)
        _detect_features(keywords, description_lower)
        _detect_providers(keywords, description_lower)
        _detect_complexity(keywords, description_lower)
        
        return keywords

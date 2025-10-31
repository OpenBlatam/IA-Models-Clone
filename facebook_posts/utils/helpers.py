from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import re
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from urllib.parse import urlparse
    from collections import Counter
from typing import Any, List, Dict, Optional
import asyncio
"""
🎯 Facebook Posts - Utilities & Helpers
=======================================

Funciones de utilidad y helpers para el sistema de Facebook posts.
"""



# ===== TEXT PROCESSING UTILITIES =====

def clean_text(text: str) -> str:
    """Limpiar y normalizar texto."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove invalid characters for Facebook
    text = re.sub(r'[^\w\s\-.,!?@#$%&*()+=:;"\'\n\r\u00A0-\uFFFF]', '', text)
    
    return text


def extract_hashtags(text: str) -> List[str]:
    """Extraer hashtags del texto."""
    hashtag_pattern = r'#(\w+)'
    hashtags = re.findall(hashtag_pattern, text)
    return [tag.lower() for tag in hashtags]


def extract_mentions(text: str) -> List[str]:
    """Extraer menciones del texto."""
    mention_pattern = r'@(\w+)'
    mentions = re.findall(mention_pattern, text)
    return mentions


def extract_urls(text: str) -> List[str]:
    """Extraer URLs del texto."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return urls


def count_emojis(text: str) -> int:
    """Contar emojis en el texto."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # misc symbols
        "\U000024C2-\U0001F251"  # enclosed characters
        "]+",
        flags=re.UNICODE
    )
    return len(emoji_pattern.findall(text))


def calculate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """Calcular tiempo de lectura en segundos."""
    word_count = len(text.split())
    reading_time_minutes = word_count / words_per_minute
    return max(int(reading_time_minutes * 60), 10)  # Mínimo 10 segundos


def get_text_complexity_score(text: str) -> float:
    """Calcular score de complejidad del texto (0-1)."""
    if not text:
        return 0.0
    
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    # Calculate average word length
    avg_word_length = sum(len(word) for word in words) / total_words
    
    # Calculate sentence count
    sentence_count = len(re.findall(r'[.!?]+', text))
    avg_sentence_length = total_words / max(sentence_count, 1)
    
    # Complexity factors
    word_complexity = min(avg_word_length / 10, 1.0)  # Normalize to 0-1
    sentence_complexity = min(avg_sentence_length / 20, 1.0)  # Normalize to 0-1
    
    return (word_complexity + sentence_complexity) / 2


# ===== CONTENT VALIDATION =====

def validate_facebook_content(text: str) -> Tuple[bool, List[str]]:
    """Validar contenido para Facebook."""
    errors = []
    
    # Length validation
    if len(text) > 2000:
        errors.append("Content exceeds Facebook's 2000 character limit")
    
    if len(text.strip()) < 10:
        errors.append("Content too short for meaningful engagement")
    
    # Content quality checks
    if not text.strip():
        errors.append("Content cannot be empty")
    
    # Check for excessive capitalization
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    if caps_ratio > 0.5:
        errors.append("Excessive use of capital letters")
    
    # Check for spam indicators
    repeated_chars = re.findall(r'(.)\1{4,}', text)
    if repeated_chars:
        errors.append("Excessive repeated characters detected")
    
    return len(errors) == 0, errors


def validate_hashtags(hashtags: List[str]) -> Tuple[bool, List[str]]:
    """Validar hashtags."""
    errors = []
    
    if len(hashtags) > 30:
        errors.append("Too many hashtags (max 30)")
    
    for hashtag in hashtags:
        if not hashtag.replace('_', '').replace('-', '').isalnum():
            errors.append(f"Invalid hashtag format: #{hashtag}")
        
        if len(hashtag) > 100:
            errors.append(f"Hashtag too long: #{hashtag}")
    
    return len(errors) == 0, errors


def validate_url(url: str) -> bool:
    """Validar formato de URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


# ===== ENGAGEMENT PREDICTION HELPERS =====

def calculate_engagement_factors(text: str, hashtags: List[str]) -> Dict[str, float]:
    """Calcular factores que influyen en el engagement."""
    factors = {
        'has_question': 0.0,
        'has_call_to_action': 0.0,
        'has_emojis': 0.0,
        'optimal_length': 0.0,
        'hashtag_count': 0.0,
        'readability': 0.0
    }
    
    # Question factor
    if '?' in text:
        factors['has_question'] = 1.0
    
    # Call to action factor
    cta_patterns = [
        r'share', r'comment', r'like', r'follow', r'click', r'visit',
        r'tell us', r'what do you think', r'let us know'
    ]
    for pattern in cta_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            factors['has_call_to_action'] = 1.0
            break
    
    # Emoji factor
    if count_emojis(text) > 0:
        factors['has_emojis'] = 1.0
    
    # Optimal length factor (80-150 words)
    word_count = len(text.split())
    if 80 <= word_count <= 150:
        factors['optimal_length'] = 1.0
    elif 50 <= word_count < 80 or 150 < word_count <= 200:
        factors['optimal_length'] = 0.7
    else:
        factors['optimal_length'] = 0.3
    
    # Hashtag count factor
    hashtag_count = len(hashtags)
    if 3 <= hashtag_count <= 7:
        factors['hashtag_count'] = 1.0
    elif 1 <= hashtag_count < 3 or 7 < hashtag_count <= 10:
        factors['hashtag_count'] = 0.7
    else:
        factors['hashtag_count'] = 0.3
    
    # Readability factor
    complexity = get_text_complexity_score(text)
    factors['readability'] = 1.0 - complexity  # Lower complexity = higher readability
    
    return factors


def predict_engagement_score(text: str, hashtags: List[str]) -> float:
    """Predecir score de engagement (0-1)."""
    factors = calculate_engagement_factors(text, hashtags)
    
    # Weighted average
    weights = {
        'has_question': 0.20,
        'has_call_to_action': 0.15,
        'has_emojis': 0.10,
        'optimal_length': 0.25,
        'hashtag_count': 0.15,
        'readability': 0.15
    }
    
    score = sum(factors[factor] * weight for factor, weight in weights.items())
    return min(max(score, 0.0), 1.0)


# ===== TIMING OPTIMIZATION =====

def get_optimal_posting_times(audience: str, timezone: str = "UTC") -> List[datetime]:
    """Obtener horarios óptimos de publicación por audiencia."""
    base_times = {
        'general': [9, 13, 17, 20],  # 9am, 1pm, 5pm, 8pm
        'professionals': [8, 12, 17],  # 8am, 12pm, 5pm
        'young_adults': [11, 15, 19, 22],  # 11am, 3pm, 7pm, 10pm
        'parents': [10, 14, 20],  # 10am, 2pm, 8pm
        'entrepreneurs': [7, 12, 18],  # 7am, 12pm, 6pm
        'students': [12, 16, 21],  # 12pm, 4pm, 9pm
        'seniors': [10, 14, 19]  # 10am, 2pm, 7pm
    }
    
    hours = base_times.get(audience.lower(), base_times['general'])
    today = datetime.now()
    
    optimal_times = []
    for hour in hours:
        optimal_time = today.replace(hour=hour, minute=0, second=0, microsecond=0)
        if optimal_time <= today:
            optimal_time += timedelta(days=1)
        optimal_times.append(optimal_time)
    
    return optimal_times


def get_best_posting_day(audience: str) -> str:
    """Obtener mejor día de la semana para publicar."""
    best_days = {
        'general': 'Tuesday',
        'professionals': 'Wednesday',
        'young_adults': 'Friday',
        'parents': 'Sunday',
        'entrepreneurs': 'Tuesday',
        'students': 'Saturday',
        'seniors': 'Thursday'
    }
    
    return best_days.get(audience.lower(), 'Tuesday')


# ===== HASHING & CACHING UTILITIES =====

def generate_content_hash(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Generar hash único para contenido."""
    content_data = text
    if metadata:
        content_data += str(sorted(metadata.items()))
    
    return hashlib.sha256(content_data.encode()).hexdigest()[:16]


def generate_cache_key(prefix: str, *args: str) -> str:
    """Generar clave de cache."""
    key_parts = [prefix] + list(args)
    return ':'.join(key_parts)


def generate_unique_id() -> str:
    """Generar ID único."""
    return str(uuid.uuid4())


# ===== PERFORMANCE MONITORING =====

class PerformanceTimer:
    """Timer para medir performance."""
    
    def __init__(self, operation_name: str):
        
    """__init__ function."""
self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self) -> Any:
        self.start_time = datetime.now()
        self.logger.debug(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds() * 1000
        
        if exc_type:
            self.logger.error(f"Operation failed: {self.operation_name} ({duration:.2f}ms)")
        else:
            self.logger.debug(f"Operation completed: {self.operation_name} ({duration:.2f}ms)")
    
    def get_duration_ms(self) -> float:
        """Obtener duración en milisegundos."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0


# ===== CONTENT OPTIMIZATION =====

def optimize_text_for_engagement(text: str) -> str:
    """Optimizar texto para mayor engagement."""
    optimized = text
    
    # Add emojis if none present
    if count_emojis(optimized) == 0:
        optimized = "✨ " + optimized
    
    # Add question if none present
    if "?" not in optimized:
        optimized += " What do you think? 💭"
    
    # Ensure proper spacing
    optimized = re.sub(r'\s+', ' ', optimized.strip())
    
    return optimized


def suggest_hashtags(text: str, max_suggestions: int = 5) -> List[str]:
    """Sugerir hashtags basados en el contenido."""
    # Extract keywords
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter common words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    
    keywords = [word for word in words if len(word) > 3 and word not in stop_words]
    
    # Get top keywords by frequency
    word_freq = Counter(keywords)
    top_keywords = [word for word, _ in word_freq.most_common(max_suggestions)]
    
    # Add some general engagement hashtags
    general_hashtags = ['trending', 'viral', 'success', 'growth', 'tips']
    
    suggestions = top_keywords + general_hashtags
    return list(set(suggestions))[:max_suggestions]


# ===== ERROR HANDLING =====

def safe_execute(func, *args, default=None, **kwargs) -> Any:
    """Ejecutar función de manera segura con valor por defecto."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.warning(f"Safe execution failed for {func.__name__}: {e}")
        return default


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """Validar campos requeridos."""
    missing_fields = []
    for field in required_fields:
        if field not in data or not data[field]:
            missing_fields.append(field)
    return missing_fields


# ===== DATA TRANSFORMATION =====

def sanitize_for_json(data: Any) -> Any:
    """Sanitizar datos para serialización JSON."""
    if isinstance(data, datetime):
        return data.isoformat()
    elif isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif hasattr(data, '__dict__'):
        return sanitize_for_json(data.__dict__)
    else:
        return data


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Aplanar diccionario anidado."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items) 
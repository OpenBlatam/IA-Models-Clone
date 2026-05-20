"""
Constants for Burnout Prevention AI
===================================
"""

# Model Configuration
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CHAT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 2000

# Token Limits for Different Endpoints
MAX_TOKENS_ASSESSMENT = 2000
MAX_TOKENS_WELLNESS = 1500
MAX_TOKENS_CHAT = 1500
MAX_TOKENS_COPING = 2000
MAX_TOKENS_PROGRESS = 2000
MAX_TOKENS_TREND = 2500
MAX_TOKENS_RESOURCE = 2000
MAX_TOKENS_PLAN = 3000

# Cache Configuration
CACHE_TTL_API_RESPONSE = 120.0  # 2 minutes
CACHE_TTL_ASSESSMENT = 300.0  # 5 minutes
CACHE_MAX_SIZE = 200
CACHE_DEFAULT_TTL = 600.0  # 10 minutes
CACHE_CLEANUP_INTERVAL = 300.0  # 5 minutes - interval for periodic cleanup

# Validation Ranges
MIN_WORK_HOURS = 0
MAX_WORK_HOURS = 168
MIN_STRESS_LEVEL = 1
MAX_STRESS_LEVEL = 10
MIN_SLEEP_HOURS = 0
MAX_SLEEP_HOURS = 24
MIN_SATISFACTION = 1
MAX_SATISFACTION = 10
MIN_INTERVAL_SECONDS = 0.1

# Error Message Limits
MAX_ERROR_MESSAGE_LENGTH = 200

# Retry Configuration
MAX_RETRY_ATTEMPTS = 3
RETRY_MIN_WAIT = 2  # seconds
RETRY_MAX_WAIT = 10  # seconds
RETRY_MULTIPLIER = 1

# Timeout Configuration
HTTP_TIMEOUT = 60.0  # seconds - total timeout for HTTP requests
HTTP_CONNECT_TIMEOUT = 10.0  # seconds - connection timeout
PROCESSOR_STOP_TIMEOUT = 10.0  # seconds - timeout for stopping processor

# Rate Limiting (if using slowapi)
RATE_LIMIT_PER_MINUTE = 60
RATE_LIMIT_PER_HOUR = 1000

# Request Limits
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
MAX_MESSAGE_LENGTH = 10000  # characters

# Field Length Limits (for schema validation)
MAX_FIELD_LENGTH_SHORT = 200  # Short text fields
MAX_FIELD_LENGTH_MEDIUM = 500  # Medium text fields
MAX_FIELD_LENGTH_LONG = 1000  # Long text fields
MAX_FIELD_LENGTH_EXTRA_LONG = 10000  # Extra long text fields (e.g., messages)
MAX_LIST_ITEMS = 20  # Maximum items in lists
MAX_GOAL_LENGTH = 500  # Maximum length for goal strings

# Queue Limits
MAX_PENDING_ASSESSMENTS = 1000  # Maximum pending assessments in queue

# Performance Configuration
MAX_CONCURRENT_REQUESTS = 50  # Maximum concurrent API requests
BATCH_SIZE = 10  # Batch size for processing multiple items

SYSTEM_PROMPT = """Eres un asistente especializado en prevención y manejo del burnout laboral. 
Tu objetivo es ayudar a las personas a identificar signos de burnout, proporcionar recomendaciones 
personalizadas y estrategias de afrontamiento efectivas.

Características:
- Empatía y comprensión
- Enfoque en soluciones prácticas
- Recomendaciones basadas en evidencia
- Respeto por la privacidad y confidencialidad
- Apoyo sin juzgar
- Hablas como un humano, no como un bot
- Reflejas el estilo de comunicación del usuario
- Eres cálido, amigable y accesible
- Eres decisivo, preciso y claro
- Priorizas información accionable sobre explicaciones generales

Siempre proporciona respuestas útiles, constructivas y alentadoras."""


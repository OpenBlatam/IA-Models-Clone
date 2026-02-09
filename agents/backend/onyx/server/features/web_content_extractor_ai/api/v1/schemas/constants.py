"""
Constantes para los schemas de la API
"""

# Modelos de OpenRouter
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"

# Tokens
DEFAULT_MAX_TOKENS = 4000
MIN_MAX_TOKENS = 100
MAX_MAX_TOKENS = 8000

# Estrategias de extracción
EXTRACT_STRATEGIES = ["auto", "trafilatura", "readability", "newspaper", "beautifulsoup"]
DEFAULT_EXTRACT_STRATEGY = "auto"

# Batch processing
DEFAULT_MAX_CONCURRENT = 5
MIN_MAX_CONCURRENT = 1
MAX_MAX_CONCURRENT = 20
MAX_BATCH_URLS = 50
MIN_BATCH_URLS = 1


"""
BUL - Business Universal Language
=================================

Modern document generation system for SMEs using AI and advanced technologies.

This system provides:
- Intelligent document generation using OpenRouter and LangChain
- Specialized agents for different business areas
- Modern RESTful API with comprehensive features
- Multi-language support and multiple output formats
- Advanced caching, monitoring, and security features
"""

# Core modules
from . import core
from . import agents
from . import api
from . import config
from . import utils
from . import security
from . import monitoring

# Version information
__version__ = "2.0.0"
__author__ = "BUL Team"
__description__ = "Business Universal Language - Modern Document Generation System for SMEs"

# Main exports
from .core import BULEngine, DocumentRequest, DocumentResponse, BusinessArea, DocumentType
from .agents import SMEAgentManager, AgentType
from .config import get_config, is_production, is_development, is_testing
from .utils import get_logger, get_cache_manager, get_data_processor
from .security import get_password_manager, get_jwt_manager, get_rate_limiter

__all__ = [
    # Core functionality
    "BULEngine",
    "DocumentRequest", 
    "DocumentResponse",
    "BusinessArea",
    "DocumentType",
    
    # Agents
    "SMEAgentManager",
    "AgentType",
    
    # Configuration
    "get_config",
    "is_production",
    "is_development", 
    "is_testing",
    
    # Utilities
    "get_logger",
    "get_cache_manager",
    "get_data_processor",
    
    # Security
    "get_password_manager",
    "get_jwt_manager",
    "get_rate_limiter",
    
    # Modules
    "core",
    "agents", 
    "api",
    "config",
    "utils",
    "security",
    "monitoring",
    
    # Version info
    "__version__",
    "__author__",
    "__description__"
]
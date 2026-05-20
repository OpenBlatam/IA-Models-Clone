"""
Chat Configuration
===================

Configuración del sistema de chat continuo.
"""

import os
import logging
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ChatConfig:
    """Configuración del sistema de chat."""
    
    # API Settings
    api_host: str = field(default_factory=lambda: os.getenv("CHAT_API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("CHAT_API_PORT", "8006")))
    
    # LLM Settings
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4"))
    llm_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    llm_base_url: Optional[str] = field(default_factory=lambda: os.getenv("LLM_BASE_URL"))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7")))
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2000")))
    
    # Chat Behavior
    auto_continue: bool = field(default_factory=lambda: os.getenv("AUTO_CONTINUE", "true").lower() == "true")
    response_interval: float = field(default_factory=lambda: float(os.getenv("RESPONSE_INTERVAL", "2.0")))
    max_consecutive_responses: int = field(default_factory=lambda: int(os.getenv("MAX_CONSECUTIVE_RESPONSES", "100")))
    max_messages_per_session: int = field(default_factory=lambda: int(os.getenv("MAX_MESSAGES_PER_SESSION", "1000")))
    
    # CORS
    cors_origins: List[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # Advanced Settings
    enable_streaming: bool = field(default_factory=lambda: os.getenv("ENABLE_STREAMING", "true").lower() == "true")
    enable_metrics: bool = field(default_factory=lambda: os.getenv("ENABLE_METRICS", "true").lower() == "true")
    
    # Storage Settings
    storage_type: str = field(default_factory=lambda: os.getenv("STORAGE_TYPE", "json"))  # json, redis
    storage_path: str = field(default_factory=lambda: os.getenv("STORAGE_PATH", "sessions"))
    redis_url: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_URL"))
    session_ttl: int = field(default_factory=lambda: int(os.getenv("SESSION_TTL", "86400")))  # 24 horas
    
    # Auto-save Settings
    auto_save: bool = field(default_factory=lambda: os.getenv("AUTO_SAVE", "true").lower() == "true")
    save_interval: float = field(default_factory=lambda: float(os.getenv("SAVE_INTERVAL", "30.0")))
    
    # Rate Limiting
    rate_limit_max_requests: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "60")))
    rate_limit_window: float = field(default_factory=lambda: float(os.getenv("RATE_LIMIT_WINDOW", "60.0")))
    rate_limit_max_concurrent: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_MAX_CONCURRENT", "10")))
    
    # Cache Settings
    enable_cache: bool = field(default_factory=lambda: os.getenv("ENABLE_CACHE", "true").lower() == "true")
    cache_size: int = field(default_factory=lambda: int(os.getenv("CACHE_SIZE", "1000")))
    cache_ttl: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "3600")))  # 1 hora
    
    # Plugins
    enable_plugins: bool = field(default_factory=lambda: os.getenv("ENABLE_PLUGINS", "true").lower() == "true")
    
    # Backups
    enable_backups: bool = field(default_factory=lambda: os.getenv("ENABLE_BACKUPS", "true").lower() == "true")
    backup_interval_hours: int = field(default_factory=lambda: int(os.getenv("BACKUP_INTERVAL_HOURS", "24")))
    backup_directory: str = field(default_factory=lambda: os.getenv("BACKUP_DIRECTORY", "backups"))
    
    def get_llm_provider(self) -> Optional[Callable]:
        """
        Obtener función proveedora de LLM según la configuración.
        
        Returns:
            Función async que genera respuestas
        """
        if self.llm_provider == "openai":
            return self._openai_provider
        elif self.llm_provider == "anthropic":
            return self._anthropic_provider
        elif self.llm_provider == "mock":
            return self._mock_provider
        else:
            logger.warning(f"Unknown LLM provider: {self.llm_provider}, using mock")
            return self._mock_provider
    
    async def _openai_provider(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Proveedor OpenAI."""
        try:
            import openai
            
            client = openai.AsyncOpenAI(
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
            )
            
            response = await client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            logger.error("OpenAI library not installed. Install with: pip install openai")
            return "Error: OpenAI library not installed"
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
            return f"Error: {str(e)}"
    
    async def _anthropic_provider(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Proveedor Anthropic."""
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(
                api_key=self.llm_api_key or os.getenv("ANTHROPIC_API_KEY"),
            )
            
            # Convertir formato de mensajes para Anthropic
            system_message = None
            anthropic_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            response = await client.messages.create(
                model=self.llm_model,
                max_tokens=self.llm_max_tokens,
                temperature=self.llm_temperature,
                system=system_message,
                messages=anthropic_messages,
                **kwargs
            )
            
            return response.content[0].text
            
        except ImportError:
            logger.error("Anthropic library not installed. Install with: pip install anthropic")
            return "Error: Anthropic library not installed"
        except Exception as e:
            logger.error(f"Error calling Anthropic: {e}")
            return f"Error: {str(e)}"
    
    async def _mock_provider(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Proveedor mock para testing."""
        import asyncio
        await asyncio.sleep(0.5)
        
        last_message = messages[-1]["content"] if messages else ""
        return f"Esta es una respuesta automática a: {last_message[:100]}... El chat continúa generando respuestas hasta que lo pausas."


from typing import Any, Dict, Optional, Tuple, List, Protocol, Union
from abc import ABC, abstractmethod
from pathlib import Path
import json
import logging
from agents.utils.config_utils import _resolve_api_key
from agents.models import InferenceResult

logger = logging.getLogger(__name__)

class AsyncLLMEngine(Protocol):
    """Protocol for any callable engine."""
    async def __call__(self, prompt: str, **kwargs) -> Union[str, InferenceResult]: ...

class BaseProvider(ABC):
    """Base class for all LLM providers."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, env_var: str = ""):
        custom_model = model
        try:
            prefs_path = Path(__file__).resolve().parent.parent / "user_preferences.json"
            if prefs_path.exists():
                import json
                data = json.loads(prefs_path.read_text())
                engine_models = data.get("engine_models", {})
                
                # Map env_var to preference key name
                env_to_key = {
                    "DEEPSEEK_API_KEY": "deepseek",
                    "GOOGLE_API_KEY": "google",
                    "OPENAI_API_KEY": "chatgpt",
                    "ANTHROPIC_API_KEY": "claude",
                    "OPENROUTER_API_KEY": "openrouter",
                }
                pref_key = env_to_key.get(env_var)
                if pref_key and pref_key in engine_models:
                    custom_model = engine_models[pref_key]
        except Exception:
            pass
            
        self.model = custom_model
        self.api_key = _resolve_api_key(env_var, api_key) if env_var else api_key
        self.timeout = 120.0

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass

    def _safe_fallback(self, thought: str, message: str, error: str = "provider_error") -> str:
        return json.dumps({
            "thought": thought,
            "tool": None,
            "tool_input": None,
            "final_answer": message,
            "metadata": {"error": error}
        })

class DummyAsyncLLM:
    """Mock engine that returns valid AgentAction JSON for testing."""
    model_name = "dummy-fallback"
    provider_name = "dummy"
    is_ensemble = False

    async def __call__(self, prompt: str, **kwargs) -> str:
        return json.dumps({
            "thought": "No hay motor LLM real configurado.",
            "tool": None,
            "tool_input": None,
            "final_answer": "âš ï¸ Motor de inferencia no configurado. Configura una API key en Settings > Engines."
        })
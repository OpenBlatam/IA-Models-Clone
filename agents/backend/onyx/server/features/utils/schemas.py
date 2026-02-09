from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from pydantic import BaseModel, Field
from typing import List, Type, TypeVar
import orjson
import random
import string

from typing import Any, List, Dict, Optional
import logging
import asyncio
T = TypeVar('T', bound='ORJSONModel')

class ORJSONModel(BaseModel):
    """BaseModel que serializa usando orjson para máxima velocidad y añade utilidades."""
    @dataclass
class Config:
        json_loads = orjson.loads
        json_dumps = lambda v, *, default: orjson.dumps(v, default=default).decode()
        orm_mode = True

    def orjson(self) -> str:
        """Serializa el modelo a JSON usando orjson."""
        return orjson.dumps(self.dict()).decode()

    @classmethod
    def parse_orjson(cls: Type[T], data: str) -> T:
        """Deserializa desde JSON usando orjson."""
        return cls.parse_raw(data)

    @classmethod
    def example(cls: Type[T]) -> T:
        return cls.parse_obj(cls._example())

    @classmethod
    def _example(cls) -> Any:
        return {}

class GenerationRequest(ORJSONModel):
    """Request para generación de texto LLM."""
    prompt: str
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_p: float = 1.0
    @classmethod
    def _example(cls) -> Any:
        return {"prompt": "Hola", "max_new_tokens": 32, "temperature": 1.0, "top_p": 1.0}
    @classmethod
    def random(cls) -> Any:
        return cls(prompt=''.join(random.choices(string.ascii_letters, k=10)), max_new_tokens=random.randint(1, 128), temperature=random.uniform(0.5,1.5), top_p=random.uniform(0.7,1.0))

class BatchGenerationRequest(ORJSONModel):
    """Request para batch de prompts."""
    prompts: List[str]
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_p: float = 1.0
    @classmethod
    def _example(cls) -> Any:
        return {"prompts": ["Hola", "¿Qué hora es?"], "max_new_tokens": 32, "temperature": 1.0, "top_p": 1.0}
    @classmethod
    def random(cls) -> Any:
        return cls(prompts=[''.join(random.choices(string.ascii_letters, k=10)) for _ in range(2)], max_new_tokens=random.randint(1, 128), temperature=random.uniform(0.5,1.5), top_p=random.uniform(0.7,1.0))

class GenerationResponse(ORJSONModel):
    """Respuesta de generación de texto."""
    result: str
    @classmethod
    def _example(cls) -> Any:
        return {"result": "¡Hola!"}
    @classmethod
    def random(cls) -> Any:
        return cls(result=''.join(random.choices(string.ascii_letters, k=20)))

class BatchGenerationResponse(ORJSONModel):
    """Respuesta para batch de generación."""
    results: List[str]
    @classmethod
    def _example(cls) -> Any:
        return {"results": ["Respuesta 1", "Respuesta 2"]}
    @classmethod
    def random(cls) -> Any:
        return cls(results=[''.join(random.choices(string.ascii_letters, k=20)) for _ in range(2)])

class TokenResponse(ORJSONModel):
    """Respuesta con access y refresh token."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    scopes: List[str]
    @classmethod
    def _example(cls) -> Any:
        return {"access_token": "abc", "refresh_token": "def", "token_type": "bearer", "expires_in": 900, "scopes": ["llm:predict"]}
    @classmethod
    def random(cls) -> Any:
        return cls(access_token=''.join(random.choices(string.ascii_letters, k=32)), refresh_token=''.join(random.choices(string.ascii_letters, k=32)), token_type="bearer", expires_in=random.randint(100,1000), scopes=["llm:predict"])

class RefreshTokenRequest(ORJSONModel):
    """Request para refrescar el access token."""
    refresh_token: str
    @classmethod
    def _example(cls) -> Any:
        return {"refresh_token": "def"}
    @classmethod
    def random(cls) -> Any:
        return cls(refresh_token=''.join(random.choices(string.ascii_letters, k=32))) 
from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import msgspec
from uuid6 import uuid7
from datetime import datetime
from typing import List, Any
import numpy as np
import zstandard as zstd
    import pandas as pd
from typing import Any, List, Dict, Optional
import logging
import asyncio
try:
except ImportError:
    pd = None

class Ad(msgspec.Struct, frozen=True, slots=True):
    """
    Modelo ultra-rápido para Ads con utilidades batch, compresión y validación.
    """
    __match_args__ = ("id", "title", "content", "metadata")
    id: str = msgspec.field(default_factory=lambda: str(uuid7()))
    title: str
    content: str
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)
    created_at: datetime = msgspec.field(default_factory=datetime.utcnow)
    updated_at: datetime = msgspec.field(default_factory=datetime.utcnow)
    created_by: str | None = None
    updated_by: str | None = None
    source: str | None = None
    version: int = 1
    trace_id: str | None = None
    is_deleted: bool = False

    def as_tuple(self) -> tuple:
        """Devuelve el ad como tupla (id, title, content, metadata)."""
        return (self.id, self.title, self.content, self.metadata)

    def to_training_example(self) -> dict:
        """Convierte el ad a un ejemplo de entrenamiento para ML."""
        return {"input": self.title, "output": self.content, "metadata": self.metadata}

    @classmethod
    def from_training_example(cls, example: dict) -> "Ad":
        """Crea un Ad desde un ejemplo de entrenamiento."""
        return cls(title=example["input"], content=example["output"], metadata=example.get("metadata", {}))

    @staticmethod
    def batch_encode(ads: List["Ad"]) -> bytes:
        """Serializa una lista de Ads a bytes usando msgspec.json."""
        return msgspec.json.encode(ads)

    @staticmethod
    def batch_decode(data: bytes) -> List["Ad"]:
        """Deserializa bytes a una lista de Ads usando msgspec.json."""
        return msgspec.json.decode(data, type=List[Ad])

    @staticmethod
    def batch_deduplicate(ads: List["Ad"]) -> List["Ad"]:
        """Elimina Ads duplicados por id en una lista."""
        seen = set()
        out = []
        for ad in ads:
            if ad.id not in seen:
                seen.add(ad.id)
                out.append(ad)
        return out

    @staticmethod
    def batch_to_training_examples(ads: List["Ad"]) -> List[dict]:
        """Convierte una lista de Ads a ejemplos de entrenamiento."""
        return [{"input": ad.title, "output": ad.content, "metadata": ad.metadata} for ad in ads]

    @staticmethod
    def batch_from_training_examples(examples: List[dict]) -> List["Ad"]:
        """Convierte una lista de ejemplos de entrenamiento a Ads."""
        return [Ad.from_training_example(ex) for ex in examples]

    @staticmethod
    def batch_as_tuples(ads: List["Ad"]) -> List[tuple]:
        """Convierte una lista de Ads a una lista de tuplas."""
        return [ad.as_tuple() for ad in ads]

    @staticmethod
    def batch_to_dicts(ads: List["Ad"]) -> List[dict]:
        """Convierte una lista de Ads a una lista de dicts."""
        return [ad.__dict__ for ad in ads]

    @staticmethod
    def batch_from_dicts(dicts: List[dict]) -> List["Ad"]:
        """Convierte una lista de dicts a Ads."""
        return [Ad(**d) for d in dicts]

    @staticmethod
    def batch_to_numpy(ads: List["Ad"]):
        """Convierte una lista de Ads a un array numpy."""
        arr = np.array([(d["id"], d["title"], d["content"], d["metadata"]) for d in Ad.batch_to_dicts(ads)], dtype=object)
        return arr

    @staticmethod
    def batch_to_pandas(ads: List["Ad"]):
        """Convierte una lista de Ads a un DataFrame de pandas."""
        if pd is None:
            raise ImportError("pandas is not installed")
        return pd.DataFrame(Ad.batch_to_dicts(ads))

    @staticmethod
    def batch_compress(ads: List["Ad"]) -> bytes:
        """Comprime una lista de Ads a bytes usando zstd."""
        data = msgspec.json.encode(ads)
        return zstd.ZstdCompressor().compress(data)

    @staticmethod
    def batch_decompress(data: bytes) -> List["Ad"]:
        """Descomprime bytes a una lista de Ads usando zstd."""
        decompressed = zstd.ZstdDecompressor().decompress(data)
        return msgspec.json.decode(decompressed, type=List[Ad])

    @staticmethod
    def validate(ad: "Ad") -> bool:
        """Valida un Ad: título y contenido no vacíos y con longitud mínima."""
        if not ad.title or not ad.content:
            return False
        if len(ad.title) < 2 or len(ad.content) < 10:
            return False
        return True

    def to_dict(self) -> dict:
        """Convierte el Ad a dict serializando fechas a ISO."""
        d = self.__dict__.copy()
        d["created_at"] = self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        d["updated_at"] = self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        return d

    @staticmethod
    def from_dict(d: dict) -> "Ad":
        """Crea un Ad desde un dict, parseando fechas si es necesario."""
        d = d.copy()
        if isinstance(d.get("created_at"), str):
            d["created_at"] = datetime.fromisoformat(d["created_at"])
        if isinstance(d.get("updated_at"), str):
            d["updated_at"] = datetime.fromisoformat(d["updated_at"])
        return Ad(**d) 
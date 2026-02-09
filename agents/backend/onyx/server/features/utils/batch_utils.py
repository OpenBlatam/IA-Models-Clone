from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List, Dict, Any, Type, TypeVar
import numpy as np
import pandas as pd

from typing import Any, List, Dict, Optional
import logging
import asyncio
T = TypeVar('T')

class BatchMethodsMixin:
    @classmethod
    def batch_to_dicts(cls: Type[T], objs: List[T]) -> List[dict]:
        return [obj.model_dump() if hasattr(obj, 'model_dump') else obj.dict() for obj in objs]

    @classmethod
    def batch_from_dicts(cls: Type[T], dicts: List[dict]) -> List[T]:
        return [cls(**d) for d in dicts]

    @classmethod
    def batch_to_numpy(cls: Type[T], objs: List[T]):
        
    """batch_to_numpy function."""
dicts = cls.batch_to_dicts(objs)
        return np.array(dicts)

    @classmethod
    def batch_to_pandas(cls: Type[T], objs: List[T]):
        
    """batch_to_pandas function."""
dicts = cls.batch_to_dicts(objs)
        return pd.DataFrame(dicts)

    @classmethod
    def batch_deduplicate(cls: Type[T], objs: List[T], key="id") -> List[T]:
        seen = set()
        result = []
        for obj in objs:
            k = getattr(obj, key, None)
            if k not in seen:
                seen.add(k)
                result.append(obj)
        return result

    @classmethod
    def to_training_example(cls: Type[T], obj: T) -> dict:
        return obj.model_dump() if hasattr(obj, 'model_dump') else obj.dict()

    @classmethod
    def from_training_example(cls: Type[T], data: dict) -> T:
        return cls(**data)

    @classmethod
    def batch_to_training_examples(cls: Type[T], objs: List[T]) -> List[dict]:
        return [cls.to_training_example(obj) for obj in objs]

    @classmethod
    def batch_from_training_examples(cls: Type[T], dicts: List[dict]) -> List[T]:
        return [cls.from_training_example(d) for d in dicts] 
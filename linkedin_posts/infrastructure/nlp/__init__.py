from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from functools import lru_cache
from typing import Protocol
from .pipelines.fast_pipeline import FastPipeline
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
NLP Infrastructure Package
==========================

Provides a unified interface (`get_nlp_pipeline`) that returns the
configured fast NLP enhancer with grammar, SEO and performance boosts.
"""



# Optional: here we could switch between different strategies (fast vs async)

class NLPPipeline(Protocol):
    async def enhance(self, text: str): ...


@lru_cache(maxsize=1)
def get_pipeline() -> NLPPipeline:
    """Return the default NLP pipeline (singleton)."""
    return FastPipeline() 
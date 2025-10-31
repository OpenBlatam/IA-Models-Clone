from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Tuple, List
from ..fast_nlp_enhancer import get_fast_nlp_enhancer

from typing import Any, List, Dict, Optional
import logging
import asyncio
class GrammarComponent:
    """Light wrapper that exposes grammar check from global enhancer."""
    def __init__(self) -> Any:
        self._enhancer = get_fast_nlp_enhancer()

    def check(self, text: str) -> Tuple[int, List[str]]:
        # reuse cached grammar function
        return self._enhancer._cached_grammar_check(text) 
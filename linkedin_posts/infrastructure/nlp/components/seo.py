from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, Any
from ..seo_analyser import analyse_seo

from typing import Any, List, Dict, Optional
import logging
import asyncio
class SEOComponent:
    def analyse(self, text: str) -> Dict[str, Any]:
        return analyse_seo(text) 
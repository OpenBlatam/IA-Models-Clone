from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
"""
Deprecated package. Use unified routes under `agents.backend.onyx.server.features.ads.api`.

This package is kept to avoid breaking older imports.
"""

import warnings

warnings.warn(
    "backend_ads is deprecated. Use unified ads.api routes instead.",
    DeprecationWarning,
    stacklevel=2,
)
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Backend Ads - ULTRA Fast Integration
Direct integration without any middleware.
"""

__all__: list[str] = []
from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .brute_forcers import (
from .exploiters import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Attackers module for cybersecurity testing and penetration testing tools.

This module provides tools for:
- Brute force attacks (password cracking, credential testing)
- Exploitation tools (vulnerability exploitation, payload generation)

All tools follow cybersecurity principles:
- Functional programming approach
- Async for I/O operations, def for CPU operations
- Type hints and Pydantic validation
- RORO pattern for tool interfaces
"""

    PasswordBruteForcer,
    CredentialTester,
    DictionaryAttacker,
    BruteForceConfig,
    BruteForceResult
)

    VulnerabilityExploiter,
    PayloadGenerator,
    ExploitFramework,
    ExploitConfig,
    ExploitResult
)

__all__ = [
    # Brute Force Tools
    'PasswordBruteForcer',
    'CredentialTester', 
    'DictionaryAttacker',
    'BruteForceConfig',
    'BruteForceResult',
    
    # Exploitation Tools
    'VulnerabilityExploiter',
    'PayloadGenerator',
    'ExploitFramework',
    'ExploitConfig',
    'ExploitResult'
] 
from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Any, List, Dict, Optional
import logging
import asyncio
broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'
accept_content = ['json']
task_serializer = 'json'
result_serializer = 'json'
timezone = 'UTC'
enable_utc = True 
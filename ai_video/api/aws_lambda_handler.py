from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi_microservice import app
from mangum import Mangum

from typing import Any, List, Dict, Optional
import logging
import asyncio
handler = Mangum(app) 
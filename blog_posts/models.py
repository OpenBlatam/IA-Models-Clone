from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

from typing import Any, List, Dict, Optional
import logging
import asyncio
class BlogPost(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        ser_json_timedelta="iso8601",
        ser_json_bytes="utf8",
        ser_json_bytes_io="utf8"
    )
    id: int = Field(..., gt=0)
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    tags: Optional[List[str]] = None
    is_published: bool = False

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "BlogPost":
        return cls.model_validate(data) 
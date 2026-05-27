import logging
import time
import uuid
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, computed_field

logger = logging.getLogger(__name__)


class Span(BaseModel):
    """A single traced event in an agent execution (Pydantic-validated)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    span_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: str = ""
    parent_id: Optional[str] = None
    name: str = ""
    agent_name: str = ""
    kind: str = Field(default="internal", description="llm_call | tool_call | routing | internal")
    input_data: str = ""
    output_data: str = ""
    status: str = Field(default="ok", description="ok | error")
    start_time: float = Field(default_factory=time.time)
    end_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[misc]
    @property
    def duration_ms(self) -> float:
        if self.end_time == 0.0:
            return 0.0
        return round((self.end_time - self.start_time) * 1000, 2)

    def finish(self, output: str = "", status: str = "ok", metadata: Optional[Dict[str, Any]] = None) -> None:
        self.end_time = time.time()
        self.output_data = output[:500]
        self.status = status
        if metadata:
            self.metadata.update(metadata)

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "agent": self.agent_name,
            "kind": self.kind,
            "input": self.input_data[:200],
            "output": self.output_data[:200],
            "status": self.status,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

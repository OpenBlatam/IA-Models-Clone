import logging
import time
import uuid
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, computed_field

logger = logging.getLogger(__name__)

from .models import Span

class Tracer:
    """
    Lightweight in-memory tracer for agent executions.

    Usage::

        tracer = Tracer()

        # Start a trace for a user request
        trace_id = tracer.start_trace("user_request", agent_name="ReActAgent")

        # Record a tool call
        span = tracer.start_span(trace_id, "web_search", kind="tool_call",
                                 input_data="search query")
        # ... tool executes ...
        span.finish(output="search results")

        # Get the full trace
        print(tracer.get_trace(trace_id))
    """

    def __init__(self, max_traces: int = 1000, persistence_path: str = "traces_history.json") -> None:
        self.max_traces = max_traces
        self.persistence_path = Path(persistence_path)
        self._traces: Dict[str, List[Span]] = {}
        self._trace_order: List[str] = []
        self._persistence_loaded = False
        import atexit
        atexit.register(self.finalize_all_on_exit)

    def finalize_all_on_exit(self) -> None:
        """Called automatically on program exit to finalize any active traces."""
        try:
            self._ensure_loaded()
            modified = False
            for tid, spans in self._traces.items():
                if spans and spans[0].end_time == 0.0:
                    spans[0].finish(output="Process exited", status="error", metadata={"exit_reason": "process_terminated"})
                    modified = True
                for span in spans[1:]:
                    if span.end_time == 0.0:
                        span.finish(output="Process exited", status="error", metadata={"exit_reason": "process_terminated"})
                        modified = True
            if modified:
                self._save_traces()
        except Exception:
            pass

    def _ensure_loaded(self) -> None:
        """Lazy-load persisted traces on first access."""
        if not self._persistence_loaded:
            self._load_traces()
            self._persistence_loaded = True

    def start_trace(self, name: str, agent_name: str = "") -> str:
        """Create a new trace and return its ID."""
        self._ensure_loaded()
        trace_id = str(uuid.uuid4())[:12]

        root_span = Span(
            trace_id=trace_id,
            name=name,
            agent_name=agent_name,
            kind="internal",
        )
        self._traces[trace_id] = [root_span]
        self._trace_order.append(trace_id)

        # Evict old traces
        while len(self._trace_order) > self.max_traces:
            old_id = self._trace_order.pop(0)
            self._traces.pop(old_id, None)

        self._save_traces()
        return trace_id

    def start_span(
        self,
        trace_id: str,
        name: str,
        kind: str = "internal",
        input_data: str = "",
        parent_id: Optional[str] = None,
        agent_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Add a new span to an existing trace."""
        self._ensure_loaded()
        span = Span(
            trace_id=trace_id,
            parent_id=parent_id,
            name=name,
            agent_name=agent_name,
            kind=kind,
            input_data=input_data[:500],
            metadata=metadata or {},
        )
        spans = self._traces.get(trace_id)
        if spans is not None:
            spans.append(span)
        return span

    def finish_trace(self, trace_id: str) -> None:
        """Mark the root span of the trace as finished."""
        spans = self._traces.get(trace_id)
        if spans:
            spans[0].finish()
            self._save_traces()

    def get_trace(self, trace_id: str) -> List[dict]:
        """Return all spans for a trace as dicts."""
        self._ensure_loaded()
        spans = self._traces.get(trace_id, [])
        return [s.to_dict() for s in spans]

    def get_recent_traces(self, limit: int = 20) -> List[dict]:
        """Return a summary of the most recent traces."""
        self._ensure_loaded()
        results = []
        for tid in reversed(self._trace_order[-limit:]):
            spans = self._traces.get(tid, [])
            if spans:
                root = spans[0]
                results.append({
                    "trace_id": tid,
                    "name": root.name,
                    "agent": root.agent_name,
                    "span_count": len(spans),
                    "duration_ms": root.duration_ms,
                    "status": root.status,
                })
        return results

    def get_stats(self) -> dict:
        """Return aggregate stats across all stored traces."""
        self._ensure_loaded()
        total_spans = sum(len(s) for s in self._traces.values())
        errors = sum(
            1
            for spans in self._traces.values()
            for s in spans
            if s.status == "error"
        )
        return {
            "total_traces": len(self._traces),
            "total_spans": total_spans,
            "error_spans": errors,
            "error_rate": round(errors / max(total_spans, 1), 4),
        }

    # ------------------------------------------------------------------
    # Persistence (uses Pydantic model_dump for serialization)
    # ------------------------------------------------------------------

    def _save_traces(self) -> None:
        """Serialize current traces to a JSON file via Pydantic model_dump."""
        try:
            data = {}
            for tid, spans in self._traces.items():
                data[tid] = [s.model_dump() for s in spans]

            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save trace history: %s", e)

    def _load_traces(self) -> None:
        """Load traces from the history file."""
        if not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for tid, spans_data in data.items():
                    spans = [Span.model_validate(s_data) for s_data in spans_data]
                    self._traces[tid] = spans
                    self._trace_order.append(tid)
            logger.info("Restored %d traces from persistence.", len(self._traces))
        except Exception as e:
            logger.error("Failed to load trace history: %s", e)

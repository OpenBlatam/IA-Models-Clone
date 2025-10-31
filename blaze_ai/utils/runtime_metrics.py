from __future__ import annotations

import threading
from typing import Any, Dict


class _RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, float]] = {}

    def record(self, engine: str, ok: bool, elapsed_ms: float) -> None:
        with self._lock:
            m = self._data.setdefault(engine, {"count": 0.0, "errors": 0.0, "sum_ms": 0.0})
            m["count"] += 1.0
            if not ok:
                m["errors"] += 1.0
            m["sum_ms"] += float(elapsed_ms)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            out: Dict[str, Any] = {}
            for name, m in self._data.items():
                count = m.get("count", 0.0)
                sum_ms = m.get("sum_ms", 0.0)
                errors = m.get("errors", 0.0)
                avg_ms = (sum_ms / count) if count > 0 else 0.0
                out[name] = {
                    "count": int(count),
                    "errors": int(errors),
                    "avg_ms": round(avg_ms, 2),
                }
            return out


runtime_metrics = _RuntimeMetrics()



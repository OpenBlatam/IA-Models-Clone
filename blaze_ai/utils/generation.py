from __future__ import annotations

from typing import Any, Dict


def format_chat_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def apply_generation_overrides(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defaults)
    out.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return out



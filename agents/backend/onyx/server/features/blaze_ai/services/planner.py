from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ScheduledItem:
    id: str
    channel: str
    content: Dict[str, Any]
    scheduled_at: datetime
    status: str = "scheduled"  # scheduled | sent | canceled


@dataclass
class Plan:
    id: str
    topic: str
    channels: List[str]
    cadence_days: int
    num_posts: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    items: List[ScheduledItem] = field(default_factory=list)


class ContentPlannerService:
    def __init__(self) -> None:
        self._plans: Dict[str, Plan] = {}
        self._scheduled: Dict[str, ScheduledItem] = {}

    def create_plan(self, topic: str, channels: List[str], cadence_days: int, num_posts: int) -> Dict[str, Any]:
        plan_id = str(uuid.uuid4())
        plan = Plan(id=plan_id, topic=topic, channels=channels, cadence_days=cadence_days, num_posts=num_posts)
        self._plans[plan_id] = plan
        return {"id": plan_id, "topic": topic, "channels": channels, "cadence_days": cadence_days, "num_posts": num_posts}

    def list_plans(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": p.id,
                "topic": p.topic,
                "channels": p.channels,
                "cadence_days": p.cadence_days,
                "num_posts": p.num_posts,
                "created_at": p.created_at.isoformat(),
                "scheduled": len(p.items),
            }
            for p in self._plans.values()
        ]

    def schedule_post(self, channel: str, content: Dict[str, Any], scheduled_at: datetime, plan_id: Optional[str] = None) -> Dict[str, Any]:
        item_id = str(uuid.uuid4())
        item = ScheduledItem(id=item_id, channel=channel, content=content, scheduled_at=scheduled_at)
        self._scheduled[item_id] = item
        if plan_id and plan_id in self._plans:
            self._plans[plan_id].items.append(item)
        return {"id": item_id, "channel": channel, "scheduled_at": scheduled_at.isoformat(), "status": item.status}

    def list_scheduled(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for it in self._scheduled.values():
            if status and it.status != status:
                continue
            out.append({
                "id": it.id,
                "channel": it.channel,
                "scheduled_at": it.scheduled_at.isoformat(),
                "status": it.status,
                "content": it.content,
            })
        return out

    def cancel(self, item_id: str) -> Dict[str, Any]:
        it = self._scheduled.get(item_id)
        if not it:
            return {"ok": False, "error": "not_found"}
        if it.status == "canceled":
            return {"ok": False, "error": "already_canceled"}
        it.status = "canceled"
        return {"ok": True, "id": it.id, "status": it.status}



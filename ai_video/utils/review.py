from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import msgspec
from typing import List, Optional

        from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
import asyncio
class ReviewInfo(msgspec.Struct, frozen=True, slots=True):
    """
    Información de permisos, revisión y aprobación.
    """
    permissions: Optional[dict] = None
    review_status: Optional[str] = None
    approval_history: List[dict] = msgspec.field(default_factory=list)

    def with_permissions(self, permissions: dict) -> 'ReviewInfo':
        return self.update(permissions=permissions)

    def set_review_status(self, status: str) -> 'ReviewInfo':
        return self.update(review_status=status)

    def add_approval_history(self, user_id: str, status: str, comment: Optional[str] = None, timestamp: Optional[str] = None) -> 'ReviewInfo':
        ts = timestamp or datetime.utcnow().isoformat()
        entry = {"user": user_id, "status": status, "timestamp": ts, "comment": comment}
        return self.update(approval_history=self.approval_history + [entry]) 
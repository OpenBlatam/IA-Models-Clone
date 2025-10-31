from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import msgspec
from typing import List, Optional

        from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
import asyncio
class ComplianceInfo(msgspec.Struct, frozen=True, slots=True):
    """
    Información de compliance, retención y auditoría.
    Integración con sistemas externos:
    - policy_ids: IDs de políticas externas (DLP, GDPR, etc)
    - retention_system_id: integración con sistemas de retención (Google Vault, AWS, etc)
    - audit_logs: logs de auditoría externos
    """
    deleted_at: Optional[str] = None
    deleted_by: Optional[str] = None
    is_archived: bool = False
    compliance_tags: List[str] = msgspec.field(default_factory=list)
    policy_ids: List[str] = msgspec.field(default_factory=list)
    retention_system_id: Optional[str] = None
    audit_logs: List[dict] = msgspec.field(default_factory=list)

    def archive(self) -> 'ComplianceInfo':
        return self.update(is_archived=True)

    def soft_delete(self, user_id: str, timestamp: Optional[str] = None) -> 'ComplianceInfo':
        ts = timestamp or datetime.utcnow().isoformat()
        return self.update(deleted_at=ts, deleted_by=user_id)

    def restore(self) -> 'ComplianceInfo':
        return self.update(deleted_at=None, deleted_by=None, is_archived=False)

    def with_compliance_tags(self, tags: List[str]) -> 'ComplianceInfo':
        return self.update(compliance_tags=tags)

    def with_policy_ids(self, policy_ids: List[str]) -> 'ComplianceInfo':
        return self.update(policy_ids=policy_ids)

    def with_retention_system(self, retention_id: str) -> 'ComplianceInfo':
        return self.update(retention_system_id=retention_id)

    def add_audit_log(self, log: dict) -> 'ComplianceInfo':
        return self.update(audit_logs=self.audit_logs + [log]) 
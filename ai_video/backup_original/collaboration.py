from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import msgspec
from typing import List, Optional, Dict

        from datetime import datetime
        from datetime import datetime
        from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
import asyncio
class CollaborationInfo(msgspec.Struct, frozen=True, slots=True):
    """
    Información colaborativa y de versionado para un video.
    Incluye comentarios, historial, propietarios, colaboradores, roles personalizados y timestamps de edición.
    Permite flujos de permisos colaborativos avanzados.
    Integración con librerías y sistemas externos:
    - chat_thread_id: integración con hilos de Slack, Discord, Teams, etc.
    - external_doc_links: integración con Google Docs, Notion, Confluence, etc.
    - comentarios soportan menciones (@user).
    """
    comments: List[dict] = msgspec.field(default_factory=list)
    history: List[dict] = msgspec.field(default_factory=list)
    owners: List[str] = msgspec.field(default_factory=list)
    collaborators: List[str] = msgspec.field(default_factory=list)
    roles: Dict[str, str] = msgspec.field(default_factory=dict)  # {user_id: role}
    last_edit: Dict[str, str] = msgspec.field(default_factory=dict)  # {user_id: timestamp}
    chat_thread_id: Optional[str] = None  # ID de hilo de chat externo (Slack, Discord, etc)
    external_doc_links: List[str] = msgspec.field(default_factory=list)  # URLs a Google Docs, Notion, etc

    def add_comment(self, user_id: str, text: str, timestamp: Optional[str] = None, mentions: Optional[List[str]] = None) -> 'CollaborationInfo':
        ts = timestamp or datetime.utcnow().isoformat()
        new_comment = {"user": user_id, "text": text, "timestamp": ts}
        if mentions:
            new_comment["mentions"] = mentions
        return self.update(comments=self.comments + [new_comment])

    def add_history(self, user_id: str, diff: dict, timestamp: Optional[str] = None) -> 'CollaborationInfo':
        ts = timestamp or datetime.utcnow().isoformat()
        new_entry = {"user": user_id, "diff": diff, "timestamp": ts}
        return self.update(history=self.history + [new_entry])

    def add_owner(self, user_id: str) -> 'CollaborationInfo':
        if user_id in self.owners:
            return self
        return self.update(owners=self.owners + [user_id])

    def add_collaborator(self, user_id: str) -> 'CollaborationInfo':
        if user_id in self.collaborators:
            return self
        return self.update(collaborators=self.collaborators + [user_id])

    def set_role(self, user_id: str, role: str) -> 'CollaborationInfo':
        roles = dict(self.roles)
        roles[user_id] = role
        return self.update(roles=roles)

    def set_last_edit(self, user_id: str, timestamp: Optional[str] = None) -> 'CollaborationInfo':
        ts = timestamp or datetime.utcnow().isoformat()
        last_edit = dict(self.last_edit)
        last_edit[user_id] = ts
        return self.update(last_edit=last_edit)

    def with_chat_thread(self, thread_id: str) -> 'CollaborationInfo':
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        return self.update(chat_thread_id=thread_id)

    def with_external_doc_links(self, links: List[str]) -> 'CollaborationInfo':
        return self.update(external_doc_links=links) 
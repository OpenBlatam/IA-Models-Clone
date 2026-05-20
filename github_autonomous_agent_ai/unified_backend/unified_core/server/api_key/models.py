from pydantic import BaseModel

from unified_core.auth.schemas import UserRole


class APIKeyArgs(BaseModel):
    name: str | None = None
    role: UserRole = UserRole.BASIC

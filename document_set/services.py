from typing import List, Optional
from uuid import UUID
from .models import DocumentSet
from .repositories import DocumentSetRepository


class DocumentSetService:
    def __init__(self, repository: DocumentSetRepository):
        self.repository = repository

    def get(self, docset_id: UUID) -> Optional[DocumentSet]:
        db_model = self.repository.get(docset_id)
        return DocumentSet.from_model(db_model) if db_model else None

    def list(self, limit: int = 50, offset: int = 0) -> List[DocumentSet]:
        rows = self.repository.list(limit=limit, offset=offset)
        return [DocumentSet.from_model(r) for r in rows]








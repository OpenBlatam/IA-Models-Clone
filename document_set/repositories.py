from typing import Optional, List
from uuid import UUID
from onyx.db.models import DocumentSet as DocumentSetDBModel
from sqlalchemy.orm import Session


class DocumentSetRepository:
    def __init__(self, session: Session):
        self.session = session

    def get(self, docset_id: UUID) -> Optional[DocumentSetDBModel]:
        return self.session.query(DocumentSetDBModel).filter(DocumentSetDBModel.id == docset_id).first()

    def list(self, limit: int = 50, offset: int = 0) -> List[DocumentSetDBModel]:
        return (
            self.session.query(DocumentSetDBModel)
            .order_by(DocumentSetDBModel.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )








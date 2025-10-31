from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException

from .models import DocumentSet
from .repositories import DocumentSetRepository
from .services import DocumentSetService


router = APIRouter(prefix="/document-sets", tags=["document_sets"])


def get_service() -> DocumentSetService:
    # NOTE: Replace this stub with your real DB session dependency
    from onyx.db.session import get_session

    session = next(get_session())
    return DocumentSetService(DocumentSetRepository(session))


@router.get("/{docset_id}", response_model=DocumentSet)
async def get_document_set(docset_id: UUID, svc: DocumentSetService = Depends(get_service)):
    result = svc.get(docset_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document set not found")
    return result


@router.get("", response_model=list[DocumentSet])
async def list_document_sets(limit: int = 50, offset: int = 0, svc: DocumentSetService = Depends(get_service)):
    return svc.list(limit=limit, offset=offset)








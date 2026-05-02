"""
Indexing logic for Copywriting records, following Onyx backend conventions.
Supports CRUD, embedding, search, and batch operations.
"""
from sqlalchemy.orm import Session
from .copywriting import Copywriting, CopywritingCreate, CopywritingRead
from typing import List, Optional, Any
from onyx.utils.logger import setup_logger

logger = setup_logger()

class CopywritingIndex:
    """Indexing and retrieval logic for Copywriting records."""

    @staticmethod
    def add(session: Session, data: CopywritingCreate) -> Copywriting:
        """Add a new copywriting record to the database."""
        record = Copywriting(
            use_case=data.use_case,
            input_data=data.input_data,
            output_data=""  # Output can be filled after LLM generation
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        logger.info(f"Added copywriting record with id={record.id}")
        return record

    @staticmethod
    def update_output(session: Session, record_id: int, output_data: str) -> Optional[Copywriting]:
        """Update the output_data of a copywriting record."""
        record = session.query(Copywriting).filter(Copywriting.id == record_id).first()
        if record:
            record.output_data = output_data
            session.commit()
            logger.info(f"Updated output for copywriting id={record_id}")
        return record

    @staticmethod
    def get(session: Session, record_id: int) -> Optional[Copywriting]:
        """Retrieve a copywriting record by id."""
        return session.query(Copywriting).filter(Copywriting.id == record_id).first()

    @staticmethod
    def list(session: Session, use_case: Optional[str] = None) -> List[Copywriting]:
        """List copywriting records, optionally filtered by use_case."""
        query = session.query(Copywriting)
        if use_case:
            query = query.filter(Copywriting.use_case == use_case)
        return query.all()

    @staticmethod
    def delete(session: Session, record_id: int) -> bool:
        """Delete a copywriting record by id."""
        record = session.query(Copywriting).filter(Copywriting.id == record_id).first()
        if record:
            session.delete(record)
            session.commit()
            logger.info(f"Deleted copywriting record id={record_id}")
            return True
        return False

    @staticmethod
    def batch_add(session: Session, data_list: List[CopywritingCreate]) -> List[Copywriting]:
        """Batch add copywriting records."""
        records = [Copywriting(use_case=data.use_case, input_data=data.input_data, output_data="") for data in data_list]
        session.add_all(records)
        session.commit()
        for record in records:
            session.refresh(record)
        logger.info(f"Batch added {len(records)} copywriting records.")
        return records

    @staticmethod
    def search(session: Session, query: str, use_case: Optional[str] = None) -> List[Copywriting]:
        """Search copywriting records by input_data or output_data."""
        q = session.query(Copywriting).filter(
            (Copywriting.input_data.ilike(f"%{query}%")) |
            (Copywriting.output_data.ilike(f"%{query}%"))
        )
        if use_case:
            q = q.filter(Copywriting.use_case == use_case)
        results = q.all()
        logger.info(f"Search for '{query}' returned {len(results)} records.")
        return results

    @staticmethod
    def embed(record: Copywriting) -> Any:
        """Stub for embedding logic (to be implemented as needed)."""
        # Example: return embedding_model.embed(record.input_data)
        logger.info(f"Embedding for copywriting id={record.id} requested.")
        return None 
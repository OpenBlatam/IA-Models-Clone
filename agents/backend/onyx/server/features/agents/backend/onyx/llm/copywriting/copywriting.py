"""
Copywriting database model and helpers for Onyx backend.
"""
from sqlalchemy.orm import registry
from sqlalchemy import Column, Integer, String, Text, DateTime, func
from pydantic import BaseModel
from typing import Optional

mapper_registry = registry()

@mapper_registry.mapped
class Copywriting:
    __tablename__ = "copywriting"
    id = Column(Integer, primary_key=True, index=True)
    use_case = Column(String(64), nullable=False)
    input_data = Column(Text, nullable=False)
    output_data = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class CopywritingCreate(BaseModel):
    use_case: str
    input_data: str

class CopywritingRead(BaseModel):
    id: int
    use_case: str
    input_data: str
    output_data: str
    created_at: Optional[str]
    class Config:
        orm_mode = True

def create_copywriting_table(engine):
    """Create the copywriting table in the database."""
    mapper_registry.metadata.create_all(engine)

__all__ = [
    "Copywriting",
    "CopywritingCreate",
    "CopywritingRead",
    "create_copywriting_table",
] 
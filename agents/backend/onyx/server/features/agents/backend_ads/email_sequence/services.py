from typing import List, Optional, Dict
from datetime import datetime
import uuid
from .models import (
    EmailSequenceRequest,
    EmailSequenceResponse,
    EmailSequenceMetrics,
    EmailTemplate,
    BrandVoice,
    AudienceProfile,
    ProjectContext
)

class EmailSequenceService:
    def __init__(self):
        # In a real implementation, this would be a database connection
        self.sequences: Dict[str, EmailSequenceResponse] = {}
        self.metrics: Dict[str, EmailSequenceMetrics] = {}

    async def create_sequence(self, request: EmailSequenceRequest) -> EmailSequenceResponse:
        """
        Create a new email sequence
        """
        sequence_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        sequence = EmailSequenceResponse(
            sequence_id=sequence_id,
            sequence_name=f"Sequence {sequence_id[:8]}",
            description=request.prompt,
            templates=[],
            estimated_completion_days=request.number_of_emails,
            created_at=now,
            updated_at=now,
            status="draft"
        )
        
        self.sequences[sequence_id] = sequence
        return sequence

    async def get_sequence(self, sequence_id: str) -> Optional[EmailSequenceResponse]:
        """
        Get a sequence by ID
        """
        return self.sequences.get(sequence_id)

    async def get_metrics(self, sequence_id: str) -> Optional[EmailSequenceMetrics]:
        """
        Get metrics for a sequence
        """
        return self.metrics.get(sequence_id)

    async def update_status(self, sequence_id: str, status: str) -> bool:
        """
        Update sequence status
        """
        if sequence_id not in self.sequences:
            return False
        
        sequence = self.sequences[sequence_id]
        sequence.status = status
        sequence.updated_at = datetime.utcnow().isoformat()
        return True

    async def add_template(self, sequence_id: str, template: EmailTemplate) -> bool:
        """
        Add a template to a sequence
        """
        if sequence_id not in self.sequences:
            return False
        
        sequence = self.sequences[sequence_id]
        sequence.templates.append(template)
        sequence.updated_at = datetime.utcnow().isoformat()
        return True

    async def list_sequences(
        self,
        status: Optional[str] = None,
        sequence_type: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[EmailSequenceResponse]:
        """
        List sequences with optional filtering
        """
        sequences = list(self.sequences.values())
        
        if status:
            sequences = [s for s in sequences if s.status == status]
        
        if sequence_type:
            sequences = [s for s in sequences if s.sequence_type == sequence_type]
        
        return sequences[offset:offset + limit]

    async def delete_sequence(self, sequence_id: str) -> bool:
        """
        Delete a sequence
        """
        if sequence_id not in self.sequences:
            return False
        
        del self.sequences[sequence_id]
        if sequence_id in self.metrics:
            del self.metrics[sequence_id]
        return True

    async def duplicate_sequence(self, sequence_id: str) -> Optional[EmailSequenceResponse]:
        """
        Create a duplicate of a sequence
        """
        if sequence_id not in self.sequences:
            return None
        
        original = self.sequences[sequence_id]
        new_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        new_sequence = EmailSequenceResponse(
            sequence_id=new_id,
            sequence_name=f"Copy of {original.sequence_name}",
            description=original.description,
            templates=original.templates.copy(),
            estimated_completion_days=original.estimated_completion_days,
            created_at=now,
            updated_at=now,
            status="draft"
        )
        
        self.sequences[new_id] = new_sequence
        return new_sequence 
import pytest
from datetime import datetime
from ..services import EmailSequenceService
from ..models import (
    EmailSequenceRequest,
    EmailTemplate,
    BrandVoice,
    AudienceProfile,
    ProjectContext
)

@pytest.fixture
def service():
    return EmailSequenceService()

@pytest.fixture
def sample_request():
    brand_voice = BrandVoice(tone="professional")
    audience = AudienceProfile(customer_stage="awareness")
    project = ProjectContext(
        project_name="Test",
        project_description="Test",
        industry="Tech"
    )
    
    return EmailSequenceRequest(
        type="email-sequence",
        prompt="Test prompt",
        target_audience="Test audience",
        goals=["goal1", "goal2"],
        brand_voice=brand_voice,
        audience_profile=audience,
        project_context=project
    )

@pytest.fixture
def sample_template():
    return EmailTemplate(
        subject="Test Subject",
        body="Test Body",
        delay_days=1,
        template_type="welcome"
    )

@pytest.mark.asyncio
async def test_create_sequence(service, sample_request):
    sequence = await service.create_sequence(sample_request)
    assert sequence.sequence_id is not None
    assert sequence.status == "draft"
    assert sequence.description == sample_request.prompt

@pytest.mark.asyncio
async def test_get_sequence(service, sample_request):
    created = await service.create_sequence(sample_request)
    retrieved = await service.get_sequence(created.sequence_id)
    assert retrieved is not None
    assert retrieved.sequence_id == created.sequence_id

@pytest.mark.asyncio
async def test_update_status(service, sample_request):
    created = await service.create_sequence(sample_request)
    success = await service.update_status(created.sequence_id, "active")
    assert success is True
    
    updated = await service.get_sequence(created.sequence_id)
    assert updated.status == "active"

@pytest.mark.asyncio
async def test_add_template(service, sample_request, sample_template):
    created = await service.create_sequence(sample_request)
    success = await service.add_template(created.sequence_id, sample_template)
    assert success is True
    
    updated = await service.get_sequence(created.sequence_id)
    assert len(updated.templates) == 1
    assert updated.templates[0].subject == sample_template.subject

@pytest.mark.asyncio
async def test_list_sequences(service, sample_request):
    # Create multiple sequences
    await service.create_sequence(sample_request)
    await service.create_sequence(sample_request)
    
    # Test listing all sequences
    sequences = await service.list_sequences()
    assert len(sequences) == 2
    
    # Test filtering by status
    sequences = await service.list_sequences(status="draft")
    assert len(sequences) == 2
    
    # Test pagination
    sequences = await service.list_sequences(limit=1)
    assert len(sequences) == 1

@pytest.mark.asyncio
async def test_delete_sequence(service, sample_request):
    created = await service.create_sequence(sample_request)
    success = await service.delete_sequence(created.sequence_id)
    assert success is True
    
    deleted = await service.get_sequence(created.sequence_id)
    assert deleted is None

@pytest.mark.asyncio
async def test_duplicate_sequence(service, sample_request, sample_template):
    # Create and populate original sequence
    original = await service.create_sequence(sample_request)
    await service.add_template(original.sequence_id, sample_template)
    
    # Duplicate sequence
    duplicate = await service.duplicate_sequence(original.sequence_id)
    assert duplicate is not None
    assert duplicate.sequence_id != original.sequence_id
    assert duplicate.sequence_name.startswith("Copy of")
    assert len(duplicate.templates) == len(original.templates)

@pytest.mark.asyncio
async def test_get_metrics(service, sample_request):
    created = await service.create_sequence(sample_request)
    metrics = await service.get_metrics(created.sequence_id)
    assert metrics is None  # Initially no metrics exist 
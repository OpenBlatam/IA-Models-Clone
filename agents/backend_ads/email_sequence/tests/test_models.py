import pytest
from datetime import datetime
from ..models import (
    BrandVoice,
    AudienceProfile,
    ContentSource,
    ProjectContext,
    EmailTemplate,
    EmailSequenceRequest,
    EmailSequenceResponse,
    EmailSequenceMetrics
)

def test_brand_voice_creation():
    brand_voice = BrandVoice(
        tone="professional",
        style="conversational",
        personality_traits=["friendly", "authoritative"],
        industry_specific_terms=["SaaS", "cloud"]
    )
    assert brand_voice.tone == "professional"
    assert brand_voice.style == "conversational"
    assert len(brand_voice.personality_traits) == 2
    assert len(brand_voice.industry_specific_terms) == 2

def test_audience_profile_creation():
    audience = AudienceProfile(
        demographics={
            "age_range": "25-35",
            "location": "US"
        },
        interests=["technology", "marketing"],
        pain_points=["time management", "automation"],
        customer_stage="awareness"
    )
    assert audience.demographics["age_range"] == "25-35"
    assert len(audience.interests) == 2
    assert audience.customer_stage == "awareness"

def test_content_source_creation():
    source = ContentSource(
        type="url",
        content="https://example.com",
        priority=1
    )
    assert source.type == "url"
    assert source.priority == 1

def test_project_context_creation():
    context = ProjectContext(
        project_name="Test Project",
        project_description="Test Description",
        industry="Technology",
        key_messages=["Message 1", "Message 2"]
    )
    assert context.project_name == "Test Project"
    assert len(context.key_messages) == 2

def test_email_template_creation():
    template = EmailTemplate(
        subject="Test Subject",
        body="Test Body",
        delay_days=1,
        template_type="welcome"
    )
    assert template.subject == "Test Subject"
    assert template.delay_days == 1
    assert template.template_type == "welcome"

def test_email_sequence_request_creation():
    brand_voice = BrandVoice(tone="professional")
    audience = AudienceProfile(customer_stage="awareness")
    project = ProjectContext(
        project_name="Test",
        project_description="Test",
        industry="Tech"
    )
    
    request = EmailSequenceRequest(
        type="email-sequence",
        prompt="Test prompt",
        target_audience="Test audience",
        goals=["goal1", "goal2"],
        brand_voice=brand_voice,
        audience_profile=audience,
        project_context=project
    )
    assert request.type == "email-sequence"
    assert len(request.goals) == 2

def test_email_sequence_response_creation():
    response = EmailSequenceResponse(
        sequence_id="test-id",
        sequence_name="Test Sequence",
        description="Test Description",
        templates=[],
        estimated_completion_days=5,
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat()
    )
    assert response.sequence_id == "test-id"
    assert response.status == "draft"

def test_email_sequence_metrics_creation():
    metrics = EmailSequenceMetrics(
        sequence_id="test-id",
        total_sent=100,
        opens=50,
        clicks=25,
        conversions=10,
        bounces=2,
        unsubscribes=1,
        revenue=1000.0,
        last_updated=datetime.utcnow().isoformat(),
        engagement_score=0.5,
        delivery_rate=0.98
    )
    assert metrics.total_sent == 100
    assert metrics.engagement_score == 0.5
    assert metrics.delivery_rate == 0.98 
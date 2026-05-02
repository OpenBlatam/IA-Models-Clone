import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from ..api import router
from ..services import EmailSequenceService

@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)
    return app

@pytest.fixture
def client(app):
    return TestClient(app)

@pytest.fixture
def email_service():
    return EmailSequenceService()

@pytest.fixture
def sample_brand_voice():
    return {
        "tone": "professional",
        "style": "conversational",
        "personality_traits": ["friendly", "authoritative"],
        "industry_specific_terms": ["SaaS", "cloud"]
    }

@pytest.fixture
def sample_audience_profile():
    return {
        "demographics": {
            "age_range": "25-35",
            "location": "US"
        },
        "interests": ["technology", "marketing"],
        "pain_points": ["time management", "automation"],
        "customer_stage": "awareness"
    }

@pytest.fixture
def sample_project_context():
    return {
        "project_name": "Test Project",
        "project_description": "Test Description",
        "industry": "Technology",
        "key_messages": ["Message 1", "Message 2"]
    }

@pytest.fixture
def sample_email_template():
    return {
        "subject": "Test Subject",
        "body": "Test Body",
        "delay_days": 1,
        "template_type": "welcome",
        "tracking_enabled": True
    }

@pytest.fixture
def sample_sequence_request(sample_brand_voice, sample_audience_profile, sample_project_context):
    return {
        "type": "email-sequence",
        "prompt": "Test prompt",
        "target_audience": "Test audience",
        "goals": ["goal1", "goal2"],
        "brand_voice": sample_brand_voice,
        "audience_profile": sample_audience_profile,
        "project_context": sample_project_context,
        "number_of_emails": 5,
        "language": "en-US",
        "timezone": "UTC"
    } 
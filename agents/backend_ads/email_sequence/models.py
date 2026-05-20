from pydantic import BaseModel
from typing import Literal, Optional, List, Dict
from datetime import datetime

class BrandVoice(BaseModel):
    tone: Literal["professional", "casual", "friendly", "authoritative", "humorous", "formal"] = "professional"
    style: Literal["conversational", "technical", "storytelling", "persuasive", "educational"] = "conversational"
    personality_traits: List[str] = []
    industry_specific_terms: List[str] = []
    brand_guidelines: Optional[dict] = None

class AudienceProfile(BaseModel):
    demographics: dict = {
        "age_range": Optional[str],
        "gender": Optional[str],
        "location": Optional[str],
        "occupation": Optional[str],
        "income_level": Optional[str]
    }
    interests: List[str] = []
    pain_points: List[str] = []
    goals: List[str] = []
    buying_behavior: Optional[dict] = None
    customer_stage: Literal["awareness", "consideration", "decision", "retention", "advocacy"] = "awareness"

class ContentSource(BaseModel):
    type: Literal["file", "text", "url", "knowledge_base"]
    content: str
    priority: int = 1
    relevance_score: Optional[float] = None

class ProjectContext(BaseModel):
    project_name: str
    project_description: str
    industry: str
    key_messages: List[str] = []
    brand_assets: List[str] = []
    content_sources: List[ContentSource] = []
    custom_variables: dict = {}

class EmailTemplate(BaseModel):
    subject: str
    body: str
    delay_days: int
    tracking_enabled: bool = True
    template_type: Literal["welcome", "follow-up", "promotional", "educational", "re-engagement"] = "follow-up"
    personalization_fields: List[str] = []
    attachments: List[str] = []
    a_b_test_variants: Optional[List[dict]] = None
    conditions: Optional[dict] = None
    ai_generated_image: Optional[dict] = None
    content_sources_used: List[str] = []
    brand_voice_applied: dict = {}
    audience_segment: Optional[dict] = None
    call_to_action: Optional[dict] = None
    dynamic_content: Optional[dict] = None

class EmailSequenceRequest(BaseModel):
    type: Literal["email-sequence"]
    prompt: str
    target_audience: str
    goals: List[str]
    number_of_emails: int = 5
    sequence_type: Literal["onboarding", "nurturing", "sales", "re-engagement", "custom"] = "nurturing"
    language: str = "en-US"
    timezone: str = "UTC"
    schedule_type: Literal["immediate", "scheduled", "triggered"] = "immediate"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    max_emails_per_day: int = 1
    fallback_sequence: Optional[str] = None
    tags: List[str] = []
    brand_voice: BrandVoice
    audience_profile: AudienceProfile
    project_context: ProjectContext
    content_sources: List[ContentSource] = []
    include_ai_generated_images: bool = False
    image_style: Optional[dict] = None
    custom_templates: Optional[List[dict]] = None
    a_b_testing_config: Optional[dict] = None
    personalization_rules: Optional[List[dict]] = None

class EmailSequenceResponse(BaseModel):
    sequence_id: str
    sequence_name: str
    description: str
    templates: List[EmailTemplate]
    estimated_completion_days: int
    status: Literal["draft", "active", "paused", "completed", "archived"] = "draft"
    created_at: str
    updated_at: str
    metrics: dict = {
        "open_rate": 0.0,
        "click_rate": 0.0,
        "conversion_rate": 0.0,
        "bounce_rate": 0.0,
        "unsubscribe_rate": 0.0,
        "revenue_generated": 0.0
    }
    audience_segment: Optional[dict] = None
    performance_insights: Optional[List[str]] = None
    content_quality_score: Optional[float] = None
    brand_voice_consistency: Optional[float] = None
    audience_relevance_score: Optional[float] = None
    generated_images: Optional[List[dict]] = None
    content_sources_summary: Optional[dict] = None

class EmailSequenceMetrics(BaseModel):
    sequence_id: str
    total_sent: int
    opens: int
    clicks: int
    conversions: int
    bounces: int
    unsubscribes: int
    revenue: float
    last_updated: str
    engagement_score: float
    delivery_rate: float
    spam_complaints: int
    device_stats: dict = {
        "mobile": 0,
        "desktop": 0,
        "tablet": 0
    }
    location_stats: dict = {}
    time_stats: dict = {
        "best_time_to_send": None,
        "average_open_time": None
    } 
"""
Test data fixtures for the ads feature.

This module provides comprehensive test data for:
- Domain entities (Ad, AdCampaign, AdGroup, AdPerformance)
- Application DTOs (requests and responses)
- Configuration objects
- Test scenarios and edge cases
"""

import pytest
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid

# Import domain entities and DTOs
from agents.backend.onyx.server.features.ads.domain.entities import Ad, AdCampaign, AdGroup, AdPerformance
from agents.backend.onyx.server.features.ads.domain.value_objects import AdStatus, AdType, Platform, Budget, TargetingCriteria
from agents.backend.onyx.server.features.ads.application.dto import (
    CreateAdRequest, CreateAdResponse, ApproveAdRequest, ApproveAdResponse,
    ActivateAdRequest, ActivateAdResponse, PauseAdRequest, PauseAdResponse,
    CreateCampaignRequest, CreateCampaignResponse, OptimizationRequest, OptimizationResponse
)


class TestDataFixtures:
    """Test data fixtures for ads feature testing."""

    @pytest.fixture
    def sample_ad_data(self):
        """Sample ad data for testing."""
        return {
            "id": "test-ad-123",
            "title": "Test Advertisement",
            "description": "A test advertisement for testing purposes",
            "status": "draft",
            "ad_type": "display",
            "platform": "facebook",
            "budget": 1000.0,
            "targeting_criteria": {
                "age_range": [25, 45],
                "interests": ["technology", "business"],
                "location": "United States",
                "gender": "all"
            },
            "creative_assets": {
                "images": ["image1.jpg", "image2.jpg"],
                "videos": ["video1.mp4"],
                "text": "Get the best deals on tech products!"
            },
            "metrics": {
                "impressions": 0,
                "clicks": 0,
                "conversions": 0,
                "spend": 0.0
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

    @pytest.fixture
    def sample_campaign_data(self):
        """Sample campaign data for testing."""
        return {
            "id": "test-campaign-123",
            "name": "Test Campaign",
            "description": "A test advertising campaign",
            "status": "active",
            "budget": 5000.0,
            "start_date": datetime.now(),
            "end_date": datetime.now() + timedelta(days=30),
            "targeting_criteria": {
                "age_range": [18, 65],
                "interests": ["technology", "business", "finance"],
                "location": "United States",
                "gender": "all"
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

    @pytest.fixture
    def sample_ad_group_data(self):
        """Sample ad group data for testing."""
        return {
            "id": "test-group-123",
            "name": "Test Ad Group",
            "description": "A test ad group within a campaign",
            "status": "active",
            "campaign_id": "test-campaign-123",
            "budget": 2000.0,
            "targeting_criteria": {
                "age_range": [25, 45],
                "interests": ["technology"],
                "location": "California",
                "gender": "all"
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

    @pytest.fixture
    def sample_performance_data(self):
        """Sample performance data for testing."""
        return {
            "id": "test-performance-123",
            "ad_id": "test-ad-123",
            "date": datetime.now().date(),
            "impressions": 1000,
            "clicks": 50,
            "conversions": 5,
            "spend": 150.0,
            "ctr": 0.05,
            "cpc": 3.0,
            "cpm": 150.0,
            "conversion_rate": 0.10,
            "roas": 2.5
        }

    @pytest.fixture
    def sample_budget_data(self):
        """Sample budget data for testing."""
        return {
            "amount": 1000.0,
            "currency": "USD",
            "duration_days": 30,
            "daily_limit": 50.0,
            "lifetime_limit": 1000.0,
            "billing_type": "daily"
        }

    @pytest.fixture
    def sample_targeting_criteria_data(self):
        """Sample targeting criteria data for testing."""
        return {
            "age_range": [25, 45],
            "interests": ["technology", "business", "finance"],
            "location": "United States",
            "gender": "all",
            "languages": ["English"],
            "devices": ["mobile", "desktop"],
            "platforms": ["facebook", "instagram"],
            "custom_audiences": ["high_value_customers"],
            "excluded_audiences": ["existing_customers"]
        }

    @pytest.fixture
    def sample_creative_assets_data(self):
        """Sample creative assets data for testing."""
        return {
            "images": [
                {
                    "url": "https://example.com/image1.jpg",
                    "alt_text": "Product showcase",
                    "dimensions": {"width": 1200, "height": 630}
                },
                {
                    "url": "https://example.com/image2.jpg",
                    "alt_text": "Brand logo",
                    "dimensions": {"width": 800, "height": 800}
                }
            ],
            "videos": [
                {
                    "url": "https://example.com/video1.mp4",
                    "duration": 30,
                    "thumbnail": "https://example.com/thumbnail1.jpg"
                }
            ],
            "text": {
                "headline": "Get the Best Deals on Tech Products!",
                "description": "Discover amazing offers on the latest technology",
                "call_to_action": "Shop Now"
            }
        }

    @pytest.fixture
    def sample_metrics_data(self):
        """Sample metrics data for testing."""
        return {
            "impressions": 1000,
            "clicks": 50,
            "conversions": 5,
            "spend": 150.0,
            "ctr": 0.05,
            "cpc": 3.0,
            "cpm": 150.0,
            "conversion_rate": 0.10,
            "roas": 2.5,
            "reach": 800,
            "frequency": 1.25,
            "engagement_rate": 0.08
        }

    @pytest.fixture
    def sample_schedule_data(self):
        """Sample schedule data for testing."""
        return {
            "start_date": datetime.now(),
            "end_date": datetime.now() + timedelta(days=30),
            "time_zones": ["UTC-5", "UTC-8"],
            "day_parts": ["morning", "afternoon", "evening"],
            "days_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"],
            "blackout_dates": [datetime.now() + timedelta(days=15)]
        }

    @pytest.fixture
    def sample_create_ad_request_data(self):
        """Sample create ad request data for testing."""
        return {
            "title": "Test Advertisement",
            "description": "A test advertisement for testing purposes",
            "brand_voice": "Professional and friendly",
            "target_audience": "Tech professionals aged 25-45",
            "platform": "facebook",
            "budget": 1000.0,
            "targeting_criteria": {
                "age_range": [25, 45],
                "interests": ["technology", "business"],
                "location": "United States",
                "gender": "all"
            },
            "creative_assets": {
                "images": ["image1.jpg", "image2.jpg"],
                "videos": ["video1.mp4"],
                "text": "Get the best deals on tech products!"
            }
        }

    @pytest.fixture
    def sample_create_campaign_request_data(self):
        """Sample create campaign request data for testing."""
        return {
            "name": "Test Campaign",
            "description": "A test advertising campaign",
            "budget": 5000.0,
            "start_date": datetime.now(),
            "end_date": datetime.now() + timedelta(days=30),
            "targeting_criteria": {
                "age_range": [18, 65],
                "interests": ["technology", "business", "finance"],
                "location": "United States",
                "gender": "all"
            }
        }

    @pytest.fixture
    def sample_optimization_request_data(self):
        """Sample optimization request data for testing."""
        return {
            "ad_id": "test-ad-123",
            "optimization_type": "performance",
            "target_metrics": ["ctr", "conversion_rate"],
            "constraints": {
                "max_budget": 1500.0,
                "min_impressions": 500
            },
            "optimization_level": "aggressive"
        }

    @pytest.fixture
    def sample_validation_error_data(self):
        """Sample validation error data for testing."""
        return {
            "field": "budget",
            "value": -100,
            "message": "Budget must be positive",
            "error_code": "INVALID_BUDGET"
        }

    @pytest.fixture
    def sample_business_rule_violation_data(self):
        """Sample business rule violation data for testing."""
        return {
            "rule": "CAMPAIGN_BUDGET_EXCEEDED",
            "message": "Campaign budget cannot exceed $10,000",
            "current_value": 12000.0,
            "max_allowed": 10000.0
        }

    @pytest.fixture
    def sample_edge_case_data(self):
        """Sample edge case data for testing."""
        return {
            "zero_budget": {
                "amount": 0.0,
                "currency": "USD",
                "duration_days": 30
            },
            "max_budget": {
                "amount": 999999.99,
                "currency": "USD",
                "duration_days": 365
            },
            "empty_targeting": {
                "age_range": [],
                "interests": [],
                "location": "",
                "gender": ""
            },
            "special_characters": {
                "title": "Ad with special chars: !@#$%^&*()",
                "description": "Description with emojis 🚀💡✨"
            }
        }

    @pytest.fixture
    def sample_performance_test_data(self):
        """Sample data for performance testing."""
        return {
            "large_dataset": {
                "ads_count": 10000,
                "campaigns_count": 1000,
                "daily_metrics": 30,
                "targeting_criteria_count": 50
            },
            "concurrent_operations": {
                "simultaneous_creates": 100,
                "simultaneous_updates": 50,
                "simultaneous_reads": 200
            },
            "memory_intensive": {
                "large_images": ["large_image_10mb.jpg"] * 100,
                "large_videos": ["large_video_100mb.mp4"] * 10,
                "extensive_targeting": {"interests": ["interest"] * 1000}
            }
        }

    @pytest.fixture
    def sample_integration_test_data(self):
        """Sample data for integration testing."""
        return {
            "cross_layer_entities": {
                "domain_entity": Ad(
                    id="integration-test-ad",
                    title="Integration Test Ad",
                    description="Ad for integration testing",
                    status="draft",
                    platform="facebook",
                    budget=1000.0
                ),
                "dto_request": CreateAdRequest(
                    title="Integration Test Ad",
                    description="Ad for integration testing",
                    brand_voice="Professional",
                    target_audience="Test audience",
                    platform="facebook",
                    budget=1000.0
                ),
                "dto_response": CreateAdResponse(
                    success=True,
                    ad_id="integration-test-ad",
                    message="Ad created successfully",
                    ad_data={"id": "integration-test-ad", "title": "Integration Test Ad"}
                )
            },
            "service_interactions": {
                "ad_service_input": {
                    "title": "Service Test Ad",
                    "description": "Ad for service testing",
                    "platform": "facebook",
                    "budget": 1000.0
                },
                "campaign_service_input": {
                    "name": "Service Test Campaign",
                    "description": "Campaign for service testing",
                    "budget": 5000.0
                }
            }
        }

    @pytest.fixture
    def sample_error_scenario_data(self):
        """Sample data for error scenario testing."""
        return {
            "database_errors": {
                "connection_timeout": "Database connection timeout after 30 seconds",
                "constraint_violation": "Unique constraint violation on ad_id",
                "deadlock": "Database deadlock detected"
            },
            "validation_errors": {
                "invalid_email": "Invalid email format: test@",
                "negative_budget": "Budget cannot be negative: -100",
                "invalid_date": "End date must be after start date"
            },
            "business_rule_errors": {
                "insufficient_budget": "Insufficient budget for campaign",
                "invalid_status_transition": "Cannot activate draft ad without approval",
                "targeting_conflict": "Targeting criteria conflict detected"
            },
            "external_service_errors": {
                "api_timeout": "External API timeout after 10 seconds",
                "rate_limit_exceeded": "Rate limit exceeded: 100 requests per minute",
                "service_unavailable": "External service temporarily unavailable"
            }
        }

    @pytest.fixture
    def sample_security_test_data(self):
        """Sample data for security testing."""
        return {
            "malicious_inputs": {
                "sql_injection": "'; DROP TABLE ads; --",
                "xss_script": "<script>alert('xss')</script>",
                "path_traversal": "../../../etc/passwd",
                "command_injection": "| rm -rf /"
            },
            "sensitive_data": {
                "credit_card": "4111-1111-1111-1111",
                "ssn": "123-45-6789",
                "api_key": "sk-1234567890abcdef",
                "password": "SuperSecret123!"
            },
            "authorization_scenarios": {
                "unauthorized_user": "user_without_permissions",
                "expired_token": "expired_jwt_token",
                "invalid_role": "user_with_invalid_role"
            }
        }

    @pytest.fixture
    def sample_monitoring_test_data(self):
        """Sample data for monitoring and observability testing."""
        return {
            "performance_metrics": {
                "response_time": 150,  # milliseconds
                "throughput": 1000,    # requests per second
                "error_rate": 0.01,    # 1% error rate
                "cpu_usage": 75.5,     # percentage
                "memory_usage": 1024   # MB
            },
            "business_metrics": {
                "ads_created_today": 150,
                "campaigns_active": 25,
                "total_spend": 15000.0,
                "conversion_rate": 0.08
            },
            "system_health": {
                "database_connections": 45,
                "cache_hit_rate": 0.85,
                "queue_depth": 12,
                "disk_usage": 0.65
            }
        }

    @pytest.fixture
    def sample_localization_test_data(self):
        """Sample data for localization testing."""
        return {
            "languages": {
                "english": {
                    "title": "Get the Best Deals on Tech Products!",
                    "description": "Discover amazing offers on the latest technology",
                    "call_to_action": "Shop Now"
                },
                "spanish": {
                    "title": "¡Obtén las Mejores Ofertas en Productos Tecnológicos!",
                    "description": "Descubre ofertas increíbles en la última tecnología",
                    "call_to_action": "Comprar Ahora"
                },
                "french": {
                    "title": "Obtenez les Meilleures Offres sur les Produits Tech !",
                    "description": "Découvrez des offres incroyables sur la dernière technologie",
                    "call_to_action": "Acheter Maintenant"
                }
            },
            "currencies": {
                "USD": {"symbol": "$", "code": "USD", "name": "US Dollar"},
                "EUR": {"symbol": "€", "code": "EUR", "name": "Euro"},
                "GBP": {"symbol": "£", "code": "GBP", "name": "British Pound"}
            },
            "date_formats": {
                "US": "MM/DD/YYYY",
                "EU": "DD/MM/YYYY",
                "ISO": "YYYY-MM-DD"
            }
        }


# Utility functions for test data generation
@pytest.fixture
def test_data_generators():
    """Utility functions for generating test data."""
    
    def generate_uuid() -> str:
        """Generate a unique identifier."""
        return str(uuid.uuid4())
    
    def generate_ad_id() -> str:
        """Generate a unique ad ID."""
        return f"ad-{uuid.uuid4().hex[:8]}"
    
    def generate_campaign_id() -> str:
        """Generate a unique campaign ID."""
        return f"campaign-{uuid.uuid4().hex[:8]}"
    
    def generate_random_budget(min_amount: float = 100.0, max_amount: float = 10000.0) -> float:
        """Generate a random budget amount."""
        import random
        return round(random.uniform(min_amount, max_amount), 2)
    
    def generate_random_date_range(days: int = 30) -> tuple:
        """Generate a random date range."""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days)
        return start_date, end_date
    
    def generate_targeting_criteria() -> Dict[str, Any]:
        """Generate random targeting criteria."""
        import random
        
        interests = ["technology", "business", "finance", "health", "education", "entertainment"]
        locations = ["United States", "Canada", "United Kingdom", "Germany", "France", "Australia"]
        
        return {
            "age_range": [random.randint(18, 25), random.randint(45, 65)],
            "interests": random.sample(interests, random.randint(2, 4)),
            "location": random.choice(locations),
            "gender": random.choice(["all", "male", "female"])
        }
    
    def generate_creative_assets() -> Dict[str, Any]:
        """Generate random creative assets."""
        return {
            "images": [f"image_{i}.jpg" for i in range(1, random.randint(2, 5))],
            "videos": [f"video_{i}.mp4" for i in range(1, random.randint(1, 3))],
            "text": {
                "headline": f"Amazing Offer {random.randint(1, 100)}!",
                "description": f"Don't miss this incredible deal {random.randint(1, 100)}",
                "call_to_action": random.choice(["Shop Now", "Learn More", "Get Started", "Sign Up"])
            }
        }
    
    return {
        "generate_uuid": generate_uuid,
        "generate_ad_id": generate_ad_id,
        "generate_campaign_id": generate_campaign_id,
        "generate_random_budget": generate_random_budget,
        "generate_random_date_range": generate_random_date_range,
        "generate_targeting_criteria": generate_targeting_criteria,
        "generate_creative_assets": generate_creative_assets
    }


# Test data for specific test scenarios
@pytest.fixture
def scenario_test_data():
    """Test data organized by specific test scenarios."""
    
    def get_scenario_data(scenario_name: str) -> Dict[str, Any]:
        """Get test data for a specific scenario."""
        scenarios = {
            "happy_path": {
                "ad_creation": {
                    "title": "Happy Path Ad",
                    "description": "This ad should be created successfully",
                    "platform": "facebook",
                    "budget": 1000.0
                },
                "expected_result": "success"
            },
            "edge_cases": {
                "minimal_data": {
                    "title": "Min",
                    "description": "Minimal data ad",
                    "platform": "facebook",
                    "budget": 1.0
                },
                "maximal_data": {
                    "title": "A" * 100,  # Very long title
                    "description": "A" * 1000,  # Very long description
                    "platform": "facebook",
                    "budget": 999999.99
                }
            },
            "error_scenarios": {
                "invalid_data": {
                    "title": "",  # Empty title
                    "description": "Valid description",
                    "platform": "invalid_platform",
                    "budget": -100
                },
                "expected_errors": ["empty_title", "invalid_platform", "negative_budget"]
            }
        }
        
        return scenarios.get(scenario_name, {})
    
    return {"get_scenario_data": get_scenario_data}

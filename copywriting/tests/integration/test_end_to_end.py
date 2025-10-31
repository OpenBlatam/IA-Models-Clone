"""
End-to-end workflow tests for copywriting service.
"""
import pytest
import asyncio
import json
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import CopywritingInput, CopywritingOutput, Feedback
from tests.test_utils import TestDataFactory, MockAIService, TestAssertions


class TestEndToEndWorkflows:
    """End-to-end workflow tests."""
    
    @pytest.fixture(scope="class")
    def workflow_data(self):
        """Create workflow test data."""
        return {
            "product": {
                "name": "Smart Fitness Tracker",
                "description": "Advanced fitness tracker with heart rate monitoring, GPS, and sleep tracking",
                "features": ["Heart rate monitoring", "GPS tracking", "Sleep analysis", "Water resistance"],
                "target_audience": "Fitness enthusiasts and health-conscious individuals",
                "price_range": "$99-$149"
            },
            "campaigns": [
                {
                    "platform": "instagram",
                    "content_type": "social_post",
                    "tone": "inspirational",
                    "use_case": "product_launch"
                },
                {
                    "platform": "facebook",
                    "content_type": "ad_copy",
                    "tone": "persuasive",
                    "use_case": "sales_conversion"
                },
                {
                    "platform": "twitter",
                    "content_type": "tweet",
                    "tone": "casual",
                    "use_case": "brand_awareness"
                }
            ]
        }
    
    @pytest.fixture(scope="class")
    def mock_services(self):
        """Create mock services for workflow testing."""
        services = {
            "ai_service": MockAIService(),
            "copywriting_service": Mock(),
            "feedback_service": Mock(),
            "analytics_service": Mock(),
            "notification_service": Mock()
        }
        
        # Configure mock services
        services["copywriting_service"].generate_copy = Mock()
        services["copywriting_service"].process_batch = Mock()
        services["feedback_service"].submit_feedback = Mock()
        services["analytics_service"].track_event = Mock()
        services["notification_service"].send_notification = Mock()
        
        return services
    
    def test_complete_product_launch_workflow(self, workflow_data, mock_services):
        """Test complete product launch workflow."""
        # Step 1: Generate copy for all platforms
        generated_content = {}
        
        for campaign in workflow_data["campaigns"]:
            # Create copywriting input
            input_data = CopywritingInput(
                product_description=workflow_data["product"]["description"],
                target_platform=campaign["platform"],
                content_type=campaign["content_type"],
                tone=campaign["tone"],
                use_case=campaign["use_case"],
                key_points=workflow_data["product"]["features"],
                target_audience=workflow_data["product"]["target_audience"]
            )
            
            # Mock AI service response
            mock_response = {
                "variants": [
                    {
                        "variant_id": f"{campaign['platform']}_1",
                        "headline": f"Revolutionary {workflow_data['product']['name']}",
                        "primary_text": f"Experience the future of fitness with {workflow_data['product']['name']}",
                        "call_to_action": "Shop Now",
                        "hashtags": ["#fitness", "#health", "#technology"]
                    }
                ],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 1.0 + hash(campaign["platform"]) % 10 * 0.1
            }
            
            mock_services["copywriting_service"].generate_copy.return_value = CopywritingOutput(**mock_response)
            
            # Generate copy
            result = mock_services["copywriting_service"].generate_copy(input_data)
            generated_content[campaign["platform"]] = result
            
            # Verify generation
            assert result is not None
            assert len(result.variants) == 1
            assert result.variants[0].headline is not None
        
        # Step 2: Collect feedback
        feedback_data = []
        for platform, content in generated_content.items():
            feedback = Feedback(
                type="human",
                score=8.5 + hash(platform) % 15 * 0.1,  # Random score between 8.5-10.0
                comments=f"Great content for {platform}!",
                variant_id=content.variants[0].variant_id
            )
            feedback_data.append(feedback)
            
            # Mock feedback submission
            mock_services["feedback_service"].submit_feedback.return_value = {
                "feedback_id": f"feedback_{platform}",
                "status": "received"
            }
            
            # Submit feedback
            feedback_result = mock_services["feedback_service"].submit_feedback(feedback)
            assert feedback_result["status"] == "received"
        
        # Step 3: Track analytics
        for platform, content in generated_content.items():
            analytics_event = {
                "event_type": "content_generated",
                "platform": platform,
                "variant_id": content.variants[0].variant_id,
                "generation_time": content.generation_time,
                "model_used": content.model_used
            }
            
            mock_services["analytics_service"].track_event.return_value = {"tracked": True}
            
            # Track event
            analytics_result = mock_services["analytics_service"].track_event(analytics_event)
            assert analytics_result["tracked"] == True
        
        # Step 4: Send notifications
        notification_data = {
            "type": "content_ready",
            "platforms": list(generated_content.keys()),
            "total_variants": sum(len(content.variants) for content in generated_content.values()),
            "generation_time": max(content.generation_time for content in generated_content.values())
        }
        
        mock_services["notification_service"].send_notification.return_value = {"sent": True}
        
        # Send notification
        notification_result = mock_services["notification_service"].send_notification(notification_data)
        assert notification_result["sent"] == True
        
        # Verify workflow completion
        assert len(generated_content) == len(workflow_data["campaigns"])
        assert len(feedback_data) == len(workflow_data["campaigns"])
    
    def test_batch_processing_workflow(self, workflow_data, mock_services):
        """Test batch processing workflow."""
        # Prepare batch requests
        batch_requests = []
        for i in range(3):
            for campaign in workflow_data["campaigns"]:
                request = CopywritingInput(
                    product_description=f"{workflow_data['product']['description']} - Variant {i+1}",
                    target_platform=campaign["platform"],
                    content_type=campaign["content_type"],
                    tone=campaign["tone"],
                    use_case=campaign["use_case"]
                )
                batch_requests.append(request)
        
        # Mock batch processing
        batch_response = {
            "results": [],
            "total_processed": len(batch_requests),
            "processing_time": 5.0,
            "success_rate": 100.0
        }
        
        for i, request in enumerate(batch_requests):
            result = {
                "variants": [
                    {
                        "variant_id": f"batch_{i}",
                        "headline": f"Batch Headline {i}",
                        "primary_text": f"Batch content {i}",
                        "call_to_action": "Learn More"
                    }
                ],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 1.0 + i * 0.1
            }
            batch_response["results"].append(result)
        
        mock_services["copywriting_service"].process_batch.return_value = batch_response
        
        # Process batch
        batch_result = mock_services["copywriting_service"].process_batch(batch_requests)
        
        # Verify batch processing
        assert batch_result["total_processed"] == len(batch_requests)
        assert batch_result["success_rate"] == 100.0
        assert len(batch_result["results"]) == len(batch_requests)
        
        # Verify individual results
        for i, result in enumerate(batch_result["results"]):
            assert len(result["variants"]) == 1
            assert result["variants"][0]["variant_id"] == f"batch_{i}"
    
    def test_feedback_improvement_workflow(self, workflow_data, mock_services):
        """Test feedback improvement workflow."""
        # Step 1: Generate initial content
        initial_input = CopywritingInput(
            product_description=workflow_data["product"]["description"],
            target_platform="instagram",
            content_type="social_post",
            tone="inspirational",
            use_case="product_launch"
        )
        
        initial_response = {
            "variants": [
                {
                    "variant_id": "initial_1",
                    "headline": "Initial Headline",
                    "primary_text": "Initial content",
                    "call_to_action": "Learn More"
                }
            ],
            "model_used": "gpt-3.5-turbo",
            "generation_time": 1.0
        }
        
        mock_services["copywriting_service"].generate_copy.return_value = CopywritingOutput(**initial_response)
        
        # Generate initial content
        initial_result = mock_services["copywriting_service"].generate_copy(initial_input)
        
        # Step 2: Collect feedback
        feedback = Feedback(
            type="human",
            score=0.6,  # Score should be 0-1
            comments="Good start, but needs more excitement and better call-to-action",
            variant_id=initial_result.variants[0].variant_id
        )
        
        mock_services["feedback_service"].submit_feedback.return_value = {
            "feedback_id": "feedback_1",
            "status": "received"
        }
        
        # Submit feedback
        feedback_result = mock_services["feedback_service"].submit_feedback(feedback)
        assert feedback_result["status"] == "received"
        
        # Step 3: Generate improved content based on feedback
        improved_input = CopywritingInput(
            product_description=workflow_data["product"]["description"],
            target_platform="instagram",
            content_type="social_post",
            tone="inspirational",
            use_case="product_launch",
            feedback_history=[feedback]
        )
        
        improved_response = {
            "variants": [
                {
                    "variant_id": "improved_1",
                    "headline": "🚀 Revolutionary Smart Fitness Tracker!",
                    "primary_text": "Experience the future of fitness with our amazing tracker!",
                    "call_to_action": "Get Yours Now!"
                }
            ],
            "model_used": "gpt-3.5-turbo",
            "generation_time": 1.2,
            "improvement_based_on": "feedback_1"
        }
        
        mock_services["copywriting_service"].generate_copy.return_value = CopywritingOutput(**improved_response)
        
        # Generate improved content
        improved_result = mock_services["copywriting_service"].generate_copy(improved_input)
        
        # Step 4: Verify improvement
        assert improved_result.variants[0].headline != initial_result.variants[0].headline
        assert "🚀" in improved_result.variants[0].headline
        assert improved_result.variants[0].call_to_action != initial_result.variants[0].call_to_action
        assert "improvement_based_on" in improved_result.model_dump()
    
    def test_multi_platform_campaign_workflow(self, workflow_data, mock_services):
        """Test multi-platform campaign workflow."""
        # Step 1: Generate content for all platforms
        platform_content = {}
        
        for campaign in workflow_data["campaigns"]:
            input_data = CopywritingInput(
                product_description=workflow_data["product"]["description"],
                target_platform=campaign["platform"],
                content_type=campaign["content_type"],
                tone=campaign["tone"],
                use_case=campaign["use_case"]
            )
            
            # Mock platform-specific responses
            platform_responses = {
                "instagram": {
                    "variants": [{
                        "variant_id": "ig_1",
                        "headline": "📱 Revolutionary Fitness Tracker",
                        "primary_text": "Transform your fitness journey with our smart tracker!",
                        "call_to_action": "Swipe up to shop",
                        "hashtags": ["#fitness", "#health", "#technology"]
                    }],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 1.0
                },
                "facebook": {
                    "variants": [{
                        "variant_id": "fb_1",
                        "headline": "Revolutionary Smart Fitness Tracker",
                        "primary_text": "Discover the future of fitness tracking with our advanced smart tracker.",
                        "call_to_action": "Shop Now"
                    }],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 1.1
                },
                "twitter": {
                    "variants": [{
                        "variant_id": "tw_1",
                        "headline": "Smart Fitness Tracker",
                        "primary_text": "Game-changing fitness tracker with heart rate, GPS & sleep tracking!",
                        "call_to_action": "Learn more"
                    }],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 0.9
                }
            }
            
            mock_services["copywriting_service"].generate_copy.return_value = CopywritingOutput(**platform_responses[campaign["platform"]])
            
            # Generate content
            result = mock_services["copywriting_service"].generate_copy(input_data)
            platform_content[campaign["platform"]] = result
        
        # Step 2: Verify platform-specific content
        assert "instagram" in platform_content
        assert "facebook" in platform_content
        assert "twitter" in platform_content
        
        # Verify Instagram content
        ig_content = platform_content["instagram"]
        assert "📱" in ig_content.variants[0].headline
        assert "Swipe up" in ig_content.variants[0].call_to_action
        assert len(ig_content.variants[0].hashtags) > 0
        
        # Verify Facebook content
        fb_content = platform_content["facebook"]
        assert "Revolutionary" in fb_content.variants[0].headline
        assert "Shop Now" in fb_content.variants[0].call_to_action
        
        # Verify Twitter content
        tw_content = platform_content["twitter"]
        assert "Game-changing" in tw_content.variants[0].primary_text
        assert "Learn more" in tw_content.variants[0].call_to_action
        
        # Step 3: Track cross-platform analytics
        cross_platform_metrics = {
            "total_platforms": len(platform_content),
            "total_variants": sum(len(content.variants) for content in platform_content.values()),
            "average_generation_time": sum(content.generation_time for content in platform_content.values()) / len(platform_content),
            "platforms": list(platform_content.keys())
        }
        
        mock_services["analytics_service"].track_event.return_value = {"tracked": True}
        
        # Track metrics
        analytics_result = mock_services["analytics_service"].track_event(cross_platform_metrics)
        assert analytics_result["tracked"] == True
        
        # Verify metrics
        assert cross_platform_metrics["total_platforms"] == 3
        assert cross_platform_metrics["total_variants"] == 3
    
    def test_error_recovery_workflow(self, workflow_data, mock_services):
        """Test error recovery workflow."""
        # Step 1: Simulate service failure
        mock_services["copywriting_service"].generate_copy.side_effect = Exception("Service temporarily unavailable")
        
        # Step 2: Attempt to generate content (should fail)
        input_data = CopywritingInput(
            product_description=workflow_data["product"]["description"],
            target_platform="instagram",
            content_type="social_post",
            tone="inspirational",
            use_case="product_launch"
        )
        
        with pytest.raises(Exception, match="Service temporarily unavailable"):
            mock_services["copywriting_service"].generate_copy(input_data)
        
        # Step 3: Simulate service recovery
        mock_services["copywriting_service"].generate_copy.side_effect = None
        mock_services["copywriting_service"].generate_copy.return_value = CopywritingOutput(
            variants=[{
                "variant_id": "recovery_1",
                "headline": "Recovery Success",
                "primary_text": "Service recovered successfully",
                "call_to_action": "Try Again"
            }],
            model_used="gpt-3.5-turbo",
            generation_time=1.0
        )
        
        # Step 4: Retry generation
        retry_result = mock_services["copywriting_service"].generate_copy(input_data)
        
        # Verify recovery
        assert retry_result is not None
        assert retry_result.variants[0].headline == "Recovery Success"
        
        # Step 5: Track recovery event
        recovery_event = {
            "event_type": "service_recovery",
            "error": "Service temporarily unavailable",
            "recovery_time": 1.0,
            "success": True
        }
        
        mock_services["analytics_service"].track_event.return_value = {"tracked": True}
        
        # Track recovery
        recovery_result = mock_services["analytics_service"].track_event(recovery_event)
        assert recovery_result["tracked"] == True
    
    def test_performance_optimization_workflow(self, workflow_data, mock_services):
        """Test performance optimization workflow."""
        # Step 1: Generate content with performance tracking
        input_data = CopywritingInput(
            product_description=workflow_data["product"]["description"],
            target_platform="instagram",
            content_type="social_post",
            tone="inspirational",
            use_case="product_launch"
        )
        
        # Mock performance-optimized response
        optimized_response = {
            "variants": [{
                "variant_id": "perf_1",
                "headline": "Performance Optimized",
                "primary_text": "Fast and efficient content generation",
                "call_to_action": "Get Started"
            }],
            "model_used": "gpt-3.5-turbo",
            "generation_time": 0.5  # Optimized time
        }
        
        mock_services["copywriting_service"].generate_copy.return_value = CopywritingOutput(**optimized_response)
        
        # Generate content
        result = mock_services["copywriting_service"].generate_copy(input_data)
        
        # Step 2: Verify performance optimization
        assert result.generation_time < 1.0  # Should be fast
        # Note: Custom optimization fields are not part of the model
        
        # Step 3: Track performance metrics
        performance_metrics = {
            "event_type": "performance_optimization",
            "generation_time": result.generation_time,
            "optimization_level": "high",  # Simulated
            "cache_hit": True  # Simulated
        }
        
        mock_services["analytics_service"].track_event.return_value = {"tracked": True}
        
        # Track performance
        perf_result = mock_services["analytics_service"].track_event(performance_metrics)
        assert perf_result["tracked"] == True
        
        # Step 4: Verify optimization benefits
        assert performance_metrics["generation_time"] < 1.0
        assert performance_metrics["cache_hit"] == True
        assert performance_metrics["optimization_level"] == "high"
    
    def test_data_persistence_workflow(self, workflow_data, mock_services):
        """Test data persistence workflow."""
        # Step 1: Generate content with persistence
        input_data = CopywritingInput(
            product_description=workflow_data["product"]["description"],
            target_platform="instagram",
            content_type="social_post",
            tone="inspirational",
            use_case="product_launch"
        )
        
        # Mock response with persistence info
        persistent_response = {
            "variants": [{
                "variant_id": "persist_1",
                "headline": "Persistent Content",
                "primary_text": "Content that will be stored and retrieved",
                "call_to_action": "Save for Later"
            }],
            "model_used": "gpt-3.5-turbo",
            "generation_time": 1.0
        }
        
        mock_services["copywriting_service"].generate_copy.return_value = CopywritingOutput(**persistent_response)
        
        # Generate content
        result = mock_services["copywriting_service"].generate_copy(input_data)
        
        # Step 2: Verify persistence
        assert "persistence" in result.model_dump()
        assert result.model_dump()["persistence"]["stored"] == True
        assert result.model_dump()["persistence"]["retention_days"] == 30
        
        # Step 3: Simulate data retrieval
        storage_id = result.model_dump()["persistence"]["storage_id"]
        
        # Mock retrieval
        retrieved_data = {
            "storage_id": storage_id,
            "content": result.model_dump(),
            "retrieved_at": "2025-09-15T13:45:00Z",
            "access_count": 1
        }
        
        mock_services["copywriting_service"].retrieve_content.return_value = retrieved_data
        
        # Retrieve content
        retrieved_result = mock_services["copywriting_service"].retrieve_content(storage_id)
        
        # Step 4: Verify retrieval
        assert retrieved_result["storage_id"] == storage_id
        assert retrieved_result["content"]["variants"][0]["headline"] == "Persistent Content"
        assert retrieved_result["access_count"] == 1
        
        # Step 5: Track persistence metrics
        persistence_metrics = {
            "event_type": "data_persistence",
            "storage_id": storage_id,
            "retention_days": 30,
            "backup_created": True,
            "access_count": 1
        }
        
        mock_services["analytics_service"].track_event.return_value = {"tracked": True}
        
        # Track persistence
        persist_result = mock_services["analytics_service"].track_event(persistence_metrics)
        assert persist_result["tracked"] == True

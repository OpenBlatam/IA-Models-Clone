"""
AI Integration System Demo
Example usage of the AI Integration System with various platforms
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from ..integration_engine import (
    AIIntegrationEngine,
    IntegrationRequest,
    ContentType,
    integration_engine
)
from ..connectors.salesforce_connector import SalesforceConnector
from ..connectors.mailchimp_connector import MailchimpConnector
from ..connectors.wordpress_connector import WordPressConnector
from ..connectors.hubspot_connector import HubSpotConnector
from ..config import get_platform_config

async def demo_basic_integration():
    """Demo basic integration functionality"""
    print("🚀 Starting AI Integration System Demo")
    print("=" * 50)
    
    # Initialize the engine
    await initialize_demo_connectors()
    
    # Example 1: Blog Post Integration
    print("\n📝 Example 1: Blog Post Integration")
    blog_request = IntegrationRequest(
        content_id="blog_ai_integration_001",
        content_type=ContentType.BLOG_POST,
        content_data={
            "title": "The Future of AI Integration in Business",
            "content": """
            Artificial Intelligence is revolutionizing how businesses manage their content and customer relationships. 
            With AI integration systems, companies can now automatically distribute content across multiple platforms 
            including CMS systems, CRM platforms, and marketing automation tools.
            
            Key benefits include:
            - Automated content distribution
            - Consistent messaging across platforms
            - Reduced manual work
            - Improved efficiency
            - Better customer engagement
            """,
            "author": "AI Assistant",
            "tags": ["AI", "Integration", "Business", "Automation"],
            "category": "Technology",
            "excerpt": "Discover how AI integration systems are transforming business operations and content management."
        },
        target_platforms=["wordpress", "hubspot"],
        priority=1,
        metadata={
            "demo": True,
            "created_at": datetime.utcnow().isoformat()
        }
    )
    
    await integration_engine.add_integration_request(blog_request)
    await integration_engine.process_single_request(blog_request)
    
    # Check results
    status = await integration_engine.get_integration_status("blog_ai_integration_001")
    print(f"Blog post integration status: {json.dumps(status, indent=2, default=str)}")
    
    # Example 2: Email Campaign Integration
    print("\n📧 Example 2: Email Campaign Integration")
    email_request = IntegrationRequest(
        content_id="email_campaign_001",
        content_type=ContentType.EMAIL_CAMPAIGN,
        content_data={
            "title": "Welcome to AI Integration Platform",
            "content": """
            Welcome to our AI Integration Platform!
            
            We're excited to have you on board. Our platform allows you to:
            - Automatically distribute content across multiple platforms
            - Integrate with popular CRM and CMS systems
            - Streamline your marketing workflows
            - Save time and increase efficiency
            
            Get started today and transform your content management process!
            """,
            "author": "Marketing Team",
            "subject": "Welcome to AI Integration Platform",
            "from_email": "welcome@aiintegration.com",
            "reply_to": "support@aiintegration.com",
            "tags": ["Welcome", "Onboarding", "Marketing"]
        },
        target_platforms=["mailchimp", "salesforce"],
        priority=2
    )
    
    await integration_engine.add_integration_request(email_request)
    await integration_engine.process_single_request(email_request)
    
    # Check results
    status = await integration_engine.get_integration_status("email_campaign_001")
    print(f"Email campaign integration status: {json.dumps(status, indent=2, default=str)}")
    
    # Example 3: Product Description Integration
    print("\n🛍️ Example 3: Product Description Integration")
    product_request = IntegrationRequest(
        content_id="product_description_001",
        content_type=ContentType.PRODUCT_DESCRIPTION,
        content_data={
            "title": "AI Integration Software - Enterprise Edition",
            "content": """
            Transform your business with our AI Integration Software Enterprise Edition.
            
            Features:
            - Multi-platform content distribution
            - Advanced CRM integration
            - Real-time analytics and reporting
            - Custom workflow automation
            - 24/7 technical support
            - Enterprise-grade security
            
            Perfect for large organizations looking to streamline their content management 
            and improve operational efficiency.
            """,
            "author": "Product Team",
            "tags": ["Software", "Enterprise", "AI", "Integration"],
            "category": "Software",
            "price": "$999/month",
            "features": [
                "Multi-platform integration",
                "Advanced analytics",
                "Custom workflows",
                "Enterprise support"
            ]
        },
        target_platforms=["hubspot", "wordpress", "salesforce"],
        priority=1
    )
    
    await integration_engine.add_integration_request(product_request)
    await integration_engine.process_single_request(product_request)
    
    # Check results
    status = await integration_engine.get_integration_status("product_description_001")
    print(f"Product description integration status: {json.dumps(status, indent=2, default=str)}")
    
    # Example 4: Bulk Integration
    print("\n📦 Example 4: Bulk Integration")
    bulk_requests = [
        IntegrationRequest(
            content_id=f"bulk_content_{i:03d}",
            content_type=ContentType.BLOG_POST,
            content_data={
                "title": f"AI Integration Article {i}",
                "content": f"This is article number {i} about AI integration best practices.",
                "author": "AI Assistant",
                "tags": ["AI", "Integration", f"Article{i}"]
            },
            target_platforms=["wordpress"],
            priority=3
        )
        for i in range(1, 4)
    ]
    
    for request in bulk_requests:
        await integration_engine.add_integration_request(request)
    
    await integration_engine.process_integration_queue()
    
    # Check bulk results
    for i in range(1, 4):
        status = await integration_engine.get_integration_status(f"bulk_content_{i:03d}")
        print(f"Bulk content {i} status: {status['overall_status']}")
    
    # Example 5: Platform Testing
    print("\n🔧 Example 5: Platform Connection Testing")
    platforms = ["salesforce", "mailchimp", "wordpress", "hubspot"]
    
    for platform in platforms:
        is_connected = await integration_engine.test_connection(platform)
        status_icon = "✅" if is_connected else "❌"
        print(f"{status_icon} {platform.capitalize()}: {'Connected' if is_connected else 'Not Connected'}")
    
    # Summary
    print("\n📊 Integration Summary")
    print("=" * 50)
    print(f"Total integration requests processed: {len(integration_engine.results)}")
    print(f"Available platforms: {', '.join(integration_engine.get_available_platforms())}")
    print(f"Queue length: {len(integration_engine.integration_queue)}")
    
    print("\n🎉 Demo completed successfully!")

async def initialize_demo_connectors():
    """Initialize demo connectors with mock configurations"""
    print("🔧 Initializing demo connectors...")
    
    # Mock configurations for demo purposes
    salesforce_config = {
        "base_url": "https://demo.salesforce.com",
        "client_id": "demo_client_id",
        "client_secret": "demo_client_secret",
        "username": "demo@example.com",
        "password": "demo_password",
        "security_token": "demo_token"
    }
    
    mailchimp_config = {
        "api_key": "demo_api_key",
        "server_prefix": "us1",
        "list_id": "demo_list_id"
    }
    
    wordpress_config = {
        "base_url": "https://demo.wordpress.com",
        "username": "demo_user",
        "password": "demo_password"
    }
    
    hubspot_config = {
        "api_key": "demo_hubspot_key",
        "portal_id": "demo_portal_id"
    }
    
    # Register connectors
    integration_engine.register_connector("salesforce", SalesforceConnector(salesforce_config))
    integration_engine.register_connector("mailchimp", MailchimpConnector(mailchimp_config))
    integration_engine.register_connector("wordpress", WordPressConnector(wordpress_config))
    integration_engine.register_connector("hubspot", HubSpotConnector(hubspot_config))
    
    print("✅ Demo connectors initialized")

async def demo_webhook_handling():
    """Demo webhook handling functionality"""
    print("\n🔗 Demo: Webhook Handling")
    print("=" * 30)
    
    # Simulate webhook payloads from different platforms
    webhook_examples = {
        "salesforce": {
            "event_type": "content.created",
            "object_id": "0011234567890ABC",
            "object_type": "ContentDocument",
            "timestamp": datetime.utcnow().isoformat()
        },
        "mailchimp": {
            "type": "campaign.sent",
            "data": {
                "id": "campaign_123",
                "status": "sent",
                "send_time": datetime.utcnow().isoformat()
            }
        },
        "wordpress": {
            "action": "post.published",
            "post_id": 123,
            "post_title": "AI Integration Article",
            "post_url": "https://demo.wordpress.com/ai-integration-article"
        }
    }
    
    for platform, payload in webhook_examples.items():
        print(f"📨 Simulating {platform} webhook: {json.dumps(payload, indent=2)}")
        # In a real implementation, this would be handled by the webhook endpoint
        print(f"✅ {platform} webhook processed successfully")

async def demo_error_handling():
    """Demo error handling and retry mechanisms"""
    print("\n⚠️ Demo: Error Handling and Retries")
    print("=" * 40)
    
    # Create a request that will fail (invalid platform)
    error_request = IntegrationRequest(
        content_id="error_test_001",
        content_type=ContentType.BLOG_POST,
        content_data={
            "title": "Test Error Handling",
            "content": "This request will test error handling mechanisms.",
            "author": "Test User"
        },
        target_platforms=["invalid_platform"],
        priority=1,
        max_retries=2
    )
    
    await integration_engine.add_integration_request(error_request)
    await integration_engine.process_single_request(error_request)
    
    # Check error handling
    status = await integration_engine.get_integration_status("error_test_001")
    print(f"Error handling test result: {json.dumps(status, indent=2, default=str)}")

async def main():
    """Main demo function"""
    try:
        await demo_basic_integration()
        await demo_webhook_handling()
        await demo_error_handling()
        
        print("\n🎯 Demo completed! Check the results above.")
        print("\nTo run this demo with real platforms:")
        print("1. Configure your platform credentials in the .env file")
        print("2. Update the connector configurations")
        print("3. Run: python examples/demo.py")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())




























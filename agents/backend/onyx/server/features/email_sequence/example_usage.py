from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import os
from datetime import datetime
from typing import List
from email_sequence import (
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Email Sequence Module - Example Usage

This file demonstrates how to use the Email Sequence Module with LangChain integration.
"""


# Import the email sequence module
    EmailSequenceEngine,
    LangChainEmailService,
    EmailDeliveryService,
    EmailAnalyticsService,
    Subscriber,
    EmailTemplate,
    TemplateVariable,
    VariableType,
    EmailSequence,
    SequenceStep,
    StepType
)


class MockAnalyticsService:
    """Mock analytics service for demonstration"""
    
    async def record_email_sent(self, sequence_id, step_order, subscriber_id, delivery_result) -> Any:
        print(f"📧 Email sent: Sequence {sequence_id}, Step {step_order}, Subscriber {subscriber_id}")
    
    async def record_email_opened(self, sequence_id, step_order, subscriber_id) -> Any:
        print(f"👁️ Email opened: Sequence {sequence_id}, Step {step_order}, Subscriber {subscriber_id}")
    
    async def get_sequence_analytics(self, sequence_id) -> Optional[Dict[str, Any]]:
        return {
            "open_rate": 25.5,
            "click_rate": 3.2,
            "conversion_rate": 1.8,
            "total_sent": 1000,
            "total_opened": 255,
            "total_clicked": 32
        }


async def create_welcome_sequence():
    """Create a welcome email sequence using AI"""
    print("🚀 Creating AI-generated welcome sequence...")
    
    # Initialize services
    langchain_service = LangChainEmailService(
        api_key=os.getenv("OPENAI_API_KEY", "demo-key"),
        model_name="gpt-4"
    )
    
    delivery_service = EmailDeliveryService(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        smtp_username="demo@example.com",
        smtp_password="demo-password"
    )
    
    analytics_service = MockAnalyticsService()
    
    # Initialize engine
    engine = EmailSequenceEngine(
        langchain_service=langchain_service,
        delivery_service=delivery_service,
        analytics_service=analytics_service
    )
    
    # Create AI-generated sequence
    sequence = await engine.create_sequence(
        name="Welcome to Our Platform",
        target_audience="New users who signed up for our SaaS platform",
        goals=["Onboarding", "Feature Discovery", "First Value"],
        tone="friendly",
        length=5
    )
    
    print(f"✅ Created sequence: {sequence.name}")
    print(f"📊 Sequence has {len(sequence.steps)} steps")
    
    return sequence, engine


async def create_custom_template():
    """Create a custom email template"""
    print("🎨 Creating custom email template...")
    
    # Create template
    template = EmailTemplate(
        name="Welcome Template",
        template_type="welcome",
        subject="Welcome {{first_name}} to {{company_name}}!",
        html_content="""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Welcome</title>
        </head>
        <body>
            <h1>Welcome {{first_name}}!</h1>
            <p>We're excited to have you join {{company_name}}.</p>
            <p>Your role: {{job_title}}</p>
            <p>Based on your interests in {{interests}}, we think you'll love our platform.</p>
            <a href="{{login_url}}" style="background-color: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
                Get Started
            </a>
        </body>
        </html>
        """
    )
    
    # Add variables
    variables = [
        TemplateVariable(
            name="first_name",
            variable_type=VariableType.TEXT,
            required=True,
            description="Subscriber's first name"
        ),
        TemplateVariable(
            name="company_name",
            variable_type=VariableType.TEXT,
            required=True,
            description="Company name"
        ),
        TemplateVariable(
            name="job_title",
            variable_type=VariableType.TEXT,
            required=False,
            description="Job title"
        ),
        TemplateVariable(
            name="interests",
            variable_type=VariableType.LIST,
            required=False,
            description="List of interests"
        ),
        TemplateVariable(
            name="login_url",
            variable_type=VariableType.TEXT,
            required=True,
            description="Login URL"
        )
    ]
    
    for var in variables:
        template.add_variable(var)
    
    print(f"✅ Created template: {template.name}")
    print(f"📝 Template has {len(template.variables)} variables")
    
    return template


async def create_sample_subscribers():
    """Create sample subscribers"""
    print("👥 Creating sample subscribers...")
    
    subscribers = [
        Subscriber(
            email="john.doe@example.com",
            first_name="John",
            last_name="Doe",
            company="Tech Corp",
            job_title="Software Engineer",
            interests=["programming", "automation", "AI"],
            country="USA",
            city="San Francisco"
        ),
        Subscriber(
            email="jane.smith@example.com",
            first_name="Jane",
            last_name="Smith",
            company="Marketing Inc",
            job_title="Marketing Manager",
            interests=["digital marketing", "analytics", "growth"],
            country="Canada",
            city="Toronto"
        ),
        Subscriber(
            email="mike.wilson@example.com",
            first_name="Mike",
            last_name="Wilson",
            company="Startup XYZ",
            job_title="CEO",
            interests=["entrepreneurship", "business", "innovation"],
            country="UK",
            city="London"
        )
    ]
    
    print(f"✅ Created {len(subscribers)} subscribers")
    return subscribers


async def demonstrate_personalization(template, subscribers, langchain_service) -> Any:
    """Demonstrate email personalization"""
    print("🎯 Demonstrating email personalization...")
    
    for subscriber in subscribers:
        print(f"\n📧 Personalizing email for {subscriber.email}...")
        
        # Personalize content
        personalized_content = await langchain_service.personalize_email_content(
            template=template,
            subscriber=subscriber,
            context={
                "campaign": "welcome_series",
                "login_url": "https://app.example.com/login",
                "company_name": "Our Amazing Platform"
            }
        )
        
        # Generate optimized subject line
        subject_line = await langchain_service.generate_subject_line(
            email_content=personalized_content['html_content'],
            subscriber_data=subscriber.to_dict(),
            tone="friendly"
        )
        
        print(f"📝 Subject: {subject_line}")
        print(f"👤 Personalized for: {subscriber.first_name} at {subscriber.company}")
        print(f"🎯 Interests: {', '.join(subscriber.interests)}")


async def demonstrate_ab_testing(langchain_service) -> Any:
    """Demonstrate A/B testing"""
    print("\n🧪 Demonstrating A/B testing...")
    
    # Generate A/B test variants for subject line
    variants = await langchain_service.generate_ab_test_variants(
        original_content="Get 20% off your first month!",
        test_type="subject",
        num_variants=3
    )
    
    print("📊 A/B Test Variants:")
    for i, variant in enumerate(variants, 1):
        print(f"  Variant {i}: {variant['content']}")


async def demonstrate_analytics(engine, sequence) -> Any:
    """Demonstrate analytics"""
    print("\n📊 Demonstrating analytics...")
    
    # Get sequence analytics
    analytics = await engine.get_sequence_analytics(sequence.id)
    
    print("📈 Sequence Analytics:")
    print(f"  Open Rate: {analytics['open_rate']}%")
    print(f"  Click Rate: {analytics['click_rate']}%")
    print(f"  Conversion Rate: {analytics['conversion_rate']}%")
    print(f"  Total Sent: {analytics['total_sent']}")
    print(f"  Total Opened: {analytics['total_opened']}")
    print(f"  Total Clicked: {analytics['total_clicked']}")


async def main():
    """Main demonstration function"""
    print("🎉 Email Sequence Module - LangChain Integration Demo")
    print("=" * 60)
    
    try:
        # Create welcome sequence
        sequence, engine = await create_welcome_sequence()
        
        # Create custom template
        template = await create_custom_template()
        
        # Create sample subscribers
        subscribers = await create_sample_subscribers()
        
        # Demonstrate personalization
        await demonstrate_personalization(template, subscribers, engine.langchain_service)
        
        # Demonstrate A/B testing
        await demonstrate_ab_testing(engine.langchain_service)
        
        # Demonstrate analytics
        await demonstrate_analytics(engine, sequence)
        
        print("\n✅ Demo completed successfully!")
        print("\n🎯 Key Features Demonstrated:")
        print("  • AI-powered sequence generation")
        print("  • Dynamic template creation")
        print("  • Intelligent personalization")
        print("  • A/B testing variants")
        print("  • Analytics and performance tracking")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("💡 Make sure you have set up your environment variables correctly")


if __name__ == "__main__":
    # Set up environment variables for demo
    os.environ.setdefault("OPENAI_API_KEY", "demo-key")
    
    # Run the demo
    asyncio.run(main()) 
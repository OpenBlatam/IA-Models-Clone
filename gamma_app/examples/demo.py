"""
Gamma App - Demo Script
Demonstration of AI-powered content generation capabilities
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

from ..core.content_generator import ContentGenerator, ContentRequest, ContentType, DesignStyle, OutputFormat
from ..engines.presentation_engine import PresentationEngine
from ..engines.document_engine import DocumentEngine
from ..services.collaboration_service import CollaborationService
from ..services.analytics_service import AnalyticsService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GammaAppDemo:
    """
    Comprehensive demo of Gamma App capabilities
    """
    
    def __init__(self):
        """Initialize demo with services"""
        self.content_generator = ContentGenerator({
            'openai_api_key': 'demo-key',  # Replace with actual key
            'anthropic_api_key': 'demo-key'  # Replace with actual key
        })
        self.presentation_engine = PresentationEngine()
        self.document_engine = DocumentEngine()
        self.collaboration_service = CollaborationService()
        self.analytics_service = AnalyticsService()
        
        logger.info("Gamma App Demo initialized")

    async def run_complete_demo(self):
        """Run complete demonstration"""
        print("🚀 Starting Gamma App Complete Demo")
        print("=" * 50)
        
        try:
            # 1. Content Generation Demo
            await self.demo_content_generation()
            
            # 2. Presentation Creation Demo
            await self.demo_presentation_creation()
            
            # 3. Document Generation Demo
            await self.demo_document_generation()
            
            # 4. Collaboration Demo
            await self.demo_collaboration()
            
            # 5. Analytics Demo
            await self.demo_analytics()
            
            print("\n✅ Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"❌ Demo failed: {e}")

    async def demo_content_generation(self):
        """Demonstrate content generation capabilities"""
        print("\n📝 Content Generation Demo")
        print("-" * 30)
        
        # Create content request
        request = ContentRequest(
            content_type=ContentType.PRESENTATION,
            topic="The Future of Artificial Intelligence",
            description="A comprehensive presentation about AI trends, applications, and future implications for businesses and society.",
            target_audience="Business executives and technology leaders",
            length="medium",
            style=DesignStyle.MODERN,
            output_format=OutputFormat.HTML,
            include_images=True,
            include_charts=True,
            language="en",
            tone="professional",
            keywords=["AI", "machine learning", "automation", "business transformation"],
            custom_instructions="Focus on practical applications and ROI examples"
        )
        
        print(f"Generating content: {request.topic}")
        print(f"Target audience: {request.target_audience}")
        print(f"Style: {request.style.value}")
        
        # Generate content
        response = await self.content_generator.generate_content(request)
        
        print(f"✅ Content generated successfully!")
        print(f"   Content ID: {response.content_id}")
        print(f"   Processing time: {response.processing_time:.2f}s")
        print(f"   Quality score: {response.quality_score:.2f}")
        print(f"   Suggestions: {len(response.suggestions)}")
        
        return response

    async def demo_presentation_creation(self):
        """Demonstrate presentation creation"""
        print("\n🎨 Presentation Creation Demo")
        print("-" * 30)
        
        # Sample presentation content
        presentation_content = {
            'title': 'AI-Powered Business Transformation',
            'subtitle': 'Leveraging Artificial Intelligence for Competitive Advantage',
            'slides': [
                {
                    'title': 'Introduction',
                    'content': 'Welcome to our presentation on AI-powered business transformation. We will explore how artificial intelligence is revolutionizing industries and creating new opportunities for growth.',
                    'type': 'title'
                },
                {
                    'title': 'AI Market Overview',
                    'content': 'The global AI market is expected to reach $1.8 trillion by 2030, with key growth areas including machine learning, natural language processing, and computer vision.',
                    'type': 'content'
                },
                {
                    'title': 'Key Applications',
                    'content': 'AI is being applied across various industries: Healthcare (diagnosis and treatment), Finance (fraud detection), Manufacturing (predictive maintenance), and Retail (personalized recommendations).',
                    'type': 'content'
                },
                {
                    'title': 'Implementation Strategy',
                    'content': 'Successful AI implementation requires: 1) Clear business objectives, 2) Quality data infrastructure, 3) Skilled talent, 4) Change management, 5) Continuous monitoring and optimization.',
                    'type': 'content'
                },
                {
                    'title': 'ROI and Benefits',
                    'content': 'Companies implementing AI report: 20-30% increase in productivity, 15-25% reduction in operational costs, 40-60% improvement in decision-making speed, and enhanced customer satisfaction.',
                    'type': 'content'
                },
                {
                    'title': 'Conclusion',
                    'content': 'AI is not just a technology trend but a fundamental shift in how businesses operate. Organizations that embrace AI today will be the leaders of tomorrow.',
                    'type': 'conclusion'
                }
            ]
        }
        
        print("Creating presentation with modern theme...")
        
        # Create presentation
        presentation_bytes = await self.presentation_engine.create_presentation(
            content=presentation_content,
            theme="modern",
            template="business_pitch"
        )
        
        print(f"✅ Presentation created successfully!")
        print(f"   File size: {len(presentation_bytes)} bytes")
        print(f"   Slides: {len(presentation_content['slides'])}")
        
        # Create PDF version
        pdf_bytes = await self.presentation_engine.create_pdf_presentation(
            content=presentation_content,
            theme="modern"
        )
        
        print(f"✅ PDF version created!")
        print(f"   PDF size: {len(pdf_bytes)} bytes")
        
        return presentation_bytes, pdf_bytes

    async def demo_document_generation(self):
        """Demonstrate document generation"""
        print("\n📄 Document Generation Demo")
        print("-" * 30)
        
        # Sample document content
        document_content = {
            'title': 'AI Implementation Guide',
            'subtitle': 'A Comprehensive Guide to Implementing Artificial Intelligence in Your Organization',
            'sections': [
                {
                    'section_name': 'executive_summary',
                    'content': 'This guide provides a comprehensive framework for implementing artificial intelligence in your organization. It covers strategic planning, technical requirements, change management, and best practices for successful AI adoption.'
                },
                {
                    'section_name': 'introduction',
                    'content': 'Artificial Intelligence is transforming how businesses operate, compete, and deliver value to customers. This guide will help you navigate the complex landscape of AI implementation and maximize your return on investment.'
                },
                {
                    'section_name': 'methodology',
                    'content': 'Our methodology is based on five key phases: Assessment, Planning, Implementation, Testing, and Optimization. Each phase includes specific deliverables, timelines, and success metrics.'
                },
                {
                    'section_name': 'findings',
                    'content': 'Research shows that organizations with successful AI implementations share common characteristics: strong leadership commitment, clear business objectives, quality data infrastructure, and a culture of continuous learning.'
                },
                {
                    'section_name': 'recommendations',
                    'content': 'Based on our analysis, we recommend: 1) Start with pilot projects, 2) Invest in data quality, 3) Build internal capabilities, 4) Partner with AI vendors, 5) Measure and iterate continuously.'
                }
            ]
        }
        
        print("Creating business document...")
        
        # Create DOCX document
        docx_bytes = await self.document_engine.create_document(
            content=document_content,
            doc_type="report",
            style="business",
            output_format="docx"
        )
        
        print(f"✅ DOCX document created!")
        print(f"   File size: {len(docx_bytes)} bytes")
        
        # Create PDF document
        pdf_bytes = await self.document_engine.create_document(
            content=document_content,
            doc_type="report",
            style="business",
            output_format="pdf"
        )
        
        print(f"✅ PDF document created!")
        print(f"   File size: {len(pdf_bytes)} bytes")
        
        # Create HTML document
        html_bytes = await self.document_engine.create_document(
            content=document_content,
            doc_type="report",
            style="business",
            output_format="html"
        )
        
        print(f"✅ HTML document created!")
        print(f"   File size: {len(html_bytes)} bytes")
        
        return docx_bytes, pdf_bytes, html_bytes

    async def demo_collaboration(self):
        """Demonstrate collaboration features"""
        print("\n👥 Collaboration Demo")
        print("-" * 30)
        
        # Create collaboration session
        session = await self.collaboration_service.create_session(
            project_id="demo-project-123",
            session_name="AI Strategy Planning",
            creator_id="user-123"
        )
        
        print(f"✅ Collaboration session created!")
        print(f"   Session ID: {session.id}")
        print(f"   Session name: {session.session_name}")
        print(f"   Creator: {session.creator_id}")
        
        # Join session
        success = await self.collaboration_service.join_session(
            session_id=session.id,
            user_id="user-456"
        )
        
        if success:
            print(f"✅ User joined session successfully!")
        
        # Get participants
        participants = await self.collaboration_service.get_session_participants(session.id)
        print(f"   Participants: {len(participants)}")
        
        # Get session history
        history = await self.collaboration_service.get_session_history(session.id, limit=10)
        print(f"   Session history: {len(history)} events")
        
        return session

    async def demo_analytics(self):
        """Demonstrate analytics capabilities"""
        print("\n📊 Analytics Demo")
        print("-" * 30)
        
        # Track some sample events
        await self.analytics_service.track_content_creation(
            user_id="user-123",
            content_id="content-456",
            content_type="presentation",
            processing_time=2.5,
            quality_score=0.85
        )
        
        await self.analytics_service.track_content_view(
            user_id="user-789",
            content_id="content-456",
            view_duration=45.2
        )
        
        await self.analytics_service.track_content_export(
            user_id="user-123",
            content_id="content-456",
            export_format="pdf",
            file_size=1024000
        )
        
        print("✅ Sample analytics events tracked!")
        
        # Get dashboard data
        dashboard_data = await self.analytics_service.get_dashboard_data(
            user_id="user-123",
            time_period="7d"
        )
        
        print(f"✅ Dashboard data retrieved!")
        print(f"   User metrics: {len(dashboard_data.get('user_metrics', {}))}")
        print(f"   Content metrics: {len(dashboard_data.get('content_metrics', []))}")
        print(f"   Recent activity: {len(dashboard_data.get('recent_activity', []))}")
        
        # Get system metrics
        system_metrics = await self.analytics_service.get_system_metrics()
        print(f"✅ System metrics retrieved!")
        print(f"   Total users: {system_metrics.total_users}")
        print(f"   Active users: {system_metrics.active_users}")
        print(f"   Total content: {system_metrics.total_content}")
        print(f"   Average response time: {system_metrics.average_response_time:.2f}s")
        
        return dashboard_data, system_metrics

    async def demo_advanced_features(self):
        """Demonstrate advanced features"""
        print("\n🔬 Advanced Features Demo")
        print("-" * 30)
        
        # Content enhancement
        print("Demonstrating content enhancement...")
        enhanced_content = await self.content_generator.enhance_content(
            content_id="demo-content-123",
            enhancement_type="improve_readability",
            instructions="Make the content more engaging and easier to read"
        )
        
        if enhanced_content:
            print("✅ Content enhanced successfully!")
        
        # Content suggestions
        suggestions = await self.content_generator.get_content_suggestions("demo-content-123")
        print(f"✅ Content suggestions retrieved: {len(suggestions)} suggestions")
        
        # Available templates
        templates = self.content_generator.get_available_templates(ContentType.PRESENTATION)
        print(f"✅ Available templates: {len(templates)}")
        
        # Available styles
        styles = self.content_generator.get_available_styles()
        print(f"✅ Available styles: {len(styles)}")
        
        # Available languages
        languages = self.content_generator.get_available_languages()
        print(f"✅ Available languages: {len(languages)}")
        
        return enhanced_content, suggestions

    def print_summary(self, results: Dict[str, Any]):
        """Print demo summary"""
        print("\n📋 Demo Summary")
        print("=" * 50)
        
        print(f"Content Generation: ✅")
        print(f"Presentation Creation: ✅")
        print(f"Document Generation: ✅")
        print(f"Collaboration Features: ✅")
        print(f"Analytics & Monitoring: ✅")
        print(f"Advanced Features: ✅")
        
        print(f"\nTotal features demonstrated: 6")
        print(f"Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

async def main():
    """Main demo function"""
    demo = GammaAppDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    asyncio.run(main())




























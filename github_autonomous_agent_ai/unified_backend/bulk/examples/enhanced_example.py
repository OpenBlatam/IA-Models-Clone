"""
BUL Enhanced Example
====================

Comprehensive example demonstrating all advanced features of the BUL system including:
- Database integration
- Document analysis and enhancement
- Advanced export formats
- Workflow automation
- Multi-language support
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime

from ..core.bul_engine import BULEngine
from ..database.database_manager import get_global_db_manager
from ..ai.document_analyzer import get_global_document_analyzer
from ..export.document_exporter import get_global_document_exporter
from ..workflow.workflow_engine import get_global_workflow_engine, WorkflowDefinition, WorkflowStep
from ..utils.webhook_manager import get_global_webhook_manager
from ..utils.cache_manager import get_global_cache_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BULEnhancedExample:
    """Enhanced example demonstrating all BUL system features."""
    
    def __init__(self):
        self.engine = BULEngine()
        self.db_manager = get_global_db_manager()
        self.document_analyzer = get_global_document_analyzer()
        self.document_exporter = get_global_document_exporter()
        self.workflow_engine = get_global_workflow_engine()
        self.webhook_manager = get_global_webhook_manager()
        self.cache_manager = get_global_cache_manager()
        
        # Enhanced example queries
        self.enhanced_queries = [
            {
                "query": "Create a comprehensive digital marketing strategy for a sustainable fashion brand targeting Gen Z consumers",
                "priority": 1,
                "business_area": "marketing",
                "description": "Advanced Marketing Strategy with AI Analysis"
            },
            {
                "query": "Develop a complete sales automation workflow for B2B SaaS companies with CRM integration and lead scoring",
                "priority": 1,
                "business_area": "sales",
                "description": "Sales Automation Workflow"
            },
            {
                "query": "Design an operational excellence framework for remote-first companies with performance metrics and KPIs",
                "priority": 2,
                "business_area": "operations",
                "description": "Operational Excellence Framework"
            },
            {
                "query": "Create a comprehensive talent acquisition and retention strategy for tech startups in competitive markets",
                "priority": 2,
                "business_area": "hr",
                "description": "Talent Strategy for Tech Startups"
            },
            {
                "query": "Generate a detailed financial model and investment strategy for a Series A funding round",
                "priority": 1,
                "business_area": "finance",
                "description": "Series A Financial Strategy"
            }
        ]
    
    async def setup_database(self):
        """Setup database and create tables."""
        logger.info("Setting up database...")
        
        try:
            await self.db_manager.create_tables_async()
            logger.info("✅ Database tables created successfully")
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
    
    async def demonstrate_document_analysis(self):
        """Demonstrate advanced document analysis features."""
        logger.info("\n🔍 Demonstrating Document Analysis Features")
        logger.info("=" * 50)
        
        # Sample document for analysis
        sample_document = """
        # Marketing Strategy for Sustainable Fashion Brand
        
        ## Executive Summary
        This comprehensive marketing strategy focuses on targeting Gen Z consumers through digital channels, emphasizing sustainability and social responsibility.
        
        ## Key Objectives
        - Increase brand awareness by 150% within 12 months
        - Achieve 25% market share in sustainable fashion segment
        - Build community of 100,000 engaged followers
        
        ## Target Audience
        Gen Z consumers (ages 18-28) who prioritize sustainability and ethical consumption.
        
        ## Marketing Channels
        - Social Media (Instagram, TikTok, YouTube)
        - Influencer Partnerships
        - Content Marketing
        - Email Marketing
        - Paid Advertising
        
        ## Budget Allocation
        - Social Media: 40%
        - Influencer Marketing: 30%
        - Content Creation: 20%
        - Paid Advertising: 10%
        
        ## Success Metrics
        - Engagement Rate: 5%+
        - Conversion Rate: 3%+
        - Customer Lifetime Value: $200+
        """
        
        # Analyze document
        analysis = await self.document_analyzer.analyze_document(sample_document, "marketing_strategy")
        
        logger.info(f"📊 Document Analysis Results:")
        logger.info(f"   Quality Score: {analysis.quality_score}/10")
        logger.info(f"   Complexity Score: {analysis.complexity_score}/10")
        logger.info(f"   Business Areas: {', '.join(analysis.business_areas)}")
        logger.info(f"   Key Points: {len(analysis.key_points)} identified")
        logger.info(f"   Recommendations: {len(analysis.recommendations)} provided")
        
        # Create summary
        summary = await self.document_analyzer.summarize_document(sample_document)
        logger.info(f"📝 Executive Summary: {summary[:100]}...")
        
        # Extract insights
        insights = await self.document_analyzer.extract_insights(sample_document)
        logger.info(f"💡 Insights Categories: {list(insights.keys())}")
        
        # Enhance document
        enhancement = await self.document_analyzer.enhance_document(sample_document)
        logger.info(f"✨ Document Enhancement:")
        logger.info(f"   Improvements Made: {len(enhancement.improvements_made)}")
        logger.info(f"   Quality Improvement: {enhancement.quality_improvement:.2f}x")
    
    async def demonstrate_advanced_export(self):
        """Demonstrate advanced document export features."""
        logger.info("\n📤 Demonstrating Advanced Export Features")
        logger.info("=" * 50)
        
        sample_content = """
        # Business Strategy Document
        
        ## Overview
        This document outlines our comprehensive business strategy for market expansion.
        
        ## Key Initiatives
        1. Product Development
        2. Market Penetration
        3. Customer Acquisition
        4. Operational Efficiency
        
        ## Timeline
        - Q1: Planning and Preparation
        - Q2: Implementation
        - Q3: Optimization
        - Q4: Evaluation
        """
        
        metadata = {
            "author": "BUL System",
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "department": "Strategy"
        }
        
        # Export to different formats
        formats = self.document_exporter.get_available_formats()
        logger.info(f"📋 Available Export Formats: {', '.join(formats)}")
        
        exported_files = []
        for format_type in formats[:3]:  # Export to first 3 formats
            try:
                file_path = await self.document_exporter.export_document(
                    content=sample_content,
                    title="Business Strategy Document",
                    format=format_type,
                    metadata=metadata
                )
                exported_files.append(file_path)
                logger.info(f"✅ Exported to {format_type.upper()}: {file_path}")
            except Exception as e:
                logger.error(f"❌ Export to {format_type} failed: {e}")
        
        # Batch export demonstration
        documents = [
            {
                "content": "Document 1 content...",
                "title": "Document 1",
                "metadata": {"type": "report"}
            },
            {
                "content": "Document 2 content...",
                "title": "Document 2",
                "metadata": {"type": "proposal"}
            }
        ]
        
        try:
            batch_files = await self.document_exporter.batch_export_documents(
                documents=documents,
                format="html"
            )
            logger.info(f"📦 Batch Export: {len(batch_files)} files exported")
        except Exception as e:
            logger.error(f"❌ Batch export failed: {e}")
    
    async def demonstrate_workflow_automation(self):
        """Demonstrate workflow automation features."""
        logger.info("\n🔄 Demonstrating Workflow Automation")
        logger.info("=" * 50)
        
        # Create a sample workflow definition
        workflow_definition = WorkflowDefinition(
            id="document_approval_workflow",
            name="Document Approval Workflow",
            description="Automated workflow for document generation and approval",
            version="1.0",
            steps=[
                WorkflowStep(
                    id="generate_document",
                    name="Generate Document",
                    description="Generate business document using AI",
                    step_type="document_generation",
                    config={
                        "query": "Create a marketing strategy for a new product launch",
                        "business_area": "marketing",
                        "priority": 1
                    }
                ),
                WorkflowStep(
                    id="review_document",
                    name="Review Document",
                    description="Review generated document",
                    step_type="approval",
                    config={
                        "approver": "marketing_manager@company.com",
                        "message": "Please review the generated marketing strategy",
                        "deadline_hours": 24
                    },
                    dependencies=["generate_document"]
                ),
                WorkflowStep(
                    id="notify_team",
                    name="Notify Team",
                    description="Notify team of approved document",
                    step_type="notification",
                    config={
                        "recipients": ["team@company.com"],
                        "subject": "Marketing Strategy Approved",
                        "message": "The marketing strategy has been approved and is ready for implementation."
                    },
                    dependencies=["review_document"]
                )
            ]
        )
        
        # Register workflow
        self.workflow_engine.register_workflow_definition(workflow_definition)
        logger.info("✅ Workflow definition registered")
        
        # Start workflow
        workflow_id = await self.workflow_engine.start_workflow(
            definition_id="document_approval_workflow",
            variables={"company": "Example Corp", "product": "New Product X"},
            created_by="system"
        )
        logger.info(f"🚀 Workflow started: {workflow_id}")
        
        # Monitor workflow
        await asyncio.sleep(5)  # Wait for workflow to process
        
        instance = self.workflow_engine.get_workflow_instance(workflow_id)
        if instance:
            logger.info(f"📊 Workflow Status: {instance.status.value}")
            logger.info(f"   Current Step: {instance.current_step}")
            logger.info(f"   Approval Requests: {len(instance.approval_requests)}")
        
        # Get workflow statistics
        stats = self.workflow_engine.get_workflow_statistics()
        logger.info(f"📈 Workflow Statistics:")
        logger.info(f"   Total Instances: {stats['total_instances']}")
        logger.info(f"   Active Workflows: {stats['active_workflows']}")
        logger.info(f"   Completed: {stats['completed_workflows']}")
    
    async def demonstrate_database_integration(self):
        """Demonstrate database integration features."""
        logger.info("\n🗄️ Demonstrating Database Integration")
        logger.info("=" * 50)
        
        # Create a sample task in database
        task_data = {
            "query": "Create a comprehensive business plan for a tech startup",
            "business_area": "strategy",
            "priority": 1,
            "status": "pending",
            "metadata": {"source": "enhanced_example", "version": "2.0"}
        }
        
        try:
            task_id = await self.db_manager.create_task(task_data)
            logger.info(f"✅ Task created in database: {task_id}")
            
            # Update task
            await self.db_manager.update_task(task_id, {
                "status": "processing",
                "started_at": datetime.now()
            })
            logger.info(f"✅ Task updated in database")
            
            # Retrieve task
            task = await self.db_manager.get_task(task_id)
            if task:
                logger.info(f"📋 Retrieved task: {task['query'][:50]}...")
            
            # Create sample document
            document_data = {
                "task_id": task_id,
                "document_type": "business_plan",
                "title": "Tech Startup Business Plan",
                "content": "This is a comprehensive business plan for a tech startup...",
                "business_area": "strategy",
                "size_bytes": 1500,
                "metadata": {"format": "markdown", "version": "1.0"}
            }
            
            doc_id = await self.db_manager.create_document(document_data)
            logger.info(f"✅ Document created in database: {doc_id}")
            
            # Record system metrics
            await self.db_manager.record_metric(
                metric_type="processing_time",
                metric_value=45.2,
                metadata={"task_id": task_id, "document_count": 1}
            )
            logger.info("✅ System metrics recorded")
            
            # Get system statistics
            stats = await self.db_manager.get_system_statistics()
            logger.info(f"📊 Database Statistics:")
            logger.info(f"   Total Tasks: {stats['total_tasks']}")
            logger.info(f"   Total Documents: {stats['total_documents']}")
            logger.info(f"   Success Rate: {stats['success_rate']:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Database operations failed: {e}")
    
    async def demonstrate_webhook_integration(self):
        """Demonstrate webhook integration features."""
        logger.info("\n🔗 Demonstrating Webhook Integration")
        logger.info("=" * 50)
        
        # Add webhook endpoint (simulated)
        from ..utils.webhook_manager import WebhookEndpoint, WebhookEvent
        
        webhook_endpoint = WebhookEndpoint(
            url="https://webhook.site/your-webhook-url",
            events=[WebhookEvent.TASK_COMPLETED, WebhookEvent.DOCUMENT_GENERATED],
            secret="your-webhook-secret",
            timeout=30,
            retry_attempts=3
        )
        
        self.webhook_manager.add_endpoint(webhook_endpoint)
        logger.info("✅ Webhook endpoint configured")
        
        # Simulate webhook notifications
        await self.webhook_manager.notify_task_created(
            task_id="example-task-123",
            query="Create a marketing strategy",
            business_area="marketing",
            priority=1
        )
        logger.info("📤 Task creation webhook sent")
        
        await self.webhook_manager.notify_document_generated(
            document_id="example-doc-456",
            document_type="marketing_strategy",
            business_area="marketing",
            file_path="/path/to/document.md"
        )
        logger.info("📤 Document generation webhook sent")
        
        # Get webhook statistics
        endpoints = self.webhook_manager.get_endpoints()
        logger.info(f"📊 Webhook Statistics:")
        logger.info(f"   Configured Endpoints: {len(endpoints)}")
    
    async def demonstrate_caching_features(self):
        """Demonstrate advanced caching features."""
        logger.info("\n💾 Demonstrating Caching Features")
        logger.info("=" * 50)
        
        # Cache some data
        await self.cache_manager.set("query_analysis", "marketing strategy analysis", ttl=300)
        await self.cache_manager.set("document_template", "template content", ttl=600)
        await self.cache_manager.set("user_preferences", {"theme": "dark", "language": "en"}, ttl=1800)
        
        logger.info("✅ Data cached successfully")
        
        # Retrieve from cache
        cached_analysis = await self.cache_manager.get("query_analysis")
        cached_template = await self.cache_manager.get("document_template")
        cached_prefs = await self.cache_manager.get("user_preferences")
        
        logger.info(f"📋 Retrieved from cache:")
        logger.info(f"   Analysis: {cached_analysis[:30]}...")
        logger.info(f"   Template: {cached_template[:30]}...")
        logger.info(f"   Preferences: {cached_prefs}")
        
        # Get cache statistics
        stats = self.cache_manager.get_stats()
        logger.info(f"📊 Cache Statistics:")
        logger.info(f"   Hit Rate: {stats['hit_rate']}%")
        logger.info(f"   Memory Entries: {stats['memory_entries']}")
        logger.info(f"   Total Hits: {stats['hits']}")
        logger.info(f"   Total Misses: {stats['misses']}")
    
    async def run_enhanced_demo(self):
        """Run the complete enhanced demonstration."""
        logger.info("🚀 Starting BUL Enhanced Example")
        logger.info("=" * 60)
        
        try:
            # Setup database
            await self.setup_database()
            
            # Demonstrate all features
            await self.demonstrate_document_analysis()
            await self.demonstrate_advanced_export()
            await self.demonstrate_workflow_automation()
            await self.demonstrate_database_integration()
            await self.demonstrate_webhook_integration()
            await self.demonstrate_caching_features()
            
            logger.info("\n🎉 Enhanced example completed successfully!")
            logger.info("All advanced features have been demonstrated:")
            logger.info("✅ Database Integration")
            logger.info("✅ Document Analysis & Enhancement")
            logger.info("✅ Advanced Export Formats")
            logger.info("✅ Workflow Automation")
            logger.info("✅ Webhook Notifications")
            logger.info("✅ Intelligent Caching")
            
        except Exception as e:
            logger.error(f"❌ Enhanced demo failed: {e}")
            raise

async def main():
    """Main function to run the enhanced example."""
    example = BULEnhancedExample()
    await example.run_enhanced_demo()

if __name__ == "__main__":
    asyncio.run(main())


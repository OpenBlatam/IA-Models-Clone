"""
Business Agents System - Demo Script
====================================

Comprehensive demo showcasing all features of the Business Agents system.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from .business_agents import BusinessAgentManager, BusinessArea
from .workflow_engine import WorkflowStatus
from .document_generator import DocumentType, DocumentFormat
from .utils.workflow_builder import WorkflowBuilder

async def demo_business_agents_system():
    """Run comprehensive demo of the Business Agents system."""
    
    print("🚀 Business Agents System - Comprehensive Demo")
    print("=" * 60)
    
    # Initialize the system
    print("\n📋 Initializing Business Agents System...")
    manager = BusinessAgentManager()
    workflow_builder = WorkflowBuilder()
    
    # Demo 1: List all agents
    print("\n🤖 Demo 1: Business Agents")
    print("-" * 30)
    
    agents = manager.list_agents()
    print(f"Total agents available: {len(agents)}")
    
    for agent in agents:
        print(f"  • {agent.name} ({agent.business_area.value})")
        print(f"    Capabilities: {len(agent.capabilities)}")
        for capability in agent.capabilities:
            print(f"      - {capability.name}: {capability.description}")
    
    # Demo 2: Execute agent capability
    print("\n⚡ Demo 2: Execute Agent Capability")
    print("-" * 40)
    
    # Get marketing agent
    marketing_agents = manager.list_agents(business_area=BusinessArea.MARKETING)
    if marketing_agents:
        agent = marketing_agents[0]
        print(f"Executing capability 'campaign_planning' on {agent.name}")
        
        result = await manager.execute_agent_capability(
            agent_id=agent.id,
            capability_name="campaign_planning",
            inputs={
                "target_audience": "B2B SaaS companies",
                "budget": 50000,
                "goals": ["lead_generation", "brand_awareness"]
            },
            parameters={"campaign_duration": "3 months"}
        )
        
        print(f"Result: {json.dumps(result, indent=2)}")
    
    # Demo 3: Create and execute workflow
    print("\n🔄 Demo 3: Workflow Creation and Execution")
    print("-" * 45)
    
    # Create a simple marketing workflow
    workflow_steps = [
        {
            "name": "Market Research",
            "step_type": "task",
            "description": "Research target market and competitors",
            "agent_type": "marketing_001",
            "parameters": {"research_depth": "comprehensive"}
        },
        {
            "name": "Content Creation",
            "step_type": "task",
            "description": "Create campaign content",
            "agent_type": "marketing_001",
            "parameters": {"content_types": ["copy", "visual"]}
        },
        {
            "name": "Campaign Launch",
            "step_type": "task",
            "description": "Launch marketing campaign",
            "agent_type": "marketing_001",
            "parameters": {"channels": ["social", "email"]}
        }
    ]
    
    # Create workflow
    workflow = await manager.create_business_workflow(
        name="Demo Marketing Campaign",
        description="Demo workflow for marketing campaign",
        business_area=BusinessArea.MARKETING,
        steps=workflow_steps,
        created_by="demo_user@example.com",
        variables={
            "campaign_name": "Q1 2024 Campaign",
            "target_audience": "B2B SaaS",
            "budget": 50000
        }
    )
    
    print(f"Created workflow: {workflow.name} (ID: {workflow.id})")
    print(f"Status: {workflow.status.value}")
    print(f"Steps: {len(workflow.steps)}")
    
    # Demo 4: Document generation
    print("\n📄 Demo 4: Document Generation")
    print("-" * 35)
    
    # Generate a business plan
    document_result = await manager.generate_business_document(
        document_type=DocumentType.BUSINESS_PLAN,
        title="Demo Business Plan 2024",
        description="Comprehensive business plan for demo purposes",
        business_area="strategy",
        created_by="demo_user@example.com",
        variables={
            "company_name": "Demo Tech Solutions",
            "industry": "SaaS",
            "target_market": "B2B Enterprise",
            "business_model": "Subscription",
            "financial_projections": "3-year forecast"
        },
        format=DocumentFormat.MARKDOWN
    )
    
    print(f"Generated document: {document_result['title']}")
    print(f"Document ID: {document_result['document_id']}")
    print(f"File path: {document_result['file_path']}")
    print(f"Size: {document_result['size_bytes']} bytes")
    
    # Demo 5: Workflow templates
    print("\n📋 Demo 5: Workflow Templates")
    print("-" * 35)
    
    templates = workflow_builder.list_templates()
    print(f"Available templates: {len(templates)}")
    
    for template in templates:
        print(f"  • {template.name} ({template.business_area.value})")
        print(f"    Category: {template.category}")
        print(f"    Nodes: {len(template.nodes)}")
        print(f"    Tags: {', '.join(template.tags)}")
    
    # Demo 6: Convert template to workflow
    print("\n🔄 Demo 6: Template to Workflow Conversion")
    print("-" * 45)
    
    if templates:
        template = templates[0]  # Use first template
        print(f"Converting template: {template.name}")
        
        steps = workflow_builder.convert_template_to_workflow(
            template_id=template.id,
            variables={
                "campaign_name": "Template Campaign",
                "target_audience": "General",
                "budget": 25000
            }
        )
        
        print(f"Converted to {len(steps)} workflow steps:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step['name']} ({step['step_type']})")
    
    # Demo 7: Business areas overview
    print("\n🏢 Demo 7: Business Areas Overview")
    print("-" * 40)
    
    business_areas = manager.get_business_areas()
    print(f"Supported business areas: {len(business_areas)}")
    
    for area in business_areas:
        agents_in_area = manager.get_agents_by_business_area(area)
        print(f"  • {area.value.title()}: {len(agents_in_area)} agents")
        
        # Show capabilities for each area
        if agents_in_area:
            total_capabilities = sum(len(agent.capabilities) for agent in agents_in_area)
            print(f"    Total capabilities: {total_capabilities}")
    
    # Demo 8: System metrics
    print("\n📊 Demo 8: System Metrics")
    print("-" * 30)
    
    agents = manager.list_agents()
    workflows = manager.list_workflows()
    
    print(f"System Overview:")
    print(f"  • Total agents: {len(agents)}")
    print(f"  • Active agents: {len([a for a in agents if a.is_active])}")
    print(f"  • Total workflows: {len(workflows)}")
    print(f"  • Business areas: {len(business_areas)}")
    
    # Show workflow status distribution
    status_counts = {}
    for workflow in workflows:
        status = workflow.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"  • Workflow status distribution:")
    for status, count in status_counts.items():
        print(f"    - {status}: {count}")
    
    # Demo 9: Advanced workflow with conditions
    print("\n🔀 Demo 9: Advanced Workflow with Conditions")
    print("-" * 50)
    
    advanced_steps = [
        {
            "name": "Lead Generation",
            "step_type": "task",
            "description": "Generate leads",
            "agent_type": "sales_001",
            "parameters": {"lead_quality": "high"}
        },
        {
            "name": "Lead Qualification",
            "step_type": "condition",
            "description": "Check if lead is qualified",
            "agent_type": "sales_001",
            "conditions": {"lead_score": ">= 70"},
            "parameters": {"qualification_criteria": "BANT"}
        },
        {
            "name": "Create Proposal",
            "step_type": "task",
            "description": "Create sales proposal",
            "agent_type": "sales_001",
            "parameters": {"customization_level": "high"}
        },
        {
            "name": "Follow Up",
            "step_type": "task",
            "description": "Follow up with prospect",
            "agent_type": "sales_001",
            "parameters": {"follow_up_frequency": "weekly"}
        }
    ]
    
    advanced_workflow = await manager.create_business_workflow(
        name="Advanced Sales Process",
        description="Sales process with conditional logic",
        business_area=BusinessArea.SALES,
        steps=advanced_steps,
        created_by="demo_user@example.com"
    )
    
    print(f"Created advanced workflow: {advanced_workflow.name}")
    print(f"Steps with conditions: {len([s for s in advanced_workflow.steps if s.conditions])}")
    
    # Demo 10: Document types and formats
    print("\n📝 Demo 10: Document Types and Formats")
    print("-" * 45)
    
    document_types = [
        DocumentType.BUSINESS_PLAN,
        DocumentType.MARKETING_STRATEGY,
        DocumentType.SALES_PROPOSAL,
        DocumentType.FINANCIAL_REPORT,
        DocumentType.OPERATIONAL_MANUAL,
        DocumentType.HR_POLICY,
        DocumentType.TECHNICAL_SPECIFICATION,
        DocumentType.PROJECT_PROPOSAL,
        DocumentType.CONTRACT,
        DocumentType.PRESENTATION,
        DocumentType.EMAIL_TEMPLATE,
        DocumentType.SOCIAL_MEDIA_POST,
        DocumentType.BLOG_POST,
        DocumentType.PRESS_RELEASE,
        DocumentType.USER_MANUAL,
        DocumentType.TRAINING_MATERIAL
    ]
    
    formats = [
        DocumentFormat.MARKDOWN,
        DocumentFormat.HTML,
        DocumentFormat.PDF,
        DocumentFormat.DOCX,
        DocumentFormat.TXT,
        DocumentFormat.JSON
    ]
    
    print(f"Supported document types: {len(document_types)}")
    for doc_type in document_types:
        print(f"  • {doc_type.value.replace('_', ' ').title()}")
    
    print(f"\nSupported formats: {len(formats)}")
    for format_type in formats:
        print(f"  • {format_type.value.upper()}")
    
    # Demo 11: Workflow validation
    print("\n✅ Demo 11: Workflow Validation")
    print("-" * 35)
    
    # Create a workflow with validation issues
    invalid_nodes = [
        {
            "id": "node1",
            "type": "task",
            "name": "Task 1",
            "description": "First task",
            "position": {"x": 100, "y": 100},
            "size": {"width": 120, "height": 60},
            "style": {},
            "data": {},
            "inputs": [],
            "outputs": ["output"]
        }
    ]
    
    invalid_connections = []
    
    validation_result = workflow_builder.validate_workflow(invalid_nodes, invalid_connections)
    print(f"Validation result: {validation_result}")
    
    # Demo 12: System capabilities summary
    print("\n🎯 Demo 12: System Capabilities Summary")
    print("-" * 45)
    
    print("Business Agents System provides:")
    print("  ✅ Multi-business area support (8 areas)")
    print("  ✅ Advanced workflow engine with conditional logic")
    print("  ✅ Comprehensive document generation (16 types)")
    print("  ✅ AI-powered content enhancement")
    print("  ✅ Visual workflow builder with templates")
    print("  ✅ Real-time execution and monitoring")
    print("  ✅ RESTful API for integration")
    print("  ✅ Extensible agent architecture")
    print("  ✅ Template-based workflow creation")
    print("  ✅ Multi-format document output")
    
    print("\n🚀 Demo completed successfully!")
    print("=" * 60)
    
    return {
        "agents_count": len(agents),
        "workflows_count": len(workflows),
        "templates_count": len(templates),
        "document_types_count": len(document_types),
        "business_areas_count": len(business_areas),
        "demo_completed": True
    }

async def demo_api_endpoints():
    """Demo API endpoints functionality."""
    
    print("\n🌐 API Endpoints Demo")
    print("-" * 25)
    
    print("Available API endpoints:")
    print("  GET  /business-agents/                    - System overview")
    print("  GET  /business-agents/agents              - List agents")
    print("  GET  /business-agents/agents/{id}         - Get agent details")
    print("  POST /business-agents/agents/{id}/execute - Execute capability")
    print("  GET  /business-agents/workflows           - List workflows")
    print("  POST /business-agents/workflows           - Create workflow")
    print("  POST /business-agents/workflows/{id}/execute - Execute workflow")
    print("  POST /business-agents/documents/generate  - Generate document")
    print("  GET  /business-agents/documents           - List documents")
    print("  GET  /business-agents/workflow-templates  - Get templates")
    print("  GET  /health                              - Health check")
    print("  GET  /system/info                         - System information")
    print("  GET  /system/metrics                      - System metrics")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_business_agents_system())
    asyncio.run(demo_api_endpoints())






























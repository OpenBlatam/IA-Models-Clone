"""
Database Manager
================

Database management and initialization utilities.
"""

from sqlalchemy import create_engine, Index
from sqlalchemy.orm import sessionmaker
from typing import Dict, Any
import logging

from .base import Base
from .user_models import User, Role
from .agent_models import BusinessAgent
from .document_models import Template, Document
from .workflow_models import Workflow
from .notification_models import Notification
from .system_models import Metric, Alert

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for handling database operations."""
    
    def __init__(self, database_url: str = "sqlite:///./business_agents.db"):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        self._create_indexes()
        
    def _create_indexes(self):
        """Create database indexes for better performance."""
        # Create indexes for better performance
        Index('idx_users_username', User.username)
        Index('idx_users_email', User.email)
        Index('idx_users_created_at', User.created_at)
        Index('idx_workflows_business_area', Workflow.business_area)
        Index('idx_workflows_status', Workflow.status)
        Index('idx_workflows_created_at', Workflow.created_at)
        Index('idx_documents_type', Document.document_type)
        Index('idx_documents_business_area', Document.business_area)
        Index('idx_documents_created_at', Document.created_at)
        Index('idx_metrics_name_timestamp', Metric.name, Metric.timestamp)
        Index('idx_alerts_level_resolved', Alert.level, Alert.is_resolved)
        Index('idx_notifications_user_read', Notification.user_id, Notification.is_read)
        
    def get_session(self):
        """Get database session."""
        return self.SessionLocal()
        
    def init_default_data(self):
        """Initialize default data."""
        session = self.get_session()
        try:
            # Create default roles
            self._create_default_roles(session)
            
            # Create default business agents
            self._create_default_agents(session)
            
            # Create default templates
            self._create_default_templates(session)
            
            session.commit()
            logger.info("Default data initialized successfully")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error initializing default data: {str(e)}")
        finally:
            session.close()
    
    def _create_default_roles(self, session):
        """Create default roles."""
        roles = [
            Role(
                name="admin",
                description="System administrator with full access",
                permissions=["*"]
            ),
            Role(
                name="manager",
                description="Manager with management permissions",
                permissions=["workflows:create", "workflows:update", "workflows:delete", "agents:manage", "analytics:view"]
            ),
            Role(
                name="user",
                description="Regular user with basic permissions",
                permissions=["workflows:view", "workflows:execute", "documents:view", "documents:create"]
            ),
            Role(
                name="viewer",
                description="Read-only user",
                permissions=["workflows:view", "documents:view", "analytics:view"]
            ),
            Role(
                name="operator",
                description="System operator with monitoring permissions",
                permissions=["monitoring:view", "monitoring:manage", "alerts:view", "alerts:manage"]
            )
        ]
        
        session.add_all(roles)
    
    def _create_default_agents(self, session):
        """Create default business agents."""
        agents = [
            BusinessAgent(
                name="Marketing Agent",
                business_area="marketing",
                description="Handles marketing campaigns, content creation, and lead generation",
                capabilities=["content_creation", "campaign_management", "lead_generation", "analytics"],
                configuration={"ai_model": "gpt-4", "max_content_length": 2000}
            ),
            BusinessAgent(
                name="Sales Agent",
                business_area="sales",
                description="Manages sales processes, customer interactions, and deal tracking",
                capabilities=["lead_qualification", "proposal_generation", "deal_tracking", "customer_communication"],
                configuration={"ai_model": "gpt-4", "response_time": "fast"}
            ),
            BusinessAgent(
                name="Operations Agent",
                business_area="operations",
                description="Handles operational tasks, process optimization, and resource management",
                capabilities=["process_optimization", "resource_management", "workflow_automation", "reporting"],
                configuration={"ai_model": "gpt-4", "optimization_level": "high"}
            ),
            BusinessAgent(
                name="HR Agent",
                business_area="hr",
                description="Manages human resources, recruitment, and employee relations",
                capabilities=["recruitment", "employee_management", "policy_creation", "training"],
                configuration={"ai_model": "gpt-4", "compliance_mode": "strict"}
            ),
            BusinessAgent(
                name="Finance Agent",
                business_area="finance",
                description="Handles financial analysis, budgeting, and reporting",
                capabilities=["financial_analysis", "budgeting", "reporting", "compliance"],
                configuration={"ai_model": "gpt-4", "accuracy_level": "high"}
            ),
            BusinessAgent(
                name="Legal Agent",
                business_area="legal",
                description="Manages legal documents, compliance, and contract review",
                capabilities=["contract_review", "compliance_checking", "document_analysis", "risk_assessment"],
                configuration={"ai_model": "gpt-4", "legal_mode": "strict"}
            ),
            BusinessAgent(
                name="Technical Agent",
                business_area="technical",
                description="Handles technical documentation, system analysis, and development support",
                capabilities=["technical_writing", "system_analysis", "code_review", "documentation"],
                configuration={"ai_model": "gpt-4", "technical_level": "advanced"}
            ),
            BusinessAgent(
                name="Content Agent",
                business_area="content",
                description="Creates and manages content across all platforms and formats",
                capabilities=["content_creation", "seo_optimization", "content_management", "publishing"],
                configuration={"ai_model": "gpt-4", "content_quality": "high"}
            )
        ]
        
        session.add_all(agents)
    
    def _create_default_templates(self, session):
        """Create default templates."""
        templates = [
            Template(
                name="Marketing Campaign Workflow",
                description="Complete workflow for creating and executing marketing campaigns",
                business_area="marketing",
                template_data={
                    "steps": [
                        {"name": "Campaign Planning", "type": "planning", "duration": 30},
                        {"name": "Content Creation", "type": "content", "duration": 60},
                        {"name": "Audience Targeting", "type": "targeting", "duration": 20},
                        {"name": "Campaign Launch", "type": "execution", "duration": 10},
                        {"name": "Performance Monitoring", "type": "monitoring", "duration": 15}
                    ]
                },
                category="campaign",
                tags=["marketing", "campaign", "automation"],
                is_public=True
            ),
            Template(
                name="Sales Process Workflow",
                description="Standard sales process from lead to close",
                business_area="sales",
                template_data={
                    "steps": [
                        {"name": "Lead Qualification", "type": "qualification", "duration": 15},
                        {"name": "Needs Assessment", "type": "assessment", "duration": 30},
                        {"name": "Proposal Creation", "type": "proposal", "duration": 45},
                        {"name": "Negotiation", "type": "negotiation", "duration": 60},
                        {"name": "Contract Signing", "type": "contract", "duration": 20}
                    ]
                },
                category="sales",
                tags=["sales", "process", "automation"],
                is_public=True
            ),
            Template(
                name="Employee Onboarding Workflow",
                description="Complete employee onboarding process",
                business_area="hr",
                template_data={
                    "steps": [
                        {"name": "Document Collection", "type": "documentation", "duration": 20},
                        {"name": "System Setup", "type": "setup", "duration": 30},
                        {"name": "Training Assignment", "type": "training", "duration": 40},
                        {"name": "Team Introduction", "type": "introduction", "duration": 15},
                        {"name": "First Week Review", "type": "review", "duration": 25}
                    ]
                },
                category="onboarding",
                tags=["hr", "onboarding", "employee"],
                is_public=True
            )
        ]
        
        session.add_all(templates)

# Global database manager instance
db_manager = DatabaseManager()

# Initialize database
db_manager.create_tables()
db_manager.init_default_data()

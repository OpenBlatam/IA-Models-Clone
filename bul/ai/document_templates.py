"""
Advanced Document Templates System for BUL
Provides intelligent document templates with smart suggestions and customization
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import json
import asyncio
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Supported document types"""
    BUSINESS_PLAN = "business_plan"
    MARKETING_PROPOSAL = "marketing_proposal"
    FINANCIAL_REPORT = "financial_report"
    PROJECT_PROPOSAL = "project_proposal"
    CONTRACT = "contract"
    PRESENTATION = "presentation"
    EMAIL_TEMPLATE = "email_template"
    PRESS_RELEASE = "press_release"
    WHITE_PAPER = "white_paper"
    CASE_STUDY = "case_study"
    PROPOSAL = "proposal"
    REPORT = "report"
    MANUAL = "manual"
    POLICY = "policy"
    PROCEDURE = "procedure"


class TemplateComplexity(str, Enum):
    """Template complexity levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class IndustryType(str, Enum):
    """Industry-specific templates"""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    EDUCATION = "education"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    CONSULTING = "consulting"
    REAL_ESTATE = "real_estate"
    LEGAL = "legal"
    NON_PROFIT = "non_profit"
    STARTUP = "startup"
    E_COMMERCE = "e_commerce"


class TemplateField(BaseModel):
    """Template field definition"""
    name: str = Field(..., description="Field name")
    label: str = Field(..., description="Human-readable label")
    type: str = Field(..., description="Field type (text, number, date, select, etc.)")
    required: bool = Field(default=True, description="Is field required")
    placeholder: Optional[str] = Field(None, description="Placeholder text")
    options: Optional[List[str]] = Field(None, description="Options for select fields")
    validation: Optional[Dict[str, Any]] = Field(None, description="Validation rules")
    ai_suggestion: Optional[str] = Field(None, description="AI-generated suggestion")


class DocumentTemplate(BaseModel):
    """Document template definition"""
    id: str = Field(..., description="Unique template ID")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    document_type: DocumentType = Field(..., description="Type of document")
    industry: Optional[IndustryType] = Field(None, description="Target industry")
    complexity: TemplateComplexity = Field(..., description="Template complexity")
    fields: List[TemplateField] = Field(..., description="Template fields")
    content_structure: Dict[str, Any] = Field(..., description="Document structure")
    ai_prompts: Dict[str, str] = Field(..., description="AI prompts for content generation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0", description="Template version")
    tags: List[str] = Field(default_factory=list, description="Template tags")


class SmartSuggestion(BaseModel):
    """AI-generated smart suggestion"""
    field_name: str = Field(..., description="Field this suggestion applies to")
    suggestion: str = Field(..., description="Suggested content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Why this suggestion was made")
    alternatives: List[str] = Field(default_factory=list, description="Alternative suggestions")


class TemplateRecommendation(BaseModel):
    """Template recommendation based on user context"""
    template: DocumentTemplate = Field(..., description="Recommended template")
    score: float = Field(..., ge=0.0, le=1.0, description="Recommendation score")
    reasoning: str = Field(..., description="Why this template was recommended")
    customization_suggestions: List[str] = Field(default_factory=list, description="Customization suggestions")


class DocumentTemplateManager:
    """Advanced document template management system"""
    
    def __init__(self):
        self.templates: Dict[str, DocumentTemplate] = {}
        self.template_cache: Dict[str, Any] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default document templates"""
        default_templates = [
            self._create_business_plan_template(),
            self._create_marketing_proposal_template(),
            self._create_financial_report_template(),
            self._create_project_proposal_template(),
            self._create_contract_template(),
            self._create_presentation_template(),
            self._create_email_template(),
            self._create_press_release_template(),
        ]
        
        for template in default_templates:
            self.templates[template.id] = template
        
        logger.info(f"Loaded {len(default_templates)} default templates")
    
    def _create_business_plan_template(self) -> DocumentTemplate:
        """Create business plan template"""
        return DocumentTemplate(
            id="business_plan_v1",
            name="Comprehensive Business Plan",
            description="Complete business plan template with financial projections and market analysis",
            document_type=DocumentType.BUSINESS_PLAN,
            complexity=TemplateComplexity.ADVANCED,
            fields=[
                TemplateField(
                    name="company_name",
                    label="Company Name",
                    type="text",
                    required=True,
                    placeholder="Enter your company name",
                    ai_suggestion="Based on your industry and target market"
                ),
                TemplateField(
                    name="industry",
                    label="Industry",
                    type="select",
                    required=True,
                    options=[industry.value for industry in IndustryType],
                    ai_suggestion="Select the primary industry your business operates in"
                ),
                TemplateField(
                    name="target_market",
                    label="Target Market",
                    type="text",
                    required=True,
                    placeholder="Describe your target customers",
                    ai_suggestion="Define your ideal customer profile"
                ),
                TemplateField(
                    name="business_model",
                    label="Business Model",
                    type="text",
                    required=True,
                    placeholder="How will you make money?",
                    ai_suggestion="Describe your revenue streams and value proposition"
                ),
                TemplateField(
                    name="funding_required",
                    label="Funding Required",
                    type="number",
                    required=False,
                    placeholder="Amount in USD",
                    validation={"min": 0, "max": 10000000}
                ),
                TemplateField(
                    name="timeline",
                    label="Implementation Timeline",
                    type="text",
                    required=True,
                    placeholder="6 months, 1 year, etc.",
                    ai_suggestion="Realistic timeline for business launch and growth"
                )
            ],
            content_structure={
                "sections": [
                    "executive_summary",
                    "company_description",
                    "market_analysis",
                    "organization_management",
                    "service_product_line",
                    "marketing_sales",
                    "funding_request",
                    "financial_projections",
                    "appendix"
                ]
            },
            ai_prompts={
                "executive_summary": "Create a compelling executive summary that highlights the key value proposition, market opportunity, and financial projections for {company_name} in the {industry} industry.",
                "market_analysis": "Analyze the {industry} market for {company_name}, including market size, growth trends, competitive landscape, and target customer segments.",
                "financial_projections": "Create realistic 3-year financial projections for {company_name} including revenue, expenses, and profitability based on the {business_model} business model."
            },
            tags=["business", "planning", "funding", "startup"],
            metadata={
                "estimated_completion_time": "2-3 hours",
                "difficulty": "advanced",
                "sections_count": 9
            }
        )
    
    def _create_marketing_proposal_template(self) -> DocumentTemplate:
        """Create marketing proposal template"""
        return DocumentTemplate(
            id="marketing_proposal_v1",
            name="Marketing Campaign Proposal",
            description="Professional marketing proposal template with ROI projections",
            document_type=DocumentType.MARKETING_PROPOSAL,
            complexity=TemplateComplexity.INTERMEDIATE,
            fields=[
                TemplateField(
                    name="campaign_name",
                    label="Campaign Name",
                    type="text",
                    required=True,
                    placeholder="Enter campaign name"
                ),
                TemplateField(
                    name="target_audience",
                    label="Target Audience",
                    type="text",
                    required=True,
                    placeholder="Describe your target audience"
                ),
                TemplateField(
                    name="budget",
                    label="Campaign Budget",
                    type="number",
                    required=True,
                    validation={"min": 1000, "max": 1000000}
                ),
                TemplateField(
                    name="duration",
                    label="Campaign Duration",
                    type="select",
                    required=True,
                    options=["1 month", "3 months", "6 months", "1 year"]
                ),
                TemplateField(
                    name="channels",
                    label="Marketing Channels",
                    type="text",
                    required=True,
                    placeholder="Social media, email, PPC, etc."
                )
            ],
            content_structure={
                "sections": [
                    "campaign_overview",
                    "target_audience_analysis",
                    "strategy_tactics",
                    "budget_breakdown",
                    "roi_projections",
                    "timeline",
                    "success_metrics"
                ]
            },
            ai_prompts={
                "campaign_overview": "Create a compelling campaign overview for {campaign_name} targeting {target_audience} with a budget of ${budget}.",
                "roi_projections": "Calculate realistic ROI projections for a {duration} marketing campaign with ${budget} budget using {channels} channels."
            },
            tags=["marketing", "campaign", "proposal", "roi"],
            metadata={
                "estimated_completion_time": "1-2 hours",
                "difficulty": "intermediate"
            }
        )
    
    def _create_financial_report_template(self) -> DocumentTemplate:
        """Create financial report template"""
        return DocumentTemplate(
            id="financial_report_v1",
            name="Financial Performance Report",
            description="Comprehensive financial report with analysis and recommendations",
            document_type=DocumentType.FINANCIAL_REPORT,
            complexity=TemplateComplexity.ADVANCED,
            fields=[
                TemplateField(
                    name="company_name",
                    label="Company Name",
                    type="text",
                    required=True
                ),
                TemplateField(
                    name="report_period",
                    label="Report Period",
                    type="select",
                    required=True,
                    options=["Q1", "Q2", "Q3", "Q4", "Annual", "Monthly"]
                ),
                TemplateField(
                    name="revenue",
                    label="Total Revenue",
                    type="number",
                    required=True,
                    validation={"min": 0}
                ),
                TemplateField(
                    name="expenses",
                    label="Total Expenses",
                    type="number",
                    required=True,
                    validation={"min": 0}
                ),
                TemplateField(
                    name="industry_benchmark",
                    label="Industry Benchmark",
                    type="text",
                    required=False,
                    placeholder="Industry average metrics"
                )
            ],
            content_structure={
                "sections": [
                    "executive_summary",
                    "financial_highlights",
                    "revenue_analysis",
                    "expense_analysis",
                    "profitability_analysis",
                    "cash_flow_analysis",
                    "key_metrics",
                    "recommendations"
                ]
            },
            ai_prompts={
                "executive_summary": "Create an executive summary for {company_name}'s {report_period} financial performance with revenue of ${revenue} and expenses of ${expenses}.",
                "recommendations": "Provide strategic recommendations for {company_name} based on their financial performance and industry benchmarks."
            },
            tags=["finance", "report", "analysis", "performance"],
            metadata={
                "estimated_completion_time": "2-3 hours",
                "difficulty": "advanced"
            }
        )
    
    def _create_project_proposal_template(self) -> DocumentTemplate:
        """Create project proposal template"""
        return DocumentTemplate(
            id="project_proposal_v1",
            name="Project Proposal",
            description="Detailed project proposal with timeline and resource requirements",
            document_type=DocumentType.PROJECT_PROPOSAL,
            complexity=TemplateComplexity.INTERMEDIATE,
            fields=[
                TemplateField(
                    name="project_name",
                    label="Project Name",
                    type="text",
                    required=True
                ),
                TemplateField(
                    name="project_type",
                    label="Project Type",
                    type="select",
                    required=True,
                    options=["Software Development", "Marketing Campaign", "Research", "Infrastructure", "Process Improvement"]
                ),
                TemplateField(
                    name="budget",
                    label="Project Budget",
                    type="number",
                    required=True,
                    validation={"min": 1000}
                ),
                TemplateField(
                    name="timeline",
                    label="Project Timeline",
                    type="text",
                    required=True,
                    placeholder="e.g., 3 months, 6 months"
                ),
                TemplateField(
                    name="team_size",
                    label="Team Size",
                    type="number",
                    required=True,
                    validation={"min": 1, "max": 50}
                )
            ],
            content_structure={
                "sections": [
                    "project_overview",
                    "objectives_goals",
                    "scope_requirements",
                    "methodology",
                    "timeline_milestones",
                    "resource_requirements",
                    "risk_assessment",
                    "success_criteria"
                ]
            },
            ai_prompts={
                "project_overview": "Create a comprehensive project overview for {project_name}, a {project_type} project with a budget of ${budget} and timeline of {timeline}.",
                "risk_assessment": "Identify potential risks and mitigation strategies for a {project_type} project with {team_size} team members."
            },
            tags=["project", "proposal", "management", "planning"],
            metadata={
                "estimated_completion_time": "1-2 hours",
                "difficulty": "intermediate"
            }
        )
    
    def _create_contract_template(self) -> DocumentTemplate:
        """Create contract template"""
        return DocumentTemplate(
            id="contract_v1",
            name="Service Contract",
            description="Professional service contract template with legal clauses",
            document_type=DocumentType.CONTRACT,
            complexity=TemplateComplexity.ADVANCED,
            fields=[
                TemplateField(
                    name="client_name",
                    label="Client Name",
                    type="text",
                    required=True
                ),
                TemplateField(
                    name="service_provider",
                    label="Service Provider",
                    type="text",
                    required=True
                ),
                TemplateField(
                    name="service_description",
                    label="Service Description",
                    type="text",
                    required=True
                ),
                TemplateField(
                    name="contract_value",
                    label="Contract Value",
                    type="number",
                    required=True,
                    validation={"min": 0}
                ),
                TemplateField(
                    name="contract_duration",
                    label="Contract Duration",
                    type="text",
                    required=True,
                    placeholder="e.g., 6 months, 1 year"
                )
            ],
            content_structure={
                "sections": [
                    "parties_information",
                    "service_description",
                    "terms_conditions",
                    "payment_terms",
                    "deliverables",
                    "intellectual_property",
                    "confidentiality",
                    "termination_clause",
                    "dispute_resolution"
                ]
            },
            ai_prompts={
                "service_description": "Create a detailed service description for {service_description} between {client_name} and {service_provider}.",
                "payment_terms": "Define clear payment terms for a contract valued at ${contract_value} with a duration of {contract_duration}."
            },
            tags=["contract", "legal", "service", "agreement"],
            metadata={
                "estimated_completion_time": "2-3 hours",
                "difficulty": "advanced",
                "legal_review_required": True
            }
        )
    
    def _create_presentation_template(self) -> DocumentTemplate:
        """Create presentation template"""
        return DocumentTemplate(
            id="presentation_v1",
            name="Business Presentation",
            description="Professional business presentation template with slide structure",
            document_type=DocumentType.PRESENTATION,
            complexity=TemplateComplexity.INTERMEDIATE,
            fields=[
                TemplateField(
                    name="presentation_title",
                    label="Presentation Title",
                    type="text",
                    required=True
                ),
                TemplateField(
                    name="audience",
                    label="Target Audience",
                    type="text",
                    required=True,
                    placeholder="e.g., investors, clients, team"
                ),
                TemplateField(
                    name="duration",
                    label="Presentation Duration",
                    type="select",
                    required=True,
                    options=["5 minutes", "10 minutes", "15 minutes", "30 minutes", "1 hour"]
                ),
                TemplateField(
                    name="key_points",
                    label="Key Points to Cover",
                    type="text",
                    required=True,
                    placeholder="List main topics to discuss"
                )
            ],
            content_structure={
                "sections": [
                    "title_slide",
                    "agenda",
                    "problem_statement",
                    "solution",
                    "market_opportunity",
                    "business_model",
                    "financial_projections",
                    "team",
                    "next_steps",
                    "questions"
                ]
            },
            ai_prompts={
                "title_slide": "Create a compelling title slide for '{presentation_title}' presentation to {audience}.",
                "agenda": "Create an agenda for a {duration} presentation covering {key_points}."
            },
            tags=["presentation", "business", "pitch", "slides"],
            metadata={
                "estimated_completion_time": "1-2 hours",
                "difficulty": "intermediate",
                "slide_count": 10
            }
        )
    
    def _create_email_template(self) -> DocumentTemplate:
        """Create email template"""
        return DocumentTemplate(
            id="email_template_v1",
            name="Professional Email Template",
            description="Professional email template with various business scenarios",
            document_type=DocumentType.EMAIL_TEMPLATE,
            complexity=TemplateComplexity.BASIC,
            fields=[
                TemplateField(
                    name="email_type",
                    label="Email Type",
                    type="select",
                    required=True,
                    options=["Follow-up", "Introduction", "Proposal", "Thank you", "Meeting request", "Project update"]
                ),
                TemplateField(
                    name="recipient",
                    label="Recipient",
                    type="text",
                    required=True,
                    placeholder="Who are you writing to?"
                ),
                TemplateField(
                    name="context",
                    label="Context",
                    type="text",
                    required=True,
                    placeholder="What is the purpose of this email?"
                ),
                TemplateField(
                    name="tone",
                    label="Tone",
                    type="select",
                    required=True,
                    options=["Professional", "Friendly", "Formal", "Casual", "Urgent"]
                )
            ],
            content_structure={
                "sections": [
                    "subject_line",
                    "greeting",
                    "opening",
                    "main_content",
                    "call_to_action",
                    "closing"
                ]
            },
            ai_prompts={
                "subject_line": "Create an effective subject line for a {email_type} email to {recipient}.",
                "main_content": "Write the main content for a {tone} {email_type} email about {context}."
            },
            tags=["email", "communication", "business", "professional"],
            metadata={
                "estimated_completion_time": "15-30 minutes",
                "difficulty": "basic"
            }
        )
    
    def _create_press_release_template(self) -> DocumentTemplate:
        """Create press release template"""
        return DocumentTemplate(
            id="press_release_v1",
            name="Press Release",
            description="Professional press release template for media outreach",
            document_type=DocumentType.PRESS_RELEASE,
            complexity=TemplateComplexity.INTERMEDIATE,
            fields=[
                TemplateField(
                    name="headline",
                    label="Headline",
                    type="text",
                    required=True,
                    placeholder="Compelling headline for the news"
                ),
                TemplateField(
                    name="company_name",
                    label="Company Name",
                    type="text",
                    required=True
                ),
                TemplateField(
                    name="news_type",
                    label="News Type",
                    type="select",
                    required=True,
                    options=["Product Launch", "Partnership", "Funding", "Award", "Expansion", "Other"]
                ),
                TemplateField(
                    name="key_quotes",
                    label="Key Quotes",
                    type="text",
                    required=False,
                    placeholder="Quotes from executives or key stakeholders"
                )
            ],
            content_structure={
                "sections": [
                    "headline",
                    "subheadline",
                    "dateline",
                    "lead_paragraph",
                    "body_paragraphs",
                    "quotes",
                    "company_information",
                    "contact_information"
                ]
            },
            ai_prompts={
                "lead_paragraph": "Create a compelling lead paragraph for a press release about {news_type} from {company_name} with the headline '{headline}'.",
                "body_paragraphs": "Write detailed body paragraphs explaining the {news_type} news from {company_name}."
            },
            tags=["press", "media", "announcement", "publicity"],
            metadata={
                "estimated_completion_time": "1-2 hours",
                "difficulty": "intermediate"
            }
        )
    
    async def get_template(self, template_id: str) -> Optional[DocumentTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)
    
    async def list_templates(
        self,
        document_type: Optional[DocumentType] = None,
        industry: Optional[IndustryType] = None,
        complexity: Optional[TemplateComplexity] = None,
        tags: Optional[List[str]] = None
    ) -> List[DocumentTemplate]:
        """List templates with optional filtering"""
        templates = list(self.templates.values())
        
        if document_type:
            templates = [t for t in templates if t.document_type == document_type]
        
        if industry:
            templates = [t for t in templates if t.industry == industry]
        
        if complexity:
            templates = [t for t in templates if t.complexity == complexity]
        
        if tags:
            templates = [t for t in templates if any(tag in t.tags for tag in tags)]
        
        return templates
    
    async def recommend_templates(
        self,
        user_context: Dict[str, Any],
        limit: int = 5
    ) -> List[TemplateRecommendation]:
        """Recommend templates based on user context"""
        recommendations = []
        
        # Simple recommendation logic - can be enhanced with ML
        for template in self.templates.values():
            score = 0.0
            reasoning_parts = []
            
            # Match document type
            if user_context.get("document_type") == template.document_type:
                score += 0.4
                reasoning_parts.append("matches your document type")
            
            # Match industry
            if user_context.get("industry") == template.industry:
                score += 0.3
                reasoning_parts.append("tailored for your industry")
            
            # Match complexity
            if user_context.get("complexity") == template.complexity:
                score += 0.2
                reasoning_parts.append("matches your experience level")
            
            # Match tags
            user_tags = user_context.get("tags", [])
            if user_tags and any(tag in template.tags for tag in user_tags):
                score += 0.1
                reasoning_parts.append("includes relevant features")
            
            if score > 0:
                recommendations.append(TemplateRecommendation(
                    template=template,
                    score=score,
                    reasoning="This template " + " and ".join(reasoning_parts),
                    customization_suggestions=self._generate_customization_suggestions(template, user_context)
                ))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:limit]
    
    def _generate_customization_suggestions(
        self,
        template: DocumentTemplate,
        user_context: Dict[str, Any]
    ) -> List[str]:
        """Generate customization suggestions for a template"""
        suggestions = []
        
        # Industry-specific suggestions
        if user_context.get("industry"):
            suggestions.append(f"Customize for {user_context['industry']} industry best practices")
        
        # Complexity-based suggestions
        if template.complexity == TemplateComplexity.BASIC:
            suggestions.append("Consider adding more detailed sections for comprehensive coverage")
        elif template.complexity == TemplateComplexity.ADVANCED:
            suggestions.append("Simplify sections if targeting a general audience")
        
        # Document type suggestions
        if template.document_type == DocumentType.BUSINESS_PLAN:
            suggestions.append("Include specific financial projections for your market")
        elif template.document_type == DocumentType.MARKETING_PROPOSAL:
            suggestions.append("Add competitor analysis and market research data")
        
        return suggestions
    
    async def generate_smart_suggestions(
        self,
        template_id: str,
        field_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> List[SmartSuggestion]:
        """Generate AI-powered smart suggestions for template fields"""
        template = await self.get_template(template_id)
        if not template:
            return []
        
        suggestions = []
        
        for field in template.fields:
            if field.name in field_data:
                continue  # Skip fields that already have data
            
            suggestion = await self._generate_field_suggestion(field, field_data, user_context)
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions
    
    async def _generate_field_suggestion(
        self,
        field: TemplateField,
        field_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Optional[SmartSuggestion]:
        """Generate suggestion for a specific field"""
        # This would integrate with AI models for intelligent suggestions
        # For now, providing basic logic-based suggestions
        
        if field.type == "select" and field.options:
            # Suggest most common option for the field
            suggestion = field.options[0]
            return SmartSuggestion(
                field_name=field.name,
                suggestion=suggestion,
                confidence=0.7,
                reasoning=f"Most commonly selected option for {field.label}",
                alternatives=field.options[1:3]
            )
        
        elif field.type == "text":
            # Generate contextual suggestion based on field name
            if "name" in field.name.lower():
                suggestion = f"Enter your {field.label.lower()}"
            elif "description" in field.name.lower():
                suggestion = f"Describe your {field.label.lower()}"
            else:
                suggestion = f"Provide details about {field.label.lower()}"
            
            return SmartSuggestion(
                field_name=field.name,
                suggestion=suggestion,
                confidence=0.6,
                reasoning=f"Contextual suggestion based on field purpose",
                alternatives=[]
            )
        
        return None
    
    async def create_custom_template(
        self,
        template_data: Dict[str, Any],
        user_id: str
    ) -> DocumentTemplate:
        """Create a custom template"""
        template_id = f"custom_{user_id}_{len(self.templates)}"
        
        template = DocumentTemplate(
            id=template_id,
            name=template_data["name"],
            description=template_data["description"],
            document_type=DocumentType(template_data["document_type"]),
            complexity=TemplateComplexity(template_data.get("complexity", "intermediate")),
            fields=[TemplateField(**field) for field in template_data["fields"]],
            content_structure=template_data["content_structure"],
            ai_prompts=template_data.get("ai_prompts", {}),
            metadata=template_data.get("metadata", {}),
            tags=template_data.get("tags", []),
            version="1.0.0"
        )
        
        self.templates[template_id] = template
        logger.info(f"Created custom template {template_id} for user {user_id}")
        
        return template
    
    async def update_template(
        self,
        template_id: str,
        updates: Dict[str, Any],
        user_id: str
    ) -> Optional[DocumentTemplate]:
        """Update an existing template"""
        template = self.templates.get(template_id)
        if not template:
            return None
        
        # Update fields
        for key, value in updates.items():
            if hasattr(template, key):
                setattr(template, key, value)
        
        template.updated_at = datetime.utcnow()
        template.version = self._increment_version(template.version)
        
        logger.info(f"Updated template {template_id} by user {user_id}")
        return template
    
    def _increment_version(self, version: str) -> str:
        """Increment version number"""
        parts = version.split(".")
        if len(parts) >= 2:
            minor = int(parts[1]) + 1
            return f"{parts[0]}.{minor}.0"
        return "1.0.0"
    
    async def get_template_statistics(self) -> Dict[str, Any]:
        """Get template usage statistics"""
        return {
            "total_templates": len(self.templates),
            "templates_by_type": {
                doc_type.value: len([t for t in self.templates.values() if t.document_type == doc_type])
                for doc_type in DocumentType
            },
            "templates_by_complexity": {
                complexity.value: len([t for t in self.templates.values() if t.complexity == complexity])
                for complexity in TemplateComplexity
            },
            "templates_by_industry": {
                industry.value: len([t for t in self.templates.values() if t.industry == industry])
                for industry in IndustryType
            },
            "average_fields_per_template": sum(len(t.fields) for t in self.templates.values()) / len(self.templates) if self.templates else 0
        }


# Global template manager instance
template_manager = DocumentTemplateManager()

















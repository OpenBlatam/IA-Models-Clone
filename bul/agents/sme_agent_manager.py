"""
SME Agent Manager
=================

Manages specialized agents for different SME business areas.
Each agent is optimized for specific business functions and document types.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime
from .bul_engine import BusinessArea, DocumentType, DocumentRequest, DocumentResponse

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of SME agents"""
    MARKETING_SPECIALIST = "marketing_specialist"
    SALES_EXPERT = "sales_expert"
    OPERATIONS_MANAGER = "operations_manager"
    HR_CONSULTANT = "hr_consultant"
    FINANCIAL_ADVISOR = "financial_advisor"
    LEGAL_CONSULTANT = "legal_consultant"
    TECH_SPECIALIST = "tech_specialist"
    CONTENT_STRATEGIST = "content_strategist"
    BUSINESS_STRATEGIST = "business_strategist"
    CUSTOMER_SERVICE_MANAGER = "customer_service_manager"

@dataclass
class AgentCapability:
    """Agent capability definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    business_areas: List[BusinessArea] = field(default_factory=list)
    document_types: List[DocumentType] = field(default_factory=list)
    expertise_level: float = 0.0
    languages: List[str] = field(default_factory=lambda: ["es", "en"])
    specializations: List[str] = field(default_factory=list)

@dataclass
class SMEAgent:
    """SME Agent definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    agent_type: AgentType = AgentType.BUSINESS_STRATEGIST
    capabilities: List[AgentCapability] = field(default_factory=list)
    experience_years: int = 0
    success_rate: float = 0.0
    total_documents_generated: int = 0
    average_rating: float = 0.0
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)

class SMEAgentManager:
    """
    SME Agent Manager
    
    Manages and coordinates specialized agents for different business areas.
    Routes document generation requests to the most appropriate agent.
    """
    
    def __init__(self):
        self.agents: Dict[str, SMEAgent] = {}
        self.agent_capabilities: Dict[AgentType, List[AgentCapability]] = {}
        self.is_initialized = False
        
        logger.info("SME Agent Manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize the agent manager with default agents"""
        try:
            await self._create_default_agents()
            await self._setup_agent_capabilities()
            self.is_initialized = True
            logger.info("SME Agent Manager fully initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SME Agent Manager: {e}")
            return False
    
    async def _create_default_agents(self):
        """Create default SME agents"""
        agents_data = [
            {
                "name": "María González - Especialista en Marketing",
                "agent_type": AgentType.MARKETING_SPECIALIST,
                "experience_years": 8,
                "success_rate": 0.92,
                "specializations": ["Marketing Digital", "Redes Sociales", "SEO", "Content Marketing"]
            },
            {
                "name": "Carlos Rodríguez - Experto en Ventas",
                "agent_type": AgentType.SALES_EXPERT,
                "experience_years": 10,
                "success_rate": 0.89,
                "specializations": ["Ventas B2B", "CRM", "Negociación", "Funnel de Ventas"]
            },
            {
                "name": "Ana Martínez - Gerente de Operaciones",
                "agent_type": AgentType.OPERATIONS_MANAGER,
                "experience_years": 12,
                "success_rate": 0.94,
                "specializations": ["Procesos", "Logística", "Calidad", "Optimización"]
            },
            {
                "name": "Luis Fernández - Consultor de RRHH",
                "agent_type": AgentType.HR_CONSULTANT,
                "experience_years": 7,
                "success_rate": 0.87,
                "specializations": ["Reclutamiento", "Capacitación", "Políticas", "Cultura Organizacional"]
            },
            {
                "name": "Patricia López - Asesora Financiera",
                "agent_type": AgentType.FINANCIAL_ADVISOR,
                "experience_years": 15,
                "success_rate": 0.96,
                "specializations": ["Finanzas Corporativas", "Presupuestos", "Análisis Financiero", "Inversiones"]
            },
            {
                "name": "Roberto Silva - Consultor Legal",
                "agent_type": AgentType.LEGAL_CONSULTANT,
                "experience_years": 20,
                "success_rate": 0.98,
                "specializations": ["Derecho Corporativo", "Contratos", "Compliance", "Propiedad Intelectual"]
            },
            {
                "name": "Sofia Chen - Especialista en Tecnología",
                "agent_type": AgentType.TECH_SPECIALIST,
                "experience_years": 6,
                "success_rate": 0.91,
                "specializations": ["Desarrollo de Software", "Sistemas", "Ciberseguridad", "Transformación Digital"]
            },
            {
                "name": "Diego Morales - Estratega de Contenido",
                "agent_type": AgentType.CONTENT_STRATEGIST,
                "experience_years": 5,
                "success_rate": 0.88,
                "specializations": ["Content Marketing", "Storytelling", "SEO", "Branding"]
            },
            {
                "name": "Isabella Ruiz - Estratega de Negocios",
                "agent_type": AgentType.BUSINESS_STRATEGIST,
                "experience_years": 18,
                "success_rate": 0.95,
                "specializations": ["Estrategia Corporativa", "Planificación", "Análisis de Mercado", "Innovación"]
            },
            {
                "name": "Miguel Torres - Gerente de Atención al Cliente",
                "agent_type": AgentType.CUSTOMER_SERVICE_MANAGER,
                "experience_years": 9,
                "success_rate": 0.93,
                "specializations": ["Atención al Cliente", "Soporte Técnico", "Retención", "Satisfacción"]
            }
        ]
        
        for agent_data in agents_data:
            agent = SMEAgent(**agent_data)
            self.agents[agent.id] = agent
            logger.info(f"Created agent: {agent.name}")
    
    async def _setup_agent_capabilities(self):
        """Setup capabilities for each agent type"""
        capabilities_map = {
            AgentType.MARKETING_SPECIALIST: [
                AgentCapability(
                    name="Estrategia de Marketing Digital",
                    description="Desarrollo de estrategias de marketing digital para PYMEs",
                    business_areas=[BusinessArea.MARKETING, BusinessArea.CONTENT],
                    document_types=[DocumentType.MARKETING_STRATEGY, DocumentType.CONTENT_STRATEGY],
                    expertise_level=0.95,
                    specializations=["SEO", "SEM", "Social Media", "Email Marketing"]
                ),
                AgentCapability(
                    name="Análisis de Mercado",
                    description="Análisis de mercado y competencia",
                    business_areas=[BusinessArea.MARKETING, BusinessArea.STRATEGY],
                    document_types=[DocumentType.BUSINESS_PLAN, DocumentType.STRATEGIC_PLAN],
                    expertise_level=0.88,
                    specializations=["Market Research", "Competitive Analysis", "Customer Personas"]
                )
            ],
            AgentType.SALES_EXPERT: [
                AgentCapability(
                    name="Estrategia de Ventas",
                    description="Desarrollo de estrategias y procesos de ventas",
                    business_areas=[BusinessArea.SALES],
                    document_types=[DocumentType.SALES_PROPOSAL, DocumentType.BUSINESS_PLAN],
                    expertise_level=0.92,
                    specializations=["Sales Funnel", "CRM", "Lead Generation", "Negotiation"]
                )
            ],
            AgentType.OPERATIONS_MANAGER: [
                AgentCapability(
                    name="Optimización de Procesos",
                    description="Mejora de procesos operativos y eficiencia",
                    business_areas=[BusinessArea.OPERATIONS],
                    document_types=[DocumentType.OPERATIONAL_MANUAL, DocumentType.BUSINESS_PLAN],
                    expertise_level=0.94,
                    specializations=["Process Mapping", "Lean", "Six Sigma", "Automation"]
                )
            ],
            AgentType.HR_CONSULTANT: [
                AgentCapability(
                    name="Gestión de Recursos Humanos",
                    description="Políticas y procesos de RRHH",
                    business_areas=[BusinessArea.HR],
                    document_types=[DocumentType.HR_POLICY, DocumentType.OPERATIONAL_MANUAL],
                    expertise_level=0.87,
                    specializations=["Recruitment", "Training", "Performance Management", "Employee Relations"]
                )
            ],
            AgentType.FINANCIAL_ADVISOR: [
                AgentCapability(
                    name="Análisis Financiero",
                    description="Análisis financiero y planificación",
                    business_areas=[BusinessArea.FINANCE],
                    document_types=[DocumentType.FINANCIAL_REPORT, DocumentType.BUSINESS_PLAN],
                    expertise_level=0.96,
                    specializations=["Financial Planning", "Budgeting", "Investment Analysis", "Risk Management"]
                )
            ],
            AgentType.LEGAL_CONSULTANT: [
                AgentCapability(
                    name="Asesoría Legal",
                    description="Asesoría legal y contractual",
                    business_areas=[BusinessArea.LEGAL],
                    document_types=[DocumentType.LEGAL_CONTRACT, DocumentType.HR_POLICY],
                    expertise_level=0.98,
                    specializations=["Corporate Law", "Contracts", "Compliance", "Intellectual Property"]
                )
            ],
            AgentType.TECH_SPECIALIST: [
                AgentCapability(
                    name="Especificaciones Técnicas",
                    description="Desarrollo de especificaciones técnicas",
                    business_areas=[BusinessArea.TECHNICAL],
                    document_types=[DocumentType.TECHNICAL_SPECIFICATION, DocumentType.OPERATIONAL_MANUAL],
                    expertise_level=0.91,
                    specializations=["Software Development", "System Architecture", "Security", "Cloud Computing"]
                )
            ],
            AgentType.CONTENT_STRATEGIST: [
                AgentCapability(
                    name="Estrategia de Contenido",
                    description="Desarrollo de estrategias de contenido",
                    business_areas=[BusinessArea.CONTENT, BusinessArea.MARKETING],
                    document_types=[DocumentType.CONTENT_STRATEGY, DocumentType.MARKETING_STRATEGY],
                    expertise_level=0.88,
                    specializations=["Content Marketing", "SEO", "Social Media", "Brand Storytelling"]
                )
            ],
            AgentType.BUSINESS_STRATEGIST: [
                AgentCapability(
                    name="Planificación Estratégica",
                    description="Desarrollo de planes estratégicos",
                    business_areas=[BusinessArea.STRATEGY],
                    document_types=[DocumentType.STRATEGIC_PLAN, DocumentType.BUSINESS_PLAN],
                    expertise_level=0.95,
                    specializations=["Strategic Planning", "Market Analysis", "Business Development", "Innovation"]
                )
            ],
            AgentType.CUSTOMER_SERVICE_MANAGER: [
                AgentCapability(
                    name="Gestión de Atención al Cliente",
                    description="Estrategias de atención y servicio al cliente",
                    business_areas=[BusinessArea.CUSTOMER_SERVICE],
                    document_types=[DocumentType.CUSTOMER_SERVICE_GUIDE, DocumentType.OPERATIONAL_MANUAL],
                    expertise_level=0.93,
                    specializations=["Customer Support", "Service Excellence", "Retention", "Satisfaction"]
                )
            ]
        }
        
        self.agent_capabilities = capabilities_map
        logger.info("Agent capabilities configured")
    
    async def get_best_agent(self, request: DocumentRequest) -> Optional[SMEAgent]:
        """Get the best agent for a specific document request"""
        if not self.is_initialized:
            await self.initialize()
        
        best_agent = None
        best_score = 0.0
        
        for agent in self.agents.values():
            if not agent.is_active:
                continue
            
            score = await self._calculate_agent_score(agent, request)
            if score > best_score:
                best_score = score
                best_agent = agent
        
        if best_agent:
            # Update agent usage statistics
            best_agent.last_used = datetime.now()
            best_agent.total_documents_generated += 1
        
        return best_agent
    
    async def _calculate_agent_score(self, agent: SMEAgent, request: DocumentRequest) -> float:
        """Calculate how well an agent matches a request with optimized scoring"""
        score = 0.0
        
        # Check if agent has capabilities for the business area
        agent_caps = self.agent_capabilities.get(agent.agent_type, [])
        
        # Business area matching (40% weight)
        business_area_match = False
        for cap in agent_caps:
            if request.business_area in cap.business_areas:
                score += cap.expertise_level * 0.4
                business_area_match = True
                break
        
        # Document type matching (30% weight)
        document_type_match = False
        for cap in agent_caps:
            if request.document_type in cap.document_types:
                score += cap.expertise_level * 0.3
                document_type_match = True
                break
        
        # Agent performance factors (20% weight)
        performance_score = (agent.success_rate * 0.6 + 
                           min(agent.experience_years / 20.0, 1.0) * 0.4)
        score += performance_score * 0.2
        
        # Availability bonus (10% weight)
        availability_bonus = 0.1 if agent.is_active else 0.0
        score += availability_bonus
        
        # Penalty for no matches
        if not business_area_match and not document_type_match:
            score *= 0.1  # Heavy penalty for no relevant capabilities
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def get_agent_by_type(self, agent_type: AgentType) -> Optional[SMEAgent]:
        """Get an agent by type"""
        for agent in self.agents.values():
            if agent.agent_type == agent_type and agent.is_active:
                return agent
        return None
    
    async def get_all_agents(self) -> List[SMEAgent]:
        """Get all active agents"""
        return [agent for agent in self.agents.values() if agent.is_active]
    
    async def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about agents"""
        total_agents = len(self.agents)
        active_agents = len([a for a in self.agents.values() if a.is_active])
        total_documents = sum(agent.total_documents_generated for agent in self.agents.values())
        avg_success_rate = sum(agent.success_rate for agent in self.agents.values()) / total_agents if total_agents > 0 else 0
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "total_documents_generated": total_documents,
            "average_success_rate": avg_success_rate,
            "agent_types": list(set(agent.agent_type.value for agent in self.agents.values())),
            "is_initialized": self.is_initialized
        }

# Global agent manager instance
_agent_manager: Optional[SMEAgentManager] = None

async def get_global_agent_manager() -> SMEAgentManager:
    """Get the global agent manager instance"""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = SMEAgentManager()
        await _agent_manager.initialize()
    return _agent_manager




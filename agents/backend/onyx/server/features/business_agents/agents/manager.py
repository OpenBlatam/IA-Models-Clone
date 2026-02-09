"""
Business Agent Manager
======================

Centralized management system for business agents.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .definitions import BusinessArea, BusinessAgent, AgentCapability, AgentExecution, ExecutionStatus
from .capabilities import capability_registry

logger = logging.getLogger(__name__)

class BusinessAgentManager:
    """Centralized management system for all business area agents."""
    
    def __init__(self):
        self.agents: Dict[str, BusinessAgent] = {}
        self.execution_history: Dict[str, AgentExecution] = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize built-in business agents."""
        try:
            # Marketing Agent
            marketing_agent = BusinessAgent(
                id="marketing_001",
                name="Marketing Strategy Agent",
                business_area=BusinessArea.MARKETING,
                description="Handles marketing strategy, campaign planning, and brand management",
                capabilities=[
                    capability_registry.get_capability("campaign_planning").get_capability_definition()
                ],
                metadata={
                    "version": "1.0.0",
                    "author": "system",
                    "tags": ["marketing", "strategy", "campaigns"]
                }
            )
            self.agents[marketing_agent.id] = marketing_agent
            
            # Sales Agent
            sales_agent = BusinessAgent(
                id="sales_001",
                name="Sales Operations Agent",
                business_area=BusinessArea.SALES,
                description="Manages sales processes, lead qualification, and customer acquisition",
                capabilities=[
                    capability_registry.get_capability("lead_qualification").get_capability_definition()
                ],
                metadata={
                    "version": "1.0.0",
                    "author": "system",
                    "tags": ["sales", "leads", "qualification"]
                }
            )
            self.agents[sales_agent.id] = sales_agent
            
            # Finance Agent
            finance_agent = BusinessAgent(
                id="finance_001",
                name="Financial Analysis Agent",
                business_area=BusinessArea.FINANCE,
                description="Provides financial analysis, budget planning, and reporting",
                capabilities=[
                    capability_registry.get_capability("budget_analysis").get_capability_definition()
                ],
                metadata={
                    "version": "1.0.0",
                    "author": "system",
                    "tags": ["finance", "analysis", "budget"]
                }
            )
            self.agents[finance_agent.id] = finance_agent
            
            logger.info(f"Initialized {len(self.agents)} business agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
    
    def get_agent(self, agent_id: str) -> Optional[BusinessAgent]:
        """Get a specific agent by ID."""
        return self.agents.get(agent_id)
    
    def list_agents(
        self, 
        business_area: Optional[BusinessArea] = None,
        is_active: Optional[bool] = None
    ) -> List[BusinessAgent]:
        """List all agents with optional filters."""
        agents = list(self.agents.values())
        
        if business_area:
            agents = [agent for agent in agents if agent.business_area == business_area]
        
        if is_active is not None:
            agents = [agent for agent in agents if agent.is_active == is_active]
        
        return agents
    
    def get_agent_capabilities(self, agent_id: str) -> List[AgentCapability]:
        """Get capabilities for a specific agent."""
        agent = self.get_agent(agent_id)
        if not agent:
            return []
        
        return agent.capabilities
    
    async def execute_agent_capability(
        self,
        agent_id: str,
        capability_name: str,
        inputs: Dict[str, Any],
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a specific agent capability."""
        start_time = datetime.now()
        
        try:
            # Validate agent exists
            agent = self.get_agent(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            if not agent.is_active:
                raise ValueError(f"Agent {agent_id} is not active")
            
            # Validate capability exists
            capability = None
            for cap in agent.capabilities:
                if cap.name == capability_name:
                    capability = cap
                    break
            
            if not capability:
                raise ValueError(f"Capability {capability_name} not found for agent {agent_id}")
            
            # Create execution record
            execution_id = f"{agent_id}_{capability_name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
            execution = AgentExecution(
                execution_id=execution_id,
                agent_id=agent_id,
                capability_name=capability_name,
                inputs=inputs,
                outputs={},
                status=ExecutionStatus.RUNNING.value,
                start_time=start_time
            )
            self.execution_history[execution_id] = execution
            
            # Execute capability
            result = await capability_registry.execute_capability(
                capability_name, inputs, parameters
            )
            
            # Update execution record
            end_time = datetime.now()
            execution.end_time = end_time
            execution.duration = (end_time - start_time).total_seconds()
            execution.outputs = result.get("result", {})
            execution.status = ExecutionStatus.COMPLETED.value if result.get("status") == "completed" else ExecutionStatus.FAILED.value
            execution.error_message = result.get("error")
            
            logger.info(f"Executed capability {capability_name} for agent {agent_id} in {execution.duration:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute capability {capability_name} for agent {agent_id}: {str(e)}")
            
            # Update execution record with error
            if execution_id in self.execution_history:
                execution = self.execution_history[execution_id]
                execution.end_time = datetime.now()
                execution.duration = (execution.end_time - execution.start_time).total_seconds()
                execution.status = ExecutionStatus.FAILED.value
                execution.error_message = str(e)
            
            raise
    
    def get_business_areas(self) -> List[BusinessArea]:
        """Get all available business areas."""
        return list(BusinessArea)
    
    def get_agents_by_business_area(self, business_area: BusinessArea) -> List[BusinessAgent]:
        """Get all agents for a specific business area."""
        return [agent for agent in self.agents.values() if agent.business_area == business_area]
    
    def get_execution_history(
        self, 
        agent_id: Optional[str] = None,
        capability_name: Optional[str] = None,
        status: Optional[ExecutionStatus] = None
    ) -> List[AgentExecution]:
        """Get execution history with optional filters."""
        executions = list(self.execution_history.values())
        
        if agent_id:
            executions = [exec for exec in executions if exec.agent_id == agent_id]
        
        if capability_name:
            executions = [exec for exec in executions if exec.capability_name == capability_name]
        
        if status:
            executions = [exec for exec in executions if exec.status == status.value]
        
        return sorted(executions, key=lambda x: x.start_time, reverse=True)
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        total_agents = len(self.agents)
        active_agents = len([agent for agent in self.agents.values() if agent.is_active])
        
        # Count by business area
        area_counts = {}
        for agent in self.agents.values():
            area = agent.business_area.value
            area_counts[area] = area_counts.get(area, 0) + 1
        
        # Execution statistics
        total_executions = len(self.execution_history)
        successful_executions = len([
            exec for exec in self.execution_history.values()
            if exec.status == ExecutionStatus.COMPLETED.value
        ])
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "business_areas": area_counts,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0
        }
    
    async def create_agent(self, agent: BusinessAgent) -> bool:
        """Create a new agent."""
        try:
            if agent.id in self.agents:
                raise ValueError(f"Agent with ID {agent.id} already exists")
            
            self.agents[agent.id] = agent
            logger.info(f"Created agent: {agent.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create agent: {str(e)}")
            return False
    
    async def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing agent."""
        try:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
            
            agent.updated_at = datetime.now()
            logger.info(f"Updated agent: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent {agent_id}: {str(e)}")
            return False
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        try:
            if agent_id not in self.agents:
                return False
            
            del self.agents[agent_id]
            logger.info(f"Deleted agent: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete agent {agent_id}: {str(e)}")
            return False

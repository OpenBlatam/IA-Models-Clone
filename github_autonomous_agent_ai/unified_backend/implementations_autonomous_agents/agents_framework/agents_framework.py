"""
Agents: An Open-source Framework
==================================

Paper: "Agents: An Open-source Framework"

Key concepts:
- Open-source agent framework
- Modular architecture
- Extensible design
- Agent composition
- Framework utilities
"""

from typing import Dict, Any, Optional, List, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class FrameworkComponent(Enum):
    """Framework components."""
    AGENT_CORE = "agent_core"
    MEMORY = "memory"
    TOOLS = "tools"
    COMMUNICATION = "communication"
    ORCHESTRATION = "orchestration"
    MONITORING = "monitoring"


class AgentType(Enum):
    """Types of agents."""
    SIMPLE = "simple"
    REACTIVE = "reactive"
    DELIBERATIVE = "deliberative"
    HYBRID = "hybrid"
    CUSTOM = "custom"


@dataclass
class FrameworkConfig:
    """Framework configuration."""
    config_id: str
    components: List[FrameworkComponent]
    agent_type: AgentType
    settings: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentInstance:
    """An agent instance in the framework."""
    instance_id: str
    agent_class: str
    agent_type: AgentType
    config: Dict[str, Any]
    status: str
    created_at: datetime = field(default_factory=datetime.now)


class AgentsFramework:
    """
    Open-source framework for building agents.
    
    Provides modular components and utilities.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize agents framework.
        
        Args:
            config: Framework configuration
        """
        self.config = config or {}
        self.components: Dict[FrameworkComponent, Any] = {}
        self.agent_instances: Dict[str, AgentInstance] = {}
        self.agent_registry: Dict[str, Type[BaseAgent]] = {}
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize framework components."""
        for component in FrameworkComponent:
            self.components[component] = {
                "active": True,
                "version": "1.0.0",
                "config": {}
            }
    
    def register_agent(
        self,
        agent_name: str,
        agent_class: Type[BaseAgent],
        agent_type: AgentType = AgentType.CUSTOM
    ):
        """
        Register an agent class.
        
        Args:
            agent_name: Name of the agent
            agent_class: Agent class
            agent_type: Type of agent
        """
        self.agent_registry[agent_name] = agent_class
    
    def create_agent(
        self,
        agent_name: str,
        instance_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseAgent]:
        """
        Create an agent instance.
        
        Args:
            agent_name: Registered agent name
            instance_name: Instance name
            config: Agent configuration
            
        Returns:
            Agent instance or None
        """
        if agent_name not in self.agent_registry:
            return None
        
        agent_class = self.agent_registry[agent_name]
        
        try:
            agent = agent_class(instance_name, config or {})
            
            # Create instance record
            instance = AgentInstance(
                instance_id=f"inst_{datetime.now().timestamp()}",
                agent_class=agent_name,
                agent_type=AgentType.CUSTOM,
                config=config or {},
                status="created"
            )
            self.agent_instances[instance_name] = instance
            
            return agent
        except Exception as e:
            print(f"Error creating agent: {e}")
            return None
    
    def get_agent(self, instance_name: str) -> Optional[BaseAgent]:
        """
        Get agent instance.
        
        Args:
            instance_name: Instance name
            
        Returns:
            Agent instance or None
        """
        # In production, this would retrieve from storage
        # For now, return None as placeholder
        return None
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agent instances."""
        return [
            {
                "instance_id": inst.instance_id,
                "agent_class": inst.agent_class,
                "agent_type": inst.agent_type.value,
                "status": inst.status,
                "created_at": inst.created_at.isoformat()
            }
            for inst in self.agent_instances.values()
        ]
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get framework status."""
        active_components = [
            comp.value for comp, info in self.components.items()
            if info.get("active", False)
        ]
        
        return {
            "total_components": len(self.components),
            "active_components": active_components,
            "registered_agents": len(self.agent_registry),
            "agent_instances": len(self.agent_instances),
            "framework_version": "1.0.0"
        }
    
    def configure_component(
        self,
        component: FrameworkComponent,
        settings: Dict[str, Any]
    ) -> bool:
        """
        Configure a framework component.
        
        Args:
            component: Component to configure
            settings: Configuration settings
            
        Returns:
            True if configured successfully
        """
        if component not in self.components:
            return False
        
        self.components[component]["config"].update(settings)
        return True
    
    def get_component_config(self, component: FrameworkComponent) -> Optional[Dict[str, Any]]:
        """Get component configuration."""
        if component not in self.components:
            return None
        
        return self.components[component].get("config", {})




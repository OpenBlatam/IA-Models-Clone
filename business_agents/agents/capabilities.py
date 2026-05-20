"""
Agent Capabilities
==================

Capability system for business agents.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from datetime import datetime

from .definitions import AgentCapability, AgentExecution, ExecutionStatus

logger = logging.getLogger(__name__)

class BaseCapability(ABC):
    """Base class for agent capabilities."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.input_types: List[str] = []
        self.output_types: List[str] = []
        self.parameters: Dict[str, Any] = {}
        self.estimated_duration: int = 300  # 5 minutes default
        self.required_permissions: List[str] = []
        self.tags: List[str] = []
    
    @abstractmethod
    async def execute(self, inputs: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the capability with given inputs and parameters."""
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate input data."""
        # Basic validation - can be overridden by subclasses
        return True
    
    def get_capability_definition(self) -> AgentCapability:
        """Get the capability definition."""
        return AgentCapability(
            name=self.name,
            description=self.description,
            input_types=self.input_types,
            output_types=self.output_types,
            parameters=self.parameters,
            estimated_duration=self.estimated_duration,
            required_permissions=self.required_permissions,
            tags=self.tags
        )

class MarketingCapability(BaseCapability):
    """Marketing-related capabilities."""
    
    def __init__(self):
        super().__init__(
            name="campaign_planning",
            description="Plan and create marketing campaigns"
        )
        self.input_types = ["target_audience", "budget", "objectives"]
        self.output_types = ["campaign_plan", "timeline", "budget_breakdown"]
        self.estimated_duration = 600  # 10 minutes
        self.tags = ["marketing", "planning", "campaign"]
    
    async def execute(self, inputs: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute campaign planning."""
        try:
            # Simulate campaign planning logic
            target_audience = inputs.get("target_audience", "general")
            budget = inputs.get("budget", 10000)
            objectives = inputs.get("objectives", "increase_awareness")
            
            # Mock campaign plan generation
            campaign_plan = {
                "target_audience": target_audience,
                "budget": budget,
                "objectives": objectives,
                "channels": ["social_media", "email", "content_marketing"],
                "timeline": "3 months",
                "expected_reach": budget * 10,
                "success_metrics": ["impressions", "clicks", "conversions"]
            }
            
            return {
                "status": "completed",
                "result": campaign_plan,
                "execution_time": 0.5
            }
            
        except Exception as e:
            logger.error(f"Campaign planning failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }

class SalesCapability(BaseCapability):
    """Sales-related capabilities."""
    
    def __init__(self):
        super().__init__(
            name="lead_qualification",
            description="Qualify and score sales leads"
        )
        self.input_types = ["lead_data", "qualification_criteria"]
        self.output_types = ["lead_score", "qualification_status", "recommendations"]
        self.estimated_duration = 300  # 5 minutes
        self.tags = ["sales", "lead_management", "qualification"]
    
    async def execute(self, inputs: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute lead qualification."""
        try:
            lead_data = inputs.get("lead_data", {})
            criteria = inputs.get("qualification_criteria", {})
            
            # Mock lead scoring logic
            score = 0
            if lead_data.get("company_size", 0) > 100:
                score += 30
            if lead_data.get("budget", 0) > 50000:
                score += 40
            if lead_data.get("timeline", "long") == "short":
                score += 30
            
            qualification_status = "qualified" if score >= 70 else "unqualified"
            
            return {
                "status": "completed",
                "result": {
                    "lead_score": score,
                    "qualification_status": qualification_status,
                    "recommendations": [
                        "Follow up within 24 hours" if score >= 70 else "Add to nurture campaign",
                        "Schedule demo call" if score >= 80 else "Send educational content"
                    ]
                },
                "execution_time": 0.3
            }
            
        except Exception as e:
            logger.error(f"Lead qualification failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }

class FinanceCapability(BaseCapability):
    """Finance-related capabilities."""
    
    def __init__(self):
        super().__init__(
            name="budget_analysis",
            description="Analyze budget and financial data"
        )
        self.input_types = ["budget_data", "expenses", "revenue"]
        self.output_types = ["analysis_report", "recommendations", "forecast"]
        self.estimated_duration = 900  # 15 minutes
        self.tags = ["finance", "analysis", "budget"]
    
    async def execute(self, inputs: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute budget analysis."""
        try:
            budget_data = inputs.get("budget_data", {})
            expenses = inputs.get("expenses", 0)
            revenue = inputs.get("revenue", 0)
            
            # Mock financial analysis
            profit = revenue - expenses
            profit_margin = (profit / revenue * 100) if revenue > 0 else 0
            
            analysis = {
                "total_revenue": revenue,
                "total_expenses": expenses,
                "net_profit": profit,
                "profit_margin": round(profit_margin, 2),
                "budget_variance": budget_data.get("variance", 0),
                "recommendations": [
                    "Optimize operational costs" if profit_margin < 10 else "Consider expansion",
                    "Review pricing strategy" if profit_margin < 5 else "Maintain current approach"
                ]
            }
            
            return {
                "status": "completed",
                "result": analysis,
                "execution_time": 0.8
            }
            
        except Exception as e:
            logger.error(f"Budget analysis failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }

class CapabilityRegistry:
    """Registry for managing agent capabilities."""
    
    def __init__(self):
        self.capabilities: Dict[str, BaseCapability] = {}
        self._register_builtin_capabilities()
    
    def _register_builtin_capabilities(self):
        """Register built-in capabilities."""
        builtin_capabilities = [
            MarketingCapability(),
            SalesCapability(),
            FinanceCapability()
        ]
        
        for capability in builtin_capabilities:
            self.register_capability(capability)
    
    def register_capability(self, capability: BaseCapability):
        """Register a new capability."""
        self.capabilities[capability.name] = capability
        logger.info(f"Registered capability: {capability.name}")
    
    def get_capability(self, name: str) -> Optional[BaseCapability]:
        """Get a capability by name."""
        return self.capabilities.get(name)
    
    def list_capabilities(self) -> List[BaseCapability]:
        """List all registered capabilities."""
        return list(self.capabilities.values())
    
    def get_capabilities_by_tag(self, tag: str) -> List[BaseCapability]:
        """Get capabilities by tag."""
        return [
            cap for cap in self.capabilities.values()
            if tag in cap.tags
        ]
    
    async def execute_capability(
        self, 
        name: str, 
        inputs: Dict[str, Any], 
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a capability."""
        capability = self.get_capability(name)
        if not capability:
            return {
                "status": "failed",
                "error": f"Capability '{name}' not found"
            }
        
        try:
            # Validate inputs
            if not capability.validate_inputs(inputs):
                return {
                    "status": "failed",
                    "error": "Invalid inputs"
                }
            
            # Execute capability
            result = await capability.execute(inputs, parameters)
            return result
            
        except Exception as e:
            logger.error(f"Capability execution failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }

# Global capability registry
capability_registry = CapabilityRegistry()

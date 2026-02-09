"""
Cost Tracking Module - Track and estimate costs.

Provides:
- Cost calculation
- Resource usage tracking
- Cost estimation
- Budget management
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Resource type."""
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ResourceUsage:
    """Resource usage record."""
    resource_type: ResourceType
    amount: float
    unit: str
    duration_seconds: float
    cost_per_unit: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def calculate_cost(self) -> float:
        """Calculate cost for this usage."""
        return self.amount * self.duration_seconds * self.cost_per_unit


@dataclass
class CostRecord:
    """Cost record."""
    id: str
    model_name: str
    benchmark_name: str
    resource_usages: List[ResourceUsage] = field(default_factory=list)
    total_cost: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_total(self) -> float:
        """Calculate total cost."""
        self.total_cost = sum(usage.calculate_cost() for usage in self.resource_usages)
        return self.total_cost


class CostTracker:
    """Cost tracking system."""
    
    def __init__(self, pricing_config: Optional[Dict[str, float]] = None):
        """
        Initialize cost tracker.
        
        Args:
            pricing_config: Pricing configuration per resource type
        """
        self.pricing_config = pricing_config or {
            ResourceType.GPU: 0.50,  # $ per GPU-hour
            ResourceType.CPU: 0.10,  # $ per CPU-hour
            ResourceType.MEMORY: 0.01,  # $ per GB-hour
            ResourceType.STORAGE: 0.05,  # $ per GB-month
            ResourceType.NETWORK: 0.01,  # $ per GB
        }
        self.records: List[CostRecord] = []
        self.budget: Optional[float] = None
        self.budget_alert_threshold: float = 0.8  # Alert at 80% of budget
    
    def record_usage(
        self,
        model_name: str,
        benchmark_name: str,
        resource_type: ResourceType,
        amount: float,
        duration_seconds: float,
        unit: str = "hour",
    ) -> CostRecord:
        """
        Record resource usage.
        
        Args:
            model_name: Model name
            benchmark_name: Benchmark name
            resource_type: Resource type
            amount: Resource amount
            duration_seconds: Duration in seconds
            unit: Unit of measurement
            
        Returns:
            Cost record
        """
        cost_per_unit = self.pricing_config.get(resource_type, 0.0)
        
        usage = ResourceUsage(
            resource_type=resource_type,
            amount=amount,
            unit=unit,
            duration_seconds=duration_seconds,
            cost_per_unit=cost_per_unit,
        )
        
        # Find or create cost record
        record_id = f"{model_name}_{benchmark_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        record = CostRecord(
            id=record_id,
            model_name=model_name,
            benchmark_name=benchmark_name,
        )
        
        record.resource_usages.append(usage)
        record.calculate_total()
        
        self.records.append(record)
        
        logger.info(f"Recorded usage: {model_name} - {benchmark_name} - ${record.total_cost:.4f}")
        
        # Check budget
        if self.budget:
            total_spent = self.get_total_cost()
            if total_spent >= self.budget * self.budget_alert_threshold:
                logger.warning(
                    f"Budget alert: {total_spent:.2f} / {self.budget:.2f} "
                    f"({total_spent/self.budget*100:.1f}%)"
                )
        
        return record
    
    def estimate_cost(
        self,
        model_name: str,
        benchmark_name: str,
        estimated_duration_seconds: float,
        gpu_count: int = 1,
        memory_gb: float = 16.0,
    ) -> float:
        """
        Estimate cost for a benchmark run.
        
        Args:
            model_name: Model name
            benchmark_name: Benchmark name
            estimated_duration_seconds: Estimated duration
            gpu_count: Number of GPUs
            memory_gb: Memory in GB
            
        Returns:
            Estimated cost
        """
        gpu_cost = (
            gpu_count *
            (estimated_duration_seconds / 3600.0) *
            self.pricing_config.get(ResourceType.GPU, 0.0)
        )
        
        memory_cost = (
            memory_gb *
            (estimated_duration_seconds / 3600.0) *
            self.pricing_config.get(ResourceType.MEMORY, 0.0)
        )
        
        return gpu_cost + memory_cost
    
    def get_total_cost(
        self,
        model_name: Optional[str] = None,
        benchmark_name: Optional[str] = None,
    ) -> float:
        """
        Get total cost.
        
        Args:
            model_name: Filter by model name
            benchmark_name: Filter by benchmark name
            
        Returns:
            Total cost
        """
        records = self.records
        
        if model_name:
            records = [r for r in records if r.model_name == model_name]
        
        if benchmark_name:
            records = [r for r in records if r.benchmark_name == benchmark_name]
        
        return sum(record.total_cost for record in records)
    
    def get_cost_breakdown(
        self,
        model_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Get cost breakdown by resource type.
        
        Args:
            model_name: Filter by model name
            
        Returns:
            Cost breakdown
        """
        records = self.records
        if model_name:
            records = [r for r in records if r.model_name == model_name]
        
        breakdown = {rt.value: 0.0 for rt in ResourceType}
        
        for record in records:
            for usage in record.resource_usages:
                breakdown[usage.resource_type.value] += usage.calculate_cost()
        
        return breakdown
    
    def set_budget(self, budget: float) -> None:
        """
        Set budget limit.
        
        Args:
            budget: Budget amount
        """
        self.budget = budget
        logger.info(f"Set budget: ${budget:.2f}")
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get budget status."""
        if not self.budget:
            return {"budget_set": False}
        
        total_spent = self.get_total_cost()
        remaining = self.budget - total_spent
        percentage = (total_spent / self.budget) * 100.0
        
        return {
            "budget": self.budget,
            "spent": total_spent,
            "remaining": remaining,
            "percentage": percentage,
            "status": "over_budget" if remaining < 0 else "within_budget",
        }













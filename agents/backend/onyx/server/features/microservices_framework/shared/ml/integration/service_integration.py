"""
Service Integration
Integration utilities for combining services.
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ServiceOrchestrator:
    """Orchestrate multiple services."""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.dependencies: Dict[str, List[str]] = {}
    
    def register_service(self, name: str, service: Any, dependencies: Optional[List[str]] = None):
        """Register a service."""
        self.services[name] = service
        self.dependencies[name] = dependencies or []
        logger.info(f"Service '{name}' registered")
    
    def get_service(self, name: str) -> Any:
        """Get a service."""
        if name not in self.services:
            raise ValueError(f"Service '{name}' not found")
        return self.services[name]
    
    def execute_workflow(self, workflow: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a workflow of service calls."""
        results = {}
        
        for step in workflow:
            service_name = step["service"]
            operation = step["operation"]
            inputs = step.get("inputs", {})
            
            # Resolve inputs from previous results
            resolved_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, str) and value.startswith("$"):
                    # Reference to previous result
                    ref = value[1:]
                    if "." in ref:
                        service, output_key = ref.split(".", 1)
                        resolved_inputs[key] = results.get(service, {}).get(output_key)
                    else:
                        resolved_inputs[key] = results.get(ref)
                else:
                    resolved_inputs[key] = value
            
            # Execute service
            service = self.get_service(service_name)
            if hasattr(service, "execute"):
                result = service.execute(operation, **resolved_inputs)
            else:
                result = getattr(service, operation)(**resolved_inputs)
            
            results[service_name] = result
        
        return results


class PipelineBuilder:
    """Build service pipelines."""
    
    def __init__(self, orchestrator: ServiceOrchestrator):
        self.orchestrator = orchestrator
        self.workflow: List[Dict[str, Any]] = []
    
    def add_step(
        self,
        service_name: str,
        operation: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> "PipelineBuilder":
        """Add a step to the pipeline."""
        self.workflow.append({
            "service": service_name,
            "operation": operation,
            "inputs": inputs or {},
        })
        return self
    
    def build(self) -> callable:
        """Build the pipeline."""
        def pipeline(initial_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Execute the pipeline."""
            # Merge initial inputs into first step
            if initial_inputs and self.workflow:
                self.workflow[0]["inputs"].update(initial_inputs)
            
            return self.orchestrator.execute_workflow(self.workflow)
        
        return pipeline




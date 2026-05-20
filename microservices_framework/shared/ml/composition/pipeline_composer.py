"""
Pipeline Composer
Compose complex ML pipelines from modular components.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """Represents a stage in a pipeline."""
    name: str
    component: Any
    inputs: List[str]
    outputs: List[str]
    config: Optional[Dict[str, Any]] = None


class PipelineComposer:
    """Compose ML pipelines from stages."""
    
    def __init__(self):
        self.stages: List[PipelineStage] = []
        self.execution_order: List[str] = []
    
    def add_stage(
        self,
        name: str,
        component: Any,
        inputs: List[str],
        outputs: List[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> "PipelineComposer":
        """Add a stage to the pipeline."""
        stage = PipelineStage(
            name=name,
            component=component,
            inputs=inputs,
            outputs=outputs,
            config=config,
        )
        self.stages.append(stage)
        return self
    
    def _resolve_execution_order(self) -> List[str]:
        """Resolve execution order based on dependencies."""
        # Simple topological sort
        stage_names = {stage.name for stage in self.stages}
        dependencies = {}
        
        for stage in self.stages:
            dependencies[stage.name] = set(stage.inputs)
        
        execution_order = []
        remaining = set(stage_names)
        
        while remaining:
            # Find stages with no unresolved dependencies
            ready = [
                name for name in remaining
                if not (dependencies[name] & remaining)
            ]
            
            if not ready:
                raise ValueError("Circular dependency detected in pipeline")
            
            # Add ready stages to execution order
            execution_order.extend(sorted(ready))
            remaining -= set(ready)
        
        return execution_order
    
    def compose(self) -> Callable:
        """Compose the pipeline into a callable function."""
        execution_order = self._resolve_execution_order()
        self.execution_order = execution_order
        
        stage_map = {stage.name: stage for stage in self.stages}
        
        def pipeline(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Execute the pipeline."""
            context = inputs.copy()
            
            for stage_name in execution_order:
                stage = stage_map[stage_name]
                
                # Get inputs for this stage
                stage_inputs = {
                    key: context[key] for key in stage.inputs if key in context
                }
                
                # Execute stage
                if hasattr(stage.component, '__call__'):
                    if stage.config:
                        result = stage.component(**stage_inputs, **stage.config)
                    else:
                        result = stage.component(**stage_inputs)
                else:
                    result = stage.component
                
                # Store outputs
                if isinstance(result, dict):
                    context.update(result)
                elif isinstance(result, (list, tuple)) and len(stage.outputs) == len(result):
                    for output_name, output_value in zip(stage.outputs, result):
                        context[output_name] = output_value
                elif len(stage.outputs) == 1:
                    context[stage.outputs[0]] = result
                else:
                    logger.warning(f"Could not map outputs for stage {stage_name}")
            
            # Return only requested outputs
            return context
        
        return pipeline
    
    def visualize(self) -> str:
        """Generate a text visualization of the pipeline."""
        lines = ["Pipeline Structure:"]
        lines.append("=" * 50)
        
        for stage_name in self.execution_order:
            stage = next(s for s in self.stages if s.name == stage_name)
            lines.append(f"\nStage: {stage.name}")
            lines.append(f"  Inputs: {', '.join(stage.inputs)}")
            lines.append(f"  Outputs: {', '.join(stage.outputs)}")
            if stage.config:
                lines.append(f"  Config: {stage.config}")
        
        return "\n".join(lines)


class TrainingPipelineComposer(PipelineComposer):
    """Specialized composer for training pipelines."""
    
    def add_data_loading(self, data_loader, name: str = "data_loader"):
        """Add data loading stage."""
        return self.add_stage(
            name=name,
            component=data_loader,
            inputs=[],
            outputs=["batches"],
        )
    
    def add_preprocessing(self, preprocessor, name: str = "preprocessor"):
        """Add preprocessing stage."""
        return self.add_stage(
            name=name,
            component=preprocessor,
            inputs=["raw_data"],
            outputs=["processed_data"],
        )
    
    def add_training(self, trainer, name: str = "trainer"):
        """Add training stage."""
        return self.add_stage(
            name=name,
            component=trainer,
            inputs=["model", "batches"],
            outputs=["trained_model", "metrics"],
        )
    
    def add_validation(self, evaluator, name: str = "validator"):
        """Add validation stage."""
        return self.add_stage(
            name=name,
            component=evaluator,
            inputs=["model", "validation_data"],
            outputs=["validation_metrics"],
        )


class InferencePipelineComposer(PipelineComposer):
    """Specialized composer for inference pipelines."""
    
    def add_tokenization(self, tokenizer, name: str = "tokenizer"):
        """Add tokenization stage."""
        return self.add_stage(
            name=name,
            component=tokenizer,
            inputs=["text"],
            outputs=["token_ids"],
        )
    
    def add_model_inference(self, model, name: str = "model"):
        """Add model inference stage."""
        return self.add_stage(
            name=name,
            component=model,
            inputs=["token_ids"],
            outputs=["logits"],
        )
    
    def add_postprocessing(self, postprocessor, name: str = "postprocessor"):
        """Add postprocessing stage."""
        return self.add_stage(
            name=name,
            component=postprocessor,
            inputs=["logits"],
            outputs=["output"],
        )




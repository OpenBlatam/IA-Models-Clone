"""Expert ML Routes"""
from fastapi import APIRouter
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from ..services.advanced_transformers_service import AdvancedTransformersService
from ..services.advanced_diffusion_service import AdvancedDiffusionService
from ..services.prompt_engineering_service import PromptEngineeringService
from ..services.model_optimization_service import ModelOptimizationService
from ..services.distributed_training_service import DistributedTrainingService
from ..services.advanced_quantization_service import AdvancedQuantizationService
from ..core.route_helpers import handle_route_errors, track_route_metrics

router = APIRouter()

transformers_service = AdvancedTransformersService()
diffusion_service = AdvancedDiffusionService()
prompt_service = PromptEngineeringService()
optimization_service = ModelOptimizationService()
distributed_service = DistributedTrainingService()
quantization_service = AdvancedQuantizationService()

@router.post("/transformers/gpt", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.create_gpt")
async def create_gpt(model_size: str = "small", vocab_size: int = 50257):
    return transformers_service.create_gpt_model(model_size, vocab_size)

@router.post("/transformers/bert", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.create_bert")
async def create_bert(model_name: str = "bert-base-uncased"):
    return transformers_service.create_bert_model(model_name)

@router.post("/transformers/t5", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.create_t5")
async def create_t5(model_size: str = "base"):
    return transformers_service.create_t5_model(model_size)

@router.post("/diffusion/controlnet", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.create_controlnet")
async def create_controlnet(base_model: str, control_type: str = "canny"):
    return diffusion_service.create_controlnet_pipeline(base_model, control_type)

@router.post("/diffusion/lora", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.create_lora")
async def create_lora_diffusion(base_model: str, lora_rank: int = 4):
    return diffusion_service.create_lora_diffusion(base_model, lora_rank)

@router.post("/prompts/template", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.create_template")
async def create_prompt_template(template_name: str, template: str, variables: List[str]):
    return prompt_service.create_prompt_template(template_name, template, variables)

@router.post("/prompts/optimize", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.optimize_prompt")
async def optimize_prompt(base_prompt: str, task: str, method: str = "few-shot"):
    return prompt_service.optimize_prompt(base_prompt, task, method)

@router.post("/prompts/rag", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.create_rag")
async def create_rag_pipeline(knowledge_base_id: str, retrieval_method: str = "semantic"):
    return prompt_service.create_rag_pipeline(knowledge_base_id, retrieval_method)

@router.post("/optimization/onnx", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.convert_onnx")
async def convert_to_onnx(model_id: str, input_shape: tuple, opset_version: int = 14):
    return optimization_service.convert_to_onnx(model_id, input_shape, opset_version)

@router.post("/optimization/tensorrt", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.optimize_tensorrt")
async def optimize_with_tensorrt(model_id: str, precision: str = "fp16"):
    return optimization_service.optimize_with_tensorrt(model_id, precision)

@router.post("/distributed/horovod", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.setup_horovod")
async def setup_horovod(model_id: str, num_workers: int = 4):
    return distributed_service.setup_horovod(model_id, num_workers)

@router.post("/distributed/deepspeed", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.setup_deepspeed")
async def setup_deepspeed(model_id: str, config: Dict[str, Any]):
    return distributed_service.setup_deepspeed(model_id, config)

@router.post("/distributed/fsdp", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.setup_fsdp")
async def setup_fsdp(model_id: str, sharding_strategy: str = "full_shard"):
    return distributed_service.setup_fsdp(model_id, sharding_strategy)

@router.post("/quantization/qat", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.apply_qat")
async def apply_qat(model_id: str, num_calibration_steps: int = 100):
    return quantization_service.apply_qat(model_id, num_calibration_steps)

@router.post("/quantization/dynamic", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.apply_dynamic_quant")
async def apply_dynamic_quantization(model_id: str):
    return quantization_service.apply_dynamic_quantization(model_id)

@router.post("/quantization/static", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("expert_ml.apply_static_quant")
async def apply_static_quantization(model_id: str):
    return quantization_service.apply_static_quantization(model_id)





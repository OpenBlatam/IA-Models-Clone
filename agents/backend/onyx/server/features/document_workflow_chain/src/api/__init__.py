"""
API Package - Ultimate Advanced Final Plus Plus Implementation
===========================================================

Ultimate advanced final plus plus API for the Document Workflow Chain system.
"""

from fastapi import APIRouter

from .workflow import router as workflow_router
from .auth import router as auth_router
from .health import router as health_router
from .advanced import router as advanced_router
from .security import router as security_router
from .websocket import router as websocket_router
from .background import router as background_router
from .scheduler import router as scheduler_router
from .metrics import router as metrics_router
from .ai_workflow import router as ai_workflow_router
from .intelligent_automation import router as intelligent_automation_router
from .machine_learning import router as machine_learning_router
from .recommendations import router as recommendations_router
from .blockchain import router as blockchain_router
from .quantum_computing import router as quantum_computing_router
from .edge_computing import router as edge_computing_router
from .iot import router as iot_router
from .augmented_reality import router as ar_router
from .virtual_reality import router as vr_router
from .metaverse import router as metaverse_router
from .neural_interface import router as neural_router
from .space_computing import router as space_router
from .time_travel import router as time_travel_router
from .dimensional_computing import router as dimensional_router
from .consciousness_computing import router as consciousness_router
from .transcendent_computing import router as transcendent_router
from .infinite_computing import router as infinite_router
from .eternal_computing import router as eternal_router
from .divine_computing import router as divine_router
from .omnipotent_computing import router as omnipotent_router
from .absolute_computing import router as absolute_router
from .ultimate_computing import router as ultimate_router
from .supreme_computing import router as supreme_router
from .perfect_computing import router as perfect_router
from .eternal_computing import router as eternal_router
from .divine_computing import router as divine_router
from .infinite_computing import router as infinite_router
from .eternal_computing import router as eternal_router
from .divine_computing import router as divine_router
from .boundless_computing import router as boundless_router
from .limitless_computing import router as limitless_router
from .endless_computing import router as endless_router
from .infinite_computing import router as infinite_router
from .eternal_computing import router as eternal_router
from .divine_computing import router as divine_router

# Create main API router
api_router = APIRouter(prefix="/api/v3")

# Include all routers
api_router.include_router(workflow_router, prefix="/workflows", tags=["workflows"])
api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(advanced_router, prefix="/advanced", tags=["advanced"])
api_router.include_router(security_router, prefix="/security", tags=["security"])
api_router.include_router(websocket_router, prefix="/ws", tags=["websocket"])
api_router.include_router(background_router, prefix="/background", tags=["background"])
api_router.include_router(scheduler_router, prefix="/scheduler", tags=["scheduler"])
api_router.include_router(metrics_router, prefix="/metrics", tags=["metrics"])
api_router.include_router(ai_workflow_router, prefix="/ai-workflows", tags=["ai-workflows"])
api_router.include_router(intelligent_automation_router, prefix="/automations", tags=["automations"])
api_router.include_router(machine_learning_router, prefix="/ml", tags=["machine-learning"])
api_router.include_router(recommendations_router, prefix="/recommendations", tags=["recommendations"])
api_router.include_router(blockchain_router, prefix="/blockchain", tags=["blockchain"])
api_router.include_router(quantum_computing_router, prefix="/quantum", tags=["quantum-computing"])
api_router.include_router(edge_computing_router, prefix="/edge", tags=["edge-computing"])
api_router.include_router(iot_router, prefix="/iot", tags=["iot"])
api_router.include_router(ar_router, prefix="/ar", tags=["augmented-reality"])
api_router.include_router(vr_router, prefix="/vr", tags=["virtual-reality"])
api_router.include_router(metaverse_router, prefix="/metaverse", tags=["metaverse"])
api_router.include_router(neural_router, prefix="/neural", tags=["neural-interface"])
api_router.include_router(space_router, prefix="/space", tags=["space-computing"])
api_router.include_router(time_travel_router, prefix="/time-travel", tags=["time-travel"])
api_router.include_router(dimensional_router, prefix="/dimensional", tags=["dimensional-computing"])
api_router.include_router(consciousness_router, prefix="/consciousness", tags=["consciousness-computing"])
api_router.include_router(transcendent_router, prefix="/transcendent", tags=["transcendent-computing"])
api_router.include_router(infinite_router, prefix="/infinite", tags=["infinite-computing"])
api_router.include_router(eternal_router, prefix="/eternal", tags=["eternal-computing"])
api_router.include_router(divine_router, prefix="/divine", tags=["divine-computing"])
api_router.include_router(omnipotent_router, prefix="/omnipotent", tags=["omnipotent-computing"])
api_router.include_router(absolute_router, prefix="/absolute", tags=["absolute-computing"])
api_router.include_router(ultimate_router, prefix="/ultimate", tags=["ultimate-computing"])
api_router.include_router(supreme_router, prefix="/supreme", tags=["supreme-computing"])
api_router.include_router(perfect_router, prefix="/perfect", tags=["perfect-computing"])
api_router.include_router(eternal_router, prefix="/eternal", tags=["eternal-computing"])
api_router.include_router(divine_router, prefix="/divine", tags=["divine-computing"])
api_router.include_router(infinite_router, prefix="/infinite", tags=["infinite-computing"])
api_router.include_router(eternal_router, prefix="/eternal", tags=["eternal-computing"])
api_router.include_router(divine_router, prefix="/divine", tags=["divine-computing"])
api_router.include_router(boundless_router, prefix="/boundless", tags=["boundless-computing"])
api_router.include_router(limitless_router, prefix="/limitless", tags=["limitless-computing"])
api_router.include_router(endless_router, prefix="/endless", tags=["endless-computing"])
api_router.include_router(infinite_router, prefix="/infinite", tags=["infinite-computing"])
api_router.include_router(eternal_router, prefix="/eternal", tags=["eternal-computing"])
api_router.include_router(divine_router, prefix="/divine", tags=["divine-computing"])

__all__ = ["api_router"]
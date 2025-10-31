"""
Metaverse & Consciousness AI API Routes for Gamma App
=====================================================

API endpoints for Metaverse AR/VR and Consciousness AI services
providing advanced immersive and self-aware AI capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging

from ..services.metaverse_ar_vr_service import (
    MetaverseARVRService,
    VRDevice,
    ARSession,
    VR3DContent,
    VRScene,
    VRInteraction,
    VRUser,
    VRDeviceType,
    ARPlatform,
    ContentType,
    InteractionType
)

from ..services.consciousness_ai_service import (
    ConsciousnessAIService,
    ConsciousnessState,
    EmotionalMemory,
    EthicalDecision,
    LearningExperience,
    SelfReflection,
    ConsciousnessLevel,
    EmotionalState,
    EthicalPrinciple,
    LearningType
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/metaverse-consciousness", tags=["Metaverse & Consciousness AI"])

# Dependency to get services
def get_metaverse_service() -> MetaverseARVRService:
    """Get Metaverse AR/VR service instance."""
    return MetaverseARVRService()

def get_consciousness_service() -> ConsciousnessAIService:
    """Get Consciousness AI service instance."""
    return ConsciousnessAIService()

@router.get("/")
async def metaverse_consciousness_root():
    """Metaverse & Consciousness AI root endpoint."""
    return {
        "message": "Metaverse AR/VR & Consciousness AI Services for Gamma App",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "services": [
            "Metaverse AR/VR",
            "Consciousness AI",
            "3D Content Management",
            "VR Scene Creation",
            "Emotional Intelligence",
            "Ethical Decision Making",
            "Self-Reflection & Learning"
        ]
    }

# ==================== METAVERSE AR/VR ENDPOINTS ====================

@router.post("/vr/devices/register")
async def register_vr_device(
    device_info: Dict[str, Any],
    metaverse_service: MetaverseARVRService = Depends(get_metaverse_service)
):
    """Register a VR device."""
    try:
        device_id = await metaverse_service.register_vr_device(device_info)
        return {
            "device_id": device_id,
            "message": "VR device registered successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error registering VR device: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register VR device: {e}")

@router.post("/ar/sessions/start")
async def start_ar_session(
    platform: ARPlatform,
    device_info: Dict[str, Any],
    metaverse_service: MetaverseARVRService = Depends(get_metaverse_service)
):
    """Start an AR session."""
    try:
        session_id = await metaverse_service.start_ar_session(platform, device_info)
        return {
            "session_id": session_id,
            "platform": platform.value,
            "message": "AR session started successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting AR session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start AR session: {e}")

@router.post("/content/3d/upload")
async def upload_3d_content(
    content_info: Dict[str, Any],
    metaverse_service: MetaverseARVRService = Depends(get_metaverse_service)
):
    """Upload 3D content for VR/AR."""
    try:
        content_id = await metaverse_service.upload_3d_content(content_info)
        return {
            "content_id": content_id,
            "message": "3D content uploaded successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error uploading 3D content: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload 3D content: {e}")

@router.post("/scenes/create")
async def create_vr_scene(
    scene_info: Dict[str, Any],
    metaverse_service: MetaverseARVRService = Depends(get_metaverse_service)
):
    """Create a VR/AR scene."""
    try:
        scene_id = await metaverse_service.create_vr_scene(scene_info)
        return {
            "scene_id": scene_id,
            "message": "VR scene created successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating VR scene: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create VR scene: {e}")

@router.post("/scenes/{scene_id}/interactions/add")
async def add_scene_interaction(
    scene_id: str,
    interaction_info: Dict[str, Any],
    metaverse_service: MetaverseARVRService = Depends(get_metaverse_service)
):
    """Add interaction to a VR scene."""
    try:
        interaction_id = await metaverse_service.add_interaction_to_scene(scene_id, interaction_info)
        return {
            "interaction_id": interaction_id,
            "scene_id": scene_id,
            "message": "Interaction added to scene successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error adding scene interaction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add scene interaction: {e}")

@router.post("/scenes/{scene_id}/join")
async def join_vr_scene(
    scene_id: str,
    user_id: str,
    avatar_info: Dict[str, Any],
    metaverse_service: MetaverseARVRService = Depends(get_metaverse_service)
):
    """Join a VR scene."""
    try:
        success = await metaverse_service.join_vr_scene(scene_id, user_id, avatar_info)
        if success:
            return {
                "scene_id": scene_id,
                "user_id": user_id,
                "message": "Successfully joined VR scene",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to join VR scene")
    except Exception as e:
        logger.error(f"Error joining VR scene: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to join VR scene: {e}")

@router.put("/users/{user_id}/position")
async def update_user_position(
    user_id: str,
    position: Tuple[float, float, float],
    rotation: Tuple[float, float, float, float],
    metaverse_service: MetaverseARVRService = Depends(get_metaverse_service)
):
    """Update user position in VR."""
    try:
        success = await metaverse_service.update_user_position(user_id, position, rotation)
        if success:
            return {
                "user_id": user_id,
                "position": position,
                "rotation": rotation,
                "message": "User position updated successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error(f"Error updating user position: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update user position: {e}")

@router.post("/users/{user_id}/hand-tracking")
async def process_hand_tracking(
    user_id: str,
    hand_data: Dict[str, Any],
    metaverse_service: MetaverseARVRService = Depends(get_metaverse_service)
):
    """Process hand tracking data."""
    try:
        result = await metaverse_service.process_hand_tracking(user_id, hand_data)
        return {
            "user_id": user_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error processing hand tracking: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process hand tracking: {e}")

@router.post("/users/{user_id}/voice-command")
async def process_voice_command(
    user_id: str,
    command: str,
    metaverse_service: MetaverseARVRService = Depends(get_metaverse_service)
):
    """Process voice command in VR."""
    try:
        result = await metaverse_service.process_voice_command(user_id, command)
        return {
            "user_id": user_id,
            "command": command,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error processing voice command: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process voice command: {e}")

@router.get("/scenes/{scene_id}/users")
async def get_scene_users(
    scene_id: str,
    metaverse_service: MetaverseARVRService = Depends(get_metaverse_service)
):
    """Get users in a VR scene."""
    try:
        users = await metaverse_service.get_scene_users(scene_id)
        return {
            "scene_id": scene_id,
            "users": users,
            "total_users": len(users),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting scene users: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scene users: {e}")

@router.get("/content/3d")
async def get_available_content(
    content_type: Optional[ContentType] = Query(None, description="Filter by content type"),
    metaverse_service: MetaverseARVRService = Depends(get_metaverse_service)
):
    """Get available 3D content."""
    try:
        content = await metaverse_service.get_available_content(content_type)
        return {
            "content": content,
            "total_content": len(content),
            "content_type_filter": content_type.value if content_type else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting available content: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get available content: {e}")

@router.get("/scenes/{scene_id}")
async def get_scene_info(
    scene_id: str,
    metaverse_service: MetaverseARVRService = Depends(get_metaverse_service)
):
    """Get VR scene information."""
    try:
        scene_info = await metaverse_service.get_scene_info(scene_id)
        if scene_info:
            return scene_info
        else:
            raise HTTPException(status_code=404, detail="Scene not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scene info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scene info: {e}")

@router.get("/metaverse/statistics")
async def get_metaverse_statistics(
    metaverse_service: MetaverseARVRService = Depends(get_metaverse_service)
):
    """Get Metaverse AR/VR service statistics."""
    try:
        stats = await metaverse_service.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting metaverse statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metaverse statistics: {e}")

# ==================== CONSCIOUSNESS AI ENDPOINTS ====================

@router.post("/consciousness/emotional-input")
async def process_emotional_input(
    input_data: Dict[str, Any],
    consciousness_service: ConsciousnessAIService = Depends(get_consciousness_service)
):
    """Process emotional input and update emotional state."""
    try:
        result = await consciousness_service.process_emotional_input(input_data)
        return {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error processing emotional input: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process emotional input: {e}")

@router.post("/consciousness/ethical-decision")
async def make_ethical_decision(
    situation: str,
    options: List[str],
    context: Dict[str, Any],
    consciousness_service: ConsciousnessAIService = Depends(get_consciousness_service)
):
    """Make an ethical decision based on principles."""
    try:
        result = await consciousness_service.make_ethical_decision(situation, options, context)
        return {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error making ethical decision: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to make ethical decision: {e}")

@router.post("/consciousness/learn")
async def learn_from_experience(
    experience_data: Dict[str, Any],
    consciousness_service: ConsciousnessAIService = Depends(get_consciousness_service)
):
    """Learn from an experience and update knowledge."""
    try:
        result = await consciousness_service.learn_from_experience(experience_data)
        return {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error learning from experience: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to learn from experience: {e}")

@router.post("/consciousness/self-reflection")
async def perform_self_reflection(
    reflection_type: str = Query("general", description="Type of self-reflection"),
    consciousness_service: ConsciousnessAIService = Depends(get_consciousness_service)
):
    """Perform self-reflection and introspection."""
    try:
        result = await consciousness_service.perform_self_reflection(reflection_type)
        return {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing self-reflection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to perform self-reflection: {e}")

@router.get("/consciousness/state")
async def get_consciousness_state(
    consciousness_service: ConsciousnessAIService = Depends(get_consciousness_service)
):
    """Get current consciousness state."""
    try:
        state = await consciousness_service.get_consciousness_state()
        return {
            "state": state,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting consciousness state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get consciousness state: {e}")

@router.get("/consciousness/emotional-history")
async def get_emotional_history(
    limit: int = Query(50, ge=1, le=100, description="Number of emotional memories to return"),
    consciousness_service: ConsciousnessAIService = Depends(get_consciousness_service)
):
    """Get emotional memory history."""
    try:
        history = await consciousness_service.get_emotional_history(limit)
        return {
            "emotional_history": history,
            "total_memories": len(history),
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting emotional history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get emotional history: {e}")

@router.get("/consciousness/learning-statistics")
async def get_learning_statistics(
    consciousness_service: ConsciousnessAIService = Depends(get_consciousness_service)
):
    """Get learning statistics."""
    try:
        stats = await consciousness_service.get_learning_statistics()
        return {
            "learning_statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting learning statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get learning statistics: {e}")

# ==================== COMBINED ENDPOINTS ====================

@router.get("/health")
async def health_check(
    metaverse_service: MetaverseARVRService = Depends(get_metaverse_service),
    consciousness_service: ConsciousnessAIService = Depends(get_consciousness_service)
):
    """Health check for both services."""
    try:
        metaverse_stats = await metaverse_service.get_statistics()
        consciousness_state = await consciousness_service.get_consciousness_state()
        
        return {
            "status": "healthy",
            "metaverse_service": {
                "status": "operational",
                "total_vr_devices": metaverse_stats.get("total_vr_devices", 0),
                "total_3d_content": metaverse_stats.get("total_3d_content", 0),
                "total_vr_scenes": metaverse_stats.get("total_vr_scenes", 0)
            },
            "consciousness_service": {
                "status": "operational",
                "consciousness_level": consciousness_state.get("level", "unknown"),
                "emotional_state": consciousness_state.get("emotional_state", "unknown"),
                "self_awareness_score": consciousness_state.get("self_awareness_score", 0.0)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/capabilities")
async def get_capabilities():
    """Get available capabilities of both services."""
    return {
        "metaverse_ar_vr": {
            "vr_devices": [device_type.value for device_type in VRDeviceType],
            "ar_platforms": [platform.value for platform in ARPlatform],
            "content_types": [content_type.value for content_type in ContentType],
            "interaction_types": [interaction_type.value for interaction_type in InteractionType],
            "capabilities": [
                "VR Device Management",
                "AR Session Management",
                "3D Content Upload & Management",
                "VR Scene Creation",
                "Real-time User Tracking",
                "Hand Tracking",
                "Voice Commands",
                "Spatial Audio",
                "Multi-user Collaboration"
            ]
        },
        "consciousness_ai": {
            "consciousness_levels": [level.value for level in ConsciousnessLevel],
            "emotional_states": [state.value for state in EmotionalState],
            "ethical_principles": [principle.value for principle in EthicalPrinciple],
            "learning_types": [learning_type.value for learning_type in LearningType],
            "capabilities": [
                "Emotional Intelligence",
                "Ethical Decision Making",
                "Self-Reflection & Introspection",
                "Continuous Learning",
                "Memory Consolidation",
                "Empathy & Understanding",
                "Creative Problem Solving",
                "Adaptive Behavior",
                "Self-Awareness"
            ]
        },
        "timestamp": datetime.now().isoformat()
    }






















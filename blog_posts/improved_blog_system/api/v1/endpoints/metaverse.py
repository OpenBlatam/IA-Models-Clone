"""
Advanced Metaverse API endpoints
"""

from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime

from ....services.advanced_metaverse_service import AdvancedMetaverseService, MetaverseWorldType, MetaverseAvatarType, MetaverseAssetType, MetaverseInteractionType
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class CreateMetaverseWorldRequest(BaseModel):
    """Request model for creating a metaverse world."""
    name: str = Field(..., description="World name")
    description: str = Field(..., description="World description")
    world_type: str = Field(..., description="World type")
    configuration: Optional[Dict[str, Any]] = Field(default=None, description="World configuration")


class CreateMetaverseAvatarRequest(BaseModel):
    """Request model for creating a metaverse avatar."""
    name: str = Field(..., description="Avatar name")
    avatar_type: str = Field(..., description="Avatar type")
    world_id: str = Field(..., description="World ID")
    appearance: Optional[Dict[str, Any]] = Field(default=None, description="Avatar appearance")
    abilities: Optional[Dict[str, Any]] = Field(default=None, description="Avatar abilities")


class CreateMetaverseAssetRequest(BaseModel):
    """Request model for creating a metaverse asset."""
    name: str = Field(..., description="Asset name")
    asset_type: str = Field(..., description="Asset type")
    world_id: str = Field(..., description="World ID")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Asset properties")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Asset metadata")


class TrackMetaverseInteractionRequest(BaseModel):
    """Request model for tracking metaverse interaction."""
    avatar_id: str = Field(..., description="Avatar ID")
    interaction_type: str = Field(..., description="Interaction type")
    target_id: Optional[str] = Field(default=None, description="Target ID")
    interaction_data: Optional[Dict[str, Any]] = Field(default=None, description="Interaction data")
    position: Optional[Dict[str, float]] = Field(default=None, description="Interaction position")
    world_id: Optional[str] = Field(default=None, description="World ID")


async def get_metaverse_service(session: DatabaseSessionDep) -> AdvancedMetaverseService:
    """Get metaverse service instance."""
    return AdvancedMetaverseService(session)


@router.post("/worlds", response_model=Dict[str, Any])
async def create_metaverse_world(
    request: CreateMetaverseWorldRequest = Depends(),
    metaverse_service: AdvancedMetaverseService = Depends(get_metaverse_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a new metaverse world."""
    try:
        # Convert world type to enum
        try:
            world_type_enum = MetaverseWorldType(request.world_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid world type: {request.world_type}")
        
        result = await metaverse_service.create_metaverse_world(
            name=request.name,
            description=request.description,
            world_type=world_type_enum,
            user_id=str(current_user.id),
            configuration=request.configuration
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Metaverse world created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create metaverse world"
        )


@router.post("/avatars", response_model=Dict[str, Any])
async def create_metaverse_avatar(
    request: CreateMetaverseAvatarRequest = Depends(),
    metaverse_service: AdvancedMetaverseService = Depends(get_metaverse_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a new metaverse avatar."""
    try:
        # Convert avatar type to enum
        try:
            avatar_type_enum = MetaverseAvatarType(request.avatar_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid avatar type: {request.avatar_type}")
        
        result = await metaverse_service.create_metaverse_avatar(
            name=request.name,
            avatar_type=avatar_type_enum,
            user_id=str(current_user.id),
            world_id=request.world_id,
            appearance=request.appearance,
            abilities=request.abilities
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Metaverse avatar created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create metaverse avatar"
        )


@router.post("/assets", response_model=Dict[str, Any])
async def create_metaverse_asset(
    request: CreateMetaverseAssetRequest = Depends(),
    metaverse_service: AdvancedMetaverseService = Depends(get_metaverse_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a new metaverse asset."""
    try:
        # Convert asset type to enum
        try:
            asset_type_enum = MetaverseAssetType(request.asset_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid asset type: {request.asset_type}")
        
        result = await metaverse_service.create_metaverse_asset(
            name=request.name,
            asset_type=asset_type_enum,
            world_id=request.world_id,
            owner_id=str(current_user.id),
            properties=request.properties,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Metaverse asset created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create metaverse asset"
        )


@router.post("/interactions", response_model=Dict[str, Any])
async def track_metaverse_interaction(
    request: TrackMetaverseInteractionRequest = Depends(),
    metaverse_service: AdvancedMetaverseService = Depends(get_metaverse_service),
    current_user: CurrentUserDep = Depends()
):
    """Track metaverse interaction."""
    try:
        # Convert interaction type to enum
        try:
            interaction_type_enum = MetaverseInteractionType(request.interaction_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid interaction type: {request.interaction_type}")
        
        result = await metaverse_service.track_metaverse_interaction(
            avatar_id=request.avatar_id,
            interaction_type=interaction_type_enum,
            target_id=request.target_id,
            interaction_data=request.interaction_data,
            position=request.position,
            world_id=request.world_id
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Metaverse interaction tracked successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to track metaverse interaction"
        )


@router.get("/analytics", response_model=Dict[str, Any])
async def get_metaverse_analytics(
    world_id: Optional[str] = Query(default=None, description="World ID"),
    avatar_id: Optional[str] = Query(default=None, description="Avatar ID"),
    interaction_type: Optional[str] = Query(default=None, description="Interaction type"),
    time_period: str = Query(default="24_hours", description="Time period"),
    metaverse_service: AdvancedMetaverseService = Depends(get_metaverse_service),
    current_user: CurrentUserDep = Depends()
):
    """Get metaverse analytics."""
    try:
        result = await metaverse_service.get_metaverse_analytics(
            world_id=world_id,
            avatar_id=avatar_id,
            interaction_type=interaction_type,
            time_period=time_period
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Metaverse analytics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get metaverse analytics"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_metaverse_stats(
    metaverse_service: AdvancedMetaverseService = Depends(get_metaverse_service),
    current_user: CurrentUserDep = Depends()
):
    """Get metaverse system statistics."""
    try:
        result = await metaverse_service.get_metaverse_stats()
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Metaverse statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get metaverse statistics"
        )


@router.get("/world-types", response_model=Dict[str, Any])
async def get_metaverse_world_types():
    """Get available metaverse world types."""
    world_types = {
        "virtual_reality": {
            "name": "Virtual Reality",
            "description": "Immersive virtual reality world",
            "icon": "🥽",
            "capabilities": ["vr", "immersion", "presence", "interaction"],
            "platforms": ["pc", "console", "mobile", "standalone"]
        },
        "augmented_reality": {
            "name": "Augmented Reality",
            "description": "Augmented reality overlay world",
            "icon": "👓",
            "capabilities": ["ar", "overlay", "real_world", "mixed"],
            "platforms": ["mobile", "glasses", "headset"]
        },
        "mixed_reality": {
            "name": "Mixed Reality",
            "description": "Mixed reality hybrid world",
            "icon": "🌐",
            "capabilities": ["mr", "hybrid", "seamless", "spatial"],
            "platforms": ["hololens", "magic_leap", "varjo"]
        },
        "gaming": {
            "name": "Gaming",
            "description": "Gaming-focused virtual world",
            "icon": "🎮",
            "capabilities": ["games", "competition", "achievements", "leaderboards"],
            "platforms": ["pc", "console", "mobile", "vr"]
        },
        "social": {
            "name": "Social",
            "description": "Social interaction virtual world",
            "icon": "👥",
            "capabilities": ["social", "communication", "networking", "community"],
            "platforms": ["pc", "mobile", "vr", "ar"]
        },
        "educational": {
            "name": "Educational",
            "description": "Educational virtual world",
            "icon": "📚",
            "capabilities": ["learning", "teaching", "simulation", "training"],
            "platforms": ["pc", "vr", "ar", "mobile"]
        },
        "business": {
            "name": "Business",
            "description": "Business and professional virtual world",
            "icon": "💼",
            "capabilities": ["meetings", "collaboration", "presentations", "networking"],
            "platforms": ["pc", "vr", "ar", "web"]
        },
        "entertainment": {
            "name": "Entertainment",
            "description": "Entertainment and media virtual world",
            "icon": "🎭",
            "capabilities": ["shows", "concerts", "movies", "events"],
            "platforms": ["pc", "vr", "ar", "mobile"]
        },
        "art": {
            "name": "Art",
            "description": "Artistic and creative virtual world",
            "icon": "🎨",
            "capabilities": ["creation", "exhibition", "gallery", "museum"],
            "platforms": ["pc", "vr", "ar", "mobile"]
        },
        "music": {
            "name": "Music",
            "description": "Music and audio virtual world",
            "icon": "🎵",
            "capabilities": ["concerts", "creation", "collaboration", "streaming"],
            "platforms": ["pc", "vr", "ar", "mobile"]
        },
        "sports": {
            "name": "Sports",
            "description": "Sports and fitness virtual world",
            "icon": "⚽",
            "capabilities": ["training", "competition", "fitness", "coaching"],
            "platforms": ["pc", "vr", "ar", "mobile"]
        },
        "travel": {
            "name": "Travel",
            "description": "Travel and exploration virtual world",
            "icon": "✈️",
            "capabilities": ["exploration", "tourism", "culture", "adventure"],
            "platforms": ["pc", "vr", "ar", "mobile"]
        },
        "shopping": {
            "name": "Shopping",
            "description": "Shopping and commerce virtual world",
            "icon": "🛒",
            "capabilities": ["shopping", "commerce", "retail", "marketplace"],
            "platforms": ["pc", "vr", "ar", "mobile"]
        },
        "healthcare": {
            "name": "Healthcare",
            "description": "Healthcare and medical virtual world",
            "icon": "🏥",
            "capabilities": ["treatment", "therapy", "training", "simulation"],
            "platforms": ["pc", "vr", "ar", "mobile"]
        },
        "real_estate": {
            "name": "Real Estate",
            "description": "Real estate and property virtual world",
            "icon": "🏠",
            "capabilities": ["viewing", "design", "investment", "management"],
            "platforms": ["pc", "vr", "ar", "mobile"]
        },
        "fashion": {
            "name": "Fashion",
            "description": "Fashion and style virtual world",
            "icon": "👗",
            "capabilities": ["design", "showcase", "try_on", "trends"],
            "platforms": ["pc", "vr", "ar", "mobile"]
        },
        "food": {
            "name": "Food",
            "description": "Food and culinary virtual world",
            "icon": "🍽️",
            "capabilities": ["cooking", "tasting", "restaurants", "recipes"],
            "platforms": ["pc", "vr", "ar", "mobile"]
        },
        "automotive": {
            "name": "Automotive",
            "description": "Automotive and transportation virtual world",
            "icon": "🚗",
            "capabilities": ["design", "testing", "showroom", "driving"],
            "platforms": ["pc", "vr", "ar", "mobile"]
        },
        "technology": {
            "name": "Technology",
            "description": "Technology and innovation virtual world",
            "icon": "💻",
            "capabilities": ["development", "testing", "showcase", "innovation"],
            "platforms": ["pc", "vr", "ar", "mobile"]
        },
        "science": {
            "name": "Science",
            "description": "Science and research virtual world",
            "icon": "🔬",
            "capabilities": ["research", "experimentation", "education", "discovery"],
            "platforms": ["pc", "vr", "ar", "mobile"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "world_types": world_types,
            "total_types": len(world_types)
        },
        "message": "Metaverse world types retrieved successfully"
    }


@router.get("/avatar-types", response_model=Dict[str, Any])
async def get_metaverse_avatar_types():
    """Get available metaverse avatar types."""
    avatar_types = {
        "human": {
            "name": "Human",
            "description": "Human-like avatar",
            "icon": "👤",
            "capabilities": ["realistic", "expressive", "customizable", "social"],
            "customization": ["face", "body", "clothing", "accessories"]
        },
        "animal": {
            "name": "Animal",
            "description": "Animal avatar",
            "icon": "🐾",
            "capabilities": ["cute", "playful", "unique", "expressive"],
            "customization": ["species", "color", "size", "accessories"]
        },
        "robot": {
            "name": "Robot",
            "description": "Robotic avatar",
            "icon": "🤖",
            "capabilities": ["futuristic", "mechanical", "precise", "customizable"],
            "customization": ["design", "color", "features", "accessories"]
        },
        "fantasy": {
            "name": "Fantasy",
            "description": "Fantasy creature avatar",
            "icon": "🧙",
            "capabilities": ["magical", "unique", "creative", "expressive"],
            "customization": ["race", "powers", "appearance", "equipment"]
        },
        "alien": {
            "name": "Alien",
            "description": "Alien creature avatar",
            "icon": "👽",
            "capabilities": ["otherworldly", "unique", "creative", "expressive"],
            "customization": ["species", "appearance", "abilities", "technology"]
        },
        "mythical": {
            "name": "Mythical",
            "description": "Mythical creature avatar",
            "icon": "🐉",
            "capabilities": ["legendary", "powerful", "unique", "majestic"],
            "customization": ["creature", "powers", "appearance", "elements"]
        },
        "abstract": {
            "name": "Abstract",
            "description": "Abstract form avatar",
            "icon": "🌀",
            "capabilities": ["artistic", "unique", "creative", "expressive"],
            "customization": ["form", "color", "texture", "animation"]
        },
        "custom": {
            "name": "Custom",
            "description": "Fully customizable avatar",
            "icon": "🎨",
            "capabilities": ["unlimited", "creative", "unique", "personal"],
            "customization": ["everything", "unlimited", "creative", "personal"]
        },
        "ai": {
            "name": "AI",
            "description": "AI-powered avatar",
            "icon": "🧠",
            "capabilities": ["intelligent", "adaptive", "learning", "autonomous"],
            "customization": ["personality", "behavior", "appearance", "abilities"]
        },
        "hybrid": {
            "name": "Hybrid",
            "description": "Hybrid creature avatar",
            "icon": "🦄",
            "capabilities": ["unique", "creative", "expressive", "versatile"],
            "customization": ["combination", "features", "appearance", "abilities"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "avatar_types": avatar_types,
            "total_types": len(avatar_types)
        },
        "message": "Metaverse avatar types retrieved successfully"
    }


@router.get("/asset-types", response_model=Dict[str, Any])
async def get_metaverse_asset_types():
    """Get available metaverse asset types."""
    asset_types = {
        "land": {
            "name": "Land",
            "description": "Virtual land parcel",
            "icon": "🏞️",
            "capabilities": ["ownership", "building", "customization", "monetization"],
            "properties": ["location", "size", "terrain", "resources"]
        },
        "building": {
            "name": "Building",
            "description": "Virtual building or structure",
            "icon": "🏢",
            "capabilities": ["construction", "customization", "functionality", "rental"],
            "properties": ["type", "size", "style", "purpose"]
        },
        "vehicle": {
            "name": "Vehicle",
            "description": "Virtual vehicle or transportation",
            "icon": "🚗",
            "capabilities": ["transportation", "customization", "performance", "ownership"],
            "properties": ["type", "speed", "capacity", "style"]
        },
        "wearable": {
            "name": "Wearable",
            "description": "Virtual clothing or accessory",
            "icon": "👕",
            "capabilities": ["fashion", "customization", "status", "collection"],
            "properties": ["type", "style", "rarity", "stats"]
        },
        "accessory": {
            "name": "Accessory",
            "description": "Virtual accessory or item",
            "icon": "💍",
            "capabilities": ["enhancement", "customization", "status", "collection"],
            "properties": ["type", "effect", "rarity", "durability"]
        },
        "furniture": {
            "name": "Furniture",
            "description": "Virtual furniture or decoration",
            "icon": "🪑",
            "capabilities": ["decoration", "functionality", "customization", "placement"],
            "properties": ["type", "style", "function", "size"]
        },
        "decoration": {
            "name": "Decoration",
            "description": "Virtual decorative item",
            "icon": "🎨",
            "capabilities": ["aesthetics", "customization", "placement", "collection"],
            "properties": ["type", "style", "size", "rarity"]
        },
        "tool": {
            "name": "Tool",
            "description": "Virtual tool or utility",
            "icon": "🔧",
            "capabilities": ["functionality", "efficiency", "customization", "upgrade"],
            "properties": ["type", "function", "efficiency", "durability"]
        },
        "weapon": {
            "name": "Weapon",
            "description": "Virtual weapon or combat item",
            "icon": "⚔️",
            "capabilities": ["combat", "damage", "customization", "upgrade"],
            "properties": ["type", "damage", "range", "durability"]
        },
        "pet": {
            "name": "Pet",
            "description": "Virtual pet or companion",
            "icon": "🐕",
            "capabilities": ["companionship", "interaction", "customization", "breeding"],
            "properties": ["species", "personality", "abilities", "rarity"]
        },
        "plant": {
            "name": "Plant",
            "description": "Virtual plant or vegetation",
            "icon": "🌱",
            "capabilities": ["growth", "harvest", "decoration", "ecology"],
            "properties": ["species", "growth_rate", "yield", "rarity"]
        },
        "mineral": {
            "name": "Mineral",
            "description": "Virtual mineral or resource",
            "icon": "💎",
            "capabilities": ["mining", "crafting", "trading", "collection"],
            "properties": ["type", "rarity", "value", "uses"]
        },
        "art": {
            "name": "Art",
            "description": "Virtual artwork or creation",
            "icon": "🖼️",
            "capabilities": ["display", "collection", "trading", "appreciation"],
            "properties": ["style", "artist", "rarity", "value"]
        },
        "music": {
            "name": "Music",
            "description": "Virtual music or audio",
            "icon": "🎵",
            "capabilities": ["playback", "collection", "trading", "creation"],
            "properties": ["genre", "artist", "duration", "quality"]
        },
        "book": {
            "name": "Book",
            "description": "Virtual book or literature",
            "icon": "📖",
            "capabilities": ["reading", "collection", "trading", "knowledge"],
            "properties": ["genre", "author", "pages", "rarity"]
        },
        "game": {
            "name": "Game",
            "description": "Virtual game or experience",
            "icon": "🎮",
            "capabilities": ["play", "competition", "achievement", "social"],
            "properties": ["genre", "difficulty", "players", "duration"]
        },
        "experience": {
            "name": "Experience",
            "description": "Virtual experience or event",
            "icon": "🌟",
            "capabilities": ["participation", "memory", "sharing", "collection"],
            "properties": ["type", "duration", "participants", "rarity"]
        },
        "service": {
            "name": "Service",
            "description": "Virtual service or utility",
            "icon": "⚙️",
            "capabilities": ["functionality", "automation", "efficiency", "customization"],
            "properties": ["type", "function", "efficiency", "cost"]
        },
        "currency": {
            "name": "Currency",
            "description": "Virtual currency or token",
            "icon": "💰",
            "capabilities": ["exchange", "trading", "value", "utility"],
            "properties": ["type", "value", "supply", "utility"]
        },
        "nft": {
            "name": "NFT",
            "description": "Non-fungible token",
            "icon": "🎫",
            "capabilities": ["ownership", "uniqueness", "trading", "verification"],
            "properties": ["contract", "token_id", "metadata", "rarity"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "asset_types": asset_types,
            "total_types": len(asset_types)
        },
        "message": "Metaverse asset types retrieved successfully"
    }


@router.get("/interaction-types", response_model=Dict[str, Any])
async def get_metaverse_interaction_types():
    """Get available metaverse interaction types."""
    interaction_types = {
        "chat": {
            "name": "Chat",
            "description": "Text-based communication",
            "icon": "💬",
            "capabilities": ["communication", "expression", "record", "search"],
            "properties": ["text", "language", "emotion", "context"]
        },
        "voice": {
            "name": "Voice",
            "description": "Voice-based communication",
            "icon": "🎤",
            "capabilities": ["communication", "expression", "record", "transcription"],
            "properties": ["audio", "language", "emotion", "quality"]
        },
        "video": {
            "name": "Video",
            "description": "Video-based communication",
            "icon": "📹",
            "capabilities": ["communication", "expression", "record", "streaming"],
            "properties": ["video", "audio", "quality", "bandwidth"]
        },
        "gesture": {
            "name": "Gesture",
            "description": "Gesture-based interaction",
            "icon": "✋",
            "capabilities": ["expression", "control", "recognition", "customization"],
            "properties": ["type", "precision", "speed", "recognition"]
        },
        "touch": {
            "name": "Touch",
            "description": "Touch-based interaction",
            "icon": "👆",
            "capabilities": ["control", "feedback", "precision", "customization"],
            "properties": ["pressure", "location", "duration", "feedback"]
        },
        "gaze": {
            "name": "Gaze",
            "description": "Eye gaze-based interaction",
            "icon": "👁️",
            "capabilities": ["control", "attention", "precision", "analytics"],
            "properties": ["direction", "duration", "focus", "intention"]
        },
        "movement": {
            "name": "Movement",
            "description": "Body movement interaction",
            "icon": "🏃",
            "capabilities": ["control", "expression", "exercise", "analytics"],
            "properties": ["type", "speed", "direction", "intensity"]
        },
        "emotion": {
            "name": "Emotion",
            "description": "Emotional expression interaction",
            "icon": "😊",
            "capabilities": ["expression", "recognition", "response", "analytics"],
            "properties": ["type", "intensity", "duration", "context"]
        },
        "expression": {
            "name": "Expression",
            "description": "Facial expression interaction",
            "icon": "😮",
            "capabilities": ["expression", "recognition", "response", "customization"],
            "properties": ["type", "intensity", "duration", "accuracy"]
        },
        "action": {
            "name": "Action",
            "description": "Action-based interaction",
            "icon": "⚡",
            "capabilities": ["execution", "control", "automation", "customization"],
            "properties": ["type", "target", "result", "feedback"]
        },
        "collaboration": {
            "name": "Collaboration",
            "description": "Collaborative interaction",
            "icon": "🤝",
            "capabilities": ["cooperation", "sharing", "creation", "productivity"],
            "properties": ["participants", "goal", "tools", "outcome"]
        },
        "competition": {
            "name": "Competition",
            "description": "Competitive interaction",
            "icon": "🏆",
            "capabilities": ["competition", "ranking", "achievement", "rewards"],
            "properties": ["type", "participants", "rules", "outcome"]
        },
        "trade": {
            "name": "Trade",
            "description": "Trading interaction",
            "icon": "🤝",
            "capabilities": ["exchange", "negotiation", "transaction", "record"],
            "properties": ["items", "value", "terms", "completion"]
        },
        "gift": {
            "name": "Gift",
            "description": "Gift-giving interaction",
            "icon": "🎁",
            "capabilities": ["giving", "receiving", "appreciation", "relationship"],
            "properties": ["item", "value", "message", "relationship"]
        },
        "share": {
            "name": "Share",
            "description": "Sharing interaction",
            "icon": "📤",
            "capabilities": ["sharing", "distribution", "access", "collaboration"],
            "properties": ["content", "audience", "permissions", "access"]
        },
        "create": {
            "name": "Create",
            "description": "Creation interaction",
            "icon": "✨",
            "capabilities": ["creation", "innovation", "expression", "ownership"],
            "properties": ["type", "content", "tools", "result"]
        },
        "destroy": {
            "name": "Destroy",
            "description": "Destruction interaction",
            "icon": "💥",
            "capabilities": ["destruction", "removal", "cleanup", "recycling"],
            "properties": ["target", "method", "result", "recovery"]
        },
        "modify": {
            "name": "Modify",
            "description": "Modification interaction",
            "icon": "🔧",
            "capabilities": ["modification", "improvement", "customization", "upgrade"],
            "properties": ["target", "changes", "tools", "result"]
        },
        "explore": {
            "name": "Explore",
            "description": "Exploration interaction",
            "icon": "🗺️",
            "capabilities": ["discovery", "navigation", "learning", "adventure"],
            "properties": ["area", "method", "discoveries", "experience"]
        },
        "learn": {
            "name": "Learn",
            "description": "Learning interaction",
            "icon": "📚",
            "capabilities": ["education", "skill_development", "knowledge", "growth"],
            "properties": ["subject", "method", "progress", "achievement"]
        },
        "teach": {
            "name": "Teach",
            "description": "Teaching interaction",
            "icon": "👨‍🏫",
            "capabilities": ["education", "knowledge_sharing", "mentoring", "guidance"],
            "properties": ["subject", "method", "students", "effectiveness"]
        },
        "work": {
            "name": "Work",
            "description": "Work interaction",
            "icon": "💼",
            "capabilities": ["productivity", "collaboration", "achievement", "compensation"],
            "properties": ["task", "tools", "collaborators", "outcome"]
        },
        "play": {
            "name": "Play",
            "description": "Play interaction",
            "icon": "🎮",
            "capabilities": ["entertainment", "relaxation", "socialization", "creativity"],
            "properties": ["activity", "participants", "rules", "enjoyment"]
        },
        "relax": {
            "name": "Relax",
            "description": "Relaxation interaction",
            "icon": "😌",
            "capabilities": ["relaxation", "stress_relief", "wellness", "recovery"],
            "properties": ["activity", "environment", "duration", "effectiveness"]
        },
        "exercise": {
            "name": "Exercise",
            "description": "Exercise interaction",
            "icon": "💪",
            "capabilities": ["fitness", "health", "strength", "endurance"],
            "properties": ["type", "intensity", "duration", "benefits"]
        },
        "meditate": {
            "name": "Meditate",
            "description": "Meditation interaction",
            "icon": "🧘",
            "capabilities": ["mindfulness", "relaxation", "focus", "wellness"],
            "properties": ["type", "duration", "environment", "benefits"]
        },
        "socialize": {
            "name": "Socialize",
            "description": "Socialization interaction",
            "icon": "👥",
            "capabilities": ["socialization", "networking", "friendship", "community"],
            "properties": ["participants", "activity", "duration", "relationship"]
        },
        "date": {
            "name": "Date",
            "description": "Dating interaction",
            "icon": "💕",
            "capabilities": ["romance", "intimacy", "relationship", "connection"],
            "properties": ["participants", "activity", "environment", "chemistry"]
        },
        "marry": {
            "name": "Marry",
            "description": "Marriage interaction",
            "icon": "💒",
            "capabilities": ["commitment", "ceremony", "relationship", "bond"],
            "properties": ["participants", "ceremony", "witnesses", "commitment"]
        },
        "divorce": {
            "name": "Divorce",
            "description": "Divorce interaction",
            "icon": "💔",
            "capabilities": ["separation", "legal", "relationship", "closure"],
            "properties": ["participants", "reason", "process", "outcome"]
        },
        "adopt": {
            "name": "Adopt",
            "description": "Adoption interaction",
            "icon": "👶",
            "capabilities": ["parenting", "care", "responsibility", "family"],
            "properties": ["adopter", "adoptee", "process", "relationship"]
        },
        "breed": {
            "name": "Breed",
            "description": "Breeding interaction",
            "icon": "🐣",
            "capabilities": ["reproduction", "genetics", "offspring", "lineage"],
            "properties": ["parents", "genetics", "offspring", "traits"]
        },
        "custom": {
            "name": "Custom",
            "description": "Custom interaction",
            "icon": "🔧",
            "capabilities": ["customization", "flexibility", "creativity", "uniqueness"],
            "properties": ["type", "parameters", "behavior", "result"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "interaction_types": interaction_types,
            "total_types": len(interaction_types)
        },
        "message": "Metaverse interaction types retrieved successfully"
    }


@router.get("/economy-types", response_model=Dict[str, Any])
async def get_metaverse_economy_types():
    """Get available metaverse economy types."""
    economy_types = {
        "virtual_currency": {
            "name": "Virtual Currency",
            "description": "In-world virtual currency",
            "icon": "🪙",
            "capabilities": ["exchange", "trading", "purchasing", "earning"],
            "properties": ["supply", "demand", "inflation", "utility"]
        },
        "cryptocurrency": {
            "name": "Cryptocurrency",
            "description": "Blockchain-based cryptocurrency",
            "icon": "₿",
            "capabilities": ["decentralized", "secure", "global", "trading"],
            "properties": ["blockchain", "consensus", "mining", "staking"]
        },
        "fiat_currency": {
            "name": "Fiat Currency",
            "description": "Traditional government currency",
            "icon": "💵",
            "capabilities": ["real_world", "stable", "regulated", "accepted"],
            "properties": ["government", "regulation", "stability", "acceptance"]
        },
        "token": {
            "name": "Token",
            "description": "Utility or governance token",
            "icon": "🎫",
            "capabilities": ["utility", "governance", "staking", "trading"],
            "properties": ["utility", "governance", "supply", "distribution"]
        },
        "nft": {
            "name": "NFT",
            "description": "Non-fungible token",
            "icon": "🖼️",
            "capabilities": ["uniqueness", "ownership", "trading", "verification"],
            "properties": ["uniqueness", "metadata", "rarity", "value"]
        },
        "land": {
            "name": "Land",
            "description": "Virtual land ownership",
            "icon": "🏞️",
            "capabilities": ["ownership", "development", "rental", "trading"],
            "properties": ["location", "size", "development", "value"]
        },
        "asset": {
            "name": "Asset",
            "description": "Virtual asset ownership",
            "icon": "📦",
            "capabilities": ["ownership", "usage", "trading", "rental"],
            "properties": ["type", "functionality", "rarity", "value"]
        },
        "service": {
            "name": "Service",
            "description": "Virtual service provision",
            "icon": "⚙️",
            "capabilities": ["provision", "automation", "efficiency", "customization"],
            "properties": ["type", "function", "efficiency", "cost"]
        },
        "experience": {
            "name": "Experience",
            "description": "Virtual experience access",
            "icon": "🌟",
            "capabilities": ["access", "participation", "memory", "sharing"],
            "properties": ["type", "duration", "participants", "value"]
        },
        "data": {
            "name": "Data",
            "description": "Virtual data and information",
            "icon": "📊",
            "capabilities": ["collection", "analysis", "trading", "insights"],
            "properties": ["type", "quality", "volume", "value"]
        },
        "attention": {
            "name": "Attention",
            "description": "User attention and engagement",
            "icon": "👁️",
            "capabilities": ["capture", "measurement", "monetization", "optimization"],
            "properties": ["duration", "quality", "engagement", "value"]
        },
        "reputation": {
            "name": "Reputation",
            "description": "User reputation and standing",
            "icon": "⭐",
            "capabilities": ["building", "maintenance", "trading", "influence"],
            "properties": ["score", "history", "influence", "value"]
        },
        "skill": {
            "name": "Skill",
            "description": "User skill and ability",
            "icon": "🎯",
            "capabilities": ["development", "demonstration", "teaching", "trading"],
            "properties": ["type", "level", "experience", "value"]
        },
        "time": {
            "name": "Time",
            "description": "User time and availability",
            "icon": "⏰",
            "capabilities": ["allocation", "trading", "optimization", "value"],
            "properties": ["duration", "quality", "availability", "value"]
        },
        "energy": {
            "name": "Energy",
            "description": "User energy and effort",
            "icon": "⚡",
            "capabilities": ["expenditure", "recovery", "trading", "optimization"],
            "properties": ["type", "amount", "recovery", "value"]
        },
        "creativity": {
            "name": "Creativity",
            "description": "User creativity and innovation",
            "icon": "💡",
            "capabilities": ["expression", "collaboration", "trading", "inspiration"],
            "properties": ["type", "quality", "uniqueness", "value"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "economy_types": economy_types,
            "total_types": len(economy_types)
        },
        "message": "Metaverse economy types retrieved successfully"
    }


@router.get("/health", response_model=Dict[str, Any])
async def get_metaverse_health(
    metaverse_service: AdvancedMetaverseService = Depends(get_metaverse_service),
    current_user: CurrentUserDep = Depends()
):
    """Get metaverse system health status."""
    try:
        # Get metaverse stats
        stats = await metaverse_service.get_metaverse_stats()
        
        # Calculate health metrics
        total_worlds = stats["data"].get("total_worlds", 0)
        total_avatars = stats["data"].get("total_avatars", 0)
        total_assets = stats["data"].get("total_assets", 0)
        total_interactions = stats["data"].get("total_interactions", 0)
        active_worlds = stats["data"].get("active_worlds", 0)
        worlds_by_type = stats["data"].get("worlds_by_type", {})
        avatars_by_type = stats["data"].get("avatars_by_type", {})
        assets_by_type = stats["data"].get("assets_by_type", {})
        
        # Calculate health score
        health_score = 100
        
        # Check world diversity
        if len(worlds_by_type) < 3:
            health_score -= 25
        elif len(worlds_by_type) > 15:
            health_score -= 5
        
        # Check avatar diversity
        if len(avatars_by_type) < 3:
            health_score -= 20
        elif len(avatars_by_type) > 10:
            health_score -= 5
        
        # Check asset diversity
        if len(assets_by_type) < 5:
            health_score -= 30
        elif len(assets_by_type) > 20:
            health_score -= 5
        
        # Check interaction activity
        if total_avatars > 0:
            interactions_per_avatar = total_interactions / total_avatars
            if interactions_per_avatar < 10:
                health_score -= 25
            elif interactions_per_avatar > 1000:
                health_score -= 10
        
        # Check active worlds
        if total_worlds > 0:
            active_ratio = active_worlds / total_worlds
            if active_ratio < 0.2:
                health_score -= 20
            elif active_ratio > 0.8:
                health_score -= 5
        
        # Check asset distribution
        if total_worlds > 0:
            assets_per_world = total_assets / total_worlds
            if assets_per_world < 5:
                health_score -= 20
            elif assets_per_world > 1000:
                health_score -= 10
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "total_worlds": total_worlds,
                "total_avatars": total_avatars,
                "total_assets": total_assets,
                "total_interactions": total_interactions,
                "active_worlds": active_worlds,
                "world_diversity": len(worlds_by_type),
                "avatar_diversity": len(avatars_by_type),
                "asset_diversity": len(assets_by_type),
                "interactions_per_avatar": interactions_per_avatar if total_avatars > 0 else 0,
                "active_ratio": active_ratio if total_worlds > 0 else 0,
                "assets_per_world": assets_per_world if total_worlds > 0 else 0,
                "worlds_by_type": worlds_by_type,
                "avatars_by_type": avatars_by_type,
                "assets_by_type": assets_by_type,
                "timestamp": datetime.utcnow().isoformat()
            },
            "message": "Metaverse health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get metaverse health status"
        )

























"""
Social Router - Handles gamification, social features, and community endpoints
"""

from fastapi import APIRouter, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import Optional
import json

from ...api.services_locator import get_service
from ...utils.logger import logger

router = APIRouter(prefix="/dermatology", tags=["social"])


@router.get("/gamification/stats/{user_id}")
async def get_gamification_stats(user_id: str):
    """Obtiene estadísticas de gamificación"""
    try:
        gamification = get_service("gamification")
        stats = gamification.get_user_stats(user_id)
        return JSONResponse(content={"success": True, "stats": stats.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/gamification/achievements/{user_id}")
async def get_user_achievements(user_id: str):
    """Obtiene logros del usuario"""
    try:
        gamification = get_service("gamification")
        achievements = gamification.get_user_achievements(user_id)
        return JSONResponse(content={
            "success": True,
            "achievements": [a.to_dict() for a in achievements]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/gamification/leaderboard")
async def get_leaderboard(
    category: Optional[str] = Query(None),
    limit: int = Query(100)
):
    """Obtiene leaderboard"""
    try:
        gamification = get_service("gamification")
        leaderboard = gamification.get_leaderboard(category, limit)
        return JSONResponse(content={
            "success": True,
            "leaderboard": [entry.to_dict() for entry in leaderboard]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/challenges/available/{user_id}")
async def get_available_challenges(user_id: str):
    """Obtiene desafíos disponibles"""
    try:
        challenge_system = get_service("challenge_system")
        challenges = challenge_system.get_available_challenges(user_id)
        return JSONResponse(content={
            "success": True,
            "challenges": [c.to_dict() for c in challenges]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/challenges/start")
async def start_challenge(
    user_id: str = Form(...),
    challenge_id: str = Form(...)
):
    """Inicia un desafío"""
    try:
        challenge_system = get_service("challenge_system")
        user_challenge = challenge_system.start_challenge(user_id, challenge_id)
        return JSONResponse(content={
            "success": True,
            "challenge": user_challenge.to_dict()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/challenges/user/{user_id}")
async def get_user_challenges(user_id: str):
    """Obtiene desafíos del usuario"""
    try:
        challenge_system = get_service("challenge_system")
        challenges = challenge_system.get_user_challenges(user_id)
        return JSONResponse(content={
            "success": True,
            "challenges": [c.to_dict() for c in challenges]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/social/follow")
async def follow_user(
    user_id: str = Form(...),
    target_user_id: str = Form(...)
):
    """Sigue a un usuario"""
    try:
        social_features = get_service("social_features")
        connection = social_features.follow_user(user_id, target_user_id)
        return JSONResponse(content={
            "success": True,
            "connection": connection.to_dict()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/social/post")
async def create_social_post(
    user_id: str = Form(...),
    content: str = Form(...),
    image_url: Optional[str] = Form(None)
):
    """Crea un post social"""
    try:
        social_features = get_service("social_features")
        post = social_features.create_post(user_id, content, image_url)
        return JSONResponse(content={"success": True, "post": post.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/social/feed/{user_id}")
async def get_social_feed(user_id: str, limit: int = Query(20)):
    """Obtiene feed social"""
    try:
        social_features = get_service("social_features")
        feed = social_features.get_feed(user_id, limit)
        return JSONResponse(content={
            "success": True,
            "feed": [post.to_dict() for post in feed]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/community/post")
async def create_community_post(
    user_id: str = Form(...),
    title: str = Form(...),
    content: str = Form(...),
    category: Optional[str] = Form(None)
):
    """Crea un post en la comunidad"""
    try:
        community_features = get_service("community_features")
        post = community_features.create_post(user_id, title, content, category)
        return JSONResponse(content={"success": True, "post": post.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/community/posts")
async def get_community_posts(
    category: Optional[str] = Query(None),
    limit: int = Query(20)
):
    """Obtiene posts de la comunidad"""
    try:
        community_features = get_service("community_features")
        posts = community_features.get_posts(category, limit)
        return JSONResponse(content={
            "success": True,
            "posts": [post.to_dict() for post in posts]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")





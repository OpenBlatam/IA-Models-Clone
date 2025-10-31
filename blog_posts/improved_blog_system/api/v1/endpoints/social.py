"""
Advanced Social Features API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime

from ....services.advanced_social_service import AdvancedSocialService, SocialGroupType, SocialEventType
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class FollowUserRequest(BaseModel):
    """Request model for following a user."""
    following_id: str = Field(..., description="ID of user to follow")


class LikeContentRequest(BaseModel):
    """Request model for liking content."""
    content_id: str = Field(..., description="Content ID")
    content_type: str = Field(default="post", description="Content type")


class ShareContentRequest(BaseModel):
    """Request model for sharing content."""
    content_id: str = Field(..., description="Content ID")
    content_type: str = Field(default="post", description="Content type")
    platform: str = Field(default="internal", description="Sharing platform")
    message: Optional[str] = Field(default=None, description="Share message")


class BookmarkContentRequest(BaseModel):
    """Request model for bookmarking content."""
    content_id: str = Field(..., description="Content ID")
    content_type: str = Field(default="post", description="Content type")
    tags: Optional[List[str]] = Field(default=None, description="Bookmark tags")


class MentionUserRequest(BaseModel):
    """Request model for mentioning a user."""
    mentioned_user_id: str = Field(..., description="ID of user to mention")
    content_id: str = Field(..., description="Content ID")
    content_type: str = Field(default="post", description="Content type")
    context: Optional[str] = Field(default=None, description="Mention context")


class CreateGroupRequest(BaseModel):
    """Request model for creating a social group."""
    name: str = Field(..., description="Group name")
    description: str = Field(..., description="Group description")
    group_type: str = Field(default="public", description="Group type")
    tags: Optional[List[str]] = Field(default=None, description="Group tags")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class JoinGroupRequest(BaseModel):
    """Request model for joining a group."""
    group_id: str = Field(..., description="Group ID")


class CreateEventRequest(BaseModel):
    """Request model for creating a social event."""
    title: str = Field(..., description="Event title")
    description: str = Field(..., description="Event description")
    event_type: str = Field(default="virtual", description="Event type")
    start_date: Optional[datetime] = Field(default=None, description="Event start date")
    end_date: Optional[datetime] = Field(default=None, description="Event end date")
    location: Optional[str] = Field(default=None, description="Event location")
    max_attendees: Optional[int] = Field(default=None, description="Maximum attendees")
    tags: Optional[List[str]] = Field(default=None, description="Event tags")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class AttendEventRequest(BaseModel):
    """Request model for attending an event."""
    event_id: str = Field(..., description="Event ID")


class SocialFeedRequest(BaseModel):
    """Request model for social feed."""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Page size")
    feed_type: str = Field(default="following", description="Feed type")


async def get_social_service(session: DatabaseSessionDep) -> AdvancedSocialService:
    """Get social service instance."""
    return AdvancedSocialService(session)


@router.post("/follow", response_model=Dict[str, Any])
async def follow_user(
    request: FollowUserRequest = Depends(),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Follow a user."""
    try:
        result = await social_service.follow_user(
            follower_id=str(current_user.id),
            following_id=request.following_id
        )
        
        return {
            "success": True,
            "data": result,
            "message": "User followed successfully"
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
            detail="Failed to follow user"
        )


@router.post("/unfollow", response_model=Dict[str, Any])
async def unfollow_user(
    request: FollowUserRequest = Depends(),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Unfollow a user."""
    try:
        result = await social_service.unfollow_user(
            follower_id=str(current_user.id),
            following_id=request.following_id
        )
        
        return {
            "success": True,
            "data": result,
            "message": "User unfollowed successfully"
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
            detail="Failed to unfollow user"
        )


@router.post("/like", response_model=Dict[str, Any])
async def like_content(
    request: LikeContentRequest = Depends(),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Like content."""
    try:
        result = await social_service.like_content(
            user_id=str(current_user.id),
            content_id=request.content_id,
            content_type=request.content_type
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Content liked successfully"
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
            detail="Failed to like content"
        )


@router.post("/unlike", response_model=Dict[str, Any])
async def unlike_content(
    request: LikeContentRequest = Depends(),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Unlike content."""
    try:
        result = await social_service.unlike_content(
            user_id=str(current_user.id),
            content_id=request.content_id,
            content_type=request.content_type
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Content unliked successfully"
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
            detail="Failed to unlike content"
        )


@router.post("/share", response_model=Dict[str, Any])
async def share_content(
    request: ShareContentRequest = Depends(),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Share content."""
    try:
        result = await social_service.share_content(
            user_id=str(current_user.id),
            content_id=request.content_id,
            content_type=request.content_type,
            platform=request.platform,
            message=request.message
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Content shared successfully"
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
            detail="Failed to share content"
        )


@router.post("/bookmark", response_model=Dict[str, Any])
async def bookmark_content(
    request: BookmarkContentRequest = Depends(),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Bookmark content."""
    try:
        result = await social_service.bookmark_content(
            user_id=str(current_user.id),
            content_id=request.content_id,
            content_type=request.content_type,
            tags=request.tags
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Content bookmarked successfully"
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
            detail="Failed to bookmark content"
        )


@router.post("/remove-bookmark", response_model=Dict[str, Any])
async def remove_bookmark(
    request: BookmarkContentRequest = Depends(),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Remove bookmark."""
    try:
        result = await social_service.remove_bookmark(
            user_id=str(current_user.id),
            content_id=request.content_id,
            content_type=request.content_type
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Bookmark removed successfully"
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
            detail="Failed to remove bookmark"
        )


@router.post("/mention", response_model=Dict[str, Any])
async def mention_user(
    request: MentionUserRequest = Depends(),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Mention a user."""
    try:
        result = await social_service.mention_user(
            user_id=str(current_user.id),
            mentioned_user_id=request.mentioned_user_id,
            content_id=request.content_id,
            content_type=request.content_type,
            context=request.context
        )
        
        return {
            "success": True,
            "data": result,
            "message": "User mentioned successfully"
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
            detail="Failed to mention user"
        )


@router.post("/groups/create", response_model=Dict[str, Any])
async def create_social_group(
    request: CreateGroupRequest = Depends(),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a social group."""
    try:
        # Convert group type to enum
        try:
            group_type = SocialGroupType(request.group_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid group type: {request.group_type}")
        
        result = await social_service.create_social_group(
            name=request.name,
            description=request.description,
            creator_id=str(current_user.id),
            group_type=group_type,
            tags=request.tags,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Social group created successfully"
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
            detail="Failed to create social group"
        )


@router.post("/groups/join", response_model=Dict[str, Any])
async def join_social_group(
    request: JoinGroupRequest = Depends(),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Join a social group."""
    try:
        result = await social_service.join_social_group(
            user_id=str(current_user.id),
            group_id=request.group_id
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Joined social group successfully"
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
            detail="Failed to join social group"
        )


@router.post("/events/create", response_model=Dict[str, Any])
async def create_social_event(
    request: CreateEventRequest = Depends(),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a social event."""
    try:
        # Convert event type to enum
        try:
            event_type = SocialEventType(request.event_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid event type: {request.event_type}")
        
        result = await social_service.create_social_event(
            title=request.title,
            description=request.description,
            creator_id=str(current_user.id),
            event_type=event_type,
            start_date=request.start_date,
            end_date=request.end_date,
            location=request.location,
            max_attendees=request.max_attendees,
            tags=request.tags,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Social event created successfully"
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
            detail="Failed to create social event"
        )


@router.post("/events/attend", response_model=Dict[str, Any])
async def attend_social_event(
    request: AttendEventRequest = Depends(),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Attend a social event."""
    try:
        result = await social_service.attend_social_event(
            user_id=str(current_user.id),
            event_id=request.event_id
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Attending social event successfully"
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
            detail="Failed to attend social event"
        )


@router.post("/feed", response_model=Dict[str, Any])
async def get_social_feed(
    request: SocialFeedRequest = Depends(),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Get social feed."""
    try:
        result = await social_service.get_social_feed(
            user_id=str(current_user.id),
            page=request.page,
            page_size=request.page_size,
            feed_type=request.feed_type
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Social feed retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get social feed"
        )


@router.get("/feed", response_model=Dict[str, Any])
async def get_social_feed_get(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Page size"),
    feed_type: str = Query(default="following", description="Feed type"),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Get social feed via GET request."""
    try:
        result = await social_service.get_social_feed(
            user_id=str(current_user.id),
            page=page,
            page_size=page_size,
            feed_type=feed_type
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Social feed retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get social feed"
        )


@router.get("/metrics/{user_id}", response_model=Dict[str, Any])
async def get_user_social_metrics(
    user_id: str,
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Get user social metrics."""
    try:
        result = await social_service.get_user_social_metrics(user_id=user_id)
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Social metrics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get social metrics"
        )


@router.get("/recommendations", response_model=Dict[str, Any])
async def get_user_recommendations(
    limit: int = Query(default=10, ge=1, le=50, description="Number of recommendations"),
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Get user recommendations."""
    try:
        result = await social_service.get_user_recommendations(
            user_id=str(current_user.id),
            limit=limit
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "User recommendations retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user recommendations"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_social_stats(
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Get social system statistics."""
    try:
        result = await social_service.get_social_stats()
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Social statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get social statistics"
        )


@router.get("/activity-types", response_model=Dict[str, Any])
async def get_activity_types():
    """Get available social activity types."""
    activity_types = {
        "post_created": {
            "name": "Post Created",
            "description": "User created a new post",
            "visibility": "Public",
            "notification": "Optional"
        },
        "post_liked": {
            "name": "Post Liked",
            "description": "User liked a post",
            "visibility": "Public",
            "notification": "Yes"
        },
        "post_shared": {
            "name": "Post Shared",
            "description": "User shared a post",
            "visibility": "Public",
            "notification": "Yes"
        },
        "post_commented": {
            "name": "Post Commented",
            "description": "User commented on a post",
            "visibility": "Public",
            "notification": "Yes"
        },
        "user_followed": {
            "name": "User Followed",
            "description": "User followed another user",
            "visibility": "Public",
            "notification": "Yes"
        },
        "user_unfollowed": {
            "name": "User Unfollowed",
            "description": "User unfollowed another user",
            "visibility": "Private",
            "notification": "No"
        },
        "profile_updated": {
            "name": "Profile Updated",
            "description": "User updated their profile",
            "visibility": "Public",
            "notification": "Optional"
        },
        "bookmark_added": {
            "name": "Bookmark Added",
            "description": "User bookmarked content",
            "visibility": "Private",
            "notification": "No"
        },
        "bookmark_removed": {
            "name": "Bookmark Removed",
            "description": "User removed bookmark",
            "visibility": "Private",
            "notification": "No"
        },
        "mention_received": {
            "name": "Mention Received",
            "description": "User was mentioned in content",
            "visibility": "Public",
            "notification": "Yes"
        },
        "tag_created": {
            "name": "Tag Created",
            "description": "User created a new tag",
            "visibility": "Public",
            "notification": "Optional"
        },
        "group_joined": {
            "name": "Group Joined",
            "description": "User joined a social group",
            "visibility": "Public",
            "notification": "Optional"
        },
        "group_left": {
            "name": "Group Left",
            "description": "User left a social group",
            "visibility": "Private",
            "notification": "No"
        },
        "event_attending": {
            "name": "Event Attending",
            "description": "User is attending an event",
            "visibility": "Public",
            "notification": "Optional"
        },
        "event_created": {
            "name": "Event Created",
            "description": "User created a social event",
            "visibility": "Public",
            "notification": "Optional"
        }
    }
    
    return {
        "success": True,
        "data": {
            "activity_types": activity_types,
            "total_types": len(activity_types)
        },
        "message": "Activity types retrieved successfully"
    }


@router.get("/group-types", response_model=Dict[str, Any])
async def get_group_types():
    """Get available social group types."""
    group_types = {
        "public": {
            "name": "Public",
            "description": "Anyone can join and see content",
            "visibility": "Public",
            "join_requirement": "None"
        },
        "private": {
            "name": "Private",
            "description": "Invitation required to join",
            "visibility": "Private",
            "join_requirement": "Invitation"
        },
        "secret": {
            "name": "Secret",
            "description": "Hidden from public view",
            "visibility": "Secret",
            "join_requirement": "Invitation"
        },
        "community": {
            "name": "Community",
            "description": "Community-focused group",
            "visibility": "Public",
            "join_requirement": "Approval"
        },
        "professional": {
            "name": "Professional",
            "description": "Professional networking group",
            "visibility": "Public",
            "join_requirement": "Verification"
        },
        "interest": {
            "name": "Interest",
            "description": "Interest-based group",
            "visibility": "Public",
            "join_requirement": "None"
        }
    }
    
    return {
        "success": True,
        "data": {
            "group_types": group_types,
            "total_types": len(group_types)
        },
        "message": "Group types retrieved successfully"
    }


@router.get("/event-types", response_model=Dict[str, Any])
async def get_event_types():
    """Get available social event types."""
    event_types = {
        "virtual": {
            "name": "Virtual",
            "description": "Online event",
            "location": "Online",
            "max_attendees": "Unlimited"
        },
        "in_person": {
            "name": "In Person",
            "description": "Physical event",
            "location": "Physical venue",
            "max_attendees": "Limited"
        },
        "hybrid": {
            "name": "Hybrid",
            "description": "Both online and in-person",
            "location": "Multiple",
            "max_attendees": "Limited"
        },
        "workshop": {
            "name": "Workshop",
            "description": "Educational workshop",
            "location": "Flexible",
            "max_attendees": "Limited"
        },
        "conference": {
            "name": "Conference",
            "description": "Professional conference",
            "location": "Flexible",
            "max_attendees": "Large"
        },
        "meetup": {
            "name": "Meetup",
            "description": "Casual meetup",
            "location": "Flexible",
            "max_attendees": "Small"
        },
        "webinar": {
            "name": "Webinar",
            "description": "Online presentation",
            "location": "Online",
            "max_attendees": "Large"
        }
    }
    
    return {
        "success": True,
        "data": {
            "event_types": event_types,
            "total_types": len(event_types)
        },
        "message": "Event types retrieved successfully"
    }


@router.get("/health", response_model=Dict[str, Any])
async def get_social_health(
    social_service: AdvancedSocialService = Depends(get_social_service),
    current_user: CurrentUserDep = Depends()
):
    """Get social system health status."""
    try:
        # Get social stats
        stats = await social_service.get_social_stats()
        
        # Calculate health metrics
        total_users = stats["data"].get("total_users", 0)
        total_follows = stats["data"].get("total_follows", 0)
        total_activities = stats["data"].get("total_activities", 0)
        social_graph_nodes = stats["data"].get("social_graph_nodes", 0)
        social_graph_edges = stats["data"].get("social_graph_edges", 0)
        
        # Calculate health score
        health_score = 100
        
        # Check social engagement
        if total_users > 0:
            avg_follows_per_user = total_follows / total_users
            if avg_follows_per_user < 5:
                health_score -= 20
            elif avg_follows_per_user > 50:
                health_score -= 10
        
        # Check activity level
        if total_users > 0:
            avg_activities_per_user = total_activities / total_users
            if avg_activities_per_user < 10:
                health_score -= 15
        
        # Check social graph connectivity
        if social_graph_nodes > 0:
            connectivity = social_graph_edges / (social_graph_nodes * (social_graph_nodes - 1) / 2)
            if connectivity < 0.01:
                health_score -= 25
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "total_users": total_users,
                "total_follows": total_follows,
                "total_activities": total_activities,
                "social_graph_nodes": social_graph_nodes,
                "social_graph_edges": social_graph_edges,
                "avg_follows_per_user": avg_follows_per_user if total_users > 0 else 0,
                "avg_activities_per_user": avg_activities_per_user if total_users > 0 else 0,
                "graph_connectivity": connectivity if social_graph_nodes > 0 else 0,
                "timestamp": datetime.utcnow().isoformat()
            },
            "message": "Social health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get social health status"
        )

























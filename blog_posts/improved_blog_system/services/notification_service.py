"""
Real-time notification service using WebSockets
"""

import asyncio
import json
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.database import User, BlogPost, Comment
from ..core.exceptions import DatabaseError


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        # Store active connections by user ID
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Store connections by WebSocket for quick lookup
        self.connection_users: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept a WebSocket connection."""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        
        self.active_connections[user_id].add(websocket)
        self.connection_users[websocket] = user_id
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.connection_users:
            user_id = self.connection_users[websocket]
            
            if user_id in self.active_connections:
                self.active_connections[user_id].discard(websocket)
                
                # Remove user entry if no connections left
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
            
            del self.connection_users[websocket]
    
    async def send_personal_message(self, message: str, user_id: str):
        """Send a message to a specific user."""
        if user_id in self.active_connections:
            # Create a copy of the set to avoid modification during iteration
            connections = self.active_connections[user_id].copy()
            
            for connection in connections:
                try:
                    await connection.send_text(message)
                except Exception:
                    # Remove broken connections
                    self.disconnect(connection)
    
    async def send_to_multiple_users(self, message: str, user_ids: List[str]):
        """Send a message to multiple users."""
        tasks = []
        for user_id in user_ids:
            tasks.append(self.send_personal_message(message, user_id))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected users."""
        tasks = []
        for user_id in self.active_connections:
            tasks.append(self.send_personal_message(message, user_id))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_connected_users(self) -> List[str]:
        """Get list of connected user IDs."""
        return list(self.active_connections.keys())
    
    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return sum(len(connections) for connections in self.active_connections.values())


class NotificationService:
    """Service for managing real-time notifications."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.connection_manager = ConnectionManager()
    
    async def send_comment_notification(
        self,
        comment: Comment,
        post: BlogPost,
        commenter: User
    ):
        """Send notification when someone comments on a post."""
        try:
            # Get post author
            post_author_query = select(User).where(User.id == post.author_id)
            post_author_result = await self.session.execute(post_author_query)
            post_author = post_author_result.scalar_one_or_none()
            
            if not post_author:
                return
            
            # Don't notify the commenter about their own comment
            if post_author.id == commenter.id:
                return
            
            notification_data = {
                "type": "comment",
                "title": "New Comment",
                "message": f"{commenter.username} commented on your post '{post.title}'",
                "data": {
                    "comment_id": comment.id,
                    "post_id": post.id,
                    "post_title": post.title,
                    "commenter_username": commenter.username,
                    "comment_content": comment.content[:100] + "..." if len(comment.content) > 100 else comment.content
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await self.connection_manager.send_personal_message(
                json.dumps(notification_data),
                str(post_author.id)
            )
            
        except Exception as e:
            raise DatabaseError(f"Failed to send comment notification: {str(e)}")
    
    async def send_like_notification(
        self,
        post: BlogPost,
        liker: User
    ):
        """Send notification when someone likes a post."""
        try:
            # Get post author
            post_author_query = select(User).where(User.id == post.author_id)
            post_author_result = await self.session.execute(post_author_query)
            post_author = post_author_result.scalar_one_or_none()
            
            if not post_author:
                return
            
            # Don't notify the liker about their own like
            if post_author.id == liker.id:
                return
            
            notification_data = {
                "type": "like",
                "title": "Post Liked",
                "message": f"{liker.username} liked your post '{post.title}'",
                "data": {
                    "post_id": post.id,
                    "post_title": post.title,
                    "liker_username": liker.username
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await self.connection_manager.send_personal_message(
                json.dumps(notification_data),
                str(post_author.id)
            )
            
        except Exception as e:
            raise DatabaseError(f"Failed to send like notification: {str(e)}")
    
    async def send_follow_notification(
        self,
        follower: User,
        following: User
    ):
        """Send notification when someone follows a user."""
        try:
            notification_data = {
                "type": "follow",
                "title": "New Follower",
                "message": f"{follower.username} started following you",
                "data": {
                    "follower_id": str(follower.id),
                    "follower_username": follower.username,
                    "follower_avatar": follower.avatar_url
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await self.connection_manager.send_personal_message(
                json.dumps(notification_data),
                str(following.id)
            )
            
        except Exception as e:
            raise DatabaseError(f"Failed to send follow notification: {str(e)}")
    
    async def send_post_published_notification(
        self,
        post: BlogPost,
        author: User
    ):
        """Send notification when a post is published."""
        try:
            # Get followers of the author (in a real implementation, you'd have a followers table)
            # For now, we'll send to all connected users
            connected_users = self.connection_manager.get_connected_users()
            
            # Remove the author from the list
            connected_users = [uid for uid in connected_users if uid != str(author.id)]
            
            if not connected_users:
                return
            
            notification_data = {
                "type": "post_published",
                "title": "New Post Published",
                "message": f"{author.username} published a new post: '{post.title}'",
                "data": {
                    "post_id": post.id,
                    "post_title": post.title,
                    "post_slug": post.slug,
                    "author_username": author.username,
                    "post_excerpt": post.excerpt
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await self.connection_manager.send_to_multiple_users(
                json.dumps(notification_data),
                connected_users
            )
            
        except Exception as e:
            raise DatabaseError(f"Failed to send post published notification: {str(e)}")
    
    async def send_system_notification(
        self,
        title: str,
        message: str,
        user_ids: Optional[List[str]] = None,
        notification_type: str = "system"
    ):
        """Send a system notification."""
        try:
            notification_data = {
                "type": notification_type,
                "title": title,
                "message": message,
                "data": {},
                "timestamp": datetime.now().isoformat()
            }
            
            if user_ids:
                await self.connection_manager.send_to_multiple_users(
                    json.dumps(notification_data),
                    user_ids
                )
            else:
                await self.connection_manager.broadcast(
                    json.dumps(notification_data)
                )
                
        except Exception as e:
            raise DatabaseError(f"Failed to send system notification: {str(e)}")
    
    async def send_mention_notification(
        self,
        mentioned_user: User,
        mentioner: User,
        post: BlogPost,
        mention_context: str
    ):
        """Send notification when someone mentions a user."""
        try:
            notification_data = {
                "type": "mention",
                "title": "You were mentioned",
                "message": f"{mentioner.username} mentioned you in a post",
                "data": {
                    "post_id": post.id,
                    "post_title": post.title,
                    "mentioner_username": mentioner.username,
                    "mention_context": mention_context
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await self.connection_manager.send_personal_message(
                json.dumps(notification_data),
                str(mentioned_user.id)
            )
            
        except Exception as e:
            raise DatabaseError(f"Failed to send mention notification: {str(e)}")
    
    def get_connection_manager(self) -> ConnectionManager:
        """Get the connection manager instance."""
        return self.connection_manager
    
    async def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        try:
            connected_users = self.connection_manager.get_connected_users()
            total_connections = self.connection_manager.get_connection_count()
            
            return {
                "connected_users": len(connected_users),
                "total_connections": total_connections,
                "active_connections": {
                    user_id: len(connections)
                    for user_id, connections in self.connection_manager.active_connections.items()
                }
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get notification stats: {str(e)}")


# Global notification service instance
notification_service: Optional[NotificationService] = None


def get_notification_service(session: AsyncSession) -> NotificationService:
    """Get notification service instance."""
    global notification_service
    if notification_service is None:
        notification_service = NotificationService(session)
    return notification_service































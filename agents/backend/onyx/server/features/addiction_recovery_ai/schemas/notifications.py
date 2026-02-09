"""
Pydantic schemas for notifications endpoints
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class NotificationResponse(BaseModel):
    """Response schema for notification"""
    notification_id: str = Field(..., description="Notification ID")
    user_id: str = Field(..., description="User ID")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    type: str = Field(..., description="Notification type")
    priority: str = Field(default="normal", description="Priority level")
    read: bool = Field(default=False, description="Whether notification is read")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    read_at: Optional[datetime] = Field(default=None, description="Read timestamp")


class NotificationsListResponse(BaseModel):
    """Response schema for notifications list"""
    user_id: str = Field(..., description="User ID")
    notifications: List[NotificationResponse] = Field(default_factory=list, description="List of notifications")
    unread_count: int = Field(default=0, ge=0, description="Number of unread notifications")
    total_count: int = Field(default=0, ge=0, description="Total number of notifications")


class ReminderResponse(BaseModel):
    """Response schema for reminder"""
    reminder_id: str = Field(..., description="Reminder ID")
    user_id: str = Field(..., description="User ID")
    title: str = Field(..., description="Reminder title")
    message: str = Field(..., description="Reminder message")
    scheduled_time: datetime = Field(..., description="Scheduled time")
    completed: bool = Field(default=False, description="Whether reminder is completed")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class RemindersListResponse(BaseModel):
    """Response schema for reminders list"""
    user_id: str = Field(..., description="User ID")
    reminders: List[ReminderResponse] = Field(default_factory=list, description="List of reminders")
    upcoming_count: int = Field(default=0, ge=0, description="Number of upcoming reminders")


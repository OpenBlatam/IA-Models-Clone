"""
Email notification service
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
from pathlib import Path

from ..config.settings import get_settings
from ..core.exceptions import ExternalServiceError


class EmailService:
    """Service for sending email notifications."""
    
    def __init__(self):
        self.settings = get_settings()
        self.smtp_server = None
        self.smtp_port = 587
        self.sender_email = None
        self.sender_password = None
    
    async def send_welcome_email(self, user_email: str, username: str) -> bool:
        """Send welcome email to new user."""
        subject = "Welcome to Our Blog Platform!"
        
        html_content = f"""
        <html>
        <body>
            <h2>Welcome to Our Blog Platform!</h2>
            <p>Hello {username},</p>
            <p>Thank you for joining our blog platform. We're excited to have you as part of our community!</p>
            <p>You can now:</p>
            <ul>
                <li>Create and publish blog posts</li>
                <li>Comment on other posts</li>
                <li>Follow your favorite authors</li>
                <li>Customize your profile</li>
            </ul>
            <p>If you have any questions, feel free to reach out to our support team.</p>
            <p>Best regards,<br>The Blog Team</p>
        </body>
        </html>
        """
        
        return await self._send_email(user_email, subject, html_content)
    
    async def send_comment_notification(
        self,
        post_author_email: str,
        post_title: str,
        commenter_name: str,
        comment_content: str,
        post_url: str
    ) -> bool:
        """Send notification when someone comments on a post."""
        subject = f"New Comment on '{post_title}'"
        
        html_content = f"""
        <html>
        <body>
            <h2>New Comment on Your Post</h2>
            <p>Hello,</p>
            <p><strong>{commenter_name}</strong> has commented on your post "<strong>{post_title}</strong>":</p>
            <div style="background-color: #f5f5f5; padding: 15px; margin: 15px 0; border-left: 4px solid #007bff;">
                <p>"{comment_content}"</p>
            </div>
            <p><a href="{post_url}" style="background-color: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">View Post</a></p>
            <p>Best regards,<br>The Blog Team</p>
        </body>
        </html>
        """
        
        return await self._send_email(post_author_email, subject, html_content)
    
    async def send_post_published_notification(
        self,
        author_email: str,
        post_title: str,
        post_url: str
    ) -> bool:
        """Send notification when a post is published."""
        subject = f"Your Post '{post_title}' Has Been Published!"
        
        html_content = f"""
        <html>
        <body>
            <h2>Your Post Has Been Published!</h2>
            <p>Congratulations!</p>
            <p>Your post "<strong>{post_title}</strong>" has been successfully published and is now live on our platform.</p>
            <p><a href="{post_url}" style="background-color: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">View Your Post</a></p>
            <p>Share it with your network and engage with your readers!</p>
            <p>Best regards,<br>The Blog Team</p>
        </body>
        </html>
        """
        
        return await self._send_email(author_email, subject, html_content)
    
    async def send_password_reset_email(
        self,
        user_email: str,
        username: str,
        reset_token: str,
        reset_url: str
    ) -> bool:
        """Send password reset email."""
        subject = "Password Reset Request"
        
        html_content = f"""
        <html>
        <body>
            <h2>Password Reset Request</h2>
            <p>Hello {username},</p>
            <p>We received a request to reset your password. If you made this request, click the button below to reset your password:</p>
            <p><a href="{reset_url}" style="background-color: #dc3545; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Reset Password</a></p>
            <p>If you didn't request a password reset, please ignore this email. Your password will remain unchanged.</p>
            <p>This link will expire in 24 hours.</p>
            <p>Best regards,<br>The Blog Team</p>
        </body>
        </html>
        """
        
        return await self._send_email(user_email, subject, html_content)
    
    async def send_weekly_digest(
        self,
        user_email: str,
        username: str,
        posts: List[Dict[str, Any]]
    ) -> bool:
        """Send weekly digest of popular posts."""
        subject = "Weekly Blog Digest - Top Posts This Week"
        
        posts_html = ""
        for post in posts:
            posts_html += f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">
                <h3><a href="{post['url']}" style="color: #007bff; text-decoration: none;">{post['title']}</a></h3>
                <p style="color: #666;">by {post['author']}</p>
                <p>{post['excerpt']}</p>
                <p style="color: #999; font-size: 0.9em;">{post['view_count']} views • {post['like_count']} likes</p>
            </div>
            """
        
        html_content = f"""
        <html>
        <body>
            <h2>Weekly Blog Digest</h2>
            <p>Hello {username},</p>
            <p>Here are the most popular posts from this week:</p>
            {posts_html}
            <p>Happy reading!</p>
            <p>Best regards,<br>The Blog Team</p>
        </body>
        </html>
        """
        
        return await self._send_email(user_email, subject, html_content)
    
    async def _send_email(self, recipient_email: str, subject: str, html_content: str) -> bool:
        """Send email using SMTP."""
        try:
            # In a real implementation, you would configure SMTP settings
            # For now, we'll simulate email sending
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Log email sending (in production, you might use a proper email service)
            print(f"Email sent to {recipient_email}: {subject}")
            
            return True
            
        except Exception as e:
            raise ExternalServiceError(
                f"Failed to send email: {str(e)}",
                service_name="email"
            )
    
    def _create_smtp_connection(self) -> smtplib.SMTP:
        """Create SMTP connection (for real implementation)."""
        # This would be implemented with actual SMTP configuration
        # For now, we'll return None to indicate we're using a mock
        return None
    
    async def send_bulk_emails(
        self,
        recipients: List[str],
        subject: str,
        html_content: str
    ) -> Dict[str, Any]:
        """Send bulk emails to multiple recipients."""
        results = {
            "sent": 0,
            "failed": 0,
            "errors": []
        }
        
        for recipient in recipients:
            try:
                success = await self._send_email(recipient, subject, html_content)
                if success:
                    results["sent"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Failed to send to {recipient}")
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Error sending to {recipient}: {str(e)}")
        
        return results































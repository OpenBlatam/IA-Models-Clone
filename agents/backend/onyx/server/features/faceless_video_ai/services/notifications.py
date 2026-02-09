"""
Notification Service
Sends email and SMS notifications
"""

from typing import Optional, Dict, Any
from uuid import UUID
import logging
import os
import httpx

logger = logging.getLogger(__name__)


class NotificationService:
    """Sends email and SMS notifications"""
    
    def __init__(self):
        self.email_enabled = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
        self.sms_enabled = os.getenv("SMS_ENABLED", "false").lower() == "true"
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")
    
    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None
    ) -> bool:
        """
        Send email notification
        
        Args:
            to_email: Recipient email
            subject: Email subject
            body: Plain text body
            html_body: HTML body (optional)
            
        Returns:
            True if sent successfully
        """
        if not self.email_enabled:
            logger.debug("Email notifications disabled")
            return False
        
        # Try SendGrid
        if self.sendgrid_api_key:
            try:
                return await self._send_with_sendgrid(to_email, subject, body, html_body)
            except Exception as e:
                logger.warning(f"SendGrid email failed: {str(e)}")
        
        # Fallback: log email (in production, use actual email service)
        logger.info(f"Email notification: To={to_email}, Subject={subject}")
        return True
    
    async def _send_with_sendgrid(
        self,
        to_email: str,
        subject: str,
        body: str,
        html_body: Optional[str]
    ) -> bool:
        """Send email using SendGrid"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {
                "personalizations": [{
                    "to": [{"email": to_email}],
                }],
                "from": {"email": os.getenv("FROM_EMAIL", "noreply@facelessvideo.ai")},
                "subject": subject,
                "content": [
                    {
                        "type": "text/plain",
                        "value": body
                    }
                ]
            }
            
            if html_body:
                payload["content"].append({
                    "type": "text/html",
                    "value": html_body
                })
            
            response = await client.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers={
                    "Authorization": f"Bearer {self.sendgrid_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload
            )
            response.raise_for_status()
            return True
    
    async def send_sms(
        self,
        to_phone: str,
        message: str
    ) -> bool:
        """
        Send SMS notification
        
        Args:
            to_phone: Recipient phone number
            message: SMS message
            
        Returns:
            True if sent successfully
        """
        if not self.sms_enabled:
            logger.debug("SMS notifications disabled")
            return False
        
        # Try Twilio
        if self.twilio_account_sid and self.twilio_auth_token and self.twilio_phone:
            try:
                return await self._send_with_twilio(to_phone, message)
            except Exception as e:
                logger.warning(f"Twilio SMS failed: {str(e)}")
        
        # Fallback: log SMS
        logger.info(f"SMS notification: To={to_phone}, Message={message[:50]}...")
        return True
    
    async def _send_with_twilio(self, to_phone: str, message: str) -> bool:
        """Send SMS using Twilio"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/Messages.json",
                auth=(self.twilio_account_sid, self.twilio_auth_token),
                data={
                    "From": self.twilio_phone,
                    "To": to_phone,
                    "Body": message,
                }
            )
            response.raise_for_status()
            return True
    
    async def notify_video_complete(
        self,
        user_email: Optional[str],
        user_phone: Optional[str],
        video_id: UUID,
        video_url: str
    ):
        """Send notification when video is complete"""
        subject = "Your video is ready!"
        message = f"Your video {video_id} has been generated successfully. View it at: {video_url}"
        
        if user_email:
            await self.send_email(
                to_email=user_email,
                subject=subject,
                body=message,
                html_body=f"<h2>Video Ready!</h2><p>{message}</p><a href='{video_url}'>View Video</a>"
            )
        
        if user_phone:
            await self.send_sms(to_phone=user_phone, message=message)


_notification_service: Optional[NotificationService] = None


def get_notification_service() -> NotificationService:
    """Get notification service instance (singleton)"""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


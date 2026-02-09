from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiosmtplib
from typing import Any, List, Dict, Optional
"""
Email Delivery Service

This module provides email delivery functionality for the email sequence system.
"""



logger = logging.getLogger(__name__)


class EmailDeliveryService:
    """
    Service for delivering emails with support for multiple providers.
    """
    
    def __init__(
        self,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        smtp_username: str = None,
        smtp_password: str = None,
        use_tls: bool = True,
        from_email: str = "noreply@example.com",
        from_name: str = "Email Sequence System"
    ):
        """
        Initialize the email delivery service.
        
        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            smtp_username: SMTP username
            smtp_password: SMTP password
            use_tls: Use TLS encryption
            from_email: Default from email address
            from_name: Default from name
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.use_tls = use_tls
        self.from_email = from_email
        self.from_name = from_name
        
        # Delivery statistics
        self.total_sent = 0
        self.total_failed = 0
        self.delivery_queue = asyncio.Queue()
        
        logger.info("Email Delivery Service initialized")
    
    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str = None,
        from_email: str = None,
        from_name: str = None,
        reply_to: str = None,
        headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Send an email asynchronously.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML content
            text_content: Plain text content
            from_email: Sender email (uses default if not provided)
            from_name: Sender name (uses default if not provided)
            reply_to: Reply-to email address
            headers: Additional email headers
            
        Returns:
            Delivery result
        """
        try:
            # Prepare email message
            message = self._create_email_message(
                to_email=to_email,
                subject=subject,
                html_content=html_content,
                text_content=text_content,
                from_email=from_email or self.from_email,
                from_name=from_name or self.from_name,
                reply_to=reply_to,
                headers=headers
            )
            
            # Send email
            result = await self._send_email_async(message)
            
            # Update statistics
            if result['success']:
                self.total_sent += 1
            else:
                self.total_failed += 1
            
            logger.info(f"Email sent to {to_email}: {result['success']}")
            return result
            
        except Exception as e:
            logger.error(f"Error sending email to {to_email}: {e}")
            self.total_failed += 1
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow(),
                'to_email': to_email
            }
    
    async def send_bulk_emails(
        self,
        emails: List[Dict[str, Any]],
        batch_size: int = 100,
        delay_between_batches: float = 1.0
    ) -> Dict[str, Any]:
        """
        Send multiple emails in batches.
        
        Args:
            emails: List of email data dictionaries
            batch_size: Number of emails per batch
            delay_between_batches: Delay between batches in seconds
            
        Returns:
            Bulk delivery results
        """
        try:
            total_emails = len(emails)
            successful_sends = 0
            failed_sends = 0
            results = []
            
            # Process emails in batches
            for i in range(0, total_emails, batch_size):
                batch = emails[i:i + batch_size]
                
                # Send batch
                batch_tasks = []
                for email_data in batch:
                    task = self.send_email(**email_data)
                    batch_tasks.append(task)
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process batch results
                for result in batch_results:
                    if isinstance(result, Exception):
                        failed_sends += 1
                        results.append({
                            'success': False,
                            'error': str(result)
                        })
                    else:
                        if result['success']:
                            successful_sends += 1
                        else:
                            failed_sends += 1
                        results.append(result)
                
                # Delay between batches
                if i + batch_size < total_emails:
                    await asyncio.sleep(delay_between_batches)
            
            logger.info(f"Bulk email delivery completed: {successful_sends} successful, {failed_sends} failed")
            
            return {
                'total_emails': total_emails,
                'successful_sends': successful_sends,
                'failed_sends': failed_sends,
                'success_rate': (successful_sends / total_emails) * 100 if total_emails > 0 else 0,
                'results': results,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error in bulk email delivery: {e}")
            raise
    
    async def queue_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str = None,
        priority: int = 1
    ) -> bool:
        """
        Queue an email for later delivery.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML content
            text_content: Plain text content
            priority: Email priority (1-10, higher is more important)
            
        Returns:
            True if queued successfully
        """
        try:
            email_data = {
                'to_email': to_email,
                'subject': subject,
                'html_content': html_content,
                'text_content': text_content,
                'priority': priority,
                'timestamp': datetime.utcnow()
            }
            
            await self.delivery_queue.put(email_data)
            logger.info(f"Email queued for {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error queuing email for {to_email}: {e}")
            return False
    
    async def process_delivery_queue(self) -> Any:
        """Process the delivery queue"""
        while True:
            try:
                # Get email from queue
                email_data = await self.delivery_queue.get()
                
                # Send email
                await self.send_email(
                    to_email=email_data['to_email'],
                    subject=email_data['subject'],
                    html_content=email_data['html_content'],
                    text_content=email_data.get('text_content')
                )
                
                # Mark task as done
                self.delivery_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing delivery queue: {e}")
    
    def _create_email_message(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str = None,
        from_email: str = None,
        from_name: str = None,
        reply_to: str = None,
        headers: Dict[str, str] = None
    ) -> MIMEMultipart:
        """Create an email message"""
        message = MIMEMultipart('alternative')
        
        # Set headers
        message['Subject'] = subject
        message['From'] = f"{from_name} <{from_email}>"
        message['To'] = to_email
        
        if reply_to:
            message['Reply-To'] = reply_to
        
        # Add custom headers
        if headers:
            for key, value in headers.items():
                message[key] = value
        
        # Add content
        if text_content:
            text_part = MIMEText(text_content, 'plain', 'utf-8')
            message.attach(text_part)
        
        html_part = MIMEText(html_content, 'html', 'utf-8')
        message.attach(html_part)
        
        return message
    
    async def _send_email_async(self, message: MIMEMultipart) -> Dict[str, Any]:
        """Send email asynchronously"""
        try:
            # Use aiosmtplib for async SMTP
            await aiosmtplib.send(
                message,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_username,
                password=self.smtp_password,
                use_tls=self.use_tls
            )
            
            return {
                'success': True,
                'timestamp': datetime.utcnow(),
                'to_email': message['To'],
                'message_id': message.get('Message-ID', 'unknown')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow(),
                'to_email': message['To']
            }
    
    def get_delivery_statistics(self) -> Dict[str, Any]:
        """Get delivery statistics"""
        return {
            'total_sent': self.total_sent,
            'total_failed': self.total_failed,
            'success_rate': (self.total_sent / (self.total_sent + self.total_failed)) * 100 if (self.total_sent + self.total_failed) > 0 else 0,
            'queue_size': self.delivery_queue.qsize(),
            'timestamp': datetime.utcnow()
        }
    
    async def close(self) -> Any:
        """Close the delivery service"""
        try:
            # Wait for queue to be processed
            await self.delivery_queue.join()
            logger.info("Email Delivery Service closed")
        except Exception as e:
            logger.error(f"Error closing Email Delivery Service: {e}")
            raise 
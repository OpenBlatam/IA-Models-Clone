"""
Gamma App - Email Utilities
Advanced email processing and template utilities
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import jinja2
import re

logger = logging.getLogger(__name__)

class EmailPriority(Enum):
    """Email priority levels"""
    LOW = "5"
    NORMAL = "3"
    HIGH = "1"

@dataclass
class EmailAttachment:
    """Email attachment"""
    filename: str
    content: bytes
    content_type: str = "application/octet-stream"
    disposition: str = "attachment"

@dataclass
class EmailMessage:
    """Email message"""
    to: Union[str, List[str]]
    subject: str
    body: str
    html_body: Optional[str] = None
    from_email: Optional[str] = None
    reply_to: Optional[str] = None
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    priority: EmailPriority = EmailPriority.NORMAL
    attachments: Optional[List[EmailAttachment]] = None
    headers: Optional[Dict[str, str]] = None

class EmailTemplate:
    """Email template class"""
    
    def __init__(self, template_path: str):
        self.template_path = Path(template_path)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_path.parent)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render email template"""
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            raise
    
    def render_html(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render HTML email template"""
        return self.render(template_name, context)
    
    def render_text(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render text email template"""
        return self.render(template_name, context)

class EmailValidator:
    """Email validation utilities"""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email address format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def normalize_email(email: str) -> str:
        """Normalize email address"""
        return email.lower().strip()
    
    @staticmethod
    def extract_domain(email: str) -> str:
        """Extract domain from email address"""
        if '@' in email:
            return email.split('@')[1].lower()
        return ""
    
    @staticmethod
    def is_disposable_email(email: str) -> bool:
        """Check if email is from disposable email service"""
        disposable_domains = [
            '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
            'mailinator.com', 'yopmail.com', 'temp-mail.org'
        ]
        domain = EmailValidator.extract_domain(email)
        return domain in disposable_domains

class EmailService:
    """Advanced email service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.use_tls = config.get('use_tls', True)
        self.use_ssl = config.get('use_ssl', False)
        self.from_email = config.get('from_email', 'noreply@gamma.app')
        self.from_name = config.get('from_name', 'Gamma App')
        
        # Initialize template engine
        template_path = config.get('template_path', './templates/email')
        self.template_engine = EmailTemplate(template_path)
        
        # Initialize validator
        self.validator = EmailValidator()
    
    def send_email(self, message: EmailMessage) -> bool:
        """Send email message"""
        try:
            # Validate recipients
            recipients = self._validate_recipients(message.to)
            if not recipients:
                logger.error("No valid recipients found")
                return False
            
            # Create message
            msg = self._create_message(message)
            
            # Send email
            return self._send_message(msg, recipients)
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def send_template_email(
        self,
        to: Union[str, List[str]],
        template_name: str,
        context: Dict[str, Any],
        subject: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Send email using template"""
        try:
            # Render template
            html_content = self.template_engine.render_html(f"{template_name}.html", context)
            text_content = self.template_engine.render_text(f"{template_name}.txt", context)
            
            # Create message
            message = EmailMessage(
                to=to,
                subject=subject or context.get('subject', 'Notification from Gamma App'),
                body=text_content,
                html_body=html_content,
                **kwargs
            )
            
            return self.send_email(message)
            
        except Exception as e:
            logger.error(f"Error sending template email: {e}")
            return False
    
    def send_bulk_email(
        self,
        messages: List[EmailMessage],
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """Send bulk emails"""
        results = {
            'total': len(messages),
            'sent': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            # Process in batches
            for i in range(0, len(messages), batch_size):
                batch = messages[i:i + batch_size]
                
                for message in batch:
                    if self.send_email(message):
                        results['sent'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Failed to send email to {message.to}")
                
                # Add delay between batches to avoid rate limiting
                if i + batch_size < len(messages):
                    import time
                    time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error sending bulk emails: {e}")
            results['errors'].append(str(e))
        
        return results
    
    def _validate_recipients(self, recipients: Union[str, List[str]]) -> List[str]:
        """Validate email recipients"""
        if isinstance(recipients, str):
            recipients = [recipients]
        
        valid_recipients = []
        for recipient in recipients:
            normalized = self.validator.normalize_email(recipient)
            if self.validator.is_valid_email(normalized):
                valid_recipients.append(normalized)
            else:
                logger.warning(f"Invalid email address: {recipient}")
        
        return valid_recipients
    
    def _create_message(self, message: EmailMessage) -> MIMEMultipart:
        """Create email message"""
        # Create message
        msg = MIMEMultipart('alternative')
        
        # Set headers
        msg['From'] = f"{self.from_name} <{message.from_email or self.from_email}>"
        msg['To'] = ', '.join(message.to) if isinstance(message.to, list) else message.to
        msg['Subject'] = message.subject
        msg['X-Priority'] = message.priority.value
        
        if message.reply_to:
            msg['Reply-To'] = message.reply_to
        
        if message.cc:
            msg['Cc'] = ', '.join(message.cc)
        
        if message.headers:
            for key, value in message.headers.items():
                msg[key] = value
        
        # Add text content
        text_part = MIMEText(message.body, 'plain', 'utf-8')
        msg.attach(text_part)
        
        # Add HTML content if provided
        if message.html_body:
            html_part = MIMEText(message.html_body, 'html', 'utf-8')
            msg.attach(html_part)
        
        # Add attachments
        if message.attachments:
            for attachment in message.attachments:
                self._add_attachment(msg, attachment)
        
        return msg
    
    def _add_attachment(self, msg: MIMEMultipart, attachment: EmailAttachment):
        """Add attachment to message"""
        try:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.content)
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'{attachment.disposition}; filename= {attachment.filename}'
            )
            msg.attach(part)
        except Exception as e:
            logger.error(f"Error adding attachment: {e}")
    
    def _send_message(self, msg: MIMEMultipart, recipients: List[str]) -> bool:
        """Send email message via SMTP"""
        try:
            # Create SMTP connection
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            
            # Enable TLS if configured
            if self.use_tls and not self.use_ssl:
                server.starttls()
            
            # Login if credentials provided
            if self.username and self.password:
                server.login(self.username, self.password)
            
            # Send email
            text = msg.as_string()
            server.sendmail(self.from_email, recipients, text)
            server.quit()
            
            logger.info(f"Email sent successfully to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email via SMTP: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test SMTP connection"""
        try:
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            
            if self.use_tls and not self.use_ssl:
                server.starttls()
            
            if self.username and self.password:
                server.login(self.username, self.password)
            
            server.quit()
            return True
            
        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            return False
    
    def get_delivery_status(self, message_id: str) -> Dict[str, Any]:
        """Get email delivery status (placeholder for future implementation)"""
        # This would integrate with email service provider APIs
        return {
            'message_id': message_id,
            'status': 'delivered',
            'timestamp': '2024-01-01T00:00:00Z'
        }

class EmailTemplateManager:
    """Email template management"""
    
    def __init__(self, template_path: str):
        self.template_path = Path(template_path)
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load email templates"""
        try:
            for template_file in self.template_path.glob('*.html'):
                template_name = template_file.stem
                self.templates[template_name] = {
                    'html': template_file.read_text(encoding='utf-8'),
                    'text': self._convert_html_to_text(template_file.read_text(encoding='utf-8'))
                }
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
    
    def _convert_html_to_text(self, html_content: str) -> str:
        """Convert HTML to plain text"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text()
        except ImportError:
            # Fallback to regex if BeautifulSoup not available
            import re
            text = re.sub(r'<[^>]+>', '', html_content)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except Exception as e:
            logger.error(f"Error converting HTML to text: {e}")
            return html_content
    
    def get_template(self, template_name: str) -> Optional[Dict[str, str]]:
        """Get email template"""
        return self.templates.get(template_name)
    
    def list_templates(self) -> List[str]:
        """List available templates"""
        return list(self.templates.keys())
    
    def create_template(
        self,
        template_name: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """Create new email template"""
        try:
            if text_content is None:
                text_content = self._convert_html_to_text(html_content)
            
            # Save HTML template
            html_file = self.template_path / f"{template_name}.html"
            html_file.write_text(html_content, encoding='utf-8')
            
            # Save text template
            text_file = self.template_path / f"{template_name}.txt"
            text_file.write_text(text_content, encoding='utf-8')
            
            # Update in-memory templates
            self.templates[template_name] = {
                'html': html_content,
                'text': text_content
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            return False
    
    def delete_template(self, template_name: str) -> bool:
        """Delete email template"""
        try:
            # Delete files
            html_file = self.template_path / f"{template_name}.html"
            text_file = self.template_path / f"{template_name}.txt"
            
            if html_file.exists():
                html_file.unlink()
            if text_file.exists():
                text_file.unlink()
            
            # Remove from in-memory templates
            if template_name in self.templates:
                del self.templates[template_name]
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting template: {e}")
            return False

# Global email service instance
email_service = None

def initialize_email_service(config: Dict[str, Any]):
    """Initialize global email service"""
    global email_service
    email_service = EmailService(config)

def send_email(message: EmailMessage) -> bool:
    """Send email using global service"""
    if email_service is None:
        raise RuntimeError("Email service not initialized")
    return email_service.send_email(message)

def send_template_email(
    to: Union[str, List[str]],
    template_name: str,
    context: Dict[str, Any],
    subject: Optional[str] = None,
    **kwargs
) -> bool:
    """Send template email using global service"""
    if email_service is None:
        raise RuntimeError("Email service not initialized")
    return email_service.send_template_email(to, template_name, context, subject, **kwargs)

def test_email_connection() -> bool:
    """Test email connection using global service"""
    if email_service is None:
        raise RuntimeError("Email service not initialized")
    return email_service.test_connection()


























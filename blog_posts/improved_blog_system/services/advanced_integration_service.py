"""
Advanced Integration Service for third-party integrations and API management
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from dataclasses import dataclass
import hashlib
import hmac
import base64

from ..models.database import Integration, WebhookEvent, APICall
from ..core.exceptions import DatabaseError, ValidationError, ExternalServiceError


@dataclass
class IntegrationConfig:
    """Integration configuration."""
    name: str
    service_type: str
    base_url: str
    api_key: str
    webhook_secret: Optional[str] = None
    rate_limit: int = 100
    timeout: int = 30


class AdvancedIntegrationService:
    """Service for advanced third-party integrations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.integrations = {}
        self.webhook_handlers = {}
        self._initialize_default_integrations()
    
    def _initialize_default_integrations(self):
        """Initialize default integrations."""
        # Social Media Integrations
        self.integrations["twitter"] = IntegrationConfig(
            name="Twitter",
            service_type="social_media",
            base_url="https://api.twitter.com/2",
            api_key="",
            rate_limit=300
        )
        
        self.integrations["facebook"] = IntegrationConfig(
            name="Facebook",
            service_type="social_media",
            base_url="https://graph.facebook.com/v18.0",
            api_key="",
            rate_limit=200
        )
        
        self.integrations["linkedin"] = IntegrationConfig(
            name="LinkedIn",
            service_type="social_media",
            base_url="https://api.linkedin.com/v2",
            api_key="",
            rate_limit=100
        )
        
        # Email Service Integrations
        self.integrations["sendgrid"] = IntegrationConfig(
            name="SendGrid",
            service_type="email",
            base_url="https://api.sendgrid.com/v3",
            api_key="",
            rate_limit=600
        )
        
        self.integrations["mailchimp"] = IntegrationConfig(
            name="Mailchimp",
            service_type="email",
            base_url="https://us1.api.mailchimp.com/3.0",
            api_key="",
            rate_limit=10
        )
        
        # Analytics Integrations
        self.integrations["google_analytics"] = IntegrationConfig(
            name="Google Analytics",
            service_type="analytics",
            base_url="https://analyticsreporting.googleapis.com/v4",
            api_key="",
            rate_limit=100
        )
        
        self.integrations["mixpanel"] = IntegrationConfig(
            name="Mixpanel",
            service_type="analytics",
            base_url="https://mixpanel.com/api/2.0",
            api_key="",
            rate_limit=1000
        )
        
        # Payment Integrations
        self.integrations["stripe"] = IntegrationConfig(
            name="Stripe",
            service_type="payment",
            base_url="https://api.stripe.com/v1",
            api_key="",
            rate_limit=100
        )
        
        self.integrations["paypal"] = IntegrationConfig(
            name="PayPal",
            service_type="payment",
            base_url="https://api.paypal.com/v1",
            api_key="",
            rate_limit=500
        )
        
        # CDN Integrations
        self.integrations["cloudflare"] = IntegrationConfig(
            name="Cloudflare",
            service_type="cdn",
            base_url="https://api.cloudflare.com/client/v4",
            api_key="",
            rate_limit=1200
        )
        
        self.integrations["aws_cloudfront"] = IntegrationConfig(
            name="AWS CloudFront",
            service_type="cdn",
            base_url="https://cloudfront.amazonaws.com/2020-05-31",
            api_key="",
            rate_limit=100
        )
    
    async def make_api_call(
        self,
        integration_name: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make an API call to an integrated service."""
        try:
            if integration_name not in self.integrations:
                raise ValidationError(f"Integration '{integration_name}' not found")
            
            config = self.integrations[integration_name]
            
            # Prepare headers
            request_headers = {
                "Content-Type": "application/json",
                "User-Agent": "BlogSystem/1.0"
            }
            
            if config.api_key:
                if integration_name == "stripe":
                    request_headers["Authorization"] = f"Bearer {config.api_key}"
                elif integration_name in ["sendgrid", "mailchimp"]:
                    request_headers["Authorization"] = f"Bearer {config.api_key}"
                else:
                    request_headers["Authorization"] = f"Bearer {config.api_key}"
            
            if headers:
                request_headers.update(headers)
            
            # Make the API call
            url = f"{config.base_url}{endpoint}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout)) as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=request_headers) as response:
                        result = await response.json()
                elif method.upper() == "POST":
                    async with session.post(url, headers=request_headers, json=data) as response:
                        result = await response.json()
                elif method.upper() == "PUT":
                    async with session.put(url, headers=request_headers, json=data) as response:
                        result = await response.json()
                elif method.upper() == "DELETE":
                    async with session.delete(url, headers=request_headers) as response:
                        result = await response.json()
                else:
                    raise ValidationError(f"Unsupported HTTP method: {method}")
                
                # Log the API call
                await self._log_api_call(
                    integration_name=integration_name,
                    endpoint=endpoint,
                    method=method,
                    status_code=response.status,
                    response_data=result
                )
                
                return {
                    "success": response.status < 400,
                    "status_code": response.status,
                    "data": result,
                    "headers": dict(response.headers)
                }
                
        except aiohttp.ClientError as e:
            raise ExternalServiceError(f"API call failed: {str(e)}", service_name=integration_name)
        except Exception as e:
            raise DatabaseError(f"Failed to make API call: {str(e)}")
    
    async def _log_api_call(
        self,
        integration_name: str,
        endpoint: str,
        method: str,
        status_code: int,
        response_data: Dict[str, Any]
    ):
        """Log API call details."""
        try:
            api_call = APICall(
                integration_name=integration_name,
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_data=response_data,
                timestamp=datetime.utcnow()
            )
            
            self.session.add(api_call)
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            # Don't raise error for logging failures
    
    async def post_to_social_media(
        self,
        platform: str,
        content: str,
        media_urls: Optional[List[str]] = None,
        hashtags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Post content to social media platforms."""
        try:
            if platform not in ["twitter", "facebook", "linkedin"]:
                raise ValidationError(f"Unsupported social media platform: {platform}")
            
            # Prepare post data based on platform
            if platform == "twitter":
                endpoint = "/tweets"
                post_data = {
                    "text": content
                }
                if media_urls:
                    post_data["media"] = {"media_ids": media_urls}
            
            elif platform == "facebook":
                endpoint = "/me/feed"
                post_data = {
                    "message": content
                }
                if media_urls:
                    post_data["link"] = media_urls[0]
            
            elif platform == "linkedin":
                endpoint = "/ugcPosts"
                post_data = {
                    "author": "urn:li:person:YOUR_PERSON_URN",
                    "lifecycleState": "PUBLISHED",
                    "specificContent": {
                        "com.linkedin.ugc.ShareContent": {
                            "shareCommentary": {
                                "text": content
                            },
                            "shareMediaCategory": "NONE"
                        }
                    },
                    "visibility": {
                        "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
                    }
                }
            
            # Add hashtags if provided
            if hashtags:
                hashtag_text = " ".join([f"#{tag}" for tag in hashtags])
                if platform == "twitter":
                    post_data["text"] += f" {hashtag_text}"
                elif platform == "facebook":
                    post_data["message"] += f" {hashtag_text}"
                elif platform == "linkedin":
                    post_data["specificContent"]["com.linkedin.ugc.ShareContent"]["shareCommentary"]["text"] += f" {hashtag_text}"
            
            # Make the API call
            result = await self.make_api_call(
                integration_name=platform,
                endpoint=endpoint,
                method="POST",
                data=post_data
            )
            
            return {
                "platform": platform,
                "success": result["success"],
                "post_id": result["data"].get("id") if result["success"] else None,
                "response": result["data"]
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to post to social media: {str(e)}")
    
    async def send_email_via_service(
        self,
        service: str,
        to_email: str,
        subject: str,
        content: str,
        from_email: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send email via integrated email service."""
        try:
            if service not in ["sendgrid", "mailchimp"]:
                raise ValidationError(f"Unsupported email service: {service}")
            
            if service == "sendgrid":
                endpoint = "/mail/send"
                email_data = {
                    "personalizations": [
                        {
                            "to": [{"email": to_email}],
                            "subject": subject
                        }
                    ],
                    "from": {"email": from_email or "noreply@blogsystem.com"},
                    "content": [
                        {
                            "type": "text/html",
                            "value": content
                        }
                    ]
                }
            
            elif service == "mailchimp":
                # This would require a campaign setup in Mailchimp
                # For simplicity, we'll use a mock implementation
                endpoint = "/campaigns"
                email_data = {
                    "type": "regular",
                    "recipients": {
                        "list_id": "YOUR_LIST_ID"
                    },
                    "settings": {
                        "subject_line": subject,
                        "from_name": "Blog System",
                        "reply_to": from_email or "noreply@blogsystem.com"
                    }
                }
            
            # Make the API call
            result = await self.make_api_call(
                integration_name=service,
                endpoint=endpoint,
                method="POST",
                data=email_data
            )
            
            return {
                "service": service,
                "success": result["success"],
                "message_id": result["data"].get("message_id") if result["success"] else None,
                "response": result["data"]
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to send email: {str(e)}")
    
    async def track_analytics_event(
        self,
        service: str,
        event_name: str,
        properties: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Track analytics event via integrated service."""
        try:
            if service not in ["google_analytics", "mixpanel"]:
                raise ValidationError(f"Unsupported analytics service: {service}")
            
            if service == "google_analytics":
                endpoint = "/reports:batchGet"
                analytics_data = {
                    "reportRequests": [
                        {
                            "viewId": "YOUR_VIEW_ID",
                            "dateRanges": [
                                {
                                    "startDate": datetime.now().strftime("%Y-%m-%d"),
                                    "endDate": datetime.now().strftime("%Y-%m-%d")
                                }
                            ],
                            "metrics": [
                                {"expression": "ga:pageviews"},
                                {"expression": "ga:sessions"}
                            ]
                        }
                    ]
                }
            
            elif service == "mixpanel":
                endpoint = "/track"
                analytics_data = {
                    "event": event_name,
                    "properties": {
                        **properties,
                        "distinct_id": user_id or "anonymous",
                        "time": int(datetime.now().timestamp())
                    }
                }
            
            # Make the API call
            result = await self.make_api_call(
                integration_name=service,
                endpoint=endpoint,
                method="POST",
                data=analytics_data
            )
            
            return {
                "service": service,
                "success": result["success"],
                "event_name": event_name,
                "response": result["data"]
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to track analytics event: {str(e)}")
    
    async def process_payment(
        self,
        service: str,
        amount: float,
        currency: str,
        description: str,
        customer_email: str
    ) -> Dict[str, Any]:
        """Process payment via integrated payment service."""
        try:
            if service not in ["stripe", "paypal"]:
                raise ValidationError(f"Unsupported payment service: {service}")
            
            if service == "stripe":
                endpoint = "/payment_intents"
                payment_data = {
                    "amount": int(amount * 100),  # Convert to cents
                    "currency": currency.lower(),
                    "description": description,
                    "receipt_email": customer_email
                }
            
            elif service == "paypal":
                endpoint = "/payments/payment"
                payment_data = {
                    "intent": "sale",
                    "payer": {
                        "payment_method": "paypal"
                    },
                    "transactions": [
                        {
                            "amount": {
                                "total": str(amount),
                                "currency": currency.upper()
                            },
                            "description": description
                        }
                    ],
                    "redirect_urls": {
                        "return_url": "https://blogsystem.com/payment/success",
                        "cancel_url": "https://blogsystem.com/payment/cancel"
                    }
                }
            
            # Make the API call
            result = await self.make_api_call(
                integration_name=service,
                endpoint=endpoint,
                method="POST",
                data=payment_data
            )
            
            return {
                "service": service,
                "success": result["success"],
                "payment_id": result["data"].get("id") if result["success"] else None,
                "response": result["data"]
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to process payment: {str(e)}")
    
    async def setup_webhook(
        self,
        integration_name: str,
        webhook_url: str,
        events: List[str]
    ) -> Dict[str, Any]:
        """Setup webhook for an integration."""
        try:
            if integration_name not in self.integrations:
                raise ValidationError(f"Integration '{integration_name}' not found")
            
            config = self.integrations[integration_name]
            
            # Generate webhook secret
            webhook_secret = self._generate_webhook_secret()
            
            # Store webhook configuration
            integration = Integration(
                name=integration_name,
                webhook_url=webhook_url,
                webhook_secret=webhook_secret,
                events=events,
                is_active=True,
                created_at=datetime.utcnow()
            )
            
            self.session.add(integration)
            await self.session.commit()
            
            # Register webhook with the service
            webhook_data = {
                "url": webhook_url,
                "events": events,
                "secret": webhook_secret
            }
            
            # Make API call to register webhook (implementation varies by service)
            result = await self.make_api_call(
                integration_name=integration_name,
                endpoint="/webhooks",
                method="POST",
                data=webhook_data
            )
            
            return {
                "integration_name": integration_name,
                "webhook_url": webhook_url,
                "webhook_secret": webhook_secret,
                "events": events,
                "registration_success": result["success"],
                "response": result["data"]
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to setup webhook: {str(e)}")
    
    def _generate_webhook_secret(self) -> str:
        """Generate a secure webhook secret."""
        import secrets
        return secrets.token_urlsafe(32)
    
    async def handle_webhook(
        self,
        integration_name: str,
        payload: Dict[str, Any],
        signature: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle incoming webhook."""
        try:
            # Verify webhook signature if provided
            if signature:
                is_valid = await self._verify_webhook_signature(
                    integration_name=integration_name,
                    payload=payload,
                    signature=signature
                )
                if not is_valid:
                    raise ValidationError("Invalid webhook signature")
            
            # Store webhook event
            webhook_event = WebhookEvent(
                integration_name=integration_name,
                event_type=payload.get("type", "unknown"),
                payload=payload,
                signature=signature,
                received_at=datetime.utcnow()
            )
            
            self.session.add(webhook_event)
            await self.session.commit()
            
            # Process webhook based on integration type
            result = await self._process_webhook_event(integration_name, payload)
            
            return {
                "integration_name": integration_name,
                "event_type": payload.get("type", "unknown"),
                "processed": True,
                "result": result
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to handle webhook: {str(e)}")
    
    async def _verify_webhook_signature(
        self,
        integration_name: str,
        payload: Dict[str, Any],
        signature: str
    ) -> bool:
        """Verify webhook signature."""
        try:
            # Get webhook secret
            integration_query = select(Integration).where(
                and_(
                    Integration.name == integration_name,
                    Integration.is_active == True
                )
            )
            integration_result = await self.session.execute(integration_query)
            integration = integration_result.scalar_one_or_none()
            
            if not integration or not integration.webhook_secret:
                return False
            
            # Verify signature (implementation varies by service)
            if integration_name == "stripe":
                # Stripe signature verification
                expected_signature = hmac.new(
                    integration.webhook_secret.encode(),
                    json.dumps(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                return hmac.compare_digest(signature, expected_signature)
            
            elif integration_name == "github":
                # GitHub signature verification
                expected_signature = "sha256=" + hmac.new(
                    integration.webhook_secret.encode(),
                    json.dumps(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                return hmac.compare_digest(signature, expected_signature)
            
            # Default verification
            return True
            
        except Exception as e:
            return False
    
    async def _process_webhook_event(
        self,
        integration_name: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process webhook event based on integration type."""
        try:
            event_type = payload.get("type", "unknown")
            
            if integration_name == "stripe":
                if event_type == "payment_intent.succeeded":
                    # Handle successful payment
                    return {"action": "payment_success", "data": payload["data"]}
                elif event_type == "payment_intent.payment_failed":
                    # Handle failed payment
                    return {"action": "payment_failed", "data": payload["data"]}
            
            elif integration_name == "github":
                if event_type == "push":
                    # Handle code push
                    return {"action": "code_push", "data": payload}
                elif event_type == "pull_request":
                    # Handle pull request
                    return {"action": "pull_request", "data": payload}
            
            # Default processing
            return {"action": "processed", "data": payload}
            
        except Exception as e:
            return {"action": "error", "error": str(e)}
    
    async def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        try:
            # Get total integrations
            total_integrations_query = select(func.count(Integration.id))
            total_integrations_result = await self.session.execute(total_integrations_query)
            total_integrations = total_integrations_result.scalar()
            
            # Get active integrations
            active_integrations_query = select(func.count(Integration.id)).where(
                Integration.is_active == True
            )
            active_integrations_result = await self.session.execute(active_integrations_query)
            active_integrations = active_integrations_result.scalar()
            
            # Get API call statistics
            api_calls_query = select(
                APICall.integration_name,
                func.count(APICall.id).label('total_calls'),
                func.count(func.case((APICall.status_code < 400, 1))).label('successful_calls')
            ).group_by(APICall.integration_name)
            
            api_calls_result = await self.session.execute(api_calls_query)
            api_calls_stats = {}
            for row in api_calls_result:
                api_calls_stats[row.integration_name] = {
                    "total_calls": row.total_calls,
                    "successful_calls": row.successful_calls,
                    "success_rate": (row.successful_calls / row.total_calls * 100) if row.total_calls > 0 else 0
                }
            
            # Get webhook statistics
            webhook_events_query = select(
                WebhookEvent.integration_name,
                func.count(WebhookEvent.id).label('total_events')
            ).group_by(WebhookEvent.integration_name)
            
            webhook_events_result = await self.session.execute(webhook_events_query)
            webhook_stats = {}
            for row in webhook_events_result:
                webhook_stats[row.integration_name] = {
                    "total_events": row.total_events
                }
            
            return {
                "total_integrations": total_integrations,
                "active_integrations": active_integrations,
                "available_integrations": list(self.integrations.keys()),
                "api_calls_stats": api_calls_stats,
                "webhook_stats": webhook_stats
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get integration stats: {str(e)}")
    
    async def get_available_integrations(self) -> Dict[str, Any]:
        """Get list of available integrations."""
        try:
            integrations_info = {}
            for name, config in self.integrations.items():
                integrations_info[name] = {
                    "name": config.name,
                    "service_type": config.service_type,
                    "base_url": config.base_url,
                    "rate_limit": config.rate_limit,
                    "timeout": config.timeout,
                    "configured": bool(config.api_key)
                }
            
            return {
                "integrations": integrations_info,
                "total": len(integrations_info),
                "by_type": self._group_integrations_by_type()
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get available integrations: {str(e)}")
    
    def _group_integrations_by_type(self) -> Dict[str, List[str]]:
        """Group integrations by service type."""
        by_type = {}
        for name, config in self.integrations.items():
            if config.service_type not in by_type:
                by_type[config.service_type] = []
            by_type[config.service_type].append(name)
        return by_type
    
    async def configure_integration(
        self,
        integration_name: str,
        api_key: str,
        additional_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Configure an integration with API key and settings."""
        try:
            if integration_name not in self.integrations:
                raise ValidationError(f"Integration '{integration_name}' not found")
            
            config = self.integrations[integration_name]
            config.api_key = api_key
            
            if additional_config:
                for key, value in additional_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            # Store configuration in database
            integration = Integration(
                name=integration_name,
                api_key=api_key,
                configuration=additional_config or {},
                is_active=True,
                created_at=datetime.utcnow()
            )
            
            self.session.add(integration)
            await self.session.commit()
            
            return {
                "integration_name": integration_name,
                "configured": True,
                "api_key_set": bool(api_key),
                "additional_config": additional_config or {}
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to configure integration: {str(e)}")
    
    async def test_integration(self, integration_name: str) -> Dict[str, Any]:
        """Test an integration by making a simple API call."""
        try:
            if integration_name not in self.integrations:
                raise ValidationError(f"Integration '{integration_name}' not found")
            
            config = self.integrations[integration_name]
            
            if not config.api_key:
                return {
                    "integration_name": integration_name,
                    "test_successful": False,
                    "error": "API key not configured"
                }
            
            # Make a test API call based on integration type
            if config.service_type == "social_media":
                endpoint = "/me" if integration_name == "facebook" else "/users/me"
            elif config.service_type == "email":
                endpoint = "/user/profile"
            elif config.service_type == "analytics":
                endpoint = "/accounts"
            elif config.service_type == "payment":
                endpoint = "/balance"
            elif config.service_type == "cdn":
                endpoint = "/user/tokens/verify"
            else:
                endpoint = "/"
            
            result = await self.make_api_call(
                integration_name=integration_name,
                endpoint=endpoint,
                method="GET"
            )
            
            return {
                "integration_name": integration_name,
                "test_successful": result["success"],
                "status_code": result["status_code"],
                "response": result["data"] if result["success"] else None,
                "error": result["data"] if not result["success"] else None
            }
            
        except Exception as e:
            return {
                "integration_name": integration_name,
                "test_successful": False,
                "error": str(e)
            }


























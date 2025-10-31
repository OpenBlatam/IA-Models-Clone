"""
Workflow Service for automated content workflows and business processes
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from dataclasses import dataclass
import json

from ..models.database import BlogPost, User, Comment, WorkflowExecution, WorkflowStep
from ..models.schemas import PostStatus
from ..core.exceptions import DatabaseError, ValidationError


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Workflow step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Workflow step definition."""
    name: str
    description: str
    function: Callable
    dependencies: List[str] = None
    retry_count: int = 3
    timeout: int = 300  # 5 minutes
    condition: Optional[Callable] = None


class WorkflowService:
    """Service for managing automated workflows."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.workflows = {}
        self._register_default_workflows()
    
    def _register_default_workflows(self):
        """Register default workflows."""
        # Content Publishing Workflow
        self.workflows["content_publishing"] = [
            WorkflowStep(
                name="validate_content",
                description="Validate content quality and completeness",
                function=self._validate_content,
                timeout=60
            ),
            WorkflowStep(
                name="generate_seo_metadata",
                description="Generate SEO metadata and tags",
                function=self._generate_seo_metadata,
                dependencies=["validate_content"],
                timeout=120
            ),
            WorkflowStep(
                name="schedule_publication",
                description="Schedule content for publication",
                function=self._schedule_publication,
                dependencies=["validate_content", "generate_seo_metadata"],
                timeout=30
            ),
            WorkflowStep(
                name="notify_subscribers",
                description="Notify subscribers about new content",
                function=self._notify_subscribers,
                dependencies=["schedule_publication"],
                timeout=180
            )
        ]
        
        # Content Moderation Workflow
        self.workflows["content_moderation"] = [
            WorkflowStep(
                name="spam_detection",
                description="Detect spam content",
                function=self._detect_spam,
                timeout=60
            ),
            WorkflowStep(
                name="toxicity_analysis",
                description="Analyze content for toxicity",
                function=self._analyze_toxicity,
                dependencies=["spam_detection"],
                timeout=90
            ),
            WorkflowStep(
                name="quality_assessment",
                description="Assess content quality",
                function=self._assess_quality,
                dependencies=["spam_detection", "toxicity_analysis"],
                timeout=120
            ),
            WorkflowStep(
                name="moderation_decision",
                description="Make moderation decision",
                function=self._make_moderation_decision,
                dependencies=["spam_detection", "toxicity_analysis", "quality_assessment"],
                timeout=30
            )
        ]
        
        # User Onboarding Workflow
        self.workflows["user_onboarding"] = [
            WorkflowStep(
                name="send_welcome_email",
                description="Send welcome email to new user",
                function=self._send_welcome_email,
                timeout=30
            ),
            WorkflowStep(
                name="create_user_profile",
                description="Create user profile and preferences",
                function=self._create_user_profile,
                timeout=60
            ),
            WorkflowStep(
                name="recommend_content",
                description="Recommend initial content to user",
                function=self._recommend_initial_content,
                dependencies=["create_user_profile"],
                timeout=90
            ),
            WorkflowStep(
                name="setup_notifications",
                description="Setup user notification preferences",
                function=self._setup_notifications,
                dependencies=["create_user_profile"],
                timeout=30
            )
        ]
    
    async def execute_workflow(
        self,
        workflow_name: str,
        context: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a workflow with given context."""
        try:
            if workflow_name not in self.workflows:
                raise ValidationError(f"Workflow '{workflow_name}' not found")
            
            workflow_steps = self.workflows[workflow_name]
            
            # Create workflow execution record
            execution = WorkflowExecution(
                workflow_name=workflow_name,
                status=WorkflowStatus.RUNNING.value,
                context=context,
                user_id=user_id,
                started_at=datetime.utcnow()
            )
            
            self.session.add(execution)
            await self.session.commit()
            
            # Execute workflow steps
            execution_result = await self._execute_workflow_steps(
                execution.id, workflow_steps, context
            )
            
            # Update execution status
            execution.status = WorkflowStatus.COMPLETED.value if execution_result["success"] else WorkflowStatus.FAILED.value
            execution.completed_at = datetime.utcnow()
            execution.result = execution_result
            
            await self.session.commit()
            
            return {
                "execution_id": execution.id,
                "workflow_name": workflow_name,
                "status": execution.status,
                "result": execution_result,
                "started_at": execution.started_at,
                "completed_at": execution.completed_at
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to execute workflow: {str(e)}")
    
    async def _execute_workflow_steps(
        self,
        execution_id: int,
        steps: List[WorkflowStep],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow steps in order."""
        try:
            completed_steps = []
            failed_steps = []
            step_results = {}
            
            for step in steps:
                try:
                    # Check dependencies
                    if step.dependencies:
                        for dep in step.dependencies:
                            if dep not in completed_steps:
                                raise ValidationError(f"Dependency '{dep}' not completed")
                    
                    # Check condition
                    if step.condition and not step.condition(context):
                        # Skip step
                        step_result = {
                            "status": StepStatus.SKIPPED.value,
                            "message": "Step condition not met"
                        }
                    else:
                        # Execute step
                        step_result = await self._execute_step(step, context)
                    
                    # Record step execution
                    step_execution = WorkflowStep(
                        execution_id=execution_id,
                        step_name=step.name,
                        status=step_result["status"],
                        result=step_result,
                        started_at=datetime.utcnow(),
                        completed_at=datetime.utcnow()
                    )
                    
                    self.session.add(step_execution)
                    await self.session.commit()
                    
                    step_results[step.name] = step_result
                    
                    if step_result["status"] == StepStatus.COMPLETED.value:
                        completed_steps.append(step.name)
                    else:
                        failed_steps.append(step.name)
                        break  # Stop on first failure
                        
                except Exception as e:
                    # Record failed step
                    step_result = {
                        "status": StepStatus.FAILED.value,
                        "error": str(e)
                    }
                    
                    step_execution = WorkflowStep(
                        execution_id=execution_id,
                        step_name=step.name,
                        status=StepStatus.FAILED.value,
                        result=step_result,
                        started_at=datetime.utcnow(),
                        completed_at=datetime.utcnow()
                    )
                    
                    self.session.add(step_execution)
                    await self.session.commit()
                    
                    step_results[step.name] = step_result
                    failed_steps.append(step.name)
                    break
            
            return {
                "success": len(failed_steps) == 0,
                "completed_steps": completed_steps,
                "failed_steps": failed_steps,
                "step_results": step_results
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to execute workflow steps: {str(e)}")
    
    async def _execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        try:
            # Execute step function with timeout
            result = await asyncio.wait_for(
                step.function(context),
                timeout=step.timeout
            )
            
            return {
                "status": StepStatus.COMPLETED.value,
                "result": result,
                "execution_time": step.timeout
            }
            
        except asyncio.TimeoutError:
            return {
                "status": StepStatus.FAILED.value,
                "error": f"Step timed out after {step.timeout} seconds"
            }
        except Exception as e:
            return {
                "status": StepStatus.FAILED.value,
                "error": str(e)
            }
    
    # Workflow Step Functions
    
    async def _validate_content(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content quality and completeness."""
        try:
            post_id = context.get("post_id")
            if not post_id:
                raise ValidationError("Post ID required")
            
            # Get post
            post_query = select(BlogPost).where(BlogPost.id == post_id)
            post_result = await self.session.execute(post_query)
            post = post_result.scalar_one_or_none()
            
            if not post:
                raise ValidationError("Post not found")
            
            # Validation checks
            validation_results = {
                "title_length": len(post.title) >= 10,
                "content_length": len(post.content) >= 100,
                "has_category": post.category is not None,
                "has_tags": post.tags is not None and len(post.tags) > 0,
                "has_excerpt": post.excerpt is not None and len(post.excerpt) > 0
            }
            
            is_valid = all(validation_results.values())
            
            return {
                "valid": is_valid,
                "validation_results": validation_results,
                "score": sum(validation_results.values()) / len(validation_results)
            }
            
        except Exception as e:
            raise DatabaseError(f"Content validation failed: {str(e)}")
    
    async def _generate_seo_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SEO metadata and tags."""
        try:
            post_id = context.get("post_id")
            if not post_id:
                raise ValidationError("Post ID required")
            
            # Get post
            post_query = select(BlogPost).where(BlogPost.id == post_id)
            post_result = await self.session.execute(post_query)
            post = post_result.scalar_one_or_none()
            
            if not post:
                raise ValidationError("Post not found")
            
            # Generate SEO metadata (mock implementation)
            seo_metadata = {
                "meta_title": post.title[:60],
                "meta_description": post.excerpt[:160] if post.excerpt else post.content[:160],
                "keywords": post.tags[:5] if post.tags else [],
                "canonical_url": f"/posts/{post.slug}",
                "og_title": post.title,
                "og_description": post.excerpt or post.content[:200],
                "og_image": post.featured_image_url or "/default-og-image.jpg"
            }
            
            # Update post with SEO metadata
            post.seo_metadata = seo_metadata
            await self.session.commit()
            
            return {
                "seo_metadata": seo_metadata,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise DatabaseError(f"SEO metadata generation failed: {str(e)}")
    
    async def _schedule_publication(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule content for publication."""
        try:
            post_id = context.get("post_id")
            publish_at = context.get("publish_at", datetime.utcnow())
            
            if not post_id:
                raise ValidationError("Post ID required")
            
            # Get post
            post_query = select(BlogPost).where(BlogPost.id == post_id)
            post_result = await self.session.execute(post_query)
            post = post_result.scalar_one_or_none()
            
            if not post:
                raise ValidationError("Post not found")
            
            # Update post status and publish date
            post.status = PostStatus.PUBLISHED.value
            post.published_at = publish_at
            await self.session.commit()
            
            return {
                "post_id": post_id,
                "status": "published",
                "published_at": publish_at.isoformat()
            }
            
        except Exception as e:
            raise DatabaseError(f"Publication scheduling failed: {str(e)}")
    
    async def _notify_subscribers(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Notify subscribers about new content."""
        try:
            post_id = context.get("post_id")
            if not post_id:
                raise ValidationError("Post ID required")
            
            # Get post
            post_query = select(BlogPost).where(BlogPost.id == post_id)
            post_result = await self.session.execute(post_query)
            post = post_result.scalar_one_or_none()
            
            if not post:
                raise ValidationError("Post not found")
            
            # Get subscribers (mock implementation)
            # In a real implementation, you would get actual subscribers
            subscribers_count = 100  # Mock count
            
            # Send notifications (mock implementation)
            notifications_sent = subscribers_count
            
            return {
                "post_id": post_id,
                "subscribers_count": subscribers_count,
                "notifications_sent": notifications_sent,
                "sent_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise DatabaseError(f"Subscriber notification failed: {str(e)}")
    
    async def _detect_spam(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect spam content."""
        try:
            post_id = context.get("post_id")
            if not post_id:
                raise ValidationError("Post ID required")
            
            # Get post
            post_query = select(BlogPost).where(BlogPost.id == post_id)
            post_result = await self.session.execute(post_query)
            post = post_result.scalar_one_or_none()
            
            if not post:
                raise ValidationError("Post not found")
            
            # Spam detection logic (mock implementation)
            spam_indicators = {
                "excessive_links": len(post.content.split("http")) > 5,
                "repetitive_text": len(set(post.content.split())) < len(post.content.split()) * 0.3,
                "suspicious_keywords": any(keyword in post.content.lower() for keyword in ["buy now", "click here", "free money"]),
                "short_content": len(post.content) < 50
            }
            
            spam_score = sum(spam_indicators.values()) / len(spam_indicators)
            is_spam = spam_score > 0.5
            
            return {
                "is_spam": is_spam,
                "spam_score": spam_score,
                "indicators": spam_indicators
            }
            
        except Exception as e:
            raise DatabaseError(f"Spam detection failed: {str(e)}")
    
    async def _analyze_toxicity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content for toxicity."""
        try:
            post_id = context.get("post_id")
            if not post_id:
                raise ValidationError("Post ID required")
            
            # Get post
            post_query = select(BlogPost).where(BlogPost.id == post_id)
            post_result = await self.session.execute(post_query)
            post = post_result.scalar_one_or_none()
            
            if not post:
                raise ValidationError("Post not found")
            
            # Toxicity analysis (mock implementation)
            toxic_keywords = ["hate", "stupid", "idiot", "kill", "die"]
            content_lower = post.content.lower()
            
            toxicity_score = sum(1 for keyword in toxic_keywords if keyword in content_lower) / len(toxic_keywords)
            is_toxic = toxicity_score > 0.3
            
            return {
                "is_toxic": is_toxic,
                "toxicity_score": toxicity_score,
                "detected_keywords": [keyword for keyword in toxic_keywords if keyword in content_lower]
            }
            
        except Exception as e:
            raise DatabaseError(f"Toxicity analysis failed: {str(e)}")
    
    async def _assess_quality(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess content quality."""
        try:
            post_id = context.get("post_id")
            if not post_id:
                raise ValidationError("Post ID required")
            
            # Get post
            post_query = select(BlogPost).where(BlogPost.id == post_id)
            post_result = await self.session.execute(post_query)
            post = post_result.scalar_one_or_none()
            
            if not post:
                raise ValidationError("Post not found")
            
            # Quality assessment (mock implementation)
            quality_metrics = {
                "readability": len(post.content.split()) / len(post.content.split('.')) > 10,
                "structure": len(post.content.split('\n\n')) > 2,
                "engagement": len(post.content) > 500,
                "uniqueness": True  # Would check against existing content
            }
            
            quality_score = sum(quality_metrics.values()) / len(quality_metrics)
            
            return {
                "quality_score": quality_score,
                "metrics": quality_metrics,
                "grade": "A" if quality_score > 0.8 else "B" if quality_score > 0.6 else "C"
            }
            
        except Exception as e:
            raise DatabaseError(f"Quality assessment failed: {str(e)}")
    
    async def _make_moderation_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make moderation decision based on analysis."""
        try:
            # Get analysis results from context
            spam_result = context.get("spam_detection", {})
            toxicity_result = context.get("toxicity_analysis", {})
            quality_result = context.get("quality_assessment", {})
            
            # Make decision
            is_spam = spam_result.get("is_spam", False)
            is_toxic = toxicity_result.get("is_toxic", False)
            quality_score = quality_result.get("quality_score", 0)
            
            if is_spam or is_toxic:
                decision = "reject"
                reason = "Content violates community guidelines"
            elif quality_score < 0.3:
                decision = "review"
                reason = "Content quality needs improvement"
            else:
                decision = "approve"
                reason = "Content meets quality standards"
            
            return {
                "decision": decision,
                "reason": reason,
                "spam_detected": is_spam,
                "toxicity_detected": is_toxic,
                "quality_score": quality_score
            }
            
        except Exception as e:
            raise DatabaseError(f"Moderation decision failed: {str(e)}")
    
    async def _send_welcome_email(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Send welcome email to new user."""
        try:
            user_id = context.get("user_id")
            if not user_id:
                raise ValidationError("User ID required")
            
            # Get user
            user_query = select(User).where(User.id == user_id)
            user_result = await self.session.execute(user_query)
            user = user_result.scalar_one_or_none()
            
            if not user:
                raise ValidationError("User not found")
            
            # Send welcome email (mock implementation)
            email_sent = True
            
            return {
                "user_id": user_id,
                "email_sent": email_sent,
                "sent_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise DatabaseError(f"Welcome email failed: {str(e)}")
    
    async def _create_user_profile(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create user profile and preferences."""
        try:
            user_id = context.get("user_id")
            if not user_id:
                raise ValidationError("User ID required")
            
            # Get user
            user_query = select(User).where(User.id == user_id)
            user_result = await self.session.execute(user_query)
            user = user_result.scalar_one_or_none()
            
            if not user:
                raise ValidationError("User not found")
            
            # Create default preferences (mock implementation)
            preferences = {
                "notifications": True,
                "email_digest": True,
                "theme": "light",
                "language": "en"
            }
            
            user.preferences = preferences
            await self.session.commit()
            
            return {
                "user_id": user_id,
                "preferences_created": True,
                "preferences": preferences
            }
            
        except Exception as e:
            raise DatabaseError(f"User profile creation failed: {str(e)}")
    
    async def _recommend_initial_content(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend initial content to user."""
        try:
            user_id = context.get("user_id")
            if not user_id:
                raise ValidationError("User ID required")
            
            # Get popular posts for recommendations (mock implementation)
            popular_posts_query = select(BlogPost).where(
                BlogPost.status == PostStatus.PUBLISHED.value
            ).order_by(desc(BlogPost.view_count)).limit(5)
            
            popular_posts_result = await self.session.execute(popular_posts_query)
            popular_posts = popular_posts_result.scalars().all()
            
            recommendations = [
                {
                    "post_id": post.id,
                    "title": post.title,
                    "excerpt": post.excerpt,
                    "category": post.category
                }
                for post in popular_posts
            ]
            
            return {
                "user_id": user_id,
                "recommendations": recommendations,
                "count": len(recommendations)
            }
            
        except Exception as e:
            raise DatabaseError(f"Content recommendation failed: {str(e)}")
    
    async def _setup_notifications(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Setup user notification preferences."""
        try:
            user_id = context.get("user_id")
            if not user_id:
                raise ValidationError("User ID required")
            
            # Setup default notifications (mock implementation)
            notification_settings = {
                "new_posts": True,
                "comments": True,
                "likes": True,
                "follows": True,
                "weekly_digest": True
            }
            
            return {
                "user_id": user_id,
                "notifications_setup": True,
                "settings": notification_settings
            }
            
        except Exception as e:
            raise DatabaseError(f"Notification setup failed: {str(e)}")
    
    async def get_workflow_executions(
        self,
        workflow_name: Optional[str] = None,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get workflow execution history."""
        try:
            # Build query
            query = select(WorkflowExecution)
            
            if workflow_name:
                query = query.where(WorkflowExecution.workflow_name == workflow_name)
            
            if status:
                query = query.where(WorkflowExecution.status == status)
            
            if user_id:
                query = query.where(WorkflowExecution.user_id == user_id)
            
            # Get total count
            count_query = select(func.count(WorkflowExecution.id))
            if workflow_name:
                count_query = count_query.where(WorkflowExecution.workflow_name == workflow_name)
            if status:
                count_query = count_query.where(WorkflowExecution.status == status)
            if user_id:
                count_query = count_query.where(WorkflowExecution.user_id == user_id)
            
            total_result = await self.session.execute(count_query)
            total = total_result.scalar()
            
            # Get executions
            query = query.order_by(desc(WorkflowExecution.started_at)).offset(offset).limit(limit)
            executions_result = await self.session.execute(query)
            executions = executions_result.scalars().all()
            
            # Format results
            execution_list = []
            for execution in executions:
                execution_list.append({
                    "id": execution.id,
                    "workflow_name": execution.workflow_name,
                    "status": execution.status,
                    "user_id": execution.user_id,
                    "started_at": execution.started_at,
                    "completed_at": execution.completed_at,
                    "context": execution.context,
                    "result": execution.result
                })
            
            return {
                "executions": execution_list,
                "total": total,
                "limit": limit,
                "offset": offset
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get workflow executions: {str(e)}")
    
    async def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics."""
        try:
            # Get total executions
            total_executions_query = select(func.count(WorkflowExecution.id))
            total_executions_result = await self.session.execute(total_executions_query)
            total_executions = total_executions_result.scalar()
            
            # Get executions by status
            status_query = select(
                WorkflowExecution.status,
                func.count(WorkflowExecution.id).label('count')
            ).group_by(WorkflowExecution.status)
            
            status_result = await self.session.execute(status_query)
            status_counts = dict(status_result.all())
            
            # Get executions by workflow
            workflow_query = select(
                WorkflowExecution.workflow_name,
                func.count(WorkflowExecution.id).label('count')
            ).group_by(WorkflowExecution.workflow_name)
            
            workflow_result = await self.session.execute(workflow_query)
            workflow_counts = dict(workflow_result.all())
            
            return {
                "total_executions": total_executions,
                "status_counts": status_counts,
                "workflow_counts": workflow_counts,
                "available_workflows": list(self.workflows.keys())
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get workflow stats: {str(e)}")
    
    def register_workflow(self, name: str, steps: List[WorkflowStep]):
        """Register a new workflow."""
        self.workflows[name] = steps
    
    def get_available_workflows(self) -> Dict[str, Any]:
        """Get available workflows."""
        workflows_info = {}
        for name, steps in self.workflows.items():
            workflows_info[name] = {
                "name": name,
                "steps": [
                    {
                        "name": step.name,
                        "description": step.description,
                        "dependencies": step.dependencies or [],
                        "timeout": step.timeout
                    }
                    for step in steps
                ],
                "step_count": len(steps)
            }
        
        return workflows_info


























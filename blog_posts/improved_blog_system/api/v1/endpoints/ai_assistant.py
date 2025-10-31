"""
Advanced AI Assistant API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime

from ....services.advanced_ai_assistant_service import AdvancedAIAssistantService, AIProvider, AIAssistantType, AIMessageType, AITaskType, AIWorkflowStatus
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class CreateAIAssistantRequest(BaseModel):
    """Request model for creating an AI assistant."""
    name: str = Field(..., description="Assistant name")
    description: str = Field(..., description="Assistant description")
    assistant_type: str = Field(..., description="Assistant type")
    provider: str = Field(default="openai", description="AI provider")
    model: str = Field(default="gpt-4", description="AI model")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    capabilities: Optional[List[str]] = Field(default=None, description="Assistant capabilities")
    configuration: Optional[Dict[str, Any]] = Field(default=None, description="Configuration")


class StartConversationRequest(BaseModel):
    """Request model for starting a conversation."""
    assistant_id: str = Field(..., description="Assistant ID")
    initial_message: Optional[str] = Field(default=None, description="Initial message")


class SendMessageRequest(BaseModel):
    """Request model for sending a message."""
    conversation_id: str = Field(..., description="Conversation ID")
    content: str = Field(..., description="Message content")
    message_type: str = Field(default="user", description="Message type")


class ExecuteTaskRequest(BaseModel):
    """Request model for executing an AI task."""
    task_type: str = Field(..., description="Task type")
    input_data: Dict[str, Any] = Field(..., description="Input data")
    assistant_id: Optional[str] = Field(default=None, description="Assistant ID")
    workflow_id: Optional[str] = Field(default=None, description="Workflow ID")


class RunWorkflowRequest(BaseModel):
    """Request model for running an AI workflow."""
    workflow_id: str = Field(..., description="Workflow ID")
    input_data: Dict[str, Any] = Field(..., description="Input data")
    assistant_id: Optional[str] = Field(default=None, description="Assistant ID")


class AddKnowledgeRequest(BaseModel):
    """Request model for adding knowledge."""
    knowledge_type: str = Field(..., description="Knowledge type")
    content: str = Field(..., description="Knowledge content")
    source: str = Field(..., description="Knowledge source")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata")


class SearchKnowledgeRequest(BaseModel):
    """Request model for searching knowledge."""
    query: str = Field(..., description="Search query")
    knowledge_type: Optional[str] = Field(default=None, description="Knowledge type")
    limit: int = Field(default=10, ge=1, le=100, description="Result limit")


async def get_ai_assistant_service(session: DatabaseSessionDep) -> AdvancedAIAssistantService:
    """Get AI assistant service instance."""
    return AdvancedAIAssistantService(session)


@router.post("/assistants", response_model=Dict[str, Any])
async def create_ai_assistant(
    request: CreateAIAssistantRequest = Depends(),
    ai_service: AdvancedAIAssistantService = Depends(get_ai_assistant_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a new AI assistant."""
    try:
        # Convert enums
        try:
            assistant_type_enum = AIAssistantType(request.assistant_type.lower())
            provider_enum = AIProvider(request.provider.lower())
        except ValueError as e:
            raise ValidationError(f"Invalid enum value: {e}")
        
        result = await ai_service.create_ai_assistant(
            name=request.name,
            description=request.description,
            assistant_type=assistant_type_enum,
            user_id=str(current_user.id),
            provider=provider_enum,
            model=request.model,
            system_prompt=request.system_prompt,
            capabilities=request.capabilities,
            configuration=request.configuration
        )
        
        return {
            "success": True,
            "data": result,
            "message": "AI assistant created successfully"
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
            detail="Failed to create AI assistant"
        )


@router.post("/conversations", response_model=Dict[str, Any])
async def start_conversation(
    request: StartConversationRequest = Depends(),
    ai_service: AdvancedAIAssistantService = Depends(get_ai_assistant_service),
    current_user: CurrentUserDep = Depends()
):
    """Start a new conversation with an AI assistant."""
    try:
        result = await ai_service.start_conversation(
            assistant_id=request.assistant_id,
            user_id=str(current_user.id),
            initial_message=request.initial_message
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Conversation started successfully"
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
            detail="Failed to start conversation"
        )


@router.post("/messages", response_model=Dict[str, Any])
async def send_message(
    request: SendMessageRequest = Depends(),
    ai_service: AdvancedAIAssistantService = Depends(get_ai_assistant_service),
    current_user: CurrentUserDep = Depends()
):
    """Send a message in a conversation."""
    try:
        # Convert message type to enum
        try:
            message_type_enum = AIMessageType(request.message_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid message type: {request.message_type}")
        
        result = await ai_service.send_message(
            conversation_id=request.conversation_id,
            content=request.content,
            message_type=message_type_enum
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Message sent successfully"
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
            detail="Failed to send message"
        )


@router.post("/tasks", response_model=Dict[str, Any])
async def execute_ai_task(
    request: ExecuteTaskRequest = Depends(),
    ai_service: AdvancedAIAssistantService = Depends(get_ai_assistant_service),
    current_user: CurrentUserDep = Depends()
):
    """Execute an AI task."""
    try:
        # Convert task type to enum
        try:
            task_type_enum = AITaskType(request.task_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid task type: {request.task_type}")
        
        result = await ai_service.execute_ai_task(
            task_type=task_type_enum,
            user_id=str(current_user.id),
            input_data=request.input_data,
            assistant_id=request.assistant_id,
            workflow_id=request.workflow_id
        )
        
        return {
            "success": True,
            "data": result,
            "message": "AI task executed successfully"
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
            detail="Failed to execute AI task"
        )


@router.post("/workflows", response_model=Dict[str, Any])
async def run_ai_workflow(
    request: RunWorkflowRequest = Depends(),
    ai_service: AdvancedAIAssistantService = Depends(get_ai_assistant_service),
    current_user: CurrentUserDep = Depends()
):
    """Run an AI workflow."""
    try:
        result = await ai_service.run_ai_workflow(
            workflow_id=request.workflow_id,
            user_id=str(current_user.id),
            input_data=request.input_data,
            assistant_id=request.assistant_id
        )
        
        return {
            "success": True,
            "data": result,
            "message": "AI workflow completed successfully"
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
            detail="Failed to run AI workflow"
        )


@router.post("/knowledge", response_model=Dict[str, Any])
async def add_knowledge(
    request: AddKnowledgeRequest = Depends(),
    ai_service: AdvancedAIAssistantService = Depends(get_ai_assistant_service),
    current_user: CurrentUserDep = Depends()
):
    """Add knowledge to the AI knowledge base."""
    try:
        result = await ai_service.add_knowledge(
            knowledge_type=request.knowledge_type,
            content=request.content,
            source=request.source,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Knowledge added successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add knowledge"
        )


@router.post("/knowledge/search", response_model=Dict[str, Any])
async def search_knowledge(
    request: SearchKnowledgeRequest = Depends(),
    ai_service: AdvancedAIAssistantService = Depends(get_ai_assistant_service),
    current_user: CurrentUserDep = Depends()
):
    """Search the AI knowledge base."""
    try:
        result = await ai_service.search_knowledge(
            query=request.query,
            knowledge_type=request.knowledge_type,
            limit=request.limit
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Knowledge search completed successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search knowledge"
        )


@router.get("/analytics", response_model=Dict[str, Any])
async def get_ai_analytics(
    user_id: Optional[str] = Query(default=None, description="User ID"),
    time_period: str = Query(default="30_days", description="Time period"),
    ai_service: AdvancedAIAssistantService = Depends(get_ai_assistant_service),
    current_user: CurrentUserDep = Depends()
):
    """Get AI usage analytics."""
    try:
        result = await ai_service.get_ai_analytics(
            user_id=user_id,
            time_period=time_period
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "AI analytics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AI analytics"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_ai_stats(
    ai_service: AdvancedAIAssistantService = Depends(get_ai_assistant_service),
    current_user: CurrentUserDep = Depends()
):
    """Get AI system statistics."""
    try:
        result = await ai_service.get_ai_stats()
        
        return {
            "success": True,
            "data": result["data"],
            "message": "AI statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AI statistics"
        )


@router.get("/providers", response_model=Dict[str, Any])
async def get_ai_providers():
    """Get available AI providers."""
    providers = {
        "openai": {
            "name": "OpenAI",
            "description": "Advanced AI models for text generation and analysis",
            "models": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
            "capabilities": ["text_generation", "text_analysis", "code_generation"],
            "pricing": "Pay per token",
            "api_required": True
        },
        "anthropic": {
            "name": "Anthropic",
            "description": "Claude AI models for safe and helpful AI assistance",
            "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "capabilities": ["text_generation", "text_analysis", "research"],
            "pricing": "Pay per token",
            "api_required": True
        },
        "google": {
            "name": "Google",
            "description": "Google's Gemini AI models for multimodal AI",
            "models": ["gemini-pro", "gemini-pro-vision", "palm-2"],
            "capabilities": ["text_generation", "image_analysis", "multimodal"],
            "pricing": "Pay per token",
            "api_required": True
        },
        "huggingface": {
            "name": "Hugging Face",
            "description": "Open-source AI models and transformers",
            "models": ["microsoft/DialoGPT-medium", "facebook/blenderbot-400M-distill"],
            "capabilities": ["text_generation", "text_analysis", "local"],
            "pricing": "Free and paid tiers",
            "api_required": False
        },
        "local": {
            "name": "Local",
            "description": "Run AI models locally on your infrastructure",
            "models": ["Custom models"],
            "capabilities": ["text_generation", "text_analysis", "custom"],
            "pricing": "Infrastructure costs only",
            "api_required": False
        },
        "custom": {
            "name": "Custom",
            "description": "Integrate with custom AI providers",
            "models": ["Custom models"],
            "capabilities": ["Custom capabilities"],
            "pricing": "Custom pricing",
            "api_required": True
        }
    }
    
    return {
        "success": True,
        "data": {
            "providers": providers,
            "total_providers": len(providers)
        },
        "message": "AI providers retrieved successfully"
    }


@router.get("/assistant-types", response_model=Dict[str, Any])
async def get_assistant_types():
    """Get available AI assistant types."""
    assistant_types = {
        "general": {
            "name": "General Assistant",
            "description": "General-purpose AI assistant for various tasks",
            "icon": "🤖",
            "use_cases": ["General questions", "Information lookup", "Casual conversation"]
        },
        "writing": {
            "name": "Writing Assistant",
            "description": "Specialized assistant for writing and content creation",
            "icon": "✍️",
            "use_cases": ["Content writing", "Grammar checking", "Style improvement"]
        },
        "coding": {
            "name": "Coding Assistant",
            "description": "Specialized assistant for programming and development",
            "icon": "💻",
            "use_cases": ["Code generation", "Debugging", "Code review"]
        },
        "research": {
            "name": "Research Assistant",
            "description": "Specialized assistant for research and analysis",
            "icon": "🔍",
            "use_cases": ["Information gathering", "Data analysis", "Report generation"]
        },
        "analysis": {
            "name": "Analysis Assistant",
            "description": "Specialized assistant for data analysis and insights",
            "icon": "📊",
            "use_cases": ["Data analysis", "Statistical analysis", "Trend identification"]
        },
        "creative": {
            "name": "Creative Assistant",
            "description": "Specialized assistant for creative projects",
            "icon": "🎨",
            "use_cases": ["Creative writing", "Brainstorming", "Artistic projects"]
        },
        "technical": {
            "name": "Technical Assistant",
            "description": "Specialized assistant for technical documentation",
            "icon": "⚙️",
            "use_cases": ["Technical writing", "Documentation", "Troubleshooting"]
        },
        "business": {
            "name": "Business Assistant",
            "description": "Specialized assistant for business tasks",
            "icon": "💼",
            "use_cases": ["Business strategy", "Planning", "Decision making"]
        },
        "educational": {
            "name": "Educational Assistant",
            "description": "Specialized assistant for learning and education",
            "icon": "🎓",
            "use_cases": ["Learning support", "Teaching assistance", "Educational content"]
        },
        "personal": {
            "name": "Personal Assistant",
            "description": "Specialized assistant for personal tasks",
            "icon": "👤",
            "use_cases": ["Personal organization", "Task management", "Productivity"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "assistant_types": assistant_types,
            "total_types": len(assistant_types)
        },
        "message": "AI assistant types retrieved successfully"
    }


@router.get("/task-types", response_model=Dict[str, Any])
async def get_task_types():
    """Get available AI task types."""
    task_types = {
        "text_generation": {
            "name": "Text Generation",
            "description": "Generate text content based on prompts",
            "icon": "📝",
            "input_required": ["prompt", "max_length"],
            "output_type": "text"
        },
        "text_summarization": {
            "name": "Text Summarization",
            "description": "Summarize long text content",
            "icon": "📄",
            "input_required": ["text", "summary_length"],
            "output_type": "text"
        },
        "text_translation": {
            "name": "Text Translation",
            "description": "Translate text between languages",
            "icon": "🌐",
            "input_required": ["text", "target_language"],
            "output_type": "text"
        },
        "text_analysis": {
            "name": "Text Analysis",
            "description": "Analyze text for sentiment, topics, etc.",
            "icon": "🔍",
            "input_required": ["text", "analysis_type"],
            "output_type": "analysis"
        },
        "code_generation": {
            "name": "Code Generation",
            "description": "Generate code based on specifications",
            "icon": "💻",
            "input_required": ["specification", "language"],
            "output_type": "code"
        },
        "code_review": {
            "name": "Code Review",
            "description": "Review code for issues and improvements",
            "icon": "🔍",
            "input_required": ["code", "language"],
            "output_type": "review"
        },
        "image_generation": {
            "name": "Image Generation",
            "description": "Generate images based on descriptions",
            "icon": "🎨",
            "input_required": ["prompt", "style"],
            "output_type": "image"
        },
        "image_analysis": {
            "name": "Image Analysis",
            "description": "Analyze images for content and features",
            "icon": "🖼️",
            "input_required": ["image", "analysis_type"],
            "output_type": "analysis"
        },
        "data_analysis": {
            "name": "Data Analysis",
            "description": "Analyze data and generate insights",
            "icon": "📊",
            "input_required": ["data", "analysis_type"],
            "output_type": "analysis"
        },
        "research": {
            "name": "Research",
            "description": "Conduct research on specific topics",
            "icon": "🔬",
            "input_required": ["topic", "scope"],
            "output_type": "research"
        },
        "writing_assistance": {
            "name": "Writing Assistance",
            "description": "Assist with writing tasks",
            "icon": "✍️",
            "input_required": ["content", "assistance_type"],
            "output_type": "text"
        },
        "content_optimization": {
            "name": "Content Optimization",
            "description": "Optimize content for SEO and engagement",
            "icon": "⚡",
            "input_required": ["content", "optimization_goals"],
            "output_type": "optimized_content"
        }
    }
    
    return {
        "success": True,
        "data": {
            "task_types": task_types,
            "total_types": len(task_types)
        },
        "message": "AI task types retrieved successfully"
    }


@router.get("/workflow-types", response_model=Dict[str, Any])
async def get_workflow_types():
    """Get available AI workflow types."""
    workflow_types = {
        "content_creation": {
            "name": "Content Creation Workflow",
            "description": "Complete workflow for creating high-quality content",
            "icon": "📝",
            "steps": ["research_topic", "generate_outline", "write_content", "review_and_edit", "optimize_seo"],
            "estimated_time": "30 minutes",
            "complexity": "Medium"
        },
        "code_review": {
            "name": "Code Review Workflow",
            "description": "Comprehensive code review and improvement workflow",
            "icon": "💻",
            "steps": ["analyze_code", "check_best_practices", "identify_issues", "suggest_improvements", "generate_report"],
            "estimated_time": "15 minutes",
            "complexity": "Medium"
        },
        "research_assistance": {
            "name": "Research Assistance Workflow",
            "description": "Complete research workflow from question to report",
            "icon": "🔍",
            "steps": ["define_research_question", "search_sources", "analyze_information", "synthesize_findings", "generate_report"],
            "estimated_time": "45 minutes",
            "complexity": "High"
        },
        "data_analysis": {
            "name": "Data Analysis Workflow",
            "description": "Complete data analysis workflow",
            "icon": "📊",
            "steps": ["data_cleaning", "exploratory_analysis", "statistical_analysis", "visualization", "insights_generation"],
            "estimated_time": "60 minutes",
            "complexity": "High"
        },
        "content_optimization": {
            "name": "Content Optimization Workflow",
            "description": "Optimize content for better performance",
            "icon": "⚡",
            "steps": ["content_analysis", "seo_optimization", "readability_improvement", "engagement_enhancement", "performance_prediction"],
            "estimated_time": "20 minutes",
            "complexity": "Medium"
        }
    }
    
    return {
        "success": True,
        "data": {
            "workflow_types": workflow_types,
            "total_types": len(workflow_types)
        },
        "message": "AI workflow types retrieved successfully"
    }


@router.get("/message-types", response_model=Dict[str, Any])
async def get_message_types():
    """Get available AI message types."""
    message_types = {
        "user": {
            "name": "User Message",
            "description": "Message sent by the user",
            "icon": "👤",
            "direction": "incoming"
        },
        "assistant": {
            "name": "Assistant Message",
            "description": "Response from the AI assistant",
            "icon": "🤖",
            "direction": "outgoing"
        },
        "system": {
            "name": "System Message",
            "description": "System-generated message",
            "icon": "⚙️",
            "direction": "system"
        },
        "function": {
            "name": "Function Message",
            "description": "Function call result",
            "icon": "🔧",
            "direction": "system"
        },
        "error": {
            "name": "Error Message",
            "description": "Error notification",
            "icon": "❌",
            "direction": "system"
        },
        "warning": {
            "name": "Warning Message",
            "description": "Warning notification",
            "icon": "⚠️",
            "direction": "system"
        },
        "info": {
            "name": "Info Message",
            "description": "Informational message",
            "icon": "ℹ️",
            "direction": "system"
        }
    }
    
    return {
        "success": True,
        "data": {
            "message_types": message_types,
            "total_types": len(message_types)
        },
        "message": "AI message types retrieved successfully"
    }


@router.get("/health", response_model=Dict[str, Any])
async def get_ai_assistant_health(
    ai_service: AdvancedAIAssistantService = Depends(get_ai_assistant_service),
    current_user: CurrentUserDep = Depends()
):
    """Get AI assistant system health status."""
    try:
        # Get AI stats
        stats = await ai_service.get_ai_stats()
        
        # Calculate health metrics
        total_assistants = stats["data"].get("total_assistants", 0)
        total_conversations = stats["data"].get("total_conversations", 0)
        total_messages = stats["data"].get("total_messages", 0)
        total_knowledge = stats["data"].get("total_knowledge", 0)
        assistants_by_type = stats["data"].get("assistants_by_type", {})
        providers_usage = stats["data"].get("providers_usage", {})
        available_providers = stats["data"].get("available_providers", 0)
        available_models = stats["data"].get("available_models", 0)
        available_workflows = stats["data"].get("available_workflows", 0)
        
        # Calculate health score
        health_score = 100
        
        # Check assistant distribution
        if total_assistants > 0:
            general_assistants = assistants_by_type.get("general", 0)
            general_ratio = general_assistants / total_assistants
            if general_ratio < 0.2:
                health_score -= 15
            elif general_ratio > 0.8:
                health_score -= 10
        
        # Check provider diversity
        if len(providers_usage) < 2:
            health_score -= 20
        elif len(providers_usage) > 5:
            health_score -= 5
        
        # Check knowledge base
        if total_knowledge < 100:
            health_score -= 15
        elif total_knowledge > 10000:
            health_score -= 5
        
        # Check conversation activity
        if total_conversations > 0:
            messages_per_conversation = total_messages / total_conversations
            if messages_per_conversation < 2:
                health_score -= 20
            elif messages_per_conversation > 50:
                health_score -= 10
        
        # Check system resources
        if available_providers < 3:
            health_score -= 15
        if available_models < 5:
            health_score -= 10
        if available_workflows < 3:
            health_score -= 10
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "total_assistants": total_assistants,
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "total_knowledge": total_knowledge,
                "assistants_by_type": assistants_by_type,
                "providers_usage": providers_usage,
                "available_providers": available_providers,
                "available_models": available_models,
                "available_workflows": available_workflows,
                "general_assistant_ratio": general_ratio if total_assistants > 0 else 0,
                "provider_diversity": len(providers_usage),
                "messages_per_conversation": messages_per_conversation if total_conversations > 0 else 0,
                "timestamp": datetime.utcnow().isoformat()
            },
            "message": "AI assistant health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AI assistant health status"
        )

























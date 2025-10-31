"""
Gamma App - Real Improvement AI Assistant
AI-powered assistant for real improvements that actually work
"""

import asyncio
import logging
import time
import json
import openai
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
import aiohttp

logger = logging.getLogger(__name__)

class AIAssistantType(Enum):
    """AI Assistant types"""
    CODE_ANALYZER = "code_analyzer"
    IMPROVEMENT_SUGGESTER = "improvement_suggester"
    IMPLEMENTATION_HELPER = "implementation_helper"
    TEST_GENERATOR = "test_generator"
    DOCUMENTATION_HELPER = "documentation_helper"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    SECURITY_ADVISOR = "security_advisor"
    CODE_REVIEWER = "code_reviewer"

class AIAssistantStatus(Enum):
    """AI Assistant status"""
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

@dataclass
class AIAssistantConfig:
    """AI Assistant configuration"""
    assistant_id: str
    name: str
    type: AIAssistantType
    description: str
    config: Dict[str, Any]
    status: AIAssistantStatus = AIAssistantStatus.ACTIVE
    enabled: bool = True
    created_at: datetime = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    success_rate: float = 0.0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class AIAssistantRequest:
    """AI Assistant request"""
    request_id: str
    assistant_id: str
    request_type: str
    input_data: Dict[str, Any]
    context: Dict[str, Any] = None
    created_at: datetime = None
    processed_at: Optional[datetime] = None
    duration: float = 0.0
    success: bool = False
    response_data: Dict[str, Any] = None
    error_message: str = ""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.context is None:
            self.context = {}
        if self.response_data is None:
            self.response_data = {}

class RealImprovementAIAssistant:
    """
    AI-powered assistant for real improvements
    """
    
    def __init__(self, project_root: str = ".", openai_api_key: str = None):
        """Initialize AI assistant"""
        self.project_root = Path(project_root)
        self.assistants: Dict[str, AIAssistantConfig] = {}
        self.requests: Dict[str, AIAssistantRequest] = {}
        self.request_logs: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize OpenAI
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            # Try to get from environment
            import os
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize with default assistants
        self._initialize_default_assistants()
        
        logger.info(f"Real Improvement AI Assistant initialized for {self.project_root}")
    
    def _initialize_default_assistants(self):
        """Initialize default AI assistants"""
        # Code Analyzer Assistant
        code_analyzer = AIAssistantConfig(
            assistant_id="code_analyzer",
            name="Code Analyzer",
            type=AIAssistantType.CODE_ANALYZER,
            description="Analyzes code for quality, performance, and security issues",
            config={
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 2000,
                "system_prompt": "You are an expert code analyzer. Analyze the provided code for quality issues, performance problems, security vulnerabilities, and suggest improvements."
            }
        )
        self.assistants[code_analyzer.assistant_id] = code_analyzer
        
        # Improvement Suggester Assistant
        improvement_suggester = AIAssistantConfig(
            assistant_id="improvement_suggester",
            name="Improvement Suggester",
            type=AIAssistantType.IMPROVEMENT_SUGGESTER,
            description="Suggests specific improvements based on code analysis",
            config={
                "model": "gpt-4",
                "temperature": 0.2,
                "max_tokens": 1500,
                "system_prompt": "You are an expert software engineer. Based on the code analysis, suggest specific, actionable improvements with implementation details."
            }
        )
        self.assistants[improvement_suggester.assistant_id] = improvement_suggester
        
        # Implementation Helper Assistant
        implementation_helper = AIAssistantConfig(
            assistant_id="implementation_helper",
            name="Implementation Helper",
            type=AIAssistantType.IMPLEMENTATION_HELPER,
            description="Helps implement specific improvements with code examples",
            config={
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 3000,
                "system_prompt": "You are an expert software engineer. Provide detailed implementation code for the requested improvement, including error handling, testing, and best practices."
            }
        )
        self.assistants[implementation_helper.assistant_id] = implementation_helper
        
        # Test Generator Assistant
        test_generator = AIAssistantConfig(
            assistant_id="test_generator",
            name="Test Generator",
            type=AIAssistantType.TEST_GENERATOR,
            description="Generates comprehensive tests for code improvements",
            config={
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 2500,
                "system_prompt": "You are an expert in software testing. Generate comprehensive unit tests, integration tests, and edge cases for the provided code."
            }
        )
        self.assistants[test_generator.assistant_id] = test_generator
        
        # Documentation Helper Assistant
        documentation_helper = AIAssistantConfig(
            assistant_id="documentation_helper",
            name="Documentation Helper",
            type=AIAssistantType.DOCUMENTATION_HELPER,
            description="Generates documentation for code improvements",
            config={
                "model": "gpt-4",
                "temperature": 0.2,
                "max_tokens": 2000,
                "system_prompt": "You are an expert technical writer. Generate clear, comprehensive documentation for the provided code, including API documentation, usage examples, and best practices."
            }
        )
        self.assistants[documentation_helper.assistant_id] = documentation_helper
        
        # Performance Optimizer Assistant
        performance_optimizer = AIAssistantConfig(
            assistant_id="performance_optimizer",
            name="Performance Optimizer",
            type=AIAssistantType.PERFORMANCE_OPTIMIZER,
            description="Optimizes code for better performance",
            config={
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 2000,
                "system_prompt": "You are an expert in performance optimization. Analyze the provided code and suggest specific optimizations for better performance, including algorithmic improvements, caching strategies, and resource optimization."
            }
        )
        self.assistants[performance_optimizer.assistant_id] = performance_optimizer
        
        # Security Advisor Assistant
        security_advisor = AIAssistantConfig(
            assistant_id="security_advisor",
            name="Security Advisor",
            type=AIAssistantType.SECURITY_ADVISOR,
            description="Identifies and fixes security vulnerabilities",
            config={
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 2000,
                "system_prompt": "You are an expert in cybersecurity. Analyze the provided code for security vulnerabilities, including injection attacks, authentication issues, authorization problems, and data protection concerns. Provide specific fixes and security best practices."
            }
        )
        self.assistants[security_advisor.assistant_id] = security_advisor
        
        # Code Reviewer Assistant
        code_reviewer = AIAssistantConfig(
            assistant_id="code_reviewer",
            name="Code Reviewer",
            type=AIAssistantType.CODE_REVIEWER,
            description="Reviews code for quality and best practices",
            config={
                "model": "gpt-4",
                "temperature": 0.2,
                "max_tokens": 2500,
                "system_prompt": "You are an expert code reviewer. Review the provided code for quality, maintainability, readability, and adherence to best practices. Provide specific feedback and suggestions for improvement."
            }
        )
        self.assistants[code_reviewer.assistant_id] = code_reviewer
    
    def create_ai_assistant(self, name: str, type: AIAssistantType, 
                          description: str, config: Dict[str, Any]) -> str:
        """Create AI assistant"""
        try:
            assistant_id = f"assistant_{int(time.time() * 1000)}"
            
            assistant = AIAssistantConfig(
                assistant_id=assistant_id,
                name=name,
                type=type,
                description=description,
                config=config
            )
            
            self.assistants[assistant_id] = assistant
            self.request_logs[assistant_id] = []
            
            logger.info(f"AI Assistant created: {name}")
            return assistant_id
            
        except Exception as e:
            logger.error(f"Failed to create AI assistant: {e}")
            raise
    
    async def process_request(self, assistant_id: str, request_type: str, 
                            input_data: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Process AI assistant request"""
        try:
            if assistant_id not in self.assistants:
                raise ValueError(f"Assistant {assistant_id} not found")
            
            assistant = self.assistants[assistant_id]
            
            if not assistant.enabled:
                raise ValueError(f"Assistant {assistant_id} is disabled")
            
            request_id = f"req_{int(time.time() * 1000)}"
            
            request = AIAssistantRequest(
                request_id=request_id,
                assistant_id=assistant_id,
                request_type=request_type,
                input_data=input_data,
                context=context or {}
            )
            
            self.requests[request_id] = request
            self.request_logs[assistant_id] = self.request_logs.get(assistant_id, [])
            
            self._log_request(assistant_id, "request_started", f"Request {request_id} started")
            
            # Update assistant status
            assistant.status = AIAssistantStatus.BUSY
            
            try:
                # Process request based on assistant type
                if assistant.type == AIAssistantType.CODE_ANALYZER:
                    result = await self._process_code_analysis(assistant, request)
                elif assistant.type == AIAssistantType.IMPROVEMENT_SUGGESTER:
                    result = await self._process_improvement_suggestion(assistant, request)
                elif assistant.type == AIAssistantType.IMPLEMENTATION_HELPER:
                    result = await self._process_implementation_help(assistant, request)
                elif assistant.type == AIAssistantType.TEST_GENERATOR:
                    result = await self._process_test_generation(assistant, request)
                elif assistant.type == AIAssistantType.DOCUMENTATION_HELPER:
                    result = await self._process_documentation_help(assistant, request)
                elif assistant.type == AIAssistantType.PERFORMANCE_OPTIMIZER:
                    result = await self._process_performance_optimization(assistant, request)
                elif assistant.type == AIAssistantType.SECURITY_ADVISOR:
                    result = await self._process_security_analysis(assistant, request)
                elif assistant.type == AIAssistantType.CODE_REVIEWER:
                    result = await self._process_code_review(assistant, request)
                else:
                    result = {"success": False, "error": f"Unknown assistant type: {assistant.type}"}
                
                # Update request
                request.processed_at = datetime.utcnow()
                request.duration = (request.processed_at - request.created_at).total_seconds()
                request.success = result["success"]
                request.response_data = result.get("data", {})
                request.error_message = result.get("error", "")
                
                # Update assistant stats
                assistant.usage_count += 1
                assistant.last_used = request.processed_at
                
                if request.success:
                    # Update success rate
                    successful_requests = len([r for r in self.requests.values() 
                                            if r.assistant_id == assistant_id and r.success])
                    assistant.success_rate = (successful_requests / assistant.usage_count) * 100
                    
                    self._log_request(assistant_id, "request_completed", f"Request {request_id} completed successfully")
                else:
                    self._log_request(assistant_id, "request_failed", f"Request {request_id} failed: {request.error_message}")
                
                # Update assistant status
                assistant.status = AIAssistantStatus.ACTIVE
                
                return request_id
                
            except Exception as e:
                # Update request with error
                request.processed_at = datetime.utcnow()
                request.duration = (request.processed_at - request.created_at).total_seconds()
                request.success = False
                request.error_message = str(e)
                
                # Update assistant status
                assistant.status = AIAssistantStatus.ERROR
                
                self._log_request(assistant_id, "request_error", f"Request {request_id} error: {str(e)}")
                
                return request_id
            
        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            raise
    
    async def _process_code_analysis(self, assistant: AIAssistantConfig, request: AIAssistantRequest) -> Dict[str, Any]:
        """Process code analysis request"""
        try:
            code = request.input_data.get("code", "")
            language = request.input_data.get("language", "python")
            
            prompt = f"""
            Analyze the following {language} code for quality issues, performance problems, and security vulnerabilities:
            
            ```{language}
            {code}
            ```
            
            Provide a detailed analysis including:
            1. Code quality issues
            2. Performance problems
            3. Security vulnerabilities
            4. Best practices violations
            5. Specific improvement suggestions
            """
            
            response = await self._call_openai(assistant, prompt)
            
            return {
                "success": True,
                "data": {
                    "analysis": response,
                    "language": language,
                    "code_length": len(code)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _process_improvement_suggestion(self, assistant: AIAssistantConfig, request: AIAssistantRequest) -> Dict[str, Any]:
        """Process improvement suggestion request"""
        try:
            analysis = request.input_data.get("analysis", "")
            code = request.input_data.get("code", "")
            language = request.input_data.get("language", "python")
            
            prompt = f"""
            Based on the following code analysis, suggest specific, actionable improvements:
            
            Analysis: {analysis}
            
            Code:
            ```{language}
            {code}
            ```
            
            Provide:
            1. Specific improvement suggestions
            2. Implementation priority
            3. Expected impact
            4. Implementation effort
            5. Code examples for each improvement
            """
            
            response = await self._call_openai(assistant, prompt)
            
            return {
                "success": True,
                "data": {
                    "suggestions": response,
                    "language": language,
                    "analysis": analysis
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _process_implementation_help(self, assistant: AIAssistantConfig, request: AIAssistantRequest) -> Dict[str, Any]:
        """Process implementation help request"""
        try:
            improvement = request.input_data.get("improvement", "")
            code = request.input_data.get("code", "")
            language = request.input_data.get("language", "python")
            
            prompt = f"""
            Help implement the following improvement:
            
            Improvement: {improvement}
            
            Current code:
            ```{language}
            {code}
            ```
            
            Provide:
            1. Complete implementation code
            2. Error handling
            3. Input validation
            4. Testing considerations
            5. Best practices
            """
            
            response = await self._call_openai(assistant, prompt)
            
            return {
                "success": True,
                "data": {
                    "implementation": response,
                    "language": language,
                    "improvement": improvement
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _process_test_generation(self, assistant: AIAssistantConfig, request: AIAssistantRequest) -> Dict[str, Any]:
        """Process test generation request"""
        try:
            code = request.input_data.get("code", "")
            language = request.input_data.get("language", "python")
            test_framework = request.input_data.get("test_framework", "pytest")
            
            prompt = f"""
            Generate comprehensive tests for the following {language} code using {test_framework}:
            
            ```{language}
            {code}
            ```
            
            Provide:
            1. Unit tests
            2. Integration tests
            3. Edge cases
            4. Mock objects
            5. Test data
            6. Test documentation
            """
            
            response = await self._call_openai(assistant, prompt)
            
            return {
                "success": True,
                "data": {
                    "tests": response,
                    "language": language,
                    "framework": test_framework
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _process_documentation_help(self, assistant: AIAssistantConfig, request: AIAssistantRequest) -> Dict[str, Any]:
        """Process documentation help request"""
        try:
            code = request.input_data.get("code", "")
            language = request.input_data.get("language", "python")
            doc_type = request.input_data.get("doc_type", "api")
            
            prompt = f"""
            Generate {doc_type} documentation for the following {language} code:
            
            ```{language}
            {code}
            ```
            
            Provide:
            1. Function/class documentation
            2. Parameter descriptions
            3. Return value descriptions
            4. Usage examples
            5. Best practices
            6. Common pitfalls
            """
            
            response = await self._call_openai(assistant, prompt)
            
            return {
                "success": True,
                "data": {
                    "documentation": response,
                    "language": language,
                    "doc_type": doc_type
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _process_performance_optimization(self, assistant: AIAssistantConfig, request: AIAssistantRequest) -> Dict[str, Any]:
        """Process performance optimization request"""
        try:
            code = request.input_data.get("code", "")
            language = request.input_data.get("language", "python")
            performance_issues = request.input_data.get("performance_issues", [])
            
            prompt = f"""
            Optimize the following {language} code for better performance:
            
            ```{language}
            {code}
            ```
            
            Performance issues identified: {performance_issues}
            
            Provide:
            1. Optimized code
            2. Performance improvements explanation
            3. Benchmarking suggestions
            4. Memory optimization
            5. Algorithm improvements
            6. Caching strategies
            """
            
            response = await self._call_openai(assistant, prompt)
            
            return {
                "success": True,
                "data": {
                    "optimization": response,
                    "language": language,
                    "issues": performance_issues
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _process_security_analysis(self, assistant: AIAssistantConfig, request: AIAssistantRequest) -> Dict[str, Any]:
        """Process security analysis request"""
        try:
            code = request.input_data.get("code", "")
            language = request.input_data.get("language", "python")
            security_context = request.input_data.get("security_context", {})
            
            prompt = f"""
            Analyze the following {language} code for security vulnerabilities:
            
            ```{language}
            {code}
            ```
            
            Security context: {security_context}
            
            Provide:
            1. Security vulnerabilities found
            2. Risk assessment
            3. Specific fixes
            4. Security best practices
            5. Input validation
            6. Authentication/authorization issues
            """
            
            response = await self._call_openai(assistant, prompt)
            
            return {
                "success": True,
                "data": {
                    "security_analysis": response,
                    "language": language,
                    "context": security_context
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _process_code_review(self, assistant: AIAssistantConfig, request: AIAssistantRequest) -> Dict[str, Any]:
        """Process code review request"""
        try:
            code = request.input_data.get("code", "")
            language = request.input_data.get("language", "python")
            review_focus = request.input_data.get("review_focus", "general")
            
            prompt = f"""
            Review the following {language} code with focus on {review_focus}:
            
            ```{language}
            {code}
            ```
            
            Provide:
            1. Code quality assessment
            2. Maintainability issues
            3. Readability improvements
            4. Best practices violations
            5. Specific recommendations
            6. Code style suggestions
            """
            
            response = await self._call_openai(assistant, prompt)
            
            return {
                "success": True,
                "data": {
                    "review": response,
                    "language": language,
                    "focus": review_focus
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _call_openai(self, assistant: AIAssistantConfig, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            config = assistant.config
            
            response = await openai.ChatCompletion.acreate(
                model=config["model"],
                messages=[
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _log_request(self, assistant_id: str, event: str, message: str):
        """Log request event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "message": message
        }
        
        if assistant_id not in self.request_logs:
            self.request_logs[assistant_id] = []
        
        self.request_logs[assistant_id].append(log_entry)
        
        logger.info(f"AI Assistant {assistant_id}: {event} - {message}")
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get request status"""
        if request_id not in self.requests:
            return None
        
        request = self.requests[request_id]
        
        return {
            "request_id": request_id,
            "assistant_id": request.assistant_id,
            "request_type": request.request_type,
            "success": request.success,
            "created_at": request.created_at.isoformat(),
            "processed_at": request.processed_at.isoformat() if request.processed_at else None,
            "duration": request.duration,
            "error_message": request.error_message,
            "response_data": request.response_data
        }
    
    def get_assistant_summary(self) -> Dict[str, Any]:
        """Get assistant summary"""
        total_assistants = len(self.assistants)
        active_assistants = len([a for a in self.assistants.values() if a.status == AIAssistantStatus.ACTIVE])
        busy_assistants = len([a for a in self.assistants.values() if a.status == AIAssistantStatus.BUSY])
        error_assistants = len([a for a in self.assistants.values() if a.status == AIAssistantStatus.ERROR])
        
        total_requests = len(self.requests)
        successful_requests = len([r for r in self.requests.values() if r.success])
        failed_requests = len([r for r in self.requests.values() if not r.success])
        
        return {
            "total_assistants": total_assistants,
            "active_assistants": active_assistants,
            "busy_assistants": busy_assistants,
            "error_assistants": error_assistants,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            "assistant_types": list(set(a.type.value for a in self.assistants.values()))
        }
    
    def get_assistant_logs(self, assistant_id: str) -> List[Dict[str, Any]]:
        """Get assistant logs"""
        return self.request_logs.get(assistant_id, [])
    
    def enable_assistant(self, assistant_id: str) -> bool:
        """Enable AI assistant"""
        try:
            if assistant_id in self.assistants:
                self.assistants[assistant_id].enabled = True
                self.assistants[assistant_id].status = AIAssistantStatus.ACTIVE
                self._log_request(assistant_id, "enabled", "AI Assistant enabled")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to enable assistant: {e}")
            return False
    
    def disable_assistant(self, assistant_id: str) -> bool:
        """Disable AI assistant"""
        try:
            if assistant_id in self.assistants:
                self.assistants[assistant_id].enabled = False
                self.assistants[assistant_id].status = AIAssistantStatus.OFFLINE
                self._log_request(assistant_id, "disabled", "AI Assistant disabled")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to disable assistant: {e}")
            return False

# Global AI assistant instance
improvement_ai_assistant = None

def get_improvement_ai_assistant() -> RealImprovementAIAssistant:
    """Get improvement AI assistant instance"""
    global improvement_ai_assistant
    if not improvement_ai_assistant:
        improvement_ai_assistant = RealImprovementAIAssistant()
    return improvement_ai_assistant














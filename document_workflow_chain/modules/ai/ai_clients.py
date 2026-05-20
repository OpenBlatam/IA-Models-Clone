"""
AI Clients Integration for Document Workflow Chain
=================================================

This module provides integration with various AI providers including OpenAI,
Anthropic, Cohere, and other AI services for document generation.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json
import time

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AIResponse:
    """Standardized AI response format"""
    content: str
    model: str
    tokens_used: int
    response_time: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class AIClient(ABC):
    """Abstract base class for AI clients"""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.config = kwargs
        self.total_tokens_used = 0
        self.total_requests = 0
    
    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> AIResponse:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    async def generate_title(self, content: str, **kwargs) -> AIResponse:
        """Generate title from content"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "model": self.model,
            "client_type": self.__class__.__name__
        }

class OpenAIClient(AIClient):
    """OpenAI GPT client implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", **kwargs):
        super().__init__(api_key, model, **kwargs)
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
    
    async def generate_text(self, prompt: str, **kwargs) -> AIResponse:
        """Generate text using OpenAI GPT"""
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional content writer who creates engaging, informative, and well-structured documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 1.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                presence_penalty=kwargs.get('presence_penalty', 0.0)
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            response_time = time.time() - start_time
            
            self.total_requests += 1
            self.total_tokens_used += tokens_used
            
            return AIResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                response_time=response_time,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def generate_title(self, content: str, **kwargs) -> AIResponse:
        """Generate title using OpenAI"""
        title_prompt = f"""
        Based on the following content, generate a compelling, SEO-friendly blog title that:
        1. Captures the main topic and value proposition
        2. Is between 50-60 characters
        3. Uses engaging language
        4. Includes relevant keywords
        
        Content:
        {content[:1000]}...
        
        Generate only the title, no additional text.
        """
        
        return await self.generate_text(title_prompt, max_tokens=100, temperature=0.8)

class AnthropicClient(AIClient):
    """Anthropic Claude client implementation"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", **kwargs):
        super().__init__(api_key, model, **kwargs)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
    
    async def generate_text(self, prompt: str, **kwargs) -> AIResponse:
        """Generate text using Anthropic Claude"""
        start_time = time.time()
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7),
                system="You are a professional content writer who creates engaging, informative, and well-structured documents.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            response_time = time.time() - start_time
            
            self.total_requests += 1
            self.total_tokens_used += tokens_used
            
            return AIResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                response_time=response_time,
                metadata={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "stop_reason": response.stop_reason
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    async def generate_title(self, content: str, **kwargs) -> AIResponse:
        """Generate title using Anthropic Claude"""
        title_prompt = f"""
        Based on the following content, generate a compelling, SEO-friendly blog title that:
        1. Captures the main topic and value proposition
        2. Is between 50-60 characters
        3. Uses engaging language
        4. Includes relevant keywords
        
        Content:
        {content[:1000]}...
        
        Generate only the title, no additional text.
        """
        
        return await self.generate_text(title_prompt, max_tokens=100, temperature=0.8)

class CohereClient(AIClient):
    """Cohere client implementation"""
    
    def __init__(self, api_key: str, model: str = "command", **kwargs):
        super().__init__(api_key, model, **kwargs)
        try:
            import cohere
            self.client = cohere.AsyncClient(api_key=api_key)
        except ImportError:
            raise ImportError("Cohere library not installed. Run: pip install cohere")
    
    async def generate_text(self, prompt: str, **kwargs) -> AIResponse:
        """Generate text using Cohere"""
        start_time = time.time()
        
        try:
            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7),
                p=kwargs.get('top_p', 1.0),
                k=kwargs.get('top_k', 0),
                stop_sequences=kwargs.get('stop_sequences', None)
            )
            
            content = response.generations[0].text
            tokens_used = response.meta.billed_units.input_tokens + response.meta.billed_units.output_tokens
            response_time = time.time() - start_time
            
            self.total_requests += 1
            self.total_tokens_used += tokens_used
            
            return AIResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                response_time=response_time,
                metadata={
                    "finish_reason": response.generations[0].finish_reason,
                    "input_tokens": response.meta.billed_units.input_tokens,
                    "output_tokens": response.meta.billed_units.output_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Cohere API error: {str(e)}")
            raise
    
    async def generate_title(self, content: str, **kwargs) -> AIResponse:
        """Generate title using Cohere"""
        title_prompt = f"""
        Based on the following content, generate a compelling, SEO-friendly blog title that:
        1. Captures the main topic and value proposition
        2. Is between 50-60 characters
        3. Uses engaging language
        4. Includes relevant keywords
        
        Content:
        {content[:1000]}...
        
        Generate only the title, no additional text.
        """
        
        return await self.generate_text(title_prompt, max_tokens=100, temperature=0.8)

class MockAIClient(AIClient):
    """Mock AI client for testing and development"""
    
    def __init__(self, api_key: str = "mock", model: str = "mock-gpt", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.call_count = 0
    
    async def generate_text(self, prompt: str, **kwargs) -> AIResponse:
        """Generate mock text based on prompt"""
        start_time = time.time()
        self.call_count += 1
        
        # Simulate API delay
        await asyncio.sleep(0.1)
        
        # Generate mock content based on prompt
        if "introducción" in prompt.lower() or "introduction" in prompt.lower():
            content = """
            La Inteligencia Artificial está revolucionando la forma en que creamos contenido. 
            Desde asistentes de escritura automatizados hasta optimización inteligente de contenido, 
            las herramientas de IA están transformando el panorama creativo. Esta guía completa 
            explora las diversas formas en que la IA puede mejorar tu proceso de creación de contenido, 
            aumentar la eficiencia y entregar resultados de mayor calidad.
            
            Los beneficios de la creación de contenido impulsada por IA son numerosos. No solo 
            acelera el proceso de escritura, sino que también ayuda a mantener la consistencia 
            en diferentes piezas de contenido. La IA puede analizar las preferencias de la audiencia, 
            optimizar para motores de búsqueda e incluso sugerir mejoras para mejorar la legibilidad y el engagement.
            """
        elif "beneficios" in prompt.lower() or "benefits" in prompt.lower():
            content = """
            Las ventajas de implementar IA en los flujos de trabajo de creación de contenido 
            van mucho más allá de la simple automatización. Uno de los beneficios más significativos 
            es la capacidad de escalar la producción de contenido sin aumentar proporcionalmente 
            los recursos humanos. La IA puede generar múltiples variaciones de contenido, probar 
            diferentes enfoques e identificar qué resuena más con tu audiencia objetivo.
            
            Además, la creación de contenido impulsada por IA permite la personalización a escala. 
            Al analizar datos de usuarios y preferencias, la IA puede crear contenido personalizado 
            que hable directamente a lectores individuales. Este nivel de personalización era 
            previamente imposible de lograr manualmente, especialmente para audiencias grandes.
            """
        elif "herramientas" in prompt.lower() or "tools" in prompt.lower():
            content = """
            El mercado está inundado de herramientas de creación de contenido con IA, cada una 
            ofreciendo capacidades y características únicas. Plataformas populares como GPT-4, 
            Claude y otros modelos de lenguaje proporcionan poderosas capacidades de generación 
            de texto. Estas herramientas pueden crear desde posts de blog y artículos hasta 
            copy de marketing y contenido para redes sociales.
            
            Al seleccionar herramientas de creación de contenido con IA, considera factores 
            como la calidad de salida, opciones de personalización, capacidades de integración 
            y costo. Algunas herramientas se especializan en tipos específicos de contenido, 
            mientras que otras ofrecen soluciones integrales para diversas necesidades de contenido. 
            La clave es elegir herramientas que se alineen con tus requisitos específicos y flujo de trabajo.
            """
        else:
            content = f"""
            Esta es una continuación de nuestra discusión sobre la creación de contenido impulsada por IA. 
            El contenido anterior ha sentado las bases para entender cómo la inteligencia artificial 
            puede transformar tu estrategia de contenido. Ahora, exploremos aspectos adicionales 
            y consideraciones que son cruciales para una implementación exitosa.
            
            A medida que continuamos esta exploración, es importante recordar que la IA es una herramienta 
            que mejora la creatividad humana en lugar de reemplazarla. Las estrategias de creación de 
            contenido más efectivas combinan la eficiencia y capacidades de la IA con la perspicacia, 
            creatividad y pensamiento estratégico humanos.
            """
        
        response_time = time.time() - start_time
        tokens_used = len(content.split()) * 1.3  # Rough estimate
        
        self.total_requests += 1
        self.total_tokens_used += int(tokens_used)
        
        return AIResponse(
            content=content,
            model=self.model,
            tokens_used=int(tokens_used),
            response_time=response_time,
            metadata={
                "mock_call_count": self.call_count,
                "generated_by": "mock_client"
            }
        )
    
    async def generate_title(self, content: str, **kwargs) -> AIResponse:
        """Generate mock title"""
        start_time = time.time()
        
        # Simple title generation based on content
        words = content.split()[:8]
        title = " ".join(words).title() + "..."
        
        # Ensure title is within character limit
        if len(title) > 60:
            title = title[:57] + "..."
        
        response_time = time.time() - start_time
        
        return AIResponse(
            content=title,
            model=self.model,
            tokens_used=10,
            response_time=response_time,
            metadata={"generated_by": "mock_client"}
        )

class AIClientFactory:
    """Factory for creating AI clients"""
    
    @staticmethod
    def create_client(client_type: str, api_key: str, model: str = None, **kwargs) -> AIClient:
        """Create AI client based on type"""
        
        client_configs = {
            "openai": {
                "class": OpenAIClient,
                "default_model": "gpt-4"
            },
            "anthropic": {
                "class": AnthropicClient,
                "default_model": "claude-3-sonnet-20240229"
            },
            "cohere": {
                "class": CohereClient,
                "default_model": "command"
            },
            "mock": {
                "class": MockAIClient,
                "default_model": "mock-gpt"
            }
        }
        
        if client_type not in client_configs:
            raise ValueError(f"Unsupported AI client type: {client_type}")
        
        config = client_configs[client_type]
        client_class = config["class"]
        default_model = config["default_model"]
        
        if not model:
            model = default_model
        
        try:
            return client_class(api_key=api_key, model=model, **kwargs)
        except ImportError as e:
            logger.error(f"Failed to create {client_type} client: {str(e)}")
            # Fallback to mock client
            logger.info("Falling back to mock client")
            return MockAIClient(api_key="fallback", model="mock-fallback")

class AIClientManager:
    """Manager for multiple AI clients with load balancing and failover"""
    
    def __init__(self):
        self.clients: List[AIClient] = []
        self.current_client_index = 0
        self.client_stats: Dict[str, Dict[str, Any]] = {}
    
    def add_client(self, client: AIClient):
        """Add a client to the manager"""
        self.clients.append(client)
        self.client_stats[client.model] = {
            "requests": 0,
            "errors": 0,
            "total_tokens": 0,
            "avg_response_time": 0.0
        }
    
    def get_next_client(self) -> AIClient:
        """Get next client using round-robin load balancing"""
        if not self.clients:
            raise RuntimeError("No AI clients available")
        
        client = self.clients[self.current_client_index]
        self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        return client
    
    async def generate_text(self, prompt: str, **kwargs) -> AIResponse:
        """Generate text using available clients with failover"""
        last_error = None
        
        for attempt in range(len(self.clients)):
            try:
                client = self.get_next_client()
                response = await client.generate_text(prompt, **kwargs)
                
                # Update stats
                self._update_client_stats(client.model, response, success=True)
                return response
                
            except Exception as e:
                last_error = e
                self._update_client_stats(client.model, None, success=False)
                logger.warning(f"Client {client.model} failed: {str(e)}")
                continue
        
        # All clients failed
        raise RuntimeError(f"All AI clients failed. Last error: {str(last_error)}")
    
    async def generate_title(self, content: str, **kwargs) -> AIResponse:
        """Generate title using available clients with failover"""
        last_error = None
        
        for attempt in range(len(self.clients)):
            try:
                client = self.get_next_client()
                response = await client.generate_title(content, **kwargs)
                
                # Update stats
                self._update_client_stats(client.model, response, success=True)
                return response
                
            except Exception as e:
                last_error = e
                self._update_client_stats(client.model, None, success=False)
                logger.warning(f"Client {client.model} failed: {str(e)}")
                continue
        
        # All clients failed
        raise RuntimeError(f"All AI clients failed. Last error: {str(last_error)}")
    
    def _update_client_stats(self, model: str, response: AIResponse, success: bool):
        """Update client statistics"""
        if model not in self.client_stats:
            self.client_stats[model] = {
                "requests": 0,
                "errors": 0,
                "total_tokens": 0,
                "avg_response_time": 0.0
            }
        
        stats = self.client_stats[model]
        
        if success and response:
            stats["requests"] += 1
            stats["total_tokens"] += response.tokens_used
            
            # Update average response time
            current_avg = stats["avg_response_time"]
            total_requests = stats["requests"]
            new_avg = ((current_avg * (total_requests - 1)) + response.response_time) / total_requests
            stats["avg_response_time"] = new_avg
        else:
            stats["errors"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all clients"""
        return {
            "total_clients": len(self.clients),
            "client_stats": self.client_stats,
            "current_client_index": self.current_client_index
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_ai_clients():
        """Test AI clients functionality"""
        print("🧪 Testing AI Clients")
        print("=" * 30)
        
        # Test mock client
        mock_client = MockAIClient()
        response = await mock_client.generate_text("Write about AI benefits")
        print(f"Mock client response: {response.content[:100]}...")
        print(f"Tokens used: {response.tokens_used}")
        
        # Test client factory
        factory = AIClientFactory()
        client = factory.create_client("mock", "test-key")
        print(f"Factory created client: {client.__class__.__name__}")
        
        # Test client manager
        manager = AIClientManager()
        manager.add_client(mock_client)
        
        response = await manager.generate_text("Test prompt")
        print(f"Manager response: {response.content[:100]}...")
        
        stats = manager.get_stats()
        print(f"Manager stats: {stats}")
    
    # Run test
    asyncio.run(test_ai_clients())



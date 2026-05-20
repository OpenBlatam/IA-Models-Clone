"""
BUL AI Integration Manager
=========================

Advanced AI integration management for the BUL system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import openai
import anthropic
from dataclasses import dataclass
from enum import Enum

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from modules.query_analyzer import QueryAnalyzer
from modules.document_processor import DocumentProcessor
from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIProvider(Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    LOCAL = "local"

@dataclass
class AIModel:
    """AI model configuration."""
    name: str
    provider: AIProvider
    max_tokens: int
    temperature: float
    cost_per_token: float
    capabilities: List[str]

class AIIntegrationManager:
    """Advanced AI integration management."""
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.available_models = {}
        self.active_models = {}
        self.usage_stats = {}
        self.init_ai_providers()
        self.setup_models()
    
    def init_ai_providers(self):
        """Initialize AI providers."""
        print("🤖 Initializing AI providers...")
        
        # OpenAI
        if self.config.openai_api_key:
            openai.api_key = self.config.openai_api_key
            print("✅ OpenAI provider initialized")
        
        # Anthropic
        if hasattr(self.config, 'anthropic_api_key') and self.config.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            print("✅ Anthropic provider initialized")
        
        # OpenRouter
        if self.config.openrouter_api_key:
            self.openrouter_api_key = self.config.openrouter_api_key
            print("✅ OpenRouter provider initialized")
    
    def setup_models(self):
        """Setup available AI models."""
        self.available_models = {
            # OpenAI Models
            "gpt-4": AIModel(
                name="gpt-4",
                provider=AIProvider.OPENAI,
                max_tokens=8192,
                temperature=0.7,
                cost_per_token=0.00003,
                capabilities=["text_generation", "analysis", "reasoning"]
            ),
            "gpt-4-turbo": AIModel(
                name="gpt-4-turbo",
                provider=AIProvider.OPENAI,
                max_tokens=128000,
                temperature=0.7,
                cost_per_token=0.00001,
                capabilities=["text_generation", "analysis", "reasoning", "long_context"]
            ),
            "gpt-3.5-turbo": AIModel(
                name="gpt-3.5-turbo",
                provider=AIProvider.OPENAI,
                max_tokens=4096,
                temperature=0.7,
                cost_per_token=0.000002,
                capabilities=["text_generation", "analysis"]
            ),
            
            # Anthropic Models
            "claude-3-opus": AIModel(
                name="claude-3-opus",
                provider=AIProvider.ANTHROPIC,
                max_tokens=200000,
                temperature=0.7,
                cost_per_token=0.000015,
                capabilities=["text_generation", "analysis", "reasoning", "long_context"]
            ),
            "claude-3-sonnet": AIModel(
                name="claude-3-sonnet",
                provider=AIProvider.ANTHROPIC,
                max_tokens=200000,
                temperature=0.7,
                cost_per_token=0.000003,
                capabilities=["text_generation", "analysis", "reasoning"]
            ),
            "claude-3-haiku": AIModel(
                name="claude-3-haiku",
                provider=AIProvider.ANTHROPIC,
                max_tokens=200000,
                temperature=0.7,
                cost_per_token=0.00000025,
                capabilities=["text_generation", "analysis"]
            ),
            
            # OpenRouter Models
            "llama-2-70b": AIModel(
                name="llama-2-70b",
                provider=AIProvider.OPENROUTER,
                max_tokens=4096,
                temperature=0.7,
                cost_per_token=0.0000007,
                capabilities=["text_generation", "analysis"]
            ),
            "mistral-7b": AIModel(
                name="mistral-7b",
                provider=AIProvider.OPENROUTER,
                max_tokens=32768,
                temperature=0.7,
                cost_per_token=0.0000002,
                capabilities=["text_generation", "analysis"]
            )
        }
        
        print(f"✅ Configured {len(self.available_models)} AI models")
    
    def get_available_models(self) -> Dict[str, AIModel]:
        """Get list of available AI models."""
        return self.available_models
    
    def select_optimal_model(self, task_type: str, complexity: str, budget: float = None) -> AIModel:
        """Select optimal model based on task requirements."""
        suitable_models = []
        
        for model_name, model in self.available_models.items():
            # Check if model supports required capabilities
            if task_type in model.capabilities:
                suitable_models.append((model_name, model))
        
        if not suitable_models:
            # Fallback to basic text generation
            suitable_models = [(name, model) for name, model in self.available_models.items() 
                             if "text_generation" in model.capabilities]
        
        if not suitable_models:
            raise ValueError("No suitable AI models available")
        
        # Select based on complexity and budget
        if complexity == "simple":
            # Prefer faster, cheaper models
            suitable_models.sort(key=lambda x: x[1].cost_per_token)
        elif complexity == "complex":
            # Prefer more capable models
            suitable_models.sort(key=lambda x: x[1].max_tokens, reverse=True)
        else:
            # Balanced selection
            suitable_models.sort(key=lambda x: x[1].cost_per_token)
        
        # Apply budget constraint if specified
        if budget is not None:
            suitable_models = [(name, model) for name, model in suitable_models 
                             if model.cost_per_token <= budget]
        
        return suitable_models[0][1] if suitable_models else suitable_models[0][1]
    
    async def generate_text(self, prompt: str, model_name: str = None, 
                          max_tokens: int = None, temperature: float = None) -> Dict[str, Any]:
        """Generate text using specified AI model."""
        if model_name is None:
            model = self.select_optimal_model("text_generation", "medium")
        else:
            model = self.available_models.get(model_name)
            if not model:
                raise ValueError(f"Model {model_name} not found")
        
        # Override model parameters if specified
        max_tokens = max_tokens or model.max_tokens
        temperature = temperature or model.temperature
        
        start_time = datetime.now()
        
        try:
            if model.provider == AIProvider.OPENAI:
                response = await self._call_openai(prompt, model.name, max_tokens, temperature)
            elif model.provider == AIProvider.ANTHROPIC:
                response = await self._call_anthropic(prompt, model.name, max_tokens, temperature)
            elif model.provider == AIProvider.OPENROUTER:
                response = await self._call_openrouter(prompt, model.name, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported provider: {model.provider}")
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Update usage statistics
            self._update_usage_stats(model.name, len(prompt), len(response.get('text', '')), processing_time)
            
            return {
                'text': response.get('text', ''),
                'model': model.name,
                'provider': model.provider.value,
                'processing_time': processing_time,
                'tokens_used': response.get('tokens_used', 0),
                'cost': response.get('cost', 0)
            }
            
        except Exception as e:
            logger.error(f"Error generating text with {model.name}: {e}")
            raise
    
    async def _call_openai(self, prompt: str, model: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Call OpenAI API."""
        try:
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            cost = tokens_used * self.available_models[model].cost_per_token
            
            return {
                'text': text,
                'tokens_used': tokens_used,
                'cost': cost
            }
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    async def _call_anthropic(self, prompt: str, model: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Call Anthropic API."""
        try:
            response = await self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            cost = tokens_used * self.available_models[model].cost_per_token
            
            return {
                'text': text,
                'tokens_used': tokens_used,
                'cost': cost
            }
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")
    
    async def _call_openrouter(self, prompt: str, model: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Call OpenRouter API."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    }
                )
                
                if response.status_code != 200:
                    raise Exception(f"OpenRouter API error: {response.status_code}")
                
                data = response.json()
                text = data['choices'][0]['message']['content']
                tokens_used = data['usage']['total_tokens']
                cost = tokens_used * self.available_models[model].cost_per_token
                
                return {
                    'text': text,
                    'tokens_used': tokens_used,
                    'cost': cost
                }
        except Exception as e:
            raise Exception(f"OpenRouter API error: {e}")
    
    def _update_usage_stats(self, model_name: str, input_tokens: int, output_tokens: int, processing_time: float):
        """Update usage statistics."""
        if model_name not in self.usage_stats:
            self.usage_stats[model_name] = {
                'total_requests': 0,
                'total_tokens': 0,
                'total_cost': 0,
                'total_time': 0,
                'average_response_time': 0
            }
        
        stats = self.usage_stats[model_name]
        stats['total_requests'] += 1
        stats['total_tokens'] += input_tokens + output_tokens
        stats['total_time'] += processing_time
        stats['average_response_time'] = stats['total_time'] / stats['total_requests']
        
        # Calculate cost
        model = self.available_models[model_name]
        cost = (input_tokens + output_tokens) * model.cost_per_token
        stats['total_cost'] += cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get AI usage statistics."""
        return self.usage_stats
    
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Get cost analysis for AI usage."""
        total_cost = sum(stats['total_cost'] for stats in self.usage_stats.values())
        total_requests = sum(stats['total_requests'] for stats in self.usage_stats.values())
        total_tokens = sum(stats['total_tokens'] for stats in self.usage_stats.values())
        
        return {
            'total_cost': total_cost,
            'total_requests': total_requests,
            'total_tokens': total_tokens,
            'average_cost_per_request': total_cost / total_requests if total_requests > 0 else 0,
            'average_tokens_per_request': total_tokens / total_requests if total_requests > 0 else 0,
            'model_breakdown': {
                model: {
                    'cost': stats['total_cost'],
                    'requests': stats['total_requests'],
                    'tokens': stats['total_tokens'],
                    'average_response_time': stats['average_response_time']
                }
                for model, stats in self.usage_stats.items()
            }
        }
    
    async def analyze_query_with_ai(self, query: str, model_name: str = None) -> Dict[str, Any]:
        """Analyze query using AI for enhanced understanding."""
        prompt = f"""
Analyze the following business query and provide detailed insights:

Query: "{query}"

Please provide:
1. Primary business area (marketing, sales, operations, hr, finance)
2. Document type needed (strategy, proposal, manual, policy, etc.)
3. Complexity level (simple, medium, complex)
4. Key requirements and constraints
5. Suggested approach and methodology
6. Estimated effort level (1-5 scale)

Format your response as JSON.
"""
        
        try:
            response = await self.generate_text(prompt, model_name)
            analysis = json.loads(response['text'])
            
            return {
                'query': query,
                'ai_analysis': analysis,
                'model_used': response['model'],
                'processing_time': response['processing_time'],
                'cost': response['cost']
            }
        except Exception as e:
            logger.error(f"Error analyzing query with AI: {e}")
            return {
                'query': query,
                'ai_analysis': None,
                'error': str(e)
            }
    
    async def enhance_document_with_ai(self, document_content: str, enhancement_type: str, 
                                     model_name: str = None) -> Dict[str, Any]:
        """Enhance document content using AI."""
        enhancement_prompts = {
            'improve_clarity': "Improve the clarity and readability of the following document while maintaining its professional tone:",
            'add_details': "Add more detailed information and examples to the following document:",
            'professional_tone': "Rewrite the following document with a more professional and polished tone:",
            'structure_improvement': "Improve the structure and organization of the following document:",
            'grammar_check': "Check and correct any grammar, spelling, or punctuation errors in the following document:"
        }
        
        if enhancement_type not in enhancement_prompts:
            raise ValueError(f"Unknown enhancement type: {enhancement_type}")
        
        prompt = f"{enhancement_prompts[enhancement_type]}\n\n{document_content}"
        
        try:
            response = await self.generate_text(prompt, model_name)
            
            return {
                'original_content': document_content,
                'enhanced_content': response['text'],
                'enhancement_type': enhancement_type,
                'model_used': response['model'],
                'processing_time': response['processing_time'],
                'cost': response['cost']
            }
        except Exception as e:
            logger.error(f"Error enhancing document with AI: {e}")
            return {
                'original_content': document_content,
                'enhanced_content': None,
                'enhancement_type': enhancement_type,
                'error': str(e)
            }
    
    def generate_ai_report(self) -> str:
        """Generate AI usage and performance report."""
        usage_stats = self.get_usage_stats()
        cost_analysis = self.get_cost_analysis()
        
        report = f"""
BUL AI Integration Report
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

AVAILABLE MODELS
----------------
Total Models: {len(self.available_models)}
"""
        
        for model_name, model in self.available_models.items():
            report += f"""
{model_name}:
  Provider: {model.provider.value}
  Max Tokens: {model.max_tokens:,}
  Temperature: {model.temperature}
  Cost per Token: ${model.cost_per_token:.8f}
  Capabilities: {', '.join(model.capabilities)}
"""
        
        report += f"""
USAGE STATISTICS
----------------
Total Requests: {cost_analysis['total_requests']}
Total Tokens: {cost_analysis['total_tokens']:,}
Total Cost: ${cost_analysis['total_cost']:.4f}
Average Cost per Request: ${cost_analysis['average_cost_per_request']:.4f}
Average Tokens per Request: {cost_analysis['average_tokens_per_request']:.0f}

MODEL BREAKDOWN
---------------
"""
        
        for model, stats in cost_analysis['model_breakdown'].items():
            report += f"""
{model}:
  Requests: {stats['requests']}
  Tokens: {stats['tokens']:,}
  Cost: ${stats['cost']:.4f}
  Average Response Time: {stats['average_response_time']:.2f}s
"""
        
        return report

def main():
    """Main AI integration manager function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL AI Integration Manager")
    parser.add_argument("--list-models", action="store_true", help="List available AI models")
    parser.add_argument("--test-model", help="Test a specific AI model")
    parser.add_argument("--analyze-query", help="Analyze a query using AI")
    parser.add_argument("--enhance-document", help="Enhance document content")
    parser.add_argument("--enhancement-type", choices=['improve_clarity', 'add_details', 'professional_tone', 'structure_improvement', 'grammar_check'],
                       default='improve_clarity', help="Type of enhancement")
    parser.add_argument("--model", help="Specific model to use")
    parser.add_argument("--report", action="store_true", help="Generate AI usage report")
    
    args = parser.parse_args()
    
    manager = AIIntegrationManager()
    
    print("🤖 BUL AI Integration Manager")
    print("=" * 40)
    
    if args.list_models:
        models = manager.get_available_models()
        print(f"\n📋 Available AI Models ({len(models)}):")
        print("-" * 50)
        
        for model_name, model in models.items():
            print(f"{model_name}:")
            print(f"  Provider: {model.provider.value}")
            print(f"  Max Tokens: {model.max_tokens:,}")
            print(f"  Cost per Token: ${model.cost_per_token:.8f}")
            print(f"  Capabilities: {', '.join(model.capabilities)}")
            print()
    
    elif args.test_model:
        async def test_model():
            try:
                response = await manager.generate_text(
                    "Write a brief business strategy for a new coffee shop.",
                    model_name=args.test_model
                )
                print(f"✅ Model test successful: {args.test_model}")
                print(f"Response: {response['text'][:200]}...")
                print(f"Processing Time: {response['processing_time']:.2f}s")
                print(f"Cost: ${response['cost']:.4f}")
            except Exception as e:
                print(f"❌ Model test failed: {e}")
        
        asyncio.run(test_model())
    
    elif args.analyze_query:
        async def analyze_query():
            try:
                analysis = await manager.analyze_query_with_ai(args.analyze_query, args.model)
                print(f"✅ Query analysis completed")
                print(f"AI Analysis: {json.dumps(analysis['ai_analysis'], indent=2)}")
                print(f"Model Used: {analysis['model_used']}")
                print(f"Processing Time: {analysis['processing_time']:.2f}s")
            except Exception as e:
                print(f"❌ Query analysis failed: {e}")
        
        asyncio.run(analyze_query())
    
    elif args.enhance_document:
        async def enhance_document():
            try:
                enhancement = await manager.enhance_document_with_ai(
                    args.enhance_document, args.enhancement_type, args.model
                )
                print(f"✅ Document enhancement completed")
                print(f"Enhancement Type: {enhancement['enhancement_type']}")
                print(f"Enhanced Content: {enhancement['enhanced_content'][:200]}...")
                print(f"Model Used: {enhancement['model_used']}")
                print(f"Processing Time: {enhancement['processing_time']:.2f}s")
            except Exception as e:
                print(f"❌ Document enhancement failed: {e}")
        
        asyncio.run(enhance_document())
    
    elif args.report:
        report = manager.generate_ai_report()
        print(report)
        
        # Save report
        report_file = f"ai_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    
    else:
        # Show quick overview
        models = manager.get_available_models()
        print(f"📋 Available Models: {len(models)}")
        print(f"💡 Use --list-models to see all models")
        print(f"💡 Use --test-model <model_name> to test a model")
        print(f"💡 Use --analyze-query 'your query' to analyze with AI")
        print(f"💡 Use --report to generate usage report")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

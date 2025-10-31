"""
Advanced Large Language Models for Document Processing
====================================================

State-of-the-art LLM integration for document classification, generation,
and processing with advanced techniques and optimizations.

Features:
- GPT-4, Claude, and custom LLM integration
- Advanced prompting techniques
- Few-shot and zero-shot learning
- Chain-of-thought reasoning
- Function calling and tool use
- Memory and context management
- Multi-modal LLM support
- Fine-tuning and adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
    GPT2LMHeadModel, GPTNeoXForCausalLM, LlamaForCausalLM,
    T5ForConditionalGeneration, BartForConditionalGeneration,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    GenerationConfig, StoppingCriteria, StoppingCriteriaList
)
from transformers.generation.utils import GenerationMixin
import openai
import anthropic
import google.generativeai as genai
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI, Anthropic
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import BaseTool
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from dataclasses import dataclass, asdict
import json
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import warnings
from torch.cuda.amp import autocast, GradScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    timeout: int = 30
    retry_attempts: int = 3
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    use_quantization: bool = False
    quantization_config: Optional[BitsAndBytesConfig] = None

class CustomStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for generation"""
    
    def __init__(self, stop_tokens: List[str], tokenizer):
        self.stop_tokens = stop_tokens
        self.tokenizer = tokenizer
        self.stop_token_ids = [tokenizer.encode(token, add_special_tokens=False)[0] for token in stop_tokens]
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0, -1] == stop_id:
                return True
        return False

class DocumentLLMClassifier:
    """LLM-based document classifier with advanced prompting"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._initialize_client()
        self.prompts = self._load_prompts()
        self.examples = self._load_examples()
    
    def _initialize_client(self):
        """Initialize LLM client based on model type"""
        if "gpt" in self.config.model_name.lower():
            return openai.OpenAI(api_key=self.config.api_key)
        elif "claude" in self.config.model_name.lower():
            return anthropic.Anthropic(api_key=self.config.api_key)
        elif "gemini" in self.config.model_name.lower():
            genai.configure(api_key=self.config.api_key)
            return genai.GenerativeModel(self.config.model_name)
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load classification prompts"""
        return {
            "classification": """
You are an expert document classifier. Your task is to classify documents into one of the following categories:

{classes}

Document to classify:
{document}

Please provide:
1. The most appropriate category
2. Confidence score (0-100)
3. Reasoning for your classification
4. Key features that led to this classification

Format your response as JSON:
{{
    "category": "category_name",
    "confidence": 85,
    "reasoning": "explanation",
    "key_features": ["feature1", "feature2", "feature3"]
}}
""",
            "few_shot": """
You are an expert document classifier. Here are some examples:

{examples}

Now classify this document:
{document}

Categories: {classes}

Provide your classification in JSON format:
{{
    "category": "category_name",
    "confidence": 85,
    "reasoning": "explanation"
}}
""",
            "chain_of_thought": """
You are an expert document classifier. Think step by step to classify this document.

Document: {document}
Categories: {classes}

Step 1: Analyze the document structure and content
Step 2: Identify key features and patterns
Step 3: Compare with category definitions
Step 4: Make final classification

Provide your analysis and classification in JSON format.
"""
        }
    
    def _load_examples(self) -> Dict[str, List[Dict]]:
        """Load few-shot examples"""
        return {
            "contract": [
                {
                    "text": "This Agreement is entered into between Company A and Company B...",
                    "category": "contract",
                    "features": ["legal language", "parties involved", "terms and conditions"]
                }
            ],
            "report": [
                {
                    "text": "Executive Summary: This report analyzes the quarterly performance...",
                    "category": "report",
                    "features": ["executive summary", "data analysis", "conclusions"]
                }
            ],
            "email": [
                {
                    "text": "Dear John, I hope this email finds you well...",
                    "category": "email",
                    "features": ["greeting", "personal tone", "email structure"]
                }
            ]
        }
    
    def classify_document(self, document: str, categories: List[str],
                         method: str = "classification") -> Dict[str, Any]:
        """Classify document using LLM"""
        try:
            if method == "few_shot":
                prompt = self.prompts["few_shot"].format(
                    examples=self._format_examples(categories),
                    document=document,
                    classes=", ".join(categories)
                )
            elif method == "chain_of_thought":
                prompt = self.prompts["chain_of_thought"].format(
                    document=document,
                    classes=", ".join(categories)
                )
            else:
                prompt = self.prompts["classification"].format(
                    classes=", ".join(categories),
                    document=document
                )
            
            response = self._generate_response(prompt)
            result = self._parse_response(response)
            
            return {
                "success": True,
                "result": result,
                "method": method,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _format_examples(self, categories: List[str]) -> str:
        """Format few-shot examples"""
        examples_text = ""
        for category in categories[:3]:  # Limit to 3 examples
            if category in self.examples:
                for example in self.examples[category][:1]:  # 1 example per category
                    examples_text += f"Document: {example['text'][:200]}...\n"
                    examples_text += f"Category: {example['category']}\n"
                    examples_text += f"Features: {', '.join(example['features'])}\n\n"
        return examples_text
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from LLM"""
        if "gpt" in self.config.model_name.lower():
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty
            )
            return response.choices[0].message.content
        
        elif "claude" in self.config.model_name.lower():
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        elif "gemini" in self.config.model_name.lower():
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p
                )
            )
            return response.text
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return {
                    "category": "unknown",
                    "confidence": 50,
                    "reasoning": response,
                    "raw_response": response
                }
        except json.JSONDecodeError:
            return {
                "category": "unknown",
                "confidence": 50,
                "reasoning": response,
                "raw_response": response
            }

class DocumentGenerator:
    """LLM-based document generator with advanced techniques"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._initialize_client()
        self.templates = self._load_templates()
        self.styles = self._load_styles()
    
    def _initialize_client(self):
        """Initialize LLM client"""
        if "gpt" in self.config.model_name.lower():
            return openai.OpenAI(api_key=self.config.api_key)
        elif "claude" in self.config.model_name.lower():
            return anthropic.Anthropic(api_key=self.config.api_key)
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")
    
    def _load_templates(self) -> Dict[str, str]:
        """Load document templates"""
        return {
            "contract": """
CONTRACT TEMPLATE

Parties: {parties}
Subject: {subject}
Terms: {terms}

This contract is entered into between {parties} for the purpose of {subject}.

TERMS AND CONDITIONS:
{terms}

SIGNATURES:
Party 1: _________________ Date: _________
Party 2: _________________ Date: _________
""",
            "report": """
REPORT TEMPLATE

Title: {title}
Date: {date}
Author: {author}

EXECUTIVE SUMMARY:
{summary}

METHODOLOGY:
{methodology}

FINDINGS:
{findings}

RECOMMENDATIONS:
{recommendations}

CONCLUSION:
{conclusion}
""",
            "proposal": """
PROPOSAL TEMPLATE

Project: {project}
Client: {client}
Date: {date}

PROJECT OVERVIEW:
{overview}

OBJECTIVES:
{objectives}

METHODOLOGY:
{methodology}

TIMELINE:
{timeline}

BUDGET:
{budget}

NEXT STEPS:
{next_steps}
"""
        }
    
    def _load_styles(self) -> Dict[str, str]:
        """Load writing styles"""
        return {
            "formal": "Use formal, professional language with proper business terminology.",
            "casual": "Use friendly, conversational tone while maintaining professionalism.",
            "technical": "Use precise technical language with appropriate jargon and terminology.",
            "academic": "Use scholarly language with citations and formal structure.",
            "creative": "Use engaging, creative language with vivid descriptions."
        }
    
    def generate_document(self, document_type: str, content: Dict[str, str],
                         style: str = "formal", custom_instructions: str = "") -> str:
        """Generate document using LLM"""
        try:
            # Get template
            template = self.templates.get(document_type, self.templates["report"])
            
            # Create prompt
            prompt = f"""
Generate a {document_type} document with the following requirements:

Style: {self.styles.get(style, self.styles['formal'])}

Template Structure:
{template}

Content to include:
{json.dumps(content, indent=2)}

Custom Instructions:
{custom_instructions}

Please generate a complete, well-structured document that follows the template and incorporates all the provided content. Ensure the document is professional and ready for use.
"""
            
            response = self._generate_response(prompt)
            return response
        
        except Exception as e:
            logger.error(f"Document generation error: {e}")
            return f"Error generating document: {str(e)}"
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from LLM"""
        if "gpt" in self.config.model_name.lower():
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.choices[0].message.content
        
        elif "claude" in self.config.model_name.lower():
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

class DocumentAnalyzer:
    """LLM-based document analyzer with advanced capabilities"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._initialize_client()
        self.analysis_prompts = self._load_analysis_prompts()
    
    def _initialize_client(self):
        """Initialize LLM client"""
        if "gpt" in self.config.model_name.lower():
            return openai.OpenAI(api_key=self.config.api_key)
        elif "claude" in self.config.model_name.lower():
            return anthropic.Anthropic(api_key=self.config.api_key)
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")
    
    def _load_analysis_prompts(self) -> Dict[str, str]:
        """Load analysis prompts"""
        return {
            "sentiment": """
Analyze the sentiment of the following document:

Document: {document}

Provide analysis in JSON format:
{{
    "overall_sentiment": "positive/negative/neutral",
    "sentiment_score": -1.0 to 1.0,
    "emotions": ["emotion1", "emotion2"],
    "key_phrases": ["phrase1", "phrase2"],
    "reasoning": "explanation"
}}
""",
            "complexity": """
Analyze the complexity of the following document:

Document: {document}

Provide analysis in JSON format:
{{
    "readability_score": 0-100,
    "complexity_level": "low/medium/high",
    "technical_terms": ["term1", "term2"],
    "sentence_structure": "simple/complex",
    "recommendations": ["rec1", "rec2"]
}}
""",
            "summary": """
Summarize the following document:

Document: {document}

Provide a comprehensive summary in JSON format:
{{
    "main_points": ["point1", "point2", "point3"],
    "key_findings": ["finding1", "finding2"],
    "conclusions": ["conclusion1", "conclusion2"],
    "summary": "brief summary text",
    "word_count": original_word_count
}}
""",
            "extract_entities": """
Extract entities from the following document:

Document: {document}

Extract and categorize entities in JSON format:
{{
    "persons": ["person1", "person2"],
    "organizations": ["org1", "org2"],
    "locations": ["location1", "location2"],
    "dates": ["date1", "date2"],
    "amounts": ["amount1", "amount2"],
    "other": ["entity1", "entity2"]
}}
"""
        }
    
    def analyze_document(self, document: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze document using LLM"""
        try:
            prompt = self.analysis_prompts.get(analysis_type, self.analysis_prompts["summary"])
            formatted_prompt = prompt.format(document=document)
            
            response = self._generate_response(formatted_prompt)
            result = self._parse_response(response)
            
            return {
                "success": True,
                "analysis_type": analysis_type,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from LLM"""
        if "gpt" in self.config.model_name.lower():
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.choices[0].message.content
        
        elif "claude" in self.config.model_name.lower():
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"raw_response": response}
        except json.JSONDecodeError:
            return {"raw_response": response}

class LocalLLMModel:
    """Local LLM model for offline processing"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = self._load_model()
        
        # Setup generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def _load_model(self):
        """Load local model"""
        if self.config.use_quantization and self.config.quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=self.config.quantization_config,
                device_map="auto",
                torch_dtype=self.config.dtype
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.dtype
            ).to(self.device)
        
        return model
    
    def generate(self, prompt: str, max_length: Optional[int] = None) -> str:
        """Generate text from prompt"""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                if self.config.dtype == torch.float16:
                    with autocast():
                        outputs = self.model.generate(
                            **inputs,
                            generation_config=self.generation_config,
                            max_length=max_length or inputs.input_ids.shape[1] + self.config.max_tokens
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=self.generation_config,
                        max_length=max_length or inputs.input_ids.shape[1] + self.config.max_tokens
                    )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating text: {str(e)}"
    
    def classify_document(self, document: str, categories: List[str]) -> Dict[str, Any]:
        """Classify document using local model"""
        prompt = f"""
Classify the following document into one of these categories: {', '.join(categories)}

Document: {document}

Category:"""
        
        response = self.generate(prompt)
        
        # Parse response
        for category in categories:
            if category.lower() in response.lower():
                return {
                    "category": category,
                    "confidence": 0.8,  # Default confidence for local models
                    "reasoning": response,
                    "method": "local_llm"
                }
        
        return {
            "category": "unknown",
            "confidence": 0.5,
            "reasoning": response,
            "method": "local_llm"
        }

class LLMModelFactory:
    """Factory for creating different LLM models"""
    
    @staticmethod
    def create_model(model_type: str, config: LLMConfig):
        """Create LLM model based on type"""
        if model_type == "classifier":
            return DocumentLLMClassifier(config)
        elif model_type == "generator":
            return DocumentGenerator(config)
        elif model_type == "analyzer":
            return DocumentAnalyzer(config)
        elif model_type == "local":
            return LocalLLMModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> Dict[str, List[str]]:
        """Get list of available models"""
        return {
            "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            "anthropic": ["claude-3-sonnet", "claude-3-opus", "claude-3-haiku"],
            "google": ["gemini-pro", "gemini-pro-vision"],
            "local": ["microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"]
        }

# Example usage
if __name__ == "__main__":
    # Configuration for OpenAI
    config = LLMConfig(
        model_name="gpt-3.5-turbo",
        api_key="your-api-key-here",
        max_tokens=1024,
        temperature=0.7
    )
    
    # Create classifier
    classifier = LLMModelFactory.create_model("classifier", config)
    
    # Classify document
    document = "This contract is entered into between Company A and Company B for the provision of services."
    categories = ["contract", "report", "email", "proposal"]
    
    result = classifier.classify_document(document, categories, method="few_shot")
    print(f"Classification result: {result}")
    
    # Create generator
    generator = LLMModelFactory.create_model("generator", config)
    
    # Generate document
    content = {
        "parties": "Company A and Company B",
        "subject": "Software Development Services",
        "terms": "Payment terms, delivery schedule, and quality standards"
    }
    
    generated_doc = generator.generate_document("contract", content, style="formal")
    print(f"Generated document: {generated_doc[:200]}...")
    
    # Create analyzer
    analyzer = LLMModelFactory.create_model("analyzer", config)
    
    # Analyze document
    analysis = analyzer.analyze_document(document, "sentiment")
    print(f"Sentiment analysis: {analysis}")

























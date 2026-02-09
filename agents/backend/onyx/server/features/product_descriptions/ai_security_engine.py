from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse
import aiohttp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from fastapi import BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator
from transformers import (
from sentence_transformers import SentenceTransformer
import gradio as gr
import structlog
from typing import Any, List, Dict, Optional
"""
AI-Powered Security Engine with Deep Learning and Transformer Models
Advanced cybersecurity analysis using PyTorch, Transformers, and LLM capabilities
"""


    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    pipeline,
    TextClassificationPipeline,
    TokenClassificationPipeline,
    GenerationConfig
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class ThreatLevel(str, Enum):
    """Threat level enumeration"""
    BENIGN = "benign"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AIModelType(str, Enum):
    """AI model types"""
    TRANSFORMER = "transformer"
    DIFFUSION = "diffusion"
    LLM = "llm"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    NER = "ner"


class SecurityDomain(str, Enum):
    """Security analysis domains"""
    NETWORK = "network"
    WEB_APPLICATION = "web_application"
    MALWARE = "malware"
    PHISHING = "phishing"
    SOCIAL_ENGINEERING = "social_engineering"
    CODE_ANALYSIS = "code_analysis"
    LOG_ANALYSIS = "log_analysis"
    THREAT_INTELLIGENCE = "threat_intelligence"


@dataclass
class AIThreatAnalysis:
    """AI-powered threat analysis result"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    threat_level: ThreatLevel = ThreatLevel.BENIGN
    confidence_score: float = 0.0
    model_used: str = ""
    analysis_type: str = ""
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    processing_time: float = 0.0


@dataclass
class SecurityInput:
    """Security analysis input"""
    content: str
    content_type: str  # text, url, code, log, image
    domain: SecurityDomain
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AISecurityResult:
    """AI security analysis result"""
    input_id: str
    threat_analysis: AIThreatAnalysis
    model_predictions: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    generated_text: Optional[str] = None
    visual_analysis: Optional[str] = None  # Base64 encoded image or path


class AISecurityConfiguration(BaseModel):
    """AI Security Engine configuration"""
    # Model configurations
    enable_transformer_models: bool = True
    enable_diffusion_models: bool = False
    enable_llm_models: bool = True
    enable_embedding_models: bool = True
    
    # Model paths and configurations
    transformer_model_path: str = Field(default="microsoft/DialoGPT-medium")
    classification_model_path: str = Field(default="microsoft/DialoGPT-medium")
    ner_model_path: str = Field(default="dbmdz/bert-large-cased-finetuned-conll03-english")
    embedding_model_path: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    llm_model_path: str = Field(default="microsoft/DialoGPT-medium")
    diffusion_model_path: str = Field(default="runwayml/stable-diffusion-v1-5")
    
    # Processing settings
    max_sequence_length: int = Field(default=512, ge=128, le=2048)
    batch_size: int = Field(default=8, ge=1, le=32)
    device: str = Field(default="auto")  # auto, cpu, cuda, mps
    
    # Confidence thresholds
    threat_detection_threshold: float = Field(default=0.7, ge=0.1, le=1.0)
    false_positive_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Performance settings
    enable_gpu_acceleration: bool = True
    enable_model_caching: bool = True
    enable_batch_processing: bool = True
    max_concurrent_analyses: int = Field(default=10, ge=1, le=50)
    
    # Output settings
    enable_attention_visualization: bool = True
    enable_embedding_analysis: bool = True
    enable_text_generation: bool = True
    enable_image_generation: bool = False
    
    @validator('device')
    def validate_device(cls, v) -> bool:
        if v == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return v


class TransformerSecurityModel(nn.Module):
    """Custom transformer model for security analysis"""
    
    def __init__(self, model_name: str, num_classes: int = 5):
        
    """__init__ function."""
super().__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_classes
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None) -> Any:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class SecurityEmbeddingModel:
    """Security-focused embedding model"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        
    """__init__ function."""
self.device = device
        self.model = SentenceTransformer(model_path, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        embeddings = self.encode([text1, text2])
        return np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )


class AISecurityEngine:
    """AI-powered security analysis engine"""
    
    def __init__(self, config: AISecurityConfiguration):
        
    """__init__ function."""
self.config = config
        self.device = torch.device(config.device)
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self._analysis_semaphore = asyncio.Semaphore(config.max_concurrent_analyses)
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self) -> Any:
        """Initialize AI models"""
        logger.info("Initializing AI Security Models", device=str(self.device))
        
        try:
            # Initialize tokenizers
            if self.config.enable_transformer_models:
                self.tokenizers['transformer'] = AutoTokenizer.from_pretrained(
                    self.config.transformer_model_path
                )
                self.tokenizers['classification'] = AutoTokenizer.from_pretrained(
                    self.config.classification_model_path
                )
                self.tokenizers['ner'] = AutoTokenizer.from_pretrained(
                    self.config.ner_model_path
                )
                
                # Add padding token if not present
                for tokenizer in self.tokenizers.values():
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
            
            # Initialize embedding model
            if self.config.enable_embedding_models:
                self.models['embedding'] = SecurityEmbeddingModel(
                    self.config.embedding_model_path,
                    self.config.device
                )
            
            # Initialize classification pipeline
            if self.config.enable_transformer_models:
                self.pipelines['classification'] = pipeline(
                    "text-classification",
                    model=self.config.classification_model_path,
                    device=self.config.device,
                    return_all_scores=True
                )
                
                self.pipelines['ner'] = pipeline(
                    "token-classification",
                    model=self.config.ner_model_path,
                    device=self.config.device
                )
            
            # Initialize LLM model
            if self.config.enable_llm_models:
                self.models['llm'] = AutoModelForCausalLM.from_pretrained(
                    self.config.llm_model_path,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None
                )
                self.tokenizers['llm'] = AutoTokenizer.from_pretrained(
                    self.config.llm_model_path
                )
                
                # Add padding token if not present
                if self.tokenizers['llm'].pad_token is None:
                    self.tokenizers['llm'].pad_token = self.tokenizers['llm'].eos_token
            
            # Initialize diffusion model (optional)
            if self.config.enable_diffusion_models:
                self.models['diffusion'] = StableDiffusionPipeline.from_pretrained(
                    self.config.diffusion_model_path,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                ).to(self.device)
            
            logger.info("AI Security Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            raise
    
    async def analyze_security_input(self, security_input: SecurityInput) -> AISecurityResult:
        """Analyze security input using AI models"""
        async with self._analysis_semaphore:
            start_time = time.time()
            
            analysis_id = str(uuid.uuid4())
            logger.info("Starting AI security analysis", 
                       analysis_id=analysis_id,
                       content_type=security_input.content_type,
                       domain=security_input.domain.value)
            
            try:
                # Initialize result
                threat_analysis = AIThreatAnalysis(
                    id=analysis_id,
                    model_used="ai_security_engine",
                    analysis_type=security_input.content_type
                )
                
                model_predictions = {}
                
                # Perform domain-specific analysis
                if security_input.domain == SecurityDomain.WEB_APPLICATION:
                    model_predictions.update(
                        await self._analyze_web_application(security_input.content)
                    )
                elif security_input.domain == SecurityDomain.MALWARE:
                    model_predictions.update(
                        await self._analyze_malware(security_input.content)
                    )
                elif security_input.domain == SecurityDomain.PHISHING:
                    model_predictions.update(
                        await self._analyze_phishing(security_input.content)
                    )
                elif security_input.domain == SecurityDomain.CODE_ANALYSIS:
                    model_predictions.update(
                        await self._analyze_code(security_input.content)
                    )
                elif security_input.domain == SecurityDomain.LOG_ANALYSIS:
                    model_predictions.update(
                        await self._analyze_logs(security_input.content)
                    )
                else:
                    # Generic analysis
                    model_predictions.update(
                        await self._analyze_generic(security_input.content)
                    )
                
                # Generate embeddings
                embeddings = None
                if self.config.enable_embedding_models:
                    embeddings = await self._generate_embeddings(security_input.content)
                
                # Generate threat analysis
                threat_analysis = await self._generate_threat_analysis(
                    model_predictions, security_input
                )
                
                # Generate recommendations
                recommendations = await self._generate_recommendations(
                    threat_analysis, security_input
                )
                threat_analysis.recommendations = recommendations
                
                # Calculate processing time
                processing_time = time.time() - start_time
                threat_analysis.processing_time = processing_time
                
                logger.info("AI security analysis completed",
                           analysis_id=analysis_id,
                           threat_level=threat_analysis.threat_level.value,
                           confidence=threat_analysis.confidence_score,
                           processing_time=processing_time)
                
                return AISecurityResult(
                    input_id=analysis_id,
                    threat_analysis=threat_analysis,
                    model_predictions=model_predictions,
                    embeddings=embeddings
                )
                
            except Exception as e:
                logger.error(f"AI security analysis failed: {e}", exc_info=True)
                raise
    
    async def _analyze_web_application(self, content: str) -> Dict[str, Any]:
        """Analyze web application security"""
        predictions = {}
        
        # Text classification for security patterns
        if 'classification' in self.pipelines:
            try:
                classification_result = self.pipelines['classification'](content)
                predictions['security_classification'] = classification_result
            except Exception as e:
                logger.warning(f"Classification failed: {e}")
        
        # Named Entity Recognition for sensitive information
        if 'ner' in self.pipelines:
            try:
                ner_result = self.pipelines['ner'](content)
                predictions['named_entities'] = ner_result
                
                # Check for sensitive entities
                sensitive_entities = [
                    entity for entity in ner_result 
                    if entity['entity'] in ['PERSON', 'ORG', 'LOC', 'MISC']
                ]
                predictions['sensitive_entities'] = sensitive_entities
            except Exception as e:
                logger.warning(f"NER failed: {e}")
        
        # Embedding-based similarity analysis
        if 'embedding' in self.models:
            try:
                # Compare with known security patterns
                security_patterns = [
                    "SQL injection vulnerability",
                    "Cross-site scripting attack",
                    "Authentication bypass",
                    "Privilege escalation",
                    "Data exposure"
                ]
                
                similarities = []
                for pattern in security_patterns:
                    similarity = self.models['embedding'].similarity(content, pattern)
                    similarities.append((pattern, similarity))
                
                predictions['security_similarities'] = similarities
            except Exception as e:
                logger.warning(f"Embedding analysis failed: {e}")
        
        return predictions
    
    async def _analyze_malware(self, content: str) -> Dict[str, Any]:
        """Analyze malware characteristics"""
        predictions = {}
        
        # Text classification for malware patterns
        if 'classification' in self.pipelines:
            try:
                classification_result = self.pipelines['classification'](content)
                predictions['malware_classification'] = classification_result
            except Exception as e:
                logger.warning(f"Malware classification failed: {e}")
        
        # Named Entity Recognition for malicious entities
        if 'ner' in self.pipelines:
            try:
                ner_result = self.pipelines['ner'](content)
                predictions['malicious_entities'] = ner_result
            except Exception as e:
                logger.warning(f"Malware NER failed: {e}")
        
        # Embedding analysis for malware signatures
        if 'embedding' in self.models:
            try:
                malware_patterns = [
                    "malware signature",
                    "trojan horse",
                    "ransomware attack",
                    "keylogger",
                    "backdoor access"
                ]
                
                similarities = []
                for pattern in malware_patterns:
                    similarity = self.models['embedding'].similarity(content, pattern)
                    similarities.append((pattern, similarity))
                
                predictions['malware_similarities'] = similarities
            except Exception as e:
                logger.warning(f"Malware embedding analysis failed: {e}")
        
        return predictions
    
    async def _analyze_phishing(self, content: str) -> Dict[str, Any]:
        """Analyze phishing attempts"""
        predictions = {}
        
        # Text classification for phishing patterns
        if 'classification' in self.pipelines:
            try:
                classification_result = self.pipelines['classification'](content)
                predictions['phishing_classification'] = classification_result
            except Exception as e:
                logger.warning(f"Phishing classification failed: {e}")
        
        # Named Entity Recognition for suspicious entities
        if 'ner' in self.pipelines:
            try:
                ner_result = self.pipelines['ner'](content)
                predictions['suspicious_entities'] = ner_result
            except Exception as e:
                logger.warning(f"Phishing NER failed: {e}")
        
        # Embedding analysis for phishing indicators
        if 'embedding' in self.models:
            try:
                phishing_patterns = [
                    "urgent action required",
                    "account suspended",
                    "verify your identity",
                    "click here immediately",
                    "bank account locked"
                ]
                
                similarities = []
                for pattern in phishing_patterns:
                    similarity = self.models['embedding'].similarity(content, pattern)
                    similarities.append((pattern, similarity))
                
                predictions['phishing_similarities'] = similarities
            except Exception as e:
                logger.warning(f"Phishing embedding analysis failed: {e}")
        
        return predictions
    
    async def _analyze_code(self, content: str) -> Dict[str, Any]:
        """Analyze code for security vulnerabilities"""
        predictions = {}
        
        # Text classification for code vulnerabilities
        if 'classification' in self.pipelines:
            try:
                classification_result = self.pipelines['classification'](content)
                predictions['vulnerability_classification'] = classification_result
            except Exception as e:
                logger.warning(f"Code classification failed: {e}")
        
        # Named Entity Recognition for code entities
        if 'ner' in self.pipelines:
            try:
                ner_result = self.pipelines['ner'](content)
                predictions['code_entities'] = ner_result
            except Exception as e:
                logger.warning(f"Code NER failed: {e}")
        
        # Embedding analysis for vulnerability patterns
        if 'embedding' in self.models:
            try:
                vulnerability_patterns = [
                    "buffer overflow",
                    "memory leak",
                    "race condition",
                    "null pointer dereference",
                    "integer overflow"
                ]
                
                similarities = []
                for pattern in vulnerability_patterns:
                    similarity = self.models['embedding'].similarity(content, pattern)
                    similarities.append((pattern, similarity))
                
                predictions['vulnerability_similarities'] = similarities
            except Exception as e:
                logger.warning(f"Code embedding analysis failed: {e}")
        
        return predictions
    
    async def _analyze_logs(self, content: str) -> Dict[str, Any]:
        """Analyze security logs"""
        predictions = {}
        
        # Text classification for log patterns
        if 'classification' in self.pipelines:
            try:
                classification_result = self.pipelines['classification'](content)
                predictions['log_classification'] = classification_result
            except Exception as e:
                logger.warning(f"Log classification failed: {e}")
        
        # Named Entity Recognition for log entities
        if 'ner' in self.pipelines:
            try:
                ner_result = self.pipelines['ner'](content)
                predictions['log_entities'] = ner_result
            except Exception as e:
                logger.warning(f"Log NER failed: {e}")
        
        # Embedding analysis for threat patterns
        if 'embedding' in self.models:
            try:
                threat_patterns = [
                    "failed login attempt",
                    "unauthorized access",
                    "suspicious activity",
                    "data breach",
                    "network intrusion"
                ]
                
                similarities = []
                for pattern in threat_patterns:
                    similarity = self.models['embedding'].similarity(content, pattern)
                    similarities.append((pattern, similarity))
                
                predictions['threat_similarities'] = similarities
            except Exception as e:
                logger.warning(f"Log embedding analysis failed: {e}")
        
        return predictions
    
    async def _analyze_generic(self, content: str) -> Dict[str, Any]:
        """Generic security analysis"""
        predictions = {}
        
        # Basic text classification
        if 'classification' in self.pipelines:
            try:
                classification_result = self.pipelines['classification'](content)
                predictions['generic_classification'] = classification_result
            except Exception as e:
                logger.warning(f"Generic classification failed: {e}")
        
        # Named Entity Recognition
        if 'ner' in self.pipelines:
            try:
                ner_result = self.pipelines['ner'](content)
                predictions['generic_entities'] = ner_result
            except Exception as e:
                logger.warning(f"Generic NER failed: {e}")
        
        return predictions
    
    async def _generate_embeddings(self, content: str) -> np.ndarray:
        """Generate embeddings for content"""
        try:
            embeddings = self.models['embedding'].encode([content])
            return embeddings[0] if len(embeddings) > 0 else None
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None
    
    async def _generate_threat_analysis(
        self, 
        model_predictions: Dict[str, Any], 
        security_input: SecurityInput
    ) -> AIThreatAnalysis:
        """Generate threat analysis from model predictions"""
        threat_analysis = AIThreatAnalysis()
        
        # Analyze classification results
        max_confidence = 0.0
        threat_scores = []
        
        for prediction_type, predictions in model_predictions.items():
            if 'classification' in prediction_type and predictions:
                for prediction in predictions:
                    if isinstance(prediction, dict) and 'score' in prediction:
                        confidence = prediction['score']
                        max_confidence = max(max_confidence, confidence)
                        threat_scores.append(confidence)
        
        # Analyze similarity results
        for prediction_type, similarities in model_predictions.items():
            if 'similarities' in prediction_type and similarities:
                for pattern, similarity in similarities:
                    max_confidence = max(max_confidence, similarity)
                    threat_scores.append(similarity)
        
        # Determine threat level based on confidence scores
        if threat_scores:
            avg_threat_score = np.mean(threat_scores)
            max_threat_score = max(threat_scores)
            
            if max_threat_score >= self.config.threat_detection_threshold:
                if max_threat_score >= 0.9:
                    threat_analysis.threat_level = ThreatLevel.CRITICAL
                elif max_threat_score >= 0.8:
                    threat_analysis.threat_level = ThreatLevel.HIGH
                elif max_threat_score >= 0.7:
                    threat_analysis.threat_level = ThreatLevel.MEDIUM
                else:
                    threat_analysis.threat_level = ThreatLevel.LOW
            else:
                threat_analysis.threat_level = ThreatLevel.BENIGN
            
            threat_analysis.confidence_score = max_threat_score
        else:
            threat_analysis.threat_level = ThreatLevel.BENIGN
            threat_analysis.confidence_score = 0.0
        
        # Generate findings
        findings = []
        for prediction_type, predictions in model_predictions.items():
            if 'similarities' in prediction_type and predictions:
                for pattern, similarity in predictions:
                    if similarity >= self.config.threat_detection_threshold:
                        findings.append(f"High similarity to '{pattern}' pattern ({similarity:.2f})")
        
        threat_analysis.findings = findings
        
        return threat_analysis
    
    async def _generate_recommendations(
        self, 
        threat_analysis: AIThreatAnalysis, 
        security_input: SecurityInput
    ) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        if threat_analysis.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            recommendations.extend([
                "Immediate action required - investigate further",
                "Implement additional security controls",
                "Monitor for similar patterns",
                "Consider incident response procedures"
            ])
        elif threat_analysis.threat_level == ThreatLevel.MEDIUM:
            recommendations.extend([
                "Review security controls",
                "Monitor for escalation",
                "Update security policies if needed"
            ])
        elif threat_analysis.threat_level == ThreatLevel.LOW:
            recommendations.extend([
                "Continue monitoring",
                "Review security best practices"
            ])
        else:
            recommendations.append("No immediate action required")
        
        # Domain-specific recommendations
        if security_input.domain == SecurityDomain.WEB_APPLICATION:
            recommendations.extend([
                "Implement input validation",
                "Use parameterized queries",
                "Enable security headers"
            ])
        elif security_input.domain == SecurityDomain.MALWARE:
            recommendations.extend([
                "Update antivirus signatures",
                "Scan system thoroughly",
                "Isolate affected systems"
            ])
        elif security_input.domain == SecurityDomain.PHISHING:
            recommendations.extend([
                "Educate users about phishing",
                "Implement email filtering",
                "Enable multi-factor authentication"
            ])
        
        return recommendations
    
    async def generate_security_report(
        self, 
        content: str, 
        domain: SecurityDomain
    ) -> str:
        """Generate comprehensive security report using LLM"""
        if 'llm' not in self.models:
            return "LLM model not available for report generation"
        
        try:
            # Create prompt for security analysis
            prompt = f"""
            Analyze the following {domain.value} content for security threats and provide a detailed report:
            
            Content: {content}
            
            Please provide a comprehensive security analysis including:
            1. Threat assessment
            2. Risk level
            3. Specific vulnerabilities or issues
            4. Recommendations for mitigation
            5. Best practices to follow
            """
            
            # Tokenize input
            inputs = self.tokenizers['llm'](
                prompt, 
                return_tensors="pt", 
                max_length=self.config.max_sequence_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.models['llm'].generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 500,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizers['llm'].eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizers['llm'].decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Extract the generated part
            report = generated_text[len(prompt):].strip()
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Error generating report: {str(e)}"
    
    async def generate_threat_visualization(
        self, 
        threat_analysis: AIThreatAnalysis,
        content: str
    ) -> Optional[str]:
        """Generate visual representation of threat analysis"""
        if 'diffusion' not in self.models or not self.config.enable_image_generation:
            return None
        
        try:
            # Create prompt based on threat level
            threat_prompts = {
                ThreatLevel.CRITICAL: "cybersecurity threat alert red warning danger",
                ThreatLevel.HIGH: "security vulnerability high risk warning",
                ThreatLevel.MEDIUM: "security concern medium risk",
                ThreatLevel.LOW: "security monitoring low risk",
                ThreatLevel.BENIGN: "secure system green checkmark"
            }
            
            prompt = threat_prompts.get(threat_analysis.threat_level, "security analysis")
            
            # Generate image
            image = self.models['diffusion'](
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
            
            # Save image (in production, you might want to save to cloud storage)
            image_path = f"threat_visualization_{threat_analysis.id}.png"
            image.save(image_path)
            
            return image_path
            
        except Exception as e:
            logger.error(f"Threat visualization failed: {e}")
            return None
    
    async def batch_analyze(
        self, 
        security_inputs: List[SecurityInput]
    ) -> List[AISecurityResult]:
        """Analyze multiple security inputs in batch"""
        logger.info(f"Starting batch analysis of {len(security_inputs)} inputs")
        
        # Process in batches
        batch_size = self.config.batch_size
        results = []
        
        for i in range(0, len(security_inputs), batch_size):
            batch = security_inputs[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.analyze_security_input(input_data) 
                for input_data in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch analysis failed: {result}")
                    # Create error result
                    error_result = AISecurityResult(
                        input_id=str(uuid.uuid4()),
                        threat_analysis=AIThreatAnalysis(
                            threat_level=ThreatLevel.BENIGN,
                            confidence_score=0.0,
                            findings=[f"Analysis failed: {str(result)}"]
                        )
                    )
                    results.append(error_result)
                else:
                    results.append(result)
        
        logger.info(f"Batch analysis completed: {len(results)} results")
        return results
    
    async def shutdown(self) -> Any:
        """Shutdown AI security engine"""
        logger.info("Shutting down AI Security Engine")
        
        # Clear models from memory
        for model_name, model in self.models.items():
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
        
        for tokenizer_name, tokenizer in self.tokenizers.items():
            del tokenizer
        
        for pipeline_name, pipeline_obj in self.pipelines.items():
            del pipeline_obj
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("AI Security Engine shutdown completed")


# FastAPI Integration
class AISecurityRequest(BaseModel):
    """AI Security analysis request"""
    content: str
    content_type: str = "text"
    domain: SecurityDomain = SecurityDomain.WEB_APPLICATION
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('content')
    def validate_content(cls, v) -> bool:
        if not v or len(v.strip()) == 0:
            raise ValueError('Content cannot be empty')
        return v


class AISecurityResponse(BaseModel):
    """AI Security analysis response"""
    analysis_id: str
    threat_level: ThreatLevel
    confidence_score: float
    findings: List[str]
    recommendations: List[str]
    processing_time: float
    model_metadata: Dict[str, Any] = Field(default_factory=dict)


# Dependency injection
_ai_security_engine: Optional[AISecurityEngine] = None


async def get_ai_security_engine() -> AISecurityEngine:
    """Get AI security engine instance"""
    global _ai_security_engine
    if _ai_security_engine is None:
        config = AISecurityConfiguration()
        _ai_security_engine = AISecurityEngine(config)
    return _ai_security_engine


async def cleanup_ai_security_engine():
    """Cleanup AI security engine on shutdown"""
    global _ai_security_engine
    if _ai_security_engine:
        await _ai_security_engine.shutdown()
        _ai_security_engine = None


# FastAPI routes
async def analyze_security(
    request: AISecurityRequest,
    engine: AISecurityEngine = Depends(get_ai_security_engine)
) -> AISecurityResponse:
    """Analyze security content using AI"""
    security_input = SecurityInput(
        content=request.content,
        content_type=request.content_type,
        domain=request.domain,
        metadata=request.metadata
    )
    
    result = await engine.analyze_security_input(security_input)
    
    return AISecurityResponse(
        analysis_id=result.threat_analysis.id,
        threat_level=result.threat_analysis.threat_level,
        confidence_score=result.threat_analysis.confidence_score,
        findings=result.threat_analysis.findings,
        recommendations=result.threat_analysis.recommendations,
        processing_time=result.threat_analysis.processing_time,
        model_metadata=result.threat_analysis.model_metadata
    )


async def generate_security_report(
    content: str,
    domain: SecurityDomain = SecurityDomain.WEB_APPLICATION,
    engine: AISecurityEngine = Depends(get_ai_security_engine)
) -> Dict[str, str]:
    """Generate comprehensive security report"""
    report = await engine.generate_security_report(content, domain)
    
    return {
        "report": report,
        "domain": domain.value,
        "timestamp": time.time()
    }


async def batch_analyze_security(
    requests: List[AISecurityRequest],
    engine: AISecurityEngine = Depends(get_ai_security_engine)
) -> List[AISecurityResponse]:
    """Batch analyze multiple security inputs"""
    security_inputs = [
        SecurityInput(
            content=req.content,
            content_type=req.content_type,
            domain=req.domain,
            metadata=req.metadata
        )
        for req in requests
    ]
    
    results = await engine.batch_analyze(security_inputs)
    
    return [
        AISecurityResponse(
            analysis_id=result.threat_analysis.id,
            threat_level=result.threat_analysis.threat_level,
            confidence_score=result.threat_analysis.confidence_score,
            findings=result.threat_analysis.findings,
            recommendations=result.threat_analysis.recommendations,
            processing_time=result.threat_analysis.processing_time,
            model_metadata=result.threat_analysis.model_metadata
        )
        for result in results
    ]


# Gradio Interface
def create_gradio_interface():
    """Create Gradio interface for AI Security Engine"""
    
    async def analyze_text(text, domain) -> Any:
        """Analyze text using AI security engine"""
        try:
            engine = await get_ai_security_engine()
            
            security_input = SecurityInput(
                content=text,
                content_type="text",
                domain=SecurityDomain(domain)
            )
            
            result = await engine.analyze_security_input(security_input)
            
            return {
                "Threat Level": result.threat_analysis.threat_level.value,
                "Confidence": f"{result.threat_analysis.confidence_score:.2f}",
                "Findings": "\n".join(result.threat_analysis.findings),
                "Recommendations": "\n".join(result.threat_analysis.recommendations),
                "Processing Time": f"{result.threat_analysis.processing_time:.2f}s"
            }
        except Exception as e:
            return {"Error": str(e)}
    
    # Create Gradio interface
    iface = gr.Interface(
        fn=analyze_text,
        inputs=[
            gr.Textbox(label="Security Content", lines=5),
            gr.Dropdown(
                choices=[domain.value for domain in SecurityDomain],
                label="Security Domain",
                value=SecurityDomain.WEB_APPLICATION.value
            )
        ],
        outputs=gr.JSON(label="Analysis Results"),
        title="AI Security Analysis Engine",
        description="Analyze security content using advanced AI models",
        examples=[
            ["SELECT * FROM users WHERE id = 1 OR 1=1", "web_application"],
            ["Your account has been suspended. Click here to verify.", "phishing"],
            ["Failed login attempt from 192.168.1.100", "log_analysis"]
        ]
    )
    
    return iface


if __name__ == "__main__":
    # Create and launch Gradio interface
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860) 
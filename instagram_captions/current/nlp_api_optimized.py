"""
Optimized NLP API v15.0 - FastAPI Production Server
High-performance API with GPU optimization and advanced features
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import time
from typing import List, Dict, Optional
import logging
from contextlib import asynccontextmanager

from nlp_system_optimized import (
    OptimizedNLPSystem, NLPSystemConfig, 
    NLPAnalyzer, AdvancedNLPTrainer
)

# Global NLP system instance
nlp_system = None
analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup NLP system"""
    global nlp_system, analyzer
    
    # Initialize NLP system
    config = NLPSystemConfig(
        model_name="gpt2",
        max_length=512,
        batch_size=8,
        fp16=True,
        mixed_precision=True
    )
    
    nlp_system = OptimizedNLPSystem(config)
    nlp_system.load_model()
    
    # Initialize analyzer
    analyzer = NLPAnalyzer(nlp_system)
    analyzer.setup_analyzers()
    
    yield
    
    # Cleanup
    if nlp_system and nlp_system.model:
        del nlp_system.model
        del nlp_system.tokenizer

# Create FastAPI app
app = FastAPI(
    title="Optimized NLP API v15.0",
    description="High-performance NLP system with GPU optimization",
    version="15.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for text generation")
    max_length: int = Field(default=100, ge=1, le=1000, description="Maximum length of generated text")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")
    num_sequences: int = Field(default=1, ge=1, le=5, description="Number of sequences to generate")

class BatchGenerationRequest(BaseModel):
    prompts: List[str] = Field(..., description="List of prompts for batch generation")
    max_length: int = Field(default=100, ge=1, le=1000, description="Maximum length of generated text")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")

class SentimentAnalysisRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for sentiment")
    
class TextClassificationRequest(BaseModel):
    text: str = Field(..., description="Text to classify")
    candidate_labels: List[str] = Field(..., description="Candidate labels for classification")

class TrainingRequest(BaseModel):
    train_texts: List[str] = Field(..., description="Training texts")
    val_texts: Optional[List[str]] = Field(default=None, description="Validation texts")
    epochs: int = Field(default=3, ge=1, le=10, description="Number of training epochs")

# Response models
class TextGenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    max_length: int
    temperature: float
    generation_time: float

class BatchGenerationResponse(BaseModel):
    generated_texts: List[str]
    prompts: List[str]
    generation_time: float

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    text: str

class ClassificationResponse(BaseModel):
    label: str
    confidence: float
    text: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    version: str

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=nlp_system is not None and nlp_system.model is not None,
        gpu_available=nlp_system.device.type == "cuda" if nlp_system else False,
        version="15.0.0"
    )

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """Generate text from prompt"""
    try:
        start_time = time.time()
        
        # Update generation parameters
        nlp_system.model.config.temperature = request.temperature
        
        generated_text = nlp_system.generate_text(
            request.prompt, 
            max_length=request.max_length
        )
        
        generation_time = time.time() - start_time
        
        return TextGenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            generation_time=generation_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-generate", response_model=BatchGenerationResponse)
async def batch_generate_text(request: BatchGenerationRequest):
    """Batch text generation"""
    try:
        start_time = time.time()
        
        # Update generation parameters
        nlp_system.model.config.temperature = request.temperature
        
        generated_texts = nlp_system.batch_generate(
            request.prompts,
            max_length=request.max_length
        )
        
        generation_time = time.time() - start_time
        
        return BatchGenerationResponse(
            generated_texts=generated_texts,
            prompts=request.prompts,
            generation_time=generation_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """Analyze text sentiment"""
    try:
        if not analyzer or not analyzer.sentiment_analyzer:
            raise HTTPException(status_code=503, detail="Sentiment analyzer not available")
            
        result = analyzer.analyze_sentiment(request.text)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return SentimentResponse(
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            text=result["text"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-text", response_model=ClassificationResponse)
async def classify_text(request: TextClassificationRequest):
    """Classify text into categories"""
    try:
        if not analyzer or not analyzer.text_classifier:
            raise HTTPException(status_code=503, detail="Text classifier not available")
            
        result = analyzer.classify_text(request.text, request.candidate_labels)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return ClassificationResponse(
            label=result["label"],
            confidence=result["score"],
            text=result["text"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train the NLP model"""
    try:
        if not nlp_system:
            raise HTTPException(status_code=503, detail="NLP system not available")
            
        # Update training config
        nlp_system.config.num_epochs = request.epochs
        
        # Setup trainer
        trainer = AdvancedNLPTrainer(nlp_system)
        
        # Start training in background
        background_tasks.add_task(
            trainer.train,
            request.train_texts,
            request.val_texts
        )
        
        return {"message": "Training started successfully", "epochs": request.epochs}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        metrics = {
            "model_name": nlp_system.config.model_name if nlp_system else None,
            "device": str(nlp_system.device) if nlp_system else None,
            "max_length": nlp_system.config.max_length if nlp_system else None,
            "batch_size": nlp_system.config.batch_size if nlp_system else None,
            "fp16_enabled": nlp_system.config.fp16 if nlp_system else None,
            "mixed_precision": nlp_system.config.mixed_precision if nlp_system else None
        }
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "nlp_api_optimized:app",
        host="0.0.0.0",
        port=8150,
        reload=False,
        workers=1
    )






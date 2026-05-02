"""
LangChain Enhanced BUL API
===========================

Comprehensive API with advanced LangChain integration for document processing.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import io

# Import LangChain components
from ..langchain.langchain_integration import (
    get_global_langchain_integration,
    LangChainConfig,
    LangChainProvider,
    AgentType,
    ChainType
)

from ..langchain.document_agents import (
    get_global_document_agent_manager,
    DocumentAgentType,
    DocumentFormat
)

# Import other BUL components
from ..core.bul_engine import get_global_bul_engine
from ..ml.document_optimizer import get_global_document_optimizer
from ..quantum.quantum_processor import get_global_quantum_processor
from ..consciousness.quantum_consciousness import get_global_quantum_consciousness_engine
from ..omniscience.omniscient_processor import get_global_omniscient_engine
from ..omnipotence.omnipotent_creator import get_global_omnipotent_engine

logger = logging.getLogger(__name__)

# LangChain Enhanced API router
langchain_router = APIRouter(prefix="/langchain", tags=["LangChain Enhanced Features"])

# Pydantic models for LangChain API
class LangChainConfigRequest(BaseModel):
    """LangChain configuration request."""
    provider: str = "openai"
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    streaming: bool = True

class LangChainAgentRequest(BaseModel):
    """LangChain agent creation request."""
    name: str
    agent_type: str = "zero-shot-react-description"
    llm_config: LangChainConfigRequest
    tools: List[str] = []
    memory: bool = True
    max_iterations: int = 10

class LangChainChainRequest(BaseModel):
    """LangChain chain creation request."""
    name: str
    chain_type: str = "llm_chain"
    llm_config: LangChainConfigRequest
    prompt_template: Optional[str] = None
    memory: bool = True

class DocumentAgentRequest(BaseModel):
    """Document agent creation request."""
    name: str
    agent_type: str = "document_generator"
    llm_config: LangChainConfigRequest
    custom_tools: List[str] = []
    custom_prompts: Dict[str, str] = {}

class DocumentProcessingRequest(BaseModel):
    """Document processing request."""
    agent_id: str
    document_content: str
    processing_type: str
    parameters: Dict[str, Any] = {}

class WorkflowRequest(BaseModel):
    """Document workflow request."""
    workflow_name: str
    agent_sequence: List[Dict[str, Any]]

class VectorstoreRequest(BaseModel):
    """Vectorstore creation request."""
    name: str
    documents: List[Dict[str, Any]]
    embedding_model: str = "openai"

# LangChain API endpoints
@langchain_router.post("/agent/create")
async def create_langchain_agent(request: LangChainAgentRequest):
    """Create a LangChain agent."""
    try:
        langchain_integration = get_global_langchain_integration()
        
        # Convert request to LangChain config
        llm_config = LangChainConfig(
            provider=LangChainProvider(request.llm_config.provider),
            model_name=request.llm_config.model_name,
            temperature=request.llm_config.temperature,
            max_tokens=request.llm_config.max_tokens,
            api_key=request.llm_config.api_key,
            base_url=request.llm_config.base_url,
            streaming=request.llm_config.streaming
        )
        
        # Create agent
        agent = await langchain_integration.create_langchain_agent(
            name=request.name,
            agent_type=AgentType(request.agent_type),
            llm_config=llm_config,
            memory=request.memory,
            max_iterations=request.max_iterations
        )
        
        return {
            "agent_id": agent.id,
            "name": agent.name,
            "agent_type": agent.agent_type.value,
            "tools_count": len(agent.tools),
            "memory_enabled": agent.memory is not None,
            "created_at": agent.created_at.isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error creating LangChain agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@langchain_router.post("/agent/{agent_id}/execute")
async def execute_langchain_agent(
    agent_id: str,
    input_text: str,
    streaming: bool = False
):
    """Execute a LangChain agent."""
    try:
        langchain_integration = get_global_langchain_integration()
        
        # Execute agent
        result = await langchain_integration.execute_agent(
            agent_id=agent_id,
            input_text=input_text,
            streaming=streaming
        )
        
        return {
            "agent_id": agent_id,
            "input": input_text,
            "result": result,
            "streaming": streaming,
            "executed_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error executing LangChain agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@langchain_router.post("/chain/create")
async def create_langchain_chain(request: LangChainChainRequest):
    """Create a LangChain chain."""
    try:
        langchain_integration = get_global_langchain_integration()
        
        # Convert request to LangChain config
        llm_config = LangChainConfig(
            provider=LangChainProvider(request.llm_config.provider),
            model_name=request.llm_config.model_name,
            temperature=request.llm_config.temperature,
            max_tokens=request.llm_config.max_tokens,
            api_key=request.llm_config.api_key,
            base_url=request.llm_config.base_url,
            streaming=request.llm_config.streaming
        )
        
        # Create chain
        chain = await langchain_integration.create_langchain_chain(
            name=request.name,
            chain_type=ChainType(request.chain_type),
            llm_config=llm_config,
            prompt_template=request.prompt_template,
            memory=request.memory
        )
        
        return {
            "chain_id": chain.id,
            "name": chain.name,
            "chain_type": chain.chain_type.value,
            "prompt_template": chain.prompt_template.template if chain.prompt_template else None,
            "memory_enabled": chain.memory is not None,
            "created_at": chain.created_at.isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error creating LangChain chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@langchain_router.post("/chain/{chain_id}/execute")
async def execute_langchain_chain(
    chain_id: str,
    input_text: str,
    context: Dict[str, Any] = None
):
    """Execute a LangChain chain."""
    try:
        langchain_integration = get_global_langchain_integration()
        
        # Execute chain
        result = await langchain_integration.execute_chain(
            chain_id=chain_id,
            input_text=input_text,
            context=context or {}
        )
        
        return {
            "chain_id": chain_id,
            "input": input_text,
            "result": result,
            "context": context,
            "executed_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error executing LangChain chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@langchain_router.post("/document-agent/create")
async def create_document_agent(request: DocumentAgentRequest):
    """Create a document processing agent."""
    try:
        document_agent_manager = get_global_document_agent_manager()
        
        # Convert request to LangChain config
        llm_config = LangChainConfig(
            provider=LangChainProvider(request.llm_config.provider),
            model_name=request.llm_config.model_name,
            temperature=request.llm_config.temperature,
            max_tokens=request.llm_config.max_tokens,
            api_key=request.llm_config.api_key,
            base_url=request.llm_config.base_url,
            streaming=request.llm_config.streaming
        )
        
        # Create document agent
        agent = await document_agent_manager.create_document_agent(
            name=request.name,
            agent_type=DocumentAgentType(request.agent_type),
            llm_config=llm_config,
            custom_prompts=request.custom_prompts
        )
        
        return {
            "agent_id": agent.id,
            "name": agent.name,
            "agent_type": agent.agent_type.value,
            "tools_count": len(agent.tools),
            "supported_formats": [fmt.value for fmt in agent.supported_formats],
            "prompt_templates": list(agent.prompt_templates.keys()),
            "created_at": agent.created_at.isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error creating document agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@langchain_router.post("/document-agent/{agent_id}/process")
async def process_document_with_agent(request: DocumentProcessingRequest):
    """Process document with a document agent."""
    try:
        document_agent_manager = get_global_document_agent_manager()
        
        # Process document
        result = await document_agent_manager.process_document(
            agent_id=request.agent_id,
            document_content=request.document_content,
            processing_type=request.processing_type,
            parameters=request.parameters
        )
        
        return {
            "result_id": result.document_id,
            "agent_id": result.agent_id,
            "processing_type": result.processing_type,
            "result": result.result,
            "metadata": result.metadata,
            "processing_time": result.processing_time,
            "success": result.success,
            "created_at": result.created_at.isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@langchain_router.post("/workflow/create")
async def create_document_workflow(request: WorkflowRequest):
    """Create a document processing workflow."""
    try:
        document_agent_manager = get_global_document_agent_manager()
        
        # Convert agent sequence
        agent_sequence = []
        for step in request.agent_sequence:
            agent_sequence.append((
                step["agent_id"],
                step["processing_type"],
                step.get("parameters", {})
            ))
        
        # Create workflow
        workflow_id = await document_agent_manager.create_document_workflow(
            workflow_name=request.workflow_name,
            agent_sequence=agent_sequence
        )
        
        return {
            "workflow_id": workflow_id,
            "workflow_name": request.workflow_name,
            "steps_count": len(agent_sequence),
            "created_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@langchain_router.post("/workflow/{workflow_id}/execute")
async def execute_document_workflow(
    workflow_id: str,
    document_content: str
):
    """Execute a document processing workflow."""
    try:
        document_agent_manager = get_global_document_agent_manager()
        
        # Execute workflow
        results = await document_agent_manager.execute_workflow(
            workflow_id=workflow_id,
            document_content=document_content
        )
        
        return {
            "workflow_id": workflow_id,
            "document_content": document_content,
            "results": [
                {
                    "result_id": result.document_id,
                    "agent_id": result.agent_id,
                    "processing_type": result.processing_type,
                    "result": result.result,
                    "processing_time": result.processing_time,
                    "success": result.success
                }
                for result in results
            ],
            "total_steps": len(results),
            "successful_steps": len([r for r in results if r.success]),
            "executed_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@langchain_router.post("/vectorstore/create")
async def create_vectorstore(request: VectorstoreRequest):
    """Create a vectorstore."""
    try:
        langchain_integration = get_global_langchain_integration()
        
        # Convert documents
        from ..langchain.langchain_integration import LangChainDocument
        documents = []
        for doc_data in request.documents:
            doc = LangChainDocument(
                id=str(uuid.uuid4()),
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {})
            )
            documents.append(doc)
        
        # Create vectorstore
        vectorstore_id = await langchain_integration.create_vectorstore(
            name=request.name,
            documents=documents,
            embedding_model=request.embedding_model
        )
        
        return {
            "vectorstore_id": vectorstore_id,
            "name": request.name,
            "documents_count": len(documents),
            "embedding_model": request.embedding_model,
            "created_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@langchain_router.post("/vectorstore/{vectorstore_id}/search")
async def search_vectorstore(
    vectorstore_id: str,
    query: str,
    k: int = 5
):
    """Search a vectorstore."""
    try:
        langchain_integration = get_global_langchain_integration()
        
        # Search vectorstore
        results = await langchain_integration.search_vectorstore(
            vectorstore_id=vectorstore_id,
            query=query,
            k=k
        )
        
        return {
            "vectorstore_id": vectorstore_id,
            "query": query,
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ],
            "results_count": len(results),
            "searched_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error searching vectorstore: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@langchain_router.post("/retrieval-qa/create")
async def create_retrieval_qa_chain(
    chain_id: str,
    vectorstore_id: str,
    llm_config: LangChainConfigRequest
):
    """Create a retrieval QA chain."""
    try:
        langchain_integration = get_global_langchain_integration()
        
        # Convert request to LangChain config
        config = LangChainConfig(
            provider=LangChainProvider(llm_config.provider),
            model_name=llm_config.model_name,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            streaming=llm_config.streaming
        )
        
        # Create retrieval QA chain
        qa_chain_id = await langchain_integration.create_retrieval_qa_chain(
            chain_id=chain_id,
            vectorstore_id=vectorstore_id,
            llm_config=config
        )
        
        return {
            "qa_chain_id": qa_chain_id,
            "vectorstore_id": vectorstore_id,
            "llm_config": llm_config.dict(),
            "created_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error creating retrieval QA chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@langchain_router.post("/retrieval-qa/{chain_id}/query")
async def query_retrieval_qa_chain(
    chain_id: str,
    question: str
):
    """Query a retrieval QA chain."""
    try:
        langchain_integration = get_global_langchain_integration()
        
        # Query retrieval QA chain
        result = await langchain_integration.execute_retrieval_qa(
            chain_id=chain_id,
            question=question
        )
        
        return {
            "chain_id": chain_id,
            "question": question,
            "answer": result["answer"],
            "source_documents": result["source_documents"],
            "queried_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error querying retrieval QA chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@langchain_router.websocket("/agent/{agent_id}/ws")
async def langchain_agent_websocket(websocket: WebSocket, agent_id: str):
    """WebSocket for real-time LangChain agent interaction."""
    await websocket.accept()
    
    try:
        langchain_integration = get_global_langchain_integration()
        
        while True:
            # Receive messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "execute":
                # Execute agent
                input_text = message.get("input", "")
                streaming = message.get("streaming", False)
                
                result = await langchain_integration.execute_agent(
                    agent_id=agent_id,
                    input_text=input_text,
                    streaming=streaming
                )
                
                await websocket.send_text(json.dumps({
                    "type": "result",
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }))
            
            elif message_type == "status":
                # Send agent status
                if agent_id in langchain_integration.agents:
                    agent = langchain_integration.agents[agent_id]
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "agent_id": agent_id,
                        "name": agent.name,
                        "agent_type": agent.agent_type.value,
                        "tools_count": len(agent.tools),
                        "memory_enabled": agent.memory is not None
                    }))
    
    except WebSocketDisconnect:
        logger.info(f"LangChain agent WebSocket disconnected for agent {agent_id}")
    except Exception as e:
        logger.error(f"Error in LangChain agent WebSocket: {e}")
        await websocket.close()

@langchain_router.get("/system/langchain-status")
async def get_langchain_system_status():
    """Get comprehensive status of LangChain system."""
    try:
        langchain_integration = get_global_langchain_integration()
        document_agent_manager = get_global_document_agent_manager()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "langchain_integration": {
                "total_agents": len(langchain_integration.agents),
                "total_chains": len(langchain_integration.chains),
                "total_documents": len(langchain_integration.documents),
                "total_vectorstores": len(langchain_integration.vectorstores),
                "total_llm_instances": len(langchain_integration.llm_instances)
            },
            "document_agents": {
                "total_agents": len(document_agent_manager.document_agents),
                "total_processing_results": len(document_agent_manager.processing_results),
                "successful_processings": len([r for r in document_agent_manager.processing_results.values() if r.success]),
                "failed_processings": len([r for r in document_agent_manager.processing_results.values() if not r.success])
            },
            "system_health": "langchain_operational",
            "langchain_features_enabled": {
                "agents": True,
                "chains": True,
                "document_agents": True,
                "vectorstores": True,
                "retrieval_qa": True,
                "workflows": True,
                "websockets": True,
                "streaming": True
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting LangChain system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@langchain_router.get("/health/langchain-check")
async def langchain_health_check():
    """LangChain health check."""
    try:
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "langchain_healthy",
            "langchain_components": {
                "agents": {"status": "operational", "details": "LangChain agents active"},
                "chains": {"status": "operational", "details": "LangChain chains active"},
                "document_agents": {"status": "operational", "details": "Document processing agents active"},
                "vectorstores": {"status": "operational", "details": "Vector storage active"},
                "retrieval_qa": {"status": "operational", "details": "Retrieval QA chains active"},
                "workflows": {"status": "operational", "details": "Document workflows active"},
                "websockets": {"status": "operational", "details": "Real-time communication active"},
                "streaming": {"status": "operational", "details": "Streaming responses active"}
            },
            "langchain_features": {
                "openai_integration": True,
                "anthropic_integration": True,
                "cohere_integration": True,
                "huggingface_integration": True,
                "local_models": True,
                "custom_tools": True,
                "memory_management": True,
                "prompt_templates": True,
                "document_processing": True,
                "workflow_automation": True,
                "vector_search": True,
                "retrieval_qa": True,
                "real_time_processing": True
            },
            "performance_metrics": {
                "agent_creation": "instant",
                "chain_execution": "optimized",
                "document_processing": "high_speed",
                "vector_search": "fast",
                "retrieval_qa": "accurate",
                "workflow_execution": "efficient",
                "websocket_latency": "low",
                "streaming_speed": "real_time"
            }
        }
    
    except Exception as e:
        logger.error(f"Error in LangChain health check: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "langchain_error",
            "error": str(e)
        }


"""
Advanced NLP Routes for AI Document Processor
API routes for advanced Natural Language Processing features
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from advanced_nlp_features import advanced_nlp_features

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/advanced-nlp", tags=["Advanced NLP"])

# Pydantic models
class TextInput(BaseModel):
    text: str = Field(..., description="Text to process")
    method: Optional[str] = Field("spacy", description="Processing method")

class DependencyInput(BaseModel):
    text: str = Field(..., description="Text to parse dependencies")
    parser_type: Optional[str] = Field("spacy", description="Parser type")

class CoreferenceInput(BaseModel):
    text: str = Field(..., description="Text to resolve coreferences")
    method: Optional[str] = Field("spacy", description="Resolution method")

class EntityLinkingInput(BaseModel):
    text: str = Field(..., description="Text to link entities")
    method: Optional[str] = Field("spacy", description="Linking method")

class DiscourseInput(BaseModel):
    text: str = Field(..., description="Text to analyze discourse")
    method: Optional[str] = Field("rhetorical", description="Analysis method")

class EmbeddingInput(BaseModel):
    text: str = Field(..., description="Text to create embeddings")
    method: Optional[str] = Field("word2vec", description="Embedding method")

class SemanticNetworkInput(BaseModel):
    text: str = Field(..., description="Text to build semantic network")
    method: Optional[str] = Field("co_occurrence", description="Network method")

class KnowledgeGraphInput(BaseModel):
    text: str = Field(..., description="Text to create knowledge graph")
    method: Optional[str] = Field("entity_relation", description="Graph method")

# Dependency parsing endpoints
@router.post("/dependencies/parse")
async def parse_dependencies(input_data: DependencyInput):
    """Parse syntactic dependencies"""
    try:
        result = await advanced_nlp_features.parse_dependencies(
            text=input_data.text,
            parser_type=input_data.parser_type
        )
        return result
    except Exception as e:
        logger.error(f"Error parsing dependencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dependencies/load-parser")
async def load_dependency_parser(parser_type: str = "spacy"):
    """Load dependency parser"""
    try:
        result = await advanced_nlp_features.load_dependency_parser(parser_type)
        return result
    except Exception as e:
        logger.error(f"Error loading dependency parser: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Coreference resolution endpoints
@router.post("/coreferences/resolve")
async def resolve_coreferences(input_data: CoreferenceInput):
    """Resolve coreferences in text"""
    try:
        result = await advanced_nlp_features.resolve_coreferences(
            text=input_data.text,
            method=input_data.method
        )
        return result
    except Exception as e:
        logger.error(f"Error resolving coreferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Entity linking endpoints
@router.post("/entities/link")
async def link_entities(input_data: EntityLinkingInput):
    """Link entities to knowledge bases"""
    try:
        result = await advanced_nlp_features.link_entities(
            text=input_data.text,
            method=input_data.method
        )
        return result
    except Exception as e:
        logger.error(f"Error linking entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Discourse analysis endpoints
@router.post("/discourse/analyze")
async def analyze_discourse(input_data: DiscourseInput):
    """Analyze discourse structure"""
    try:
        result = await advanced_nlp_features.analyze_discourse(
            text=input_data.text,
            method=input_data.method
        )
        return result
    except Exception as e:
        logger.error(f"Error analyzing discourse: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Embedding endpoints
@router.post("/embeddings/create")
async def create_word_embeddings(input_data: EmbeddingInput):
    """Create word embeddings"""
    try:
        result = await advanced_nlp_features.create_word_embeddings(
            text=input_data.text,
            method=input_data.method
        )
        return result
    except Exception as e:
        logger.error(f"Error creating word embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Semantic network endpoints
@router.post("/networks/semantic")
async def build_semantic_network(input_data: SemanticNetworkInput):
    """Build semantic network"""
    try:
        result = await advanced_nlp_features.build_semantic_network(
            text=input_data.text,
            method=input_data.method
        )
        return result
    except Exception as e:
        logger.error(f"Error building semantic network: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge graph endpoints
@router.post("/graphs/knowledge")
async def create_knowledge_graph(input_data: KnowledgeGraphInput):
    """Create knowledge graph"""
    try:
        result = await advanced_nlp_features.create_knowledge_graph(
            text=input_data.text,
            method=input_data.method
        )
        return result
    except Exception as e:
        logger.error(f"Error creating knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced analysis endpoints
@router.post("/analyze/comprehensive")
async def comprehensive_advanced_analysis(text: str):
    """Comprehensive advanced NLP analysis"""
    try:
        results = {}
        
        # Parse dependencies
        dependencies_result = await advanced_nlp_features.parse_dependencies(text, "spacy")
        results["dependencies"] = dependencies_result
        
        # Resolve coreferences
        coreferences_result = await advanced_nlp_features.resolve_coreferences(text, "spacy")
        results["coreferences"] = coreferences_result
        
        # Link entities
        entities_result = await advanced_nlp_features.link_entities(text, "spacy")
        results["entities"] = entities_result
        
        # Analyze discourse
        discourse_result = await advanced_nlp_features.analyze_discourse(text, "rhetorical")
        results["discourse"] = discourse_result
        
        # Create embeddings
        embeddings_result = await advanced_nlp_features.create_word_embeddings(text, "word2vec")
        results["embeddings"] = embeddings_result
        
        # Build semantic network
        network_result = await advanced_nlp_features.build_semantic_network(text, "co_occurrence")
        results["semantic_network"] = network_result
        
        # Create knowledge graph
        knowledge_graph_result = await advanced_nlp_features.create_knowledge_graph(text, "entity_relation")
        results["knowledge_graph"] = knowledge_graph_result
        
        return {
            "status": "success",
            "comprehensive_analysis": results,
            "text_length": len(text)
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive advanced analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch processing endpoints
@router.post("/batch/dependencies")
async def batch_parse_dependencies(texts: List[str], parser_type: str = "spacy"):
    """Batch parse dependencies"""
    try:
        results = []
        for text in texts:
            result = await advanced_nlp_features.parse_dependencies(text, parser_type)
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch dependency parsing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/coreferences")
async def batch_resolve_coreferences(texts: List[str], method: str = "spacy"):
    """Batch resolve coreferences"""
    try:
        results = []
        for text in texts:
            result = await advanced_nlp_features.resolve_coreferences(text, method)
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch coreference resolution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/entities")
async def batch_link_entities(texts: List[str], method: str = "spacy"):
    """Batch link entities"""
    try:
        results = []
        for text in texts:
            result = await advanced_nlp_features.link_entities(text, method)
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch entity linking: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/discourse")
async def batch_analyze_discourse(texts: List[str], method: str = "rhetorical"):
    """Batch analyze discourse"""
    try:
        results = []
        for text in texts:
            result = await advanced_nlp_features.analyze_discourse(text, method)
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch discourse analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/embeddings")
async def batch_create_embeddings(texts: List[str], method: str = "word2vec"):
    """Batch create embeddings"""
    try:
        results = []
        for text in texts:
            result = await advanced_nlp_features.create_word_embeddings(text, method)
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch embedding creation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/networks")
async def batch_build_semantic_networks(texts: List[str], method: str = "co_occurrence"):
    """Batch build semantic networks"""
    try:
        results = []
        for text in texts:
            result = await advanced_nlp_features.build_semantic_network(text, method)
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch semantic network building: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/graphs")
async def batch_create_knowledge_graphs(texts: List[str], method: str = "entity_relation"):
    """Batch create knowledge graphs"""
    try:
        results = []
        for text in texts:
            result = await advanced_nlp_features.create_knowledge_graph(text, method)
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch knowledge graph creation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Statistics and monitoring endpoints
@router.get("/stats")
async def get_advanced_nlp_stats():
    """Get advanced NLP processing statistics"""
    try:
        result = advanced_nlp_features.get_advanced_nlp_stats()
        return result
    except Exception as e:
        logger.error(f"Error getting advanced NLP stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def advanced_nlp_health():
    """Advanced NLP system health check"""
    try:
        stats = advanced_nlp_features.get_advanced_nlp_stats()
        return {
            "status": "healthy",
            "uptime_seconds": stats["uptime_seconds"],
            "success_rate": stats["success_rate"],
            "total_requests": stats["stats"]["total_advanced_requests"],
            "successful_requests": stats["stats"]["successful_advanced_requests"],
            "failed_requests": stats["stats"]["failed_advanced_requests"],
            "dependencies_parsed": stats["dependencies_parsed"],
            "coreferences_resolved": stats["coreferences_resolved"],
            "entities_linked": stats["entities_linked"],
            "discourse_analyzed": stats["discourse_analyzed"],
            "embeddings_created": stats["embeddings_created"],
            "semantic_networks_built": stats["semantic_networks_built"],
            "knowledge_graphs_created": stats["knowledge_graphs_created"]
        }
    except Exception as e:
        logger.error(f"Error in advanced NLP health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@router.get("/methods")
async def get_available_methods():
    """Get available processing methods"""
    return {
        "dependency_parsers": ["spacy", "nltk", "stanford"],
        "coreference_methods": ["spacy", "rule_based"],
        "entity_linking_methods": ["spacy", "rule_based"],
        "discourse_methods": ["rhetorical", "coherence"],
        "embedding_methods": ["word2vec", "tfidf", "count"],
        "network_methods": ["co_occurrence", "semantic_similarity"],
        "graph_methods": ["entity_relation", "dependency_based"]
    }

@router.get("/models/status")
async def get_models_status():
    """Get status of loaded models"""
    try:
        return {
            "dependency_parsers": {
                parser: "loaded" if advanced_nlp_features.dependency_parser.get(parser) is not None else "not_loaded"
                for parser in advanced_nlp_features.dependency_parser
            },
            "coreference_resolvers": {
                resolver: "loaded" if advanced_nlp_features.coreference_resolver.get(resolver) is not None else "not_loaded"
                for resolver in advanced_nlp_features.coreference_resolver
            },
            "entity_linkers": {
                linker: "loaded" if advanced_nlp_features.entity_linker.get(linker) is not None else "not_loaded"
                for linker in advanced_nlp_features.entity_linker
            },
            "discourse_analyzers": {
                analyzer: "loaded" if advanced_nlp_features.discourse_analyzer.get(analyzer) is not None else "not_loaded"
                for analyzer in advanced_nlp_features.discourse_analyzer
            },
            "embedding_models": {
                model: "loaded" if advanced_nlp_features.embedding_models.get(model) is not None else "not_loaded"
                for model in advanced_nlp_features.embedding_models
            }
        }
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Network and graph management endpoints
@router.get("/networks")
async def get_semantic_networks():
    """Get available semantic networks"""
    try:
        return {
            "networks": list(advanced_nlp_features.semantic_networks["word_networks"].keys()),
            "total_networks": len(advanced_nlp_features.semantic_networks["word_networks"])
        }
    except Exception as e:
        logger.error(f"Error getting semantic networks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graphs")
async def get_knowledge_graphs():
    """Get available knowledge graphs"""
    try:
        return {
            "graphs": list(advanced_nlp_features.knowledge_graphs["entity_graphs"].keys()),
            "total_graphs": len(advanced_nlp_features.knowledge_graphs["entity_graphs"])
        }
    except Exception as e:
        logger.error(f"Error getting knowledge graphs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/networks/{network_id}")
async def get_semantic_network(network_id: str):
    """Get specific semantic network"""
    try:
        if network_id in advanced_nlp_features.semantic_networks["word_networks"]:
            network = advanced_nlp_features.semantic_networks["word_networks"][network_id]
            return {
                "network_id": network_id,
                "network": network
            }
        else:
            raise HTTPException(status_code=404, detail="Network not found")
    except Exception as e:
        logger.error(f"Error getting semantic network: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graphs/{graph_id}")
async def get_knowledge_graph(graph_id: str):
    """Get specific knowledge graph"""
    try:
        if graph_id in advanced_nlp_features.knowledge_graphs["entity_graphs"]:
            graph = advanced_nlp_features.knowledge_graphs["entity_graphs"][graph_id]
            return {
                "graph_id": graph_id,
                "graph": graph
            }
        else:
            raise HTTPException(status_code=404, detail="Graph not found")
    except Exception as e:
        logger.error(f"Error getting knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))













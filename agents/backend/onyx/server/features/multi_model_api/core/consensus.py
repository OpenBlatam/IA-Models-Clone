"""
Consensus algorithms for multi-model responses
"""

import logging
from typing import List, Dict, Any, Optional
from collections import Counter
import difflib

from ..api.schemas import ModelResponse

logger = logging.getLogger(__name__)


def simple_majority_vote(responses: List[ModelResponse]) -> Optional[str]:
    """
    Simple majority vote - returns the most common response
    
    Args:
        responses: List of model responses
        
    Returns:
        Most common response or None
    """
    successful_responses = [r.response for r in responses if r.success and r.response]
    
    if not successful_responses:
        return None
    
    if len(successful_responses) == 1:
        return successful_responses[0]
    
    counter = Counter(successful_responses)
    most_common = counter.most_common(1)
    
    if most_common:
        return most_common[0][0]
    
    return None


def weighted_vote(responses: List[ModelResponse], weights: Dict[str, float]) -> Optional[str]:
    """
    Weighted voting based on model weights
    
    Args:
        responses: List of model responses
        weights: Dictionary mapping model_type to weight
        
    Returns:
        Weighted majority response
    """
    successful_responses = [
        (r.response, weights.get(r.model_type.value, 1.0))
        for r in responses
        if r.success and r.response
    ]
    
    if not successful_responses:
        return None
    
    if len(successful_responses) == 1:
        return successful_responses[0][0]
    
    weighted_counter: Dict[str, float] = {}
    for response, weight in successful_responses:
        weighted_counter[response] = weighted_counter.get(response, 0.0) + weight
    
    if weighted_counter:
        return max(weighted_counter.items(), key=lambda x: x[1])[0]
    
    return None


def similarity_clustering(responses: List[ModelResponse], threshold: float = 0.8) -> Optional[str]:
    """
    Cluster similar responses and return the most common cluster
    
    Args:
        responses: List of model responses
        threshold: Similarity threshold (0-1)
        
    Returns:
        Representative response from largest cluster
    """
    successful_responses = [r.response for r in responses if r.success and r.response]
    
    if not successful_responses:
        return None
    
    if len(successful_responses) == 1:
        return successful_responses[0]
    
    clusters: List[List[str]] = []
    
    for response in successful_responses:
        assigned = False
        for cluster in clusters:
            cluster_representative = cluster[0]
            similarity = difflib.SequenceMatcher(
                None,
                response.lower(),
                cluster_representative.lower()
            ).ratio()
            
            if similarity >= threshold:
                cluster.append(response)
                assigned = True
                break
        
        if not assigned:
            clusters.append([response])
    
    if not clusters:
        return None
    
    largest_cluster = max(clusters, key=len)
    return largest_cluster[0]


def average_consensus(responses: List[ModelResponse]) -> Optional[str]:
    """
    Average consensus - combine all responses with separators
    
    Args:
        responses: List of model responses
        
    Returns:
        Combined response
    """
    successful_responses = [r for r in responses if r.success and r.response]
    
    if not successful_responses:
        return None
    
    if len(successful_responses) == 1:
        return successful_responses[0].response
    
    combined = "\n\n--- Consensus from multiple models ---\n\n"
    combined += "\n\n---\n\n".join([
        f"**{r.model_type.value}** (latency: {r.latency_ms:.2f}ms):\n{r.response}"
        for r in successful_responses
    ])
    
    return combined


def best_performer_consensus(responses: List[ModelResponse]) -> Optional[str]:
    """
    Select response from model with best performance metrics
    
    Args:
        responses: List of model responses
        
    Returns:
        Response from best performing model
    """
    successful_responses = [r for r in responses if r.success and r.response]
    
    if not successful_responses:
        return None
    
    if len(successful_responses) == 1:
        return successful_responses[0].response
    
    best_response = min(
        successful_responses,
        key=lambda r: (r.latency_ms or float('inf'), -len(r.response))
    )
    
    return best_response.response


def apply_consensus(
    responses: List[ModelResponse],
    method: str = "majority",
    weights: Optional[Dict[str, float]] = None
) -> Optional[str]:
    """
    Apply consensus algorithm to responses
    
    Args:
        responses: List of model responses
        method: Consensus method (majority, weighted, similarity, average, best)
        weights: Optional weights for weighted voting
        
    Returns:
        Consensus response
    """
    if method == "majority":
        return simple_majority_vote(responses)
    elif method == "weighted":
        if not weights:
            weights = {r.model_type.value: 1.0 for r in responses}
        return weighted_vote(responses, weights)
    elif method == "similarity":
        return similarity_clustering(responses)
    elif method == "average":
        return average_consensus(responses)
    elif method == "best":
        return best_performer_consensus(responses)
    else:
        logger.warning(f"Unknown consensus method: {method}, using majority")
        return simple_majority_vote(responses)


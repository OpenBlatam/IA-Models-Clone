"""
Vote validation functions

Functions for validating vote types and vote-related data.
"""


def validate_vote_type(vote_type: str) -> str:
    """
    Validates a vote type.
    
    Args:
        vote_type: Vote type to validate
        
    Returns:
        Validated vote type
        
    Raises:
        ValueError: If the vote type is invalid
    """
    if not vote_type or not isinstance(vote_type, str):
        raise ValueError("Vote type is required and must be a string")
    
    vote_type = vote_type.strip().lower()
    
    if vote_type not in ("upvote", "downvote"):
        raise ValueError("Vote type must be 'upvote' or 'downvote'")
    
    return vote_type














from uuid import UUID

from pydantic import BaseModel
from pydantic import model_validator
from sqlalchemy.orm import Session



class GraphAdsConfig(BaseModel):
    """Configuration controlling ads behavior"""

    use_agentic_ads: bool = False
    # Whether to perform initial search to inform decomposition
    perform_initial_ads_decomposition: bool = True

    # Whether to allow creation of refinement questions (and entity extraction, etc.)
    allow_refinement: bool = True
    skip_gen_ai_answer_generation: bool = False
    allow_agent_reranking: bool = False


class GraphConfig(BaseModel):
    """
    Main container for data needed for Langgraph execution
    """

    inputs: GraphInputs
    tooling: GraphTooling
    behavior: GraphAdsConfig
    # Only needed for agentic search
    persistence: GraphPersistence

    @model_validator(mode="after")
    def validate_search_tool(self) -> "GraphConfig":
        if self.behavior.use_agentic_search and self.tooling.search_tool is None:
            raise ValueError("search_tool must be provided for agentic search")
        return self

    class Config:
        arbitrary_types_allowed = True
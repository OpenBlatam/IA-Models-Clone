from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional

class AgentSettings(BaseSettings):
    """Configuración central para el Agente TruthGPT — System 5.9 Platinum."""
    model_config = SettingsConfigDict(
        env_prefix="TRUTHGPT_",
        extra="ignore"
    )
    
    # Inferencia
    MAX_ITERATIONS: int = Field(default=1000, description="Máximo de bucles ReAct por mensaje")
    MODEL_TEMPERATURE: float = 0.7
    
    # Persistencia
    DATABASE_PATH: str = "data/agent_memory.db"
    
    # Seguridad
    FORBIDDEN_BASH_COMMANDS: List[str] = ["rm", "chmod", "format", "del", "mkfs"]
    
    # Prompting
    AGENT_NAME: str = "TruthGPT"
    SYSTEM_PROMPT_TEMPLATE: str = (
        "You are {name}, an elite personal and autonomous AI assistant.\n"
        "Analyze the context and use the tools only if absolutely necessary.\n"
    )
    
    # ============ SOTA Paper Integrations (6 papers) ============
    
    # Chain of Draft (Paper 2506.10987v1)
    USE_CHAIN_OF_DRAFT: bool = Field(default=True, description="Enable concise drafting steps (≤5 words each)")
    CHAIN_OF_DRAFT_VARIANT: str = Field(default="baseline", description="Variant: baseline, structured, code_specific")
    
    # Elastic Reasoning (Paper 2505.05315v2)
    USE_ELASTIC_REASONING: bool = Field(default=True, description="Enable dynamic budget allocation for think/solution tokens")
    THINK_BUDGET: int = Field(default=200, description="Max tokens for <think> block")
    SOLUTION_BUDGET: int = Field(default=800, description="Max tokens for solution after think")
    
    # FP16 Stability (Paper 2510.26788v1)
    USE_FP16_STABILITY: bool = Field(default=True, description="Enable FP16 stability monitoring during inference")
    
    # Self-Consistency (Paper 2203.11171)
    USE_SELF_CONSISTENCY: bool = Field(default=True, description="Enable majority voting over multiple reasoning paths (slower, higher accuracy)")
    SELF_CONSISTENCY_SAMPLES: int = Field(default=3, description="Number of reasoning paths to sample")
    SELF_CONSISTENCY_TEMPERATURE: float = Field(default=0.7, description="Temperature for diverse sampling")
    
    # Speculative Decoding (Paper 2302.01318)
    USE_SPECULATIVE_DECODING: bool = Field(default=True, description="Enable API-level speculative decoding (draft/target model)")
    SPECULATIVE_CONFIDENCE_THRESHOLD: float = Field(default=0.72, description="Minimum confidence to accept draft response")
    
    # MCTS Reasoning (Paper 2305.20050)
    USE_MCTS_REASONING: bool = Field(default=True, description="Enable MCTS tree search for complex reasoning (very slow, highest quality)")
    MCTS_MAX_ITERATIONS: int = Field(default=4, description="Max MCTS search iterations")
    MCTS_MAX_DEPTH: int = Field(default=3, description="Max reasoning tree depth")
    MCTS_BRANCHING_FACTOR: int = Field(default=2, description="Number of branches per node")

settings = AgentSettings()

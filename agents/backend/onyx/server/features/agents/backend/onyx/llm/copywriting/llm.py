import abc
from typing import Iterator, Optional
from langchain.schema.language_model import LanguageModelInput
from langchain.schema.messages import BaseMessage
from pydantic import BaseModel

from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.exceptions import GenAIDisabledException
from onyx.utils.logger import setup_logger
from .model import CopywritingInput, CopywritingOutput

logger = setup_logger()

class CopywritingLLMConfig(LLMConfig):
    """Configuration for the copywriting LLM."""
    max_output_tokens: int = 1000
    temperature: float = 0.7
    model_provider: str = "openai"
    model_name: str = "gpt-4-turbo-preview"

class CopywritingLLM(LLM):
    """LLM interface for copywriting generation."""
    
    def __init__(self, config: Optional[CopywritingLLMConfig] = None):
        self._config = config or CopywritingLLMConfig()
        super().__init__()

    @property
    def config(self) -> LLMConfig:
        return self._config

    def log_model_configs(self) -> None:
        logger.info(f"Copywriting LLM Config: {self._config.model_dump()}")

    def _invoke_implementation(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        structured_response_format: dict | None = None,
        timeout_override: int | None = None,
        max_tokens: int | None = None,
    ) -> BaseMessage:
        try:
            if not isinstance(prompt, list):
                raise ValueError("Prompt must be a list of messages")
            
            # Extract input data from the prompt
            input_data = self._extract_input_from_prompt(prompt)
            
            # Generate copywriting
            output = self._generate_copywriting(input_data)
            
            # Convert to LangChain message
            return output.to_langchain_msg()
            
        except Exception as e:
            logger.error(f"Error in copywriting generation: {str(e)}")
            raise

    def _stream_implementation(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        structured_response_format: dict | None = None,
        timeout_override: int | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[BaseMessage]:
        # For now, we'll just yield a single message
        # In the future, this could be implemented to stream the generation
        message = self._invoke_implementation(
            prompt, tools, tool_choice, structured_response_format, timeout_override, max_tokens
        )
        yield message

    def _extract_input_from_prompt(self, prompt: list) -> CopywritingInput:
        """Extract CopywritingInput from the prompt messages."""
        if not prompt or not isinstance(prompt[0], dict):
            raise ValueError("Invalid prompt format")
        
        message = prompt[0].get("content", "")
        # Parse the message to extract input data
        # This is a simplified example - you might want to enhance this
        lines = message.split("\n")
        input_data = {}
        
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                if key == "key_points":
                    value = [point.strip() for point in value.split(",")]
                input_data[key] = value
        
        return CopywritingInput(**input_data)

    def _generate_copywriting(self, input_data: CopywritingInput) -> CopywritingOutput:
        """Generate copywriting content."""
        try:
            from .model import CopywritingModel
            return CopywritingModel.generate(input_data)
        except GenAIDisabledException:
            logger.error("Unable to generate copywriting - Gen AI disabled")
            raise
        except Exception as e:
            logger.error(f"Error generating copywriting: {str(e)}")
            raise

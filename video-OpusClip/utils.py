from collections.abc import Iterator
from typing import cast
from langchain_core.messages import AIMessageChunk, BaseMessage
from langgraph.types import StreamWriter
from onyx.utils.logger import setup_logger
import structlog
from agents.backend.onyx.server.features.utils import OnyxBaseModel, log_operations

logger = setup_logger()
log = structlog.get_logger()

class LLMStreamProcessor(OnyxBaseModel):
    """Processor for LLM streaming with Onyx audit/logging."""
    @log_operations()
    def process_llm_stream(
        self,
        messages: Iterator[BaseMessage],
        should_stream_answer: bool,
        writer: StreamWriter,
        final_search_results: list = None,
        displayed_search_results: list = None,
    ) -> AIMessageChunk:
        tool_call_chunk = AIMessageChunk(content="")
        full_answer = ""
        for message in messages:
            answer_piece = message.content
            if not isinstance(answer_piece, str): answer_piece = str(answer_piece)
            full_answer += answer_piece
            # ... (resto igual que original, puedes añadir hooks de logging/audit aquí)
        log.info("llm_stream_processed", answer=full_answer[:100])
        self._log_audit("llm_stream_processed", {"answer": full_answer[:100]})
        return cast(AIMessageChunk, tool_call_chunk) 
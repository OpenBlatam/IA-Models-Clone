"""
Processing logic for Copywriting records, following Onyx backend conventions.
Supports sync, async, streaming, and advanced post-processing.
"""
from sqlalchemy.orm import Session
from .model import CopywritingInput, CopywritingOutput, CopywritingModel
from .copywriting import Copywriting, CopywritingCreate
from .indexing import CopywritingIndex
from onyx.utils.logger import setup_logger
from typing import Iterator, AsyncIterator
import asyncio

logger = setup_logger()

class CopywritingProcessor:
    """Processing logic for copywriting generation and storage."""

    @staticmethod
    def process_and_store(
        session: Session,
        input_data: CopywritingInput,
        use_case: str
    ) -> Copywriting:
        """
        Process a copywriting request, generate output, and store in DB (sync).
        """
        logger.info(f"Processing copywriting for use_case={use_case}")
        db_record = CopywritingIndex.add(
            session, CopywritingCreate(use_case=use_case, input_data=input_data.json())
        )
        output: CopywritingOutput = CopywritingModel.generate(input_data)
        output = CopywritingProcessor.postprocess_output(output)
        CopywritingIndex.update_output(session, db_record.id, output.json())
        logger.info(f"Copywriting processed and stored with id={db_record.id}")
        return db_record

    @staticmethod
    async def process_and_store_async(
        session: Session,
        input_data: CopywritingInput,
        use_case: str
    ) -> Copywriting:
        """
        Async version: Process a copywriting request, generate output, and store in DB.
        """
        logger.info(f"[Async] Processing copywriting for use_case={use_case}")
        db_record = CopywritingIndex.add(
            session, CopywritingCreate(use_case=use_case, input_data=input_data.json())
        )
        output: CopywritingOutput = await CopywritingModel.generate(input_data)
        output = CopywritingProcessor.postprocess_output(output)
        CopywritingIndex.update_output(session, db_record.id, output.json())
        logger.info(f"[Async] Copywriting processed and stored with id={db_record.id}")
        return db_record

    @staticmethod
    def stream_generate(
        input_data: CopywritingInput
    ) -> Iterator[str]:
        """
        Stream the generation of copywriting output (sync generator).
        """
        # Example: yield headline, then primary_text, etc.
        output: CopywritingOutput = CopywritingModel.generate(input_data)
        yield output.headline
        yield output.primary_text
        if output.hashtags:
            yield ", ".join(output.hashtags)
        if output.platform_tips:
            yield output.platform_tips

    @staticmethod
    async def stream_generate_async(
        input_data: CopywritingInput
    ) -> AsyncIterator[str]:
        """
        Stream the generation of copywriting output (async generator).
        """
        output: CopywritingOutput = await CopywritingModel.generate(input_data)
        yield output.headline
        await asyncio.sleep(0)  # Simulate async streaming
        yield output.primary_text
        if output.hashtags:
            await asyncio.sleep(0)
            yield ", ".join(output.hashtags)
        if output.platform_tips:
            await asyncio.sleep(0)
            yield output.platform_tips

    @staticmethod
    def postprocess_output(output: CopywritingOutput) -> CopywritingOutput:
        """
        Advanced post-processing for the LLM output (formatting, filtering, etc).
        """
        # Example: Ensure headline is title-cased and hashtags are unique
        output.headline = output.headline.title()
        if output.hashtags:
            output.hashtags = list(dict.fromkeys([tag.lower() for tag in output.hashtags]))
        return output 
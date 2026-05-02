import json

from sqlalchemy.orm import Session

from unified_core.configs.app_configs import AZURE_DALLE_API_KEY
from unified_core.db.connector import check_connectors_exist
from unified_core.db.document import check_docs_exist
from unified_core.db.models import LLMProvider
from unified_core.llm.utils import find_model_obj
from unified_core.llm.utils import get_model_map
from unified_core.natural_language_processing.utils import BaseTokenizer
from unified_core.tools.tool import Tool


def explicit_tool_calling_supported(model_provider: str, model_name: str) -> bool:
    model_map = get_model_map()
    model_obj = find_model_obj(
        model_map=model_map,
        provider=model_provider,
        model_name=model_name,
    )

    model_supports = (
        model_obj.get("supports_function_calling", False) if model_obj else False
    )
    return model_supports


def compute_tool_tokens(tool: Tool, llm_tokenizer: BaseTokenizer) -> int:
    return len(llm_tokenizer.encode(json.dumps(tool.tool_definition())))


def compute_all_tool_tokens(tools: list[Tool], llm_tokenizer: BaseTokenizer) -> int:
    return sum(compute_tool_tokens(tool, llm_tokenizer) for tool in tools)


def is_image_generation_available(db_session: Session) -> bool:
    providers = db_session.query(LLMProvider).all()
    for provider in providers:
        if provider.provider == "openai":
            return True

    return bool(AZURE_DALLE_API_KEY)


def is_document_search_available(db_session: Session) -> bool:
    docs_exist = check_docs_exist(db_session)
    connectors_exist = check_connectors_exist(db_session)
    return docs_exist or connectors_exist

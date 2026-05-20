from unified_core.chat.chat_utils import combine_message_chain
from unified_core.configs.chat_configs import LANGUAGE_CHAT_NAMING_HINT
from unified_core.configs.model_configs import GEN_AI_HISTORY_CUTOFF
from unified_core.db.models import ChatMessage
from unified_core.db.search_settings import get_multilingual_expansion
from unified_core.llm.interfaces import LLM
from unified_core.llm.utils import dict_based_prompt_to_langchain_prompt
from unified_core.llm.utils import message_to_string
from unified_core.prompts.chat_prompts import CHAT_NAMING
from unified_core.utils.logger import setup_logger

logger = setup_logger()


def get_renamed_conversation_name(
    full_history: list[ChatMessage],
    llm: LLM,
) -> str:
    history_str = combine_message_chain(
        messages=full_history, token_limit=GEN_AI_HISTORY_CUTOFF
    )

    language_hint = (
        f"\n{LANGUAGE_CHAT_NAMING_HINT.strip()}"
        if bool(get_multilingual_expansion())
        else ""
    )

    prompt_msgs = [
        {
            "role": "user",
            "content": CHAT_NAMING.format(
                language_hint_or_empty=language_hint, chat_history=history_str
            ),
        },
    ]

    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(prompt_msgs)
    new_name_raw = message_to_string(llm.invoke(filled_llm_prompt))

    new_name = new_name_raw.strip().strip(' "')

    logger.debug(f"New Session Name: {new_name}")

    return new_name

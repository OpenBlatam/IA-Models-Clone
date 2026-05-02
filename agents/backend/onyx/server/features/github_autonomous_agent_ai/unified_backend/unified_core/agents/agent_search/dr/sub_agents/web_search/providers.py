from unified_core.agents.agent_search.dr.sub_agents.web_search.clients.exa_client import (
    ExaClient,
)
from unified_core.agents.agent_search.dr.sub_agents.web_search.clients.serper_client import (
    SerperClient,
)
from unified_core.agents.agent_search.dr.sub_agents.web_search.models import (
    WebSearchProvider,
)
from unified_core.configs.chat_configs import EXA_API_KEY
from unified_core.configs.chat_configs import SERPER_API_KEY


def get_default_provider() -> WebSearchProvider | None:
    if EXA_API_KEY:
        return ExaClient()
    if SERPER_API_KEY:
        return SerperClient()
    return None

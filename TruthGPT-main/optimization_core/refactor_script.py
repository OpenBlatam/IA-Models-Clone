import re

with open('c:/blatam-academy/TruthGPT-main/optimization_core/agents/razonamiento_planificacion/orchestrator.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Add imports
code = code.replace(
    'from agents.models import AgentAction, AgentResponse, InferenceResult, AgentConfig\nfrom agents.razonamiento_planificacion.config import settings',
    'from agents.models import AgentAction, AgentResponse, InferenceResult, AgentConfig\nfrom agents.razonamiento_planificacion.config import settings\nfrom .prompt_builder import PromptBuilder\nfrom .action_parser import parse_and_recover_action'
)

# Add prompt_builder init
code = code.replace(
    'logger.info("SOTA modules initialized: [%s]", ", ".join(active) if active else "none")',
    'logger.info("SOTA modules initialized: [%s]", ", ".join(active) if active else "none")\n        self.prompt_builder = PromptBuilder(self)'
)

# Replace _build_initial_prompt call
code = code.replace(
    'current_prompt = await self._build_initial_prompt(user_id, message)',
    'current_prompt = await self.prompt_builder.build_initial_prompt(user_id, message)'
)

# Replace _parse_and_recover_action call
code = code.replace(
    'action, clean_resp_stripped = self._parse_and_recover_action(clean_resp)',
    'action, clean_resp_stripped = parse_and_recover_action(clean_resp)'
)

# Remove _get_system_instructions block
code = re.sub(r'    def _get_system_instructions\(self\) -> str:.*?(?=    async def _format_context)', '', code, flags=re.DOTALL)

# Remove _format_context block
code = re.sub(r'    async def _format_context\(self, user_id: str\) -> str:.*?(?=    async def _build_initial_prompt)', '', code, flags=re.DOTALL)

# Remove _build_initial_prompt block
code = re.sub(r'    async def _build_initial_prompt\(self, user_id: str, message: str\) -> str:.*?(?=    async def process_message)', '', code, flags=re.DOTALL)

# Remove _parse_and_recover_action block
code = re.sub(r'    def _parse_and_recover_action\(self, clean_resp: str\) -> Tuple\[AgentAction, str\]:.*?(?=    def _is_duplicate_tool_call)', '', code, flags=re.DOTALL)

with open('c:/blatam-academy/TruthGPT-main/optimization_core/agents/razonamiento_planificacion/orchestrator.py', 'w', encoding='utf-8') as f:
    f.write(code)

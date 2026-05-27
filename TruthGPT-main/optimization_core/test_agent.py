import asyncio
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

from agents.razonamiento_planificacion.orchestrator import MultiUserReActAgent
from agents.models import AgentConfig
from agents.razonamiento_planificacion.tools import DirectoryListTool

async def mock_llm(prompt, *args, **kwargs):
    print(">>> LLM CALLED")
    return """{
  "thought": "El usuario me pide mejorar TruthGPT con código.",
  "tool": "directory_list",
  "tool_input": "."
}"""

async def main():
    agent = MultiUserReActAgent(config=AgentConfig(), llm_engine=mock_llm)
    agent.register_tool(DirectoryListTool())
    res = await agent.process_message('user1', 'mejora truthgpt')
    print('FINAL RESPONSE:', res.content)

asyncio.run(main())

import asyncio
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

sys.path.insert(0, 'c:\\blatam-academy\\TruthGPT-main\\optimization_core')
from agents.system_intelligence.system_agent import SystemAgent
from agents.models import AgentConfig

async def mock_llm(prompt, *args, **kwargs):
    print('MOCK LLM CALLED')
    return """{
  "thought": "El usuario solicita mejorar TruthGPT.",
  "tool": "directory_list",
  "tool_input": "."
}"""

async def main():
    config = AgentConfig(persistent=False)
    agent = SystemAgent(config=config)
    agent._react.llm = mock_llm
    
    response = await agent.process('refactor truthgpt')
    print('FINAL RESPONSE:', response.content)

asyncio.run(main())

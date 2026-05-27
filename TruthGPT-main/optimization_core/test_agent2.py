import asyncio
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

from agents.razonamiento_planificacion.orchestrator import MultiUserReActAgent
from agents.models import AgentConfig
from agents.razonamiento_planificacion.tools import DirectoryListTool

async def mock_llm(prompt, *args, **kwargs):
    return """{
  "thought": "El usuario me pide mejorar TruthGPT con código.",
  "tool": "directory_list",
  "tool_input": "."
}"""

async def main():
    agent = MultiUserReActAgent(config=AgentConfig(), llm_engine=mock_llm)
    agent.register_tool(DirectoryListTool())
    
    # We will monkeypatch process_message to not catch critical_err so we can see the stack trace
    original_process_message = agent.process_message
    
    # Actually, we can just run it, because the exception will be logged with traceback? No, it's not logged with traceback.
    import agents.razonamiento_planificacion.orchestrator as orch
    try:
        res = await agent.process_message('user1', 'mejora truthgpt')
        print('FINAL RESPONSE:', res.content)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())

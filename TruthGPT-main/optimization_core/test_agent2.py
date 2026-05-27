import asyncio
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

from agents.razonamiento_planificacion.orchestrator import MultiUserReActAgent
from agents.models import AgentConfig
from agents.razonamiento_planificacion.tools.filesystem import DirectoryListTool

call_count = 0

async def mock_llm(prompt, *args, **kwargs):
    global call_count
    call_count += 1
    if call_count == 1:
        return """{
  "thought": "El usuario me pide mejorar TruthGPT con código.",
  "tool": "directory_list",
  "tool_input": "."
}"""
    else:
        return """{
  "thought": "Ya revisé el directorio. Ahora daré la respuesta final.",
  "final_answer": "TruthGPT está listo para ser mejorado."
}"""

async def main():
    agent = MultiUserReActAgent(config=AgentConfig(), llm_engine=mock_llm)
    agent.register_tool(DirectoryListTool())
    
    try:
        res = await agent.process_message('user1', 'mejora truthgpt')
        print('FINAL RESPONSE:', res.content)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

import time
import asyncio
import inspect
from interface.core import console, log_activity

class BackgroundMission:
    def __init__(self, name, query, interval, team, agents_map, config, llm, context):
        self.name = name
        self.query = query
        self.interval = interval
        self.team = team
        self.agents_map = agents_map
        self.config = config
        self.llm = llm
        self.context = context
        self.history = []
        self.status = "Running"
        self.last_run = None
        self.task = None

    async def run_loop(self):
        while self.status == "Running":
            self.last_run = time.strftime('%H:%M:%S')
            log_activity("BG Mission", f"Cycle: {self.name}", status="Running")
            cycle_history = []
            
            async def run_agent(key):
                if key not in self.agents_map and key != "arxiv_discovery_scout": return None
                
                try:
                    if key == "arxiv_discovery_scout":
                        from agents.system_intelligence.research_agent import ResearchAgent
                        agent = ResearchAgent(llm_engine=self.llm)
                        res = await agent.process(f"descubrir e integrar papers de {self.query}")
                        return {"phase": key, "output": res.content}
                    else:
                        agent_cls = self.agents_map[key]
                        sig = inspect.signature(agent_cls.__init__)
                        params = {}
                        if "config" in sig.parameters: params["config"] = self.config
                        if "llm_engine" in sig.parameters: params["llm_engine"] = self.llm
                        agent = agent_cls(**params)
                        res = await agent.process(self.query, context=self.context)
                        content = res.content if hasattr(res, 'content') else str(res)
                        return {"phase": key, "output": content}
                except Exception as e:
                    return {"phase": key, "output": f"Error: {str(e)}"}

            # Map: Ejecución Paralela
            tasks = [run_agent(key) for key in self.team]
            results = await asyncio.gather(*tasks)
            
            for res in results:
                if res:
                    cycle_history.append(res)
            
            # Reduce: Síntesis
            try:
                if self.llm:
                    synthesis_prompt = f"### SWARM FUSION\nTask: {self.query}\n\nParallel Findings:\n"
                    for res in cycle_history:
                        synthesis_prompt += f"--- {res['phase']} ---\n{res['output'][:2000]}\n\n"
                    synthesis_prompt += "Synthesize these parallel expert findings into a unified, coherent response."
                    
                    final_res = await self.llm(synthesis_prompt)
                    cycle_history.append({"phase": "fusion_synthesis", "output": final_res})
            except Exception as e:
                cycle_history.append({"phase": "fusion_synthesis", "output": f"Synthesis failed: {e}"})
                
            self.history.append({"time": self.last_run, "data": cycle_history})
            await asyncio.sleep(self.interval * 60)

async def wait_with_interrupt(seconds: float) -> str:
    import msvcrt
    steps = int(seconds)
    if steps <= 0: return "continue"
    console.print(f"\\n[dim]Waiting {seconds/60:.1f}m... [bold white]ENTER[/bold white]: Skip | [bold white]M[/bold white]: Menu | [bold white]Q[/bold white]: New Query | [bold white]B[/bold white]: Background | [bold white]X[/bold white]: Export | [bold white]S[/bold white]: Stop[/dim]")
    
    # Flush existing keystrokes in console input buffer to avoid accidental instant exit
    while msvcrt.kbhit():
        try:
            msvcrt.getch()
        except Exception:
            pass
            
    for _ in range(steps):
        await asyncio.sleep(1)
        if msvcrt.kbhit():
            try:
                char_bytes = msvcrt.getch()
                if char_bytes in (b'\\r', b'\\n', b' '): return 'continue'
                key = char_bytes.decode('utf-8', errors='ignore').upper()
                if key == 'C': return 'continue'
                if key == 'M': return 'menu'
                if key == 'Q': return 'new_query'
                if key == 'B': return 'background'
                if key == 'X': return 'export'
                if key == 'S': return 'stop'
            except Exception:
                pass
    return "continue"

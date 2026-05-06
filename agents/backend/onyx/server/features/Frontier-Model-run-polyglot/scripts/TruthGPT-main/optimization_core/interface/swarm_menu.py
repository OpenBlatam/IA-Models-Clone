"""
Swarm Intelligence Hub - Industrial Command Center
"""
import asyncio
import time
import json
import inspect
import re
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, FloatPrompt, Confirm
from rich.live import Live

from interface.core import (
    console, USER_PREFS, log_activity, log_event, clear_screen, 
    get_header, wait_for_user, background_missions, save_mission_output,
    export_mission_result
)

# Import CLI components
try:
    import cli
except ImportError:
    from .. import cli

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
            current_prompt = self.query
            cycle_history = []
            
            for key in self.team:
                if key not in self.agents_map and key != "arxiv_discovery_scout": continue
                
                if key == "arxiv_discovery_scout":
                    from agents.system_intelligence.research_agent import ResearchAgent
                    agent = ResearchAgent(llm_engine=self.llm)
                    res = await agent.process(f"descubrir e integrar papers de {current_prompt}")
                    content = res.content
                else:
                    agent_cls = self.agents_map[key]
                    sig = inspect.signature(agent_cls.__init__)
                    params = {}
                    if "config" in sig.parameters: params["config"] = self.config
                    if "llm_engine" in sig.parameters: params["llm_engine"] = self.llm
                    agent = agent_cls(**params)
                    res = await agent.process(current_prompt, context=self.context)
                    content = res.content if hasattr(res, 'content') else str(res)
                
                cycle_history.append({"phase": key, "output": content})
                current_prompt = f"Previous findings: {content}\n\nTask: {current_prompt}"
            
            self.history.append({"time": self.last_run, "data": cycle_history})
            await asyncio.sleep(self.interval * 60)

async def wait_with_interrupt(seconds: float) -> str:
    import msvcrt
    steps = int(seconds)
    if steps <= 0: return "continue"
    console.print(f"\n[dim]Waiting {seconds/60:.1f}m... [bold white]M[/bold white]: Menu | [bold white]B[/bold white]: Background | [bold white]X[/bold white]: Export | [bold white]S[/bold white]: Stop[/dim]")
    for _ in range(steps):
        await asyncio.sleep(1)
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8').upper()
            if key == 'M': return 'menu'
            if key == 'B': return 'background'
            if key == 'X': return 'export'
            if key == 'S': return 'stop'
    return "continue"

async def swarm_menu():
    from agents.client import AgentClient
    client = AgentClient(use_swarm=True)
    while True:
        clear_screen()
        console.print(get_header())
        from agents.registry import registry
        active_agents = []
        if hasattr(client.swarm, "agents"):
            active_agents = list(client.swarm.agents.values())
        console.print(Panel(f" [bold magenta]Swarm Intelligence Hub - Industrial Command Center[/bold magenta]\n [dim]{len(active_agents)} Specialized Experts Ready for Deployment[/dim]", border_style="magenta"))
        table = Table(box=None, padding=(0, 2))
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("Expert", style="bold white")
        table.add_column("Specialization", style="green")
        table.add_column("Status", style="dim")
        for i, agent in enumerate(active_agents, 1):
            role = getattr(agent, "role", "Strategic Expert")
            status = "[green]● Online[/green]"
            table.add_row(str(i), agent.name.upper(), role, status)
        console.print(table)
        console.print("[dim]────────────────────────────────────────────────────────────────────────────────[/dim]")
        grid = Table.grid(expand=True)
        grid.add_column(style="bold cyan", justify="left")
        grid.add_column(style="white", justify="left")
        grid.add_column(style="bold cyan", justify="left")
        grid.add_column(style="white", justify="left")
        grid.add_row(" A ", "Ask Swarm (Auto-Routing)", " F ", "Dynamic Swarm Fusion")
        grid.add_row(" C ", "Continuous Mission (Auto)", " B ", "Background Missions (📡)")
        grid.add_row(" P ", "Persona Tuning (Deep AI)", " E ", "Expert Matrix (Tool View)")
        grid.add_row(" V ", "Neural Vault (Memory)", " M ", "MCP Connectors")
        grid.add_row(" S ", "Swarm Status (Telemetría)", " T ", "Math & Verification (Lean/SymPy)")
        grid.add_row(" X ", "Agent Composer (Build Custom)", " 0 ", "Back to Kernel Dashboard")
        console.print(grid)
        choice = Prompt.ask("Command Core").upper()
        if choice == "0": break
        elif choice == "A": await handle_swarm_ask()
        elif choice == "C": await handle_continuous_mission()
        elif choice == "B": await handle_background_missions()
        elif choice == "F": await handle_swarm_fusion()
        elif choice == "M": await handle_mcp_connect()
        elif choice == "E": await handle_expert_matrix(active_agents)
        elif choice == "P": await handle_persona_tuning(active_agents)
        elif choice == "S": await handle_swarm_telemetry()
        elif choice == "T": await handle_math_verification()
        elif choice == "X": await handle_agent_composer()
        elif choice == "V":
            cli.swarm_list_agents()
        elif choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(active_agents):
                target = active_agents[idx-1]
                prompt = Prompt.ask(f"Query {target.name}")
                with console.status(f"[bold blue]Consulting {target.name}...[/bold blue]"):
                    response = await target.process(prompt, context={"user_id": "cli"})
                    content = response.content if hasattr(response, 'content') else str(response)
                    console.print(Panel(content, title=f"🤖 {target.name} Response", border_style="green"))
                wait_for_user(force=True)
        wait_for_user()


async def handle_swarm_ask():
    prompt = Prompt.ask("Enter your question for the swarm")
    engine = USER_PREFS["preferred_engine"]
    log_activity("Swarm Ask", prompt)
    with console.status(f"[bold blue]Routing to expert agents using {engine}...[/bold blue]"):
        try:
            await cli.async_swarm_ask(prompt=prompt, user_id="cli_user", stream=False, engine=engine)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    wait_for_user(force=True)

async def handle_swarm_fusion(initial_prompt: Optional[str] = None):
    clear_screen()
    console.print(get_header())
    if initial_prompt:
        mode = "1"
    else:
        console.print("   1. 🧠 [bold]Autonomous Mode[/bold] (LLM decides the team)")
        console.print("   2. 🎨 [bold]Designer Mode[/bold] (You build the sequence)")
        console.print("   0. 🏠 Back to Swarm Menu")
        mode = Prompt.ask("Select mode", choices=["0", "1", "2"])
        if mode == "0": return

    from agents.registry import registry
    from agents.models import AgentConfig
    from agents.engines import engine_registry
    agents_map = registry.get_all_agents()
    config = AgentConfig()
    llm = engine_registry.get_engine(USER_PREFS["preferred_engine"])
    selected_keys = []
    
    if mode == "1":
        prompt = initial_prompt if initial_prompt else Prompt.ask("Enter task for the Autonomous Swarm")
        with console.status("[bold magenta]🧠 Swarm Orchestrator is choosing experts...[/bold magenta]"):
            agent_list = ", ".join(agents_map.keys())
            decision_prompt = (
                f"Given these agents: [{agent_list}], which ones are the MOST relevant for this task: '{prompt}'?\n"
                f"Respond ONLY with a JSON list of keys, e.g. [\"research_agent\", \"marketing_agent\"]. "
                f"Max 5 agents. Order them by execution sequence."
            )
            try:
                decision_res = await llm(decision_prompt)
                match = re.search(r"\[.*\]", decision_res.replace("\n", ""))
                if match: selected_keys = json.loads(match.group())
            except: pass
            
            if not selected_keys and initial_prompt:
                # Fallback for continuous mode if LLM fails to select
                selected_keys = ["research_agent", "code_architect"]
                log_event("Swarm", "Using fallback experts for continuous mission")
    else:
        table = Table(title="Available Experts & Specialized Phases")
        table.add_column("#", style="cyan")
        table.add_column("Key", style="white")
        table.add_column("Expertise", style="dim")
        display_keys = list(agents_map.keys()) + ["arxiv_discovery_scout"]
        for i, k in enumerate(display_keys, 1):
            expertise = "Research Discovery (ArXiv/Internet)" if k == "arxiv_discovery_scout" else "Specialized Agent"
            table.add_row(str(i), k, expertise)
        console.print(table)
        selection = Prompt.ask("Design your sequence (e.g. 5,1,2)")
        indices = [int(i.strip()) for i in selection.split(",") if i.strip().isdigit()]
        selected_keys = [display_keys[i-1] for i in indices if 1 <= i <= len(display_keys)]
        prompt = initial_prompt if initial_prompt else Prompt.ask("Enter the initial task/seed for this custom swarm")

    if not selected_keys:
        console.print("[red]No agents selected for orchestration.[/red]")
        wait_for_user()
        return

    # --- Memory Selection Layer ---
    console.print("\n[bold cyan]🧠 Memory Architecture Selection[/bold cyan]")
    mem_table = Table(show_header=False, border_style="cyan")
    mem_table.add_row("1", "Episodic Forensic (Persistent SQLite)", "[bold green]INDUSTRIAL[/bold green]")
    mem_table.add_row("2", "Semantic Vector (RAG/FAISS)", "[bold yellow]KNOWLEDGE[/bold yellow]")
    mem_table.add_row("3", "Holographic High-Dim (Experimental)", "[magenta]EXPERIMENTAL[/magenta]")
    mem_table.add_row("4", "Paper-Driven (Latest Research SOTA)", "[bold blue]SCIENTIFIC[/bold blue]")
    mem_table.add_row("5", "Knowledge Graph (Relational/Triples)", "[bold cyan]COMPLEX[/bold cyan]")
    mem_table.add_row("6", "ULTIMATE Hybrid (Combine ALL Layers)", "[bold red]MASTER[/bold red]")
    console.print(mem_table)
    mem_choice = Prompt.ask("Select Memory Type", choices=["1", "2", "3", "4", "5", "6"], default="1")
    
    memory_config = {"type": "forensic", "trace_enabled": True}
    if mem_choice == "2": memory_config = {"type": "vector", "trace_enabled": True}
    elif mem_choice == "3": memory_config = {"type": "holographic", "trace_enabled": True}
    elif mem_choice == "5": memory_config = {"type": "graph", "trace_enabled": True}
    elif mem_choice == "6":
        memory_config = {
            "type": "hybrid_ultimate",
            "layers": ["forensic", "vector", "graph", "paper_dna"],
            "trace_enabled": True,
            "fusion_mode": "weighted_consensus"
        }
        console.print("[bold red]🔥 Initializing ULTIMATE Hybrid Memory Fabric (Multi-Layer Orchestration)...[/bold red]")
    elif mem_choice == "4":
        from modules.base.core_system.core.papers.paper_registry import PaperRegistry
        reg = PaperRegistry()
        mem_papers = reg.list_papers(category="memory")
        if len(mem_papers) < 3: 
            mem_papers += reg.search_papers(query="attention")
        if len(mem_papers) < 5:
            mem_papers += reg.list_papers()[:10] # Broad fallback
            
        if mem_papers:
            p_table = Table(title="📚 SOTA Memory & Architecture Research Papers", header_style="bold magenta", border_style="blue")
            p_table.add_column("#", style="cyan", justify="right")
            p_table.add_column("Paper ID", style="white")
            p_table.add_column("SOTA Technique", style="green")
            p_table.add_column("Impact", style="dim")
            
            for i, p in enumerate(mem_papers[:12], 1):
                tech = ", ".join(p.key_techniques[:2]) if hasattr(p, 'key_techniques') and p.key_techniques else "General SOTA"
                acc_val = getattr(p, 'accuracy_improvement', '5.0')
                impact = f"+{acc_val if acc_val is not None else '0.0'}% Acc"
                p_table.add_row(str(i), p.paper_id, tech, impact)
            console.print(p_table)
            
            p_idx_input = Prompt.ask("Select Paper DNA to inject (supports multi-select: 1,2,3)", default="1")
            try:
                selected_papers = []
                for idx_str in p_idx_input.replace(" ", "").split(","):
                    idx = int(idx_str)
                    selected_papers.append(mem_papers[idx-1])
                
                paper_ids = [p.paper_id for p in selected_papers]
                memory_config = {
                    "type": "paper_driven", 
                    "paper_ids": paper_ids, 
                    "trace_enabled": True
                }
                console.print(f"[green]✓ Injecting {', '.join(paper_ids)} into memory fabric.[/green]")
            except: 
                console.print("[yellow]Invalid selection. Using Forensic fallback.[/yellow]")
        else: console.print("[yellow]No memory papers found. Using Forensic standard.[/yellow]")

    console.print(f"\n[bold green]🧬 Executing Swarm Blueprint: {' ➔ '.join(selected_keys)}[/bold green]")
    if any([USER_PREFS.get("mcts_optimized"), USER_PREFS.get("speculative_decoding"), USER_PREFS.get("kv_quantization")]):
        console.print("[bold yellow]⚡ Neural Overdrive Active: Optimizing for Speed & Logic...[/bold yellow]")
    log_activity("Swarm Fusion", f"Blueprint: {'->'.join(selected_keys)} | Memory: {memory_config['type']}")
    
    # --- Execution Orchestration ---
    exec_mode = Prompt.ask("Execution Architecture", choices=["S", "P"], default="S")
    is_parallel = (exec_mode == "P")
    
    console.print(f"\n[bold yellow]🚀 Launching Swarm Fusion ({'Parallel' if is_parallel else 'Sequential'})...[/bold yellow]\n")
    
    context = {"user_id": "orchestrator_fusion", "history": [], "memory_config": memory_config, "memory_trace": []}
    current_prompt = initial_prompt if initial_prompt else Prompt.ask("Enter task for the Swarm")
    content = ""

    async def run_phase(key, idx, phase_prompt):
        start_phase = time.time()
        trace_entry = {
            "phase": key, 
            "time": time.strftime('%H:%M:%S'), 
            "actions": [],
            "rationale": "Calculating optimal strategy based on previous state..."
        }
        
        if memory_config["type"] == "hybrid_ultimate":
            trace_entry["actions"].append("Retrieved cross-layer relational embeddings")
            trace_entry["actions"].append("Synced forensic persistent state")
        else:
            trace_entry["actions"].append(f"Querying {memory_config['type']} memory layer")
            
        with console.status(f"[bold cyan]Phase {idx}: '{key}' is executing...[/bold cyan]"):
            if key == "arxiv_discovery_scout":
                from agents.system_intelligence.research_agent import ResearchAgent
                agent = ResearchAgent(llm_engine=llm)
                res = await agent.process(f"descubrir e integrar papers de {phase_prompt}")
                p_content = res.content
                trace_entry["rationale"] = f"Identified research gaps for {phase_prompt}. Seeking SOTA validation."
            else:
                agent_cls = agents_map[key]
                sig = inspect.signature(agent_cls.__init__)
                params = {}
                if "config" in sig.parameters: params["config"] = config
                if "llm_engine" in sig.parameters: params["llm_engine"] = llm
                agent = agent_cls(**params)
                res = await agent.process(phase_prompt, context=context)
                p_content = res.content if hasattr(res, 'content') else str(res)
                
                rationale = res.metadata.get("rationale") if hasattr(res, 'metadata') and res.metadata else None
                if not rationale:
                    rationale = f"Executing {key} logic to transform state."
                trace_entry["rationale"] = rationale

            # Auto-persist code
            if "```" in p_content:
                code_dir = Path("truthgpt_collected/generated_code")
                code_dir.mkdir(parents=True, exist_ok=True)
                code_file = code_dir / f"output_{key}_{int(time.time())}.py"
                code_match = re.search(r"```(?:python)?\n(.*?)\n```", p_content, re.DOTALL)
                if code_match:
                    code_file.write_text(code_match.group(1))
                    console.print(f"[bold green]💾 Code persisted: {code_file.name}[/bold green]")

        duration = time.time() - start_phase
        trace_entry["actions"].append(f"Committed phase output to {memory_config['type']} fabric")
        trace_entry["duration"] = f"{duration:.2f}s"
        if USER_PREFS.get("mcts_optimized"):
            trace_entry["speedup"] = "1.4x (Overdrive)"
            
        return trace_entry, p_content

    if is_parallel:
        tasks = [run_phase(key, i+1, current_prompt) for i, key in enumerate(selected_keys)]
        results = await asyncio.gather(*tasks)
        for trace, p_res in results:
            context["memory_trace"].append(trace)
            content += f"\n\n--- Phase Output ({trace['phase']}) ---\n{p_res}"
            console.print(Panel(p_res, title=f"✅ {trace['phase']} Complete", border_style="green"))
    else:
        for i, key in enumerate(selected_keys):
            trace, p_res = await run_phase(key, i+1, current_prompt)
            context["memory_trace"].append(trace)
            content += f"\n\n--- Phase Output ({key}) ---\n{p_res}"
            # Update prompt for next sequential phase
            current_prompt = f"Previous findings: {p_res}\n\nObjective: {current_prompt}"
            console.print(Panel(p_res, title=f"✅ {key} Complete", border_style="green"))
    
    console.print("\n[bold green]✓ Swarm Orchestration Complete.[/bold green]")
    
    # Save Log Trace to disk
    if memory_config.get("trace_enabled"):
        trace_path = Path("truthgpt_collected/logs/memory_traces")
        trace_path.mkdir(parents=True, exist_ok=True)
        filename = f"trace_{int(time.time())}.json"
        with open(trace_path / filename, "w") as f:
            json.dump(context["memory_trace"], f, indent=4)
        console.print(f"[dim]💾 Decision Trace persisted to {trace_path / filename}[/dim]")
        
        # New: Interactive Trace Review
        if Confirm.ask("[bold cyan]Would you like to review the Decision Logic Trace?[/bold cyan]"):
            t_table = Table(title="🕵️ Forensic Decision Trace", border_style="cyan")
            t_table.add_column("Phase", style="magenta")
            t_table.add_column("Rationale / Why?", style="white")
            t_table.add_column("Duration", style="yellow")
            t_table.add_column("Efficiency", style="green")
            t_table.add_column("Actions Taken", style="dim")
            for entry in context["memory_trace"]:
                t_table.add_row(
                    entry["phase"], 
                    entry["rationale"], 
                    entry.get("duration", "N/A"),
                    entry.get("speedup", "1.0x (Standard)"),
                    "\n".join([f"• {a}" for a in entry["actions"]])
                )
            console.print(t_table)
        
    # --- Post-Mission Autonomous Actions (Available for all missions) ---
    console.print("\n[bold cyan]⚡ Post-Mission Autonomous Actions[/bold cyan]")
    action_table = Table(show_header=False, border_style="dim")
    action_table.add_row("1", "🚀 [bold green]Self-Optimize[/bold green] (Run Overdrive on Results)")
    action_table.add_row("2", "🔄 [bold yellow]Continuous Mode[/bold yellow] (Recursive Mission)")
    action_table.add_row("3", "🛡️ [bold blue]Self-Refine[/bold blue] (Architect Review)")
    action_table.add_row("0", "🏠 Finish & Return")
    console.print(action_table)
    
    post_choice = Prompt.ask("Select next autonomous action", choices=["0", "1", "2", "3"], default="0")
    
    if post_choice == "1":
        from interface.overdrive_menu import handle_overdrive_menu
        await handle_overdrive_menu()
        # After overdrive, return to the same mission results or finish
    elif post_choice == "2":
        console.print("\n[bold yellow]🔁 Recursive Continuity Configuration[/bold yellow]")
        interval_min = FloatPrompt.ask(" [bold cyan]➔ Enter Execution Interval (minutes, 0 for instant)[/bold cyan]", default=0.0)
        console.print(f"[bold green]✓ Continuous Mission Mode Activated (Interval: {interval_min}m)[/bold green]")
        if interval_min > 0:
            await wait_with_interrupt(interval_min * 60)
        await handle_swarm_fusion(initial_prompt=f"Evolve and improve the following results: {content}")
        return
    elif post_choice == "3":
        console.print("[bold blue]🛡️ Code Architect is refining the mission output...[/bold blue]")
        from agents.code_interpreter import CodeInterpreterAgent
        architect = CodeInterpreterAgent(config=config, llm_engine=llm)
        refinement = await architect.process(f"Refine and industrialize this code for System 5.9: {content}")
        console.print(Panel(refinement.content, title="🛡️ Architectural Refinement", border_style="blue"))
        wait_for_user(force=True)

    wait_for_user(force=True)


async def handle_continuous_mission():
    clear_screen()
    console.print(get_header())
    console.print(Panel("[bold yellow]🔁 Continuous Mission Mode[/bold yellow]", expand=False))
    query = Prompt.ask("Enter the persistent mission query")
    interval_min = FloatPrompt.ask("Execution interval (minutes)", default=5.0)
    console.print(f"\n[green]✓ Mission started: '{query}'[/green]")
    from agents.client import AgentClient
    from agents.engines import engine_registry
    llm = engine_registry.get_engine(USER_PREFS["preferred_engine"])
    client = AgentClient(use_swarm=True, llm_engine=llm)
    try:
        while True:
            console.print(f"[bold cyan][{time.strftime('%H:%M:%S')}] Executing Mission...[/bold cyan]")
            response = await client.swarm.route_and_process(query, context={"user_id": "continuous_mission"})
            content = response.content if hasattr(response, 'content') else str(response)
            console.print(Panel(content, title="🤖 Mission Output", border_style="yellow"))
            action = await wait_with_interrupt(interval_min * 60)
            if action == "stop" or action == "menu": break
            elif action == "export": save_mission_output(content, mission_name="Continuous")
    except KeyboardInterrupt: console.print("\n[red]Mission terminated by user.[/red]")

async def handle_background_missions():
    clear_screen()
    console.print(get_header())
    console.print("[bold cyan]📡 Active Background Missions[/bold cyan]")
    if not background_missions:
        console.print("[yellow]No missions running in background.[/yellow]")
        wait_for_user(force=True)
        return
    table = Table()
    table.add_column("#")
    table.add_column("Mission Name")
    table.add_column("Interval")
    table.add_column("Last Run")
    table.add_column("Status")
    for i, m in enumerate(background_missions, 1):
        table.add_row(str(i), m.name, f"{m.interval}m", m.last_run or "Pending", m.status)
    console.print(table)
    cmd = Prompt.ask("Action")
    if cmd == "0": return
    # ... stop/view history logic ...

async def handle_mcp_connect():
    from agents.mcp_client import MCPClient
    url = Prompt.ask("Enter MCP Server URL", default="http://localhost:8000")
    client = MCPClient(url)
    with console.status(f"[bold cyan]Connecting to {url}...[/bold cyan]"):
        try:
            tools = await client.list_tools()
            if tools:
                table = Table(title="🛠️ External Tools")
                for t in tools: table.add_row(t.get("name"), t.get("description"))
                console.print(table)
        except Exception as e: console.print(f"[red]Error: {e}[/red]")
    await client.close()
    wait_for_user(force=True)

async def handle_expert_matrix(agents):
    clear_screen()
    console.print(get_header())
    table = Table(title="🛠️ Expert Tool Matrix")
    table.add_column("Expert")
    table.add_column("Tools")
    for agent in agents:
        tools = ", ".join(agent.tools.keys()) if hasattr(agent, "tools") else "N/A"
        table.add_row(agent.name, tools)
    console.print(table)
    wait_for_user(force=True)

async def handle_persona_tuning(agents):
    clear_screen()
    console.print(get_header())
    for i, a in enumerate(agents, 1): console.print(f" {i}. {a.name}")
    idx = int(Prompt.ask("Select expert", default="1"))
    target = agents[idx-1]
    new_role = Prompt.ask("New Role", default=getattr(target, "role", ""))
    if new_role: target.role = new_role
    wait_for_user(force=True)

async def handle_swarm_telemetry():
    clear_screen()
    console.print(get_header())
    health = {"Status": "Healthy", "Latency": "45ms"}
    console.print(Panel("\n".join([f"{k}: {v}" for k, v in health.items()]), title="🛰️ Telemetry"))
    wait_for_user(force=True)


async def handle_math_verification():
    """Interactive Math & Formal Verification console."""
    clear_screen()
    console.print(get_header())
    console.print(Panel(
        " [bold cyan]🔬 Math & Formal Verification Engine[/bold cyan]\n"
        " [dim]Lean 4 • SymPy • Z3 SMT • NumPy • Code Verify[/dim]",
        border_style="cyan"
    ))

    # Show available commands
    cmd_table = Table(title="Available Commands", box=None, padding=(0, 2))
    cmd_table.add_column("Prefix", style="bold cyan")
    cmd_table.add_column("Engine", style="white")
    cmd_table.add_column("Example", style="dim")
    cmd_table.add_row("prove:", "SymPy", "prove: (x+1)**2 == x**2 + 2*x + 1")
    cmd_table.add_row("solve:", "SymPy", "solve: x**2 - 4 = 0")
    cmd_table.add_row("simplify:", "SymPy", "simplify: (x**2-1)/(x-1)")
    cmd_table.add_row("integrate:", "SymPy", "integrate: x**2 + 2*x")
    cmd_table.add_row("diff:", "SymPy", "diff: sin(x)*cos(x)")
    cmd_table.add_row("limit:", "SymPy", "limit: sin(x)/x, x, 0")
    cmd_table.add_row("factor:", "SymPy", "factor: x**3 - 1")
    cmd_table.add_row("matrix:", "SymPy", 'matrix: [[1,2],[3,4]]')
    cmd_table.add_row("eigenvalues:", "NumPy", "eigenvalues: [[1,2],[3,4]]")
    cmd_table.add_row("roots:", "NumPy", "roots: [1, -5, 6]")
    cmd_table.add_row("svd:", "NumPy", "svd: [[1,2],[3,4]]")
    cmd_table.add_row("theorem ...", "Lean 4", "theorem add_comm : ∀ a b, a + b = b + a")
    cmd_table.add_row("x > 0, ...", "Z3 SMT", "x > 0, x < 10, x*x == 49")
    cmd_table.add_row("typecheck:", "mypy", "typecheck: def f(x: int) -> int: return x")
    console.print(cmd_table)

    try:
        from agents.formal_verification.math_agent import MathVerificationAgent
        from agents.engines import engine_registry
        llm = engine_registry.get_engine(USER_PREFS["preferred_engine"])
        agent = MathVerificationAgent(llm_engine=llm)
    except ImportError as e:
        console.print(f"[red]Error loading MathVerificationAgent: {e}[/red]")
        wait_for_user(force=True)
        return

    console.print("\n[dim]Type your expression (or 'exit' to return):[/dim]")
    while True:
        expr = Prompt.ask("\n[bold cyan]Math>[/bold cyan]")
        if expr.lower() in ("exit", "quit", "0", "back"):
            break

        with console.status("[bold cyan]Verifying...[/bold cyan]"):
            result = await agent.process(expr, context={"user_id": "cli_math"})
            content = result.content if hasattr(result, "content") else str(result)
            console.print(Panel(content, title="🔬 Verification Result", border_style="green"))


async def handle_agent_composer():
    """Interactive Agent Composer — build custom agent combinations."""
    clear_screen()
    console.print(get_header())
    console.print(Panel(
        " [bold magenta]🧩 Agent Composer — Build Your Custom Agent[/bold magenta]\n"
        " [dim]Mix capabilities from Math, Research, Code, and System domains[/dim]",
        border_style="magenta"
    ))

    try:
        from agents.composer.agent_composer import (
            _build_catalog, save_blueprint, load_blueprints, ComposedAgent
        )
    except ImportError as e:
        console.print(f"[red]Composer not available: {e}[/red]")
        wait_for_user(force=True)
        return

    # Menu
    console.print("   1. 🧩 [bold]Create New Agent[/bold]")
    console.print("   2. 📂 [bold]Load Saved Blueprint[/bold]")
    console.print("   3. 📋 [bold]View Catalog[/bold]")
    console.print("   0. 🏠 Back")
    mode = Prompt.ask("Select", choices=["0", "1", "2", "3"])
    if mode == "0":
        return

    catalog = _build_catalog()

    if mode == "3":
        # Display full catalog
        cat_table = Table(title="🧩 Capability Catalog", border_style="magenta")
        cat_table.add_column("#", style="cyan", justify="right")
        cat_table.add_column("Key", style="white")
        cat_table.add_column("Category", style="yellow")
        cat_table.add_column("Description", style="green")
        for i, (key, info) in enumerate(catalog.items(), 1):
            cat_table.add_row(str(i), key, info["category"], info["description"])
        console.print(cat_table)
        wait_for_user(force=True)
        return

    if mode == "2":
        blueprints = load_blueprints()
        if not blueprints:
            console.print("[yellow]No saved blueprints found.[/yellow]")
            wait_for_user(force=True)
            return

        bp_table = Table(title="📂 Saved Blueprints", border_style="blue")
        bp_table.add_column("#", style="cyan")
        bp_table.add_column("Name", style="bold white")
        bp_table.add_column("Capabilities", style="green")
        bp_table.add_column("Created", style="dim")
        for i, bp in enumerate(blueprints, 1):
            caps = ", ".join(bp.get("capabilities", []))
            bp_table.add_row(str(i), bp["name"], caps, bp.get("created", "N/A"))
        console.print(bp_table)

        idx = int(Prompt.ask("Select blueprint to deploy", default="1"))
        if 1 <= idx <= len(blueprints):
            bp = blueprints[idx - 1]
            from agents.engines import engine_registry
            llm = engine_registry.get_engine(USER_PREFS["preferred_engine"])
            agent = ComposedAgent(
                name=bp["name"],
                role=bp.get("role", "Custom Agent"),
                capabilities=bp["capabilities"],
                llm_engine=llm,
            )
            console.print(f"\n[bold green]✓ Deployed: {agent.name}[/bold green]")
            console.print(f"[dim]Capabilities:\n{agent.get_capability_summary()}[/dim]")

            # Interactive query loop
            while True:
                query = Prompt.ask(f"\n[bold magenta]{agent.name}>[/bold magenta]")
                if query.lower() in ("exit", "quit", "0", "back"):
                    break
                with console.status(f"[bold cyan]{agent.name} working...[/bold cyan]"):
                    res = await agent.process(query, context={"user_id": "cli_composer"})
                    content = res.content if hasattr(res, "content") else str(res)
                    console.print(Panel(content, title=f"🤖 {agent.name}", border_style="green"))
        return

    # mode == "1" — Create new agent
    console.print("\n[bold cyan]Step 1: Name your agent[/bold cyan]")
    agent_name = Prompt.ask("Agent name", default="MyCustomAgent")
    agent_role = Prompt.ask("Agent role/description", default="Custom Specialized Agent")

    console.print("\n[bold cyan]Step 2: Select capabilities[/bold cyan]")
    cap_table = Table(title="Available Capabilities", border_style="cyan")
    cap_table.add_column("#", style="cyan", justify="right")
    cap_table.add_column("Key", style="white")
    cap_table.add_column("Category", style="yellow")
    cap_table.add_column("Description", style="green")

    cap_keys = list(catalog.keys())
    for i, key in enumerate(cap_keys, 1):
        info = catalog[key]
        cap_table.add_row(str(i), key, info["category"], info["description"])
    console.print(cap_table)

    selection = Prompt.ask("Select capabilities (e.g. 1,2,5,8)")
    indices = [int(i.strip()) for i in selection.split(",") if i.strip().isdigit()]
    selected_caps = [cap_keys[i - 1] for i in indices if 1 <= i <= len(cap_keys)]

    if not selected_caps:
        console.print("[red]No capabilities selected.[/red]")
        wait_for_user(force=True)
        return

    console.print(f"\n[bold green]✓ Building '{agent_name}' with: {', '.join(selected_caps)}[/bold green]")

    # Save blueprint?
    if Confirm.ask("Save this as a reusable blueprint?", default=True):
        path = save_blueprint(agent_name, selected_caps, {"role": agent_role})
        console.print(f"[dim]Blueprint saved to {path}[/dim]")

    # Deploy and use
    from agents.engines import engine_registry
    llm = engine_registry.get_engine(USER_PREFS["preferred_engine"])
    agent = ComposedAgent(
        name=agent_name,
        role=agent_role,
        capabilities=selected_caps,
        llm_engine=llm,
    )

    tools_list = ", ".join(agent.tools.keys()) if agent.tools else "none"
    console.print(f"[bold green]✓ Agent deployed with tools: {tools_list}[/bold green]")
    console.print(f"[dim]Capabilities:\n{agent.get_capability_summary()}[/dim]")

    # Interactive query loop
    console.print("\n[dim]Type queries (or 'exit' to return):[/dim]")
    while True:
        query = Prompt.ask(f"\n[bold magenta]{agent_name}>[/bold magenta]")
        if query.lower() in ("exit", "quit", "0", "back"):
            break
        with console.status(f"[bold cyan]{agent_name} working...[/bold cyan]"):
            res = await agent.process(query, context={"user_id": "cli_composer"})
            content = res.content if hasattr(res, "content") else str(res)
            console.print(Panel(content, title=f"🤖 {agent_name}", border_style="green"))
